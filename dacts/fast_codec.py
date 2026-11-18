"""A fast stand-in for KVAE's decoder (owner 2026-10-25: "I don't want the codec to even be half the processing time,
it needs to be fast"). KVAE's decoder is a DAC-style conv stack (~32 GMAC per second of audio, most of it in 768..96
channel convs at 0.4..24 kHz); on v13XS it was 88% of the CPU time. This one is Vocos-style: ConvNeXt blocks at 100
frames/s predict a short-time spectrum (magnitude + phase) and an inverse FFT makes the waveform -- ~2 GMAC per second
of audio, and it writes 44.1 kHz directly (no 48 -> 44.1 kHz resampler, no 20.1 kHz tone to filter out).

It is distilled from KVAE's decoder (tools/kvae_student_train.py): same input (raw KVAE latents, 50 Hz), and it learns
to sound like `codec.kvae_to_audio` on real and TTS latents. models/kvae-student.pt.
"""
import torch
import torch.nn as nn

SR, HOP, N_FFT = 44100, 441, 1764  # (100 frames/s, 2 per KVAE latent frame; 40 ms windows like Vocos's 24 kHz model)


_MATMUL, _BASIS = False, {}  # (_MATMUL: force the matmul path on the CPU, for tests)


def _idft_matmul(spec, window):
    """irfft(spec, N_FFT, dim=1) * window as ONE fp32 matmul with a fixed (N_FFT, N_FFT + 2) matrix (2026-11-01: on the
    GPU, even with the plans cached, the cuFFT call was still 23% of v14xl-vtv's main thread; ~9 GFLOP for 30 s of audio
    is nothing for the GPU, but too slow on the CPU, so the site's CPU decoding keeps the FFT). Same sum as irfft: the
    imaginary parts of the DC and Nyquist bins are ignored, the other bins count twice."""
    key = (window.device, window.data_ptr())
    if key not in _BASIS:
        k, t = torch.arange(N_FFT // 2 + 1, dtype=torch.float64), torch.arange(N_FFT, dtype=torch.float64)
        a = 2 * torch.pi * t[:, None] * k[None, :] / N_FFT
        w = torch.full((N_FFT // 2 + 1,), 2.0, dtype=torch.float64)
        w[0] = w[-1] = 1.0
        s = -torch.sin(a) * w
        s[:, 0] = s[:, -1] = 0.0
        b = torch.cat([torch.cos(a) * w, s], 1) * window.double().cpu()[:, None] / N_FFT
        _BASIS[key] = b.to(device=window.device, dtype=torch.float32)
    tf32 = torch.backends.cuda.matmul.allow_tf32  # (full fp32: TF32's 10-bit mantissa would add noise at about -60 dB)
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.autocast(spec.device.type, enabled=False):
            return _BASIS[key] @ torch.cat([spec.real, spec.imag], 1).float()  # (B, N_FFT, n)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32


def istft(spec, window, length):
    """torch.istft(spec, N_FFT, HOP, N_FFT, window, center=True, length=length), the same steps (inverse FFT, window,
    overlap-add, divide by the summed squared window, trim) without its window check, which reads a value back from the
    GPU (2026-11-01; py-spy had torch.istft at 29% of v14xl-vtv's main thread -- most of it turned out to be cuFFT
    plans, below)."""
    n = spec.shape[-1]
    if spec.is_cuda or _MATMUL:
        frames = _idft_matmul(spec, window)
    else:
        # (the frames padded with silent ones to a multiple of 128, then dropped: each frame is its own inverse FFT, so
        # the same values -- but cuFFT gets a few dozen sizes instead of a new one per clip, and building a plan for a new
        # size (loading kernel modules) was what took the time: py-spy --native, 2026-11-01)
        m = -(-n // 128) * 128
        frames = torch.fft.irfft(torch.nn.functional.pad(spec, (0, m - n)), N_FFT, dim=1)[..., :n] * window[:, None]  # (B, N_FFT, n)
    out = N_FFT + HOP * (n - 1)
    fold = lambda f: torch.nn.functional.fold(f, (1, out), (1, N_FFT), stride=(1, HOP))[:, 0, 0]  # noqa: E731
    y = fold(frames)
    env = fold((window ** 2)[None, :, None].expand(1, N_FFT, n))
    s = N_FFT // 2
    return y[:, s:s + length] / env[:, s:s + length]


class ConvNeXt(nn.Module):
    def __init__(self, dim, hidden, scale):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1, self.pw2 = nn.Linear(dim, hidden), nn.Linear(hidden, dim)
        self.gamma = nn.Parameter(torch.full((dim,), scale))

    def forward(self, x):  # (B, dim, F)
        y = self.norm(self.dw(x).transpose(1, 2))
        return x + (self.gamma * self.pw2(torch.nn.functional.gelu(self.pw1(y)))).transpose(1, 2)


class FastDecoder(nn.Module):
    def __init__(self, mean, std, dim=512, layers=12, mult=3):
        super().__init__()
        self.settings = dict(dim=dim, layers=layers, mult=mult)
        self.register_buffer("mean", torch.as_tensor(mean).float().view(1, -1, 1))
        self.register_buffer("std", torch.as_tensor(std).float().view(1, -1, 1))
        self.up = nn.ConvTranspose1d(self.mean.shape[1], dim, 4, stride=2, padding=1)  # 50 -> 100 frames/s
        self.norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList(ConvNeXt(dim, mult * dim, 1 / layers) for _ in range(layers))
        self.final = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, N_FFT + 2)
        self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)

    def forward(self, z):
        """raw KVAE latents (B, 64, T) at 50 Hz -> audio (B, T * 882) at 44.1 kHz."""
        x = self.up((z.float() - self.mean) / self.std)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        for b in self.blocks:
            x = b(x)
        with torch.autocast(x.device.type, enabled=False):  # (the spectrum in fp32: phase is fragile)
            mag, ph = self.head(self.final(x.transpose(1, 2).float())).transpose(1, 2).chunk(2, dim=1)
        spec = torch.polar(torch.exp(mag).clamp(max=100.0), ph)
        return istft(spec, self.window, x.shape[-1] * HOP)

    def forward_rows(self, z, lengths):
        """several rows of different lengths in ONE network pass (2026-11-07, you: "make the jobs bigger"): raw latents
        (B, 64, T) + lengths (frames, ints) -> [audio (n_b * 882,)]. Each row gives what forward(z[b:b+1, :, :n_b])
        does: everything past a row's end is zeroed before each convolution (where that call's zero padding sits, so
        the convolutions see the same zeros) and the inverse STFT runs per row on its own frames (its edge
        normalisation depends on where the row ends)."""
        B, _, T = z.shape
        dev = z.device
        # (2026-11-09) lengths copied ONCE through pinned memory: torch.as_tensor(list, device=cuda) is a pageable copy
        # that waits for all queued GPU work (the last step's optimizer) -- it brought back the step-start drain
        lt = torch.tensor(lengths, dtype=torch.long)
        lt = lt.pin_memory().to(dev, non_blocking=True) if dev.type == "cuda" else lt.to(dev)
        m50 = (torch.arange(T, device=dev)[None] < lt[:, None])[:, None]  # (B, 1, T)
        x = self.up(((z.float() - self.mean) / self.std) * m50)
        m = (torch.arange(x.shape[-1], device=dev)[None] < 2 * lt[:, None])[:, None]
        x = self.norm(x.transpose(1, 2)).transpose(1, 2) * m
        for b in self.blocks:
            x = b(x) * m
        with torch.autocast(x.device.type, enabled=False):
            mag, ph = self.head(self.final(x.transpose(1, 2).float())).transpose(1, 2).chunk(2, dim=1)
        spec = torch.polar(torch.exp(mag).clamp(max=100.0), ph)
        return [istft(spec[b:b + 1, :, :2 * n], self.window, 2 * n * HOP)[0] for b, n in enumerate(lengths)]

    def state(self):
        return {"settings": self.settings, "model": self.state_dict()}

    @classmethod
    def load(cls, path, device="cpu"):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck["model"]
        m = cls(sd["mean"].flatten(), sd["std"].flatten(), **ck["settings"])
        m.load_state_dict(sd)
        return m.to(device).eval()
