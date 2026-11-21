"""DAC helpers: loading the codec and mapping codes <-> normalized continuous latents.

The model's target is, per DAC frame, the concatenation of the (L2-normalized) codebook
vectors selected by each of the first n_q quantizers: an (n_q * 8)-dim vector. At decode
time DAC's `from_latents` snaps each 8-dim chunk to its nearest codebook entry (cosine),
so generated latents are always decoded through valid codes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SAMPLE_RATE = 44100
HOP = 512


def load_dac(device="cuda"):
    import dac  # only for DAC-era checkpoints

    path = dac.utils.download(model_type="44khz")
    return dac.DAC.load(str(path)).to(device).eval().requires_grad_(False)


def normalized_codebooks(dac_model) -> torch.Tensor:
    """(n_q_total, codebook_size, 8) unit-norm codebook vectors."""
    return torch.stack([F.normalize(q.codebook.weight.detach(), dim=-1) for q in dac_model.quantizer.quantizers]).cpu()


class LatentSpace(nn.Module):
    def __init__(self, codebooks: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        n_q, k, d = codebooks.shape
        self.register_buffer("flat", codebooks.reshape(n_q * k, d).float())
        self.register_buffer("offsets", torch.arange(n_q) * k)
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float())
        self.n_q = n_q

    @classmethod
    def from_stats(cls, stats: dict, n_q: int):
        d = n_q * 8
        return cls(stats["codebooks"][:n_q], stats["mean"][:d], stats["std"][:d])

    def state(self):
        n_q, d = self.n_q, self.n_q * 8
        return {"codebooks": self.flat.reshape(n_q, -1, 8).cpu(), "mean": self.mean.cpu(), "std": self.std.cpu()}

    def encode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes (B, n_q, T) long -> normalized latents (B, T, n_q*8)."""
        b, n_q, t = codes.shape
        z = F.embedding(codes + self.offsets[None, :n_q, None], self.flat)  # B, n_q, T, 8
        z = z.permute(0, 2, 1, 3).reshape(b, t, n_q * 8)
        return (z - self.mean) / self.std

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.std + self.mean


@torch.no_grad()
def latents_to_audio(dac_model, latents: torch.Tensor) -> torch.Tensor:
    """latents (B, T, n_q*8) un-normalized -> audio (B, T*HOP) at 44.1kHz."""
    z_q, _, _ = dac_model.quantizer.from_latents(latents.transpose(1, 2).float())
    return dac_model.decode(z_q)[:, 0]


# ---------------------------------------------------------------- KVAE-Audio (continuous)
# 48 kHz, 64-d latent at 50 Hz; generated latents are decoded as-is (no codebooks, no snapping).
KVAE_SR, KVAE_HOP, KVAE_DIM = 48000, 960, 64


def load_kvae(device="cuda"):
    from huggingface_hub import hf_hub_download
    from kvae_audio import KVAEAudio

    m = KVAEAudio.load(hf_hub_download("kandinskylab/KVAE-Audio", "kvae-audio.pt"), map_location="cpu")
    return m.to(device).eval().requires_grad_(False)


class KVAELatent(nn.Module):
    """Per-dim standardization of KVAE latents (the model's target space)."""

    codec = "kvae"

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float())

    @classmethod
    def from_stats(cls, stats: dict, n_q=None):
        return cls(stats["mean"], stats["std"])

    def state(self):
        return {"codec": "kvae", "mean": self.mean.cpu(), "std": self.std.cpu()}

    def encode(self, lat: torch.Tensor) -> torch.Tensor:
        """raw latents (B, T, 64) -> normalized."""
        return (lat.float() - self.mean) / self.std

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.std + self.mean


KVAE_WHOLE = 1200  # frames (~24 s): takes up to this long decode in one piece, exactly as before
KVAE_PIECE, KVAE_CTX, KVAE_XF = 800, 64, 8  # longer: 800-frame pieces, 64 frames of context each side, 2x8-frame crossfade


def _kvae_decode(kvae_model, lat: torch.Tensor) -> torch.Tensor:
    """(11-21, you: "work on 4gb vram") a one-minute song decoded whole asked a 4 GB card for 3.5 GB + 460 MB more: long
    latents decode in overlapping pieces (each with context around it, joined by a short linear crossfade whose two
    ramps add up to 1). (B, T, 64) -> (B, N) at 48 kHz."""
    T = lat.shape[1]
    if T <= KVAE_WHOLE:
        return kvae_model.decode(lat.transpose(1, 2).float())[:, 0]
    out = None
    for s in range(0, T, KVAE_PIECE):
        e = min(T, s + KVAE_PIECE)
        ks, ke = max(0, s - KVAE_XF), min(T, e + KVAE_XF)  # kept, overlapping the neighbours by 2 * XF frames
        a0, b0 = max(0, ks - KVAE_CTX), min(T, ke + KVAE_CTX)
        y = kvae_model.decode(lat[:, a0:b0].transpose(1, 2).float())[:, 0]
        spf = y.shape[-1] // (b0 - a0)
        if out is None:
            out = torch.zeros(y.shape[0], T * spf, device=y.device, dtype=y.dtype)
        seg = y[:, (ks - a0) * spf:(ke - a0) * spf]
        w = torch.ones(seg.shape[-1], device=y.device, dtype=y.dtype)
        n = 2 * KVAE_XF * spf
        if ks > 0:
            w[:n] = torch.linspace(0, 1, n + 2, device=y.device, dtype=y.dtype)[1:-1]
        if ke < T:
            w[-n:] = torch.linspace(1, 0, n + 2, device=y.device, dtype=y.dtype)[1:-1]
        out[:, ks * spf:ks * spf + seg.shape[-1]] += seg * w
        del y
    return out


@torch.no_grad()
def kvae_to_audio(kvae_model, latents: torch.Tensor) -> torch.Tensor:
    """latents (B, T, 64) un-normalized -> audio (B, N) at 44.1 kHz (decoded at 48 kHz, resampled)."""
    a = _kvae_decode(kvae_model, latents)
    # KVAE's decoder adds a steady tone at ~20.1 kHz (+35-45 dB, on every clip): cut everything above
    # 19.5 kHz with a short raised-cosine edge before resampling (nothing audible lives up there).
    # The FFT runs at the next power of two: every take has its own length, and cuFFT planning an odd length took
    # ~25 ms of a 36 ms filter; padded, 2.4 ms (2026-10-24). Zero-padding also stops the end wrapping into the start.
    n = a.shape[-1]
    m = 1 << max(0, n - 1).bit_length()
    X = torch.fft.rfft(a, n=m, dim=-1)
    f = torch.fft.rfftfreq(m, 1 / KVAE_SR).to(a.device)
    g = 0.5 - 0.5 * torch.cos(torch.pi * torch.clamp((19800.0 - f) / 300.0, 0, 1))
    a = torch.fft.irfft(X * g, n=m, dim=-1)[..., :n]
    return _resampler(a.device)(a)


_RESAMPLERS = {}


def _resampler(device):
    """48 -> 44.1 kHz with its kernel built once per device (AF.resample rebuilt it on every call); same output."""
    import torchaudio
    k = str(device)
    if k not in _RESAMPLERS:
        _RESAMPLERS[k] = torchaudio.transforms.Resample(KVAE_SR, SAMPLE_RATE).to(device)
    return _RESAMPLERS[k]


def frame_rate(cfg) -> float:
    return KVAE_SR / KVAE_HOP if getattr(cfg, "codec", "dac") == "kvae" else SAMPLE_RATE / HOP


def latent_space(state: dict):
    """Rebuild the latent space a checkpoint was trained with."""
    if state.get("codec") == "kvae":
        return KVAELatent(state["mean"], state["std"])
    return LatentSpace(state["codebooks"], state["mean"], state["std"])
