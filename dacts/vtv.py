"""Voice-to-voice (v14xl-vtv; owner 2026-10-30: "just add it to v14-xl and name it v14xl-vtv", "it will support both
tts and vtv right"): the decoder generates from what a RECORDING carries instead of from text -- its content (ContentVec,
a tester's pick: a HuBERT trained to drop who is speaking) plus its pitch / voicing / loudness contour (dacts.contour) --
and the voice from the reference prompt (+ the voice table for learned voices), exactly as cloning does. So the same
model does text-to-speech and speech-to-speech; your performance (timing, melody, stress, laughs) carries over.

Training (train.py, cfg.use_vtv): on up to cfg.vtv_rows rows a step, the row's own latents are decoded (fast codec),
pitch- AND formant-shifted by a random +-cfg.vtv_shift semitones (played back at another rate), read by ContentVec and
time-warped back onto the row's frames: content that still carries the speaker's timbre would now point to the wrong
voice, so the model has to take the voice from the prompt. The contour comes from the real (unshifted) latents.
DacTTS.forward swaps those rows' text conditioning for vtv_in(content, contour).

Exact pitch (you 2026-10-31: "just teach it to handle sticking to pitch well", "like exact pitch", "singing mode does
it exactly, it doesnt have to be exact for speech"): cfg.vtv_exact_prob of the VTV rows are flagged (DacTTS.vtv_exact;
singing mode calls convert(exact_pitch=True)). Their content is read from audio bent by a random pitch curve (a new target
in +-cfg.vtv_exact_shift every ~cfg.vtv_exact_seg s), so the source's own melody -- even its shape -- is no guide; their
contour's pitch is RMVPE's on the real audio where RMVPE hears voicing (exactly what singing mode feeds at inference,
dacts/sing.py), and they get pitch-following losses (cfg.vtv_pitch_weight: ContourNet in DacTTS.flow_loss;
cfg.vtv_rmvpe_weight: RMVPE on decoded audio, PitchLoss below).
"""

import math

import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

CV_DIM = 768  # ContentVec's last layer (its README: no final_proj)
CV_SR = 16000


def precast_(module, dtype):
    """Frozen module run under autocast -> its Linear / Conv weights stored in autocast's dtype once (autocast passes
    them through: the kernels see the same bits). Norm layers keep fp32 (autocast runs them in fp32); weight_norm is
    computed once, the same op as at every call before."""
    from torch.nn.utils import parametrize
    for m in module.modules():
        if parametrize.is_parametrized(m, "weight"):
            parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            m.to(dtype)
    return module


def bucket_len(n, steps=16):
    """n rounded up to the next of `steps` lengths per octave (<= 1/steps more)."""
    q = 1 << max(0, n.bit_length() - 1 - int(math.log2(steps)))
    return -(-n // q) * q


def _conv_pad(m, args):
    x = args[0]
    L = x.shape[-1]
    p, d, k, s = m.padding[0], m.dilation[0], m.kernel_size[0], m.stride[0]
    m._want = (L + 2 * p - d * (k - 1) - 1) // s + 1
    return (F.pad(x, (0, bucket_len(L) - L)),) + tuple(args[1:])


def _conv_cut(m, args, y):
    return y[..., : m._want]


def bucket_convs_(module):
    """Every zero-padded Conv1d in `module` gets its input padded with zeros at the end to a bucket length (16 per
    octave) and its output cut back to the true length: the same values (the extra zeros only reach outputs that are
    cut), but cuDNN sees a few hundred input lengths instead of a new one per clip. Each new one cost a heuristics
    query + a runtime kernel build on the CPU: ~13% of v14xl-vtv's main thread in ContentVec (py-spy --native,
    2026-11-01)."""
    for m in module.modules():
        if isinstance(m, nn.Conv1d) and m.padding_mode == "zeros" and not isinstance(m.padding, str):
            m.register_forward_pre_hook(_conv_pad)
            m.register_forward_hook(_conv_cut)
    return module


_MEL_MM = __import__("os").environ.get("MEL_MATMUL", "1") != "0"  # (MEL_MATMUL=0: torch.stft again; _MEL_FORCE: the matmul on the CPU, for tests)
_MEL_FORCE, _MEL_BASIS = False, {}


def rmvpe_mel(tr, audio):
    """tr.mel(audio) (third_party.rmvpe MelSpectrogram: torch.stft, |.|, mel, log) with the FFT as ONE fp32 matmul on the
    GPU (2026-11-10: py-spy had torch.stft there at 16% of v15xl's main thread -- cuFFT again, as in fast_codec's istft:
    every clip has a new frame count). Same frames (reflect padding, hop 160, Hann 1024) and the same sums; differentiable
    (PitchLoss). audio (B, n) at 16 kHz -> (B, 128, 1 + n // 160)."""
    m = tr.mel
    if not (_MEL_MM and (audio.is_cuda or _MEL_FORCE)):
        return m(audio)
    N, key = m.n_fft, (m.window.device, m.window.data_ptr())
    if key not in _MEL_BASIS:
        t, k = torch.arange(N, dtype=torch.float64), torch.arange(N // 2 + 1, dtype=torch.float64)
        a = 2 * torch.pi * t[:, None] * k[None, :] / N
        b = torch.cat([torch.cos(a), -torch.sin(a)], 1) * m.window.double().cpu()[:, None]  # (N, N + 2): real | imaginary
        _MEL_BASIS[key] = b.to(device=m.window.device, dtype=torch.float32)
    tf32 = torch.backends.cuda.matmul.allow_tf32  # (full fp32, as fast_codec._idft_matmul)
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.autocast(audio.device.type, enabled=False):
            x = F.pad(audio.float()[:, None], (N // 2, N // 2), mode="reflect")[:, 0]
            spec = x.unfold(-1, N, m.hop_length) @ _MEL_BASIS[key]  # (B, frames, N + 2)
            mag = torch.complex(spec[..., :N // 2 + 1], spec[..., N // 2 + 1:]).abs()  # (complex abs: the stft path's gradient)
            return torch.log(torch.clamp(mag @ m.mel_basis.T, min=m.clamp)).transpose(1, 2)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32


class BucketResample(nn.Module):
    """torchaudio Resample with the input padded with zeros at the end to a bucket length and the output cut to the
    true one (its own zero padding sits there anyway: the same samples), for cuDNN's sake as bucket_convs_."""

    def __init__(self, src, dst):
        super().__init__()
        import torchaudio.transforms as AT
        self.rs = AT.Resample(src, dst)
        g = math.gcd(src, dst)
        self.o, self.n = src // g, dst // g

    def forward(self, w):
        L = w.shape[-1]
        return self.rs(F.pad(w, (0, bucket_len(L) - L)))[..., : int(math.ceil(self.n * L / self.o))]


_CODEC_ROWS = __import__("os").environ.get("VTV_CODEC_ROWS", "1") != "0"  # (2026-11-07) ContentFeatures.batch: one codec pass for all rows


def to_device_async(x, device):
    """CPU tensor -> device without waiting for queued GPU work (a pageable copy syncs the stream; a pinned one doesn't)."""
    return x.pin_memory().to(device, non_blocking=True) if torch.device(device).type == "cuda" else x.to(device)


def contour_feats(c):
    """(B, T, 3) dacts.contour [semitones re 100 Hz, voiced logit, dB / 10] -> network-sized features."""
    return torch.stack([c[..., 0] / 12, torch.sigmoid(c[..., 1]), c[..., 2]], -1)


class ContentFeatures(nn.Module):
    def __init__(self, device="cuda", codec_path=str(Path(__file__).resolve().parents[1] / "models" / "fast-codec.pt")):
        super().__init__()
        from transformers import HubertModel
        from transformers.utils import logging as tlog

        from .fast_codec import SR, FastDecoder
        lv = tlog.get_verbosity()
        tlog.set_verbosity_error()  # (the checkpoint's legacy final_proj is unused on purpose)
        self.cv = HubertModel.from_pretrained("lengyue233/content-vec-best").to(device).eval().requires_grad_(False)
        tlog.set_verbosity(lv)
        if str(device).startswith("cuda"):  # (2026-11-01, speedups/ZIPPY_REPORT.md 3: autocast caches casts only of
            precast_(self.cv, torch.bfloat16)  # weights that train, so a frozen fp32 ContentVec was re-cast every call)
            bucket_convs_(self.cv)
        self.codec = FastDecoder.load(codec_path, device).requires_grad_(False)
        if str(device).startswith("cuda"):
            bucket_convs_(self.codec)
        self.codec_sr = SR

    @torch.no_grad()
    def from_audio(self, wav, sr, n_frames, shift_st=0.0):
        """mono audio (N,) at sr -> (n_frames, 768) content per 50 Hz frame; shift_st: pitch + formants moved by that
        many semitones first (the audio played back faster / slower), then the features stretched back in time."""
        # (the shifted rate rounded to 100 Hz, ~0.04 semitones: resample's filter bank has new_rate / gcd rows, and an
        # arbitrary rate like 52437 Hz -> 16000 made it thousands of rows: GPU memory blew up -- bench2 f1, 2026-10-30)
        src = int(round(sr * 2 ** (shift_st / 12) / 100)) * 100
        x = self._resampler(src, wav.device)(wav.float()[None]) if src != CV_SR else wav.float()[None]  # (filter cached)
        x = (x - x.mean()) / (x.std() + 1e-7)  # (ContentVec's feature extractor normalizes each clip)
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            h = self.cv(x).last_hidden_state.float()  # (1, ~N/320, 768)
        return F.interpolate(h.transpose(1, 2), size=n_frames, mode="linear", align_corners=False)[0].T

    def _resampler(self, src, device):
        """src Hz -> 16 kHz with its filter kept (AF.resample rebuilds the filter at every call)."""
        cache = self.__dict__.setdefault("_rs", {})
        if src not in cache:
            cache[src] = BucketResample(src, CV_SR).to(device)
        return cache[src]

    @torch.no_grad()
    def from_audio_bent(self, wav, sr, n_frames, shift, seg):
        """from_audio with the pitch + formants bent by a random curve: knots every 0.5-1.5 x seg seconds, each a
        random shift in [-shift, shift] semitones, straight glides between them. The audio is played back at that
        varying rate (one anti-alias low-pass, then one interpolation onto the 16 kHz grid), read by ContentVec in one
        pass, and every 50 Hz frame takes the features from where its moment landed."""
        x = wav.float()
        n_s, dev = x.shape[0], x.device
        dur = n_s / sr
        knots = [0.0]
        while knots[-1] < dur:
            knots.append(knots[-1] + seg * (0.5 + torch.rand(()).item()))
        # the curve (a few thousand points) is worked out on the CPU and copied over without waiting: on the GPU, its
        # knots' copy and the scalars read back made the step wait for all queued GPU work (2026-11-01, ZIPPY_REPORT 1B;
        # the knot shifts now come from the CPU generator: other dice, same distribution)
        kt = torch.tensor(knots)
        ks = (2 * torch.rand(len(knots)) - 1) * shift  # semitones at the knots
        g = torch.linspace(0, dur, max(2, int(dur * 1000)) + 1)  # 1 ms grid of input time
        j = (torch.searchsorted(kt, g, right=True) - 1).clamp(0, len(knots) - 2)
        w = (g - kt[j]) / (kt[j + 1] - kt[j])
        rate = 2 ** ((ks[j] * (1 - w) + ks[j + 1] * w) / 12)  # playback speed at each input moment
        tau = torch.cat([g.new_zeros(1), torch.cumsum((g[1:] - g[:-1]) / (0.5 * (rate[1:] + rate[:-1])), 0)])  # output time
        r_max, tau_end = float(rate.max()), float(tau[-1])
        g, tau = to_device_async(g, dev), to_device_async(tau, dev)
        fc = min(0.5 * CV_SR / r_max, 0.45 * sr) / sr  # (cycles / sample: nothing above 8 kHz after speeding up)
        k, hw = self._sinc_grid(dev)
        h = 2 * fc * torch.sinc(2 * fc * k) * hw
        xl = F.conv1d(x[None, None], (h / h.sum())[None, None], padding=128)[0, 0]
        n_out = int(tau_end * CV_SR)
        t_in = self._interp(torch.arange(n_out, device=dev, dtype=torch.float32) / CV_SR, tau, g)  # output -> input time
        p = (t_in * sr).clamp(0, n_s - 1)
        i0 = p.floor().long().clamp(max=n_s - 2)
        fr = p - i0.float()
        y = (xl[i0] * (1 - fr) + xl[i0 + 1] * fr)[None]
        y = (y - y.mean()) / (y.std() + 1e-7)
        with torch.autocast(y.device.type, dtype=torch.bfloat16, enabled=y.is_cuda):
            h_ = self.cv(y).last_hidden_state.float()[0]  # (M, 768) at 50 Hz of the bent audio
        t = (torch.arange(n_frames, device=dev, dtype=torch.float32) + 0.5) * (dur / n_frames)  # frame centres
        i = (self._interp(t, g, tau) * (h_.shape[0] / tau_end) - 0.5).clamp(0, h_.shape[0] - 1)  # (from_audio's stretch)
        i0 = i.floor().long()
        i1 = (i0 + 1).clamp(max=h_.shape[0] - 1)
        w = (i - i0.float())[:, None]
        return h_[i0] * (1 - w) + h_[i1] * w

    def _sinc_grid(self, dev):
        """the anti-alias filter's taps and window (the same every call)."""
        cache = self.__dict__.setdefault("_sg", {})
        if str(dev) not in cache:
            cache[str(dev)] = (torch.arange(-128, 129, device=dev, dtype=torch.float32),
                               torch.hann_window(257, periodic=False, device=dev))
        return cache[str(dev)]

    @staticmethod
    def _interp(x, xp, fp):
        """np.interp for increasing xp (torch, on any device)."""
        j = (torch.searchsorted(xp, x, right=True) - 1).clamp(0, len(xp) - 2)
        w = ((x - xp[j]) / (xp[j + 1] - xp[j]).clamp(min=1e-12)).clamp(0, 1)
        return fp[j] * (1 - w) + fp[j + 1] * w

    @torch.no_grad()
    def rmvpe(self, wavs16, chunk=512, ctx=64, group=16):
        """RMVPE F0 (Hz per 10 ms, numpy) of several 16 kHz clips in one batched pass: each clip's mel is cut into
        windows of `chunk` frames overlapping by 2 x ctx and the middles are stitched (`group` windows at a time).
        4 clips of ~18 s: 104 ms vs 286 ms one clip at a time; vs the whole-clip pass median 0.5 cents, 1.3% of frames
        > 50 cents apart, 2.7% voicing flips (2026-10-31, Emilia EN)."""
        from . import sing
        tr = sing.tracker(wavs16[0].device)
        sal, spans = self._rmvpe_sal(tr, wavs16, chunk, ctx, group)
        sal = sal.cpu().numpy()  # (windows, chunk, 360)
        out, k = [], 0
        for sp in spans:
            rows = []
            for a, b in sp:
                rows.append(sal[k, ctx:ctx + b - a])
                k += 1
            out.append(tr.decode(np.concatenate(rows)))
        return out

    @staticmethod
    def _rmvpe_sal(tr, wavs16, chunk, ctx, group):
        hop, fill = chunk - 2 * ctx, math.log(tr.mel.clamp)
        mels = [rmvpe_mel(tr, w.float()[None])[0] for w in wavs16]  # (128, n) each
        wins, spans = [], []
        for m in mels:
            n = m.shape[1]
            pad = F.pad(m, (ctx, chunk), value=fill)
            spans.append([])
            for s0 in range(0, n, hop):
                wins.append(pad[:, s0:s0 + chunk])
                spans[-1].append((s0, min(s0 + hop, n)))
        x = torch.stack(wins)
        if x.is_cuda and _RG_ON:  # (2026-11-07) whole groups as one CUDA-graph replay each: the frozen pitch net ran
            # eager, hundreds of small launches per group at a new batch size each time (py-spy: ~9% of the main thread)
            nw = len(x)
            x = torch.cat([x, x.new_full((-nw % group,) + tuple(x.shape[1:]), fill)]) if nw % group else x
            sal = torch.cat([_rmvpe_graphed(tr.model, x[i:i + group]) for i in range(0, len(x), group)])[:nw]
            return sal, spans
        with torch.autocast(x.device.type, dtype=torch.float16, enabled=x.is_cuda):  # (fp16: within 0.1 cents of fp32,
            sal = torch.cat([tr.model(x[i:i + group]).float() for i in range(0, len(x), group)])  # half the memory)
        return sal, spans

    @torch.no_grad()
    def rmvpe_frames(self, wavs16, frames, per, chunk=512, ctx=64, group=16):
        """rmvpe() + sing.frame_pitch, all on the clips' device (2026-11-01, ZIPPY_REPORT 1C: reading the salience back
        made the step wait for all queued GPU work): [(semitones float64 (n,), voiced bool (n,))] for frames[i] model
        frames of `per` F0 frames each. Same maths in float64; numpy's sums run in another order (last-bit rounding)."""
        from . import sing
        tr = sing.tracker(wavs16[0].device)
        sal, spans = self._rmvpe_sal(tr, wavs16, chunk, ctx, group)
        cents = self.__dict__.get("_cents64")
        if cents is None:
            cents = self._cents64 = torch.as_tensor(tr.cents, device=sal.device)  # (float64, once)
        out, k = [], 0
        for sp, n in zip(spans, frames):
            s = torch.cat([sal[k + q, ctx:ctx + b - a] for q, (a, b) in enumerate(sp)])
            k += len(sp)
            # rmvpe.RMVPE.decode: the local average around the peak (first max, as np.argmax), in cents
            center = s.argmax(1) + 4
            sp_ = F.pad(s, (4, 4))
            idx = center[:, None] + torch.arange(-4, 5, device=s.device)
            w = sp_.gather(1, idx)
            c = (w.double() * cents[idx]).sum(1) / w.sum(1).clamp(min=1e-9).double()
            f0 = torch.where(sp_.max(1).values <= 0.03, 0.0, 10 * 2 ** (c / 1200))
            # sing.frame_pitch
            f = f0.new_zeros(n * per)
            m = min(len(f0), n * per)
            f[:m] = f0[:m]
            f = f.view(n, per)
            v = f > 0
            st = torch.where(v, 12 * torch.log2(f.clamp(min=1e-3) / 100), 0.0).sum(1) / v.sum(1).clamp(min=1)
            out.append((st, v.double().mean(1) >= 0.5))
        return out

    @torch.no_grad()
    def from_latents(self, z_raw, n_frames, shift_st=0.0):
        """one row's raw KVAE latents (64, T) -> (n_frames, 768)."""
        wav = self.codec(z_raw[None, :, :n_frames].float())[0]
        return self.from_audio(wav, self.codec_sr, n_frames, shift_st)

    @torch.no_grad()
    def batch(self, codes, y_len, rows: int, shift: float, min_frames: int = 100, exact=None, force=None, force_exact=None):
        """codes (B, T, 64) raw latents (as the loader gives them), y_len (B,) frames -> (content (B, T, 768) bf16,
        rows (B,) bool) or None.
        Up to `rows` random rows of >= min_frames (2 s); each gets its own random shift in [-shift, shift] semitones.
        exact: (prob, shift, seg, patch, start (B,) frames or None) -> each picked row is an exact-pitch row with that
        probability (content bent by a random pitch curve; RMVPE pitch from frame start[b] on: a cross prompt keeps
        ContourNet's, as at inference); then returns (content, rows, exact (B,) bool, pitch (B, T/patch) semitones,
        voiced (B, T/patch)).
        force: (B,) CPU bool, rows that are always picked (cfg.vtv_only_prefix: singing), on top of `rows`; their
        exact-pitch probability is force_exact (None: as the others)."""
        B, T, _ = codes.shape
        dev = codes.device
        y_len = y_len.cpu()  # (pass the loader's CPU copy: reading the GPU one waited for the last step's optimizer)
        long_ = y_len >= min_frames
        forced = (long_ & force).nonzero().flatten().tolist() if force is not None else []
        ok = (long_ & ~force if force is not None else long_).nonzero().flatten().tolist()
        if not ok and not forced:
            return None
        pick = forced + [ok[i] for i in torch.randperm(len(ok))[:rows].tolist()]
        fset = set(forced)
        out = torch.zeros(B, T, CV_DIM, device=dev, dtype=torch.bfloat16)
        on = torch.zeros(B, dtype=torch.bool)  # (on the CPU: setting a GPU element per row launched a kernel each, 8% of the main thread)
        if exact is not None:
            x_prob, x_shift, x_seg, patch, start = exact
            xr = torch.zeros(B, dtype=torch.bool)
            xst, xv = torch.zeros(B, T // patch, device=dev), torch.zeros(B, T // patch, dtype=torch.bool, device=dev)
        todo = []  # exact rows: (row, first step, steps, 16 kHz audio) for one RMVPE call
        wavs = {}
        if pick and _CODEC_ROWS and hasattr(self.codec, "forward_rows"):  # (2026-11-07) every row's audio in one codec
            # pass instead of one per row: the same audio per row (FastDecoder.forward_rows), fewer, bigger GPU jobs
            ns_ = [int(y_len[b]) for b in pick]
            # (2026-11-09) index through a pinned copy: codes[list] made a pageable copy that waited for the GPU
            zs_ = codes.index_select(0, to_device_async(torch.tensor(pick, dtype=torch.long), dev))[:, :max(ns_)].transpose(1, 2).float()
            wavs = dict(zip(pick, self.codec.forward_rows(zs_, ns_)))
        for b in pick:
            n = int(y_len[b])
            if exact is not None and torch.rand(()).item() < (force_exact if b in fset and force_exact is not None else x_prob):
                wav = wavs[b] if b in wavs else self.codec(codes[b].T[None, :, :n].float())[0]
                out[b, :n] = self.from_audio_bent(wav, self.codec_sr, n, x_shift, x_seg).to(torch.bfloat16)
                k0 = (int(start[b]) if start is not None else 0) // patch
                ns = n // patch
                if ns - k0 >= 1:
                    todo.append((b, k0, ns, self._resampler(self.codec_sr, wav.device)(wav[None, k0 * patch * len(wav) // n:])[0]))
                xr[b] = True
            else:
                s = (2 * torch.rand(()).item() - 1) * shift
                out[b, :n] = (self.from_audio(wavs[b], self.codec_sr, n, s) if b in wavs
                              else self.from_latents(codes[b].T, n, s)).to(torch.bfloat16)
            on[b] = True
        if todo:
            for (b, k0, ns, _), (st, v) in zip(todo, self.rmvpe_frames([w for *_, w in todo], [ns - k0 for _, k0, ns, _ in todo], 2 * patch)):
                xst[b, k0:ns], xv[b, k0:ns] = st.float(), v
        if exact is not None:
            return out, to_device_async(on, dev), to_device_async(xr, dev), xst, xv
        return out, to_device_async(on, dev)


class PitchLoss(nn.Module):
    """Exact-pitch rows, measured the way the result is judged (cfg.vtv_rmvpe_weight; you 2026-10-31: "like exact
    pitch"): a ~3 s crop of the one-jump clean estimate is decoded (fast codec) and read by RMVPE (frozen, with
    gradient), and compared with RMVPE on the same crop of the real recording on its voiced frames: the pitch in
    semitones (RMVPE's local average around the real pitch's bin, 20-cent bins: cents-level) + the salience curve
    (binary cross-entropy, so a pitch far off -- e.g. an octave -- is pulled back too)."""

    def __init__(self, latent, feats: ContentFeatures, crop: int = 150, rows: int = 2):
        super().__init__()
        self.latent, self.f, self.crop, self.rows = latent, feats, crop, rows

    def salience(self, z):
        """normalized latents (N, L, 64) -> RMVPE salience (N, 10 ms frames, 360), differentiable."""
        from . import sing
        tr = sing.tracker(z.device)
        tr.model.fc[0].train()  # (cuDNN's GRU backward needs train mode; a 1-layer GRU has no dropout: same output)
        a = self.f.codec(self.latent.denormalize(z.float()).transpose(1, 2))
        mel = rmvpe_mel(tr, self.f._resampler(self.f.codec_sr, a.device)(a))
        n = mel.shape[-1]
        mel = F.pad(mel, (0, 32 * ((n - 1) // 32 + 1) - n), value=math.log(tr.mel.clamp))
        return tr.model(mel)[:, :n]

    def forward(self, x1_hat, x1, gen):
        """x1_hat / x1: (N, T, 64) normalized latents (prediction / real); gen: (N, T) frames the model generated (on
        the CPU since 2026-11-01, DacTTS.flow_loss: picking the crops no longer waits for the GPU)."""
        from . import sing
        L = self.crop
        picks = []
        for i in torch.randperm(x1.shape[0]).tolist():
            g = gen[i]
            ok = (g.unfold(0, L, 1).all(-1)).nonzero().flatten() if g.shape[0] >= L else g.new_zeros(0)
            if len(ok):
                picks.append((i, int(ok[torch.randint(len(ok), (1,))])))
            if len(picks) >= self.rows:
                break
        if not picks:
            return None
        zh = torch.stack([x1_hat[i, s:s + L] for i, s in picks])
        zr = torch.stack([x1[i, s:s + L] for i, s in picks])
        with torch.autocast(device_type=zh.device.type, enabled=False):
            with torch.no_grad():
                sr = self.salience(zr)
            sh = self.salience(zh)
            voiced = (sr.max(-1).values > 0.03).float()  # (RMVPE's own threshold)
            cents = self.__dict__.get("_cents")
            if cents is None:  # (once: a numpy -> GPU copy waits for all queued GPU work)
                cents = self._cents = torch.as_tensor(sing.tracker(zh.device).cents, device=sh.device, dtype=torch.float32)  # (368,)
            idx = sr.argmax(-1, keepdim=True) + torch.arange(9, device=sh.device)  # the real pitch's bin +-4 (padded)
            sr_p, sh_p = F.pad(sr, (4, 4)), F.pad(sh, (4, 4))
            wr, wh = sr_p.gather(-1, idx), sh_p.gather(-1, idx)
            cr = (wr * cents[idx]).sum(-1) / wr.sum(-1).clamp(min=1e-6)
            ch = (wh * cents[idx]).sum(-1) / wh.sum(-1).clamp(min=1e-6)
            bce = (F.binary_cross_entropy(sh.clamp(1e-6, 1 - 1e-6), sr, reduction="none")  # (minus its floor, the
                   - F.binary_cross_entropy(sr.clamp(1e-6, 1 - 1e-6), sr, reduction="none")).mean(-1)  # real's entropy: 0 = same)
            per = (ch - cr).abs() / 100 + 10 * bce
            # (no "any frame voiced?" check, 2026-11-01: it waited for the GPU; an all-unvoiced crop now adds 0 with a
            # zero gradient instead of no term -- the same update)
            return (per * voiced).sum() / voiced.sum().clamp(min=1)


# ---- RMVPE as CUDA graphs (2026-11-07, "I want the cpu sending jobs to the gpu to be quicker") ------------------------
# One graph per input shape (in training: always (group, 128, chunk)). Checked against the eager pass when captured;
# anything off (or a capture error) -> that shape stays eager. RMVPE_GRAPHS=0 switches it off.
import os as _os  # noqa: E402

_RG_ON = _os.environ.get("RMVPE_GRAPHS", "1") != "0"
_RG = {}


def _rmvpe_eager(model, xg):
    with torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):  # (no weight-cast cache across graph capture)
        return model(xg).float()


def _rmvpe_graphed(model, xg):
    key = (id(model), tuple(xg.shape), xg.dtype, xg.device.index)
    e = _RG.get(key)
    if e is None:
        e = False
        try:
            si = xg.clone()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    ref = _rmvpe_eager(model, si)
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, capture_error_mode="thread_local"):  # (train.py's VTV thread captures while
                so = _rmvpe_eager(model, si)  # the main thread keeps launching: only this thread's work is captured)
            g.replay()
            diff = float((so - ref).abs().max())
            ok = diff <= 1e-3
            print(f"RMVPE CUDA graph for {tuple(xg.shape)}: max diff vs eager {diff:.2e} -> {'on' if ok else 'OFF (eager)'}",
                  flush=True)
            if ok:
                e = (g, si, so)
        except Exception as ex:  # (capture unsupported here: stay eager)
            print(f"RMVPE CUDA graph for {tuple(xg.shape)} failed ({type(ex).__name__}: {ex}); eager", flush=True)
        _RG[key] = e
    if e is False:
        return _rmvpe_eager(model, xg)
    g, si, so = e
    si.copy_(xg)
    g.replay()
    return so.clone()
