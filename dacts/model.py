"""Text -> DAC latents.

Matcha-TTS-style: a text encoder predicts a per-token prior mean `mu` and log-duration.
During training, monotonic alignment search (MAS) aligns `mu` to the target latents, which
supervises the durations with no external aligner. A flow-matching transformer then turns
noise into latents conditioned on the frame-aligned `mu`. Because `mu` already approximates
the target, the decoder only learns a residual refinement, which converges quickly and
samples well in ~10 ODE steps.
"""
import math
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .mas import maximum_path


def to_host(x):
    """GPU tensor -> (pinned CPU copy, event), queued without waiting; host_wait() then waits only until the GPU has
    reached this point, not for whatever was queued after it (2026-11-01, speedups/ZIPPY_REPORT.md)."""
    if not x.is_cuda:
        return x, None
    h = torch.empty(x.shape, dtype=x.dtype, pin_memory=True)
    h.copy_(x, non_blocking=True)
    ev = torch.cuda.Event()
    ev.record()
    return h, ev


def host_wait(hv):
    h, ev = hv
    if ev is not None:
        ev.synchronize()
    return h


def seq_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    max_len = max_len or int(lengths.max())
    return torch.arange(max_len, device=lengths.device)[None] < lengths[:, None]


def uniform_attn(x_len: torch.Tensor, y_len: torch.Tensor, tx: int, ty: int) -> torch.Tensor:
    """(B, Tx, Ty) 0/1: each line's tokens spread evenly over its frames (frame j -> token floor(j * nx / ny)). A
    position hint only (cfg.implicit_align): the decoder learns the real timing itself with full self-attention."""
    j = torch.arange(ty, device=y_len.device)[None].float()
    idx = torch.floor(j * x_len[:, None].float() / y_len[:, None].clamp(min=1).float()).long()
    idx = torch.minimum(idx, (x_len[:, None] - 1).clamp(min=0))
    a = F.one_hot(idx, tx).float() * (j < y_len[:, None]).float()[..., None]  # B, Ty, Tx
    return a.transpose(1, 2)


def patchify(z: torch.Tensor, p: int) -> torch.Tensor:
    b, t, d = z.shape
    return z.reshape(b, t // p, p * d)


def unpatchify(z: torch.Tensor, p: int) -> torch.Tensor:
    b, t, d = z.shape
    return z.reshape(b, t * p, d // p)


def patch_contour(net, z: torch.Tensor, p: int) -> torch.Tensor:
    """dacts.contour on patched latents (B, T, p*64) -> (B, T, 3) per decoder step: the net reads single KVAE frames,
    so it runs on the frames and pitch / voicing / loudness are averaged over each step's p frames (owner
    2026-10-26: patch 2 = 25 steps/s; the alignment that uses these is per step)."""
    if p == 1:
        return net(z)
    c = net(unpatchify(z, p))
    b, t = z.shape[:2]
    return c[:, : t * p].reshape(b, t, p, c.shape[-1]).mean(2)


# inference: all guidance passes of a step in one decoder call (False = one call per pass, the old way; A/B)
BATCH_GUIDANCE = True
# inference, prompted cloning: before every decoder evaluation, the prompt frames of the noisy input are set to what
# training showed there (the real prompt + its noise at that t; the loss never covered them, so the model's own
# trajectory there drifts off the training inputs). LongCat-AudioDiT (arXiv 2603.29339): SIM 0.769 -> 0.812.
PROMPT_PIN = False
# inference: adaptive projected guidance (APG, arXiv 2410.02416 / LongCat ablation) instead of plain CFG, e.g.
# {"eta": 0.5, "beta": -0.3}: the guidance direction in clean-audio space, split into the part parallel to the
# prediction (scaled by eta) and the part across it; beta = momentum. eta=1, beta=0 is exactly plain CFG.
# Measured 2026-10-21 (v12, 6 held-out cloned + 2 built-in voices, 128 clips): SIM +0.003 (41/64 wins), WER equal
# (3/64 lines with an error each). On for the website (playground passes apg=...), off here so A/B renders stay
# comparable. (Prompt pinning in the same test: SIM -0.009, more WER: not used.)
APG = None
# a thread can ask for the plain (uncompiled) modules: the website's takes while its start-up warm-up is still compiling
# (they would otherwise wait for each compile). Per thread: the warm-up thread itself keeps compiling. (2026-10-24)
EAGER = threading.local()


def eager() -> bool:
    return getattr(EAGER, "on", False)


# OFF for the website since 2026-10-24 (owner: "slight decoder artifacting", "make it sound real"): on the same real clips
# (24, text aligned, evals/hf_loss_probe.py) APG left 45% of the recording's 8-20 kHz energy, plain guidance 80%, and
# plain guidance's treble texture matches the recording's (spectral flatness 0.179 vs 0.183; APG 0.197). It had bought
# only SIM +0.003. The duller, smeared top end was the artifact, not KVAE (rebuilt latents decode like real ones).
APG_WEBSITE = None


def rope(t: int, dim: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    f = torch.outer(torch.arange(t, device=device).float(), inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def _rms_norm(x, weight):
    """F.rms_norm, spelled out while exporting to ONNX (no RMSNormalization op before opset 23)."""
    if torch.onnx.is_in_onnx_export():
        # statistics in fp32 so an fp16 graph cannot overflow on x^2
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + torch.finfo(torch.float32).eps)).to(x.dtype) * weight
    return F.rms_norm(x, (x.shape[-1],), weight)


def timestep_embedding(t: torch.Tensor, dim: int = 256) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
    args = t.float()[:, None] * 1000 * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)


# 2026-10-22 training speed: FlexAttention (Triton flash-style kernels). PyTorch's Windows builds have no FlashAttention
# and the memory-efficient kernel's backward is slow. Off unless train.py switches a decoder to it (cfg.flex_attn ->
# FlowDecoder.use_flex): the decoder then builds one block mask per forward from its key padding mask. Same maths.
try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except ImportError:  # older torch
    create_block_mask = flex_attention = None


# called as PASS_HOOK(i, n) after decoder pass i of n (synthesize's flow steps): the website shows each pass as it
# finishes (dashboard/server.py PublicQueue). None = nothing called.
PASS_HOOK = None


class Attention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.hd = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.q_norm = nn.Parameter(torch.ones(self.hd))
        self.k_norm = nn.Parameter(torch.ones(self.hd))

    def forward(self, x, mask, rot, bm=None):
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        # weight cast to activation dtype so bf16 hits the fused kernel
        q = apply_rope(_rms_norm(q, self.q_norm.to(q.dtype)), *rot)
        k = apply_rope(_rms_norm(k, self.k_norm.to(k.dtype)), *rot)
        # no attention-prob dropout: it forces a much slower backward kernel; residual dropout regularizes
        if bm is not None:  # FlexAttention with the decoder's block mask (key padding)
            o = flex_attention(q, k, v, block_mask=bm)
        else:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :])
        return self.out(o.transpose(1, 2).reshape(b, t, d))


class CrossBlock(nn.Module):
    """Text tokens attend to language-model token features. Output projection starts at zero, so
    adding it to a trained encoder changes nothing until it learns something useful."""

    def __init__(self, d, heads):
        super().__init__()
        self.h, self.hd = heads, d // heads
        self.n = nn.LayerNorm(d)
        self.q, self.kv = nn.Linear(d, d), nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, d)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, mem, mem_mask, gate=None):
        """gate: (B,) 0/1 per row (rows without a memory: 0, and their mask needs one True so attention stays finite)."""
        b, t, d = x.shape
        q = self.q(self.n(x)).view(b, t, self.h, self.hd).transpose(1, 2)
        k, v = self.kv(mem).view(b, mem.shape[1], 2, self.h, self.hd).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype), attn_mask=mem_mask[:, None, None, :])
        o = self.out(o.transpose(1, 2).reshape(b, t, d))
        return x + (o if gate is None else o * gate.to(o.dtype)[:, None, None])


class SeqConv(nn.Module):
    """1D 'same' convolution over (B, T, C) done as time-shifts + one GEMM.

    Mathematically a Conv1d, but stays channels-last: no cuDNN layout transposes, and the
    shifts fuse into the surrounding Triton kernels under torch.compile.
    """

    def __init__(self, d_in, d_out, k):
        super().__init__()
        self.k = k
        self.proj = nn.Linear(k * d_in, d_out)

    def forward(self, x, mask):
        x = x * mask[..., None]
        p = self.k // 2
        xp = F.pad(x, (0, 0, p, p))
        t = x.shape[1]
        return self.proj(torch.cat([xp[:, i : i + t] for i in range(self.k)], dim=-1))


class ConvFFN(nn.Module):
    """FFN whose first layer is a k=3 conv: cheap local context, smoother outputs."""

    def __init__(self, d, mult, dropout):
        super().__init__()
        self.c1 = SeqConv(d, d * mult, 3)
        self.c2 = nn.Linear(d * mult, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        return self.c2(self.drop(F.gelu(self.c1(x, mask))))


class Block(nn.Module):
    def __init__(self, d, heads, mult, dropout, cond_dim=None):
        super().__init__()
        affine = cond_dim is None
        self.n1 = nn.LayerNorm(d, elementwise_affine=affine)
        self.n2 = nn.LayerNorm(d, elementwise_affine=affine)
        self.attn = Attention(d, heads)
        self.ffn = ConvFFN(d, mult, dropout)
        self.drop = nn.Dropout(dropout)
        self.ada = None
        if cond_dim is not None:  # adaLN-zero
            self.ada = nn.Linear(cond_dim, 6 * d)
            nn.init.zeros_(self.ada.weight)
            nn.init.zeros_(self.ada.bias)

    def forward(self, x, mask, rot, c=None, bm=None):
        if self.ada is None:
            x = x + self.drop(self.attn(self.n1(x), mask, rot, bm))
            return x + self.drop(self.ffn(self.n2(x), mask))
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(F.silu(c))[:, None].chunk(6, dim=-1)
        x = x + g1 * self.drop(self.attn(self.n1(x) * (1 + sc1) + sh1, mask, rot, bm))
        return x + g2 * self.drop(self.ffn(self.n2(x) * (1 + sc2) + sh2, mask))


class ConvLN(nn.Module):
    def __init__(self, d_in, d_out, k, dropout):
        super().__init__()
        self.conv = SeqConv(d_in, d_out, k)
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        return self.drop(self.norm(F.relu(self.conv(x, mask))))


class FlowDurationPredictor(nn.Module):
    """Stochastic durations: flow matching over per-token log(1 + duration).

    A deterministic (MSE) predictor yields the same metronomic rhythm every time; sampling
    durations from a learned distribution gives natural, varied timing.
    """

    def __init__(self, d, hidden=256, layers=4, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(d + 1, hidden)
        self.t_mlp = nn.Sequential(nn.Linear(256, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.layers = nn.ModuleList([ConvLN(hidden, hidden, 3, dropout) for _ in range(layers)])
        self.out = nn.Linear(hidden, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.span_in = None  # cfg.use_dur_span: (present, log1p group frames / 4, log1p notes / 2) per token

    def add_span(self, hidden=256):
        self.span_in = nn.Linear(3, hidden)
        nn.init.zeros_(self.span_in.weight)
        nn.init.zeros_(self.span_in.bias)

    def velocity(self, w_t, t, h, mask, span=None):
        x = self.inp(torch.cat([h, w_t[..., None]], dim=-1)) + self.t_mlp(timestep_embedding(t))[:, None]
        if span is not None and self.span_in is not None:
            x = x + self.span_in(span.to(x.dtype))
        for layer in self.layers:
            x = x + layer(x, mask)
        return self.out(x).squeeze(-1) * mask

    def forward(self, w_t, t, h, mask, span=None):
        """= velocity, through __call__: what DacTTS.compile_modules compiles (loss() called velocity() directly, so
        the predictor's compile never applied in training)."""
        return self.velocity(w_t, t, h, mask, span)

    def loss(self, w1, h, mask, span=None):
        t = torch.sigmoid(torch.randn(w1.shape[0], device=w1.device))
        w0 = torch.randn_like(w1)
        w_t = (1 - t[:, None]) * w0 + t[:, None] * w1
        v = self(w_t, t, h, mask, span)  # (the noise above stays eager: same draws as before)
        return ((v.float() - (w1 - w0)) ** 2 * mask).sum() / mask.sum()

    @torch.no_grad()
    def sample(self, h, mask, steps=10, temperature=0.8, span=None):
        # (the whole loop compiled into one graph on the GPU once DacTTS.enable_fast_decoder ran; the CPU keeps the
        # plain one. The noise is drawn here, outside it: the same seed gives the same durations either way)
        w = torch.randn(h.shape[:2], device=h.device) * temperature
        fast = self.__dict__.get("_fast_integrate") if h.is_cuda and not eager() else None
        return (fast or self.integrate)(w, h, mask, span, steps)

    def integrate(self, w, h, mask, span=None, steps=10):
        for i in range(steps):
            t = torch.full((h.shape[0],), i / steps, device=h.device)
            w = w + self.velocity(w, t, h, mask, span) / steps
        return w * mask


class GlobalCond(nn.Module):
    """Utterance-level conditioning: speaker identity, style exaggeration (0 narration .. 1 acting)
    volume / vocal effort (0..0.1 whisper, ~0.5 normal voice, 1 shouting) and speaking rate
    (0 slow, 0.5 speaker's normal pace, 1 fast)."""

    def __init__(self, cfg: Config, d: int = 256):
        super().__init__()
        self.d = d
        self.spk = nn.Embedding(cfg.n_speakers, d) if cfg.n_speakers > 1 else None
        self.style = nn.Sequential(nn.Linear(64, d), nn.SiLU(), nn.Linear(d, d)) if cfg.use_style else None
        self.volume = nn.Sequential(nn.Linear(64, d), nn.SiLU(), nn.Linear(d, d)) if cfg.use_volume else None
        self.lang = nn.Embedding(getattr(cfg, "n_langs", 8), d) if getattr(cfg, "use_lang", False) else None
        self.lang_mean = None  # (T2s) each language's mean voice row; with lang_Q: the voice vector's language part
        if getattr(cfg, "lang_steer", 0) > 0 and self.spk is not None and self.lang is not None:
            self.lang_mean = nn.Embedding(getattr(cfg, "n_langs", 8), d)
            self.register_buffer("lang_Q", torch.zeros(cfg.lang_steer, d))  # (all zero = no effect until the grow fills it)
        for name in ("mood", "dominance", "whisper", "clean", "band"):  # later controls: start with no effect
            m = None
            if getattr(cfg, f"use_{name}", False):
                m = nn.Sequential(nn.Linear(64, d), nn.SiLU(), nn.Linear(d, d))
                nn.init.zeros_(m[-1].weight)
                nn.init.zeros_(m[-1].bias)
            setattr(self, name, m)
        for name in ("sing", "bpm", "bpb", "key"):  # singing labels (cfg.use_music): start with no effect
            m = None
            if getattr(cfg, "use_music", False):
                m = nn.Sequential(nn.Linear(64, d), nn.SiLU(), nn.Linear(d, d))
                nn.init.zeros_(m[-1].weight)
                nn.init.zeros_(m[-1].bias)
            setattr(self, name, m)
        self.genre = None
        if getattr(cfg, "use_music", False):  # multi-hot genre vector (dacts/music.py GENRES)
            self.genre = nn.Linear(getattr(cfg, "n_genres", 22), d)
            nn.init.zeros_(self.genre.weight)
            nn.init.zeros_(self.genre.bias)
        self.guid = None  # (cfg.use_guid: the guidance the teacher used, 1..5; starts with no effect)
        if getattr(cfg, "use_guid", False):
            self.guid = nn.Sequential(nn.Linear(64, d), nn.SiLU(), nn.Linear(d, d))
            nn.init.zeros_(self.guid[-1].weight)
            nn.init.zeros_(self.guid[-1].bias)
        self.spk_enc = None
        if getattr(cfg, "zs_weight", 0) > 0 and self.spk is not None:  # zero-shot: reference x-vector -> voice vector
            self.spk_enc = nn.Sequential(nn.LayerNorm(cfg.zs_in), nn.Linear(cfg.zs_in, 1024), nn.GELU(), nn.Dropout(0.1),
                                         nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, d))
        self.speed = None
        if getattr(cfg, "use_speed", False):
            self.speed = nn.Sequential(nn.Linear(64, d), nn.SiLU(), nn.Linear(d, d))
            nn.init.zeros_(self.speed[-1].weight)  # starts with no effect, so it can be added to a trained model
            nn.init.zeros_(self.speed[-1].bias)

    def forward(self, spk=None, style=None, volume=None, speed=None, lang=None, keep=None, mood=None,
                dominance=None, whisper=None, clean=None, band=None, spk_override=None,
                sing=None, bpm=None, bpb=None, key=None, genre=None, guid=None):
        """Components passed as None are left out (e.g. no speaker -> voice not pinned).

        keep: optional {"spk"|"style"|"volume"|"speed": (B,) 0/1} per-sample masks used for condition
        dropout in training; a dropped component contributes exactly what None does (nothing),
        so leaving a label out at inference is a mode the model has trained on.
        """
        g = None
        parts = (("spk", self.spk, spk, lambda m, x: m(x)),
                 ("style", self.style, style, lambda m, x: m(timestep_embedding(x, 64))),
                 ("volume", self.volume, volume, lambda m, x: m(timestep_embedding(x, 64))),
                 ("speed", self.speed, speed, lambda m, x: m(timestep_embedding(x, 64))),
                 ("lang", self.lang, lang, lambda m, x: m(x)),
                 ("mood", self.mood, mood, lambda m, x: m(timestep_embedding(x, 64))),
                 ("dominance", self.dominance, dominance, lambda m, x: m(timestep_embedding(x, 64))),
                 ("whisper", self.whisper, whisper, lambda m, x: m(timestep_embedding(x, 64))),
                 ("clean", self.clean, clean, lambda m, x: m(timestep_embedding(x, 64))),
                 ("band", self.band, band, lambda m, x: m(timestep_embedding(x, 64))),
                 ("sing", self.sing, sing, lambda m, x: m(timestep_embedding(x, 64))),
                 ("bpm", self.bpm, bpm, lambda m, x: m(timestep_embedding(x, 64))),
                 ("bpb", self.bpb, bpb, lambda m, x: m(timestep_embedding(x, 64))),
                 ("key", self.key, key, lambda m, x: m(timestep_embedding(x, 64))),
                 ("genre", self.genre, genre, lambda m, x: m(x.to(m.weight.dtype))),
                 ("guid", getattr(self, "guid", None), guid, lambda m, x: m(timestep_embedding(x / 5.0, 64))))
        for name, module, value, embed in parts:
            if module is None or value is None:
                continue
            e = embed(module, value)
            if name == "spk" and getattr(self, "spk_fix", None) is not None:  # (inference: the language leak fix, infer.py)
                f = self.spk_fix.to(e.dtype)
                e = e + (f.repeat(e.shape[0] // f.shape[0], 1) if f.shape[0] != e.shape[0] else f)
            if name == "spk" and spk_override is not None:  # (vectors, row mask): encoder voice instead of the table
                z, use = spk_override
                e = torch.where(use[:, None], z.to(e.dtype), e)
            if name == "spk" and getattr(self, "lang_mean", None) is not None and lang is not None:  # (T2s) the language
                q = self.lang_Q.to(e.dtype)  # part of the voice vector := the tag's language (table row and spk_enc alike)
                e = e - ((e - self.lang_mean(lang).to(e.dtype)) @ q.t()) @ q
            if keep is not None and name in keep:
                e = e * keep[name][:, None].to(e.dtype)
            g = e if g is None else g + e
        return g


PROMPT_LABELS = ("whisper", "style", "mood", "dominance", "volume", "speed", "clean", "band")


CLAB_DIM = 4 * len(PROMPT_LABELS)  # per label: value, known, (clip - asked) difference, difference known


def clip_label_features(lab, target=None):
    """Per-position labels of the reference clip each position came from -> features.
    lab (B, T, L) in [0, 1], NaN = unknown / not a reference position; target (B, L) the labels asked for the new
    line, NaN = not asked. -> (B, T, 4L): [value * known, known, (value - target) * both known, both known].
    A position with no clip labels at all is all zeros (exactly "no input")."""
    lab = lab.float()
    known = torch.isfinite(lab)  # (inf = unknown too, as NaN)
    v = torch.where(known, lab, torch.zeros_like(lab))
    if target is None:
        tk, tv = torch.zeros_like(known), torch.zeros_like(v)
    else:
        t = target.float()[:, None, :].expand_as(lab)
        tk = torch.isfinite(t)
        tv = torch.where(tk, t, torch.zeros_like(t))
    both = (known & tk).float()
    return torch.cat([v * known.float(), known.float(), (v - tv) * both, both], dim=-1)


def asked_labels(raw, keep=None, b=None, device=None):
    """The new line's labels as the model sees them, in PROMPT_LABELS order: (B, L), NaN = not given / dropped."""
    cols = []
    for k in PROMPT_LABELS:
        v = raw.get(k) if raw else None
        if v is None or not torch.is_tensor(v) or not v.is_floating_point():
            v = torch.full((b,), float("nan"), device=device)
        v = v.float().reshape(-1).to(device)
        if v.numel() == 1 and b and b > 1:
            v = v.expand(b)
        if keep is not None and k in keep:
            v = torch.where(keep[k].to(device) > 0, v, torch.full_like(v, float("nan")))
        cols.append(v)
    return torch.stack(cols, -1)


class PromptLabels(nn.Module):
    """How the prompt (the given reference audio a line continues from) was delivered: the line's own labels, for the
    prompt (cfg.use_prompt_labels, owner 2026-10-22). With "this reference is whispered / acted / noisy" as an input,
    the model can tell how the reference sounds from how the new line should sound (a whispered prompt + whisper 0 =
    this voice talking normally). Every label optional (NaN = unknown), all hidden together part of the time. Starts
    with no effect (zero last layer); added to the global condition only on rows that have a prompt."""

    def __init__(self, d: int = 256):
        super().__init__()
        n = len(PROMPT_LABELS)
        self.net = nn.Sequential(nn.Linear(n * 64 + n, d), nn.SiLU(), nn.Linear(d, d))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, labels, has, keep=None):
        """labels {name: (B,)} in [0, 1], NaN / missing = unknown; has (B,) bool: rows with a prompt;
        keep {name: (B,)} 0/1 (training dropout) or None."""
        b, dev = has.shape[0], has.device
        es, ms = [], []
        for k in PROMPT_LABELS:
            v = labels.get(k) if labels else None
            v = torch.full((b,), float("nan"), device=dev) if v is None else v.to(dev).float().reshape(b)
            ok = torch.isfinite(v)  # (inf = unknown too)
            if keep is not None and k in keep:
                ok = ok & (keep[k] > 0)
            es.append(timestep_embedding(torch.where(torch.isfinite(v), v, torch.full_like(v, 0.5)), 64) * ok[:, None])
            ms.append(ok)
        x = torch.cat(es + [torch.stack(ms, -1).float()], -1)
        w = self.net[0].weight
        return self.net(x.to(w.dtype)) * has[:, None].to(w.dtype)


NOTE_PERIODS = (0.5, 1.0, 2.0, 4.0, 6.0, 12.0, 24.0, 48.0)  # semitones (12: the note name, 24 / 48: the register)
NOTE_FEATS = 2 * len(NOTE_PERIODS) + 3
BEAT_FEATS = 9  # (P4, Solver 11-19) beat track per frame: sin/cos beat, sin/cos subdivision, sin/cos two beats, log2(bpm/100), triple, known
TECHNIQUES = ("Control", "Breathy", "Glissando", "Falsetto", "Mixed_Voice", "Pharyngeal", "Vibrato")  # (11-19) singing style, one-hot
TECH_FEATS = len(TECHNIQUES)


def note_features(notes, n_in=None):
    """(11-18, Solver) note track (B, T, 4) [(MIDI - 60) / 24, sung, onset, known] -> (B, T, 19): sin / cos of the pitch in
    semitones at NOTE_PERIODS (zero where nothing is sung) + the sung / onset / known flags. One scalar made neighbouring
    semitones 0.042 apart -- the singing branch learned the contour (r 0.67) but missed the note by ~1.7 semitones."""
    st = notes[..., 0].float() * 24.0
    sung = notes[..., 1:2].float()
    per = torch.tensor(NOTE_PERIODS, device=notes.device, dtype=torch.float32)
    ang = 2 * math.pi * st[..., None] / per
    out = torch.cat([torch.sin(ang) * sung, torch.cos(ang) * sung, notes[..., 1:4].float()], -1)
    if n_in is not None and n_in >= NOTE_FEATS + BEAT_FEATS:  # (P4) the beat track rides after the 4 note columns
        beat = notes[..., 4:4 + BEAT_FEATS].float() if notes.shape[-1] >= 4 + BEAT_FEATS else out.new_zeros(*out.shape[:-1], BEAT_FEATS)
        out = torch.cat([out, beat], -1)
    if n_in is not None and n_in == NOTE_FEATS + BEAT_FEATS + TECH_FEATS:  # (11-19) + the singing style (one-hot; zeros = unknown)
        a = 4 + BEAT_FEATS
        tech = notes[..., a:a + TECH_FEATS].float() if notes.shape[-1] >= a + TECH_FEATS else out.new_zeros(*out.shape[:-1], TECH_FEATS)
        out = torch.cat([out, tech], -1)
    return out


PCTX_DIM = 6  # [log length, pitch, pitch movement, voiced share, loudness, known]


def token_prosody(attn, contour, known_frames):
    """How each token was really said, for tokens whose audio is known (a prompt): (B, Tx, PCTX_DIM).

    attn (B, Tx, Ty) 0/1 token -> frame alignment; contour (B, Ty, 3) from dacts.contour (pitch in semitones
    re 100 Hz, voiced logit, loudness dB/10); known_frames (B, Ty) bool. A token counts as known when all its
    frames are (zero-length tokens: the frame at their position). Unknown tokens get all zeros."""
    a = attn.float()
    ty = a.shape[-1]
    dur = a.sum(-1)
    kf = known_frames.float()
    all_known = (a * (1 - kf)[:, None]).sum(-1) == 0
    pos = (dur.cumsum(-1) - dur / 2).long().clamp(0, ty - 1)  # frame at the token's middle / position
    known = torch.where(dur > 0, all_known, known_frames.gather(1, pos)) & (a.sum(-1) >= 0)
    vm = torch.sigmoid(contour[..., 1].float())
    vw = a * vm[:, None]
    vs = vw.sum(-1)
    p = contour[..., 0].float()[:, None]
    pitch = (vw * p).sum(-1) / vs.clamp(min=1e-3)
    move = ((vw * p ** 2).sum(-1) / vs.clamp(min=1e-3) - pitch ** 2).clamp(min=0).sqrt()
    has = (vs > 0.5).float()
    loud = (a * contour[..., 2].float()[:, None]).sum(-1) / dur.clamp(min=1)
    f = torch.stack([torch.log1p(dur) / 3, pitch / 12 * has, move / 6 * has, vs / dur.clamp(min=1), loud,
                     torch.ones_like(dur)], -1)
    return f * known[..., None].float()


class TextEncoder(nn.Module):
    def __init__(self, cfg: Config, vocab_size: int):
        super().__init__()
        d = cfg.enc_dim
        self.heads = cfg.enc_heads
        self.emb = nn.Embedding(vocab_size, d, padding_idx=0)
        nn.init.normal_(self.emb.weight, 0.0, d**-0.5)
        self.emph = None  # per-token emphasis 0..2 (only where specified); zero-init: no effect until trained
        if getattr(cfg, "use_emph", False):
            self.emph = nn.Sequential(nn.Linear(64, d), nn.SiLU(), nn.Linear(d, d))
            nn.init.zeros_(self.emph[-1].weight)
            nn.init.zeros_(self.emph[-1].bias)
        self.pctx = None  # prompt prosody per token (DacTTS.token_prosody); zero-init: no effect until trained
        if getattr(cfg, "use_pctx", False):
            self.pctx = nn.Sequential(nn.Linear(PCTX_DIM, d), nn.SiLU(), nn.Linear(d, d))
            nn.init.zeros_(self.pctx[-1].weight)
            nn.init.zeros_(self.pctx[-1].bias)
        self.tlab = None  # per-clip reference labels on the reference words (cfg.use_clip_labels); zero-init
        if getattr(cfg, "use_clip_labels", False):
            self.tlab = nn.Linear(CLAB_DIM, d, bias=False)
            nn.init.zeros_(self.tlab.weight)
        self.g_proj = nn.Linear(256, d)
        self.prenet = nn.ModuleList([ConvLN(d, d, 5, cfg.dropout) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(d, cfg.enc_heads, cfg.ffn_mult, cfg.dropout) for _ in range(cfg.enc_layers)])
        self.norm = nn.LayerNorm(d)
        self.proj_mu = nn.Linear(d, cfg.token_dim)
        # (cfg.swift_rm: no old head at all -- DacTTS.rhythm, Swift-RM, is the duration model)
        self.dur = None if getattr(cfg, "swift_rm", False) else FlowDurationPredictor(d, dropout=cfg.dropout)
        if self.dur is not None and getattr(cfg, "use_dur_span", False):
            self.dur.add_span()
        self.lm_in = None
        if getattr(cfg, "lm_name", ""):  # cross-attention to frozen LM features of (line, previous line)
            self.lm_in = nn.Linear(cfg.lm_dim, d)
            self.lm_seg = nn.Embedding(2, d)  # 0 = current line, 1 = previous line
            self.lm_norm = nn.LayerNorm(d)
            self.xattn = nn.ModuleList([CrossBlock(d, cfg.enc_heads) for _ in range(cfg.lm_layers)])

    def forward(self, tokens, mask, g=None, lm=None, emph=None, pctx=None, tlab=None):
        """lm: optional (features (B, L, lm_dim), mask (B, L), segment ids (B, L)) from dacts.lm.TextLM.
        pctx: optional (B, Tx, PCTX_DIM) prosody of the tokens whose audio is known (prompt), zeros elsewhere.
        tlab: optional (B, Tx, CLAB_DIM) clip_label_features of the reference words, zeros elsewhere."""
        x = self.emb(tokens)
        if tlab is not None and self.tlab is not None:
            x = x + self.tlab(tlab.to(x.dtype))
        if pctx is not None and self.pctx is not None:
            x = x + self.pctx(pctx.to(x.dtype))
        if emph is not None and self.emph is not None:  # (values (B,Tx) 0..2, specified (B,Tx) bool)
            ev, es = emph
            x = x + self.emph(timestep_embedding(ev.flatten() / 2, 64).to(x.dtype)).view(*ev.shape, -1) * es[..., None].to(x.dtype)
        if g is not None:
            x = x + self.g_proj(g)[:, None]
        h = x
        for layer in self.prenet:
            h = layer(h, mask)
        x = x + h
        rot = rope(x.shape[1], x.shape[-1] // self.heads, x.device)
        for blk in self.blocks:
            x = blk(x, mask, rot)
        if self.lm_in is not None and lm is not None:
            feats, lm_mask, seg = lm
            mem = self.lm_norm(self.lm_in(feats.to(x.dtype)) + self.lm_seg(seg))
            for xb in self.xattn:
                x = xb(x, mem, lm_mask)
        x = self.norm(x)
        mu = self.proj_mu(x) * mask[..., None]
        return mu, x


class FlowDecoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d, c = cfg.dec_dim, cfg.token_dim
        self.heads = cfg.dec_heads
        # input: noisy latents + aligned prior mean + aligned encoder states (richer than mu alone)
        self.inp = nn.Linear(2 * c + cfg.enc_dim, d)
        # prompted cloning: [known latents, known flag] per frame; no bias + zero init, so "no prompt"
        # (all zeros) is exactly the model without this input, and costs nothing when skipped
        self.ctx_in = None
        if getattr(cfg, "use_prompt", False):
            self.ctx_in = nn.Linear(c + 1, d, bias=False)
            nn.init.zeros_(self.ctx_in.weight)
        self.clab_in = None  # per-clip reference labels per known frame (cfg.use_clip_labels), after ctx's c + 1
        if getattr(cfg, "use_clip_labels", False):
            self.clab_in = nn.Linear(CLAB_DIM, d, bias=False)
            nn.init.zeros_(self.clab_in.weight)
        self.note_in = None
        if getattr(cfg, "use_notes", False):  # notes: [pitch, sung, onset, known] per frame, starts with no effect
            # (11-18, singing branch) cfg.note_sincos: the pitch as sin / cos features (note_features) instead of one number
            self.note_in = nn.Linear((NOTE_FEATS if getattr(cfg, "note_sincos", False) else 4) + (BEAT_FEATS if getattr(cfg, "note_beat", False) else 0)
                                     + (TECH_FEATS if getattr(cfg, "note_tech", False) else 0), d, bias=False)
            nn.init.zeros_(self.note_in.weight)
        self.beat_in = None
        if getattr(cfg, "use_beat", False):  # metronome: beat + bar position per frame (sin/cos), starts with no effect
            self.beat_in = nn.Linear(4, d, bias=False)
            nn.init.zeros_(self.beat_in.weight)
        self.t_mlp = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        self.g_proj = nn.Linear(256, d)
        self.blocks = nn.ModuleList(
            [Block(d, cfg.dec_heads, cfg.ffn_mult, cfg.dropout, cond_dim=d) for _ in range(cfg.dec_layers)]
        )
        self.skip_up = None
        if getattr(cfg, "dec_skips", False):  # U-Net long skips: second-half block i <- output of block (L-1-i)
            half, r = cfg.dec_layers // 2, getattr(cfg, "dec_skip_rank", 256)
            self.skip_norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(half)])
            self.skip_down = nn.ModuleList([nn.Linear(d, r, bias=False) for _ in range(half)])
            self.skip_up = nn.ModuleList([nn.Linear(r, d, bias=False) for _ in range(half)])
            for m in self.skip_up:
                nn.init.zeros_(m.weight)  # starts as no skip at all: a grown model is unchanged
        self.ref_x = None  # v14 reference segment: cross-attention to its frames (zero-init: a grown model is unchanged)
        if getattr(cfg, "use_ref_seg", False):
            self.ref_every = cfg.ref_seg_every
            self.ref_x = nn.ModuleList([CrossBlock(d, cfg.dec_heads) for _ in range(cfg.dec_layers // cfg.ref_seg_every)])
        # (2026-11-13, you: "fix the top suspect") v16 XL looked at the reference only twice (1024 x 11, every 4th) where
        # v15 XL did 11 times: cfg.ref_x_all adds a look-up after every OTHER block as well, in a separate list so the
        # trained ref_x keep their names and places (and their optimizer state); zero-init, so the model is unchanged
        self.ref_x2, self.ref_x2_idx = None, None
        if self.ref_x is not None and getattr(cfg, "ref_x_all", False):
            own = [i for i in range(cfg.dec_layers)
                   if not ((i + 1) % self.ref_every == 0 and i // self.ref_every < len(self.ref_x))]
            self.ref_x2_idx = [own.index(i) if i in own else -1 for i in range(cfg.dec_layers)]
            self.ref_x2 = nn.ModuleList([CrossBlock(d, cfg.dec_heads) for _ in own])
        self.norm_out = nn.LayerNorm(d, elementwise_affine=False)
        self.ada_out = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, c)
        self.use_flex = False  # FlexAttention in the blocks (set by train.py for cfg.flex_attn; not saved)
        for m in (self.ada_out, self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x_t, t, cond, mask, g=None, tap=None, ctx=None, beat=None, notes=None, ref=None):
        """cond: (B, T, token_dim + enc_dim) = [aligned mu, aligned encoder states]; g: global cond.
        ctx: (B, T, token_dim + 1) known latents + known flag (prompted cloning), or None.
        ref: (memory (B, Tr, dec_dim), mask (B, Tr), on (B,)) from DacTTS.ref_memory, or None."""
        h = self.inp(torch.cat([x_t, cond.to(x_t.dtype)], dim=-1))
        if ctx is not None and self.ctx_in is not None:
            k = self.ctx_in.in_features
            h = h + self.ctx_in(ctx[..., :k].to(h.dtype))
            if ctx.shape[-1] > k and self.clab_in is not None:  # [.., clip label features] (cfg.use_clip_labels)
                h = h + self.clab_in(ctx[..., k:].to(h.dtype))
        if beat is not None and self.beat_in is not None:  # (B, T, 4); rows without a tempo are all zero
            h = h + self.beat_in(beat.to(h.dtype))
        if notes is not None and self.note_in is not None:  # (B, T, 4); unknown frames are all zero
            if self.note_in.in_features != notes.shape[-1] or notes.shape[-1] > 4:
                notes = note_features(notes, self.note_in.in_features)
            h = h + self.note_in(notes.to(h.dtype))
        c = self.t_mlp(timestep_embedding(t).to(self.t_mlp[0].weight.dtype))
        if g is not None:
            c = c + self.g_proj(g)
        rot = rope(h.shape[1], h.shape[-1] // self.heads, h.device)
        bm = None
        if self.use_flex and h.is_cuda:  # one block mask for all blocks: every query sees the clip's real frames
            # padded to whole 128-frame blocks: the block mask is built over that grid, and reading past the
            # end of the key mask was an illegal memory access
            km = F.pad(mask.bool(), (0, (-h.shape[1]) % 128))
            bm = create_block_mask(lambda b, hh, q, kv: km[b, kv], h.shape[0], None, h.shape[1], h.shape[1], device=h.device)
        tapped = []
        n, saved = len(self.blocks), []
        half = n // 2 if self.skip_up is not None else 0
        for i, blk in enumerate(self.blocks):
            if half and i >= n - half:  # second half: + the mirror block's output (block n-1-i)
                k = i - (n - half)
                h = h + self.skip_up[k](self.skip_down[k](self.skip_norm[k](saved[n - 1 - i])))
            h = blk(h, mask, rot, c, bm)
            if ref is not None and self.ref_x is not None and (i + 1) % self.ref_every == 0 and i // self.ref_every < len(self.ref_x):
                h = self.ref_x[i // self.ref_every](h, ref[0], ref[1], ref[2])
            elif ref is not None and self.ref_x2 is not None and self.ref_x2_idx[i] >= 0:
                h = self.ref_x2[self.ref_x2_idx[i]](h, ref[0], ref[1], ref[2])
            if i < half:
                saved.append(h)
            # (any(==), not `in`: compiled with dynamic=True, `i in tap` was always False and nothing got tapped, 2026-10-29)
            if tap is not None and (any(i == j for j in tap) if isinstance(tap, tuple) else i == tap):
                tapped.append(h)
        sh, sc = self.ada_out(F.silu(c))[:, None].chunk(2, dim=-1)
        out = self.out(self.norm_out(h) * (1 + sc) + sh) * mask[..., None]
        return out if tap is None else (out, tapped if isinstance(tap, tuple) else tapped[0])


class DeliveryEncoder(nn.Module):
    """Reference latents (another clip of the same speaker, a few seconds) -> a delivery vector added to the
    global conditioning, so text encoder, durations and decoder all hear HOW the person talks (energy, pace,
    pitch swings), not just what they sound like. Zero-initialised output: adding it changes nothing at first."""

    def __init__(self, c: int, d_model: int = 256, d_out: int = 256, layers: int = 3, heads: int = 4):
        super().__init__()
        self.inp = nn.Linear(c, d_model)
        self.blocks = nn.ModuleList([Block(d_model, heads, 2, 0.1) for _ in range(layers)])
        self.pool = nn.Linear(d_model, 1)
        self.out = nn.Linear(d_model, d_out)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, ref, mask):
        """ref: (B, T, c) normalized latents, mask: (B, T) bool -> (B, d_out)."""
        h = self.inp(ref)
        rot = rope(h.shape[1], h.shape[-1] // self.blocks[0].attn.h, h.device)
        for b in self.blocks:
            h = b(h, mask, rot)
        a = self.pool(h).squeeze(-1).masked_fill(~mask, -1e4).softmax(-1)
        return self.out((a[..., None] * h).sum(1))


class VoiceRefEncoder(nn.Module):
    """Reference audio latents (other clips of the speaker, 3-20 s) -> a voice vector in the speaker table's space.
    Reads the sound itself, unlike the x-vector path (a speaker-verification summary that drops much of how a voice
    sounds); trained end to end through the TTS loss, plus a pull toward the speaker's table row."""

    def __init__(self, c: int, d_out: int, d_model: int = 384, layers: int = 4, heads: int = 6):
        super().__init__()
        self.inp = nn.Linear(c, d_model)
        self.blocks = nn.ModuleList([Block(d_model, heads, 2, 0.1) for _ in range(layers)])
        self.pool = nn.Linear(d_model, 4)  # 4 attention-pooling heads
        self.out = nn.Sequential(nn.LayerNorm(4 * d_model), nn.Linear(4 * d_model, 1024), nn.GELU(), nn.Linear(1024, d_out))

    def forward(self, ref, mask):
        h = self.inp(ref)
        rot = rope(h.shape[1], h.shape[-1] // self.blocks[0].attn.h, h.device)
        for b in self.blocks:
            h = b(h, mask, rot)
        a = self.pool(h).masked_fill(~mask[..., None], -1e4).softmax(1)  # B, T, 4
        return self.out(torch.einsum("btk,btd->bkd", a, h).flatten(1))


class RefSegEncoder(nn.Module):
    """v14 reference segment: other clips of the speaker (latents at the model's token width) -> per-frame features
    at the decoder width, which the decoder cross-attends to (FlowDecoder.ref_x). Frame by frame, not pooled: the
    decoder can look up how this voice sounds on the sounds it is making."""

    def __init__(self, c: int, d_out: int, d_model: int = 384, layers: int = 4, heads: int = 6, tokens: int = 0):
        super().__init__()
        self.inp = nn.Linear(c, d_model)
        self.blocks = nn.ModuleList([Block(d_model, heads, 2, 0.1) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, d_out)
        # tokens > 0 (owner 2026-10-28: "something that tells something about the voice and itself also runs very
        # fast"): learned queries read the encoded frames once, so the decoder attends to `tokens` voice tokens
        # instead of every reference frame (up to 1000) -- made once per voice at inference
        self.tokens = tokens
        if tokens:
            self.query = nn.Parameter(torch.randn(tokens, d_model) * 0.02)
            self.pool = nn.MultiheadAttention(d_model, heads, batch_first=True)
            self.pool_norm = nn.LayerNorm(d_model)

    def forward(self, ref, mask):
        h = self.inp(ref)
        rot = rope(h.shape[1], h.shape[-1] // self.blocks[0].attn.h, h.device)
        for b in self.blocks:
            h = b(h, mask, rot)
        if not self.tokens:
            return self.out(self.norm(h)) * mask[..., None]
        h = self.norm(h)
        q = self.query[None].expand(h.shape[0], -1, -1).to(h.dtype)
        t = self.pool(q, h, h, key_padding_mask=~mask, need_weights=False)[0]
        return self.out(self.pool_norm(t + q))


class DacTTS(nn.Module):
    def __init__(self, cfg: Config, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.encoder = TextEncoder(cfg, vocab_size)
        if getattr(cfg, "swift_rm", False):  # v16: the rhythm model is part of the model (state_dict "rhythm.*")
            from .rhythm import SwiftRM
            self.rhythm = SwiftRM(h_dim=cfg.enc_dim, g_dim=256, d=cfg.swift_d, layers=cfg.swift_layers,
                                  heads=max(1, cfg.swift_d // 64), dropout=0.1)
        self.decoder = FlowDecoder(cfg)
        self.global_cond = GlobalCond(cfg)
        # dims of a patchified token that belong to the first `align_q` codebooks of each frame
        lat = cfg.latent_dim
        n_align = lat if getattr(cfg, "codec", "dac") == "kvae" else min(cfg.align_q, cfg.n_q) * 8  # KVAE: all dims
        idx = [f * lat + d for f in range(cfg.patch) for d in range(n_align)]
        self.register_buffer("align_idx", torch.tensor(idx), persistent=False)
        if getattr(cfg, "sing_vowel_prior", 0.0) > 0:  # which token ids are vowels (sung alignment prior)
            from . import text as _T
            self.register_buffer("vowel_tok", torch.tensor(
                [_T._INV.get(i, "") in _T._VOWEL_CHARS for i in range(vocab_size)]), persistent=False)
        self.repa_proj = None
        if getattr(cfg, "repa_weight", 0) > 0:  # training only: maps a decoder layer to speech-SSL features
            d = cfg.dec_dim
            self.repa_proj = nn.Sequential(nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, 2 * d), nn.SiLU(),
                                           nn.Linear(2 * d, cfg.repa_dim))
        self.spk_repa_proj, self.spk_repa_layers = None, ()
        if getattr(cfg, "spk_repa_weight", 0) > 0:  # training only: middle decoder layers -> the speaker's x-vector
            n_ = cfg.dec_layers
            self.spk_repa_layers = tuple(sorted({int(v) for v in cfg.spk_repa_layers.split(",") if v.strip()})
                                         or (n_ // 6, n_ // 3, n_ // 2))
            d = cfg.dec_dim
            self.spk_repa_proj = nn.Sequential(nn.Linear(d * len(self.spk_repa_layers), 2 * d), nn.SiLU(),
                                               nn.Linear(2 * d, cfg.zs_in))
        self.vtv_in = None  # voice-to-voice (dacts/vtv.py): ContentVec content + contour -> the decoder's conditioning
        if getattr(cfg, "use_vtv", False):
            from .vtv import CV_DIM
            self.vtv_in = nn.Sequential(nn.LayerNorm(CV_DIM + 3), nn.Linear(CV_DIM + 3, cfg.token_dim + cfg.enc_dim))
        # "exact pitch" mode (you 2026-10-31: "singing mode does it exactly, it doesnt have to be exact for speech"): a
        # vector added to a VTV row's conditioning when the output must hit the given pitch frame by frame; zero at
        # the start, so a model it is added to behaves exactly as before until it learns the mode (cfg.vtv_exact_prob)
        self.vtv_exact = nn.Parameter(torch.zeros(cfg.token_dim + cfg.enc_dim)) \
            if self.vtv_in is not None and getattr(cfg, "vtv_exact_prob", 0) > 0 else None
        self.dlv_enc = DeliveryEncoder(cfg.token_dim) if getattr(cfg, "use_dlv", False) else None
        self.prompt_lab = PromptLabels(self.global_cond.d) if getattr(cfg, "use_prompt_labels", False) else None
        self.voice_enc = (VoiceRefEncoder(cfg.token_dim, self.global_cond.spk.weight.shape[1])
                          if getattr(cfg, "use_vre", False) and cfg.n_speakers > 1 else None)
        self.ref_enc = (RefSegEncoder(cfg.token_dim, cfg.dec_dim, cfg.ref_seg_dim, cfg.ref_seg_layers,
                                      tokens=getattr(cfg, "ref_seg_tokens", 0))
                        if getattr(cfg, "use_ref_seg", False) else None)
        self.code_head = None
        if getattr(cfg, "code_head", False):  # picks the actual DAC codes (dacts/codehead.py)
            from .codehead import CodeHead
            self.code_head = CodeHead(cfg)

    def ref_memory(self, lat, mask, on=None):
        """Reference segment latents (B, Tr, token_dim) normalized + frame mask (B, Tr) -> the decoder's `ref` input.
        on (B,) bool: rows that use it (default: rows with any reference frame)."""
        has = mask.any(-1)
        on = has if on is None else on & has
        mask = mask.clone()
        mask[:, 0] |= ~has  # (a row without frames attends to one zero frame, gated off: no NaN)
        mem = self.ref_enc(lat, mask)
        if mem.shape[1] != mask.shape[1]:  # voice tokens: every token is real
            mask = torch.ones(mem.shape[:2], dtype=torch.bool, device=mem.device)
        return mem, mask, on.float()

    @property
    def fast_decoder(self):
        """The decoder for inference: its torch.compile'd copy once enable_fast_decoder() ran, else the decoder."""
        fast = self.__dict__.get("_fast_decoder")
        return self.decoder if fast is None or eager() else fast

    def text_encoder(self, on_gpu: bool):
        """The text encoder for inference: its compiled copy on the GPU (enable_fast_decoder), else the plain one."""
        fast = self.__dict__.get("_fast_encoder") if on_gpu and not eager() else None
        return self.encoder if fast is None else fast

    def enable_fast_decoder(self):
        """Inference: compile the decoder (Inductor, dynamic lengths), used only while the model is on the GPU -- on the
        CPU the plain decoder runs (the website parks models on the CPU). ~1.3-2x per line, measured 2026-10-21.
        Kept outside the module tree (no extra state_dict keys); the first line of each new shape compiles (~seconds).
        Also the text encoder and the duration sampler's 10-step loop (every line, at every quality): eager, they were
        ~8% of a take, mostly kernel launches (evals/stage_profile.py, 2026-10-24)."""
        self.__dict__["_fast_decoder"] = torch.compile(self.decoder, dynamic=True)
        self.__dict__["_fast_encoder"] = torch.compile(self.encoder, dynamic=True)
        if self.encoder.dur is not None:
            self.encoder.dur.__dict__["_fast_integrate"] = torch.compile(self.encoder.dur.integrate, dynamic=True)
        return self

    def compile_modules(self):
        """Compile the heavy parts with Inductor (Triton kernels); MAS stays outside the graph."""
        self.encoder.compile(dynamic=True)
        if self.encoder.dur is not None:
            self.encoder.dur.compile(dynamic=True)
        else:  # (Swift-RM: its velocity network, as the old head's)
            self.rhythm.compile(dynamic=True)
        self.decoder.compile(dynamic=True)
        if self.ref_enc is not None:
            self.ref_enc.compile(dynamic=True)
        # 2026-10-30 (speedups/REPORT.md): these ran eager, ~1200 kernel launches a step on a launch-bound loop: the
        # voice / label conditioning, the prompt labels, the delivery encoder
        for m in (self.global_cond, self.prompt_lab, self.dlv_enc):
            if m is not None:
                m.compile(dynamic=True)

    def forward(self, tokens, x_len, y, y_len, spk=None, style=None, volume=None, speed=None, lang=None, codes=None,
                lm=None, labels=None, repa=None, ref=None, dlv=None, word_ids=None, contour=None, corrupt=None,
                known_len=None, vre=None, vre_use=0.0, notext=None, notes=None, prompt_labels=None,
                prompt_patch_labels=None, prompt_token_labels=None, hints=None, vtv=None):
        """corrupt: (damaged latents (B,Ty,D) like y, has (B,) bool) for prompt-robustness training, or None.
        vtv: (content (B,Ty,768), rows (B,) bool) from dacts.vtv: those rows are conditioned on the recording's content
        + contour instead of the text (voice-to-voice), or None.
        repa: (B, Ty, repa_dim) speech-SSL targets per frame (REPA), or None.
        lm: LM features of (line, previous line's text); labels: {"mood"|"dominance"|"whisper": (B,)}.
        NaN in any label tensor = that clip has no such label (treated like a dropped label)."""
        labels = dict(labels or {})
        raw_labels = {"style": style, "volume": volume, "speed": speed, **labels}  # (before NaN -> dropped below)
        """y: (B, Ty, token_dim) normalized, patchified target latents; y_len in patches.
        spk: (B,) speaker ids; style, volume, speed: (B,) in [0, 1]."""
        x_mask = seq_mask(x_len, tokens.shape[1])
        y_mask = seq_mask(y_len, y.shape[1])
        keep = None
        if self.training and getattr(self.cfg, "label_drop", 0) > 0:
            b = tokens.shape[0]
            # also drop *all* labels together sometimes: independent drops alone almost never leave a
            # clip fully unlabeled, which is exactly how it is used with every label left off
            all_kept = (torch.rand(b, device=tokens.device) >= getattr(self.cfg, "label_drop_all", 0.0)).float()
            keep = {k: (torch.rand(b, device=tokens.device) >= self.cfg.label_drop).float() * all_kept
                    for k in ("spk", "style", "volume", "speed", "lang", "mood", "dominance", "whisper", "clean", "band",
                              "sing", "bpm", "bpb", "key", "genre")}  # owner: every label gets dropout (unconditional too)
            if getattr(self.cfg, "lang_never_drop", False):  # (T1: the language tag is always given)
                keep.pop("lang")
        named = {"style": style, "volume": volume, "speed": speed, **labels}
        for k, v in named.items():  # missing labels (NaN) -> dropped for that clip
            if v is not None and v.is_floating_point():
                # (isfinite: an infinite label is dropped like a missing one, code review 2026-10-29 -- see v13xl's speed=inf)
                ok = torch.isfinite(v) if v.dim() == 1 else torch.isfinite(v).all(-1)  # vector labels (genre): per row
                all_ok = (hints or {}).get("labels_ok", {}).get(k)  # (train.py knows it from the CPU batch: no wait)
                if not (bool(ok.all()) if all_ok is None else all_ok):
                    keep = dict(keep) if keep is not None else {}
                    keep[k] = keep.get(k, torch.ones(v.shape[0], device=v.device)) * ok.float()
                    f_ = 0.5 if v.dim() == 1 else 0.0
                    named[k] = torch.nan_to_num(v, nan=f_, posinf=f_, neginf=f_)
        style, volume, speed = named["style"], named["volume"], named["speed"]
        labels = {k: named[k] for k in labels}
        p_ex = getattr(self.cfg, "spk_drop_expressive", 0.0)
        if self.training and p_ex > 0 and style is not None:
            # expressive clips, speaker hidden, style label kept: the model learns what acting does on its own, so
            # style_transfer (speaker-free style direction at inference) works for voices that only ever read
            b = tokens.shape[0]
            ok_style = keep["style"] if keep is not None and "style" in keep else torch.ones(b, device=tokens.device)
            hide = (style >= getattr(self.cfg, "expressive_style", 0.7)) & (ok_style > 0) & (torch.rand(b, device=tokens.device) < p_ex)
            keep = dict(keep) if keep is not None else {}
            keep["spk"] = keep.get("spk", torch.ones(b, device=tokens.device)) * (~hide).float()
            keep["style"] = torch.maximum(keep.get("style", torch.ones(b, device=tokens.device)), hide.float())
        p_sing = getattr(self.cfg, "sing_spk_drop", 0.0)
        if self.training and p_sing > 0 and labels.get("sing") is not None:
            # sung clips with the singer hidden (sing label kept): the model learns what singing is on its own, so any
            # voice -- including speakers who never sang -- can sing (same idea as spk_drop_expressive for acting)
            b = tokens.shape[0]
            ok_sing = keep["sing"] if keep is not None and "sing" in keep else torch.ones(b, device=tokens.device)
            hide = (labels["sing"] > 0.75) & (ok_sing > 0) & (torch.rand(b, device=tokens.device) < p_sing)
            keep = dict(keep) if keep is not None else {}
            keep["spk"] = keep.get("spk", torch.ones(b, device=tokens.device)) * (~hide).float()
            keep["sing"] = torch.maximum(keep.get("sing", torch.ones(b, device=tokens.device)), hide.float())
        prompted, cross = None, None
        if known_len is not None and (bool((known_len > 0).any()) if (hints or {}).get("cross_any") is None
                                      else hints["cross_any"]):  # cross-clip rows: clip A given, clip B learned
            cross = known_len > 0
        if self.training and getattr(self.cfg, "use_prompt", False):
            b = tokens.shape[0]
            prompted = (torch.rand(b, device=tokens.device) < self.cfg.prompt_prob) & (y_len * self.cfg.patch >= 100)  # (y_len: steps)
            if cross is not None:
                prompted = prompted & ~cross
            hide = (prompted if cross is None else prompted | cross) & (torch.rand(b, device=tokens.device) < self.cfg.prompt_spk_drop)
            keep = dict(keep) if keep is not None else {}
            keep["spk"] = keep.get("spk", torch.ones(b, device=tokens.device)) * (~hide).float()
        zs_out, override = {}, None
        if ref is not None and self.global_cond.spk_enc is not None and spk is not None:
            ok = torch.isfinite(ref).all(-1)
            if bool(ok.any()) if (hints or {}).get("ref_any") is None else hints["ref_any"]:
                z = self.global_cond.spk_enc(torch.nan_to_num(ref, nan=0.0, posinf=0.0, neginf=0.0).float())
                with torch.autocast(device_type=ref.device.type, enabled=False):
                    target = self.global_cond.spk.weight[spk].detach().float()
                    err = ((z.float() - target) ** 2).mean(-1) / target.pow(2).mean().clamp(min=1e-6)
                    zs_out["zs"] = (err * ok).sum() / ok.sum()
                use = ok & (torch.rand(ok.shape, device=ok.device) < getattr(self.cfg, "zs_prob", 0.5))
                override = (z, use)
        ref_seg = None
        if vre is not None and self.ref_enc is not None:  # v14 reference segment (patched like the targets by train.py)
            lat, m = vre
            on = torch.rand(m.shape[0], device=m.device) < self.cfg.ref_seg_prob
            ref_seg = self.ref_memory(lat.to(y.dtype), m, on)
        if vre is not None and self.voice_enc is not None and spk is not None:  # voice from reference audio
            lat, m = vre
            ok = m.any(-1)
            if bool(ok.any()):
                zv = self.voice_enc(lat, m)
                with torch.autocast(device_type=lat.device.type, enabled=False):
                    target = self.global_cond.spk.weight[spk].detach().float()
                    err = ((zv.float() - target) ** 2).mean(-1) / target.pow(2).mean().clamp(min=1e-6)
                    zs_out["vre"] = (err * ok).sum() / ok.sum()
                use_v = ok & (torch.rand(ok.shape, device=ok.device) < vre_use)  # 0 while it warms up
                if override is None:
                    override = (zv, use_v)
                else:
                    override = (torch.where(use_v[:, None], zv.to(override[0].dtype), override[0]), use_v | override[1])
        spk_rep = None
        if self.spk_repa_proj is not None and ref is not None:  # Speaker-REPA targets: OTHER clips' x-vectors of the voice
            ok_r = torch.isfinite(ref).all(-1)
            if override is not None:  # (rows whose voice vector was made from that same x-vector: the loss would be trivial)
                ok_r = ok_r & ~override[1]
            spk_rep = (torch.nan_to_num(ref, nan=0.0, posinf=0.0, neginf=0.0).float(), ok_r)
        g = self.global_cond(spk, style, volume, speed, lang, keep, **labels, spk_override=override)
        zs_rows = None  # zero-shot rows as the site clones: a prompt from ANOTHER recording, and no learned table vector
        if cross is not None and getattr(self.cfg, "critic_zs", False):  # (hidden, or the encoder's vector from a clip)
            spk_off = (keep["spk"] <= 0) if keep is not None and "spk" in keep else torch.zeros_like(cross)
            zs_rows = cross & (spk_off | (override[1] if override is not None else torch.zeros_like(cross)))
        nt = notext.to(y.device) if notext is not None and (bool(notext.any()) if (hints or {}).get("notext_any") is None
                                                            else hints["notext_any"]) else None
        if self.prompt_lab is not None:  # how the prompt was delivered (cfg.use_prompt_labels)
            b = tokens.shape[0]
            has = torch.zeros(b, dtype=torch.bool, device=tokens.device)
            own = torch.zeros_like(has)  # the prompt is part of this clip: its labels are the clip's own
            for m_, same in ((prompted, True), (nt, True), (cross, False)):
                if m_ is not None:
                    has, own = has | m_, (own | m_) if same else (own & ~m_)
            if hints is not None or bool(has.any()):  # (rows without a prompt get exactly 0 from prompt_lab)
                given = prompt_labels or {}
                pl = {}
                for k in PROMPT_LABELS:
                    o, p_ = raw_labels.get(k), given.get(k)
                    v = torch.full((b,), float("nan"), device=tokens.device)
                    if p_ is not None:
                        v = p_.to(v.device).float()
                    if o is not None and o.is_floating_point():
                        v = torch.where(own, o.float(), v)
                    pl[k] = v
                pkeep = None
                if self.training:
                    all_kept = (torch.rand(b, device=tokens.device) >= getattr(self.cfg, "prompt_label_drop_all", 0.3)).float()
                    pkeep = {k: (torch.rand(b, device=tokens.device) >= getattr(self.cfg, "label_drop", 0.0)).float() * all_kept
                             for k in PROMPT_LABELS}
                e = self.prompt_lab(pl, has, pkeep)
                g = e if g is None else g + e.to(g.dtype)
        if dlv is not None and self.dlv_enc is not None:  # (ref latents (B,T,c), mask (B,T)): how this person talks
            ref_lat, ref_mask = dlv
            use = ref_mask.any(-1) & (torch.rand(ref_mask.shape[0], device=ref_mask.device) < self.cfg.dlv_prob)
            if hints is not None or bool(use.any()):  # (unused rows: * use = 0)
                e = self.dlv_enc(ref_lat.to(g.dtype if g is not None else ref_lat.dtype), ref_mask) * use[:, None]
                g = e if g is None else g + e.to(g.dtype)
        known = None
        edit_rows = None  # (speech editing) short-gap rows: their context stays the real recording (no damaged copy)
        if prompted is not None and (hints is not None or bool(prompted.any())):  # (none prompted: known all False)
            # prompted rows: a random 70-100% span is generated, the rest of the clip is given as the prompt
            b, L = y.shape[0], y_mask.sum(1)
            span = (L.float() * (0.7 + 0.3 * torch.rand(b, device=y.device))).long().clamp(min=1)
            start = (torch.rand(b, device=y.device) * (L - span + 1).float()).long()
            ep = float(getattr(self.cfg, "edit_span_prob", 0.0))
            if ep > 0:  # (11-22, speech editing) short gaps with audio on both sides: 5-25 % of the clip, start in 10-90 %
                short = torch.rand(b, device=y.device) < ep
                s_span = (L.float() * (0.05 + 0.20 * torch.rand(b, device=y.device))).long().clamp(min=10)
                lo = (L.float() * 0.1).long()
                room = (L - s_span - 2 * lo).clamp(min=0)
                s_start = lo + (torch.rand(b, device=y.device) * (room + 1).float()).long()
                span, start = torch.where(short, s_span, span), torch.where(short, s_start, start)
                edit_rows = short & prompted
                if corrupt is not None:  # (Solver 11-22) an edit continues the room / noise / microphone of its
                    corrupt = (corrupt[0], corrupt[1] & ~edit_rows)  # neighbours: never the damaged copy as its context
            idx = torch.arange(y_mask.shape[1], device=y.device)[None]
            known = y_mask & ~((idx >= start[:, None]) & (idx < (start + span)[:, None])) & prompted[:, None]
        if cross is not None:
            idx = torch.arange(y_mask.shape[1], device=y.device)[None]
            kc = y_mask & (idx < known_len[:, None])
            known = kc if known is None else known | kc
        if nt is not None:  # untranscribed rows: the first 20-50 % is given, the rest continues that voice
            idx = torch.arange(y_mask.shape[1], device=y.device)[None]
            k = (y_mask.sum(1).float() * (0.2 + 0.3 * torch.rand(nt.shape[0], device=y.device))).long()
            kn = y_mask & (idx < k[:, None]) & nt[:, None]
            known = kn if known is None else (known & ~nt[:, None]) | kn
        clab = tlab = None
        if self.decoder.clab_in is not None and known is not None:
            # per-clip reference labels: cross rows bring theirs (one row of labels per reference patch / word, from
            # the loader); a same-clip prompt (prompted / untranscribed rows) carries the row's own labels
            b, ty = y.shape[0], y.shape[1]
            lab = torch.full((b, ty, len(PROMPT_LABELS)), float("nan"), device=y.device)
            if prompt_patch_labels is not None:
                n = min(ty, prompt_patch_labels.shape[1])
                lab[:, :n] = prompt_patch_labels[:, :n].to(y.device).float()
            own = asked_labels(raw_labels, None, b, y.device)
            same = torch.zeros(b, dtype=torch.bool, device=y.device)
            for m_ in (prompted, nt):
                if m_ is not None:
                    same = same | m_
            if cross is not None:
                same = same & ~cross
            lab = torch.where((same[:, None] & known)[..., None], own[:, None, :].expand_as(lab), lab)
            tl = prompt_token_labels.to(y.device).float() if prompt_token_labels is not None else None
            if self.training:  # all of a row's clip labels hidden at once now and then (unlabelled references)
                hide = torch.rand(b, device=y.device) < getattr(self.cfg, "prompt_label_drop_all", 0.3)
                lab = torch.where(hide[:, None, None], torch.full_like(lab, float("nan")), lab)
                if tl is not None:
                    tl = torch.where(hide[:, None, None], torch.full_like(tl, float("nan")), tl)
            asked = asked_labels(raw_labels, keep, b, y.device)  # (what the model is told about the new line)
            clab = clip_label_features(lab, asked) * known[..., None].float()
            if tl is not None and self.encoder.tlab is not None:
                tlab = torch.zeros(b, tokens.shape[1], CLAB_DIM, device=y.device)
                n = min(tokens.shape[1], tl.shape[1])
                tlab[:, :n] = clip_label_features(tl[:, :n], asked)
        emph, pctx = None, None
        want_emph = self.training and self.encoder.emph is not None and word_ids is not None and contour is not None
        want_pctx = self.training and self.encoder.pctx is not None and known is not None and contour is not None
        if want_emph or want_pctx:
            # alignment from a first, gradient-free encoder pass: measures each word's prominence (pitch /
            # loudness / length vs the rest of the sentence) and how the known (prompt) tokens were said
            with torch.no_grad():
                mu0, _ = self.encoder(tokens, x_mask, g, lm, tlab=tlab)
                with torch.autocast(device_type=y.device.type, enabled=False):
                    a0 = mu0.float().index_select(-1, self.align_idx)
                    y0 = y.float().index_select(-1, self.align_idx)
                    lp = 2 * a0 @ y0.transpose(1, 2) - (y0 ** 2).sum(-1)[:, None, :] - (a0 ** 2).sum(-1)[:, :, None]
                    attn0 = maximum_path(lp, x_mask[:, :, None] & y_mask[:, None, :])
                if want_emph:
                    emph = self.prominence(attn0, word_ids, contour.float(), y_mask, nw=(hints or {}).get("word_max"))
                if want_pctx:
                    pctx = token_prosody(attn0, contour.float(), known) * x_mask[..., None].float()
        mu_x, h = self.encoder(tokens, x_mask, g, lm, emph=emph, pctx=pctx, tlab=tlab)

        with torch.autocast(device_type=y.device.type, enabled=False):
            mu_a = mu_x.detach().float().index_select(-1, self.align_idx)
            y_a = y.float().index_select(-1, self.align_idx)
            logp = 2 * mu_a @ y_a.transpose(1, 2) - (y_a**2).sum(-1)[:, None, :] - (mu_a**2).sum(-1)[:, :, None]
            vp = float(getattr(self.cfg, "sing_vowel_prior", 0.0))
            if vp > 0 and notes is not None and getattr(self, "vowel_tok", None) is not None:
                # sung frames inside a note: vowels get a bonus, so the hold lands on the vowel like a singer's (plain
                # MAS split sung words like speech and taught long m / n hums). Scale: the per-frame spread over tokens
                sungf = torch.zeros(y.shape[0], y.shape[1], device=y.device)
                n_ = min(sungf.shape[1], notes.shape[1])
                sungf[:, :n_] = (notes[:, :n_, 1].float() > 0).float() * (notes[:, :n_, 3].float() > 0).float()
                sungf = sungf * y_mask.float()
                if bool(sungf.any()):
                    xm = x_mask[:, :, None].float()
                    mean = (logp * xm).sum(1, keepdim=True) / xm.sum(1, keepdim=True).clamp(min=1)
                    sd = (((logp - mean) ** 2 * xm).sum(1) / xm.sum(1).clamp(min=1)).sqrt()  # B, Ty
                    scale = (sd * sungf).sum(1) / sungf.sum(1).clamp(min=1)  # B
                    vt = self.vowel_tok[tokens.clamp(min=0, max=self.vowel_tok.shape[0] - 1)].float() * x_mask.float()
                    logp = logp + vp * scale[:, None, None] * vt[:, :, None] * sungf[:, None, :]
            attn = maximum_path(logp, x_mask[:, :, None] & y_mask[:, None, :])  # B, Tx, Ty
            fd = (hints or {}).get("forced_dur")
            if fd is not None:  # (11-18, singing branch) stored alignments: each token covers its given frames (no MAS)
                fd = fd.to(y.device).long() * x_mask.long()
                ends = fd.cumsum(1)
                fr = torch.arange(y.shape[1], device=y.device)
                attn = ((fr[None, None, :] >= (ends - fd)[:, :, None]) & (fr[None, None, :] < ends[:, :, None])
                        & x_mask[:, :, None] & y_mask[:, None, :]).to(attn.dtype)

        dur = attn.sum(-1)
        tm = x_mask if nt is None else x_mask & ~nt[:, None]  # untranscribed rows: no text, no durations to learn
        nfe = None
        if notes is not None and getattr(self.cfg, "use_notes", False):
            nfe = torch.zeros(y.shape[0], y.shape[1], notes.shape[-1], device=y.device, dtype=torch.float32)  # (4, or 4 + beat track)
            n_ = min(nfe.shape[1], notes.shape[1])
            nfe[:, :n_] = notes[:, :n_].float()
            if self.training and getattr(self.cfg, "note_drop", 0) > 0:  # whole rows without notes: still sings / talks
                keep_n = (torch.rand(y.shape[0], device=y.device) >= self.cfg.note_drop).float()
                nfe = nfe * keep_n[:, None, None]
        span = None
        if self.encoder.dur is not None and self.encoder.dur.span_in is not None and nfe is not None and word_ids is not None:
            # rows sung with notes: each token is told how long its word (or the rest it belongs to) really lasted and
            # how many notes the word had; the model learns how singers spread that time over the letters
            has = (nfe[..., 3] > 0).any(1)  # B
            if bool(has.any()):
                with torch.no_grad():
                    wid = word_ids.to(y.device)
                    run = torch.cumsum(torch.cat([torch.zeros_like(wid[:, :1]), (wid[:, 1:] != wid[:, :-1]).long()], 1), 1)
                    on = (attn @ (nfe[..., 2] * y_mask.float())[..., None]).squeeze(-1)  # note onsets per token
                    gf = torch.zeros_like(dur).scatter_add_(1, run, dur * x_mask).gather(1, run)
                    gn = torch.zeros_like(on).scatter_add_(1, run, on * x_mask).gather(1, run)
                    pres = (has[:, None] & x_mask).float()
                    span = torch.stack([pres, torch.log1p(gf) / 4 * pres, torch.log1p(gn) / 2 * pres], -1)
        text_any = bool(tm.any()) if (hints or {}).get("text_any") is None else hints["text_any"]
        if not text_any:
            dur_loss = mu_x.sum() * 0
        elif self.encoder.dur is None:  # v16: Swift-RM on the encoder's states + the global condition (both detached,
            # as Swift-RM v1 was trained: the encoder learns from the prior / decoder; timing learns to read it)
            dur_loss = self.rhythm.loss(torch.log1p(dur), h.detach(), tm, g.detach())
        else:
            dur_loss = self.encoder.dur.loss(torch.log1p(dur), h.detach(), tm, span)

        attn_t = attn.transpose(1, 2).to(mu_x.dtype)
        mu_y = attn_t @ mu_x
        h_y = attn_t @ h.to(mu_x.dtype)
        pm = y_mask if nt is None else y_mask & ~nt[:, None]
        n = pm.sum().clamp(min=1) * self.cfg.token_dim
        prior_loss = (0.5 * (y - mu_y.float()) ** 2 * pm[..., None]).sum() / n
        out = {"prior": prior_loss, "dur": dur_loss, **zs_out}
        cond = torch.cat([mu_y, h_y], dim=-1)
        if getattr(self.cfg, "implicit_align", False):  # v13: the decoder never sees the alignment (prior/dur losses
            # above still train the encoder and the length predictor); text spread evenly, timing learned from audio
            ua = uniform_attn(x_mask.sum(1), y_mask.sum(1), x_mask.shape[1], y_mask.shape[1]).transpose(1, 2).to(mu_x.dtype)
            cond = torch.cat([ua @ mu_x, ua @ h.to(mu_x.dtype)], dim=-1)
        pitch_rows = None  # VTV rows in exact-pitch mode with the pitch-following loss (cfg.vtv_pitch_weight)
        if (hints or {}).get("pitch_rows") is not None:  # (11-18, singing branch) sung rows judged by RMVPE too
            pitch_rows = hints["pitch_rows"].to(y.device)
        if vtv is not None and self.vtv_in is not None and contour is not None:  # voice-to-voice rows: the recording's
            from .vtv import contour_feats                                       # content + contour, not the text
            content, vrows = vtv[0], vtv[1]
            exact = vtv[2] if len(vtv) > 2 else None
            cv = contour
            if exact is not None:  # exact-pitch rows: the pitch channel is the recording's RMVPE pitch where RMVPE
                xst, xv = vtv[3], vtv[4]  # hears voicing (as singing mode asks at inference, dacts/sing.py)
                n_ = min(cv.shape[1], xst.shape[1])
                p_ = cv[..., 0].clone()
                p_[:, :n_] = torch.where(exact[:, None] & xv[:, :n_], xst[:, :n_].to(p_.dtype), p_[:, :n_])
                cv = torch.cat([p_[..., None], cv[..., 1:]], -1)
            c = self.vtv_in(torch.cat([content[:, : cond.shape[1]].float(), contour_feats(cv.float())], -1))
            if exact is not None and self.vtv_exact is not None:  # the "exact pitch" mode input (zero at the start)
                c = c + exact[:, None, None].to(c.dtype) * self.vtv_exact.to(c.dtype)
            cond = torch.where(vrows[:, None, None], c.to(cond.dtype), cond)
            if nt is not None:  # (untranscribed rows still get content: no text needed)
                nt = nt & ~vrows
            if exact is not None and (getattr(self.cfg, "vtv_pitch_weight", 0) > 0 or getattr(self.cfg, "vtv_rmvpe_weight", 0) > 0):
                pitch_rows = exact & vrows
        beat = None
        if getattr(self.cfg, "use_beat", False) and labels.get("bpm") is not None:
            kb = keep["bpm"] if keep is not None and "bpm" in keep else torch.ones(y.shape[0], device=y.device)
            beat = self.beat_features(labels["bpm"], kb, y.shape[1])  # dropped / missing tempo -> no metronome
        out.update(self.flow_loss(y, cond, y_mask, g, codes=codes if self.code_head is not None else None, repa=repa,
                                  prompted=prompted, corrupt=corrupt, known=known, nocond=nt, beat=beat, notes=nfe,
                                  clab=clab, hints=hints, ref_seg=ref_seg, spk_rep=spk_rep, pitch_rows=pitch_rows, zs_rows=zs_rows))
        return out

    def prominence(self, attn, word_ids, contour, y_mask, nw=None):
        """Per-word emphasis 0..2 (1 = typical for this sentence) from aligned pitch, loudness and length.
        attn (B,Tx,Ty) 0/1, word_ids (B,Tx) word index or -1, contour (B,Ty,3) from dacts.contour.
        Returns (values (B,Tx), specified (B,Tx) bool) with a random emph_keep share of words specified."""
        nw = (int(word_ids.max().clamp(min=0)) if nw is None else int(nw)) + 1
        W = F.one_hot(word_ids.clamp(min=0), nw).float() * (word_ids >= 0)[..., None].float()  # B,Tx,Nw
        wf = W.transpose(1, 2) @ attn  # B,Nw,Ty: each word's frames
        frames, ntok = wf.sum(-1), W.sum(1)
        vm = (torch.sigmoid(contour[..., 1]) > 0.5).float() * y_mask.float()
        vf = (wf * vm[:, None]).sum(-1)
        pitch = (wf * (contour[..., 0] * vm)[:, None]).sum(-1) / vf.clamp(min=1)
        loud = (wf * contour[..., 2][:, None]).sum(-1) / frames.clamp(min=1)
        dur = frames / ntok.clamp(min=1)
        valid = (frames >= 2) & (ntok > 0)

        def z(x, m):
            mf = m.float()
            n = mf.sum(1, keepdim=True).clamp(min=1)
            mu = (x * mf).sum(1, keepdim=True) / n
            sd = (((x - mu) ** 2 * mf).sum(1, keepdim=True) / n).sqrt().clamp(min=0.3)
            return (x - mu) / sd * mf

        hasp = valid & (vf >= 2)
        prom = (z(pitch, hasp) + z(loud, valid) + z(dur, valid)) / (2 + hasp.float())
        e = (1 + 0.5 * prom).clamp(0, 2)
        enough = valid.sum(1, keepdim=True) >= 2  # emphasis is relative: needs at least two words
        spec = valid & enough & (torch.rand(valid.shape, device=valid.device) < getattr(self.cfg, "emph_keep", 0.3))
        idx = word_ids.clamp(min=0)
        return e.gather(1, idx), spec.gather(1, idx) & (word_ids >= 0)

    def beat_features(self, bpm_norm, present, n):
        """Metronome per decoder frame: sin/cos of the beat and bar (4/4) position, from the tempo label (dacts/music.py
        bpm_norm, 0..1 log scale). Clips start on a downbeat, so frame 0 is beat 1. present (B,) 0/1 gates rows."""
        from .codec import KVAE_HOP, KVAE_SR
        from .music import BPM_HI, BPM_LO
        bpm = BPM_LO * (BPM_HI / BPM_LO) ** bpm_norm.float().clamp(0, 1)  # (B,)
        sec = torch.arange(n, device=bpm.device).float() * (self.cfg.patch * KVAE_HOP / KVAE_SR)  # frame start times
        beats = sec[None] * (bpm[:, None] / 60.0)  # (B, n) beats since the clip start
        ph = torch.stack([beats, beats / 4.0], -1) * (2 * math.pi)
        f = torch.cat([ph.sin(), ph.cos()], -1)  # (B, n, 4)
        return f * present.float()[:, None, None]

    def flow_loss(self, x1, cond, mask, g, codes=None, repa=None, prompted=None, corrupt=None, known=None, nocond=None,
                  beat=None, notes=None, clab=None, hints=None, ref_seg=None, spk_rep=None, pitch_rows=None, zs_rows=None):
        b = x1.shape[0]
        t = torch.sigmoid(torch.randn(b, device=x1.device))  # logit-normal: focus on mid-noise levels
        t_min = getattr(self.cfg, "real_t_min", 0.0)  # (v13S on real recordings, 2026-10-25: only the end of the path,
        if t_min:                                     # where its last steps shape the fine texture; 0 = everywhere)
            t = t_min + (1 - t_min) * t
        shift = getattr(self.cfg, "prompt_t_shift", 1.0)  # (2026-10-30 voice-match plateau: rows with a prompt spend
        if shift != 1.0 and known is not None:             # more of their t near the noise end, where the voice is set;
            has = (known & mask).any(1)                    # SD3-style shift, t = 0 noise, 1 data)
            t = torch.where(has, t / (t + shift * (1 - t)), t)
        x0 = torch.randn_like(x1)
        s = self.cfg.sigma_min
        tt = t[:, None, None]
        x_t = (1 - (1 - s) * tt) * x0 + tt * x1
        u = x1 - (1 - s) * x0
        cond_in = cond
        keep = torch.ones(b, device=x1.device)
        if self.training and self.cfg.cfg_drop > 0:
            keep = (torch.rand(b, device=x1.device) > self.cfg.cfg_drop).float()
            cond = cond * keep.to(cond.dtype)[:, None, None]
        if nocond is not None:  # untranscribed rows: no text conditioning (but their prompt stays)
            cond = cond * (~nocond).to(cond.dtype)[:, None, None]
        ctx, gen = None, mask
        if known is not None:  # prompted rows (known frames chosen in forward)
            known = known & mask & (keep > 0)[:, None]  # CFG-dropped rows: no prompt either
            src = x1
            if corrupt is not None:  # the prompt part from a damaged recording of the same clip: take the voice,
                zc, has = corrupt    # not the room / noise / microphone (the generated part stays clean)
                use = has & (torch.rand(b, device=x1.device) < getattr(self.cfg, "corrupt_prob", 0.0))
                if hints is not None or bool(use.any()):  # (no row used: torch.where keeps x1)
                    src = torch.where(use[:, None, None], zc[:, : x1.shape[1]].to(x1.dtype), x1)
            ctx = torch.cat([src * known[..., None], known[..., None].to(x1.dtype)], dim=-1)
            if self.decoder.clab_in is not None:  # + the clip labels of each known frame (cfg.use_clip_labels)
                cl = clab if clab is not None else torch.zeros(*known.shape, CLAB_DIM, device=x1.device)
                ctx = torch.cat([ctx, (cl * known[..., None].to(cl.dtype)).to(x1.dtype)], dim=-1)
            gen = mask & ~known  # loss only where the model had to generate
        if ref_seg is not None:  # CFG-dropped rows: no reference segment either (the unconditional pass has no voice)
            ref_seg = (ref_seg[0], ref_seg[1], ref_seg[2] * keep)
        use_repa = repa is not None and self.repa_proj is not None
        spk_layers = self.spk_repa_layers if spk_rep is not None else ()
        taps = tuple(sorted(set(([self.cfg.repa_layer] if use_repa else []) + list(spk_layers))))
        vp_host = None
        if self.training and pitch_rows is not None and getattr(self.cfg, "vtv_rmvpe_weight", 0) > 0:
            # which rows / frames the RMVPE pitch loss takes, copied to the CPU before the decoder is queued: reading
            # them after it waited for the whole decoder forward (2026-11-01, speedups/ZIPPY_REPORT.md 2D)
            sel = pitch_rows & (t >= getattr(self.cfg, "vtv_rmvpe_t_min", 0.6)) & (keep > 0)
            vp_host = to_host(torch.cat([sel[:, None], gen], 1))
        v = self.decoder(x_t, t, cond, mask, g, tap=taps or None, ctx=ctx, beat=beat, notes=notes, ref=ref_seg)
        H = {}
        if taps:
            v, hs = v
            H = dict(zip(taps, hs))
        if use_repa:
            hid = H[self.cfg.repa_layer]
        v = v.float()
        out = {"fm": ((v - u) ** 2 * gen[..., None]).sum() / (gen.sum() * self.cfg.token_dim)}
        td = getattr(self, "_teacher_dec", None)  # frozen teacher decoder (set by train.py; not a submodule, not saved)
        if self.training and td is not None and getattr(self.cfg, "teacher_weight", 0.0) > 0:
            with torch.no_grad():  # (same inputs as the student, without the v14 reference segment: its memory is
                vt = td(x_t, t, cond, mask, g, ctx=ctx, beat=beat, notes=notes).float()  # the student's own size)
            out["teach"] = ((v - vt) ** 2 * gen[..., None]).sum() / (gen.sum() * self.cfg.token_dim)
        cn = getattr(self, "_contour_ref", None)  # frozen ContourNet (set by train.py; not a submodule, not saved)
        if self.training and getattr(self.cfg, "pitch_weight", 0) > 0 and notes is not None and cn is not None:
            sungf = (notes[..., 1] > 0) & (notes[..., 3] > 0) & gen
            sel = (t >= getattr(self.cfg, "pitch_t_min", 0.5)) & sungf.any(1) & (keep > 0)
            if bool(sel.any()):
                with torch.autocast(device_type=x1.device.type, enabled=False):
                    x1_hat = x_t.float()[sel] + (1 - tt[sel]) * v[sel]  # one-jump clean estimate, WITH gradient
                    ch = cn(x1_hat)
                    with torch.no_grad():
                        ct = cn(x1.float()[sel])
                    vm = (sungf[sel] & (torch.sigmoid(ct[..., 1]) > 0.5)).float()
                    if bool(vm.any()):
                        hl = F.huber_loss(ch[..., 0], ct[..., 0], reduction="none", delta=1.0)
                        out["pitch"] = (hl * vm).sum() / vm.sum()
        if vp_host is not None:
            # ... and the same rows judged by RMVPE on decoded audio (dacts.vtv.PitchLoss, run by train.py)
            hs = host_wait(vp_host)
            rows = hs[:, 0].nonzero().flatten()
            if len(rows):
                p = self.cfg.patch
                ri = rows.pin_memory().to(x1.device, non_blocking=True) if x1.is_cuda else rows
                x1_hat = x_t.float() + (1 - tt) * v  # one-jump clean estimate, WITH gradient
                out["_vpitch"] = (unpatchify(x1_hat.index_select(0, ri), p), unpatchify(x1.float().index_select(0, ri), p),
                                  hs[rows, 1:].repeat_interleave(p, dim=1))  # (generated frames: CPU, for PitchLoss)
        if self.training and pitch_rows is not None and cn is not None and getattr(self.cfg, "vtv_pitch_weight", 0) > 0:
            # exact-pitch VTV rows (you 2026-10-31: "just teach it to handle sticking to pitch well", "like exact pitch"):
            # the one-jump clean estimate's pitch (frozen ContourNet, with gradient) must match the target's on its
            # voiced frames; the same net on both sides, so its own bias cancels. Huber at 0.5 st: small misses count
            sel = pitch_rows & (t >= getattr(self.cfg, "pitch_t_min", 0.5)) & (keep > 0)
            if hints is not None or bool(sel.any()):
                p = self.cfg.patch
                with torch.autocast(device_type=x1.device.type, enabled=False):
                    x1_hat = x_t.float() + (1 - tt) * v  # (all rows: no data-dependent shapes; unselected rows weigh 0)
                    ch = patch_contour(cn, x1_hat, p)
                    with torch.no_grad():
                        ct = patch_contour(cn, x1.float(), p)
                    vm = (gen & sel[:, None] & (torch.sigmoid(ct[..., 1]) > 0.5)).float()
                    hl = F.huber_loss(ch[..., 0], ct[..., 0], reduction="none", delta=0.5)
                    out["vtv_pitch"] = (hl * vm).sum() / vm.sum().clamp(min=1)
        if self.training and getattr(self.cfg, "critic_weight", 0) > 0 and known is not None and getattr(self, "_critic_now", False):
            # the voice critic's batch (dacts/voice_critic.py; train.py runs it after this step's backward): prompted
            # rows with text, windows holding >= 2 s of prompt and >= 3.2 s the model generates (decoder steps)
            from .voice_critic import critic_windows
            p = self.cfg.patch
            ok = (keep > 0) if nocond is None else (keep > 0) & ~nocond
            alt = getattr(self, "_critic_alt", None)  # (11-14 Solver, cfg.critic_skip_alt) damaged-target rows: not judged
            if alt is not None and getattr(self.cfg, "critic_skip_alt", False):
                alt = alt.to(ok.device)
                ok = ok & ~((alt == 1) | ((alt == 2) & bool(getattr(self.cfg, "critic_skip_up", False))))
            if zs_rows is not None and bool((ok & zs_rows).any()):  # critic_zs (owner 2026-10-31: "make sure the model learns
                ok = ok & zs_rows  # to increase voice similarity and zero shot performance with that critic")
            ce = float(getattr(self.cfg, "critic_edit", 0.0))
            out["_critic"] = None
            eo = bool(getattr(self, "_critic_edit_only", False))  # (11-24 critic pool) a word-gap-only harvest step
            if ce > 0 and (eo or float(torch.rand(())) < ce):  # (11-22) voice match on the filled-in words of word-gap rows
                from .voice_critic import critic_edit_windows
                ok_e = (keep > 0) if nocond is None else (keep > 0) & ~nocond  # (not limited to zero-shot rows)
                out["_critic"] = critic_edit_windows(x1, cond, mask, ctx, g, ref_seg, gen, known & mask, ok_e,
                                                     rows=self.cfg.critic_rows, win=self.cfg.critic_win // p,
                                                     gap_min=-(-15 // p), gap_max=(int(getattr(self.cfg, "critic_crop", 150)) - 20) // p)
            if out["_critic"] is None and not eo:
                out["_critic"] = critic_windows(x1, cond, mask, ctx, g, ref_seg, gen, known & mask, ok,
                                                rows=self.cfg.critic_rows, win=self.cfg.critic_win // p,
                                                k_min=-(-int(getattr(self.cfg, "critic_k_min", 100)) // p), g_min=-(-160 // p))
        if self.training and getattr(self.cfg, "sim_weight", 0) > 0:  # for the speaker-similarity loss (train.py)
            sel = (t >= self.cfg.sim_t_min) & (keep > 0)
            if bool(sel.any()):
                p = self.cfg.patch
                x1_hat = x_t.float() + (1 - tt) * v  # one-jump clean estimate, WITH gradient
                out["_sim"] = (unpatchify(x1_hat[sel], p), unpatchify(x1.float()[sel], p),
                               gen[sel].repeat_interleave(p, dim=1))
        if use_repa:  # REPA: -cosine similarity between projected layer and SSL features of the real audio
            p = self.repa_proj(hid).float()
            m = mask & (repa.abs().sum(-1) > 0)  # frames with a target
            cos = F.cosine_similarity(p, repa.float(), dim=-1)
            out["repa"] = -(cos * m).sum() / m.sum().clamp(min=1)
        if spk_layers:  # Speaker-REPA (TLA-SA, arXiv 2511.09995; research idea 1, 2026-10-28): middle layers, pooled over
            # the frames the model generates, must say who is talking (cosine to the voice's x-vector)
            w = (gen & (keep > 0)[:, None]).float()[..., None]
            pooled = torch.cat([(H[l].float() * w).sum(1) / w.sum(1).clamp(min=1) for l in spk_layers], -1)
            xv, ok_r = spk_rep
            rows = (ok_r & (w.sum((1, 2)) > 0)).float()
            cos = F.cosine_similarity(self.spk_repa_proj(pooled).float(), xv.float(), dim=-1)
            out["spk_repa"] = ((1 - cos) * rows).sum() / rows.sum().clamp(min=1)
        if codes is not None:
            # code picker, trained on this same pass: rows whose noise level falls where sampling hands
            # over to it (~45% of a batch) give the one-jump clean estimate it sees at inference
            lo, hi = self.cfg.code_t_range
            sel = (t >= lo) & (t <= hi) & (keep > 0)
            if bool(sel.any()):
                p = self.cfg.patch
                x1_hat = (x_t.float() + (1 - tt) * v).detach()[sel]
                ce, acc = self.code_head.loss(unpatchify(x1_hat, p), cond_in.detach()[sel], codes[sel],
                                              mask[sel].repeat_interleave(p, dim=1))
                out.update({"code": ce, "code_acc": acc.mean(), "code_acc1": acc[0], "code_acc9": acc[-1]})
        return out

    @torch.no_grad()
    def synthesize(self, tokens, x_len, spk=None, style=None, volume=None, speed=None, lang=None, durations=None,
                   length_scale=1.0,
                   steps=8, guidance=3.0, temperature=0.85, dur_temperature=0.8, sway=-1.0, solver="midpoint",
                   cfg_until=0.7, label_guidance=1.0, pick_codes=False, code_temperature=1.0, code_top_p=0.95,
                   code_t=None, lm=None, labels=None, spk_guidance=1.0, prompt=None, schedule=None, trace=None,
                   prompt_guidance=1.0, dlv=None, emph=None, contour_net=None, prompt_guidance_until=1.0,
                   whisper_guidance=1.0, band_guidance=1.0, lang_guidance=1.0, style_transfer=0.0, apg=None,
                   target_len=None, len_quantum=None, notes=None, word_frames=None, word_ids=None,
                   word_vowels=None, note_guidance=1.0, note_cfg=False, prompt_labels=None, emph_stretch=0.0, word_ms=None,
                   word_tok=None, durations_only=False, align_to=None, start_t=None, record=None, ref_seg=None, short_fix=True,
                   cfg_from=0.0, prompt_guidance_from=0.0, anti_voice=0.0,
                   np_drop_pe=False, guid_w=None, infill=None):
        """Returns normalized latents (B, Ty*patch, latent_dim) and frame lengths.
        word_ms / word_tok: a length in ms per word (None = the model's) and each token's word index (-1: none) --
        the site's Timing mode (owner 2026-10-23). durations_only: return the token durations (frames) right after they
        are decided (the site shows the duration model's word lengths before anything is rendered).

        `durations` (B, Tx), in patches, overrides the duration predictor.
        `sway` < 0 concentrates ODE steps near t=0 where the coarse structure forms (F5-TTS).
        `cfg_until` < 1 skips the unconditional pass once t >= cfg_until (guidance matters early; halves late steps).
        `prompt` {"latents": (1, Tp, D) normalized, "n_tok": prompt tokens at the start of `tokens`}: prompted
        cloning (batch of 1): the prompt's text is aligned to its audio, the rest is generated as its continuation.
        `label_guidance` > 1 exaggerates what style / volume / speed change, by steering away from the same
        speaker without those labels (they are dropped in training, so the model knows both). Helps speakers
        whose data never whispers / shouts / acts, at one extra decoder pass per step.
        `whisper_guidance` > 1 (with labels["whisper"] set): steer along whisper vs the SAME voice explicitly
        normal (whisper = 0); the whisper direction is learned from the voices that have real whispers.
        `align_to` (B, Ty, D) normalized real latents of this text: durations from aligning the text to them (as in
        training), so the result lines up with that clip; `start_t`: start the flow part-way, from those latents re-noised
        to t (training's x_t) -- the model's own version of a real clip, frame for frame (decoder fine-tune data, evals).
        `record` (a dict): filled with what a distilled student learns from (v13XS, tools/xs_teacher.py): the token
        durations, the encoder's outputs mu_x / h, the starting noise x0 and the final latents x1 (normalized, patched).
        `ref_seg` (1, Tr, D) normalized latents of other clips of the voice (v14 reference segment, cfg.use_ref_seg):
        every pass hears it except the unconditional and the no-prompt (prompt guidance) ones.
        `cfg_from` / `prompt_guidance_from` > 0: no guidance while t < that (the noisiest steps) -- a guidance interval
        (Kynkaanniemi et al. 2024: guidance at high noise mostly hurts; research idea 2, 2026-10-28).
        `anti_voice` > 0 (with a prompt): one more pass whose prompt has its voice colour removed (each latent dimension's
        mean over the prompt subtracted: same words and timing, an average voice) and steer away from it -- a negative
        prompt for timbre against the drift toward an average voice (research idea 6, 2026-10-28).
        """
        if getattr(self.global_cond, "guid", None) is not None:  # (cfg.use_guid: the guidance is
            # a condition the model learned, one pass per step: no unguided pass. guid_w: the knob set to that instead and
            # `guidance` kept as plain two-pass guidance on top -- the old way, for A/Bs)
            w_ = float(guidance if guid_w is None else guid_w)
            labels = {**(labels or {}), "guid": torch.full((tokens.shape[0],), w_, device=tokens.device)}
            guidance = 1.0 if guid_w is None else guidance
        p = self.cfg.patch
        if p > 1:  # references arrive as KVAE frames (make_prompt, the site's voice files) -> decoder steps
            def steps_of(z):
                if z is None or z.shape[-1] != self.cfg.token_dim // p:
                    return z
                z = z if z.dim() == 3 else z[None]
                return patchify(z[:, : z.shape[1] // p * p], p)
            dlv = steps_of(dlv)
            ref_seg = steps_of(ref_seg)
            if prompt is not None:
                prompt = {**prompt, "latents": steps_of(prompt["latents"])}
        x_mask = seq_mask(x_len, tokens.shape[1])
        tlab = clab_p = None
        if prompt is not None and self.decoder.clab_in is not None and (prompt.get("frame_labels") is not None
                                                                         or prompt.get("token_labels") is not None):
            b_ = tokens.shape[0]
            asked = asked_labels({"style": style, "volume": volume, "speed": speed, **(labels or {})}, None, b_, tokens.device)
            if prompt.get("frame_labels") is not None:
                fl = prompt["frame_labels"].to(tokens.device).float()[None].expand(b_, -1, -1)
                clab_p = clip_label_features(fl, asked)
            if prompt.get("token_labels") is not None and self.encoder.tlab is not None:
                tk = prompt["token_labels"].to(tokens.device).float()[None].expand(b_, -1, -1)
                tlab = torch.zeros(b_, tokens.shape[1], CLAB_DIM, device=tokens.device)
                n_ = min(tokens.shape[1], tk.shape[1])
                tlab[:, :n_] = clip_label_features(tk[:, :n_], asked)
        pe = None  # how the prompt was delivered (cfg.use_prompt_labels): {name: float or (B,)}, each optional
        if prompt is not None and self.prompt_lab is not None:
            b_ = tokens.shape[0]
            pl = {k: torch.as_tensor(v, dtype=torch.float32, device=tokens.device).reshape(-1).expand(b_)
                  for k, v in (prompt_labels or {}).items() if v is not None}
            pe = self.prompt_lab(pl, torch.ones(b_, dtype=torch.bool, device=tokens.device))

        def gc(*a, **k):  # every global condition of this line (also the guidance variants) includes the prompt's labels
            g_ = self.global_cond(*a, **k)
            return g_ if pe is None else (pe if g_ is None else g_ + pe.to(g_.dtype))
        emph_raw = emph
        if emph is not None:  # the site goes up to 3: the encoder learned 0..2, the rest is stretch + a pause (below)
            emph = (emph[0].clamp(0, 2), emph[1])
        g = gc(spk, style, volume, speed, lang, **(labels or {}))
        g_dlv = None  # the delivery reference's part of g: every guidance variant keeps it (code review 2026-10-29)
        if dlv is not None and self.dlv_enc is not None:  # delivery reference latents (1, T, c) of the voice
            e = self.dlv_enc(dlv.float(), torch.ones(dlv.shape[:2], dtype=torch.bool, device=dlv.device))
            e = e.expand(tokens.shape[0], -1)
            g_dlv = e
            g = e if g is None else g + e.to(g.dtype)
        encoder = self.text_encoder(tokens.is_cuda)
        mu_x, h = encoder(tokens, x_mask, g, lm, emph=emph, tlab=tlab)
        dur_cont = None  # unrounded durations (prompt pacing scales these, then rounds)
        if align_to is not None:  # teacher-forced: the text aligned to the real latents (monotonic, like training)
            ya = align_to.float().to(mu_x.device)
            mu_a, y_a = mu_x.float().index_select(-1, self.align_idx), ya.index_select(-1, self.align_idx)
            logp = 2 * mu_a @ y_a.transpose(1, 2) - (y_a ** 2).sum(-1)[:, None, :] - (mu_a ** 2).sum(-1)[:, :, None]
            m = x_mask.bool()[:, :, None] & torch.ones(ya.shape[:2], dtype=torch.bool, device=ya.device)[:, None, :]
            if prompt is not None:  # (prompt frames first: the line's tokens only to the frames after them)
                n_, tp_ = int(prompt["n_tok"]), prompt["latents"].shape[1]
                durations = torch.zeros(m.shape[:2], device=m.device)
                durations[:, n_:] = maximum_path(logp[:, n_:, tp_:].contiguous(), m[:, n_:, tp_:].contiguous()).sum(-1)
            else:
                durations = maximum_path(logp, m).sum(-1)
        rhythm = getattr(self, "rhythm", None)  # Swift-RM (dacts/rhythm.py) when attached: the rhythm model, else the old head
        pctx = None  # (set below for a prompt: the lambda reads it when it runs -- Swift-RM 2 hears the prompt's prosody too)
        # (Swift-RM 2 re-reads the line with its own text encoder: it gets the same inputs as the original)
        dur_sample = ((lambda h_, m_: rhythm.sample(h_, m_, steps=10, temperature=dur_temperature, g=g, tokens=tokens, lm=lm,
                                                    emph=emph, pctx=pctx, tlab=tlab)) if rhythm is not None
                      else (lambda h_, m_: self.encoder.dur.sample(h_, m_, steps=10, temperature=dur_temperature)))
        def tame(dc):  # (11-18, you: "grey doesn't handle short words, 1-3 well") one-word lines are rare in training, so the
            # rhythm model gives them absurd lengths ("No." 3-6 s, one sound 40-110 frames) and the decoder fills the time
            # with made-up talk. Short lines only (<= 30 tokens of new text): no sound longer than 25 frames, and the
            # line no slower than 7 frames per token (normal speech: 3-7). Longer lines are never touched.
            if not short_fix:
                return dc
            n0 = int(prompt["n_tok"]) if prompt is not None else 0
            dc = dc.clone()
            for b_ in range(dc.shape[0]):
                n_new = int(x_mask[b_].sum()) - n0
                if n_new <= 0 or n_new > 30:
                    continue
                seg = dc[b_, n0:n0 + n_new].clamp(max=25.0 * length_scale)
                cap = 7.0 * n_new * length_scale
                if float(seg.sum()) > cap:
                    seg = seg * (cap / float(seg.sum()))
                dc[b_, n0:n0 + n_new] = seg
            return dc
        if durations is None:
            logw = dur_sample(h, x_mask)
            dur_cont = tame(torch.expm1(logw.float()).clamp(min=0) * length_scale * x_mask)
            durations = torch.round(dur_cont)
        tp = 0
        plain_enc = None  # encoder output without the prompt's prosody (pctx), when that was given
        if prompt is not None:  # prompt tokens get the durations of the prompt audio (monotonic alignment)
            pl, n = prompt["latents"].float(), int(prompt["n_tok"])
            tp = pl.shape[1]
            mu_a = mu_x[:, :n].float().index_select(-1, self.align_idx)
            y_a = pl.index_select(-1, self.align_idx)
            logp = 2 * mu_a @ y_a.transpose(1, 2) - (y_a ** 2).sum(-1)[:, None, :] - (mu_a ** 2).sum(-1)[:, :, None]
            m = torch.ones(1, n, tp, dtype=torch.bool, device=pl.device)
            durations = durations.clone()
            pa = maximum_path(logp, m)  # (1, n, tp) prompt token -> frame alignment
            actual = pa.sum(-1)  # how long each prompt token really lasted
            if self.encoder.pctx is not None and contour_net is not None:
                # tell the encoder how the prompt was said (lengths, pitch, loudness per token) and plan the new
                # line again: it continues the speaker's intonation and rhythm, not only the voice
                with torch.autocast(device_type=pl.device.type, enabled=False):
                    c = patch_contour(contour_net, pl.float(), self.cfg.patch)
                    f = token_prosody(pa, c, torch.ones(1, tp, dtype=torch.bool, device=pl.device))
                pctx = torch.zeros(*tokens.shape, f.shape[-1], device=pl.device)
                pctx[:, :n] = f.expand(tokens.shape[0], -1, -1)
                plain_enc = (mu_x, h)  # (the prompt-less guidance pass reads the text without the prompt's prosody)
                mu_x, h = encoder(tokens, x_mask, g, lm, emph=emph, pctx=pctx, tlab=tlab)
                if dur_cont is not None:  # new durations from the informed encoder
                    logw = dur_sample(h, x_mask)
                    dur_cont = tame(torch.expm1(logw.float()).clamp(min=0) * length_scale * x_mask)
                    durations = torch.round(dur_cont)
            timing = float(prompt.get("timing", 0.0)) if align_to is None else 0.0  # (teacher-forced: real timing)
            if timing > 0:  # speak at the prompt's pace: the duration predictor never sees the prompt audio,
                # so compare what it predicts for the prompt's own tokens with what was really said, and scale
                # the new line by that ratio (separately for speech and for pauses: spaces / punctuation)
                pred = durations[:, :n].float()
                pause = prompt.get("pause")
                pause = torch.zeros_like(pred, dtype=torch.bool) if pause is None else pause[:, : tokens.shape[1]].bool()
                scale = torch.ones_like(durations, dtype=torch.float)
                for sel in (~pause, pause):
                    p_sum, a_sum = (pred * sel[:, :n]).sum(), (actual.float() * sel[:, :n]).sum()
                    if p_sum >= 10:  # enough evidence (>= 0.2 s predicted)
                        f = float((a_sum / p_sum).clamp(0.5, 2.5)) ** timing
                        scale = torch.where(sel, torch.full_like(scale, f), scale)
                # scale the unrounded durations and round their running total: most tokens last 1-2 frames,
                # so rounding each one alone would swallow a 1.3x slow-down (or double a 1.6x one)
                d = (durations.float() if dur_cont is None else dur_cont) * scale * x_mask
                ends = torch.round(d.cumsum(1))
                durations = torch.diff(ends, dim=1, prepend=torch.zeros_like(ends[:, :1])) * x_mask
            durations[:, :n] = actual
        if emph is not None and emph_stretch > 0:
            # (owner 2026-10-23: "time per word slower than average" is what stress sounds like) the marked words'
            # tokens last longer by their emphasis -- 2.0 -> x(1 + emph_stretch) -- and softened ones a little shorter;
            # on top of what the encoder does with the values. Rounded on the running total (most tokens are 1-2
            # frames: rounding each alone would swallow the stretch). A prompt's own tokens are never marked.
            ev, es = emph_raw
            ev, es = ev[:, : durations.shape[1]].float(), es[:, : durations.shape[1]]
            f = torch.where(ev >= 1, 1 + emph_stretch * (ev - 1), 1 - 0.25 * emph_stretch * (1 - ev))
            f = torch.where(es, f, torch.ones_like(f))
            if bool((f != 1).any()):
                d = (dur_cont if (dur_cont is not None and prompt is None) else durations.float()) * f * x_mask
                # above 2: a beat of silence before the word (up to 0.12 s at 3), on the token just before it
                from .codec import frame_rate
                fps = frame_rate(self.cfg) / max(1, int(getattr(self.cfg, "patch", 1)))
                lead = (ev - 2).clamp(min=0, max=1) * 0.12 * fps
                first = es & ~F.pad(es[:, :-1], (1, 0), value=False)  # the first token of each marked word
                for b_, t_ in first.nonzero().tolist():
                    if t_ > 0 and float(lead[b_, t_]) > 0:
                        d[b_, t_ - 1] = d[b_, t_ - 1] + lead[b_, t_]
                ends = torch.round(d.cumsum(1))
                durations = torch.diff(ends, dim=1, prepend=torch.zeros_like(ends[:, :1])) * x_mask
        if word_frames is not None and word_ids is not None and prompt is None and not getattr(self.cfg, "implicit_align", False):
            # sing given notes. word_frames[w] = (first note start, sung end[, next word's start]): the word fills its
            # notes, the gap after it (a rest) goes to the separators that follow it (silence, like pauses in speech);
            # tokens before the first word get the lead-in. Inside a word the consonants keep their predicted (spoken)
            # length and the vowels (word_vowels) hold the rest -- stretching every letter evenly sang words in slow
            # motion ("dying / reading a book very slowly", owner 2026-10-19). Words without vowels stretch evenly.
            d = (dur_cont if dur_cont is not None else durations.float())[0].clamp(min=0.3)
            slot, gap_of, last_w = [], [], -1  # per token: word index / -1 lead-in / None separator (+ the word before)
            for w in word_ids:
                if w is not None and w >= 0:
                    slot.append(w); last_w = w; gap_of.append(None)
                else:
                    slot.append(None if last_w >= 0 else -1); gap_of.append(last_w if last_w >= 0 else None)
            span = {-1: float(word_frames[0][0])}
            gaps = {}
            for w, wf in enumerate(word_frames):
                a_, b_ = wf[0], wf[1]
                c_ = wf[2] if len(wf) > 2 else b_
                if not any(gw == w for gw in gap_of):  # nowhere to rest: the word holds the gap too
                    b_ = c_
                span[w] = float(max(1, b_ - a_)); gaps[w] = float(max(0, c_ - b_))
            learned = self.encoder.dur is not None and self.encoder.dur.span_in is not None
            if learned:  # the duration model was trained to spread a sung word's time itself (cfg.use_dur_span)
                feat = torch.zeros(1, d.shape[0], 3, device=d.device)
                for k, (sl, gw) in enumerate(zip(slot, gap_of)):  # (not `g`: that is the global conditioning)
                    fr, nn_ = (span[-1], 0) if sl == -1 else (span[sl], word_frames[sl][3] if len(word_frames[sl]) > 3 else 1)                         if sl is not None else (gaps.get(gw, 0.0), 0)
                    feat[0, k] = torch.tensor([1.0, math.log1p(fr) / 4, math.log1p(nn_) / 2])
                logw = self.encoder.dur.sample(h[:1], x_mask[:1], steps=10, temperature=dur_temperature, span=feat)
                d = torch.expm1(logw.float()).clamp(min=0.3)[0]
            vow = torch.tensor([bool(v) for v in word_vowels], device=d.device) if (word_vowels is not None and not learned) else None
            new = torch.zeros_like(d)

            def fill(idx, frames, hold):
                if not idx:
                    return
                dd = d[idx]
                if hold and vow is not None and frames > float(dd.sum()) and bool(vow[idx].any()):
                    v = vow[idx].float()
                    part = dd + (frames - float(dd.sum())) * v * dd / float((v * dd).sum())
                else:
                    part = dd / dd.sum() * frames
                ends = torch.round(part.cumsum(0)); new[idx] = torch.diff(ends, prepend=torch.zeros(1, device=d.device))
            fill([k for k, sl in enumerate(slot) if sl == -1], span[-1], False)
            for w in range(len(word_frames)):
                fill([k for k, sl in enumerate(slot) if sl == w], span[w], True)
                if gaps[w] > 0:
                    fill([k for k, gw in enumerate(gap_of) if gw == w], gaps[w], False)
            durations = new[None].to(durations.dtype)
        if (target_len is not None or len_quantum is not None) and word_frames is None:  # notes set the timing themselves
            # owner: optional length override, in patches, for the new line (not the prompt): keep the predicted
            # rhythm, stretch / squeeze it to the asked total; round the running total so short tokens aren't all
            # rounded the same way. len_quantum (patches per bar, singing): the duration model still decides the
            # length, which is then rounded to whole bars (at least one)
            n0 = int(prompt["n_tok"]) if prompt is not None else 0
            d = durations[:, n0:].float().clamp(min=0) * x_mask[:, n0:]
            d = torch.where(d.sum(1, keepdim=True) > 0, d, x_mask[:, n0:].float())  # nothing predicted: spread evenly
            if target_len is not None:
                tgt = torch.as_tensor(target_len, dtype=torch.float, device=d.device).reshape(-1, 1).expand(d.shape[0], 1)
            else:
                q = float(len_quantum)
                tgt = (d.sum(1, keepdim=True) / q).round().clamp(min=1) * q
            ends = torch.round((d * (tgt / d.sum(1, keepdim=True))).cumsum(1))
            durations = durations.clone()
            durations[:, n0:] = torch.diff(ends, dim=1, prepend=torch.zeros_like(ends[:, :1])) * x_mask[:, n0:]
        if word_ms is not None and word_tok is not None and any(v is not None for v in word_ms):
            # each word lasts what the site asks: its tokens scaled together (their shares kept), rounded on the
            # running total; the pauses at punctuation are ids after the words (0 ms allowed), plain spaces keep theirs
            from .codec import frame_rate
            fps = frame_rate(self.cfg) / max(1, int(getattr(self.cfg, "patch", 1)))
            d = durations.float().clone()
            wt = word_tok[:, : d.shape[1]].to(d.device)
            for w, ms in enumerate(word_ms):
                if ms is None:
                    continue
                sel = wt == w
                cur = float((d * sel).sum())
                if cur > 0:
                    d = torch.where(sel, d * (float(ms) / 1000.0 * fps / cur), d)
            ends = torch.round(d.cumsum(1))
            durations = torch.diff(ends, dim=1, prepend=torch.zeros_like(ends[:, :1])) * x_mask
        inf = None
        if infill is not None and prompt is not None:  # (11-22, you: "take the surrounding words, then completely noise out
            # the bad word, and denoise that word with what it's supposed to be") speech editing: the prompt = the audio before
            # the edit; the line = new word(s) + the words after; infill["latents"] = the real audio after the edit. Those
            # words are aligned to it (as align_to), it is known context (ctx) and pinned to its real latents at every step,
            # so only the new word's frames are made -- seeing both sides. Default off: nothing else changes.
            n0, nn = int(prompt["n_tok"]), int(infill["n_new"])
            suf = infill["latents"].float().to(mu_x.device)
            suf = suf if suf.dim() == 3 else suf[None]
            mu_s = mu_x[:, n0 + nn:].float().index_select(-1, self.align_idx)
            y_s = suf.index_select(-1, self.align_idx).expand(mu_s.shape[0], -1, -1)
            lp = 2 * mu_s @ y_s.transpose(1, 2) - (y_s ** 2).sum(-1)[:, None, :] - (mu_s ** 2).sum(-1)[:, :, None]
            ms_ = x_mask.bool()[:, n0 + nn:, None] & torch.ones(1, 1, y_s.shape[1], dtype=torch.bool, device=lp.device)
            durations = durations.clone()
            if infill.get("mid_frames"):  # a fixed length for the new word(s) (e.g. the old word's, scaled)
                dm = durations[:, n0:n0 + nn].float().clamp(min=0.3)
                dm = dm.clamp(max=2.5 * float(dm.median()))  # no pause spike: the rhythm model expects a break after a prompt
                ends = torch.round((dm * (float(infill["mid_frames"]) / dm.sum(1, keepdim=True))).cumsum(1))
                durations[:, n0:n0 + nn] = torch.diff(ends, dim=1, prepend=torch.zeros_like(ends[:, :1])).to(durations.dtype)
            if infill.get("seg_frames"):  # (11-24) per-word lengths: [(tok_start, tok_end, frames)] relative to the new tokens --
                # the unchanged neighbours keep their REAL length from the recording, only the edited word's is new; inside a
                # word the model's own split over its letters is kept (one total for the phrase squeezed its last word away)
                for t0_, t1_, fr_ in infill["seg_frames"]:
                    a_, b_ = n0 + int(t0_), n0 + int(t1_)
                    dm = durations[:, a_:b_].float().clamp(min=0.3)
                    if dm.shape[1] > 2:  # no pause spike inside a word (the break the model expects after a prompt sat on
                        dm = dm.clamp(max=2.5 * float(dm.median()))  # the first letter: "did" = a long pause + a rushed word)
                    ends = torch.round((dm * (float(fr_) / dm.sum(1, keepdim=True))).cumsum(1))
                    durations[:, a_:b_] = torch.diff(ends, dim=1, prepend=torch.zeros_like(ends[:, :1])).to(durations.dtype)
            if infill.get("lead_frames"):  # room for the pause the model puts after a prompt (cut out afterwards)
                durations[:, n0] = durations[:, n0] + int(infill["lead_frames"])
            durations[:, n0 + nn:] = maximum_path(lp.contiguous(), ms_.contiguous()).sum(-1).to(durations.dtype)
            if infill.get("suf_seg_frames"):  # (11-24) the words after placed by the recording's REAL word times (the MAS
                # above gave them a flat ~100 ms each; with text and known audio out of step the edit dropped "men")
                for t0_, t1_, fr_ in infill["suf_seg_frames"]:
                    a_, b_ = n0 + nn + int(t0_), n0 + nn + int(t1_)
                    dm = durations[:, a_:b_].float().clamp(min=0.3)
                    if dm.shape[1] > 2:
                        dm = dm.clamp(max=2.5 * float(dm.median()))
                    ends = torch.round((dm * (float(fr_) / dm.sum(1, keepdim=True))).cumsum(1))
                    durations[:, a_:b_] = torch.diff(ends, dim=1, prepend=torch.zeros_like(ends[:, :1])).to(durations.dtype)
            s0 = int(tp + durations[0, n0:n0 + nn].sum())
            inf = (s0, suf)
            infill["mid_frames_used"] = s0 - tp
        if durations_only:
            if record is not None:  # (tools/rhythm_data.py: the aligned timing + what the rhythm model reads)
                record.update(durations=durations.detach(), h=h.detach().float(), g=None if g is None else g.detach().float(),
                              x_len=x_len, tokens=tokens.detach(), lm=lm)
            return durations
        if record is not None:
            record.update(durations=durations.detach(), mu_x=mu_x.detach().float(), h=h.detach().float(), x_len=x_len)
        y_len = durations.sum(1).long().clamp(min=1)
        ends = durations.cumsum(1)
        frames = torch.arange(int(y_len.max()), device=tokens.device)[None, None]
        attn = ((frames >= (ends - durations)[..., None]) & (frames < ends[..., None])).transpose(1, 2).float()
        if getattr(self.cfg, "implicit_align", False):  # v13: durations only set the length; the decoder places the text
            attn = uniform_attn(x_mask.sum(1), y_len, x_mask.shape[1], int(y_len.max())).transpose(1, 2)
        cond = torch.cat([attn @ mu_x.float(), attn @ h.float()], dim=-1)
        # the no-prompt guidance pass: training's prompt-less rows had no prompt prosody in the text either (code review
        # 2026-10-29); the same alignment, the encoder's first (uninformed) reading
        cond_np = cond if plain_enc is None or not np_drop_pe else torch.cat(
            [attn @ plain_enc[0].float(), attn @ plain_enc[1].float()], dim=-1)
        y_mask = seq_mask(y_len, cond.shape[1])

        # integrate in fp32 (the decoder itself may run under bf16 autocast)
        x = torch.randn(cond.shape[0], cond.shape[1], self.cfg.token_dim, device=cond.device) * temperature
        if start_t is not None and align_to is not None:  # part-way: the real latents re-noised to start_t
            x0 = torch.randn_like(x)
            x = (1 - (1 - self.cfg.sigma_min) * start_t) * x0 + start_t * align_to.float().to(x.device)[:, : x.shape[1]]
        if record is not None:
            record["x0"] = x.detach().clone()
        ctx = ctx2 = None
        if prompt is not None:
            extra = CLAB_DIM if self.decoder.clab_in is not None else 0
            ctx = torch.zeros(cond.shape[0], cond.shape[1], self.cfg.token_dim + 1 + extra, device=cond.device)
            ctx[:, :tp, :self.cfg.token_dim] = prompt["latents"].float()
            ctx[:, :tp, self.cfg.token_dim] = 1.0
            if extra and clab_p is not None:
                n_ = min(tp, clab_p.shape[1])
                ctx[:, :n_, self.cfg.token_dim + 1:] = clab_p[:, :n_]
        if inf is not None and ctx is not None and infill.get("ctx", True):  # (speech editing) the real audio after the edit: known frames too
            s0_, suf_ = inf
            ctx[:, s0_:s0_ + suf_.shape[1], :self.cfg.token_dim] = suf_
            ctx[:, s0_:s0_ + suf_.shape[1], self.cfg.token_dim] = 1.0
        ctx_av = None
        if ctx is not None and anti_voice:  # the same prompt, voice colour removed
            ctx_av = ctx.clone()
            pl = ctx_av[:, :tp, :self.cfg.token_dim]
            ctx_av[:, :tp, :self.cfg.token_dim] = pl - pl.mean(1, keepdim=True)
        use_av = ctx_av is not None
        use_cfg = guidance != 1.0
        if use_cfg:
            cond2, mask2 = torch.cat([cond, torch.zeros_like(cond)]), torch.cat([y_mask, y_mask])
            ctx2 = None if ctx is None else torch.cat([ctx, torch.zeros_like(ctx)])  # unconditional: no text, no prompt
            g2 = None if g is None else torch.cat([g, g])  # guidance is over text; speaker/style kept
        use_lg = label_guidance != 1.0 and any(a is not None for a in (style, volume, speed))
        g_plain = gc(spk, None, None, None, lang) if use_lg else None  # same voice, no delivery labels
        if g_plain is not None and g_dlv is not None:  # (was missing: label guidance also scaled the delivery vector)
            g_plain = g_plain + g_dlv.to(g_plain.dtype)
        # speaker guidance: steer away from "no speaker" to hold the voice's identity (e.g. in another language)
        use_sg = spk_guidance != 1.0 and spk is not None
        use_bg = band_guidance != 1.0 and labels is not None and labels.get("band") is not None
        g_lowband = None
        if use_bg:  # same voice and labels, but band-limited: steer away from it (voices only heard muffled)
            g_lowband = gc(spk, style, volume, speed, lang, **{**labels, "band": torch.zeros_like(labels["band"])})
            if dlv is not None and self.dlv_enc is not None:
                g_lowband = g_lowband + (g - gc(spk, style, volume, speed, lang, **labels))
        use_wg = whisper_guidance != 1.0 and labels is not None and labels.get("whisper") is not None
        g_normal = None
        if use_wg:  # same voice and labels, but explicitly not whispering
            g_normal = gc(spk, style, volume, speed, lang,
                                        **{**labels, "whisper": torch.zeros_like(labels["whisper"])})
            if dlv is not None and self.dlv_enc is not None:
                g_normal = g_normal + (g - gc(spk, style, volume, speed, lang, **labels))
        g_nospk = gc(None, style, volume, speed, lang, **(labels or {})) if use_sg else None
        if g_nospk is not None and g_dlv is not None:  # "no speaker", same delivery: speaker guidance steers the voice only
            g_nospk = g_nospk + g_dlv.to(g_nospk.dtype)  # (code review 2026-10-29: it scaled the delivery vector too)
        # style transfer: the delivery change "style label vs neutral" measured WITHOUT a speaker (what acting does across
        # all voices, learned from the clips trained with the speaker hidden), added to this voice's prediction --
        # so voices that only ever read can follow the Style slider (2 extra decoder passes per step)
        use_st = style_transfer != 0.0 and style is not None
        g_st_on = g_st_off = None
        if use_st:
            g_st_on = gc(None, style, volume, speed, lang, **(labels or {}))
            g_st_off = gc(None, torch.full_like(style, 0.5), volume, speed, lang, **(labels or {}))
        # language guidance: steer toward "this language" vs the same voice with no language label (dropped 15% in
        # training) -- pulls a voice whose identity carries another language's accent toward native pronunciation
        use_langg = lang_guidance != 1.0 and lang is not None
        g_nolang = None
        if use_langg:
            g_nolang = gc(spk, style, volume, speed, None, **(labels or {}))
            if dlv is not None and self.dlv_enc is not None:
                g_nolang = g_nolang + (g - gc(spk, style, volume, speed, lang, **(labels or {})))

        beat = None
        if getattr(self.cfg, "use_beat", False) and labels is not None and labels.get("bpm") is not None:
            beat = self.beat_features(labels["bpm"], torch.ones(cond.shape[0], device=cond.device), cond.shape[1])
        import functools
        nfe = None
        if notes is not None and getattr(self.cfg, "use_notes", False):  # (B, T, 4) per frame, padded / cut to the length
            nfe = torch.zeros(cond.shape[0], cond.shape[1], notes.shape[-1], device=cond.device)  # (4, or 4 + beat track)
            n_ = min(nfe.shape[1], notes.shape[1]); nfe[:, :n_] = notes[:, :n_].float().to(cond.device)
        decoder = self.fast_decoder if cond.is_cuda else self.decoder  # compiled copy on the GPU (enable_fast_decoder)
        rmem = None
        if ref_seg is not None and self.ref_enc is not None:  # v14 reference segment, encoded once for every step
            r = ref_seg.float().to(cond.device)
            r = r if r.dim() == 3 else r[None]
            rmem, rmask, _ = self.ref_memory(r, torch.ones(r.shape[:2], dtype=torch.bool, device=r.device))

        ref_cache = {}

        def refs(on):
            """The decoder's `ref` for len(on) rows (on: 1 = hears the reference); None without one."""
            if rmem is None:
                return None
            k_ = tuple(on)
            if k_ not in ref_cache:  # (a list -> CUDA copy on every evaluation before; code review 2026-10-29)
                on_ = torch.as_tensor(on, dtype=torch.float32, device=cond.device)
                ref_cache[k_] = (rmem.expand(len(on), -1, -1), rmask.expand(len(on), -1), on_)
            return ref_cache[k_]

        bsz = cond.shape[0]
        decoder = functools.partial(decoder, ref=refs([1.0] * bsz))
        dec = functools.partial(decoder, beat=beat, notes=nfe)  # every pass hears the metronome + the notes
        if record is not None:  # the decoder's exact inputs (tools/vm_collect.py: preference fine-tuning on these takes)
            record.update(cond=cond.detach(), y_mask=y_mask, g=None if g is None else g.detach(), tp=tp,
                          ctx=None if ctx is None else ctx.detach(), ref=refs([1.0] * bsz))
        # note guidance: steer toward "these notes" vs the same pass without notes (rows trained note-less, note_drop)
        # -- holds the written pitch and timing tighter (1 extra decoder pass per step)
        use_ng = note_guidance != 1.0 and nfe is not None
        dec_nonotes = functools.partial(decoder, beat=beat, notes=torch.zeros_like(nfe)) if use_ng else None
        dec2 = functools.partial(decoder, beat=None if beat is None else torch.cat([beat, beat]),
                                 notes=None if nfe is None else torch.cat([nfe, torch.zeros_like(nfe) if note_cfg else nfe]),
                                 ref=refs([1.0] * bsz + [0.0] * bsz))  # (conditional, unconditional)

        use_np = ctx is not None and prompt_guidance != 1.0
        # the no-prompt pass as training's prompt-less rows saw it: without the prompt-label embedding (training adds it
        # only on prompted rows; code review 2026-10-29). OFF by default: measured, the training-consistent pass LOWERS
        # voice match (site model -0.017 +- 0.003, v14XL-L -0.020 on Seed-TTS EN 150) -- prompt guidance 2 was tuned on
        # the old one, whose extra prompt information pushes harder toward the voice
        g_np = g - pe.to(g.dtype) if (np_drop_pe and pe is not None and g is not None) else g
        apg_prev = [None]
        # the step time as a Python float, set by the step loop before every velocity call: float(tval) on the GPU
        # schedule was a device sync 2-3x per evaluation in a launch-bound loop (code review 2026-10-29)
        t_now = [0.0]

        def cfg_combine(x_, tval, vc, vu):
            """Classifier-free guidance: plain (vu + w (vc - vu)), or APG when APG is set (see APG at the top)."""
            A = APG if apg is None else (apg or None)  # per call (e.g. the website), else the module default
            if A is None:
                return vu + guidance * (vc - vu)
            one_t = (1 - tval).clamp(min=1e-3)
            mu_c = x_ + one_t * vc                    # clean-audio estimates (conditional / unconditional)
            diff = one_t * (vc - vu)                  # mu_c - mu_u
            if apg_prev[0] is not None:
                diff = diff + A.get("beta", -0.3) * apg_prev[0]
            apg_prev[0] = diff
            m = y_mask[..., None].to(diff.dtype)      # per row, over the real frames x latent dims
            par = (diff * mu_c * m).sum((1, 2), keepdim=True) / (mu_c * mu_c * m).sum((1, 2), keepdim=True).clamp(min=1e-8) * mu_c
            mu = mu_c + (guidance - 1.0) * ((diff - par) + A.get("eta", 0.5) * par)
            return (mu - x_) / one_t

        stacked = {}

        def velocity_batched(x, tval):
            """Same as velocity(), but every guidance pass of this step in ONE decoder call (stacked on the batch):
            a TTS line is short, so a pass alone leaves the GPU mostly idle (2026-10-21: ~2x faster with
            speaker guidance on). Passes differ only in global cond / text cond / prompt / notes; a missing prompt or
            note input equals zeros exactly (bias-free, zero-init input layers), so all passes fit one batch."""
            zc = torch.zeros_like(ctx) if ctx is not None else None
            zn = torch.zeros_like(nfe) if nfe is not None else None
            P = [("c", g, cond, ctx, nfe)]  # (name, global cond, text cond, prompt, notes)
            cfg_now = use_cfg and cfg_from <= t_now[0] < cfg_until
            if cfg_now:
                P.append(("u", g, torch.zeros_like(cond), zc, zn if note_cfg else nfe))  # note_cfg: no-text pass also note-less (singing: guidance pulls to the written pitch)
            for on, name, gv in ((use_lg, "lg", g_plain), (use_bg, "bg", g_lowband), (use_wg, "wg", g_normal),
                                 (use_sg, "sg", g_nospk), (use_langg, "lang", g_nolang)):
                if on:
                    P.append((name, gv, cond, ctx, nfe))
            if use_ng:
                P.append(("ng", g, cond, ctx, zn))
            if use_st:
                P += [("st_on", g_st_on, cond, ctx, nfe), ("st_off", g_st_off, cond, ctx, nfe)]
            if use_np and prompt_guidance_from <= t_now[0] < prompt_guidance_until:
                P.append(("np", g_np, cond_np, zc, nfe))
            if use_av and prompt_guidance_from <= t_now[0] < prompt_guidance_until:
                P.append(("av", g, cond, ctx_av, nfe))
            n, b = len(P), x.shape[0]
            key = tuple(p[0] for p in P)
            if key not in stacked:  # (the same passes -> the same stacked inputs: made once, code review 2026-10-29)
                cat = lambda i: torch.cat([p[i] for p in P])  # noqa: E731
                stacked[key] = (cat(2), torch.cat([y_mask] * n), None if g is None else cat(1), None if ctx is None else cat(3),
                                None if beat is None else torch.cat([beat] * n), None if nfe is None else cat(4),
                                refs([0.0 if p[0] in ("u", "np") else 1.0 for p in P for _ in range(b)]))
            c_, m_, g_, x_ctx, b_, n_, r_ = stacked[key]
            out = decoder(torch.cat([x] * n), tval.expand(n * b), c_, m_, g_, ctx=x_ctx, beat=b_, notes=n_, ref=r_).float()
            V = {p[0]: out[i * b:(i + 1) * b] for i, p in enumerate(P)}
            vc = V["c"]
            v = cfg_combine(x, tval, vc, V["u"]) if cfg_now else vc
            for on, name, w in ((use_lg, "lg", label_guidance), (use_bg, "bg", band_guidance),
                                (use_wg, "wg", whisper_guidance), (use_sg, "sg", spk_guidance),
                                (use_langg, "lang", lang_guidance), (use_ng, "ng", note_guidance)):
                if on:
                    v = v + (w - 1.0) * (vc - V[name])
            if use_st:
                v = v + style_transfer * (V["st_on"] - V["st_off"])
            if "np" in V:
                v = v + (prompt_guidance - 1.0) * (vc - V["np"])
            if "av" in V:
                v = v + anti_voice * (vc - V["av"])
            return v

        def velocity(x, tval):
            if BATCH_GUIDANCE and (g is not None or not any((use_lg, use_bg, use_wg, use_sg, use_langg, use_st))):
                return velocity_batched(x, tval)
            t = tval.expand(x.shape[0])
            if use_cfg and cfg_from <= t_now[0] < cfg_until:
                vc, vu = dec2(torch.cat([x, x]), torch.cat([t, t]), cond2, mask2, g2, ctx=ctx2).float().chunk(2)
                v = cfg_combine(x, tval, vc, vu)
            else:
                vc = dec(x, t, cond, y_mask, g, ctx=ctx).float()
                v = vc
            if use_lg:
                vp = dec(x, t, cond, y_mask, g_plain, ctx=ctx).float()
                v = v + (label_guidance - 1.0) * (vc - vp)
            if use_bg:
                vb = dec(x, t, cond, y_mask, g_lowband, ctx=ctx).float()
                v = v + (band_guidance - 1.0) * (vc - vb)
            if use_wg:
                vw = dec(x, t, cond, y_mask, g_normal, ctx=ctx).float()
                v = v + (whisper_guidance - 1.0) * (vc - vw)
            if use_sg:
                vn = dec(x, t, cond, y_mask, g_nospk, ctx=ctx).float()
                v = v + (spk_guidance - 1.0) * (vc - vn)
            if use_langg:
                vl = dec(x, t, cond, y_mask, g_nolang, ctx=ctx).float()
                v = v + (lang_guidance - 1.0) * (vc - vl)
            if use_ng:
                vnn = dec_nonotes(x, t, cond, y_mask, g, ctx=ctx).float()
                v = v + (note_guidance - 1.0) * (vc - vnn)
            if use_st:
                v_on = dec(x, t, cond, y_mask, g_st_on, ctx=ctx).float()
                v_off = dec(x, t, cond, y_mask, g_st_off, ctx=ctx).float()
                v = v + style_transfer * (v_on - v_off)
            # lean harder on what the prompt implies; until < 1: only in the early steps, where the coarse
            # intonation / rhythm forms (later steps refine sound, where extra pull costs naturalness)
            if ctx is not None and prompt_guidance != 1.0 and prompt_guidance_from <= t_now[0] < prompt_guidance_until:
                vnp = dec(x, t, cond_np, y_mask, g_np, ref=refs([0.0] * x.shape[0])).float()
                v = v + (prompt_guidance - 1.0) * (vc - vnp)
            if use_av and prompt_guidance_from <= t_now[0] < prompt_guidance_until:
                v = v + anti_voice * (vc - dec(x, t, cond, y_mask, g, ctx=ctx_av).float())
            return v

        if schedule is not None:  # explicit step times 0 = t0 < ... < tN = 1 (e.g. an EPSS-style schedule)
            ts = torch.as_tensor(schedule, dtype=torch.float32, device=x.device)
            steps = len(ts) - 1
        else:
            u = torch.linspace(0, 1, steps + 1, device=x.device)
            ts = u + sway * (torch.cos(math.pi / 2 * u) - 1 + u)
        if start_t is not None and align_to is not None:  # (the same schedule, squeezed into start_t .. 1)
            ts = float(start_t) + (1 - float(start_t)) * ts
        handover = None
        if pick_codes and self.code_head is not None:
            code_t = self.cfg.code_t if code_t is None else code_t
        pin_x0 = x / max(float(temperature), 1e-6) if ((PROMPT_PIN or (infill is not None and infill.get("pin_prompt"))) and tp) else None  # unit noise, as in training
        pin_x1 = prompt["latents"].float().to(x.device) if pin_x0 is not None else None

        inf_x0 = x / max(float(temperature), 1e-6) if (inf is not None and infill.get("pin", True)) else None  # (11-24 pin=False: the real audio after only as known context)

        soft = infill.get("soft") if infill is not None else None  # (11-24) {"frames", "lat", "until"}: frames held to the
        # ORIGINAL recording's latents while t < until, then free -- the regenerated neighbour words keep the recording's own
        # timing / pitch / flow (a timing plan alone: Grey waited ~0.7 s, then rushed the words at 0.78x)
        soft_x0 = x / max(float(temperature), 1e-6) if soft is not None else None
        if soft is not None:
            soft_f = torch.as_tensor(soft["frames"], dtype=torch.long, device=x.device)
            soft_l = soft["lat"].float().to(x.device)

        def pin(x_, tv):
            if pin_x0 is None and inf_x0 is None and (soft_x0 is None or float(tv) >= float(soft["until"])):
                return x_
            x_ = x_.clone()
            if soft_x0 is not None and float(tv) < float(soft["until"]):
                x_[:, soft_f] = (1 - (1 - self.cfg.sigma_min) * tv) * soft_x0[:, soft_f] + tv * soft_l
            if pin_x0 is not None:
                x_[:, :tp] = (1 - (1 - self.cfg.sigma_min) * tv) * pin_x0[:, :tp] + tv * pin_x1
            if inf_x0 is not None:  # (speech editing) the real audio after the edit, at this step's noise level
                s0_, suf_ = inf
                e_ = s0_ + suf_.shape[1]
                x_[:, s0_:e_] = (1 - (1 - self.cfg.sigma_min) * tv) * inf_x0[:, s0_:e_] + tv * suf_
            return x_
        ts_f = ts.tolist()  # (one sync for the whole schedule; t_now gives each evaluation its time as a float)
        for i in range(steps):
            x = pin(x, ts[i])
            t_now[0] = ts_f[i]
            if pick_codes and self.code_head is not None and ts_f[i] >= code_t:
                handover = x + (1 - ts[i]) * velocity(x, ts[i])  # one-jump clean estimate; picker does the rest
                break
            dt = ts[i + 1] - ts[i]
            if solver == "midpoint":  # 2nd order: 2 evaluations per step, far less smoothing per evaluation
                x_mid = pin(x + 0.5 * dt * velocity(x, ts[i]), ts[i] + 0.5 * dt)
                t_now[0] = ts_f[i] + 0.5 * (ts_f[i + 1] - ts_f[i])
                x = x + dt * velocity(x_mid, ts[i] + 0.5 * dt)
            else:
                v_i = velocity(x, ts[i])
                if trace is not None:  # (t, velocity) along the path: where the trajectory bends
                    trace.append((ts_f[i], v_i.detach().float().cpu()))
                x = x + dt * v_i
            # (only in the thread that set it: a visitor's hook, run by the warm-up thread, stopped the warm-up with
            # "the visitor left", 2026-10-24)
            if PASS_HOOK is not None and getattr(PASS_HOOK, "thread", threading.get_ident()) == threading.get_ident():
                PASS_HOOK(i + 1, steps)
        if pick_codes and self.code_head is not None:  # -> also the DAC codes chosen by the code picker
            p = self.cfg.patch
            x = x if handover is None else handover
            codes = self.code_head.sample(unpatchify(x, p), cond, seq_mask(y_len * p, x.shape[1] * p),
                                          temperature=code_temperature, top_p=code_top_p)
            return unpatchify(x, p), y_len * p, codes
        if record is not None:
            record.update(x1=x.detach().clone(), y_len=y_len)
        if tp:  # only the continuation
            x, y_len = x[:, tp:], y_len - tp
        return unpatchify(x, self.cfg.patch), y_len * self.cfg.patch

    @torch.no_grad()
    def convert(self, content, contour, g=None, g_nospk=None, prompt=None, ref_seg=None, steps=8, guidance=2.0,
                temperature=0.85, sway=-1.0, spk_guidance=1.0, exact_pitch=False, tts_cond=None, g_tts=None, accent=0.0,
                record=None, solver="midpoint", batched=False):
        """Voice-to-voice (cfg.use_vtv, dacts/vtv.py; v14xl-vtv 2026-10-30): the decoder conditioned on a recording's
        content (ContentVec, (1, T, 768)) + contour ((1, T, 3), patch_contour of its normalized latents) instead of
        text, as on training's VTV rows. The voice comes from `g` (a learned voice) and / or `prompt` {"latents"
        (1, Tp, D) normalized, "content", "contour"}: zero-shot, the reference's frames come first and the converted
        recording continues them, as prompted cloning does. g_nospk: the same global condition without the speaker
        (speaker guidance). exact_pitch: the "exact pitch" mode (singing; trained with cfg.vtv_exact_prob, ignored by
        models without it): the output follows the contour's pitch channel frame by frame, which should then be RMVPE
        pitch as in training (dacts/sing.py). Returns normalized latents (1, T * patch, latent_dim) of the converted part.
        accent (you 2026-11-07: "accent strength from 0 to 150"; untrained shortcut, evals/vtv_accent_blend.py): with
        tts_cond -- a TTS conditioning of the clip's transcript aligned to these frames (synthesize(align_to=...,
        record=...)["cond"], g_tts its g) -- the conditional velocity is (1 - accent) v_vtv + accent v_tts: 0 = the
        source's pronunciation, 1 = the target voice's own accent on the source's timing, > 1 pushes further."""
        from .vtv import contour_feats
        tp = 0
        if prompt is not None:
            tp = prompt["latents"].shape[1]
            content = torch.cat([prompt["content"].float(), content.float()], 1)
            contour = torch.cat([prompt["contour"].float(), contour.float()], 1)
        cond = self.vtv_in(torch.cat([content.float(), contour_feats(contour.float())], -1))
        if exact_pitch and getattr(self, "vtv_exact", None) is not None:  # (whole row, prompt included, as in training)
            cond = cond + self.vtv_exact.to(cond.dtype)
        b, n = cond.shape[:2]
        y_mask = torch.ones(b, n, dtype=torch.bool, device=cond.device)
        ctx = None
        if prompt is not None:
            extra = CLAB_DIM if self.decoder.clab_in is not None else 0
            ctx = torch.zeros(b, n, self.cfg.token_dim + 1 + extra, device=cond.device)
            ctx[:, :tp, :self.cfg.token_dim] = prompt["latents"].float()
            ctx[:, :tp, self.cfg.token_dim] = 1.0
        ref = None
        if ref_seg is not None and self.ref_enc is not None:
            r = ref_seg.float().to(cond.device)
            r = r if r.dim() == 3 else r[None]
            rmem, rmask, _ = self.ref_memory(r, torch.ones(r.shape[:2], dtype=torch.bool, device=r.device))
            ref = lambda on: (rmem.expand(len(on), -1, -1), rmask.expand(len(on), -1),  # noqa: E731
                              torch.as_tensor(on, dtype=torch.float32, device=cond.device))
        decoder = self.fast_decoder if cond.is_cuda else self.decoder
        use_sg = spk_guidance != 1.0 and g_nospk is not None

        use_acc = tts_cond is not None and accent > 0
        if use_acc:
            tts_cond = tts_cond.to(cond.device, cond.dtype)[:, :n]
            if tts_cond.shape[1] < n:
                tts_cond = torch.cat([tts_cond, tts_cond.new_zeros(b, n - tts_cond.shape[1], tts_cond.shape[2])], 1)

        # batched (2026-11-11, tools/enh_pairs.py: "why does this take so long"): the conditional and the unconditional
        # pass of a step in ONE decoder call, as synthesize's velocity_batched -- the loop is bound by kernel launches,
        # so two calls of one row cost about twice one call of two rows. Same maths; opt-in (the site's calls unchanged),
        # and only for the plain case (no accent blend, no speaker guidance).
        both = None
        if batched and guidance != 1.0 and not use_acc and not use_sg:
            both = (torch.cat([cond, torch.zeros_like(cond)]), torch.cat([y_mask, y_mask]),
                    None if g is None else torch.cat([g, g]),
                    None if ctx is None else torch.cat([ctx, torch.zeros_like(ctx)]),
                    None if ref is None else ref([1.0] * b + [0.0] * b))

        def velocity(x, tval):
            if both is not None:
                vc, vu = decoder(torch.cat([x, x]), tval.expand(2 * b), both[0], both[1], both[2], ctx=both[3],
                                 ref=both[4]).float().chunk(2)
                return vu + guidance * (vc - vu)
            t = tval.expand(b)
            vc = decoder(x, t, cond, y_mask, g, ctx=ctx, ref=None if ref is None else ref([1.0] * b)).float()
            if use_acc:  # (the accent blend: the target's own pronunciation of the transcript, on the clip's frames)
                vt = decoder(x, t, tts_cond, y_mask, g if g_tts is None else g_tts, ctx=ctx,
                             ref=None if ref is None else ref([1.0] * b)).float()
                vc = vc + accent * (vt - vc)
            v = vc
            if guidance != 1.0:  # unconditional: no content, no reference (as training's dropped rows)
                vu = decoder(x, t, torch.zeros_like(cond), y_mask, g, ctx=None if ctx is None else torch.zeros_like(ctx),
                             ref=None if ref is None else ref([0.0] * b)).float()
                v = vu + guidance * (vc - vu)
            if use_sg:
                vn = decoder(x, t, cond, y_mask, g_nospk, ctx=ctx, ref=None if ref is None else ref([1.0] * b)).float()
                v = v + (spk_guidance - 1.0) * (vc - vn)
            return v

        x = torch.randn(b, n, self.cfg.token_dim, device=cond.device) * temperature
        if record is not None:  # (tools/xs_vtv_teacher.py: Smol learns the straight path from this noise to the result)
            record["x0"] = x.detach().clone()
        u = torch.linspace(0, 1, steps + 1, device=x.device)
        ts = u + sway * (torch.cos(math.pi / 2 * u) - 1 + u)
        if solver == "euler":  # (Smol v14.5: distilled for N plain Euler steps on a uniform grid, as synthesize's)
            ts = torch.linspace(0, 1, steps + 1, device=x.device)
        for i in range(steps):  # midpoint (Euler for few-step distilled models)
            dt = ts[i + 1] - ts[i]
            if solver == "euler":
                x = x + dt * velocity(x, ts[i])
            else:
                x_mid = x + 0.5 * dt * velocity(x, ts[i])
                x = x + dt * velocity(x_mid, ts[i] + 0.5 * dt)
            # (the website's progress bar + its stop when the visitor left / it ran too long: as synthesize does)
            if PASS_HOOK is not None and getattr(PASS_HOOK, "thread", threading.get_ident()) == threading.get_ident():
                PASS_HOOK(i + 1, steps)
        if record is not None:
            record["x1"] = x.detach().clone()
        return unpatchify(x[:, tp:], self.cfg.patch)
