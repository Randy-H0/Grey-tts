"""Swift-RM, the rhythm model (you 2026-11-08: "we might need a better rhythm etc model, because the ones we have might
not be designed for this big of a project", "needs to communicate more with other things", "the big decoder is better at
handling pitch and volume. we should make a new rhythm model").

The old head (model.FlowDurationPredictor, 1M parameters): 4 convolutions over the text encoder's output, each seeing
one token either side -- a sound's length decided from ~4 characters around it; it trained detached from the encoder
and its loss stayed flat for all of v14.5 XL (0.274 -> 0.266 over 1M steps). Rhythm probe (evals/rhythm_probe.py):
per-sound log-length correlation with real recordings 0.14-0.36, most real pauses missed.

Swift-RM: the same job and the same interface (flow matching over log(1 + frames) per token, sampled in 10 Euler steps
at a temperature, so takes keep their natural variety), but a transformer over the WHOLE line (8 layers x 384, ~15M
parameters): every sound's length is decided with the full sentence in view (phrasing, where the pauses go, which words
carry the stress), and it also reads the global condition g directly (the voice, the language and the delivery labels:
speed, style, mood ...). Pitch and loudness stay with the decoder. It replaces only the duration sampling of a trained
model: DacTTS.rhythm = SwiftRM.load(...) (model.DacTTS.synthesize checks for it), nothing else changes.
"""
import math
from pathlib import Path

import torch
import torch.nn as nn


def t_embed(t, dim=256):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t.float()[:, None] * 1000 * f[None]
    return torch.cat([a.sin(), a.cos()], -1)


class SwiftRM(nn.Module):
    def __init__(self, h_dim=256, g_dim=256, d=384, layers=8, heads=6, dropout=0.1):
        super().__init__()
        self.cfg = dict(h_dim=h_dim, g_dim=g_dim, d=d, layers=layers, heads=heads, dropout=dropout)
        self.inp = nn.Linear(h_dim + 1, d)
        self.g_proj = nn.Linear(g_dim, d)
        self.t_mlp = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        self.pos = nn.Conv1d(d, d, 5, padding=2, groups=d)  # (local order: a depthwise convolution as position signal)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True,
                                                                activation="gelu") for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.ctx = {}  # {"g": (B, g_dim)} set by the caller before sample() (DacTTS.synthesize)

    def velocity(self, w_t, t, h, mask, g=None):
        x = self.inp(torch.cat([h.float(), w_t.float()[..., None]], -1)) + self.t_mlp(t_embed(t))[:, None]
        if g is not None:
            x = x + self.g_proj(g.float().expand(h.shape[0], -1))[:, None]
        m = mask.bool()
        x = x + self.pos((x * m[..., None]).transpose(1, 2)).transpose(1, 2)
        for b in self.blocks:
            x = b(x, src_key_padding_mask=~m)
        return self.out(self.norm(x)).squeeze(-1) * m

    def forward(self, w_t, t, h, mask, g=None):
        """= velocity, through __call__: what DacTTS.compile_modules compiles when Swift-RM is inside the model (v16)."""
        return self.velocity(w_t, t, h, mask, g)

    def loss(self, w1, h, mask, g=None):
        t = torch.sigmoid(torch.randn(w1.shape[0], device=w1.device))
        w0 = torch.randn_like(w1)
        w_t = (1 - t[:, None]) * w0 + t[:, None] * w1
        v = self(w_t, t, h, mask, g)
        m = mask.float()
        return ((v - (w1 - w0)) ** 2 * m).sum() / m.sum()

    @torch.no_grad()
    def sample(self, h, mask, steps=10, temperature=0.8, span=None, g=None, **_):  # (**_: v2's extra inputs, unused here)
        g = self.ctx.get("g") if g is None else g
        w = torch.randn(h.shape[:2], device=h.device) * temperature
        for i in range(steps):
            t = torch.full((h.shape[0],), i / steps, device=h.device)
            w = w + self.velocity(w, t, h, mask, g) / steps
        return w * mask

    def save(self, path, **meta):
        torch.save({"cfg": self.cfg, "model": self.state_dict(), **meta}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ck = torch.load(Path(path), map_location="cpu", weights_only=False)
        m = cls(**ck["cfg"])
        m.load_state_dict(ck["model"])
        return m.eval().to(device)


class SwiftRM2(nn.Module):
    """Swift-RM 2 (you 2026-11-08: "option 2"): Swift-RM with its OWN copy of the text encoder, trained for timing.
    v1 reads the encoder output the decoder uses -- features shaped only by the sound, never by timing -- and levelled
    off (held-out per-sound correlation ~0.26). Here the rhythm side has its own TextEncoder (v14.5 XL's, copied: the
    same weights at the start, so it starts exactly at v1), fed the same inputs as the original -- tokens, the global
    condition g (voice + labels), XLM-R's features of the line, Stress marks, the reference's prosody for clones -- and
    trained by the rhythm loss end to end. XLM-R stays frozen and the decoder keeps the original encoder: sound and
    cloning can't change, only the timing."""

    def __init__(self, enc_cfg: dict, vocab: int, head_cfg: dict):
        super().__init__()
        from .config import Config
        from .model import TextEncoder
        self.enc_cfg, self.vocab = dict(enc_cfg), int(vocab)
        self.encoder = TextEncoder(Config.from_dict(enc_cfg), vocab)
        self.encoder.dur = None  # (the old rhythm head inside the copy isn't used)
        self.head = SwiftRM(**head_cfg)
        self.ctx = {}

    def h_of(self, tokens, mask, g=None, lm=None, emph=None, pctx=None, tlab=None):
        _, h = self.encoder(tokens, mask, g, lm, emph=emph, pctx=pctx, tlab=tlab)
        return h

    def loss(self, w1, tokens, mask, g=None, lm=None, g_head=None):
        h = self.h_of(tokens, mask, g, lm)
        return self.head.loss(w1, h, mask, g if g_head is None else g_head)

    @torch.no_grad()
    def sample(self, h, mask, steps=10, temperature=0.8, span=None, g=None, tokens=None, lm=None, emph=None, pctx=None,
               tlab=None):
        """The same call as SwiftRM.sample (h = the original encoder's output, used only when tokens aren't given)."""
        if tokens is not None:
            h = self.h_of(tokens, mask.bool(), g, lm, emph, pctx, tlab)
        return self.head.sample(h, mask, steps, temperature, g=g)

    def save(self, path, **meta):
        torch.save({"kind": "swift-rm2", "enc_cfg": self.enc_cfg, "vocab": self.vocab, "head_cfg": self.head.cfg,
                    "model": self.state_dict(), **meta}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ck = torch.load(Path(path), map_location="cpu", weights_only=False)
        m = cls(ck["enc_cfg"], ck["vocab"], ck["head_cfg"])
        m.load_state_dict(ck["model"])
        return m.eval().to(device)


def load_rhythm(path, device="cpu"):
    """Swift-RM v1 or v2 from its file."""
    ck = torch.load(Path(path), map_location="cpu", weights_only=False, mmap=True)
    return (SwiftRM2 if ck.get("kind") == "swift-rm2" else SwiftRM).load(path, device)
