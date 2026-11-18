"""A fast stand-in for KVAE's encoder (owner 2026-10-28: "let's think about making a faster kvae encoder that stays as
faithful as it can"). KVAE's encoder is a DAC-style conv stack (86M) that starts on the raw 48 kHz samples (64 channels
at 48 kHz, 128 at 24 kHz, ...) and ends in one self-attention over the whole clip at 50 Hz; at ~17 clips/s on the GPU
it is why encoding a new dataset takes most of a day, and on the CPU (clone uploads, Grey Studio) it is slow too.

This one works on a short-time spectrum at 100 frames/s instead of the samples: a complex STFT (40 ms windows, magnitude
compressed, phase kept) -> ConvNeXt blocks at 100 Hz -> stride 2 to KVAE's 50 Hz -> ConvNeXt blocks -> a few
transformer blocks over the whole clip (KVAE attends over the whole clip too, so the student must see it all) -> the 64
latent means. The fast decoder's mirror (dacts/fast_codec.py).

Distilled from KVAE (tools/kvae_encoder_train.py) on the latents we already stored for 12 datasets: the target for a
clip is exactly KVAE's encoding of it. models/kvae-encoder.pt.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fast_codec import ConvNeXt

SR, HOP, N_FFT = 48000, 480, 1920  # 100 frames/s, 2 per KVAE latent frame


class Attn(nn.Module):
    def __init__(self, dim, heads, mult=4):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.o = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.ff = nn.Sequential(nn.Linear(dim, mult * dim), nn.GELU(), nn.Linear(mult * dim, dim))

    def forward(self, x, pad):  # x (B, T, dim), pad (B, T) True = padding
        b, t, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, t, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        q, k = rope(q), rope(k)
        mask = None if pad is None else ~pad[:, None, None, :]
        x = x + self.o(F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(b, t, d))
        return x + self.ff(self.n2(x))


def rope(x):
    """rotary position (B, H, T, hd)."""
    t, hd = x.shape[-2], x.shape[-1]
    f = torch.exp(-math.log(10000.0) * torch.arange(0, hd, 2, device=x.device, dtype=torch.float32) / hd)
    a = torch.arange(t, device=x.device, dtype=torch.float32)[:, None] * f[None]
    cos, sin = a.cos().to(x.dtype), a.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class FastEncoder(nn.Module):
    def __init__(self, mean, std, dim=512, layers_hi=8, layers_lo=4, attn=4, heads=8, mult=3):
        super().__init__()
        self.settings = dict(dim=dim, layers_hi=layers_hi, layers_lo=layers_lo, attn=attn, heads=heads, mult=mult)
        self.register_buffer("mean", torch.as_tensor(mean).float().view(1, 1, -1))  # (the latents' own spread: the
        self.register_buffer("std", torch.as_tensor(std).float().view(1, 1, -1))    # head predicts normalized latents)
        self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)
        self.inp = nn.Linear(2 * (N_FFT // 2 + 1), dim)
        self.norm = nn.LayerNorm(dim)
        n = layers_hi + layers_lo
        self.hi = nn.ModuleList(ConvNeXt(dim, mult * dim, 1 / n) for _ in range(layers_hi))
        self.down = nn.Conv1d(dim, dim, 4, stride=2, padding=1)  # 100 -> 50 frames/s
        self.lo = nn.ModuleList(ConvNeXt(dim, mult * dim, 1 / n) for _ in range(layers_lo))
        self.attn = nn.ModuleList(Attn(dim, heads) for _ in range(attn))
        self.final = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, self.mean.shape[-1])

    def features(self, wav):
        """(B, N) 48 kHz -> (B, F, 2 * bins): the STFT, magnitude compressed (^0.3) with its phase kept (real, imag)."""
        with torch.autocast(wav.device.type, enabled=False):
            X = torch.stft(wav.float(), N_FFT, HOP, N_FFT, self.window, center=True, return_complex=True)
            m = X.abs().clamp(min=1e-7)
            X = X * (m ** 0.3 / m)
            return torch.cat([X.real, X.imag], 1).transpose(1, 2)

    def forward(self, wav, frames=None, lengths=None):
        """(B, N) audio at 48 kHz, levelled like encode_kvae.py (speech at -20 dB + limiter) -> raw KVAE latent means
        (B, T, 64) at 50 Hz. frames: T (default N / 960 rounded up); lengths: (B,) samples per row, for padded batches."""
        b, n = wav.shape
        t = frames or -(-n // 960)
        x = self.norm(self.inp(self.features(wav))).transpose(1, 2)  # (B, dim, F)
        for blk in self.hi:
            x = blk(x)
        x = self.down(x)
        for blk in self.lo:
            x = blk(x)
        x = x[..., :t]
        if x.shape[-1] < t:
            x = F.pad(x, (0, t - x.shape[-1]), mode="replicate")
        x = x.transpose(1, 2)
        pad = None
        if lengths is not None:
            pad = torch.arange(t, device=wav.device)[None] >= (-(-lengths.to(wav.device) // 960))[:, None]
        for blk in self.attn:
            x = blk(x, pad)
        return self.head(self.final(x)).float() * self.std + self.mean

    def state(self):
        return {"settings": self.settings, "model": self.state_dict()}

    @classmethod
    def load(cls, path, device="cpu"):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck["model"]
        m = cls(sd["mean"].flatten(), sd["std"].flatten(), **ck["settings"])
        m.load_state_dict(sd)
        return m.to(device).eval()
