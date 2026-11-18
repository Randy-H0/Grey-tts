"""Accent judge on KVAE latents: reads (normalized) latents, predicts the English accent (dacts.accent.ACCENTS).

Trained once (train_accent_judge.py) on real clips, then frozen and used as an extra training loss: the
TTS model's one-step clean estimate of the frames it generates is judged, and the judge's accent is pulled
toward the speaker's accent label. Works directly on latents, so it costs a few small convolutions per
step (no audio decoding). Its inputs are augmented in training to look like those blurry one-step
estimates (temporal smoothing + shrinkage + noise), so it judges the estimates the way it judges real audio.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .accent import ACCENTS


class AccentJudge(nn.Module):
    def __init__(self, dim: int = 64, hidden: int = 256):
        super().__init__()
        layers, c = [], dim
        for dil in (1, 2, 4, 8, 1, 2):
            layers += [nn.Conv1d(c, hidden, 5, padding=2 * dil, dilation=dil), nn.GroupNorm(8, hidden), nn.GELU()]
            c = hidden
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(2 * hidden, len(ACCENTS))

    def forward(self, z, mask=None):
        """z (B, T, dim) normalized latents, mask (B, T) bool -> logits (B, len(ACCENTS))."""
        h = self.net(z.transpose(1, 2))  # B, H, T
        m = torch.ones(z.shape[:2], device=z.device, dtype=torch.bool) if mask is None else mask
        mf = m[:, None].to(h.dtype)
        n = mf.sum(-1).clamp(min=1)
        mu = (h * mf).sum(-1) / n
        sd = (((h - mu[..., None]) ** 2 * mf).sum(-1) / n).clamp(min=1e-6).sqrt()
        return self.out(torch.cat([mu, sd], -1))


def soft_ce(logits, target):
    """Cross-entropy against soft targets (rows of probabilities), per row."""
    return -(target * F.log_softmax(logits.float(), -1)).sum(-1)


def load_judge(path, device="cuda"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    j = AccentJudge(**ck.get("args", {}))
    j.load_state_dict(ck["model"])
    return j.to(device).eval().requires_grad_(False)
