"""Pitch / voicing / loudness read straight from KVAE latents (50 frames/s).

Word emphasis is learned from how much each word sticks out of its sentence (pitch, loudness, length).
Pitch and loudness need audio, but the latents already encode them: this small conv net predicts them
from latents, so prominence can be measured on the fly during training for ALL data (including datasets
whose raw audio was deleted). Trained once by train_contour.py on clips that still have audio.
"""
import torch
import torch.nn as nn


class ContourNet(nn.Module):
    """(B, T, 64) normalized latents -> (B, T, 3): pitch [semitones re 100 Hz], voiced logit, loudness [dB / 10]."""

    def __init__(self, c: int = 64, d: int = 192):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(c, d, 5, padding=2), nn.GELU(), nn.Conv1d(d, d, 5, padding=2), nn.GELU(),
                                 nn.Conv1d(d, d, 5, padding=2), nn.GELU(), nn.Conv1d(d, 3, 1))

    def forward(self, z):
        return self.net(z.transpose(1, 2)).transpose(1, 2)


def load_contour(path, device="cuda"):
    m = ContourNet()
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.to(device).eval().requires_grad_(False)
