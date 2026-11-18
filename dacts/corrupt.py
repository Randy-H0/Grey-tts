"""Recording-condition damage for prompt-robustness training (make_corrupt.py).

Everything keeps the timing of the clip (no shifts, reverb tails cut at the clip's end), so the damaged
latents line up frame for frame with the clean ones: the model is shown a damaged prompt and still has
to continue in the clean voice, which teaches it to take the voice from a prompt and leave the room,
the noise and the microphone behind.

All noise is made here (no third-party audio): coloured noise, mains hum, synthetic music (chords +
percussion), other speakers' speech (babble, from our own data), reverb, phone / muffled bandwidth,
clipping and 8-bit companding.
"""
import math
import random

import torch
import torchaudio.functional as AF

SR = 48000


def _rms(x):
    return x.pow(2).mean().sqrt().clamp(min=1e-6)


def _mix(x, n, snr_db):
    return x + n * (_rms(x) / _rms(n)) * 10 ** (-snr_db / 20)


def coloured_noise(n, device, slope=None):
    """slope 0 = white, -1 = pink, -2 = brown (power ~ f^slope)."""
    slope = random.choice([0.0, -1.0, -2.0]) if slope is None else slope
    X = torch.fft.rfft(torch.randn(n, device=device))
    f = torch.fft.rfftfreq(n, 1 / SR).to(device).clamp(min=20.0)
    return torch.fft.irfft(X * f ** (slope / 2), n=n)


def hum(n, device):
    t = torch.arange(n, device=device) / SR
    f0 = random.choice([50.0, 60.0])
    return sum(random.uniform(0.2, 1.0) / k * torch.sin(2 * math.pi * f0 * k * t + random.uniform(0, 6.3))
               for k in range(1, 8)) + 0.05 * coloured_noise(n, device, -1.0) / 10


def music(n, device):
    """Chords of harmonic tones changing every 0.4-1.5 s, a bass line and noise-burst percussion."""
    t = torch.arange(n, device=device) / SR
    out = torch.zeros(n, device=device)
    pos, root = 0, random.randint(45, 60)
    while pos < n:
        ln = int(random.uniform(0.4, 1.5) * SR)
        seg = slice(pos, min(n, pos + ln))
        tt = t[seg] - t[pos]
        env = torch.exp(-tt * random.uniform(0.5, 4.0))
        chord = root + random.choice([0, 5, 7, -3, 2]) + torch.tensor(random.choice([[0, 4, 7], [0, 3, 7], [0, 4, 7, 11], [0, 3, 7, 10]]))
        for m in chord.tolist() + [chord[0].item() - 12]:
            f = 440 * 2 ** ((m - 69) / 12)
            bright = random.uniform(0.3, 0.8)
            out[seg] += env * sum(bright ** k * torch.sin(2 * math.pi * f * (k + 1) * tt) for k in range(5))
        pos += ln
    beat = int(SR * 60 / random.uniform(80, 150))
    for k in range(0, n, beat // 2):  # hats every half beat, kicks / snares on beats
        L = min(n - k, int(0.12 * SR))
        dec = torch.exp(-torch.arange(L, device=device) / SR * random.choice([60.0, 25.0]))
        hit = coloured_noise(L, device, 0.0 if (k // (beat // 2)) % 2 else -2.0) if L > 32 else torch.zeros(L, device=device)
        out[k:k + L] += 0.6 * dec * hit / _rms(hit)
    return out


def reverb(x):
    rt60 = random.uniform(0.25, 1.2)
    L = int(min(rt60, 1.0) * SR)
    tt = torch.arange(L, device=x.device) / SR
    ir = coloured_noise(L, x.device, -0.5) * torch.exp(-6.9 * tt / rt60)
    ir = AF.lowpass_biquad(ir, SR, random.uniform(3000, 9000))
    ir[0] = 0
    ir = ir / ir.abs().sum().clamp(min=1e-6) * random.uniform(1.5, 6.0)
    ir[0] = 1.0  # direct path at t = 0: no timing shift
    y = AF.fftconvolve(x, ir)[: len(x)]
    return y


def phone(x):
    y = AF.resample(x, SR, 8000)
    X = torch.fft.rfft(y)
    f = torch.fft.rfftfreq(len(y), 1 / 8000).to(x.device)
    y = torch.fft.irfft(X * ((f > 300) & (f < 3400)), n=len(y))
    if random.random() < 0.7:  # 8-bit mu-law, like old telephony
        p = y.abs().max().clamp(min=1e-6)
        y = AF.mu_law_decoding(AF.mu_law_encoding(y / p, 256), 256) * p
    return AF.resample(y, 8000, SR)[: len(x)]


def muffle(x):
    return lowpass(x, random.uniform(1800, 5000))


def clip(x):
    g = 10 ** (random.uniform(6, 18) / 20)
    p = x.abs().max().clamp(min=1e-6)
    return (x / p * g).clamp(-1, 1) * p


class Bank:
    """Pre-made music tracks and room responses (made once on the GPU, then cut at random offsets):
    synthesising them per clip was most of the cost."""

    def __init__(self, device, n_music=48, music_s=16.0, n_ir=96):
        self.music = torch.stack([music(int(music_s * SR), device) for _ in range(n_music)])
        self.music = self.music / self.music.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        self.irs = [room(device) for _ in range(n_ir)]

    def get_music(self, n):
        m = self.music[random.randrange(len(self.music))]
        k = n // len(m) + 2
        m = m.repeat(k)
        s = random.randrange(len(m) - n)
        return m[s:s + n]

    def reverb(self, x):
        ir = random.choice(self.irs)
        return AF.fftconvolve(x, ir)[: len(x)]


def room(device):
    rt60 = random.uniform(0.25, 1.2)
    L = int(min(rt60, 1.0) * SR)
    tt = torch.arange(L, device=device) / SR
    ir = coloured_noise(L, device, -0.5) * torch.exp(-6.9 * tt / rt60)
    ir = AF.lowpass_biquad(ir, SR, random.uniform(3000, 9000))
    ir[0] = 0
    ir = ir / ir.abs().sum().clamp(min=1e-6) * random.uniform(1.5, 6.0)
    ir[0] = 1.0  # direct path at t = 0: no timing shift
    return ir


def lowpass(x, fc):
    """Steep FFT low-pass (a cosine edge 15% wide): fast on long clips, no time shift."""
    X = torch.fft.rfft(x)
    f = torch.fft.rfftfreq(len(x), 1 / SR).to(x.device)
    g = 0.5 + 0.5 * torch.cos(torch.pi * torch.clamp((f - fc) / (0.15 * fc), 0, 1))
    return torch.fft.irfft(X * g, n=len(x))


def corrupt(x, others=(), bank=None):
    """x (N,) 48 kHz mono -> damaged copy, same length; others: other speakers' audio for babble.
    Returns (audio, list of applied damage names)."""
    n, dev, done = len(x), x.device, []
    kinds = ["noise", "music", "babble", "hum", "reverb", "phone", "muffle", "clip"]
    weights = [3, 3, 2 if others else 0, 1, 3, 1.5, 1.5, 1]
    for k in set(random.choices(kinds, weights, k=random.choice([1, 1, 2, 2, 3]))):
        if k == "reverb":
            x = bank.reverb(x) if bank is not None else reverb(x)
        elif k == "phone":
            x = phone(x)
        elif k == "muffle":
            x = muffle(x)
        elif k == "clip":
            x = clip(x)
        else:
            if k == "noise":
                nz, snr = coloured_noise(n, dev), random.uniform(0, 20)
            elif k == "music":
                nz, snr = (bank.get_music(n) if bank is not None else music(n, dev)), random.uniform(-3, 15)
            elif k == "hum":
                nz, snr = hum(n, dev), random.uniform(5, 25)
            else:
                o = [y for y in others if len(y)][: random.randint(1, 3)]
                nz = sum(torch.cat([y] * (n // len(y) + 1))[:n] for y in o)
                snr = random.uniform(3, 15)
            x = _mix(x, nz, snr)
        done.append(k)
    return x, done
