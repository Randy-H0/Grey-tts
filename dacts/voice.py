"""Voice-quality measures (DiffSinger-style variance parameters), from harmonic / noise separation.

Per 10 ms voiced frame (YIN f0), the spectrum is split with a harmonic comb: bins near k*f0 are
harmonic, bins between harmonics estimate the noise floor, which is assumed flat across the band.
  breathiness = aperiodic share of the voiced sound, dB  (airy / breathy voice is high)
  tension     = harmonic energy at 1-5 kHz relative to the first harmonic, dB (pressed voice is high)
Utterance values are medians over voiced frames; clips that are hardly voiced (whispers) are marked.
"""
import numpy as np
import torch

from .style import yin

SR = 16000
NFFT, HOP = 1024, 160


@torch.no_grad()
def voice_features(w16: torch.Tensor):
    """w16: 16 kHz mono tensor (on the device to compute on). -> dict or None (too little voicing)."""
    f0, voiced, rms_db, ap = yin(w16, return_ap=True)
    x = torch.nn.functional.pad(w16, (NFFT // 2, NFFT // 2))
    spec = torch.stft(x, NFFT, HOP, window=torch.hann_window(NFFT, device=w16.device), center=False,
                      return_complex=True).abs().pow(2).T  # (frames, bins)
    n = min(len(f0), spec.shape[0])
    f0, voiced, spec, ap = f0[:n], voiced[:n], spec[:n], ap[:n]
    loud = 10 * torch.log10(spec.sum(1) + 1e-10)
    speech = loud > loud.max() - 35
    vf = float((voiced & speech).sum()) / max(1, int(speech.sum()))
    sel = voiced & speech & (f0 > 60) & (f0 < 600)
    if int(sel.sum()) < 10:
        return {"voiced_frac": round(vf, 3), "breath_db": None, "tension_db": None}
    f0s, S = f0[sel], spec[sel]  # (m,), (m, bins)
    freqs = torch.arange(S.shape[1], device=S.device) * (SR / NFFT)
    h = freqs[None] / f0s[:, None]  # harmonic number per bin
    dist = (h - h.round()).abs() * f0s[:, None]  # Hz from the nearest harmonic
    band = (freqs[None] > 0.5 * f0s[:, None]) & (freqs[None] < 7500)
    harm = band & (dist < 0.18 * f0s[:, None]) & (h.round() >= 1)
    gap = band & (dist > 0.35 * f0s[:, None])
    noise_density = (S * gap).sum(1) / gap.sum(1).clamp(min=1)
    total = (S * band).sum(1)
    noise = noise_density * band.sum(1)
    harmonic = ((S * harm).sum(1) - noise_density * harm.sum(1)).clamp(min=1e-10)
    # breathiness: YIN aperiodicity (pitch-robust), over periodic-enough loud frames; breathy voice
    # often fails YIN's 0.15 voicing threshold, so use a looser one here
    per = speech & (f0 > 60) & (f0 < 600) & (ap < 0.45)
    breath = 10 * torch.log10(ap[per]) if int(per.sum()) >= 10 else torch.tensor([float("nan")])
    h1 = harm & (h.round() == 1)
    high = harm & (freqs[None] >= 1000) & (freqs[None] <= 5000)
    e1 = ((S * h1).sum(1) - noise_density * h1.sum(1)).clamp(min=1e-10)
    eh = ((S * high).sum(1) - noise_density * high.sum(1)).clamp(min=1e-10)
    tension = 10 * torch.log10(eh / e1)
    return {"voiced_frac": round(vf, 3), "breath_db": round(float(breath.median()), 3),
            "tension_db": round(float(tension.median()), 3),
            "f0_st": round(float(12 * torch.log2(f0s.median() / 55.0)), 3), "harmonic_db": round(float(10 * torch.log10(harmonic.median())), 2)}


def assign_relative(items, key, out_key, whisper_value=None):
    """Speaker-relative 0..1 label from a raw dB feature: 0.5 = that speaker's median, 0 / 1 at the
    corpus 5th / 95th percentile of deviations (clipped). Items without the feature get
    `whisper_value` if they are whispers (voiced_frac < 0.1), else 0.5."""
    vals = {}
    for it in items:
        v = (it.get("voice") or {}).get(key)
        if v is not None:
            vals.setdefault(it["spk_name"], []).append(v)
    med = {s: float(np.median(v)) for s, v in vals.items()}
    rel = [(it["voice"][key] - med[it["spk_name"]]) for it in items if (it.get("voice") or {}).get(key) is not None]
    lo, hi = np.percentile(rel, [5, 95])
    for it in items:
        v = (it.get("voice") or {}).get(key)
        if v is None:
            vf = (it.get("voice") or {}).get("voiced_frac", 1.0)
            it[out_key] = whisper_value if (whisper_value is not None and vf < 0.1) else 0.5
            continue
        r = v - med[it["spk_name"]]
        lab = 0.5 + 0.5 * (r / hi if r >= 0 else r / -lo)
        it[out_key] = round(float(np.clip(lab, 0.0, 1.0)), 4)
    return [round(float(lo), 3), round(float(hi), 3)]
