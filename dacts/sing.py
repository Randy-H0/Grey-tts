"""Singing mode for voice-to-voice (you 2026-10-31, after VTV on two sung vocal stems: "pitch isnt great, maybe add a
singing mode (on the site too) which is just a checkbox that uses a pitch method that does well").

One method, three parts (Synthesizer.convert(..., singing=True)):
1. The melody: RMVPE (third_party/rmvpe, the pitch tracker RVC uses; trained on singing) reads the source's F0.
   Grey's own ContourNet is a small speech-trained net: on the sung stems it called 26-60% of the frames voiced where
   RMVPE heard 68-82%, and its pitch was off by 1.5-2 semitones on average (0.7 on speech).
2. The range: the melody moves by whole octaves into the target voice's range (its median F0 from the reference clip,
   or from a sample line of a learned voice), so the tune stays the same tune. The model is conditioned on that
   melody (the contour's pitch channel, semitones re 100 Hz).
3. The exact notes: the model's contour is 50 frames/s and it only roughly follows it, so its output is re-pitched
   with WORLD (pyworld): the output's own spectral envelope + aperiodicity, resynthesised on the intended F0, in the
   frames where the melody is voiced; everything else (consonants, breaths, silence) stays the model's audio.

Measured 2026-10-31 (v14xl-vtv 85k, CPU, two 20 s vocal stems -> your voice; F0 judged by WORLD's Harvest against the
intended melody, voice by Seed-TTS-eval's WavLM-large vs your other recording):
- in tune (within 50 cents, octave-folded; Billie / Dream stems): plain VTV 20% / 12%; melody in the contour 21-24% /
  10% (the 85k model barely follows its contour); + WORLD re-pitch 92% / 86-88% (median error 5-8 cents).
- voice match (Billie stem): plain 0.196 (it sang at her pitch); whole-octave move 0.462; RMVPE melody 0.537; + WORLD
  0.335 -- WORLD's analysis-resynthesis alone costs ~0.12 (0.537 -> 0.420 resynthesised on its own F0).
  The "exact pitch" training (from v14xl-vtv 85k) should make the model hit the notes itself; WORLD stays the guarantee.
- the slider (re-pitch strength s, StoneMask-refined melody; Billie / Dream): in tune 21 / 11% at s=0, 28 / 16% at 0.3,
  36 / 22% at 0.5, 53 / 30% at 0.7, 86 / 71% at 1 (vs the refined melody; the old RMVPE-only melody at 1: 78 / 63%);
  voice match 0.537 / 0.474 at 0, then 0.389 / 0.407, 0.367 / 0.394, 0.339 / 0.379, 0.314 / 0.349: most of the cost is
  WORLD resynthesis itself (~0.12-0.13; FFT size, D4C threshold, keeping the band above 3-5 kHz: all within 0.01).
"""
from pathlib import Path

import numpy as np
import torch

RMVPE_PATH = Path(__file__).resolve().parents[1] / "models" / "rmvpe" / "rmvpe.pt"
FPS = 100  # RMVPE / WORLD frames per second (10 ms)
_TRACKERS = {}


def tracker(device="cpu"):
    k = str(device)
    if k not in _TRACKERS:
        import sys
        root = str(Path(__file__).resolve().parents[1])
        if root not in sys.path:
            sys.path.insert(0, root)
        from third_party.rmvpe.rmvpe import RMVPE
        _TRACKERS[k] = RMVPE(RMVPE_PATH, device)
    return _TRACKERS[k]


def f0_of(w: np.ndarray, sr: int, device="cpu", refine: bool = True) -> np.ndarray:
    """mono audio -> F0 in Hz per 10 ms (0 = unvoiced). refine: WORLD's StoneMask moves each voiced frame onto the
    instantaneous frequency (you 2026-10-31 on the first singing takes: "a bit of an autotune effect"): RMVPE alone keeps
    only ~21 cents of frame-to-frame movement on the Billie stem, refined 37 -- the singer's own wobble, scoops, vibrato."""
    import torchaudio.functional as AF
    x = torch.as_tensor(np.ascontiguousarray(w, dtype=np.float32))
    x16 = AF.resample(x, sr, 16000) if sr != 16000 else x
    f0 = tracker(device)(x16.numpy())
    if not refine:
        return f0
    import pyworld as pw
    r = pw.stonemask(np.ascontiguousarray(w, dtype=np.float64), f0.astype(np.float64), np.arange(len(f0)) / FPS, sr)
    ok = (f0 > 0) & (np.abs(np.log2(np.maximum(r, 1e-3) / np.maximum(f0, 1e-3))) < 0.25)  # (never a jump > 3 st)
    return np.where(ok, r, f0)


def median_hz(f0: np.ndarray) -> float:
    v = f0[f0 > 0]
    return float(np.median(v)) if len(v) else 0.0


def octave_shift(src_f0: np.ndarray, target_hz: float) -> int:
    """Whole octaves that bring the melody's median nearest the target voice's median F0."""
    s = median_hz(src_f0)
    return int(np.round(np.log2(target_hz / s))) if s > 0 and target_hz > 0 else 0


def frame_pitch(f0: np.ndarray, n: int, per: int = 2):
    """10 ms F0 -> per model frame (`per` F0 frames each, 50 frames/s for KVAE): (semitones re 100 Hz, voiced) where
    at least half of the frame is voiced (log-mean over its voiced part)."""
    f = np.zeros(n * per)
    m = min(len(f0), n * per)
    f[:m] = f0[:m]
    f = f.reshape(n, per)
    v = f > 0
    st = np.where(v, 12 * np.log2(np.maximum(f, 1e-3) / 100), 0).sum(1) / np.maximum(v.sum(1), 1)
    return st, v.mean(1) >= 0.5


def impose(y: np.ndarray, sr: int, f0_new: np.ndarray, device="cpu", strength: float = 1.0,
           fade_ms: float = 20.0) -> np.ndarray:
    """Re-pitch `y` toward `f0_new` (Hz per 10 ms, 0 = keep y there) with WORLD: y's envelope (CheapTrick, analysed
    with y's own F0) and aperiodicity (D4C), synthesised on f_y^(1 - strength) * f_new^strength (a blend in log-F0: the
    Singing slider, you 2026-10-31 "maybe allow a little bit of variation? or make singing a slider"), cross-faded into
    y where it's voiced. Below 1, frames y itself left unvoiced stay as they are."""
    import pyworld as pw
    x = np.ascontiguousarray(y, dtype=np.float64)
    t = np.arange(len(x) // (sr // FPS) + 1) / FPS
    n = len(t)

    def fit(f):
        g = np.zeros(n)
        g[: min(n, len(f))] = f[:n]
        return g
    fy = fit(f0_of(y, sr, device)).astype(np.float64)
    fm = fit(f0_new).astype(np.float64)
    s = float(min(1.0, max(0.0, strength)))
    both = (fy > 0) & (fm > 0)
    fn = np.where(both, np.exp((1 - s) * np.log(np.maximum(fy, 1e-3)) + s * np.log(np.maximum(fm, 1e-3))),
                  fm if s >= 1.0 else 0.0)
    sp = pw.cheaptrick(x, fy, t, sr)
    ap = pw.d4c(x, fy, t, sr)
    syn = pw.synthesize(fn, sp, ap, sr, 1000 / FPS)[: len(x)]
    syn = np.pad(syn, (0, len(x) - len(syn)))
    m = (fn > 0).astype(np.float64)
    k = max(1, int(fade_ms / (1000 / FPS)))
    m = np.convolve(m, np.ones(k) / k, "same")  # ~20 ms fades between the resynthesised and the model's audio
    m = np.interp(np.arange(len(x)) / sr, t, m)
    return (m * syn + (1 - m) * x).astype(np.float32)

