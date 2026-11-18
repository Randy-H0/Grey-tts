"""Style-exaggeration score: 0 = flat narration ... 1 = full-on acting.

Per utterance we measure prosodic variability (pitch spread, pitch range, intonation movement,
loudness variability). Each feature is z-scored over the whole corpus, averaged, and the result
is percentile-ranked so scores are uniform in [0, 1]. Ranking is global (not per speaker), so a
speaker who only narrates stays near 0.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as AF

FEATURES = ("f0_std", "f0_range", "f0_vel", "energy_std")
_SR = 16000
_HOP = 160            # 10 ms
_W = 512              # integration window
_TAU_MIN = _SR // 500  # 500 Hz
_TAU_MAX = _SR // 60   # 60 Hz


@torch.no_grad()
def yin(x: torch.Tensor, threshold: float = 0.15, return_ap: bool = False):
    """YIN pitch tracker on 16 kHz mono audio. Returns f0 (Hz, per 10 ms frame) and voiced mask."""
    n = _W + _TAU_MAX + 1
    x = F.pad(x, (n // 2, n // 2))
    frames = x.unfold(0, n, _HOP)  # N, n
    nfft = 1 << (n * 2 - 1).bit_length()
    a = torch.fft.rfft(frames[:, :_W], nfft)
    b = torch.fft.rfft(frames, nfft)
    r = torch.fft.irfft(a.conj() * b, nfft)[:, : _TAU_MAX + 1]  # cross-correlation, tau = 0..TAU_MAX
    c = F.pad(frames.pow(2).cumsum(1), (1, 0))
    taus = torch.arange(_TAU_MAX + 1, device=x.device)
    e2 = c[:, taus + _W] - c[:, taus]
    d = (c[:, _W : _W + 1] + e2 - 2 * r).clamp(min=0)
    # cumulative mean normalized difference
    cmnd = d[:, 1:] * taus[1:] / d[:, 1:].cumsum(1).clamp(min=1e-8)
    cmnd = torch.cat([torch.ones_like(cmnd[:, :1]), cmnd], 1)
    seg = cmnd[:, _TAU_MIN:]
    below = seg < threshold
    voiced = below.any(1)
    tau = torch.where(voiced, below.int().argmax(1), seg.argmin(1))
    for _ in range(32):  # descend to the local minimum after the first dip
        nxt = (tau + 1).clamp(max=seg.shape[1] - 1)
        step = seg.gather(1, nxt[:, None])[:, 0] < seg.gather(1, tau[:, None])[:, 0]
        tau = torch.where(step, nxt, tau)
    # parabolic interpolation
    l = seg.gather(1, (tau - 1).clamp(min=0)[:, None])[:, 0]
    m = seg.gather(1, tau[:, None])[:, 0]
    rr = seg.gather(1, (tau + 1).clamp(max=seg.shape[1] - 1)[:, None])[:, 0]
    den = l - 2 * m + rr
    shift = torch.where(den.abs() > 1e-8, 0.5 * (l - rr) / den, torch.zeros_like(den)).clamp(-1, 1)
    f0 = _SR / (tau.float() + _TAU_MIN + shift)
    # also require some energy
    rms_db = 10 * torch.log10(frames[:, :_W].pow(2).mean(1) + 1e-10)
    voiced = voiced & (rms_db > rms_db.max() - 35)
    if return_ap:  # YIN aperiodicity at the chosen period ~ aperiodic share of the frame's power
        return f0, voiced, rms_db, (m - shift * (l - rr) / 4).clamp(1e-4, 1.0)
    return f0, voiced, rms_db


def _median_filter(x: torch.Tensor, k: int = 5) -> torch.Tensor:
    p = k // 2
    return F.pad(x[None, None], (p, p), mode="replicate")[0, 0].unfold(0, k, 1).median(1).values


@torch.no_grad()
def prosody_features(wav: torch.Tensor, sr: int) -> dict:
    """Raw per-utterance features (NaN where pitch is not measurable, e.g. whisper)."""
    x = AF.resample(wav, sr, _SR) if sr != _SR else wav
    f0, voiced, rms_db = yin(x)
    speech = rms_db > rms_db.max() - 40
    feats = {k: float("nan") for k in FEATURES}
    feats["energy_std"] = float(rms_db[speech].std()) if speech.sum() > 10 else float("nan")
    # vocal-effort inputs: active speech level (power mean over frames within 30 dB of the peak)
    # and the fraction of those frames that are voiced (~0 for whispers)
    active = rms_db > rms_db.max() - 30
    feats["level_db"] = float(10 * torch.log10((10 ** (rms_db[active] / 10)).mean()))
    feats["voiced_frac"] = float((voiced & active).sum() / active.sum().clamp(min=1))
    if voiced.sum() >= 20:
        st = 12 * torch.log2(f0 / 55.0)
        st = torch.where(voiced, _median_filter(st), st)  # suppress octave glitches
        v = st[voiced]
        feats["f0_std"] = float(v.std())
        q = torch.quantile(v, torch.tensor([0.05, 0.95], device=v.device))
        feats["f0_range"] = float(q[1] - q[0])
        both = voiced[1:] & voiced[:-1]
        if both.sum() >= 10:
            dv = (st[1:] - st[:-1])[both].abs().clamp(max=3.0)  # semitones per 10 ms
            feats["f0_vel"] = float(dv.mean())
    return feats


def _rank01(x: np.ndarray) -> np.ndarray:
    order = x.argsort(kind="stable")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    return ranks / max(1, len(x) - 1)


AROUSAL_WEIGHT = 1.0  # v14 (owner 2026-10-27, "remove it completely and rely on the other 65%"): Style = Energy only.
# Was 0.65 + 0.35 * (1 - P(neutral)) from emotion2vec, but that "acting" verdict is mostly noise: the first and second
# half of the SAME clip agree at r 0.36-0.46 (energy 0.86), the acted yes/no flips on 1 clip in 3. v13 and older
# models were trained on the 0.65 / 0.35 mix.


def _emo_ok(it) -> bool:
    e = it.get("emotion") or {}
    return e.get("arousal") is not None and (AROUSAL_WEIGHT >= 1.0 or e.get("nonneutral") is not None)


def _raw_style(items):
    arousal = np.array([it["emotion"]["arousal"] for it in items], dtype=np.float64)
    if AROUSAL_WEIGHT >= 1.0:
        return _rank01(arousal)
    nonneutral = np.array([it["emotion"]["nonneutral"] for it in items], dtype=np.float64)
    return AROUSAL_WEIGHT * _rank01(arousal) + (1 - AROUSAL_WEIGHT) * nonneutral


def assign_scores(items: list[dict]) -> None:
    """Adds 'style' in [0, 1] (corpus-wide percentile rank) to each item.

    With learned emotion features ('emotion': arousal): rank(arousal), the Energy (AROUSAL_WEIGHT: v13 and older mixed
    in 0.35 * P(non-neutral)). Otherwise falls back to prosodic variability.
    """
    has = [it for it in items if _emo_ok(it)]
    if has and len(has) < len(items):  # emotion measured on part of the clips (preprocess --emotion_every): score
        # those the usual way, the rest get no style label (training treats it as a dropped label)
        for it, s in zip(has, _rank01(_raw_style(has))):
            it["style"] = round(float(s), 4)
        for it in items:
            if not _emo_ok(it):
                it["style"] = None
        return
    if has:
        raw = _raw_style(items)
    else:
        mat = np.array([[it["prosody"].get(k, math.nan) for k in FEATURES] for it in items], dtype=np.float64)
        mu = np.nanmean(mat, 0)
        sd = np.nanstd(mat, 0) + 1e-8
        raw = np.nanmean((mat - mu) / sd, 1)    # average over available features
        raw = np.where(np.isnan(raw), np.nanmedian(raw), raw)
    for it, s in zip(items, _rank01(raw)):
        it["style"] = round(float(s), 4)


WHISPER_VOICED_FRAC = 0.1  # measured on EARS: whispers 0.00-0.06, all voiced styles >= 0.13
WHISPER_TOP = 0.1          # volume 0 .. 0.1 = whisper, 0.1 .. 1 = voiced speech


def assign_volume(items: list[dict]) -> float:
    """Adds 'volume' in [0, 1]: 0 .. 0.1 whisper, 0.1 .. 1 voiced speech from soft to shouting.

    Level is measured relative to each speaker's median voiced level, so microphone gain and
    natural loudness differences between speakers/corpora cancel. Within each band the mapping
    is linear in dB between robust percentiles and clipped, so a single extreme clip cannot
    squash everyone else. Returns the volume of a speaker's typical (median) voice.
    """
    level = np.array([it["prosody"]["level_db"] for it in items], dtype=np.float64)
    whisper = np.array([it["prosody"]["voiced_frac"] < WHISPER_VOICED_FRAC for it in items])
    spk = np.array([it["speaker"] for it in items])
    rel = np.zeros_like(level)
    for s in np.unique(spk):
        m = spk == s
        ref = np.median(level[m & ~whisper]) if (m & ~whisper).any() else np.median(level[m])
        rel[m] = level[m] - ref

    def band(x, lo_q, hi_q, a, b, mask):
        lo, hi = np.percentile(x[mask], [lo_q, hi_q]) if mask.any() else (0.0, 1.0)
        return a + (b - a) * np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1), (lo, hi)

    voiced_v, (lo, hi) = band(rel, 2, 99.5, WHISPER_TOP, 1.0, ~whisper)
    whisper_v, (wlo, whi) = band(rel, 5, 95, 0.0, WHISPER_TOP, whisper)
    vol = np.where(whisper, whisper_v, voiced_v)
    for it, v, r in zip(items, vol, rel):
        it["volume"] = round(float(v), 4)
        it["prosody"]["level_rel_db"] = round(float(r), 2)
    normal = WHISPER_TOP + (1 - WHISPER_TOP) * float(np.clip((0 - lo) / max(hi - lo, 1e-6), 0, 1))
    print(f"volume: voiced range {lo:+.1f} .. {hi:+.1f} dB re speaker median; whispers {whisper.mean() * 100:.1f}%; "
          f"typical voice = {normal:.2f}")
    return {"volume_normal": round(normal, 3),
            "volume_map_db": {"whisper": [round(float(wlo), 2), round(float(whi), 2)],
                              "voiced": [round(float(lo), 2), round(float(hi), 2)]}}


def assign_speed(items: list[dict]) -> dict:
    """Adds 'speed' in [0, 1]: 0.5 = the speaker's median pace, 0 = slow, 1 = fast.

    Rate = letters per second of clip (pauses count as slowness). Measured relative to each
    speaker's median (log2 ratio), then mapped piecewise-linearly so 0.5 is exactly the median and
    0 / 1 sit at the corpus 5th / 95th percentiles (clipped beyond, so outliers can't stretch it).
    Checked on EARS: 'slow' readings are 0.71x speaker median, 'fast' 1.40x.
    """
    import re as _re
    from .text import normalize

    from .text import japanese_to_kana, _moras

    def units(it):
        if it.get("lang", "en") == "ja":  # moras (kana) per second
            kana = _re.sub(r"[^\u30a1-\u30f6ー]", "", japanese_to_kana(it["text"], accent=False))
            return len(_moras(kana))
        if it.get("lang", "en") == "zh":
            # characters (one syllable each) per second. Counting a-z letters, as below, gave Chinese 0 units: every
            # Chinese set got one constant speed, and Emilia ZH (a few Latin words) got speed = Infinity, which turned
            # every v13xl weight into NaN within 100 steps (2026-10-22). Latin words inside: ~3 letters per syllable
            return len(_re.findall(r"[\u3400-\u9fff\uf900-\ufaff]", it["text"])) + \
                len(_re.sub(r"[^a-z]", "", it["text"].lower())) / 3
        if it.get("lang", "en") == "ko":  # (2026-11-08) Hangul syllables per second, like Chinese characters; Latin /3
            return len(_re.findall(r"[가-힣]", it["text"])) + len(_re.sub(r"[^a-z]", "", it["text"].lower())) / 3
        if it.get("lang", "en") == "fr":  # (2026-11-08) French: its own numbers + letters (accents count as letters)
            from .text import normalize_fr
            try:
                return len(_re.sub(r"[^a-zàâçéèêëîïôûùüÿœæ]", "", normalize_fr(it["text"])))
            except (OverflowError, ValueError):
                return len(_re.sub(r"[^a-zàâçéèêëîïôûùüÿœæ]", "", it["text"].lower()))
        try:
            return len(_re.sub(r"[^a-z]", "", normalize(it["text"])))
        except (OverflowError, ValueError):  # (a number too long to spell: count its letters only)
            return len(_re.sub(r"[^a-z]", "", it["text"].lower()))

    rate = np.array([units(it) / max(1e-3, it["frames"] * 512 / 44100) for it in items])
    spk = np.array([it["speaker"] for it in items])
    rel = np.zeros_like(rate)
    for s in np.unique(spk):
        m = spk == s
        rel[m] = np.log2(np.maximum(rate[m], 1e-3) / np.median(np.maximum(rate[m], 1e-3)))
    lo, hi = np.percentile(rel, [5, 95])
    for it, r in zip(items, rel):
        it["speed"] = round(float(rate_rel_to_speed(r, lo, hi)), 4)
        it["prosody"]["rate_rel_log2"] = round(float(r), 3)
    print(f"speed: 0 = {2 ** lo:.2f}x, 1 = {2 ** hi:.2f}x speaker median pace")
    return {"speed_map_log2": [round(float(lo), 3), round(float(hi), 3)]}


def rate_rel_to_speed(rel_log2: float, lo: float, hi: float) -> float:
    # lo / hi of 0 (a set whose clips all measured the same pace): stay in [0, 1], never divide by zero
    if rel_log2 < 0:
        return 0.5 - 0.5 * (min(1.0, rel_log2 / lo) if lo < 0 else 1.0)
    return 0.5 + 0.5 * (min(1.0, rel_log2 / hi) if hi > 0 else float(rel_log2 > 0))


def volume_to_rel_db(volume: float, vmap: dict) -> float:
    """Inverse of assign_volume's mapping: natural level (dB) relative to the speaker's typical voice."""
    if volume < WHISPER_TOP:
        lo, hi = vmap["whisper"]
        return lo + volume / WHISPER_TOP * (hi - lo)
    lo, hi = vmap["voiced"]
    return lo + (volume - WHISPER_TOP) / (1 - WHISPER_TOP) * (hi - lo)
