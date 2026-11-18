"""A cloned voice's typical delivery, measured from its reference recording.

Zero-shot clones otherwise speak with average delivery: the model's style / mood / dominance controls are
left off, so an energetic showman comes out flat (half his real pitch range). This measures the reference
the same way the training data was labelled (emotion model: arousal / valence / dominance; pitch range)
and places it in the training set's distribution (configs/delivery_reference.json) -> default labels 0..1.
"""
import json
from pathlib import Path

import numpy as np
import torch

REF = Path(__file__).resolve().parent.parent / "configs" / "delivery_reference.json"


def _pct(x, q):
    return float(np.interp(x, q, np.linspace(0, 1, len(q))))


@torch.no_grad()
def estimate(wav: torch.Tensor, sr: int = 44100, device: str = "cuda", scorer=None) -> dict:
    """wav: mono reference (any length). -> {"style", "mood", "dominance"} in 0..1 (0.5 = typical)."""
    from .emotion import EmotionScorer
    from .style import prosody_features

    q = json.loads(REF.read_text(encoding="utf-8"))
    scorer = scorer or EmotionScorer(device)
    n = 6 * sr
    parts = [wav[i:i + n] for i in range(0, max(1, len(wav) - n // 3), n)] or [wav]
    em, pr = [], []
    for p in parts:
        if len(p) < sr * 1.5:
            continue
        em.append(scorer(p.float(), sr))
        f = prosody_features(p.float().to(device), sr)
        if f.get("f0_range") is not None and np.isfinite(f["f0_range"]):
            pr.append(f["f0_range"])
    if not em:
        return {}
    a = float(np.mean([e["arousal"] for e in em]))
    v = float(np.mean([e["valence"] for e in em]))
    d = float(np.mean([e["dominance"] for e in em]))
    style = _pct(a, q["arousal"])
    if pr:  # excitement + how much the pitch moves
        style = 0.5 * style + 0.5 * _pct(float(np.median(pr)), q["f0_range"])
    return {"style": round(style, 3), "mood": round(_pct(v, q["valence"]), 3), "dominance": round(_pct(d, q["dominance"]), 3)}
