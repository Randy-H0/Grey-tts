"""English accent: the label every English voice gets (label_accent.py) and the measurement for clones.

An accent is a probability vector over ACCENTS, from a wav2vec2 English-accent classifier averaged over a
few clips of the voice (VCTK's own accent metadata overrides it where it maps onto these classes). Used as a
conditioning label (cfg.use_accent) and as the target of the latent-space accent judge (cfg.accent_judge).
"""
import numpy as np
import torch
import torchaudio.functional as AF

ACCENTS = ("us", "england", "indian", "australia", "canada")
CLASSIFIER = "dima806/english_accents_classification"
_clf = {}


def classifier(device: str = "cpu"):
    if device not in _clf:
        from transformers import pipeline

        _clf[device] = pipeline("audio-classification", model=CLASSIFIER, device=device)
    return _clf[device]


@torch.no_grad()
def measure(wav, sr: int, device: str = "cpu") -> list:
    """Mono audio -> probabilities in ACCENTS order (classifier on ~10 s windows, averaged)."""
    w = torch.as_tensor(np.asarray(wav, np.float32))
    x = AF.resample(w, sr, 16000).numpy() if sr != 16000 else w.numpy()
    win, out = 16000 * 10, []
    for s in range(0, max(1, len(x) - 16000 * 2), win):  # skip a <2 s tail
        seg = x[s:s + win]
        r = {d["label"]: d["score"] for d in classifier(device)({"raw": seg, "sampling_rate": 16000}, top_k=None)}
        out.append([r.get(a, 0.0) for a in ACCENTS])
    return [round(float(v), 4) for v in np.mean(out, 0)]
