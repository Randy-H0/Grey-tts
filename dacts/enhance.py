"""Enhance clone: refine a zero-shot voice vector on its own reference audio (the whole model stays frozen).

The first attempt (zeroshot.fit) started from the AVERAGE voice and drifted: the fitted voices sounded less
like the person than plain zero-shot. Here:
  * start from the voice encoder's zero-shot vector (the point we want to improve, not replace)
  * an anchor penalty keeps the vector near it (fit what the encoder missed, not the microphone)
  * snapshots along the way; the caller keeps a snapshot only if a speaker verifier prefers it

Training pairs come from the reference itself: 3-12 s stretches between pauses, transcribed by Whisper,
encoded like the training data (Synthesizer.make_prompt).
"""
import time

import torch

from . import text as T
from .model import patchify


@torch.no_grad()
def make_pairs(synth, wav44: torch.Tensor, device: str, max_pairs: int = 12, log=print):
    from .auto_prompt import candidates, transcribe

    pairs = []
    for s, e in candidates(wav44, 3.0, 12.0, max_pairs):
        clip = wav44[int(s * 44100): int(e * 44100)]
        text = transcribe(clip, device)
        if len(text) < 8:
            continue
        p = synth.make_prompt(clip, text, max_seconds=12.0)
        pairs.append({"z": p["latents"].float().cpu(), "text": text, "lang": p["lang"]})
    log(f"  enhance: {len(pairs)} transcribed stretches, {sum(len(p['z']) for p in pairs) / 50:.0f} s")
    return pairs


def _batch(synth, pairs):
    cfg, dev, pp = synth.model.cfg, synth.device, synth.model.cfg.patch
    ids = [T.encode(p["text"], cfg.add_blank, p["lang"]) for p in pairs]
    vocab = synth.model.encoder.emb.num_embeddings
    ids = [[i if i < vocab else T.BLANK for i in s] for s in ids]
    x_len = torch.tensor([len(s) for s in ids], device=dev)
    tokens = torch.zeros(len(ids), int(x_len.max()), dtype=torch.long, device=dev)
    for i, s in enumerate(ids):
        tokens[i, : len(s)] = torch.tensor(s)
    n = [len(p["z"]) // pp * pp for p in pairs]
    y = torch.zeros(len(pairs), max(n), pairs[0]["z"].shape[1], device=dev)
    for i, p in enumerate(pairs):
        y[i, : n[i]] = p["z"][: n[i]].to(dev)
    y_len = torch.tensor([k // pp for k in n], device=dev)
    lang = torch.tensor([T.LANGS.get(p["lang"], 0) for p in pairs], device=dev)
    lm = synth.text_lm([p["text"] for p in pairs], [None] * len(pairs)) if synth.text_lm is not None else None
    return tokens, x_len, patchify(y, pp), y_len, lang, lm


def enhance(synth, slot: int, pairs, steps: int = 200, lr: float = 2e-3, anchor: float = 1.0, batch: int = 3,
            snapshots=(50, 100, 200), log=print):
    """Refine table row `slot` (already holding the zero-shot vector). Returns {step: vector} incl. step 0."""
    model = synth.model
    table = model.global_cond.spk.weight
    v0 = table[slot].detach().float().clone()
    vec = torch.nn.Parameter(v0.clone())
    opt = torch.optim.Adam([vec], lr=lr)
    was = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    hook = model.global_cond.spk.register_forward_hook(
        lambda m, inp, out: torch.where((inp[0] == slot)[:, None], vec[None].to(out.dtype), out))
    snaps, t0, step = {0: v0.cpu()}, time.time(), 0
    batches = [_batch(synth, pairs[i: i + batch]) for i in range(0, len(pairs), batch)]
    scale = float(v0.pow(2).mean())
    try:
        while step < max(snapshots):
            for tokens, x_len, y, y_len, lang, lm in batches:
                spk = torch.full((len(x_len),), slot, device=synth.device)
                L = model(tokens, x_len, y, y_len, spk, None, None, None, lang, lm=lm)
                reg = anchor * (vec - v0).pow(2).mean() / scale
                loss = L["fm"] + L["dur"] + reg
                opt.zero_grad()
                loss.backward()
                opt.step()
                step += 1
                if step in snapshots:
                    snaps[step] = vec.detach().float().cpu().clone()
                    log(f"  enhance step {step}: fm {L['fm'].item():.3f} dur {L['dur'].item():.3f} "
                        f"drift {float((vec - v0).norm() / v0.norm()):.3f} ({time.time() - t0:.0f} s)")
                if step >= max(snapshots):
                    break
    finally:
        hook.remove()
        for p, w in zip(model.parameters(), was):
            p.requires_grad_(w)
        with torch.no_grad():
            table[slot] = v0.to(table)
    return snaps
