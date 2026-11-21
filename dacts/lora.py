"""LoRA for Grey fine-tunes (you 2026-11-17: "we might also wanna add lora-ing for people who don't have a lot of compute").

Low-rank adapters on the decoder's linear layers: W x + (B A x) * alpha / rank, with A (rank x in) and B (out x rank), B
starting at zero (so step 0 is exactly the base model). Only the adapters (+ the voice table, for the new voices) train:
no gradients or AdamW state for the ~300M frozen decoder weights, which is where the memory goes.

The adapters are added IN PLACE to the existing nn.Linear modules (extra parameters `lora_A` / `lora_B` + a patched
forward), so every base weight keeps its state_dict name: a base checkpoint loads unchanged (strict=False: the adapters
are "missing" and keep their init), and a fine-tuned checkpoint holds the base weights + `<name>.lora_A/B`.
merge() folds them into the weights (W += scale * B @ A) and removes them: inference costs exactly the same as the base.
"""
import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_TARGETS = ("decoder.blocks.",)  # where the adapters go (prefixes of module names)
SKIP = ("ada",)  # (the per-block AdaLN modulation reads the global condition: leave the voice path to the voice table)


def _forward(self, x):
    w, bias = self.weight, self.bias
    if w.dtype != x.dtype:  # (frozen weights kept in bf16: half_frozen; fp32 inputs, e.g. sampling outside autocast)
        w = w.to(x.dtype)
        bias = None if bias is None else bias.to(x.dtype)
    y = F.linear(x, w, bias)
    if not getattr(self, "lora_on", True):  # (11-18) a switched branch (singing): off = exactly the base layer
        return y
    a = self.lora_A.to(x.dtype)
    b = self.lora_B.to(x.dtype)
    return y + F.linear(F.linear(x, a), b) * self.lora_scale


def top_blocks(model: nn.Module, n: int) -> tuple:
    """Prefixes of the top n decoder blocks (all of them for n <= 0)."""
    total = len(model.decoder.blocks)
    lo = 0 if n <= 0 else max(0, total - n)
    return tuple(f"decoder.blocks.{i}." for i in range(lo, total))


def add(model: nn.Module, rank: int = 16, alpha: float = 16.0, targets=DEFAULT_TARGETS) -> list:
    """Adapters on every nn.Linear whose name starts with one of `targets`. Returns their parameter names."""
    names = []
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear) or not name.startswith(tuple(targets)) or name.rsplit(".", 1)[-1] in SKIP:
            continue
        if hasattr(m, "lora_A"):
            continue
        dev = m.weight.device
        m.lora_A = nn.Parameter(torch.empty(rank, m.in_features, device=dev))
        nn.init.kaiming_uniform_(m.lora_A, a=math.sqrt(5))
        m.lora_B = nn.Parameter(torch.zeros(m.out_features, rank, device=dev))
        m.lora_scale = alpha / rank
        m.forward = types.MethodType(_forward, m)
        names += [f"{name}.lora_A", f"{name}.lora_B"]
    return names


def half_frozen(model: nn.Module, dtype=torch.bfloat16) -> int:
    """(11-18, you: "optimize") The adapted layers' frozen weights stored in bf16: training multiplies them in bf16
    anyway (autocast), so the maths is the same and their memory halves (~0.6 GB on a v16+ decoder). -> weights converted"""
    n = 0
    for _, m in model.named_modules():
        if isinstance(m, nn.Linear) and hasattr(m, "lora_A") and not m.weight.requires_grad:
            m.weight.data = m.weight.data.to(dtype)
            if m.bias is not None and not m.bias.requires_grad:
                m.bias.data = m.bias.data.to(dtype)
            n += m.weight.numel()
    return n


def freeze_for_lora(model: nn.Module, also=("global_cond.spk.weight",)) -> tuple:
    """requires_grad only for the adapters (+ `also`, e.g. the voice table for the new voices). -> (trainable, total) counts"""
    tr = tot = 0
    for n, p in model.named_parameters():
        p.requires_grad_(".lora_" in n or n in also)
        tot += p.numel()
        tr += p.numel() if p.requires_grad else 0
    return tr, tot


def state(model_or_sd) -> dict:
    sd = model_or_sd.state_dict() if isinstance(model_or_sd, nn.Module) else model_or_sd
    return {k: v for k, v in sd.items() if ".lora_" in k}


@torch.no_grad()
def merge(model: nn.Module) -> int:
    """W += scale * B @ A for every adapter, then the adapters are removed (plain nn.Linear again)."""
    n = 0
    for _, m in model.named_modules():
        if isinstance(m, nn.Linear) and hasattr(m, "lora_A"):
            m.weight += (m.lora_B.float() @ m.lora_A.float() * m.lora_scale).to(m.weight.dtype)
            del m.lora_A, m.lora_B
            del m.forward  # (back to nn.Linear.forward)
            n += 1
    return n


def merge_sd(sd: dict, scale: float) -> dict:
    """The same on a state dict: base weights + `.lora_A/B` -> merged weights, no adapters (+ the voice rows folded)."""
    sd = fold_voice_rows(sd)
    out = {k: v for k, v in sd.items() if ".lora_" not in k}
    for k in [k for k in sd if k.endswith(".lora_A")]:
        base = k[: -len(".lora_A")]
        A, B = sd[k].float(), sd[base + ".lora_B"].float()
        w = out[base + ".weight"]
        out[base + ".weight"] = (w.float() + B @ A * scale).to(w.dtype)
    return out


# ---------------------------------------------------------------- the new voices' rows only (11-18)
# (you 11-18: "remove the voice table") A LoRA fine-tune used to train the WHOLE voice table (~466k rows, ~120M weights:
# gradients + AdamW state ~1.5 GB) to fit one or two new voices. Now the table stays frozen and each trained voice gets
# one small row of its own (`voice_delta`), added on lookup. Snapshots fold it back into the table (fold_voice_rows), so
# every file outside training looks exactly as before.
DELTA, IDS = "global_cond.spk.voice_delta", "global_cond.spk.voice_ids"


def _emb_forward(self, idx):
    out = F.embedding(idx, self.weight)
    slot = self.voice_slot[idx.clamp(min=0)]  # (-1: not a trained voice)
    hit = slot >= 0
    if bool(hit.any()):
        add = torch.zeros_like(out)
        add[hit] = self.voice_delta[slot[hit]].to(out.dtype)
        out = out + add
    return out


def voice_rows(model: nn.Module, ids) -> int:
    """Frozen table + a trainable row for each voice id in `ids`. Returns the number of rows."""
    emb = model.global_cond.spk
    ids = sorted({int(i) for i in ids})
    dev = emb.weight.device
    emb.voice_delta = nn.Parameter(torch.zeros(len(ids), emb.weight.shape[1], device=dev))
    emb.register_buffer("voice_ids", torch.tensor(ids, dtype=torch.long, device=dev))
    slot = torch.full((emb.weight.shape[0],), -1, dtype=torch.long, device=dev)
    slot[emb.voice_ids] = torch.arange(len(ids), device=dev)
    emb.register_buffer("voice_slot", slot, persistent=False)
    emb.weight.requires_grad_(False)
    emb.forward = types.MethodType(_emb_forward, emb)
    return len(ids)


def fold_voice_rows(sd: dict) -> dict:
    """A state dict with `voice_delta` -> the plain one: the rows added into the table, the extra keys gone."""
    if DELTA not in sd:
        return sd
    out = {k: v for k, v in sd.items() if k not in (DELTA, IDS)}
    w = out["global_cond.spk.weight"].clone()
    w[sd[IDS].long()] += sd[DELTA].to(w.dtype)
    out["global_cond.spk.weight"] = w
    return out


def set_branch(model: nn.Module, on: bool) -> int:
    """(11-18) Switch every adapter on / off (the singing branch: on only for requests with notes). -> adapters switched"""
    n = 0
    for _, m in model.named_modules():
        if isinstance(m, nn.Linear) and hasattr(m, "lora_A"):
            m.lora_on = bool(on)
            n += 1
    return n
