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
    y = F.linear(x, self.weight, self.bias)
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
    """The same on a state dict: base weights + `.lora_A/B` -> merged weights, no adapters."""
    out = {k: v for k, v in sd.items() if ".lora_" not in k}
    for k in [k for k in sd if k.endswith(".lora_A")]:
        base = k[: -len(".lora_A")]
        A, B = sd[k].float(), sd[base + ".lora_B"].float()
        w = out[base + ".weight"]
        out[base + ".weight"] = (w.float() + B @ A * scale).to(w.dtype)
    return out
