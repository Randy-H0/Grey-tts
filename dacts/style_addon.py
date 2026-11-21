"""(11-19, you: "can people also train talking style loras? like a certain accent ... sure, you can build it") Style
add-ons: a LoRA trained on several speakers who share a way of talking (an accent, a manner). Each speaker had its own
voice row while training, so the adapters hold what they share, not who they are -- and can go on ANY voice of the same
base model (library voices, clones, Grey's own), at a strength from 0 to 1.

File (<user>/styles/<id>.pt, written by app/trainer.finish for a "style" job):
    {"kind": "style", "adapter_of": "<base>.pt", "name", "step", "lora_rank", "lora_alpha", "lora": {"<module>.lora_A": ..,
     "<module>.lora_B": ..}, "speakers": n}

Applied by adding strength * alpha / rank * B @ A to each adapted layer's weight IN PLACE (a compiled decoder reads its
weights as inputs, so nothing recompiles). The original weights of those layers are kept on the CPU and copied back
exactly when the style changes or goes: no drift however often it switches."""
import os
from pathlib import Path

import torch

_CACHE = {}  # path -> (mtime, loaded file)


def style_dir() -> Path:
    base = os.environ.get("GREY_USER") or str(Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Grey Studio")
    return Path(base) / "styles"


def _load(path: Path) -> dict:
    m = path.stat().st_mtime
    c = _CACHE.get(str(path))
    if c is None or c[0] != m:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck.get("kind") != "style":
            raise ValueError(f"{path.name} is not a style add-on")
        _CACHE[str(path)] = c = (m, ck)
    return c[1]


def list_styles() -> list:
    out = []
    d = style_dir()
    for f in sorted(d.glob("*.pt")) if d.is_dir() else []:
        try:
            ck = _load(f)
        except Exception:  # noqa: BLE001  (a broken file: not listed)
            continue
        out.append({"id": f.stem, "name": ck.get("name") or f.stem, "base": ck.get("adapter_of"), "step": ck.get("step"),
                    "speakers": ck.get("speakers")})
    return out


def _modules(model) -> dict:
    """name -> module, with torch.compile's "_orig_mod." taken out of the names"""
    return {n.replace("_orig_mod.", ""): m for n, m in model.named_modules()}


def restore(synth) -> None:
    orig = getattr(synth, "_style_orig", None)
    if orig:
        mods = _modules(synth.model)
        with torch.no_grad():
            for name, w in orig.items():
                W = mods[name].weight
                W.copy_(w.to(W.device, W.dtype))
    synth._style_orig = {}
    synth._style_addon = None


def apply(synth, base_file: str, style_id: str | None, strength: float = 1.0) -> None:
    """Puts style `style_id` on `synth` (a Synthesizer of the model file `base_file`) at `strength` 0..1; None or 0 =
    plain. Cheap when nothing changed (every request calls it)."""
    strength = max(0.0, min(1.5, float(strength if strength is not None else 1.0)))
    want = (style_id, round(strength, 3)) if style_id and strength > 0 else None
    if getattr(synth, "_style_addon", None) == want:
        return
    restore(synth)
    if want is None:
        return
    path = style_dir() / f"{Path(style_id).name}.pt"
    if not path.exists():
        raise ValueError(f"style add-on \"{style_id}\" not found")
    ck = _load(path)
    if ck.get("adapter_of") and Path(base_file).name != ck["adapter_of"]:
        raise ValueError(f"the style \"{ck.get('name') or style_id}\" was trained on {ck['adapter_of']}: pick that model to use it")
    scale = float(ck["lora_alpha"]) / float(ck["lora_rank"]) * strength
    mods = _modules(synth.model)
    orig = {}
    try:
        with torch.no_grad():
            for k, A in ck["lora"].items():
                if not k.endswith(".lora_A"):
                    continue
                name = k[: -len(".lora_A")]
                B = ck["lora"][name + ".lora_B"]
                m = mods.get(name)
                if m is None or not hasattr(m, "weight"):
                    continue
                W = m.weight
                orig[name] = W.detach().to("cpu", copy=True)
                W.add_((B.float() @ A.float() * scale).to(W.device, W.dtype))
    except Exception:
        synth._style_orig = orig
        restore(synth)
        raise
    synth._style_orig = orig
    synth._style_addon = want
