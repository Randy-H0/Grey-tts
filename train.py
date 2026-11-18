"""Train text -> DAC latent model.

  python train.py --data data/ljspeech_dac --out runs/lj
  python train.py --data ... --out ... --set dec_layers=6 batch_patches=8000
"""
import argparse
import os

# (2026-11-10) torch's CPU thread pool (Intel OpenMP) spins for 200 ms after every parallel CPU op; one batch-sized copy
# per step kept 13 threads spinning all the time (~7 cores, measured). 0 = they sleep at once (same results)
os.environ.setdefault("KMP_BLOCKTIME", "0")
# return cached GPU memory instead of hoarding it (varying batch shapes fragment the cache)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6")
import copy
import json
import math
import random
import time
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dacts import text as T
from dacts.codec import SAMPLE_RATE, KVAELatent, LatentSpace, frame_rate, latent_space
from dacts.config import Config
from dacts.data import BucketBatchSampler, CodesDataset, PlannedBatchSampler, collate, load_datasets
from dacts.infer import Synthesizer
from dacts.model import DacTTS, patch_contour, patchify

_T0 = time.time()  # startup timestamps (2026-11-07: "training takes 30 minutes to initialize")


def _st(what):
    print(f"STARTUP +{time.time() - _T0:7.1f} s  {what}", flush=True)


def param_groups(model):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if n.startswith("code_head."):  # own optimizer (see main)
            continue
        (no_decay if p.ndim < 2 or "emb" in n or "spk" in n else decay).append((n, p))
    return decay, no_decay


class CpuAverage:
    """The weight average (EMA) kept in system RAM instead of on the GPU (2026-11-10, you: "apply the fixes"): it held
    1.9 GB of GPU memory that the batch can use. Every `every` steps the weights are copied to pinned RAM (queued on
    the GPU, no wait) and a side thread blends them in with the weight those steps add up to (decay ** every): the
    average of the same weights, sampled every `every` steps instead of every step (half-life ~3500 steps).
    EMA_CPU=0 in the environment brings the old on-GPU average back."""

    def __init__(self, ema, model_params, every):
        self.ema, self.every, self.src, self.th, self.err = ema, every, model_params, None, None
        ema.to("cpu")
        self.params = list(ema.parameters())
        self.stage = [torch.empty(p.shape, dtype=p.dtype).pin_memory() for p in self.params]

    def join(self):
        if self.th is not None:
            self.th.join()
            self.th = None
        if self.err is not None:
            err, self.err = self.err, None
            raise RuntimeError(f"weight average update failed: {err!r}")

    def update(self, w):
        import threading
        self.join()
        with torch.no_grad():
            for s, p in zip(self.stage, self.src):
                s.copy_(p, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()

        def work():
            try:
                while not ev.query():  # (sleeping, not spinning: the training thread needs the CPU)
                    time.sleep(0.005)
                with torch.no_grad():
                    for e, s in zip(self.params, self.stage):
                        e.lerp_(s, w)
            except Exception as e:  # noqa: BLE001  (raised in the training thread at the next join)
                self.err = e
        self.th = threading.Thread(target=work, daemon=True)
        self.th.start()

    def to_gpu(self, dev):  # (for the sample set: the Synthesizer runs the averaged weights on the GPU)
        self.join()
        self.ema.to(dev)

    def to_cpu(self):
        self.ema.to("cpu")


def opt_groups(model, cfg):
    """The AdamW parameter groups: decay / no decay (+ the speaker table at spk_lr_mult, + a new voice encoder at 5x)."""
    decay, no_decay = ([p for _, p in group] for group in param_groups(model))
    if getattr(cfg, "lora_rank", 0) > 0:  # (11-17) LoRA: only the adapters (own rate) + the voice table get an optimizer
        named = dict(model.named_parameters())
        ad = [p for n, p in named.items() if ".lora_" in n]
        rest = [p for n, p in named.items() if p.requires_grad and ".lora_" not in n]
        groups = [{"params": ad, "weight_decay": 0.0, "lr": cfg.lr * getattr(cfg, "lora_lr_mult", 20.0)}]
        if rest:
            groups.append({"params": rest, "weight_decay": 0.0, "lr": cfg.lr * cfg.spk_lr_mult})
        return ad + rest, [], groups
    groups = [{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}]
    if cfg.spk_lr_mult != 1.0:  # each voice's row is only updated when that voice is in the batch (and on only
        spk_w = model.global_cond.spk.weight  # part of those rows: zero-shot / prompted rows replace it): faster
        groups[1]["params"] = [p for p in no_decay if p is not spk_w]
        groups.append({"params": [spk_w], "weight_decay": 0.0, "lr": cfg.lr * cfg.spk_lr_mult})
    prefixes = tuple(x for x in getattr(cfg, "new_params", "").split(",") if x)
    if getattr(model, "prompt_lab", None) is not None:  # prompt labels (added to a trained model): new too
        prefixes += ("prompt_lab.",)
    if getattr(model.decoder, "clab_in", None) is not None:  # per-clip reference labels: new too
        prefixes += ("decoder.clab_in.", "encoder.tlab.")
    if getattr(model, "ref_enc", None) is not None:  # v14 reference segment: new too
        prefixes += ("ref_enc.", "decoder.ref_x.")
    if getattr(model, "vtv_exact", None) is not None:  # the exact-pitch mode input (2026-10-31): new too
        prefixes += ("vtv_exact",)
    if prefixes and getattr(cfg, "new_lr_mult", 1.0) != 1.0:  # parameters a grow added: their own learning rate
        names = {id(p): n for n, p in model.named_parameters()}
        for g_ in list(groups):
            new = [p for p in g_["params"] if names[id(p)].startswith(prefixes)]
            if new:
                g_["params"] = [p for p in g_["params"] if not names[id(p)].startswith(prefixes)]
                groups.append({**{k: v for k, v in g_.items() if k != "params"}, "params": new,
                               "lr": g_.get("lr", cfg.lr) * cfg.new_lr_mult})
    rlm = getattr(cfg, "rhythm_lr_mult", 1.0)
    if rlm != 1.0 and getattr(model, "rhythm", None) is not None:  # Swift-RM inside (cfg.swift_rm): its own learning rate
        rp = set(id(p) for p in model.rhythm.parameters())
        for g_ in groups:
            g_["params"] = [p for p in g_["params"] if id(p) not in rp]
        groups.append({"params": list(model.rhythm.parameters()), "weight_decay": 0.01, "lr": cfg.lr * rlm})
    if getattr(model, "voice_enc", None) is not None:  # new module on a fine-tune: its own (5x) learning rate
        ve = set(id(p) for p in model.voice_enc.parameters())
        for g_ in groups:
            g_["params"] = [p for p in g_["params"] if id(p) not in ve]
        groups.append({"params": list(model.voice_enc.parameters()), "weight_decay": 0.01, "lr": cfg.lr * 5})
    llm = getattr(cfg, "lang_lr_mult", 1.0)
    if llm != 1.0 and getattr(model.global_cond, "lang", None) is not None:  # (T1 11-17: the language table's own rate; LAST, so the
        # resume's lr copy by group position still lines up with the checkpoint's groups)
        lp = [model.global_cond.lang.weight]
        if getattr(model.global_cond, "lang_mean", None) is not None:  # (T2s: + the language means, Ukrainian's must move)
            lp.append(model.global_cond.lang_mean.weight)
        ids = set(id(p) for p in lp)
        for g_ in groups:
            g_["params"] = [p for p in g_["params"] if id(p) not in ids]
        groups.append({"params": lp, "weight_decay": 0.0, "lr": cfg.lr * llm})
    return decay, no_decay, groups


def ckpt_param_order(ck):
    """Parameter names in the order the checkpoint's optimizer indexed them: its own model (built from its saved
    config on the meta device, no memory) grouped the same way."""
    old_cfg = Config.from_dict(ck.get("cfg_opt") or ck["cfg"])  # (cfg_opt: a file whose model was changed, e.g. tools/v15swift_init.py)
    with torch.device("meta"):
        old = DacTTS(old_cfg, ck["model"]["encoder.emb.weight"].shape[0])
    name = {id(p): n for n, p in old.named_parameters()}
    return [name[id(p)] for g in opt_groups(old, old_cfg)[2] for p in g["params"]]


def load_optimizer_by_name(opt, ck, model, missing):
    """Restore AdamW state even if parameters were added since the checkpoint (they start fresh), or the
    parameters were regrouped (e.g. the speaker table got its own learning rate): matched by name, in the
    order param_groups() gave the checkpoint's optimizer."""
    n_old = sum(len(g["params"]) for g in ck["opt"]["param_groups"])
    n_new = sum(len(g["params"]) for g in opt.param_groups)
    removed = set(ck["model"]) - set(model.state_dict())
    if not missing and not removed and len(opt.param_groups) == len(ck["opt"]["param_groups"]) and n_old == n_new:
        opt.load_state_dict(ck["opt"])
        return
    # the old optimizer indexed its params in ITS group order: the same groups as this optimizer (e.g. the speaker table
    # in its own group, spk_lr_mult), minus the new params -- not param_groups(model), which keeps the table inside
    # group 1 (that misaligned every param after it: "params, grads, exp_avgs ... must have same dtype")
    param_name = {id(p): n for n, p in model.named_parameters()}
    if removed or len(opt.param_groups) != len(ck["opt"]["param_groups"]):  # parameters gone since the checkpoint (e.g.
        # singing inputs switched off), or regrouped (11-17: a new group moves params to the end): its own order, by name
        old_names = ckpt_param_order(ck)
        assert len(old_names) == n_old, (len(old_names), n_old)
    else:
        old_names = [param_name[id(p)] for g in opt.param_groups for p in g["params"] if param_name[id(p)] not in set(missing)]
    old_state = ck["opt"]["state"]
    old_idx = {name: i for i, name in enumerate(old_names)}
    # rebuild a state dict in the NEW optimizer's layout and let torch's own loader cast / place it (a hand copy
    # left the fused AdamW with mismatched state tensors: "params, grads, exp_avgs ... must have same dtype")
    sd = opt.state_dict()
    new_state, restored, k = {}, 0, 0
    for group in opt.param_groups:
        for p in group["params"]:
            i = old_idx.get(param_name.get(id(p)))
            if i is not None and i in old_state:
                new_state[k] = old_state[i]
                restored += 1
            k += 1
    sd["state"] = new_state
    for g_new, g_old in zip(sd["param_groups"], ck["opt"]["param_groups"]):
        g_new["lr"], g_new["initial_lr"] = g_old["lr"], g_old.get("initial_lr", g_old["lr"])
    opt.load_state_dict(sd)
    print(f"resume: optimizer state restored for {restored} params")


def _low_priority_worker(_):
    """DataLoader workers: below-normal priority (the desktop first; see main)."""
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:  # noqa: BLE001
        pass


def patch_ref(z, mask, p):
    """A reference clip (B, T, 64) + frame mask -> decoder steps (B, T//p, p*64) + step mask: the delivery encoder is
    built at the model's token width and inference feeds it make_prompt's (patched) latents (owner 2026-10-26: patch 2)."""
    if p == 1:
        return z, mask
    t = z.shape[1] // p * p
    return patchify(z[:, :t], p), mask[:, :t].reshape(mask.shape[0], -1, p).any(-1)


def main():
    _st("imports done")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, nargs="+", help="one or more preprocessed dataset dirs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--context", default="data/context.jsonl", help="previous-line context (make_context.py)")
    ap.add_argument("--ckpt_dir", default=None, help="where checkpoints go (default: --out)")
    ap.add_argument("--base_speakers", action="store_true",
                    help="with --init: speaker ids + conditioning from the init checkpoint (no base data needed)")
    ap.add_argument("--init", default=None, help="start from these weights (e.g. grow.py output); fresh optimizer")
    args = ap.parse_args()
    # owner 2026-10-22 ("my entire pc is lagging"): the data workers (the heavy CPU users) run below normal so the
    # desktop comes first. The main process stays normal: it was below normal too, but the profile showed training
    # waits on this one thread launching GPU work (CPU-bound on Windows), not on the GPU -- 2.2 -> ~3 it/s when raised.

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ck_dir = Path(args.ckpt_dir) if args.ckpt_dir else out
    ck_dir.mkdir(parents=True, exist_ok=True)
    ck_path = ck_dir / "last.pt"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False) if args.resume and ck_path.exists() else None
    init = torch.load(args.init, map_location="cpu", weights_only=False) if args.init and not ck else None
    _st("checkpoint file read (torch.load)")
    cfg = Config.from_dict((ck or init)["cfg"]) if (ck or init) else Config()
    cfg.override(args.set)
    # (2026-11-09, you: "allow you to change stuff like this when you pause the run") runs/<run>/SET: one key=value per
    # line (# comments), applied on top of the launcher's --set at EVERY start -- so a change made while the run is
    # paused (a gate check, a RESTART) takes effect at the resume, no launcher edit. tools/v15_gate.py moves
    # runs/<run>/SET.next into SET while training is paused: queue a change there and it lands at the next check.
    set_f = Path(args.out) / "SET"
    if set_f.exists():
        for ln in set_f.read_text(encoding="utf-8-sig").splitlines():  # (-sig: a byte-order mark silently dropped line 1, critic_weight, from 102.1k to 200k; Solver 11-13)
            ln = ln.split("#", 1)[0].strip()
            if not ln:
                continue
            try:
                cfg.override([ln])
                print(f"SET {ln} (runs/.../SET)", flush=True)
            except (KeyError, ValueError) as e:
                print(f"SET line ignored ({e!r}): {ln}", flush=True)
    bo = Path(args.out) / "BATCH_OVERRIDE"  # (2026-11-07) tools/train_watchdog.py lowered the batch after a GPU overflow
    if bo.exists():  # (11-14 Solver review: after SET, as a ceiling -- SET's batch_patches line no longer undoes it)
        try:
            cfg.batch_patches = min(cfg.batch_patches, int(bo.read_text().split()[0]))
            print(f"batch_patches {cfg.batch_patches} (runs/.../BATCH_OVERRIDE ceiling, set by the watchdog after a GPU overflow)", flush=True)
        except (OSError, ValueError, IndexError):
            pass
    if cfg.repa_until and ck is not None and ck["step"] >= cfg.repa_until:
        cfg._repa_off = True  # past repa_until: the loader skips the REPA features entirely

    src0 = ck or init
    use_base = args.base_speakers and src0 is not None
    fast = None
    if getattr(cfg, "fast_data", False):  # prebuilt shared data index (dacts/fastdata.py): seconds, one copy for all workers
        from dacts import fastdata
        fast = fastdata.prepare(args.data, cfg, src0["speakers"] if use_base else None, args.context)
        _st("fast data cache opened")
    if fast is not None:
        items, speakers, stats, conditioning = None, fast.speakers, fast.stats, fast.conditioning
    else:
        items, speakers, stats, conditioning = load_datasets(args.data, src0["speakers"] if use_base else None, slim=True)
    if use_base:  # volume/level maps of the base model, not of the (small) new dataset
        conditioning = dict(src0.get("conditioning", conditioning))
    if getattr(cfg, "exclude_speakers", "") and items is not None:  # removed voices (the fast path filters in its cache)
        gone = set(filter(None, cfg.exclude_speakers.split(",")))
        n0 = len(items)
        items = [i for i in items if i.get("spk_name") not in gone]
        print(f"excluded voices: {n0 - len(items)} clips left out")
    if cfg.only_speakers:  # single-speaker fine-tunes: filter after loading so speaker ids match the base model
        wanted = set(cfg.only_speakers.split(","))
        items = [i for i in items if i.get("spk_name") in wanted]
        print(f"only speakers {sorted(wanted)}: {len(items)} utterances")
    conditioning.setdefault("volume_normal", 0.5)
    # each voice's own language (majority of its clips): lets inference know when a voice speaks a foreign one
    from collections import Counter
    _langs = {}
    for it in items or []:
        _langs.setdefault(it.get("spk_name"), Counter())[it.get("lang", "en")] += 1
    speaker_langs = dict((src0 or {}).get("speaker_langs", {}))
    speaker_langs.update(fast.speaker_langs if fast is not None else {s: c.most_common(1)[0][0] for s, c in _langs.items() if s})
    if cfg.lm_name and Path(args.context).exists() and fast is None:  # previous line (text) of each clip (cache: built in)
        ctx = {}
        for line in open(args.context, encoding="utf-8"):
            c = json.loads(line)
            ctx[c["id"]] = c
        for it in items:
            c = ctx.get(it["id"])
            if c:
                it["prev_text"] = c["prev_text"]
        print(f"context: {sum(1 for i in items if i.get('prev_text'))} of {len(items)} clips have a previous line")
    cfg.n_speakers = max(cfg.n_speakers, len(speakers))
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=1))
    vtv_only_spk = None  # voices whose rows are always voice-to-voice (cfg.vtv_only_prefix: the sing_* singers)
    if getattr(cfg, "vtv_only_prefix", "") and getattr(cfg, "use_vtv", False):
        vtv_only_spk = torch.zeros(max(cfg.n_speakers, max(speakers.values(), default=0) + 1), dtype=torch.bool)
        for name, i in speakers.items():
            vtv_only_spk[i] = str(name).startswith(cfg.vtv_only_prefix)
        print(f"voice-to-voice only: {int(vtv_only_spk.sum())} voices named {cfg.vtv_only_prefix}*")

    if cfg.max_vram_gb:
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        torch.cuda.set_per_process_memory_fraction(min(1.0, cfg.max_vram_gb / total))
        print(f"VRAM cap: {cfg.max_vram_gb:.1f} of {total:.1f} GB")
    if torch.cuda.is_available():
        # 2026-10-23: cuFFT keeps a plan, with its own GPU workspace outside PyTorch's pool (and the cap), for every FFT
        # length it meets. A sample set decodes ~165 clips of different lengths (kvae_to_audio's rfft / irfft): the
        # plans piled up, the card overflowed and training froze right after the set at step 165000. Keep only a few.
        torch.backends.cuda.cufft_plan_cache[0].max_size = 8
    try:  # 2026-10-23: the main process only (it feeds the GPU, one core): above normal, so the desktop, a browser or
        # Remote Desktop don't starve it -- measured 2.6-3.5 -> 4.1-4.3 it/s. The data workers stay below normal.
        import psutil
        psutil.Process().nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
    except Exception:  # noqa: BLE001
        pass
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"

    src_latent = (ck or init or {}).get("latent")  # keep the normalization the weights were trained with
    if src_latent and src_latent.get("codec", "dac") == cfg.codec:
        latent = latent_space(src_latent).to(dev)
    else:  # new run, or switching codec: normalization from this data
        latent = (KVAELatent.from_stats(stats) if cfg.codec == "kvae" else LatentSpace.from_stats(stats, cfg.n_q)).to(dev)
    cross_on = getattr(cfg, "cross_prob", 0) > 0
    if fast is not None:  # everything below, prebuilt (same split, caps, oversampling, filter, pools)
        val_items, train_items, train_ds = [dict(v) for v in fast.val_items], None, fast.train_ds
        print(f"validation: {len(val_items)} clips, held out (cache)")
        print(f"train: {len(train_ds)} utterances, {fast.hours:.2f} h (prebuilt: {fast.dir.name})")
        print("  by language: " + ", ".join(f"{k} {v:.1f} h" for k, v in sorted(fast.by_lang.items())))
        if cross_on and getattr(cfg, "plan_cross", False):
            from dacts.fastdata import FastPlannedBatchSampler
            sampler = FastPlannedBatchSampler(train_ds, cfg.batch_patches, longest_first=getattr(cfg, "longest_first", False))
            _st("batch plan built")
        else:
            raise SystemExit("fast_data needs plan_cross=True (the cache plans prompts with the batches)")
    else:
        random.Random(0).shuffle(items)
        split = str(getattr(cfg, "val_split", "auto") or "auto").strip().lower()
        both = split == "both" or (split == "auto" and len(items) < 20)  # tiny data (e.g. a 15 s clip): train on every clip
        n_val = max(1, round(len(items) * float(split[:-1]) / 100)) if split.endswith("%") else cfg.n_val
        n_val = max(0, min(n_val, len(items) - (0 if both else 1)))  # held out: always leave something to train on
        val_items, train_items = items[:n_val], items[n_val:]
        # make sure the sample set covers every language present
        for lang in sorted({i.get("lang", "en") for i in items} - {i.get("lang", "en") for i in val_items}):
            extra = next((i for i in train_items if i.get("lang", "en") == lang), None)
            if extra is not None and (both or len(train_items) > 1):
                train_items.remove(extra)
                val_items.append(extra)
        if both:
            train_items = list(items)
        print(f"validation: {len(val_items)} clips, {'also trained on' if both else 'held out'} (val_split={split})")
        if getattr(cfg, "corrupt_prob", 0) > 0:  # which clips have a damaged copy (make_corrupt.py): "corrupt:2" oversamples them
            have = {}
            for it in train_items:
                r = it.get("root", "")
                if r not in have:
                    f = Path(r) / "shards" / "latents_corrupt_index.json"
                    have[r] = set(json.loads(f.read_text(encoding="utf-8"))) if f.exists() else set()
                it["has_corrupt"] = it["id"] in have[r]
            print(f"damaged copies for prompt robustness: {sum(i['has_corrupt'] for i in train_items)} clips")
        for spec in filter(None, getattr(cfg, "voice_cap", "").split(",")):  # audiobook-style sets: a few narrators with
            key, cap = spec.split(":")                                        # thousands of clips each must not dominate
            rnd, per, keep = random.Random(0), {}, []
            mine = [i for i in train_items if Path(i.get("root", "")).name == key]
            rnd.shuffle(mine)
            for i in mine:
                per[i.get("spk_name")] = per.get(i.get("spk_name"), 0) + 1
                if per[i.get("spk_name")] <= int(cap):
                    keep.append(i)
            drop = {id(i) for i in mine} - {id(i) for i in keep}
            train_items = [i for i in train_items if id(i) not in drop]
            print(f"voice cap {key}: {len(mine)} -> {len(keep)} clips ({len(per)} voices, <= {cap} each)")
        for spec in filter(None, cfg.oversample.split(",")):  # "ja:2" a language, "a character voice:10" a speaker, "kvae_x:4" a dataset dir,
            key, k = spec.split(":")                           # "expresso*:3" every speaker whose name starts with "expresso"
            hit = ((lambda i: str(i.get("spk_name", "")).startswith(key[:-1])) if key.endswith("*") else
                   (lambda i: (i.get("whisper") or 0) >= 0.5) if key == "whisper" else  # "whisper:3" real whispers
                   (lambda i: i.get("has_corrupt", False)) if key == "corrupt" else
                   (lambda i: key in (i.get("lang", "en"), i.get("spk_name"), Path(i.get("root", "")).name)))
            train_items += [i for i in train_items if hit(i)] * (int(k) - 1)
        train_ds = CodesDataset(args.data[0], train_items, cfg)
        print(f"train: {len(train_ds)} utterances, {sum(i['frames'] for i in train_ds.items) / frame_rate(cfg) / 3600:.2f} h")
        by_lang = {}
        for i in train_ds.items:
            by_lang[i.get("lang", "en")] = by_lang.get(i.get("lang", "en"), 0) + i["frames"] / frame_rate(cfg) / 3600
        print("  by language: " + ", ".join(f"{k} {v:.1f} h" for k, v in sorted(by_lang.items())))
        # cross-clip rows can carry a prompt up to as long as the clip itself (+3 s): budget for the worst case
        cross_on = getattr(cfg, "cross_prob", 0) > 0
        if cross_on and getattr(cfg, "plan_cross", False):  # prompts planned up front: batches packed by real length
            sampler = PlannedBatchSampler(train_ds, cfg.batch_patches, longest_first=getattr(cfg, "longest_first", False))
        else:
            sampler = BucketBatchSampler([2 * i["patches"] + 150 if cross_on else i["patches"] for i in train_ds.items], cfg.batch_patches)
    try:  # the dashboard's "audio processed" = steps / batches per pass x hours per pass (real batches, not the cap)
        (out / "epoch.json").write_text(json.dumps({"hours": fast.hours if fast is not None else
                                                    sum(i["frames"] for i in train_ds.items) / frame_rate(cfg) / 3600,
                                                    "batches": len(sampler), "clips": len(train_ds)}), encoding="utf-8")
    except OSError:
        pass
    replay = os.environ.get("NAN_REPLAY")  # debugging: train only on batches recorded in a nan_batches.jsonl, repeatedly
    if replay:
        recs = [json.loads(l) for l in open(replay, encoding="utf-8")]
        rb = [[(r[0], None if r[3] is None else (tuple(r[3][0]), r[3][1])) for r in x["rows"]] for x in recs if x.get("rows")]
        sampler = [b for _ in range(int(os.environ.get("NAN_REPLAY_TIMES", "20"))) for b in rb]
        print(f"NAN_REPLAY: {len(rb)} recorded batches x {len(sampler) // max(1, len(rb))}")
    _st("data index + sampler ready")
    dl = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate, num_workers=cfg.num_workers,
                    worker_init_fn=_low_priority_worker,
                    persistent_workers=cfg.num_workers > 0, pin_memory=True)

    src = ck or init
    vocab = src["model"]["encoder.emb.weight"].shape[0] if src else T.VOCAB_SIZE  # English-only runs: smaller vocab
    model = DacTTS(cfg, vocab).to(dev)
    _st("model built")
    if getattr(cfg, "grad_ckpt", False):  # (11-17) recompute decoder activations in the backward pass: less memory
        import functools
        import torch.utils.checkpoint as _ckp  # (wraps each decoder block here: model.py stays as the main runs compiled it)

        def _ck(fwd, *a, **k):
            return _ckp.checkpoint(fwd, *a, use_reentrant=False, **k) if torch.is_grad_enabled() else fwd(*a, **k)
        for _b in model.decoder.blocks:
            _b.forward = functools.partial(_ck, _b.forward)
        print("decoder: gradient checkpointing on", flush=True)
    if getattr(cfg, "lora_rank", 0) > 0:  # (11-17) LoRA fine-tune: adapters on the decoder, everything else frozen
        from dacts import lora as LORA
        _ln = LORA.add(model, cfg.lora_rank, cfg.lora_alpha, LORA.top_blocks(model, getattr(cfg, "lora_top_blocks", 0)))
        # (11-18, you: "remove the voice table") only the trained voices' own rows learn, not the whole ~466k-row table
        _own = [x for x in str(cfg.only_speakers or "").split(",") if x] or             sorted({i.get("spk_name") for i in (items or []) if i.get("spk_name")})
        _vids = [speakers[n] for n in _own if n in speakers]
        _rows = LORA.voice_rows(model, _vids) if (_vids and getattr(cfg, "lora_train_voice", True)) else 0
        _tr, _tot = LORA.freeze_for_lora(model, (LORA.DELTA,) if _rows else
                                         (("global_cond.spk.weight",) if getattr(cfg, "lora_train_voice", True) else ()))
        if getattr(cfg, "lora_bf16_base", True) and str(dev).startswith("cuda"):
            print(f"LoRA: {LORA.half_frozen(model) / 1e6:.0f}M frozen weights stored in bf16", flush=True)
        print(f"LoRA rank {cfg.lora_rank}: {len(_ln) // 2} adapted layers, {_rows} voice rows of their own (table frozen), "
              f"{_tr / 1e6:.1f}M of {_tot / 1e6:.1f}M weights train", flush=True)
    text_lm = None
    if cfg.lm_name:  # frozen multilingual LM (not saved in checkpoints)
        from dacts.lm import TextLM
        text_lm = TextLM(cfg.lm_name, device=dev)
        cfg.lm_dim = text_lm.dim
        if cfg.compile:  # 2026-10-30: eager it is ~300 small kernels every step (frozen, fp16: compiled, same maths)
            text_lm.enable_fast()
    # (11-18, you: "remove ... and optimize") no weight average in LoRA runs: a second full copy of the model (+ a CPU
    # thread averaging it) for adapters that don't need it -- the trained weights themselves are saved and sampled
    no_ema = getattr(cfg, "lora_rank", 0) > 0
    ema = model if no_ema else copy.deepcopy(model).eval().requires_grad_(False)
    ema_params, model_params = list(ema.parameters()), list(model.parameters())
    n_enc = sum(p.numel() for p in model.encoder.parameters()) / 1e6
    n_dec = sum(p.numel() for p in model.decoder.parameters()) / 1e6
    print(f"params: encoder {n_enc:.1f}M, decoder {n_dec:.1f}M")

    decay, no_decay, groups = opt_groups(model, cfg)
    main_params = decay + no_decay  # everything but the code picker
    opt = torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.98), fused=True)

    def lr_at(step):
        if getattr(cfg, "lr_drop_at", 0) and step >= cfg.lr_drop_at:
            return cfg.lr_drop * _lr_at(step)
        return _lr_at(step)

    def _lr_at(step):
        if cfg.lr_restart and step >= cfg.lr_restart:  # warm restart: short warmup, then cosine to max_steps
            s = step - cfg.lr_restart
            if s < cfg.warmup:
                return cfg.lr_restart_peak * (s + 1) / cfg.warmup
            p = min(1.0, (s - cfg.warmup) / max(1, cfg.max_steps - cfg.lr_restart - cfg.warmup))
            return 0.1 + (cfg.lr_restart_peak - 0.1) * 0.5 * (1 + math.cos(math.pi * p))
        if step < cfg.warmup:
            return (step + 1) / cfg.warmup
        p = min(1.0, (step - cfg.warmup) / max(1, cfg.max_steps - cfg.warmup))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    head_opt = head_sched = None
    if model.code_head is not None:  # code picker: separate optimizer + clipping, inputs detached
        head_opt = torch.optim.AdamW(model.code_head.parameters(), lr=cfg.code_lr, betas=(0.9, 0.98), weight_decay=0.01)
        head_sched = torch.optim.lr_scheduler.LambdaLR(head_opt, lambda s: min(1.0, (s + 1) / 500))
        print(f"code picker: {sum(p.numel() for p in model.code_head.parameters()) / 1e6:.1f}M params")
    step = 0
    if ck:
        _res = model.load_state_dict(ck["model"], strict=False)
        missing = _res.missing_keys
        if any(k.startswith("decoder.blocks.") for k in list(_res.unexpected_keys) + list(missing)):  # (11-14 Solver review)
            raise SystemExit(f"resume: decoder depth differs from the checkpoint (cfg.dec_layers={cfg.dec_layers}, checkpoint "
                             f"{ck['cfg'].get('dec_layers')}): fix --set / SET")
        ema.load_state_dict(ck["ema"] if ck.get("ema") is not None else ck["model"], strict=False)
        if missing:
            print(f"resume: new parameters start fresh: {sorted({k.rsplit('.', 2)[0] for k in missing})}")
        load_optimizer_by_name(opt, ck, model, missing)
        sched.load_state_dict(ck["sched"])
        sched.base_lrs = [g["initial_lr"] for g in opt.param_groups]  # (the checkpoint may have fewer groups)
        step = ck["step"]
        if head_opt is not None and "head_opt" in ck:
            head_opt.load_state_dict(ck["head_opt"])
            head_sched.load_state_dict(ck["head_sched"])
        print(f"resumed at step {step}")
    if init is not None:
        model.load_state_dict(init["model"], strict=False)  # new modules (e.g. code picker) start fresh
        # (11-18, a user's full fine-tune: KeyError 'ema') the app's downloads are slimmed copies with only "model" (= the
        # averaged weights): start the average from those
        ema.load_state_dict(init["ema"] if init.get("ema") is not None else init["model"], strict=False)
        print(f"initialized from {args.init} (grown from {init.get('grown_from')})")
    # 2026-11-12 (you: "it doesn't need to be that full"): the loaded checkpoint stayed in RAM for the whole run -- a
    # second copy of every weight (+ the optimizer's state on a resume); nothing below reads it
    ck = init = src = src0 = None
    import gc
    gc.collect()
    nan_where = []  # NAN_REPLAY: the first modules whose output was non-finite, per step
    if replay and not cfg.compile:
        def _hook(name):
            def f(mod, inp, out):
                o = out[0] if isinstance(out, (tuple, list)) and out and torch.is_tensor(out[0]) else out
                if torch.is_tensor(o) and o.is_floating_point() and len(nan_where) < 8 and not bool(torch.isfinite(o).all()):
                    ins = [bool(torch.isfinite(x).all()) for x in inp if torch.is_tensor(x) and x.is_floating_point()]
                    nan_where.append(f"{name} ({type(mod).__name__}, inputs finite: {ins})")
            return f
        for name, mod in model.named_modules():
            if name:
                mod.register_forward_hook(_hook(name))
    if getattr(cfg, "flex_attn", False):
        model.decoder.use_flex = True  # (before compiling: Dynamo reads it)
        print("decoder attention: FlexAttention")
    if cfg.compile:
        if os.environ.get("INDUCTOR_CPP_WRAPPER") == "1":  # (2026-11-07) compiled graphs launch their kernels from C++,
            # not one Python call per kernel (py-spy: Triton's Python launcher ~17% of the main thread). Same kernels.
            import torch._inductor.config as icfg
            icfg.cpp_wrapper = True
            print("inductor: cpp_wrapper on", flush=True)
        model.compile_modules()  # in place: state_dict keys unchanged
        # 2026-10-30: Dynamo keeps 8 compiled graphs per function, then runs every new variant EAGER for the rest of the
        # run (v14xl-ts1 / -vc1 .err: TextEncoder.forward hit the limit). Grad on / off, 1-row batches (~19% of batches at
        # batch_patches 2600: Dynamo specialises size 1) and sizes that equal a fixed width when compiling each take one
        import torch._dynamo.config as dcfg
        for k in ("recompile_limit", "cache_size_limit"):
            if hasattr(dcfg, k):
                setattr(dcfg, k, max(getattr(dcfg, k), 64))

    writer = SummaryWriter(out / "tb")
    sim_loss = None
    if getattr(cfg, "sim_weight", 0) > 0:  # speaker-similarity loss (frozen KVAE decoder + WavLM-SV)
        from dacts.sim_loss import SimLoss
        sim_loss = SimLoss(latent, dev, rows=cfg.sim_rows)
    vtv_feat = None
    if getattr(cfg, "use_vtv", False):  # voice-to-voice rows (dacts/vtv.py, v14xl-vtv 2026-10-30): frozen ContentVec + fast codec
        from dacts.vtv import ContentFeatures
        vtv_feat = ContentFeatures(dev)
    vtv_pitch = None
    if vtv_feat is not None and cfg.vtv_exact_prob > 0 and cfg.vtv_rmvpe_weight > 0:  # (exact pitch, 2026-10-31)
        from dacts.vtv import PitchLoss
        vtv_pitch = PitchLoss(latent, vtv_feat)
    _st("checkpoint loaded, side models ready")
    critic = None
    if getattr(cfg, "critic_weight", 0) > 0:  # voice critic in the loop (dacts/voice_critic.py, 2026-10-30)
        from dacts.voice_critic import VoiceCritic
        critic = VoiceCritic(latent, dev, steps=cfg.critic_steps, compile_mode=getattr(cfg, "critic_compile", ""),
                             sv_model=getattr(cfg, "critic_sv", "microsoft/wavlm-base-plus-sv"),
                             asr_model=getattr(cfg, "critic_asr", ""), asr_weight=getattr(cfg, "critic_asr_weight", 0.0),
                             contrast_weight=getattr(cfg, "critic_contrast_weight", 0.0),
                             contrast_m=getattr(cfg, "critic_contrast_m", 0.2), contrast_tau=getattr(cfg, "critic_contrast_tau", 0.05),
                             contrast_bank=getattr(cfg, "critic_contrast_bank", 256),
                             sv2_model=getattr(cfg, "critic_sv2", ""), sv_weight=getattr(cfg, "critic_sv_weight", 1.0),
                             sv2_weight=getattr(cfg, "critic_sv2_weight", 0.0))
        if critic.sv2 is not None and max(1, int(getattr(cfg, "critic_group", 1))) < 2:
            raise SystemExit("critic_sv2 needs critic_group > 1 (it lives in the grouped path)")
        if critic.contrast_w > 0 and max(1, int(getattr(cfg, "critic_group", 1))) < 2:
            raise SystemExit("critic_contrast_weight needs critic_group > 1 (it lives in the grouped path)")
        if critic.asr is not None and max(1, int(getattr(cfg, "critic_group", 1))) < 2:
            raise SystemExit("critic_asr needs critic_group > 1 (the word critic lives in the grouped path)")
        critic.group = max(1, int(getattr(cfg, "critic_group", 1)))
        critic.bf16 = bool(getattr(cfg, "critic_bf16", False))  # (2026-11-09) decode + fingerprint in bf16
        if cfg.critic_fast or getattr(cfg, "critic_compile", ""):  # (the compiled decoder also meets the critic's shapes / grad modes: room for more variants)
            import torch._dynamo.config as dcfg
            for k in ("recompile_limit", "cache_size_limit"):
                if hasattr(dcfg, k):
                    setattr(dcfg, k, max(getattr(dcfg, k), 32))
    synth = None
    if cfg.sample_texts.startswith("@"):  # one text per line in a file
        cfg.sample_texts = "|".join(Path(cfg.sample_texts[1:]).read_text(encoding="utf-8").splitlines())
    if cfg.sample_texts:  # unseen lines, spoken by the first sample speaker: tests beyond the training text
        v0 = val_items[0]
        val_items = val_items + [{"text": t.strip(), "speaker": v0.get("speaker", 0), "lang": T.detect_lang(t)}
                                 for t in cfg.sample_texts.split("|") if t.strip()]
    val_texts = [i["text"] for i in val_items]
    # random sample set (owner 2026-10-18): per language, talked and sung lines drawn from the data, random voices
    # (sometimes none), random tempo / key / genre -- different at every sample step. Pools built once here.
    rand_pool = {}
    composer = None  # composer v3 (models/composer3.pt), loaded at the first sample step when cfg.use_notes
    if getattr(cfg, "sample_random", 0) > 0 and fast is not None:
        rand_pool = dict(fast.rand_pool)  # built with the cache, same draws
        print("random sample pools: " + ", ".join(f"{l}/{m} {len(p)}" for (l, m), p in sorted(rand_pool.items())))
    elif getattr(cfg, "sample_random", 0) > 0:
        rng0 = random.Random(0)
        for it in train_items:
            t = it.get("text") or ""
            if it.get("notext") or not (20 <= len(t) <= 160):
                continue
            mode = "sing" if (it.get("sing") or 0) >= 0.5 else "talk"
            pool = rand_pool.setdefault((it.get("lang", "en"), mode), [])
            if len(pool) < 3000:
                pool.append((t, it.get("spk_name")))
            elif rng0.random() < 0.01:
                pool[rng0.randrange(3000)] = (t, it.get("spk_name"))
        print("random sample pools: " + ", ".join(f"{l}/{m} {len(p)}" for (l, m), p in sorted(rand_pool.items())))
    repa_on = lambda s: not cfg.repa_until or s < cfg.repa_until  # noqa: E731  REPA helps most early
    contour_net = None
    pitch_ref = getattr(cfg, "pitch_weight", 0) > 0 or (getattr(cfg, "use_vtv", False) and cfg.vtv_exact_prob > 0
                                                       and cfg.vtv_pitch_weight > 0)  # (+ exact-pitch VTV rows)
    if getattr(cfg, "use_emph", False) or getattr(cfg, "use_pctx", False) or pitch_ref:
        # frozen latents -> pitch / loudness net: measures word emphasis and the prompt part's prosody on the fly,
        # and scores sung pitch (cfg.pitch_weight)
        from dacts.contour import load_contour
        contour_net = load_contour(cfg.contour_path, dev)
        if str(dev).startswith("cuda"):  # (a batch-length input each step: bucketed for cuDNN, same values; 2026-11-01)
            from dacts.vtv import bucket_convs_
            bucket_convs_(contour_net)
        if pitch_ref:
            object.__setattr__(model, "_contour_ref", contour_net)  # plain attribute: not trained, not saved
    if getattr(cfg, "teacher_ckpt", "") and getattr(cfg, "teacher_weight", 0) > 0:  # frozen teacher decoder (2026-11-13)
        from dacts.model import FlowDecoder
        tck = torch.load(cfg.teacher_ckpt, map_location="cpu", weights_only=False, mmap=True)
        tcfg = Config.from_dict(tck["cfg"])
        tdec = FlowDecoder(tcfg)
        tsd = {k[len("decoder."):]: v.float() for k, v in tck["ema"].items() if k.startswith("decoder.")}
        miss = tdec.load_state_dict(tsd, strict=False)
        if miss.missing_keys:
            raise SystemExit(f"teacher {cfg.teacher_ckpt}: decoder weights missing {miss.missing_keys[:5]}")
        tdec = tdec.to(dev).eval().requires_grad_(False)
        del tck, tsd
        if cfg.compile:
            tdec.compile(dynamic=True)
        object.__setattr__(model, "_teacher_dec", tdec)  # plain attribute: not trained, not saved
        print(f"teacher: {cfg.teacher_ckpt} decoder {tcfg.dec_dim}x{tcfg.dec_layers}, "
              f"{sum(p.numel() for p in tdec.parameters()) / 1e6:.0f}M, weight {cfg.teacher_weight}", flush=True)
    zs_refs = [x.split("=", 1) for x in filter(None, cfg.zs_sample_refs.split(","))]
    zs_texts = cfg.zs_sample_texts
    if zs_texts.startswith("@"):
        zs_texts = "|".join(Path(zs_texts[1:]).read_text(encoding="utf-8").splitlines())
    zs_texts = [t.strip() for t in zs_texts.split("|") if t.strip()] if zs_refs else []

    # 2026-11-01 (you: "keep count of it ... the exact number"): seconds of audio in every batch the model trained on --
    # the clips, and the prompt audio in front of them -- added up across restarts in <out>/heard.json (written at every
    # log step, so steps replayed after a crash count too); tools/heard_total.py adds all runs up
    heard_f, fps = out / "heard.json", frame_rate(cfg)
    try:
        heard = json.loads(heard_f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        heard = {"clips_s": 0.0, "prompts_s": 0.0, "steps": 0, "since_step": step}

    def _plain(sd):  # (11-18) LoRA's own voice rows folded into the table: snapshots look as before
        if getattr(cfg, "lora_rank", 0) > 0:
            from dacts import lora as LORA
            return LORA.fold_voice_rows(sd)
        return sd

    def save(snap=True):  # snap=False: only the resume point (last.pt), no snapshot file (a stop between snapshots)
        if ema_cpu is not None:
            ema_cpu.join()
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step, "cfg": cfg.to_dict(), "latent": latent.state(),
                    "speakers": speakers, "conditioning": conditioning, "speaker_langs": speaker_langs,
                    **({"head_opt": head_opt.state_dict(), "head_sched": head_sched.state_dict()} if head_opt else {})},
                   ck_path)
        if Path("models").is_dir() and not args.ckpt_dir:  # project's models/ folder: what the website loads
            w = {k: (v.half() if v.is_floating_point() else v) for k, v in _plain(ema.state_dict()).items()}
            tmp = Path("models") / f"{out.name}.tmp"
            torch.save({"model": w, "ema": w, "step": step, "cfg": cfg.to_dict(), "latent": latent.state(),
                        "speakers": speakers, "conditioning": conditioning, "speaker_langs": speaker_langs,
                        "source": str(ck_path)}, tmp)
            tmp.replace(Path("models") / f"{out.name}.pt")
        if snap and cfg.keep_ckpts and (not getattr(cfg, "keep_every", 0) or step % cfg.keep_every == 0):  # small snapshot to listen to / promote later (no optimizer, fp16 EMA weights)
            w = {k: (v.half() if v.is_floating_point() else v) for k, v in _plain(ema.state_dict()).items()}
            torch.save({"model": w, "ema": w, "step": step, "cfg": cfg.to_dict(), "latent": latent.state(),
                        "speakers": speakers, "conditioning": conditioning, "speaker_langs": speaker_langs},
                       ck_dir / f"step_{step:06d}.pt")

    model.train()
    acc, t0 = {}, time.time()
    nan_skips = nan_run = 0
    oom_skips = oom_run = 0
    # non-finite steps, found on the GPU (2026-10-22): the CPU reads each step's flag once its copy has landed
    pending_bad, n_ok, alloc_prev = [], 0.0, {}

    def check_bad(wait=False):
        nonlocal_state = check_bad.__dict__
        while pending_bad and (wait or pending_bad[0][1].query()):
            flag, ev, at, dg, lv, g = pending_bad.pop(0)
            ev.synchronize()
            if float(flag) == 0:
                nonlocal_state["run"] = 0
                continue
            nonlocal_state["skips"] = nonlocal_state.get("skips", 0) + 1
            nonlocal_state["run"] = nonlocal_state.get("run", 0) + 1
            if nonlocal_state["skips"] <= 20:
                try:
                    rows = epoch_batches[dg["bi"]] if dg["bi"] < len(epoch_batches) else []
                    c = dg["ctx"]
                    rec = {"step": at, "losses": {k: float(v.float()) for k, v in lv.items()}, "grad_norm": float(g),
                           "rows": [[int(r), train_ds.items[int(r)]["id"], str(train_ds.items[int(r)]["root"]),
                                     None if pl is None else [list(map(int, pl[0])), int(pl[1])]] for r, pl in rows],
                           "x_len": dg["x_len"].tolist(), "y_len": dg["y_len"].tolist(),
                           "spk": dg["spk"].tolist() if dg["spk"] is not None else None,
                           "lang": dg["lang"].tolist() if dg["lang"] is not None else None,
                           "known_len": c["known_len"].tolist() if "known_len" in c else None,
                           "notext": c["notext"].tolist() if "notext" in c else None,
                           "texts": c.get("texts"), "labels": {k: v.tolist() for k, v in c["labels"].items()}}
                    with open(out / "nan_batches.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, default=str) + "\n")
                except Exception as e:  # noqa: BLE001  (diagnosis must never stop training)
                    print(f"(nan batch not saved: {e})")
            print(f"step {at}: non-finite loss / gradient, update skipped on the GPU ({nonlocal_state['skips']} so far, "
                  "see nan_batches.jsonl)", flush=True)
            if nonlocal_state["run"] >= 50:
                raise SystemExit("50 non-finite steps in a row: stopping (see nan_batches.jsonl)")

    # where a step's time goes (2026-10-22): CUDA events between the phases on every PHASE_EVERY-th step
    # (2026-10-30: + "crit", the voice critic, which "opt" used to include -- every timed step is a critic step when
    # critic_every divides 20 -- and "oom", time lost on batches skipped for memory, which "data" used to include)
    PHASE_EVERY, phase_names = 20, ("prep", "fwd", "bwd", "crit", "opt")
    prof, prof_at = None, None
    if (out / "PROFILE").exists():  # a one-off profile, 300 steps after this start
        prof_at = step + 300
        (out / "PROFILE").unlink()
    phase_acc, phase_n, data_ms, t_body = {k: 0.0 for k in phase_names}, 0, 0.0, None
    phase_ev = [torch.cuda.Event(enable_timing=True) for _ in range(6)] if torch.cuda.is_available() else None
    # (2026-11-10) on the timed steps also: the VTV targets' share of "fwd", and the peak memory per part of the step
    # (which part sets the peak decides how far the batch can grow). Host-side counters: no sync
    vtv_ev = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    vtv_ms = 0.0
    mem_pk = {"hold": 0, "vtv": 0, "main": 0, "crit": 0, "opt": 0}

    def _pk(name):
        v = torch.cuda.max_memory_allocated()
        mem_pk[name], mem_pk["hold"] = max(mem_pk[name], v), max(mem_pk["hold"], v)
        torch.cuda.reset_peak_memory_stats()
    oom_ms = 0.0
    def _vtv_make(b):  # (2026-11-07) the VTV rows' targets, built one batch ahead on a second thread (dacts/ahead.py)
        codes_g = b[2].to(dev, non_blocking=True)
        spk_ = b[4]
        vf = vtv_only_spk[spk_.clamp(0, len(vtv_only_spk) - 1)] & (spk_ >= 0) if vtv_only_spk is not None else None
        return codes_g, vtv_feat.batch(codes_g, b[3], cfg.vtv_rows, cfg.vtv_shift, exact=(
            cfg.vtv_exact_prob, cfg.vtv_exact_shift, cfg.vtv_exact_seg, cfg.patch, b[9].get("known_len"))
            if cfg.vtv_exact_prob > 0 else None, force=vf,
            force_exact=cfg.vtv_only_exact if cfg.vtv_only_exact >= 0 else None)
    use_ahead = vtv_feat is not None and torch.cuda.is_available() and os.environ.get("VTV_AHEAD", "0") == "1"  # (off: measured slower 11-07, the GIL)
    if use_ahead:
        from dacts.ahead import Ahead
        print("VTV targets: one batch ahead on a second thread (VTV_AHEAD=1)", flush=True)
    ema_cpu = None
    if torch.cuda.is_available() and int(os.environ.get("EMA_CPU", "50")) > 0 and not no_ema:
        ema_cpu = CpuAverage(ema, model_params, int(os.environ.get("EMA_CPU", "50")))
        torch.cuda.empty_cache()
        print(f"weight average: in system RAM, updated every {ema_cpu.every} steps (EMA_CPU)", flush=True)
    _st("setup done, entering the training loop")
    if torch.cuda.is_available():
        print(f"GPU memory held before the first step: {torch.cuda.memory_allocated() / 2**30:.2f} GB "
              f"(weights, their average, side models; the optimizer's state comes with the first steps)", flush=True)
    _st_first = step
    while step < cfg.max_steps:
        epoch_batches, bi = list(getattr(sampler, "batches", None) or []), -1  # the loader yields them in this order
        import gc as gc_mod  # (the ~1.4M batch lists + ~5.6M row tuples out of the garbage collector's full passes: those
        gc_mod.collect()  # walk everything and can stall a step for seconds; speedups/ZIPPY_REPORT.md item 11)
        gc_mod.freeze()
        for (tokens, x_len, codes, y_len, spk, style, volume, speed, lang, ctx), pre in (
                Ahead(dl, _vtv_make, dev) if use_ahead else ((b_, None) for b_ in dl)):
            bi += 1
            if prof_at is not None and step == prof_at:
                try:
                    from torch.profiler import ProfilerActivity, profile
                    prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
                    prof.__enter__()
                except Exception as e:  # noqa: BLE001
                    print(f"(profile not started: {e})", flush=True)
                    prof, prof_at = None, None
            if prof is not None and step == prof_at + 3:
                try:
                    torch.cuda.synchronize()
                    prof.__exit__(None, None, None)
                    print(f"profile of steps {prof_at}..{prof_at + 2} (CUDA time, top kernels):\n"
                          + prof.key_averages().table(sort_by="cuda_time_total", row_limit=30), flush=True)
                    # (2026-11-01, ZIPPY_REPORT 0: waits for the GPU have no CUDA time -- the CPU table shows them)
                    print("CPU time:\n" + prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30), flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"(profile failed: {e})", flush=True)
                prof, prof_at = None, None
            if t_body is not None:
                data_ms += (time.perf_counter() - t_body) * 1000  # waiting for the loader
            t_step = time.perf_counter()
            timed = phase_ev is not None and step % PHASE_EVERY == 0
            nan_where.clear()
            x_len_cpu, y_len_cpu = x_len, y_len
            # answers the model would otherwise wait for the GPU to give (DacTTS.forward `hints`), from the CPU batch
            nanfree = lambda v: bool(torch.isfinite(v).all()) if torch.is_tensor(v) and v.is_floating_point() else None  # noqa: E731  (inf too)
            hints = {"labels_ok": {k: ok for k, ok in [("style", nanfree(style)), ("volume", nanfree(volume)),
                                                        ("speed", nanfree(speed))]
                                   + [(k, nanfree(v)) for k, v in ctx["labels"].items()] if ok is not None}}
            if "known_len" in ctx:
                hints["cross_any"] = bool((ctx["known_len"] // cfg.patch > 0).any())
            if "ref" in ctx:
                hints["ref_any"] = bool((~torch.isnan(ctx["ref"]).any(-1)).any())
            if "notext" in ctx:
                hints["notext_any"] = bool(ctx["notext"].any())
            nt_cpu = ctx["notext"].bool() if "notext" in ctx else torch.zeros(len(x_len), dtype=torch.bool)
            hints["text_any"] = bool(((x_len > 0) & ~nt_cpu).any())
            if "word_ids" in ctx:
                hints["word_max"] = int(ctx["word_ids"].max().clamp(min=0))
            tokens, x_len, y_len = (t.to(dev, non_blocking=True) for t in (tokens, x_len, y_len))
            codes = pre[0] if isinstance(pre, tuple) else codes.to(dev, non_blocking=True)
            nb = dict(non_blocking=True)  # (pinned by the loader: a blocking copy would wait for all queued GPU work)
            diag = {"bi": bi, "x_len": x_len_cpu, "y_len": y_len_cpu, "spk": spk, "lang": lang, "ctx": ctx}  # (CPU, for nan_batches)
            vtv_force = vtv_only_spk[spk.clamp(0, len(vtv_only_spk) - 1)] & (spk >= 0) if vtv_only_spk is not None else None
            spk = spk.to(dev, **nb) if cfg.n_speakers > 1 else None
            style = style.to(dev, **nb) if cfg.use_style else None
            volume = volume.to(dev, **nb) if cfg.use_volume else None
            speed = speed.to(dev, **nb) if cfg.use_speed else None
            lang = lang.to(dev, **nb) if cfg.use_lang else None
            oom = False
            if timed:
                phase_ev[0].record()
                _pk("hold")
            try:  # 2026-10-22: a heavy batch now and then needs more than the 20 GB cap; skip just that batch
                with torch.no_grad():
                    y = patchify(latent.encode(codes), cfg.patch)
                    if text_lm is None:
                        lmf = None
                    elif "lm_tokens" in ctx:  # tokenized by the loader workers
                        lmf = text_lm.from_tokens(*ctx["lm_tokens"])
                    else:
                        lmf = text_lm(ctx["texts"], ctx["prev_texts"])
                if timed:
                    phase_ev[1].record()
                model._critic_now = critic is not None and step % cfg.critic_every == 0
                model._critic_alt = ctx["alt_tgt"].to(dev, non_blocking=True) if "alt_tgt" in ctx and model._critic_now else None
                if isinstance(pre, BaseException):
                    raise pre  # (the ahead thread's failure, e.g. out of memory: handled here as the inline call's was)
                vtv = pre[1] if pre is not None else vtv_feat.batch(codes, y_len_cpu, cfg.vtv_rows, cfg.vtv_shift, exact=(  # (exact pitch: 2026-10-31)
                    cfg.vtv_exact_prob, cfg.vtv_exact_shift, cfg.vtv_exact_seg, cfg.patch, ctx.get("known_len"))
                    if cfg.vtv_exact_prob > 0 else None, force=vtv_force,  # (sing_* rows: always, cfg.vtv_only_prefix)
                    force_exact=cfg.vtv_only_exact if cfg.vtv_only_exact >= 0 else None) if vtv_feat is not None else None
                if timed:
                    vtv_ev.record()
                    _pk("vtv")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    losses = model(tokens, x_len, y, y_len // cfg.patch, spk, style, volume, speed, lang,
                                   codes=codes if head_opt is not None else None, lm=lmf,
                                   labels={k: v.to(dev, **nb) for k, v in ctx["labels"].items() if getattr(cfg, f"use_{k}", False)
                                           or (k in ("sing", "bpm", "bpb", "key", "genre") and getattr(cfg, "use_music", False))},
                                   repa=ctx["repa"].to(dev, non_blocking=True) if "repa" in ctx and repa_on(step) else None,
                                   ref=ctx["ref"].to(dev, non_blocking=True) if "ref" in ctx else None,
                                   dlv=patch_ref(latent.encode(ctx["dlv"][0].to(dev, **nb)), ctx["dlv"][1].to(dev, **nb), cfg.patch) if "dlv" in ctx else None,
                                   word_ids=ctx["word_ids"].to(dev, **nb) if (contour_net is not None and getattr(cfg, "use_emph", False)) or getattr(cfg, "use_dur_span", False) else None,
                                   contour=patch_contour(contour_net, y, cfg.patch) if contour_net is not None else None,
                                   known_len=(ctx["known_len"].to(dev, **nb) // cfg.patch) if "known_len" in ctx else None,
                                   corrupt=(patchify(latent.encode(ctx["corrupt"][0].to(dev, non_blocking=True)), cfg.patch),
                                            ctx["corrupt"][1].to(dev, **nb)) if "corrupt" in ctx else None,
                                   vre=((patch_ref if getattr(cfg, "use_ref_seg", False) else lambda z, m, p: (z, m))(  # (ref segment: decoder steps)
                                       latent.encode(ctx["vre"][0].to(dev, non_blocking=True)), ctx["vre"][1].to(dev, **nb), cfg.patch)) if "vre" in ctx else None,
                                   vre_use=cfg.vre_prob if step >= getattr(cfg, "vre_warmup", 0) else 0.0,
                                   notext=ctx["notext"].to(dev, **nb) if "notext" in ctx else None,
                                   notes=ctx["notes"].to(dev, non_blocking=True) if "notes" in ctx else None,
                                   prompt_labels={k: v.to(dev, **nb) for k, v in ctx["prompt_labels"].items()} if "prompt_labels" in ctx else None,
                                   prompt_patch_labels=ctx["prompt_patch_labels"].to(dev, **nb) if "prompt_patch_labels" in ctx else None,
                                   prompt_token_labels=ctx["prompt_token_labels"].to(dev, **nb) if "prompt_token_labels" in ctx else None,
                                   hints=hints, vtv=vtv)
                loss = losses["prior"] + losses["dur"] + losses["fm"] + losses.get("code", 0.0)
                if "repa" in losses:
                    loss = loss + cfg.repa_weight * losses["repa"]
                if "spk_repa" in losses:
                    loss = loss + cfg.spk_repa_weight * losses["spk_repa"]
                if "zs" in losses:
                    loss = loss + cfg.zs_weight * losses["zs"]
                if "teach" in losses:  # frozen teacher decoder (cfg.teacher_ckpt, 2026-11-13)
                    loss = loss + cfg.teacher_weight * losses["teach"]
                if "vre" in losses:
                    loss = loss + getattr(cfg, "vre_weight", 1.0) * losses["vre"]
                if "pitch" in losses:
                    loss = loss + cfg.pitch_weight * losses["pitch"]
                if "vtv_pitch" in losses:  # exact-pitch VTV rows follow the contour (dacts/vtv.py, 2026-10-31)
                    loss = loss + cfg.vtv_pitch_weight * losses["vtv_pitch"]
                critic_in = losses.pop("_critic", None)
                sim_in = losses.pop("_sim", None)
                if sim_in is not None and sim_loss is not None:
                    s = sim_loss(*sim_in)
                    if s is not None:
                        losses["sim"] = s
                        loss = loss + cfg.sim_weight * s
                vp_in = losses.pop("_vpitch", None)
                if vp_in is not None and vtv_pitch is not None:  # exact-pitch rows judged by RMVPE (dacts.vtv.PitchLoss)
                    s = vtv_pitch(*vp_in)
                    if s is not None:
                        losses["vtv_rmvpe"] = s
                        loss = loss + cfg.vtv_rmvpe_weight * s

                opt.zero_grad(set_to_none=True)
                if head_opt is not None:
                    head_opt.zero_grad(set_to_none=True)
                if timed:
                    phase_ev[2].record()
                loss.backward()
                if timed:
                    phase_ev[3].record()
                    _pk("main")
                if critic_in is not None:  # the voice critic, after the main step's graph is freed: its gradients add
                    # to this step's (decoder only). It backpropagates itself, one row's fingerprint at a time (all rows'
                    # graphs at once ran out of memory on most critic steps, 2026-10-30): same gradients
                    # (11-14 Solver: the logged gn never saw a critic step) gradient norm before / after the critics' backward
                    _gn = lambda: torch.linalg.vector_norm(torch.stack([n_.float() for n_ in torch._foreach_norm([q.grad for q in main_params if q.grad is not None])]))  # noqa: E731
                    gnf = _gn().detach()
                    n_cr = float(critic_in.get("n_real", 0))  # (11-14 Solver: real, scored rows per critic step)
                    if getattr(cfg, "critic_eval", False):  # (11-14 Solver) no dropout in the take the critic judges: rollout AND
                        model.decoder.eval()  # its backward (checkpointing recomputes the forward there) in eval mode
                    try:
                        r = critic(model.decoder, critic_in, cfg.patch, fast=cfg.critic_fast, weight=cfg.critic_weight)
                    finally:
                        model.decoder.train()
                    critic_in = None
                    if r is not None:
                        losses["critic"] = r[1]  # (logged: mean voice match of the model's own takes, WavLM-SV)
                        losses["gnf"], losses["gnc"] = gnf, _gn().detach()  # (logged: grad norm without / with the critics)
                        losses["crit_n"] = torch.full((), n_cr, device=gnf.device)
                        losses["crit_on"] = torch.ones((), device=gnf.device)  # (11-14 Solver: critic steps that ran, the divisor)
                        if critic.last_cos2 is not None:
                            losses["critic2"] = critic.last_cos2  # (logged: the second verifier's voice match, ERes2NetV2)
                        if critic.last_word is not None:
                            losses["wordcrit"] = critic.last_word  # (logged: KL real-vs-take letters, word critic)
                        if critic.last_contrast is not None:  # (logged: contrast loss, weight on wrong prompts, margin)
                            losses["contrast"], losses["c_wrong"], losses["c_margin"] = critic.last_contrast.unbind()
            except torch.OutOfMemoryError:
                oom = True
            if oom:  # (outside the except: its traceback would keep the failed step's tensors alive)
                y = lmf = loss = losses = None
                opt.zero_grad(set_to_none=True)
                if head_opt is not None:
                    head_opt.zero_grad(set_to_none=True)
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                oom_skips, oom_run = oom_skips + 1, oom_run + 1
                print(f"step {step}: out of GPU memory on a heavy batch, skipped it ({oom_skips} so far)", flush=True)
                if oom_run >= 20:
                    raise SystemExit("20 out-of-memory batches in a row: stopping")
                oom_ms += (time.perf_counter() - t_step) * 1000
                t_body = time.perf_counter()  # (the next loader wait starts now, not at the last good step)
                continue
            oom_run = 0
            if timed:
                phase_ev[4].record()  # (after the critic)
                if getattr(model, "_critic_now", False):
                    _pk("crit")
            gn = torch.nn.utils.clip_grad_norm_(main_params, cfg.grad_clip)
            if head_opt is None:  # 2026-10-22 (speed): checked on the GPU, read a step later (check_bad); never applied
                bad = (~(torch.isfinite(gn) & torch.isfinite(loss.detach()))).float()  # (0-dim, like the optimizer's step counters)
                opt.found_inf, opt.grad_scale = bad, None  # the fused AdamW skips the whole update when bad = 1
                opt.step()
                del opt.found_inf, opt.grad_scale
                opt.zero_grad(set_to_none=True)  # (11-14 Solver: free the gradients now, not after the next forward: ~2 GB)
                sched.step()
                flag = bad.to("cpu", non_blocking=True)
                ev = torch.cuda.Event()
                ev.record()
                pending_bad.append((flag, ev, step, diag, {k: v.detach() for k, v in losses.items() if torch.is_tensor(v) and v.numel() == 1}, gn.detach()))
                check_bad()
            elif not (torch.isfinite(gn) and torch.isfinite(loss)):
                # 2026-10-22: one non-finite step turned every v13xl weight into NaN within 100 steps. Never apply it:
                # skip the update (and the EMA), keep the batch for diagnosis, stop if it keeps happening
                nan_skips, nan_run = nan_skips + 1, nan_run + 1
                opt.zero_grad(set_to_none=True)
                if nan_skips <= 20:
                    try:
                        bad = {k: float(v.detach().float()) for k, v in losses.items() if torch.is_tensor(v) and v.numel() == 1}
                        rows = epoch_batches[bi] if bi < len(epoch_batches) else []
                        rec = {"step": step, "losses": bad, "grad_norm": float(gn),
                               "rows": [[int(r), train_ds.items[int(r)]["id"], str(train_ds.items[int(r)]["root"]),
                                         None if p is None else [list(map(int, p[0])), int(p[1])]] for r, p in rows],
                               "x_len": x_len.tolist(),
                               "y_len": y_len.tolist(), "spk": spk.tolist() if spk is not None else None,
                               "lang": lang.tolist() if lang is not None else None,
                               "known_len": ctx["known_len"].tolist() if "known_len" in ctx else None,
                               "notext": ctx["notext"].tolist() if "notext" in ctx else None,
                               "texts": ctx.get("texts"), "labels": {k: v.tolist() for k, v in ctx["labels"].items()}}
                        with open(out / "nan_batches.jsonl", "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, default=str) + "\n")
                    except Exception as e:  # noqa: BLE001  (diagnosis must never stop training)
                        print(f"(nan batch not saved: {e})")
                print(f"step {step}: non-finite loss / gradient, update skipped ({nan_skips} so far, see nan_batches.jsonl)", flush=True)
                if nan_where:
                    print("  first non-finite outputs: " + " | ".join(nan_where[:4]), flush=True)
                if nan_run >= 50:
                    raise SystemExit("50 non-finite steps in a row: stopping (see nan_batches.jsonl)")
                continue
            else:
                nan_run = 0
                opt.step()
                opt.zero_grad(set_to_none=True)  # (11-14: as above)
                sched.step()
            if head_opt is not None:
                torch.nn.utils.clip_grad_norm_(model.code_head.parameters(), 1.0)
                head_opt.step()
                head_opt.zero_grad(set_to_none=True)  # (11-14: as above)
                head_sched.step()

            known = ctx.get("known_len")  # (CPU tensors: no GPU wait)
            prompt_f = int(known.sum()) if torch.is_tensor(known) else 0
            heard["clips_s"] += (int(y_len_cpu.sum()) - prompt_f) / fps
            heard["prompts_s"] += prompt_f / fps
            heard["steps"] += 1
            step += 1
            if step - _st_first <= 24 or step - _st_first in (50, 100, 200, 500):
                _st(f"step {step} done (critic {'yes' if getattr(model, '_critic_now', False) else 'no'})")

            d = min(cfg.ema, (1 + step) / (10 + step))
            if no_ema:
                pass
            elif ema_cpu is not None:
                if step % ema_cpu.every == 0:
                    ema_cpu.update(1 - d ** ema_cpu.every)
            else:
                with torch.no_grad():
                    torch._foreach_lerp_(ema_params, model_params, 1 - d)
            if timed:
                phase_ev[5].record()
                phase_ev[5].synchronize()
                _pk("opt")
                if vtv_ev.query():  # (recorded this step: not on a batch skipped for memory before it)
                    vtv_ms += phase_ev[1].elapsed_time(vtv_ev)
                for k, name in enumerate(phase_names):
                    phase_acc[name] += phase_ev[k].elapsed_time(phase_ev[k + 1])
                phase_n += 1

            # accumulate on GPU; only sync when logging (a skipped non-finite step adds nothing)
            okf = (1 - bad) if head_opt is None else 1.0
            n_ok = n_ok + okf
            for k, v in losses.items():
                acc[k] = acc.get(k, 0.0) + (torch.where(bad > 0, 0.0, v.detach()) if head_opt is None else v.detach())
            if step % cfg.log_every == 0:
                dt = time.time() - t0
                check_bad(wait=True)
                n_good = max(1.0, float(n_ok))
                # (11-14 Solver) critic values: mean over the critic steps that actually ran (crit_on), not over every slot;
                # crit_ran = the share of critic slots that ran (empty windows / no eligible rows skip the critic)
                CK = ("critic", "critic2", "wordcrit", "contrast", "c_wrong", "c_margin", "gnf", "gnc", "crit_n")
                n_c = float(acc.pop("crit_on", 0.0))
                acc = {k: float(v) / (max(1.0, n_c) if k in CK else n_good) for k, v in acc.items()}
                acc["crit_ran"] = n_c / max(1.0, n_good / cfg.critic_every)
                n_ok = 0.0
                msg = " ".join(f"{k} {v:.4f}" for k, v in acc.items())
                peak = max(mem_pk["hold"], torch.cuda.max_memory_allocated()) / 2**30 if torch.cuda.is_available() else 0.0
                torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
                ph = ""
                if phase_n:
                    ph = f" | ms data {data_ms / cfg.log_every:.0f} " + " ".join(f"{k} {v / phase_n:.0f}" for k, v in phase_acc.items())
                    ph += f" oom {oom_ms / cfg.log_every:.0f} (vtv {vtv_ms / phase_n:.0f} of fwd)"
                    ph += " | GB " + " ".join(f"{k} {mem_pk[k] / 2**30:.1f}" for k in ("vtv", "main", "crit", "opt"))
                vtv_ms = 0.0
                mem_pk = dict.fromkeys(mem_pk, 0)
                ms_ = torch.cuda.memory_stats() if torch.cuda.is_available() else {}  # (host-side counters: no sync)
                al = {k: ms_.get(k, 0) for k in ("num_alloc_retries", "num_device_alloc", "num_device_free")}
                d_al = {k: v - alloc_prev.get(k, 0) for k, v in al.items()}
                alloc_prev.update(al)
                # 2026-10-22: speed fell within long runs (4.5 -> 2.5 it/s) and recovered after restarts: is the
                # allocator, squeezed by the VRAM cap, retrying / freeing and re-requesting memory more and more?
                am = f" | alloc retries {d_al['num_alloc_retries']} mallocs {d_al['num_device_alloc']} frees {d_al['num_device_free']}"
                res = torch.cuda.memory_reserved() / 2**30 if torch.cuda.is_available() else 0.0
                print(f"step {step} | {msg} | gn {gn:.2f} | lr {sched.get_last_lr()[0]:.2e} | {cfg.log_every / dt:.1f} it/s | peak {peak:.1f} GB res {res:.1f}{ph}{am}")
                try:
                    heard_f.write_text(json.dumps({**heard, "step": step}), encoding="utf-8")
                except OSError:
                    pass
                # 2026-10-22: asked to restart (tools/train_watchdog.py, when the GPU spilled into system RAM and steps
                # crawl): save right here, so no steps are lost, and exit 3 -- the launcher starts again in 20 s
                if (out / "RESTART").exists():
                    try:
                        (out / "RESTART").unlink()
                    except OSError:
                        pass
                    save()
                    print(f"step {step}: saved; restarting as asked (RESTART file)", flush=True)
                    writer.flush()
                    os._exit(3)  # (not sys.exit: DataLoader workers can hold a normal exit up)
                phase_acc, phase_n, data_ms, oom_ms = {k: 0.0 for k in phase_names}, 0, 0.0, 0.0
                for k, v in acc.items():
                    writer.add_scalar(f"loss/{k}", v, step)
                writer.add_scalar("train/grad_norm", gn, step)
                acc, t0 = {}, time.time()

            t_body = time.perf_counter()
            if step % cfg.ckpt_every == 0:
                save()
                t_body = None  # (saving and samples aren't the loader's wait)
            if (step % cfg.sample_every == 0 or step >= cfg.max_steps) and val_texts:  # + a final set
                # first hand the training step's cached memory back (2026-10-22: at step 35000 the sample set was
                # loaded on top of ~20 GB of cache; the card overflowed into system RAM and training crawled for
                # half an hour instead of crashing) -- the sample set then fits inside what training already had
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                if synth is None:
                    if ema_cpu is not None:
                        ema_cpu.to_gpu(dev)
                    synth = Synthesizer(ema, latent, device=dev, text_lm=text_lm)
                    if zs_refs:  # the zero-shot clones' voice fingerprint model (WavLM) runs on the CPU: no GPU memory
                        from dacts.spkemb import SpeakerEmbedder
                        synth._spk_embedder = SpeakerEmbedder("cpu")
                sdir = out / "samples" / f"{step:07d}"
                sdir.mkdir(parents=True, exist_ok=True)
                # each val text with its own speaker: both ends of the style spectrum at typical
                # volume, plus whispered and shouted
                vn = conditioning["volume_normal"]
                # one sample per control extreme; everything else at the speaker's normal
                base = {"style": 0.5, "volume": vn, "speed": 0.5}
                combos = [("style0", {"style": 0.0}), ("style1", {"style": 1.0})]
                if cfg.use_whisper:  # whisper is its own control; volume = quiet .. loud
                    combos += [("quiet", {"volume": 0.05}), ("loud", {"volume": 1.0}), ("whisper", {"whisper": 1.0})]
                else:
                    combos += [("whisper", {"volume": 0.05}), ("shout", {"volume": 1.0})]
                if cfg.use_speed:
                    combos += [("slow", {"speed": 0.0}), ("fast", {"speed": 1.0})]
                if cfg.use_mood:
                    combos += [("sad", {"mood": 0.0}), ("happy", {"mood": 1.0})]
                if cfg.use_dominance:
                    combos += [("meek", {"dominance": 0.0}), ("assertive", {"dominance": 1.0})]
                if getattr(cfg, "use_clean", False):
                    base["clean"] = 1.0  # every sample generated clean, plus one deliberately "dirty"
                if getattr(cfg, "use_band", False):
                    base["band"] = 1.0  # always full band (band-limited sources only teach voices / pronunciation)
                    combos += [("dirty", {"clean": 0.0})]
                for i, it in enumerate(val_items if getattr(cfg, "sample_val", True) else []):
                    for name, over in combos:
                        w = synth(it["text"], speaker=it.get("speaker", 0), lang=it.get("lang", "en"),
                                  **{**base, **over})[0]
                        tag = f"{i}_{name}"
                        sf.write(sdir / f"{tag}.wav", w, SAMPLE_RATE)
                        writer.add_audio(f"sample/{tag}", torch.from_numpy(w), step, SAMPLE_RATE)
                if zs_refs and model.global_cond.spk_enc is not None:  # zero-shot clones from reference audio
                    for name, ref in zs_refs:
                        # everything a clone can use: voice encoder (1 min), prompt (one sentence + its text,
                        # <name>_prompt.wav/.txt), delivery encoder, measured delivery (<name>_delivery.json)
                        synth.add_voice(f"zs-{name}", ref)
                        d = Path(ref).parent
                        extra, lab = {}, {}
                        pw, pt = d / f"{name}_prompt.wav", d / f"{name}_prompt.txt"
                        if getattr(cfg, "use_prompt", False) and pw.exists() and pt.exists():
                            extra["prompt"] = synth.make_prompt(str(pw), pt.read_text(encoding="utf-8").strip())
                        if getattr(cfg, "use_dlv", False):
                            extra["dlv"] = synth.make_delivery(ref)
                        if (d / f"{name}_delivery.json").exists():
                            lab = json.loads((d / f"{name}_delivery.json").read_text())
                        for j, t in enumerate(zs_texts):
                            torch.manual_seed(j)
                            w = synth(t, speaker=f"zs-{name}", **{**base, **lab}, **extra)[0]
                            sf.write(sdir / f"{len(val_texts) + j}_zs-{name}.wav", w, SAMPLE_RATE)
                sset = []
                rng = random.Random(step)
                if getattr(cfg, "use_notes", False) and composer is None and Path("models/composer3.pt").exists():
                    from dacts import composer3 as C3_
                    ck3 = torch.load("models/composer3.pt", map_location=dev, weights_only=False)
                    cm_, mm_ = C3_.ChordModel().to(dev), C3_.Composer3().to(dev)
                    cm_.load_state_dict(ck3["chord"]); mm_.load_state_dict(ck3["melody"]); cm_.eval(); mm_.eval()
                    composer = (cm_, mm_)
                if getattr(cfg, "sample_random", 0) > 0 and rand_pool:
                    # owner: N random lines per language, talked and sung, random voices (1 in 5: no voice at all),
                    # random tempo / time signature / key / genre (key and genre sometimes left out) -- new every step
                    from dacts.music import KEYS
                    COMMON = ["pop", "rock", "hiphop", "rnb", "folk", "country", "electronic", "indie", "soul", "jazz",
                              "dance", "alternative"]
                    rng = random.Random(step)
                    for lang in ("en", "es", "ja", "nl", "pl", "zh", "de"):
                        for mode in (("talk", "sing") if getattr(cfg, "use_music", False) else ("talk",)):
                            # no singing data in this language (Dutch, Polish): its speech lines and voices sing
                            pool = rand_pool.get((lang, mode)) or rand_pool.get((lang, "talk")) or []
                            for k in range(min(cfg.sample_random, len(pool))):
                                t = rng.choice(pool)[0]
                                voice = None if rng.random() < 0.2 else rng.choice(pool)[1]
                                e = {"file": f"set-{lang}-{mode}-{k + 1}.wav", "lang": lang, "mode": mode, "text": t, "speaker": voice}
                                if mode == "sing":
                                    e.update(bpm=rng.choice([70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 128, 130, 140, 150, 160, 170]),
                                             beats_per_bar=4,  # owner: 4/4 only, common tempos
                                             key=rng.choice(KEYS) if rng.random() < 0.8 else None,
                                             genre=rng.sample(COMMON, rng.choice([1, 1, 2])) if rng.random() < 0.8 else None)  # owner: common genres
                                sset.append(e)
                elif getattr(cfg, "sample_set", ""):  # fixed lines (configs/sample_set_v12.json), the same every time
                    sset = json.loads(Path(cfg.sample_set).read_text(encoding="utf-8"))
                if sset:
                    for e in sset:
                        kw = dict(base)
                        if e.get("mode") == "sing" and getattr(cfg, "use_music", False):
                            kw.update(sing=True, bpm=e.get("bpm"), key=e.get("key"), genre=e.get("genre"),
                                      beats_per_bar=e.get("beats_per_bar", 4))
                            if getattr(cfg, "use_notes", False) and composer is not None and e.get("bpm"):
                                # sing a composer melody (chords -> notes, in key, on the 8th grid) over the metronome
                                from dacts.composer3 import compose
                                key = e.get("key") or rng.choice(["C major", "G major", "A minor", "D major", "E minor"])
                                from dacts.text import song_words
                                words = song_words(e["text"], e["lang"])  # the voice's own words (CJK: no spaces)
                                from dacts.composer3 import song_plan
                                cbpm, bars, six = song_plan(words, e["bpm"])  # sung pace; slow songs in double time
                                torch.manual_seed(step + len(sset))
                                res = compose(composer[0], composer[1], words, key.split()[0], key.split()[1], cbpm,
                                              e.get("genre"), bars, e["lang"] if e["lang"] in ("en", "es") else None)
                                kw["notes"] = [(on * six, ln * six, midi, w) for on, ln, midi, w in res["notes"]]
                                kw["key"] = key; e["key"] = key; e["chords"] = res["chords"]; e["bars"] = bars * e["bpm"] / cbpm  # song bars (double time: half)
                            kw["clean"] = None  # owner: sung samples without the clean label (it changes singing a lot)
                        torch.manual_seed(0)
                        spk = speakers.get(e["speaker"]) if e.get("speaker") else None  # None: no voice conditioning
                        w = synth(e["text"], speaker=spk, lang=e["lang"], **kw)[0]
                        sf.write(sdir / e["file"], w, SAMPLE_RATE)
                    (sdir / "sample_set.json").write_text(json.dumps(sset, ensure_ascii=False, indent=1), encoding="utf-8")
                (sdir / "texts.txt").write_text("\n".join(val_texts + zs_texts), encoding="utf-8")
                # give the GPU back to training: the Synthesizer (its voice embedder for the zero-shot clones, ~0.4 GB)
                # and the cache its long lines left behind. v13xl (2026-10-22) ran out of memory on the first step after
                # its first sample set at step 5000 (19.3 of 20 GB): the memory test only measured training steps
                synth = None
                if ema_cpu is not None:
                    ema_cpu.to_cpu()
                import gc
                gc.collect()
                torch.backends.cuda.cufft_plan_cache[0].clear()  # (the set's FFT plans: see the cap above)
                torch.cuda.empty_cache()
                model.train()
                t0 = time.time()
            if step >= cfg.max_steps:
                break
            # (11-18, you: "asap allow people to resume training") a STOP file in the run folder (Grey Studio's Stop button):
            # save the resume point and end cleanly -- instead of being killed and losing everything since the last save
            if step % 10 == 0 and (out / "STOP").exists():
                print(f"stop requested at step {step}: saving so training can resume from here", flush=True)
                save(snap=False)
                try:
                    (out / "STOP").unlink()
                except OSError:
                    pass
                return
    save()


if __name__ == "__main__":
    main()
