"""Grow a trained model deeper without changing what it does (function-preserving).

New transformer blocks are interleaved with the old ones and start as exact identities:
  decoder blocks: adaLN-zero gates are zero -> block returns its input
  encoder blocks: attention/FFN output projections are zero -> residual adds nothing
The vocabulary is extended (Japanese symbols) and a language embedding is added at zero, so the
grown model produces the same output as the original for English input. Training then continues
from it (fresh optimizer, short warmup).

  python grow.py runs/v1/last.pt runs/v2/init.pt --dec_layers 20 --enc_layers 10 --use_lang
"""
import argparse
import copy

import torch

import dacts  # noqa: F401
from dacts import text as T
from dacts.config import Config
from dacts.model import DacTTS


def interleave_map(n_old, n_new):
    """Old block i -> new position; new blocks spread evenly between the old ones."""
    step = n_new / n_old
    return {i: int(round(i * step)) for i in range(n_old)}


def zero_encoder_block(blk):
    for lin in (blk.attn.out, blk.ffn.c2):
        torch.nn.init.zeros_(lin.weight)
        torch.nn.init.zeros_(lin.bias)


def zero_decoder_block(blk):
    torch.nn.init.zeros_(blk.ada.weight)
    torch.nn.init.zeros_(blk.ada.bias)


def grow_state(old_sd, old_cfg, new_model, new_cfg):
    new_sd = new_model.state_dict()
    out = {}
    enc_map = interleave_map(old_cfg.enc_layers, new_cfg.enc_layers)
    dec_map = interleave_map(old_cfg.dec_layers, new_cfg.dec_layers)
    for k, v in old_sd.items():
        nk = k
        if k.startswith("encoder.blocks."):
            i = int(k.split(".")[2])
            nk = f"encoder.blocks.{enc_map[i]}." + k.split(".", 3)[3]
        elif k.startswith("decoder.blocks."):
            i = int(k.split(".")[2])
            nk = f"decoder.blocks.{dec_map[i]}." + k.split(".", 3)[3]
        if nk not in new_sd:
            continue
        if v.shape[0] < new_sd[nk].shape[0] and v.shape[1:] == new_sd[nk].shape[1:]:  # extended vocab / speakers
            w = new_sd[nk].clone()
            w[: v.shape[0]] = v
            out[nk] = w
            continue
        assert new_sd[nk].shape == v.shape, (k, nk, v.shape, new_sd[nk].shape)
        out[nk] = v
    return out, enc_map, dec_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--dec_layers", type=int, default=20)
    ap.add_argument("--enc_layers", type=int, default=10)
    ap.add_argument("--use_lang", action="store_true")
    ap.add_argument("--n_speakers", type=int, default=None)
    args = ap.parse_args()

    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    old_cfg = Config.from_dict(ck["cfg"])
    if "use_speed" not in ck["cfg"]:
        old_cfg.use_speed = False
    new_cfg = copy.deepcopy(old_cfg)
    new_cfg.dec_layers, new_cfg.enc_layers = args.dec_layers, args.enc_layers
    new_cfg.use_lang = args.use_lang or old_cfg.use_lang
    if args.n_speakers:
        new_cfg.n_speakers = max(args.n_speakers, old_cfg.n_speakers)

    torch.manual_seed(0)
    new_model = DacTTS(new_cfg, T.VOCAB_SIZE)
    # new blocks start as identities; new language embedding starts at zero effect
    enc_map = interleave_map(old_cfg.enc_layers, new_cfg.enc_layers)
    dec_map = interleave_map(old_cfg.dec_layers, new_cfg.dec_layers)
    for j, blk in enumerate(new_model.encoder.blocks):
        if j not in enc_map.values():
            zero_encoder_block(blk)
    for j, blk in enumerate(new_model.decoder.blocks):
        if j not in dec_map.values():
            zero_decoder_block(blk)
    if new_model.global_cond.lang is not None:
        torch.nn.init.zeros_(new_model.global_cond.lang.weight)

    result = {"cfg": new_cfg.to_dict(), "step": 0, "latent": ck["latent"], "speakers": ck.get("speakers", {}),
              "conditioning": ck.get("conditioning", {}), "grown_from": {"path": args.src, "step": ck["step"]}}
    for key in ("model", "ema"):
        sd, _, _ = grow_state(ck[key], old_cfg, new_model, new_cfg)
        full = new_model.state_dict()
        full.update(sd)
        result[key] = {k: v.clone() for k, v in full.items()}

    # verify: grown model == original model on English input (same noise, same durations)
    old_model = DacTTS(old_cfg, ck["model"]["encoder.emb.weight"].shape[0]).eval()
    old_model.load_state_dict(ck["ema"])
    grown = DacTTS(new_cfg, T.VOCAB_SIZE).eval()
    grown.load_state_dict(result["ema"])
    ids = torch.tensor([T.encode("Function preserving growth check.", True, "en")])
    x_len = torch.tensor([ids.shape[1]])
    spk, sty, vol = torch.tensor([3]), torch.tensor([0.6]), torch.tensor([0.4])
    kw = dict(spk=spk, style=sty, volume=vol, steps=4, guidance=2.0, durations=torch.full((1, ids.shape[1]), 2.0))
    if old_cfg.use_speed:
        kw["speed"] = torch.tensor([0.5])
    torch.manual_seed(1)
    a, _ = old_model.synthesize(ids, x_len, **kw)
    torch.manual_seed(1)
    b, _ = grown.synthesize(ids, x_len, lang=torch.tensor([0]) if new_cfg.use_lang else None, **kw)
    diff = (a - b).abs().max().item()
    n_old = sum(p.numel() for p in old_model.parameters()) / 1e6
    n_new = sum(p.numel() for p in grown.parameters()) / 1e6
    print(f"grown {n_old:.1f}M -> {n_new:.1f}M params; max output difference vs original: {diff:.2e}")
    assert diff < 1e-3, "growth was not function-preserving"
    torch.save(result, args.dst)
    print("->", args.dst)


if __name__ == "__main__":
    main()
