"""Init checkpoint for v13xl: v12 grown deeper (~2x parameters), German letters, and the new voices -- all
function-preserving, so v13xl starts out sounding exactly like v12 and only gets better from there.

  - depth: decoder 20 -> 44 blocks, text encoder 10 -> 16 (grow.py: new blocks interleaved, starting as exact
    identities -- adaLN-zero gates / zero output projections)
  - text table 165 -> 169: German-only ä ß ö ü start as copies of a, s, ö, ü (dacts.text._DE; German = language 6)
  - voice table: every v12 voice keeps its row; each new voice (Emilia, German sets) starts where the model's own voice
    encoder puts its averaged fingerprint (grow_v12.py). Ids follow load_datasets order, so give the data dirs exactly
    as in the training command (v12's dirs first, new sets after). Without dirs: voices unchanged (a test grow).
  - verified: same output as v12 (same noise, same durations) on an English line.

  python grow_v13xl.py runs/v12/last.pt runs/v13xl/init.pt [data dirs, in training order ...]"""
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dacts import text as T
from dacts.config import Config
from dacts.model import DacTTS
from grow import interleave_map, zero_decoder_block, zero_encoder_block

DEC_LAYERS, ENC_LAYERS = 44, 16
DE_FROM = {"ä": "a", "ß": "s", T._DE_OE: "ö", T._DE_UE: "ü"}


def grow_voices(w, ck, dirs):
    from dacts.data import load_datasets
    # start from v12's table, as training does (--base_speakers): its retired singing voices (jam_*, gts_*) have no
    # data dirs any more but keep their rows, so every other id stays where v12 learned it; new voices are appended
    _, speakers, _, _ = load_datasets(dirs, base_speakers=ck["speakers"])
    for name, i in ck["speakers"].items():  # old voices must keep their rows
        assert speakers.get(name) == i, f"speaker id mismatch for {name}: {speakers.get(name)} vs {i}"
    old = w["global_cond.spk.weight"]
    n = int(math.ceil((len(speakers) + 64) / 1024) * 1024)
    used = sorted(ck["speakers"].values())
    g = torch.Generator().manual_seed(0)
    table = torch.randn(n, old.shape[1], generator=g) * old[used].std(0) + old[used].mean(0)
    table[: old.shape[0]] = old
    zs_in = ck["cfg"].get("zs_in", 512)
    enc = nn.Sequential(nn.LayerNorm(zs_in), nn.Linear(zs_in, 1024), nn.GELU(), nn.Dropout(0.1),
                        nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, old.shape[1]))
    enc.load_state_dict({k[len("global_cond.spk_enc."):]: v for k, v in w.items() if k.startswith("global_cond.spk_enc.")})
    enc.eval()
    new = {s: i for s, i in speakers.items() if s not in ck["speakers"]}
    fp = {}
    for d in map(Path, dirs):
        if not (d / "spkemb.npy").exists():
            continue
        ids = json.loads((d / "spkemb_ids.json").read_text(encoding="utf-8"))
        E = np.load(d / "spkemb.npy", mmap_mode="r")
        spk_of = {}
        for line in open(d / "manifest.jsonl", encoding="utf-8"):
            it = json.loads(line)
            if it.get("spk_name") in new:
                spk_of[it["id"]] = it["spk_name"]
        for r, uid in enumerate(ids):
            s = spk_of.get(uid)
            if s:
                fp.setdefault(s, []).append(np.asarray(E[r], np.float32))
    done = 0
    with torch.no_grad():
        names = [s for s in new if s in fp]
        for k in range(0, len(names), 4096):
            part = names[k:k + 4096]
            x = torch.nn.functional.normalize(torch.tensor(np.stack([np.mean(fp[s], 0) for s in part])), dim=-1)
            for s, row in zip(part, enc(x)):
                table[new[s]] = row
            done += len(part)
    w["global_cond.spk.weight"] = table
    print(f"voices: {len(ck['speakers'])} kept, {len(new)} new ({done} from the voice encoder, {len(new) - done} random), "
          f"table {n} rows", flush=True)
    return speakers, n


def main():
    src, dst, dirs = sys.argv[1], sys.argv[2], sys.argv[3:]
    ck = torch.load(src, map_location="cpu", weights_only=False)
    old_cfg = Config.from_dict(ck["cfg"])
    new_cfg = copy.deepcopy(old_cfg)
    new_cfg.dec_layers, new_cfg.enc_layers = DEC_LAYERS, ENC_LAYERS
    new_cfg.dec_skips, new_cfg.dec_skip_rank = True, 256  # U-Net long skips for the deep decoder (start as no-ops)
    old_vocab = ck["ema"]["encoder.emb.weight"].shape[0]

    torch.manual_seed(0)
    fresh = DacTTS(new_cfg, T.VOCAB_SIZE)
    enc_map = interleave_map(old_cfg.enc_layers, new_cfg.enc_layers)
    dec_map = interleave_map(old_cfg.dec_layers, new_cfg.dec_layers)
    for j, blk in enumerate(fresh.encoder.blocks):
        if j not in enc_map.values():
            zero_encoder_block(blk)
    for j, blk in enumerate(fresh.decoder.blocks):
        if j not in dec_map.values():
            zero_decoder_block(blk)
    base = fresh.state_dict()
    # what the grow adds (new blocks, the skips): training gives these their own learning rate (cfg.new_lr_mult)
    new_cfg.new_params = ",".join([f"decoder.blocks.{j}." for j in range(DEC_LAYERS) if j not in dec_map.values()]
                                  + [f"encoder.blocks.{j}." for j in range(ENC_LAYERS) if j not in enc_map.values()]
                                  + ["decoder.skip_"])

    w = {}
    for k, v in ck["ema"].items():
        nk = k
        if k.startswith(("encoder.blocks.", "decoder.blocks.")):
            part, _, rest = k.split(".", 3)[0], None, k.split(".", 3)
            i = int(rest[2])
            nk = f"{rest[0]}.blocks.{(enc_map if rest[0] == 'encoder' else dec_map)[i]}.{rest[3]}"
        if nk not in base:
            continue
        w[nk] = v.float().clone() if v.is_floating_point() else v.clone()
    for k, v in base.items():  # new blocks (identities) and anything else new
        w.setdefault(k, v.clone())
    # text table: German letters start as their nearest old letters
    emb = w["encoder.emb.weight"]
    if emb.shape[0] < T.VOCAB_SIZE:
        grown = torch.cat([emb, torch.zeros(T.VOCAB_SIZE - emb.shape[0], emb.shape[1])])
        for c, b in DE_FROM.items():
            grown[T._VOCAB[c]] = emb[T._VOCAB[b]]
        w["encoder.emb.weight"] = grown
    speakers, n_spk = ck["speakers"], ck["cfg"]["n_speakers"]
    if dirs:
        speakers, n_spk = grow_voices(w, ck, dirs)
    cfg = {**new_cfg.to_dict(), "n_speakers": n_spk}

    # verify: grown == v12 on English (same noise, same durations); voice table rows of old voices unchanged
    old_model = DacTTS(old_cfg, old_vocab).eval()
    old_model.load_state_dict(ck["ema"], strict=False)
    gcfg = Config.from_dict(cfg)
    grown_model = DacTTS(gcfg, T.VOCAB_SIZE).eval()
    grown_model.load_state_dict(w, strict=False)
    ids = torch.tensor([T.encode("Function preserving growth check.", old_cfg.add_blank, "en")])
    kw = dict(spk=torch.tensor([3]), style=torch.tensor([0.6]), volume=torch.tensor([0.4]), speed=torch.tensor([0.5]),
              lang=torch.tensor([0]), steps=4, guidance=2.0, durations=torch.full((1, ids.shape[1]), 2.0))
    with torch.no_grad():
        torch.manual_seed(1); a, _ = old_model.synthesize(ids, torch.tensor([ids.shape[1]]), **kw)
        torch.manual_seed(1); b, _ = grown_model.synthesize(ids, torch.tensor([ids.shape[1]]), **kw)
    diff = (a - b).abs().max().item()
    n_old = sum(p.numel() for p in old_model.parameters()) / 1e6
    n_new = sum(p.numel() for p in grown_model.parameters()) / 1e6
    print(f"grown {n_old:.1f}M -> {n_new:.1f}M params (decoder {old_cfg.dec_layers} -> {DEC_LAYERS} blocks, encoder "
          f"{old_cfg.enc_layers} -> {ENC_LAYERS}, text {old_vocab} -> {T.VOCAB_SIZE}); max output difference vs v12: {diff:.2e}",
          flush=True)
    assert diff < 1e-3, "growth was not function-preserving"
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": w, "ema": w, "step": 0, "cfg": cfg, "latent": ck["latent"], "speakers": speakers,
                "conditioning": ck.get("conditioning", {}), "speaker_langs": dict(ck.get("speaker_langs", {})),
                "grown_from": f"{src} (step {ck['step']})"}, dst)
    print("->", dst, flush=True)


if __name__ == "__main__":
    main()
