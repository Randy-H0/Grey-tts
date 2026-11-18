"""Speaker-verification embeddings per clip (WavLM-base-plus-sv x-vectors, 512-d): the input of the
zero-shot voice encoder. Writes <dataset>/spkemb.npy (N, 512, fp16, L2-normalized) and
<dataset>/spkemb_ids.json (clip id per row). Resumable per dataset (skips finished ones).

  python extract_spkemb.py data/kvae_norm=data/clips.txt data/kvae_ja=data/clips_ja.txt ...
  SPKEMB_MAX_PER_SPEAKER=20  only that many clips per voice (evenly spread): training uses the fingerprints as a pool of
                             OTHER clips of the same voice to average, and the voice-table init averages them -- a
                             fixed pool of ~20 gives the same variety at a fraction of the work (Emilia: ~60 clips/voice)
  SPKEMB_COMPILE=1           torch.compile the WavLM (VS 2022 env, CC = its cl.exe): 1.9x faster, identical
                             fingerprints (cosine 1.000000 on 300 Emilia clips, 2026-10-20)
The 16 kHz resample + feature extractor run in the audio workers (2026-10-28: in the main process they took 7.5 CPU
cores and capped Emilia at ~21 clips/s while the workers idled); every 25k clips a part file is saved
(<dataset>/spkemb_parts/), so a crash resumes from the last part.
  SPKEMB_BATCH_S=100         clips sorted by length, ~this many seconds of audio per WavLM call, each batch trimmed to
                             its shortest clip (median 11 ms trimmed, cosine >= 0.9996 vs one at a time). One clip per
                             call spent ~90 ms waiting on a GPU sync for ~12 ms of work (2026-10-28: 18 clips/s); 0 = off
  SPKEMB_MAX_S=8             only the first 8 s of a longer clip (owner 2026-10-28, chose it to save ~1.5 h on emi_en2:
                             vs the whole clip on 40 clips of 13-30 s, cosine median 0.993, worst 0.970); 0 = whole
  SPKEMB_PROCS=4             that many processes side by side on slices of the clips (didn't help one-at-a-time: Windows
                             time-slices the GPU between processes); same fingerprints
"""
import os
import json
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from label_clean_gpu import Clips
from preprocess import read_metadata

from dacts.spkemb import MODEL, SpeakerEmbedder  # noqa: F401

PART = 25000
MAX_S = int(os.environ.get("SPKEMB_MAX_S", "0"))


def _as_is(b):
    return b


class Prepped(Clips):
    """Clips -> (id, WavLM input) in the worker processes."""

    def __getitem__(self, i):
        uid, w = super().__getitem__(i)
        if w is None or len(w) <= 4410:
            return uid, None
        if not hasattr(self, "fe"):
            from transformers import AutoFeatureExtractor
            self.fe = AutoFeatureExtractor.from_pretrained(MODEL)
        x = SpeakerEmbedder.prep(self.fe, w)
        if MAX_S:
            x = {k: v[:, : MAX_S * 16000] for k, v in x.items()}
        return uid, x


def main():
    import subprocess
    import torch
    shard = os.environ.get("SPKEMB_SHARD")  # "i/n": a child process doing every n-th clip from i
    procs = 1 if shard else int(os.environ.get("SPKEMB_PROCS", "1"))
    si, sn = map(int, shard.split("/")) if shard else (0, 1)
    emb = None
    cap = int(os.environ.get("SPKEMB_MAX_PER_SPEAKER", "0"))
    for pair in sys.argv[1:]:
        d, lst = pair.split("=")
        d = Path(d)
        if (d / "spkemb.npy").exists():
            print(f"{d.name}: already done", flush=True)
            continue
        frames = {}
        for l in open(d / "manifest.jsonl", encoding="utf-8"):
            it = json.loads(l)
            frames[it["id"]] = it.get("frames", 0)
        want = list(frames)
        rows = {r[0]: r for r in read_metadata(lst, "list", "")}
        todo = [rows[i] for i in want if i in rows]
        if cap:
            by = {}
            for r in todo:
                by.setdefault(r[3], []).append(r)
            todo = [r for rs in by.values() for r in (rs if len(rs) <= cap else [rs[int(k * len(rs) / cap)] for k in range(cap)])]
        if not shard:
            print(f"{d.name}: {len(todo)} clips" + (f" (<= {cap} per voice)" if cap else ""), flush=True)
        parts = d / "spkemb_parts"
        parts.mkdir(exist_ok=True)

        def load_parts():
            out, ids = [], []
            for f in sorted(parts.glob("*.npz")):
                z = np.load(f)
                out += list(z["emb"])
                ids += [str(i) for i in z["ids"]]
            return out, ids
        out, ids = load_parts()
        done, n0 = set(ids), len(ids)
        todo = [r for r in todo if r[0] not in done]
        if n0 and not shard:
            print(f"{d.name}: {n0} from part files, {len(todo)} left", flush=True)
        if procs > 1 and len(todo) > 1000:  # children on slices, then gather their part files here
            kids = [subprocess.Popen([sys.executable, "-u", __file__, pair], env={**os.environ, "SPKEMB_SHARD": f"{i}/{procs}"},
                                     stdout=sys.stdout, stderr=sys.stderr)  # (Windows: a redirected log isn't inherited otherwise)
                    for i in range(procs)]
            codes = [k.wait() for k in kids]
            if any(codes):
                raise SystemExit(f"{d.name}: a fingerprint process failed {codes}")
            out, ids = load_parts()
            todo = []
        todo = todo[si::sn]
        tag = f"s{si}_" if shard else ""
        if todo and emb is None:
            if shard and torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(min(1.0, 3 * 2**30 / torch.cuda.get_device_properties(0).total_memory))
            emb = SpeakerEmbedder()
            if os.environ.get("SPKEMB_COMPILE") == "1":
                emb.model = torch.compile(emb.model, dynamic=True)
        new_out, new_ids = [], []

        def save_part():
            if new_ids:
                k = len(list(parts.glob(f"{tag}*.npz")))
                np.savez(parts / f"{tag}{k:04d}.npz", emb=np.stack(new_out), ids=np.array(new_ids))
                out.extend(new_out)
                ids.extend(new_ids)
                new_out.clear()
                new_ids.clear()

        workers = max(2, 12 // sn)
        budget = float(os.environ.get("SPKEMB_BATCH_S", "100"))
        if budget > 0 and todo:  # batches of similar length, trimmed to the shortest (no padding)
            fr_ = lambda r: min(frames.get(r[0], 0), MAX_S * 50) if MAX_S else frames.get(r[0], 0)  # noqa: E731
            todo.sort(key=fr_)
            batches, cur = [], []
            for j, r in enumerate(todo):
                if cur and (len(cur) + 1) * fr_(r) / 50 > budget:
                    batches.append(cur)
                    cur = []
                cur.append(j)
            batches.append(cur)
            dl = DataLoader(Prepped(todo), batch_sampler=batches, num_workers=workers, prefetch_factor=4, collate_fn=_as_is)
            n, t0 = 0, __import__("time").time()
            for bi, b in enumerate(dl):
                b = [(u, x) for u, x in b if x is not None]
                # (the manifest's length can differ from the loaded clip's: groups within 100 ms, so no clip loses more;
                # one group per batch for almost all, a few small extra calls for the odd ones -- min cosine 0.945 -> ~1)
                b.sort(key=lambda ux: len(ux[1]["input_values"][0]))
                groups, g = [], []
                for ux in b:
                    if g and len(ux[1]["input_values"][0]) - len(g[0][1]["input_values"][0]) > 1600:
                        groups.append(g)
                        g = []
                    g.append(ux)
                if g:
                    groups.append(g)
                for b in groups:
                    xs = [x["input_values"][0] for _, x in b]
                    L = min(len(v) for v in xs)
                    try:
                        E = emb.embed_batch(torch.stack([v[:L] for v in xs])).numpy().astype(np.float16)
                        new_out.extend(E)
                        new_ids.extend(u for u, _ in b)
                    except Exception as e:  # noqa: BLE001  (one at a time then: one odd clip must not stop the run)
                        print(f"batch {bi} one at a time: {e}", flush=True)
                        for u, x in b:
                            try:
                                new_out.append(emb.embed(x).numpy().astype(np.float16))
                                new_ids.append(u)
                            except Exception as e2:  # noqa: BLE001
                                print(f"skip {u}: {e2}", flush=True)
                n_old, n = n, n + len(batches[bi])
                if n // 10000 != n_old // 10000:
                    r = n / (__import__("time").time() - t0)
                    print(f"{d.name}: {n0 + n}/{n0 + len(todo)}  {r:.0f} clips/s  eta {(len(todo) - n) / r / 60:.0f} min", flush=True)
                if len(new_ids) >= PART:
                    save_part()
            todo = []
        for k, (uid, x) in enumerate(DataLoader(Prepped(todo), batch_size=None, num_workers=workers, prefetch_factor=8) if todo else []):
            if x is not None:
                try:
                    new_out.append(emb.embed(x).numpy().astype(np.float16))
                    new_ids.append(uid)
                except Exception as e:  # noqa: BLE001  (one odd clip must not stop a 100k-clip run)
                    print(f"skip {uid}: {e}", flush=True)
            if k % 10000 == 0:
                print(f"{d.name}{' ' + shard if shard else ''}: {k}/{len(todo)}", flush=True)
            if len(new_ids) >= PART:
                save_part()
        save_part()
        if shard:
            continue  # (the parent writes spkemb.npy)
        np.save(d / "spkemb.npy", np.stack(out))
        (d / "spkemb_ids.json").write_text(json.dumps(ids), encoding="utf-8")
        print(f"{d.name}: done ({len(ids)})", flush=True)
        for f in parts.glob("*.npz"):
            f.unlink()
        parts.rmdir()


if __name__ == "__main__":
    main()
