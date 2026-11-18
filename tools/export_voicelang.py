"""The language subspace of a model's voice table (the language fix, dacts/infer.py _voice_lang_fix), for any model:
<model>.voicelang.npz next to it. Same recipe as notes_solver/leak/export_subspace.py (v15+), but the languages and voices
come from the checkpoint's own speaker_langs, so a model with new languages (v16+: Ukrainian) gets them too.
  Q (k, 256) orthonormal: the directions between the language means + a logistic regression's;  mu (256,);  mus (L, 256); langs
    .venv/Scripts/python.exe tools/export_voicelang.py models/v16plus.pt
"""
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

PER_LANG = 3000


def main(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    spk, sl, W = ck["speakers"], ck.get("speaker_langs") or {}, ck["model"]["global_cond.spk.weight"]
    langs = sorted({v for v in sl.values() if v})
    rng = np.random.RandomState(0)
    rows, labs = [], []
    for li, lg in enumerate(langs):
        names = [n for n, v in sl.items() if v == lg and n in spk and not n.startswith(("sing_", "mk_", "lib_", "clone:", "design:"))]
        pick = rng.permutation(len(names))[:PER_LANG]
        rows += [spk[names[i]] for i in pick]
        labs += [li] * len(pick)
        print(f"  {lg}: {len(names)} voices, {len(pick)} used")
    X = W[torch.tensor(rows)].float().numpy()
    y = np.array(labs)
    mus = np.stack([X[y == i].mean(0) for i in range(len(langs))])
    mu = mus.mean(0)
    sd = X.std(0) + 1e-6
    clf = LogisticRegression(C=1.0, max_iter=400).fit((X - mu) / sd, y)
    D = np.concatenate([mus - mu, clf.coef_ / sd])
    _, S, Vt = np.linalg.svd(D, full_matrices=False)
    Q = Vt[S > 1e-6 * S[0]].astype(np.float32)
    out = path.with_suffix(".voicelang.npz")
    np.savez(out, Q=Q, mu=mu.astype(np.float32), mus=mus.astype(np.float32), langs=np.array(langs))
    print(f"saved {out}: Q {Q.shape}, languages {langs}; a straight line reads the language off these rows {clf.score((X - mu) / sd, y):.1%} of the time")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
