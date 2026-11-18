"""DNSMOS P.835 (Microsoft DNS Challenge, CC BY 4.0): reference-free speech quality scores 1..5.
  SIG = speech signal quality, BAK = background (5 = no audible noise), OVRL = overall.
Model: third_party/dnsmos/sig_bak_ovr.onnx. Follows the official dnsmos_local.py (9.01 s windows,
1 s hop, short clips tiled, non-personalized polynomial mapping)."""
from pathlib import Path

import numpy as np

MODEL = Path(__file__).resolve().parent.parent / "third_party" / "dnsmos" / "sig_bak_ovr.onnx"
SR, LEN = 16000, int(9.01 * 16000)
_P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
_P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
_P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])


class DNSMOS:
    def __init__(self, threads: int = 2):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(MODEL), so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, w16: np.ndarray) -> dict:
        """w16: 16 kHz mono float array -> {'sig', 'bak', 'ovrl'} (means over windows)."""
        a = np.asarray(w16, dtype=np.float32)
        while len(a) < LEN:
            a = np.concatenate([a, a])
        n_win = max(1, int(np.floor(len(a) / SR) - np.floor(LEN / SR) + 1))
        out = []
        for i in range(n_win):
            seg = a[i * SR: i * SR + LEN]
            if len(seg) < LEN:
                break
            s, b, o = self.sess.run(None, {self.inp: seg[None]})[0][0]
            out.append((_P_SIG(s), _P_BAK(b), _P_OVR(o)))
        s, b, o = np.mean(out, 0)
        return {"sig": float(s), "bak": float(b), "ovrl": float(o)}
