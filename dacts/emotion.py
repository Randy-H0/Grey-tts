"""Learned expressiveness signals for the style score.

- arousal: audeering wav2vec2-large-robust-12-ft-emotion-msp-dim (MSP-Podcast), ~0..1
- nonneutral: 1 - P(neutral) from emotion2vec+ large; catches low-arousal acting (sad, whisper,
  menace) that arousal alone scores like narration.
Both are CC BY-NC(-SA); see the model cards.
"""
import numpy as np
import torch
import torch.nn as nn
import torchaudio.functional as AF

AROUSAL_MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
E2V_MODEL = "emotion2vec/emotion2vec_plus_large"
SR = 16000


class _RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        return self.out_proj(self.dropout(torch.tanh(self.dense(self.dropout(x)))))


def _arousal_model(device):
    from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel

    class EmotionModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = _RegressionHead(config)
            self.post_init()

        def forward(self, x, lengths=None):
            if lengths is None:
                h = self.wav2vec2(x)[0].mean(1)
                return self.classifier(h)  # arousal, dominance, valence
            # batch of padded clips: attention mask + mean over each clip's own frames (= unbatched result)
            am = (torch.arange(x.shape[1], device=x.device)[None] < lengths[:, None]).long()
            hs = self.wav2vec2(x, attention_mask=am)[0]
            n = self.wav2vec2._get_feat_extract_output_lengths(lengths)
            fm = (torch.arange(hs.shape[1], device=x.device)[None] < n[:, None]).to(hs.dtype)
            h = (hs * fm[..., None]).sum(1) / n[:, None].to(hs.dtype)
            return self.classifier(h)

    return EmotionModel.from_pretrained(AROUSAL_MODEL).to(device).eval()


class EmotionScorer:
    def __init__(self, device="cuda", acting=True):
        """acting=False: no emotion2vec (v14's Style is Energy only -- 'nonneutral' stays None; it was ~65% of the
        emotion labelling time)."""
        self.device = device
        self.arousal = _arousal_model(device)
        self.e2v = None
        if acting:
            try:  # optional (heavy, FunASR): without it 'nonneutral' is None and style uses arousal only
                from funasr import AutoModel
                self.e2v = AutoModel(model=E2V_MODEL, hub="hf", disable_update=True, device=device)
            except ImportError:
                pass
        self._neutral_idx = None

    @torch.no_grad()
    def batch(self, wavs, sr: int) -> list:
        """Many clips at once (same results as calling one by one, several times faster on a GPU)."""
        xs = [w.float().to(self.device) for w in wavs]  # resample on the GPU (CPU resampling dominated)
        xs = [AF.resample(x, sr, SR) if sr != SR else x for x in xs]
        lens = torch.tensor([len(x) for x in xs], device=self.device)
        pad = torch.zeros(len(xs), int(lens.max()), device=self.device)
        for i, x in enumerate(xs):
            x = x.to(self.device)
            pad[i, : len(x)] = (x - x.mean()) / (x.std() + 1e-7)
        with torch.autocast("cuda", dtype=torch.float16, enabled=str(self.device).startswith("cuda")):
            adv = self.arousal(pad, lens)
        adv = adv.float().tolist()
        nonneutral = [None] * len(xs)
        if self.e2v is not None:
            res = self.e2v.generate([x.cpu().numpy() for x in xs], granularity="utterance", extract_embedding=False,
                                    disable_pbar=True, batch_size=len(xs))
            if self._neutral_idx is None:
                self._neutral_idx = next(i for i, l in enumerate(res[0]["labels"]) if "neutral" in l)
            nonneutral = [float(1 - np.asarray(r["scores"])[self._neutral_idx] / np.asarray(r["scores"]).sum()) for r in res]
        return [{"arousal": a, "dominance": d, "valence": v, "nonneutral": nn_} for (a, d, v), nn_ in zip(adv, nonneutral)]

    @torch.no_grad()
    def __call__(self, wav: torch.Tensor, sr: int) -> dict:
        """wav: 1D float tensor. Returns {'arousal', 'valence', 'dominance', 'nonneutral'}."""
        x = AF.resample(wav.float(), sr, SR) if sr != SR else wav.float()
        x = x.to(self.device)
        xn = (x - x.mean()) / (x.std() + 1e-7)  # Wav2Vec2 feature-extractor normalization
        with torch.autocast("cuda", dtype=torch.float16, enabled=str(self.device).startswith("cuda")):
            a, d, v = self.arousal(xn[None])[0].float().tolist()
        if self.e2v is None:
            return {"arousal": a, "dominance": d, "valence": v, "nonneutral": None}
        res = self.e2v.generate(x.cpu().numpy(), granularity="utterance", extract_embedding=False, disable_pbar=True)[0]
        if self._neutral_idx is None:
            self._neutral_idx = next(i for i, l in enumerate(res["labels"]) if "neutral" in l)
        p = np.asarray(res["scores"], dtype=np.float64)
        return {"arousal": a, "dominance": d, "valence": v, "nonneutral": float(1 - p[self._neutral_idx] / p.sum())}
