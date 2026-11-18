"""Frozen multilingual language model features for the text encoder.

The TTS text encoder reads letters / kana, so it does not know which words matter in a sentence
(where stress goes, what is a question, how a reply relates to what was said before). This runs a
pretrained multilingual LM over "current line </s></s> previous line" and returns its token
features; the text encoder cross-attends to them. Not part of the TTS checkpoint (frozen).
"""
import torch

DEFAULT = "FacebookAI/xlm-roberta-base"


class TextLM:
    def __init__(self, name: str = DEFAULT, device: str = "cuda", max_len: int = 112, prev_words: int = 30):
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name)
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(name, torch_dtype=dtype).to(device).eval().requires_grad_(False)
        self.device, self.max_len, self.prev_words = device, max_len, prev_words
        self.dim = self.model.config.hidden_size
        self._fast = None

    def enable_fast(self):
        """Inference: a compiled copy of the LM for the GPU (eager it is ~150 small kernels per line); the CPU keeps the
        plain one (the website parks models in RAM)."""
        self._fast = torch.compile(self.model, dynamic=True)
        return self

    @staticmethod
    def tokenize(tok, text, prev_text, max_len=112, prev_words=30):
        """CPU part (runs in DataLoader workers): (line, previous line) -> (ids, seg) lists."""
        p = " ".join(prev_text.split()[-prev_words:]) if prev_text else None
        e = tok(text, p, truncation="longest_first" if p else True, max_length=max_len)
        sid = e.sequence_ids() if hasattr(e, "sequence_ids") else [0] * len(e["input_ids"])
        return e["input_ids"], [1 if s == 1 else 0 for s in sid]

    @torch.no_grad()
    def from_tokens(self, ids, mask, seg):
        """GPU part: padded (B, L) ids / bool mask / seg from collate -> same outputs as __call__."""
        ids, mask, seg = (t.to(self.device, non_blocking=True) for t in (ids, mask, seg))  # (pinned: no wait for queued work)
        lm = self._fast if self._fast is not None and ids.is_cuda else self.model  # (enable_fast: train.py with compile)
        out = lm(input_ids=ids, attention_mask=mask.long(), output_hidden_states=True)
        feats = torch.stack(out.hidden_states[-4:]).mean(0)
        return feats.float() * mask[..., None], mask, seg

    @torch.no_grad()
    def __call__(self, texts, prev_texts=None):
        """texts: list[str]; prev_texts: list[str | None] (previous line, if any).
        -> feats (B, L, dim) float32, mask (B, L) bool, seg (B, L) long (0 current line, 1 previous)."""
        prev_texts = prev_texts or [None] * len(texts)
        # only the end of the previous line (what a reply reacts to); keeps the LM cheap
        prev_texts = [" ".join(p.split()[-self.prev_words:]) if p else None for p in prev_texts]
        # trim the longer of (line, previous line) first; a very long line can need trimming itself
        enc = [self.tok(t, p, truncation="longest_first" if p else True, max_length=self.max_len) for t, p in zip(texts, prev_texts)]
        n = max(len(e["input_ids"]) for e in enc)
        ids = torch.full((len(enc), n), self.tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros(len(enc), n, dtype=torch.bool)
        seg = torch.zeros(len(enc), n, dtype=torch.long)
        for i, e in enumerate(enc):
            k = len(e["input_ids"])
            ids[i, :k] = torch.tensor(e["input_ids"])
            mask[i, :k] = True
            sid = e.sequence_ids() if hasattr(e, "sequence_ids") else [0] * k
            seg[i, :k] = torch.tensor([1 if s == 1 else 0 for s in sid])
        ids, mask, seg = ids.to(self.device), mask.to(self.device), seg.to(self.device)
        from .model import eager
        lm = self._fast if (self._fast is not None and ids.is_cuda and not eager() and next(self.model.parameters()).is_cuda) else self.model
        out = lm(input_ids=ids, attention_mask=mask.long(), output_hidden_states=True)
        feats = torch.stack(out.hidden_states[-4:]).mean(0)  # last layers: lexical + contextual
        return feats.float() * mask[..., None], mask, seg
