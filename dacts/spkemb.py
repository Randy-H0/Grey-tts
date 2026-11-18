"""Speaker-verification x-vectors (WavLM-base-plus-sv): input of the zero-shot voice encoder."""
import torch
import torchaudio.functional as AF

MODEL = "microsoft/wavlm-base-plus-sv"


class SpeakerEmbedder:
    """Reference audio -> unit-norm 512-d x-vector (also used at inference for zero-shot voices)."""

    def __init__(self, device="cuda"):
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self.fe = AutoFeatureExtractor.from_pretrained(MODEL)
        self.model = WavLMForXVector.from_pretrained(MODEL).to(device).eval()
        self.device = device

    @staticmethod
    def prep(fe, wav: torch.Tensor, sr: int = 44100) -> dict:
        """The CPU half (16 kHz + the feature extractor): extract_spkemb.py runs it in its audio workers."""
        w = AF.resample(wav.float().cpu(), sr, 16000).numpy()
        if len(w) < 16000:  # the x-vector network needs ~1 s: loop very short clips
            w = __import__("numpy").resize(w, 16000) if len(w) else __import__("numpy").zeros(16000, "float32")
        return fe(w, sampling_rate=16000, return_tensors="pt")

    @torch.no_grad()
    def embed(self, x: dict) -> torch.Tensor:
        e = self.model(**{k: v.to(self.device) for k, v in x.items()}).embeddings[0]
        return torch.nn.functional.normalize(e.float(), dim=-1).cpu()

    @torch.no_grad()
    def embed_batch(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T) prepped 16 kHz inputs of ONE length (no padding: the first conv layer's group norm runs over the whole
        clip, so zero padding would change the fingerprint) -> (B, 512) unit vectors."""
        e = self.model(input_values=x.to(self.device)).embeddings
        return torch.nn.functional.normalize(e.float(), dim=-1).cpu()

    def __call__(self, wav: torch.Tensor, sr: int = 44100) -> torch.Tensor:
        return self.embed(self.prep(self.fe, wav, sr))

    def from_long(self, wav: torch.Tensor, sr: int = 44100, chunk_s: float = 6.0) -> torch.Tensor:
        """Average over ~6 s chunks of a longer reference (mean of unit vectors, renormalized)."""
        n = int(chunk_s * sr)
        parts = [wav[i:i + n] for i in range(0, max(1, len(wav) - n // 3), n)] or [wav]
        es = [self(p, sr) for p in parts if len(p) > sr]  # >= 1 s
        return torch.nn.functional.normalize(torch.stack(es or [self(wav, sr)]).mean(0), dim=-1)


def load_reference(path, sr=44100) -> torch.Tensor:
    """Any audio file -> mono float tensor at `sr`."""
    from .audio_io import read_any

    w, r = read_any(path)
    w = torch.from_numpy(w.mean(1))
    return AF.resample(w, r, sr) if r != sr else w
