from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import text as T
from .codec import (HOP, KVAE_HOP, KVAE_SR, SAMPLE_RATE, kvae_to_audio, latent_space, latents_to_audio, load_dac,
                    load_kvae)
from .config import Config
from .model import DacTTS

# the fast codec (owner 2026-10-25: "add a fast codec switch"): dacts/fast_codec.FastDecoder, a ~20M distilled stand-in
# for KVAE's decoder (~27x faster on the CPU). A fixed snapshot (models/fast-codec.pt), not the training run's moving file.
FAST_CODEC = Path(__file__).resolve().parents[1] / "models" / "fast-codec.pt"
_FAST = {}


def fast_decoder(device):
    k = str(device)
    if k not in _FAST:
        from .fast_codec import FastDecoder
        _FAST[k] = FastDecoder.load(FAST_CODEC, device)
    return _FAST[k]
from .style import volume_to_rel_db

# Training audio keeps natural studio levels (median active-speech level ~-34 dB on
# EARS/Expresso), so raw output is quiet. `master` brings typical speech to a normal level.
DEFAULT_DATA_LEVEL_DB = -34.3


def speech_level_db(w: np.ndarray) -> float:
    """Active-speech level: power mean over 512-sample frames within 30 dB of the loudest."""
    fr = w[: len(w) // 512 * 512].reshape(-1, 512)
    if not len(fr):
        return -100.0
    db = 10 * np.log10((fr.astype(np.float64) ** 2).mean(1) + 1e-12)
    act = db > db.max() - 30
    return float(10 * np.log10((10 ** (db[act] / 10)).mean()))


def limit(w: np.ndarray, ceiling_db=-1.0, attack_ms=5.0, release_ms=60.0) -> np.ndarray:
    """Look-ahead peak limiter: smooth gain that keeps |w| <= ceiling without hard clipping."""
    c = 10 ** (ceiling_db / 20)
    w = w.astype(np.float32)
    if len(w) == 0 or np.abs(w).max() <= c:
        return w  # nothing to limit
    # envelope on 64-sample blocks (~1.5 ms): same behaviour, ~64x less work than per sample
    B = 64
    n = len(w)
    x = torch.from_numpy(np.pad(w, (0, (-n) % B)))
    blk = x.abs().view(1, 1, -1, B).amax(-1)                          # (1, 1, n/B) block peaks
    a = max(1, int(SAMPLE_RATE * attack_ms / 1000 / B)) | 1
    r = max(a, int(SAMPLE_RATE * release_ms / 1000 / B)) | 1
    env = F.max_pool1d(blk, r, 1, r // 2)                            # hold peaks for the release time
    gain = torch.clamp(c / env.clamp(min=1e-9), max=1.0)
    gain = -F.max_pool1d(-gain, a, 1, a // 2)                        # min over attack window ...
    gain = F.avg_pool1d(gain, a, 1, a // 2, count_include_pad=False)  # ... then smooth: never overshoots
    gain = F.interpolate(gain, scale_factor=B, mode="linear", align_corners=False)[0, 0, :n]
    return (torch.from_numpy(w) * gain).clamp(-c, c).numpy()


def master(w: np.ndarray, loudness_db=-20.0, dynamics=0.4, rel_db=None, data_level_db=DEFAULT_DATA_LEVEL_DB,
           ceiling_db=-1.0) -> np.ndarray:
    """Typical speech -> `loudness_db`; keeps `dynamics` (0..1) of the natural loudness difference
    vs typical speech (whisper quieter, shout louder); then peak-limits.

    rel_db: natural level of this take relative to typical speech. For models trained on
    level-normalized audio it comes from the volume setting; otherwise (natural-level training
    data) it is measured from the output itself.
    """
    lvl = speech_level_db(w)
    if rel_db is None:
        rel_db = lvl - data_level_db
    target = loudness_db + dynamics * rel_db
    return limit(w * 10 ** ((target - lvl) / 20), ceiling_db)


def native_lang(speaker) -> str:
    """Language a training voice spoke (from its name)."""
    s = str(speaker)
    if s.startswith("es_"):
        return "es"
    if s.startswith(("jsut", "jvnv", "tsukuyomi", "amitaro", "ritsu")):
        return "ja"
    return "en"



def can_compile() -> bool:
    """torch.compile needs Triton and a C compiler (Windows: cl.exe, see the Triton notes): a PC without them -- Grey
    Studio on someone else's computer -- runs the plain modules instead of failing on the first line."""
    import importlib.util
    import os
    import shutil
    if importlib.util.find_spec("triton") is None:
        return False
    return os.name != "nt" or bool(os.environ.get("CC") or shutil.which("cl"))

class Synthesizer:
    def __init__(self, model: DacTTS, latent, dac_model=None, device="cuda", text_lm=None):
        self.model = model.eval().to(device)
        self.text_lm = text_lm
        if text_lm is None and getattr(model.cfg, "lm_name", ""):
            from .lm import TextLM
            self.text_lm = TextLM(model.cfg.lm_name, device=device)
        self.latent = latent.to(device)
        self.kvae = getattr(model.cfg, "codec", "dac") == "kvae"
        # decoder of the codec the model generates for (KVAE: continuous latents, no codebooks)
        self.dac = dac_model or (load_kvae(device) if self.kvae else load_dac(device))
        self.device = device
        self.speakers = {}
        self.volume_normal = 0.5
        self.data_level_db = DEFAULT_DATA_LEVEL_DB
        self.volume_map = None  # set for models trained on level-normalized audio

    @staticmethod
    def resolve_adapter(ck: dict, path) -> dict:
        """(11-17) A LoRA fine-tune saved as an adapter file ({"adapter_of": "<base>.pt", "lora": {...}, "voices": {name:
        row}, ...}, a few MB) -> the full checkpoint it stands for: the base's weights with the adapters folded in, the
        new voices appended to the voice table. Looked up next to the file, then in <install>/models."""
        from . import lora as LORA
        here = Path(path).parent
        cands = [here / ck["adapter_of"], Path(__file__).resolve().parents[1] / "models" / ck["adapter_of"]]
        bp = next((p for p in cands if p.exists()), None)
        if bp is None:
            raise FileNotFoundError(f"{Path(path).name} is a voice for {ck['adapter_of']}, which isn't installed")
        base = torch.load(bp, map_location="cpu", weights_only=False)
        sd = dict(base.get("ema") or base["model"])
        sd = LORA.merge_sd({**sd, **ck["lora"]}, ck["lora_alpha"] / ck["lora_rank"])
        W = sd["global_cond.spk.weight"]
        names = list(ck["voices"])
        speakers = dict(base.get("speakers") or {})
        used = set(speakers.values())
        free = [i for i in range(W.shape[0]) if i not in used]
        if len(free) < len(names):  # (no spare rows left: grow the table)
            W = torch.cat([W, W.new_zeros(len(names) - len(free), W.shape[1])])
            free += list(range(W.shape[0] - (len(names) - len(free)), W.shape[0]))
        W = W.clone()
        for n, i in zip(names, free):
            W[i] = ck["voices"][n].to(W.dtype)
            speakers[n] = i
        sd["global_cond.spk.weight"] = W
        cfg = {**base["cfg"], "n_speakers": W.shape[0], "only_speakers": ",".join(names), "lora_rank": 0}
        return {**base, "model": sd, "ema": sd, "cfg": cfg, "speakers": speakers, "name": ck.get("name"), "step": ck.get("step"),
                "speaker_langs": {**(base.get("speaker_langs") or {}), **(ck.get("speaker_langs") or {})}}

    @classmethod
    def from_checkpoint(cls, path, device="cuda", use_ema=True, compile=False, fast=False):
        fast = fast and can_compile()
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if "adapter_of" in ck:
            ck = cls.resolve_adapter(ck, path)
        cfg = Config.from_dict(ck["cfg"])
        if "use_speed" not in ck["cfg"]:  # checkpoints from before the speed control
            cfg.use_speed = False
        model = DacTTS(cfg, ck["model"]["encoder.emb.weight"].shape[0])  # vocab size of that checkpoint
        if getattr(cfg, "lora_rank", 0) > 0:  # (11-17) a LoRA fine-tune: base weights + adapters, folded together
            from . import lora as LORA
            LORA.add(model, cfg.lora_rank, cfg.lora_alpha, LORA.top_blocks(model, getattr(cfg, "lora_top_blocks", 0)))
        model.load_state_dict(ck["ema" if use_ema and "ema" in ck else "model"])
        if getattr(cfg, "lora_rank", 0) > 0:
            LORA.merge(model)
        lat = ck["latent"]
        if compile:  # slower first call (compilation), faster afterwards
            model.compile_modules()
        elif fast:  # inference: compiled decoder on the GPU only (the CPU keeps the plain one)
            model.enable_fast_decoder()
        synth = cls(model, latent_space(lat), device=device)
        if fast and synth.text_lm is not None:  # (the text LM too: compiled copy on the GPU only)
            synth.text_lm.enable_fast()
        synth.speakers = ck.get("speakers", {})
        synth.speaker_langs = ck.get("speaker_langs", {})
        vlp = Path(path).with_suffix(".voicelang.npz")  # the language subspace of this model's voice table (optional)
        synth.voice_lang = None
        if vlp.exists():
            import numpy as _np
            d = _np.load(vlp)
            synth.voice_lang = {"Q": torch.from_numpy(d["Q"]).float(), "mu": torch.from_numpy(d["mu"]).float(),
                                "mus": torch.from_numpy(d["mus"]).float(), "langs": [str(x) for x in d["langs"]]}
        cond = ck.get("conditioning", {})
        synth.volume_normal = cond.get("volume_normal", 0.5)
        synth.data_level_db = cond.get("level_db_median", DEFAULT_DATA_LEVEL_DB)
        if "encode_level_db" in cond:
            synth.volume_map = cond["volume_map_db"]
        return synth

    @torch.no_grad()
    def add_voice(self, name: str, audio, sr: int = 44100):
        """Zero-shot voice from reference audio (file path(s) or 1-D tensor(s) at `sr`). It becomes an
        ordinary voice (one speaker-table row), so generation costs exactly the same. Returns its vector."""
        enc = self.model.global_cond.spk_enc
        if enc is None:
            raise ValueError("this model has no zero-shot voice encoder")
        from .spkemb import SpeakerEmbedder, load_reference

        if getattr(self, "_spk_embedder", None) is None:
            self._spk_embedder = SpeakerEmbedder(self.device)
        clips = audio if isinstance(audio, (list, tuple)) else [audio]
        wavs = [load_reference(a, sr) if isinstance(a, (str, Path)) else a for a in clips]
        x = self._spk_embedder.from_long(torch.cat(wavs), sr)
        vec = enc.eval()(x[None].to(self.device))[0]
        return self.set_voice(name, vec)

    def _voice_lang_fix(self, speaker, own, langs, mode):
        """Language leak fix (Solver 11-17, notes_solver/V15PLUS_CONTINUATION.md section 5): a voice's table row tells
        the model its language (every voice spoke only one in training), so an English voice speaking French came out
        English. For lines in another language than the voice's own, the row's part in the 16-direction language
        subspace (models/<model>.voicelang.npz, fitted on that model's table) is moved to the all-language mean
        ("erase") or to the target language's mean ("steer"). Returns the per-line change (B, dim) or None."""
        vl = getattr(self, "voice_lang", None)
        if vl is None or not any(l != own for l in langs):
            return None
        W = self.model.global_cond.spk.weight
        v = W[int(speaker)].detach().float()
        Q, mu, mus, names = vl["Q"].to(v.device), vl["mu"].to(v.device), vl["mus"].to(v.device), vl["langs"]
        rows = []
        for l in langs:
            if l == own or (mode == "steer" and l not in names):
                rows.append(torch.zeros_like(v))
                continue
            c = mus[names.index(l)] if mode == "steer" else mu
            rows.append(-(Q.T @ (Q @ (v - c))))
        return torch.stack(rows).to(W.dtype)

    def _lang_id(self, lang: str) -> int:
        """The language slot; a language this model was never given a slot for (Ukrainian on a 9-language model, 2026-11-16)
        is an error instead of an out-of-range embedding."""
        i = T.LANGS[lang]
        if i >= getattr(self.model.cfg, "n_langs", 8):
            raise ValueError(f"this model can't speak '{lang}'")
        return i

    @torch.no_grad()
    def make_prompt(self, audio, text: str, lang: str = None, max_seconds: float = 10.0, sr: int = 44100,
                    labels: dict = None) -> dict:
        """Reference audio (path or 1-D tensor at `sr`) + its exact transcript -> a prompt for __call__(prompt=...).
        Encoded like the training data (speech level -20 dB, KVAE at 48 kHz); longer audio is cut to max_seconds."""
        import torchaudio.functional as AF

        if not self.kvae:
            raise ValueError("prompted cloning needs a KVAE model")
        if isinstance(audio, (str, Path)):
            from .spkemb import load_reference
            audio = load_reference(audio, sr)
        w = audio.float().cpu().numpy()[: int(max_seconds * sr)]
        w = limit(w * 10 ** ((-20.0 - speech_level_db(w)) / 20))
        w48 = AF.resample(torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)), sr, KVAE_SR)
        step = 20 * KVAE_SR  # long references in 20 s pieces (60 s at once needs > 8 GB of activations)
        zs = [self.dac.encode(w48[i:i + step][None, None].to(self.device), sample_rate=KVAE_SR)[0][0].T
              for i in range(0, len(w48), step) if len(w48[i:i + step]) >= KVAE_SR // 10]
        z = torch.cat(zs)
        out = {"latents": self.latent.encode(z[None].float())[0], "text": text, "lang": lang or T.detect_lang(text, T.langs_of(getattr(self.model.cfg, 'n_langs', 8)))}
        if labels:  # how the reference was delivered, e.g. {"whisper": 1.0} (models with cfg.use_prompt_labels)
            out["labels"] = {k: float(v) for k, v in labels.items() if v is not None}
        return out

    @staticmethod
    def combine_prompts(prompts: list) -> dict:
        """Several make_prompt() results -> one reference whose clips keep their own labels, e.g. a voice's normal,
        whispered and excited lines: [make_prompt(a, ta, labels={}), make_prompt(b, tb, labels={"whisper": 1}), ...].
        Models trained with cfg.use_clip_labels take the voice from all of them and the delivery from the clip whose
        labels match the request. Keep the total near the training range (<= ~15 s)."""
        parts = [{"latents": q["latents"], "text": q["text"], "lang": q.get("lang"), "labels": dict(q.get("labels") or {})}
                 for q in prompts]
        return {"latents": torch.cat([q["latents"] for q in parts]), "text": " ".join(q["text"] for q in parts),
                "lang": parts[0]["lang"], "parts": parts}

    @torch.no_grad()
    def make_delivery(self, audio, sr: int = 44100, seconds: float = 8.0) -> torch.Tensor:
        """Reference audio -> normalized latents (T, D) of its middle `seconds`: the delivery encoder's input."""
        if isinstance(audio, (str, Path)):
            from .spkemb import load_reference
            audio = load_reference(audio, sr)
        n = int(seconds * sr)
        s = max(0, (len(audio) - n) // 2)
        return self.make_prompt(audio[s:s + n], "", max_seconds=seconds, sr=sr)["latents"].cpu()

    def make_reference(self, audios, sr: int = 44100, seconds: float = 20.0) -> torch.Tensor:
        """Reference clips of a voice (paths or arrays; one or several) -> normalized latents (T, D) of up to `seconds`
        in all, the clips one after another: the v14 reference segment (models trained with use_ref_seg), as in training."""
        if isinstance(audios, (str, Path)) or (hasattr(audios, "ndim") and audios.ndim == 1):
            audios = [audios]
        parts, left = [], seconds
        for a in audios:
            if left < 1.0:
                break
            if isinstance(a, (str, Path)):
                from .spkemb import load_reference
                a = load_reference(a, sr)
            a = a[: int(left * sr)]
            parts.append(self.make_prompt(a, "", max_seconds=left, sr=sr)["latents"].cpu())
            left -= len(a) / sr
        return torch.cat(parts) if parts else None

    @torch.no_grad()
    def vtv_pitch(self, source, speaker=None, prompt=None, lang="en", sr: int = 44100) -> dict:
        """The Pitch slider's recommendation for convert (site, 2026-11-01): the clip's median F0 against the target
        voice's -> speech: the semitones between them (clamped to +-12, as transpose="auto"); singing: the whole
        octaves singing mode picks on its own (keeps the song's key). Grey (no speaker, no prompt): no target, 0."""
        from . import sing
        if getattr(self, "_vtv", None) is None:
            from .vtv import ContentFeatures
            self._vtv = ContentFeatures(self.device)
        w = np.asarray(source, dtype=np.float32)
        if sr != 44100:
            import torchaudio.functional as AF
            w = AF.resample(torch.from_numpy(w), sr, 44100).numpy()
        if prompt is not None:
            z = (prompt["latents"] if isinstance(prompt, dict) else prompt).float().to(self.device)
            a = self._vtv.codec(self.latent.denormalize(z[None])[0].T[None].float())[0].float().cpu().numpy()
            tgt = sing.median_hz(sing.f0_of(a, 44100, self.device, refine=False))
        else:
            tgt = self.voice_hz(speaker, lang) if speaker is not None else 0.0
        f0 = sing.f0_of(w, 44100, self.device, refine=False)
        src = sing.median_hz(f0)
        ok = tgt > 0 and src > 0
        return {"source_hz": round(src, 1), "target_hz": round(tgt, 1),
                "speech": float(np.clip(12 * np.log2(tgt / src), -12, 12)) if ok else 0.0,
                "singing": 12 * sing.octave_shift(f0, tgt) if ok else 0}

    @torch.no_grad()
    def convert(self, source, speaker=None, prompt=None, lang="en", clean=1.0, steps=8, guidance=2.0, temperature=0.85,
                spk_guidance=1.0, ref_seg="prompt", mastering=True, loudness_db=-20.0, dynamics=0.4, max_seconds=30.0,
                sr: int = 44100, singing=False, fast_codec=False, transpose="auto", accent=0.0, text=None,
                prompt_text=None, prompt_lang=None, solver=None):
        """Voice-to-voice (models trained with cfg.use_vtv, v14xl-vtv): `source` (audio path / 1-D array at `sr`, or
        normalized latents (T, D) e.g. a training clip's) said again in another voice, keeping its words, timing,
        melody and stress. The voice: `speaker` (a learned voice: name or id) and / or `prompt` (make_prompt(...) or
        audio: zero-shot, as cloning). ref_seg="prompt": the prompt also as the v14 reference segment (as the site's
        clones). singing: dacts/sing.py -- the melody read by RMVPE, moved by whole octaves into the target voice's
        range, and the output re-pitched toward it (WORLD): 0 / False = off, up to 1 / True = exactly on the melody
        (the site's Singing slider). fast_codec: decode with dacts/fast_codec (as the site's
        takes; ~27x faster on the CPU). transpose (speech, singing off): "auto" = the source moved by the semitones
        between its median pitch and the target voice's (clamped to +-12; Grey: none), a number = that many semitones,
        None / 0 = as recorded. With singing: "auto" / None = whole octaves into the voice's range (as before), a number
        (0 included) = the melody moved by exactly that many semitones (the site's Pitch slider, you 2026-11-01: "a
        lot of bleed from the original audio ... a slider from -24 to +24"). Returns float32 audio at 44.1 kHz."""
        from .model import patch_contour
        if getattr(self.model, "vtv_in", None) is None:
            raise ValueError("this model wasn't trained for voice-to-voice (cfg.use_vtv)")
        if getattr(self, "_vtv", None) is None:
            from .vtv import ContentFeatures
            self._vtv = ContentFeatures(self.device)
        if getattr(self, "_contour", None) is None:
            from .contour import load_contour
            self._contour = load_contour(self.model.cfg.contour_path, self.device)
        p = self.model.cfg.patch

        def latents(a):  # -> normalized latents (T, D) on the device, whole decoder steps
            z = a if (torch.is_tensor(a) and a.dim() == 2) else self.make_prompt(
                a if isinstance(a, (str, Path)) else torch.as_tensor(a), "", max_seconds=max_seconds, sr=sr)["latents"]
            z = z.float().to(self.device)
            return z[: z.shape[0] // p * p]

        def features(z, shift_st=0.0):  # content (1, T/p, 768) from the latents decoded by the fast codec (as in
            n = z.shape[0] // p           # training) + contour
            raw = self.latent.denormalize(z[None])[0].T  # (64, T)
            content = self._vtv.from_latents(raw, z.shape[0], shift_st)  # (T, 768) per KVAE frame
            content = content[: n * p].reshape(n, p, -1).mean(1)[None]
            zp = z[None] if p == 1 else z.reshape(1, n, p * z.shape[-1])
            return content, patch_contour(self._contour, zp, p)

        def decoded(z):  # normalized latents -> audio at 44.1 kHz (fast codec: what the content is read from)
            return self._vtv.codec(self.latent.denormalize(z[None])[0].T[None].float())[0].float().cpu().numpy()

        import time as _time
        times, t_last = {}, [_time.perf_counter()]

        def lap(name):  # seconds per stage (self.last_times; the website logs them): GPU work finished first
            if str(self.device).startswith("cuda"):
                torch.cuda.synchronize()
            now = _time.perf_counter()
            times[name] = round(times.get(name, 0.0) + now - t_last[0], 2)
            t_last[0] = now

        zs = latents(source)
        zp = latents(prompt["latents"] if isinstance(prompt, dict) else prompt) if prompt is not None else None
        lap("encode")
        melody, shift = None, 0.0

        def target_hz():  # the target voice's natural median F0 (0 = no target: Grey)
            from . import sing
            return sing.median_hz(sing.f0_of(decoded(zp), 44100, self.device, refine=False)) if zp is not None else (
                self.voice_hz(speaker, lang) if speaker is not None else 0.0)
        if singing:  # dacts/sing.py: the melody (RMVPE), moved by whole octaves into the target voice's range
            from . import sing
            src_f0 = sing.f0_of(decoded(zs), 44100, self.device)
            if isinstance(transpose, (int, float)) and not isinstance(transpose, bool):
                shift = float(transpose)
            else:
                shift = 12.0 * sing.octave_shift(src_f0, target_hz())
            melody = src_f0 * 2.0 ** (shift / 12)
            lap("melody")
        elif transpose == "auto":  # (you 2026-10-31: "allow the pitch to be transposed up or down automatically to fit the
            from . import sing      # persons natural voice. I notice a lot of the source voice comes through")
            tgt = target_hz()
            src = sing.median_hz(sing.f0_of(decoded(zs), 44100, self.device, refine=False)) if tgt > 0 else 0.0
            shift = float(np.clip(12 * np.log2(tgt / src), -12, 12)) if tgt > 0 and src > 0 else 0.0
            lap("transpose")
        elif transpose:
            shift = float(transpose)
        self.last_shift = shift
        content, contour = features(zs)
        # the contour asks for the target's range (ContourNet: semitones re 100 Hz); the content stays as recorded.
        # Measured 2026-10-31 (8 s cross-gender clips, 4 pairs, WavLM-large voice match to the target): off -0.02 / 0.31 /
        # 0.14 / 0.03 -> contour moved 0.31 / 0.47 / 0.18 / 0.39 (output pitch 156 -> 229 Hz etc.); the content moved
        # along too (4 or 12 st, as training's +-4 resample) gave mixed results and once -0.05: contour only
        if shift and not singing:
            contour[0, :, 0] += shift
        if singing:  # the model's pitch channel = that melody where it's voiced (elsewhere ContourNet's, moved alike)
            st, v = sing.frame_pitch(melody, contour.shape[1], 2 * p)
            st, v = torch.as_tensor(st, dtype=contour.dtype, device=contour.device), torch.as_tensor(v, device=contour.device)
            contour[0, :, 0] = torch.where(v, st, contour[0, :, 0] + shift)
        pr = None
        if zp is not None:
            pc, pk = features(zp)
            pr = {"latents": zp[None] if p == 1 else zp.reshape(1, -1, p * zp.shape[-1]), "content": pc, "contour": pk}
        lap("content")
        spk = None
        if speaker is not None and self.model.cfg.n_speakers > 1:
            sid = self.speakers[speaker] if isinstance(speaker, str) else int(speaker)
            spk = torch.full((1,), sid, device=self.device)
        lng = (torch.tensor([self._lang_id(lang)], device=self.device)
               if getattr(self.model.cfg, "use_lang", False) and lang is not None else None)
        lab = {}
        if clean is not None and getattr(self.model.cfg, "use_clean", False):
            lab["clean"] = torch.full((1,), float(clean), device=self.device)
        if getattr(self.model.global_cond, "guid", None) is not None:  # (cfg.use_guid, Smol v14.5: one pass per step)
            lab["guid"] = torch.full((1,), float(guidance), device=self.device)
            guidance = 1.0
        xp = {}  # singing: the model's "exact pitch" mode (trained from v14xl-vtv 85k on; older DacTTS.convert lack it)
        if singing:
            import inspect
            if "exact_pitch" in inspect.signature(self.model.convert).parameters:
                xp["exact_pitch"] = True
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(self.device).startswith("cuda")):
            g = self.model.global_cond(spk, None, None, None, lng, **lab)
            g_nospk = self.model.global_cond(None, None, None, None, lng, **lab) if spk is not None else None
            acc = {}
            if accent and accent > 0:  # Accent strength (you 2026-11-07): the target voice's own pronunciation
                acc = self._accent_cond(zs, zp, speaker, lang, clean, text, prompt_text, prompt_lang, accent, lap)
            if solver:  # (a distilled Smol: "euler", its own N steps)
                xp["solver"] = solver
            z = self.model.convert(content, contour, g=g, g_nospk=g_nospk, prompt=pr, **xp, **acc,
                                   ref_seg=pr["latents"] if (pr is not None and ref_seg == "prompt") else None,
                                   steps=steps, guidance=guidance, temperature=temperature, spk_guidance=spk_guidance)
        lap("model")
        if fast_codec and self.kvae and FAST_CODEC.exists():
            audio = fast_decoder(z.device)(self.latent.denormalize(z.float()).transpose(1, 2))
        else:
            audio = kvae_to_audio(self.dac, self.latent.denormalize(z.float())) if self.kvae else latents_to_audio(
                self.dac, self.latent.denormalize(z.float()))
        w = audio[0].float().cpu().numpy()
        lap("codec")
        if melody is not None:  # the notes: re-pitched toward the melody (WORLD) by the slider's amount
            w = sing.impose(w, 44100, melody, self.device, strength=1.0 if singing is True else float(singing))
            lap("repitch")
        self.last_melody, self.last_times = melody, times
        return master(w, loudness_db, dynamics, None, self.data_level_db) if mastering else w

    def _accent_cond(self, zs, zp, speaker, lang, clean, text, prompt_text, prompt_lang, accent, lap):
        """convert's accent blend inputs: the clip's transcript (given, else Whisper's) read by the TTS side in the target
        voice and aligned to the clip's frames (monotonic, as training: synthesize(align_to=...)). {} when it can't be
        done (no words heard; a voice file without its prompt's transcript; the alignment not covering the frames)."""
        self.last_accent = 0.0  # (what was applied: 0 when skipped -- the site's take details show this)
        if zp is not None and not prompt_text:  # (checked before Whisper: no CPU spent on a clip we can't use)
            return {}
        if not text:
            from .auto_prompt import transcribe
            w = self._vtv.codec(self.latent.denormalize(zs[None])[0].T[None].float())[0].float().cpu()
            text = transcribe(w, "cuda" if str(self.device).startswith("cuda") else "cpu")
            lap("transcribe")
        self.last_text = text = (text or "").strip()
        if not text:
            return {}
        prompt = None
        if zp is not None:
            if not prompt_text:
                return {}
            prompt = {"latents": zp, "text": prompt_text, "lang": prompt_lang or lang}
        rec = {}
        al = zs[None] if zp is None else torch.cat([zp, zs])[None]
        cuda = str(self.device).startswith("cuda")
        from . import model as M
        hook, M.PASS_HOOK = M.PASS_HOOK, None  # (the alignment pass isn't one of the take's passes on the progress bar)
        try:
          with torch.random.fork_rng(devices=[self.device] if cuda else []):  # (the take's own noise untouched)
            self(text, speaker=speaker, lang="auto", mastering=False, whisper=0.0, clean=clean, steps=1,
                 fast_codec=True, prompt=prompt, align_to=al, record=rec,
                 keep_prompt_across_languages=True)  # (frames only: a dropped prompt would misalign the words)
        finally:
            M.PASS_HOOK = hook
        c = rec.get("cond")
        if c is None or c.shape[1] != al.shape[1]:
            print(f"accent: alignment didn't cover the clip ({None if c is None else c.shape[1]} vs {al.shape[1]} frames): off", flush=True)
            return {}
        lap("accent")
        self.last_accent = float(accent)
        return {"tts_cond": c.float(), "g_tts": rec.get("g"), "accent": float(accent)}

    @torch.no_grad()
    def voice_hz(self, speaker, lang="en") -> float:
        """A learned voice's median F0 (singing mode's target range): RMVPE on one sample line, kept per voice."""
        cache = self.__dict__.setdefault("_voice_hz", {})
        if speaker not in cache:
            from . import sing
            line = {"en": "I was thinking we could take the long way home tonight, if you want to.",
                    "nl": "Ik dacht dat we vanavond de lange weg naar huis konden nemen, als je wilt."}.get(lang)
            # (a fixed seed: unseeded, Wren's line came out at 148-171 Hz from take to take; the caller's RNG untouched)
            from . import model as M
            hook, M.PASS_HOOK = M.PASS_HOOK, None  # (not one of the take's passes on the site's progress bar)
            try:
              with torch.random.fork_rng(devices=[self.device] if str(self.device).startswith("cuda") else []):
                torch.manual_seed(0)
                # (whisper 0 + clean 1 as the site sends them: left out, Wren whispers most takes -- no pitch to read)
                w = self(line or "I was thinking we could take the long way home tonight, if you want to.",
                         speaker=speaker, lang=lang if line else "en", mastering=False, whisper=0.0, clean=1.0,
                         steps=4, fast_codec=True)[0]
            finally:
                M.PASS_HOOK = hook
            cache[speaker] = sing.median_hz(sing.f0_of(w, 44100, self.device))
        return cache[speaker]

    @torch.no_grad()
    def voice_from_latents(self, lat: torch.Tensor, chunk: int = 1000) -> torch.Tensor:
        """Normalized reference latents (T, D), e.g. make_prompt(...)["latents"] -> voice vector from the voice
        reference encoder (models trained with use_vre): mean over <= 20 s windows, as in training."""
        enc = self.model.voice_enc.eval()
        lat = lat.float().to(self.device)
        parts = [lat[i:i + chunk] for i in range(0, max(1, len(lat) - chunk // 4), chunk)] or [lat]
        vs = [enc(p[None], torch.ones(1, len(p), dtype=torch.bool, device=self.device))[0] for p in parts if len(p) >= 100]
        return torch.stack(vs or [enc(lat[None], torch.ones(1, len(lat), dtype=torch.bool, device=self.device))[0]]).mean(0)

    @torch.no_grad()
    def set_voice(self, name: str, vec: torch.Tensor):
        """Put a voice vector into a free speaker slot under `name` (or overwrite that name's slot)."""
        table = self.model.global_cond.spk.weight
        if name in self.speakers:
            slot = self.speakers[name]
        else:  # (the taken slots kept as a set: rebuilding it from ~195k voices took every clone request; rebuilt only
            # when voices were added / removed elsewhere -- code review 2026-10-29)
            used = getattr(self, "_used_slots", None)
            if used is None or len(used) != len(self.speakers):
                used = self._used_slots = set(self.speakers.values())
            slot = next(i for i in range(table.shape[0] - 1, 0, -1) if i not in used)
        table[slot] = vec.to(table)
        self.speakers[name] = slot
        if getattr(self, "_used_slots", None) is not None:
            self._used_slots.add(slot)
        return vec.detach().float().cpu()

    @torch.no_grad()
    def __call__(self, texts, speaker=0, style=0.5, volume="typical", speed=0.5, lang="auto", mastering=True,
                 code_picker=True, prev_text=None, mood=None, dominance=None, whisper=None, clean=1.0, band=1.0,
                 spk_guidance="auto", prompt=None, dlv=None, prompt_timing=1.0,
                 loudness_db=-20.0, fast_codec=False,
                 dynamics=0.4, **kw):
        """texts: str or list[str]; speaker: id or name; style: 0 = narration .. 1 = acting;
        volume: 0..0.1 whisper .. 1 shouting, or "typical" (default) for a speaker's normal voice.
        style / volume / speed / speaker = None leave that label out (the model picks something
        natural; it was trained with labels randomly dropped).
        mastering: normalize typical speech to `loudness_db`, keeping `dynamics` of the natural
        whisper/shout loudness differences, then peak-limit (False = raw model output).
        Returns list of float32 numpy arrays at 44.1kHz."""
        spk_name = speaker if isinstance(speaker, str) else {v: k for k, v in self.speakers.items()}.get(speaker)
        if isinstance(speaker, str):
            speaker = self.speakers[speaker] if speaker in self.speakers else int(speaker)
        texts = [texts] if isinstance(texts, str) else texts
        # lang: "en" / "ja" / "auto" (Japanese if the text has kana or kanji) / None (label left out;
        # the text is still read as detected)
        detected = [T.detect_lang(t, T.langs_of(getattr(self.model.cfg, 'n_langs', 8))) for t in texts]
        langs = detected if lang in ("auto", None) else [lang] * len(texts)
        # a prompt in another language (an English clone reference, a Dutch line) makes the model continue IN the
        # prompt's language: owner's clone, Dutch line on v11 sounded 51-84% Dutch with it, 97-99% without. The
        # delivery reference stays (12 lines: P(Dutch) 0.94 with vs 0.93 without, identity 0.897 vs 0.893).
        keep = kw.pop("keep_prompt_across_languages", False)
        prompt_lang_ = None if prompt is None else (prompt.get("lang") or T.detect_lang(prompt["text"], T.langs_of(getattr(self.model.cfg, 'n_langs', 8))))
        if prompt is not None and not keep and prompt_lang_ not in langs:
            prompt = None
        # a line without final punctuation reads as "the speaker isn't finished" (conversational training clips are
        # often cut mid-thought), so the model trails into the next word: end every line as a finished sentence
        _END = tuple(".!?…。！？」』)\"'”")
        texts = [t if t.rstrip().endswith(_END) else t.rstrip() + ("。" if l == "ja" else ".") for t, l in zip(texts, langs)]
        # a (normal-voice) prompt clip makes the model continue in that delivery and overrides the whisper label
        # (a tester clone: voiced share 0.55 with the prompt vs 0.17 without): whispering clones go on the voice alone
        # A model with prompt labels (cfg.use_prompt_labels) is told how the prompt was delivered (prompt["labels"],
        # e.g. {"whisper": 1} for a whispered one), so it keeps the prompt: a labelled prompt + whisper = this voice
        # whispering. Unlabelled prompts and older models: as before.
        knows = (getattr(self.model, "prompt_lab", None) is not None and bool((prompt or {}).get("labels"))) or (
            getattr(self.model.decoder, "clab_in", None) is not None
            and any(q.get("labels") for q in ((prompt or {}).get("parts") or [prompt or {}])))
        if prompt is not None and whisper is not None and whisper >= 0.5 and not knows and not prompt.get("whispered"):
            prompt = None  # (a whispered clip of the voice itself is kept: continuing it whispers -- any model)
        if whisper is not None and whisper >= 0.5:
            # whisper clips carry no clean label in training, so clean=1 ("studio-clean", learned only from
            # voiced speech) pulled whispers back to a normal voice (Tsukuyomi: aperiodicity 0.07 -> 0.73 without)
            clean = None
            # and steer along whisper vs the same voice explicitly normal (VCTK p281: 0.63 -> 0.72)
            kw.setdefault("whisper_guidance", 2.0)
        # speaker guidance (holds identity): auto = 3 when speaking a language that isn't the voice's own
        if spk_guidance == "auto":
            sl = getattr(self, "speaker_langs", {}) or {}
            if spk_name in sl or not sl:  # a voice the model trained on (old checkpoints: guessed from its name)
                own = sl.get(spk_name) or native_lang(spk_name)
            else:  # a zero-shot voice (a clone, an eval prompt): counted as English, ON PURPOSE. The code review
                # (2026-10-29) found every clone got 3x speaker guidance on non-English lines by accident; measured, it
                # HELPS: Seed-TTS zh, 200 lines, SIM 0.664 -> 0.689 (+0.025 +- 0.003, 156/200), CER 6.4 -> 6.9%.
                # English clone lines (1 here) are being tested at 2 / 3 (data/seedtts_eval/vm/_sweep10.cmd).
                own = "en"
            spk_guidance = 3.0 if spk_name is not None and own is not None and any(l != own for l in langs) else 1.0
        kw["spk_guidance"] = float(spk_guidance)
        lang_fix = kw.pop("lang_fix", "off")  # "erase" / "steer": the voice's language taken out of its vector (below)
        voice_own = None
        gq = getattr(self.model.global_cond, "lang_Q", None)
        if getattr(self.model.global_cond, "lang_mean", None) is not None and gq is not None and bool(gq.abs().sum() > 0):
            lang_fix = "off"  # (T2s live inside the model; v15uk runs with lang_Q = 0, so S0 stays on there)
        if lang_fix in ("erase", "steer") and spk_name is not None:
            sl = getattr(self, "speaker_langs", {}) or {}
            voice_own = sl.get(spk_name) or (native_lang(spk_name) if (spk_name in sl or not sl) else (prompt_lang_ or "en"))
        marked = any("]:" in t for t in texts)
        T.mixed_script(True)  # Japanese lines with English / Spanish words: each part in its own script
        try:
            enc = [T.encode_marked(t, self.model.cfg.add_blank, l) for t, l in zip(texts, langs)]
        finally:
            T.mixed_script(False)
        nv = self.model.encoder.emb.weight.shape[0]  # older models lack newer letters (e.g. German a-umlaut / sharp s)
        ids = [T.fit_vocab(e[0], nv) for e in enc]
        word_ms, durations_only = kw.pop("word_ms", None), kw.pop("durations_only", False)  # the site's Timing mode
        gap_ms = kw.pop("gap_ms", None)  # (and its pauses: ms per gap_ids gap, None = the model's own)
        line_ids = list(ids[0])
        seconds = kw.pop("seconds", None)  # optional length override for the line (the prompt keeps its own timing)
        # sing given notes (composer output, cfg.use_notes): [(onset s, duration s, MIDI, word index)], words = text.split()
        song_notes = kw.pop("notes", None)
        # singing: whole bars. bpm (+ beats_per_bar, default 4) -> the predicted length is rounded to whole bars;
        # bars=N sets the bar count by hand (overrides both the duration model's choice and `seconds`)
        bpm, bars, bpb = kw.pop("bpm", None), kw.pop("bars", None), kw.pop("beats_per_bar", 4)
        # singing labels (cfg.use_music; each optional, all trained with dropout): sing=True/False, key="A minor",
        # genre="pop" or ["pop", "rnb"]; a bpm implies singing unless sing=False is given
        sing, key, genre = kw.pop("sing", None), kw.pop("key", None), kw.pop("genre", None)
        if sing is None and bpm:
            sing = True
        bar_s = 60.0 / float(bpm) * float(bpb) if bpm else None
        if bar_s and bars:
            seconds = float(bars) * bar_s
        blank = kw.pop("blank_seconds", None)
        if blank and bar_s:  # no words + a tempo: whole bars too (bars=N, or the asked seconds rounded to bars)
            blank = float(bars) * bar_s if bars else max(1, round(float(blank) / bar_s)) * bar_s
        if blank:  # speech from blankness: no words, one BLANK token for `blank` seconds (training: cfg.text_hide)
            spp = (KVAE_HOP * SAMPLE_RATE / KVAE_SR if self.kvae else HOP) * self.model.cfg.patch  # samples per patch
            ids, texts, prompt, marked = [[T.BLANK] for _ in ids], ["" for _ in texts], None, False
            kw["durations"] = torch.full((len(ids), 1), max(1, round(float(blank) * SAMPLE_RATE / spp)), device=self.device)
        elif seconds:
            spp = (KVAE_HOP * SAMPLE_RATE / KVAE_SR if self.kvae else HOP) * self.model.cfg.patch
            kw["target_len"] = max(1, round(float(seconds) * SAMPLE_RATE / spp))
        elif bar_s:
            spp = (KVAE_HOP * SAMPLE_RATE / KVAE_SR if self.kvae else HOP) * self.model.cfg.patch
            kw["len_quantum"] = bar_s * SAMPLE_RATE / spp  # patches per bar (fractional: no drift over many bars)
        if song_notes and getattr(self.model.cfg, "use_notes", False) and len(texts) == 1:
            spf = (KVAE_HOP / KVAE_SR if self.kvae else HOP / SAMPLE_RATE) * self.model.cfg.patch  # seconds per frame
            wids = T.word_ids(ids[0])
            n_words = max([w for w in wids if w is not None and w >= 0], default=-1) + 1
            starts = {}
            ends_w = {}  # each word's sung end (its last note's end); the time to the next word is a rest
            for on, du, midi, w in song_notes:
                if w is not None and w >= 0:
                    starts[w] = min(starts.get(w, on), on); ends_w[w] = max(ends_w.get(w, 0.0), on + du)
            end = max(on + du for on, du, _, _ in song_notes)
            if getattr(self.model.cfg, "implicit_align", False):  # v13: the notes only set the length; the model
                # itself places the words on them (no word timing given)
                kw["target_len"] = int(round((end + 0.3) / spf)) + 1
                kw.pop("len_quantum", None)
            elif n_words and starts:
                # words without a note of their own share the gap before the next noted word (or the tail after
                # the last one) instead of dropping the note timing altogether -- that made the voice skip words
                if len(starts) < n_words:
                    end = max(end, max(starts.values()) + 0.25 * sum(1 for w in range(n_words) if w > max(starts)))
                    known = sorted(starts) + [n_words]
                    first = known[0]
                    for w in range(first):  # leading words: squeeze into 0..first start
                        starts[w] = starts[first] * w / max(1, first)
                    for k0, k1 in zip(known, known[1:]):
                        t1 = starts[k1] if k1 < n_words else end
                        for w in range(k0 + 1, k1):
                            starts[w] = starts[k0] + (t1 - starts[k0]) * (w - k0) / (k1 - k0)
                bounds = [starts[w] for w in range(n_words)] + [end]
                sung = [min(bounds[w + 1], ends_w.get(w, bounds[w + 1])) for w in range(n_words)]
                cnt = {}
                for _, _, _, w in song_notes:
                    if w is not None and w >= 0:
                        cnt[w] = cnt.get(w, 0) + 1
                kw["word_frames"] = [(round(bounds[w] / spf), max(round(bounds[w] / spf) + 1, round(sung[w] / spf)),
                                      round(bounds[w + 1] / spf), cnt.get(w, 1)) for w in range(n_words)]
                kw["word_ids"] = wids
                kw["word_vowels"] = T.vowel_flags(ids[0])
                prompt = None  # the notes set the timing (a prompt would bring its own)
            nf = torch.zeros(1, int(round((end + 0.3) / spf)) + 1, 4)
            nf[..., 3] = 1.0
            for on, du, midi, _ in song_notes:
                a_, b_ = int(round(on / spf)), max(int(round((on + du) / spf)), int(round(on / spf)) + 1)
                nf[0, a_:b_, 0] = (midi - 60.0) / 24.0; nf[0, a_:b_, 1] = 1.0; nf[0, a_, 2] = 1.0
            kw["notes"] = nf.to(self.device)
        if marked and getattr(self.model.encoder, "emph", None) is not None:  # '[word]:2.0' emphasis marks
            n = max(len(i) for i in ids) + (len(T.encode(prompt["text"], self.model.cfg.add_blank, prompt.get("lang") or T.detect_lang(prompt["text"], T.langs_of(getattr(self.model.cfg, 'n_langs', 8))))) if prompt is not None else 0)
            ev = torch.ones(len(ids), n, device=self.device)
            es = torch.zeros(len(ids), n, dtype=torch.bool, device=self.device)
            off = n - max(len(i) for i in ids)  # prompted cloning puts the prompt's tokens first
            for b, (_, vals) in enumerate(enc):
                for k, v in enumerate(vals):
                    if v is not None:
                        ev[b, off + k], es[b, off + k] = v, True
            kw["emph"] = (ev, es)
        texts = [T.strip_marks(t) for t in texts]  # the language model reads the plain text
        if dlv is not None and getattr(self.model, "dlv_enc", None) is not None:  # how this voice talks
            kw["dlv"] = dlv[None].to(self.device) if dlv.dim() == 2 else dlv.to(self.device)
        if kw.get("ref_seg") is not None:  # v14 reference segment: other clips of the voice (make_reference)
            r = kw.pop("ref_seg")
            if getattr(self.model, "ref_enc", None) is not None:
                kw["ref_seg"] = r[None].to(self.device) if r.dim() == 2 else r.to(self.device)
        else:
            kw.pop("ref_seg", None)
        if prompt is not None:  # prompted cloning: the prompt's text goes first, the new line continues it
            if len(texts) != 1:
                raise ValueError("prompted cloning takes one text at a time")
            parts = prompt.get("parts") or [{"latents": prompt["latents"], "text": prompt["text"], "lang": prompt.get("lang"),
                                             "labels": prompt.get("labels") or {}}]
            per = [T.fit_vocab(T.encode(q["text"], self.model.cfg.add_blank, q.get("lang") or T.detect_lang(q["text"], T.langs_of(getattr(self.model.cfg, 'n_langs', 8)))), nv)
                   for q in parts]  # (each clip's words on their own, as in training)
            pids = [i for q in per for i in q]
            ids[0] = pids + ids[0]
            clab = None
            if getattr(self.model.decoder, "clab_in", None) is not None and any(q.get("labels") for q in parts):
                from .model import PROMPT_LABELS
                vec = lambda lab: [float(lab[k]) if lab.get(k) is not None else float("nan") for k in PROMPT_LABELS]  # noqa: E731
                fr = torch.tensor([vec(q.get("labels") or {}) for q in parts for _ in range(len(q["latents"]))])
                tk = torch.tensor([vec(q.get("labels") or {}) for q, pq in zip(parts, per) for _ in pq])
                clab = (fr, tk)
            pause_ids = {T._VOCAB[c] for c in " .,!?-:;\"()¿¡…、。" if c in T._VOCAB}  # where pauses live
            if getattr(self.model.encoder, "pctx", None) is not None:  # prompt prosody needs the pitch net
                if getattr(self, "_contour", None) is None:
                    from .contour import load_contour
                    self._contour = load_contour(self.model.cfg.contour_path, self.device)
                kw["contour_net"] = self._contour
            kw["prompt"] = {"latents": prompt["latents"][None].to(self.device), "n_tok": len(pids),
                            "pause": torch.tensor([[i in pause_ids for i in ids[0]]], device=self.device),
                            "timing": float(prompt_timing)}  # 0 = model's own pace, 1 = the prompt's pace
            if clab is not None:  # every reference patch / word: the labels of its own clip
                kw["prompt"]["frame_labels"], kw["prompt"]["token_labels"] = clab[0].to(self.device), clab[1].to(self.device)
            if getattr(self.model, "prompt_lab", None) is not None and prompt.get("labels"):
                kw["prompt_labels"] = {k: float(v) for k, v in prompt["labels"].items() if v is not None}
            if prev_text is None:
                prev_text = prompt["text"]
        vocab = self.model.encoder.emb.num_embeddings
        ids = [[i if i < vocab else T.BLANK for i in seq] for seq in ids]  # English-only models: drop unknown
        line_words = None
        if word_ms is not None or gap_ms is not None or durations_only:  # each token's word (the new line only: a prompt's tokens come first)
            wid = T.word_ids(line_ids)
            nwd = max(wid) + 1 if wid and max(wid) >= 0 else 0
            gid, gap_marks, gap_after = T.gap_ids(line_ids, wid)  # pauses: numbered after the words
            ng = len(gap_marks)
            off = len(ids[0]) - len(line_ids)
            wt = torch.full((1, len(ids[0])), -1, dtype=torch.long)
            wt[0, off: off + len(wid)] = torch.tensor([w if w >= 0 else (nwd + g if g >= 0 else -1) for w, g in zip(wid, gid)])
            kw["word_tok"] = wt.to(self.device)
            line_words = ["".join(T._INV.get(line_ids[k], "") for k in range(len(wid)) if wid[k] == w) for w in range(nwd)]
            if word_ms is not None or gap_ms is not None:
                fit = lambda v, n: [None if x is None else float(x) for x in list(v or [])[:n]] + [None] * max(0, n - len(v or []))
                kw["word_ms"] = fit(word_ms, nwd) + fit(gap_ms, ng)
            kw["durations_only"] = bool(durations_only)
        x_len = torch.tensor([len(i) for i in ids], device=self.device)
        tokens = torch.zeros(len(ids), int(x_len.max()), dtype=torch.long, device=self.device)
        for i, seq in enumerate(ids):
            tokens[i, : len(seq)] = torch.tensor(seq)
        # speaker=None -> no speaker conditioning (voice is not pinned to a training speaker)
        spk = (torch.full((len(ids),), speaker, device=self.device)
               if self.model.cfg.n_speakers > 1 and speaker is not None else None)
        sty = (torch.full((len(ids),), float(style), device=self.device)
               if self.model.cfg.use_style and style is not None else None)
        volume = self.volume_normal if volume == "typical" else volume
        whispering = (whisper is not None and whisper >= 0.5 and getattr(self.model.cfg, "use_whisper", False))
        if whispering:  # whisper is its own control; volume (normal-voice loudness) does not apply
            volume = None
        vol = (torch.full((len(ids),), float(volume), device=self.device)
               if getattr(self.model.cfg, "use_volume", False) and volume is not None else None)
        # speed: 0 slow .. 0.5 speaker's normal pace .. 1 fast; None leaves it to the model
        spd = (torch.full((len(ids),), float(speed), device=self.device)
               if getattr(self.model.cfg, "use_speed", False) and speed is not None else None)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(self.device).startswith("cuda")):
            lng = (torch.tensor([self._lang_id(l) for l in langs], device=self.device)
                   if getattr(self.model.cfg, "use_lang", False) and lang is not None else None)
            use_picker = code_picker and getattr(self.model, "code_head", None) is not None and not self.kvae
            # newer controls (0..1, None = left out): mood sad..happy, dominance meek..assertive,
            # whisper normal..whispered (when whispering, volume is ignored: it means normal-voice loudness)
            lab = {}
            # clean: recording cleanliness, 1 = studio-clean (default), 0 = like the hissy recordings
            for k, v in (("mood", mood), ("dominance", dominance), ("whisper", whisper), ("clean", clean), ("band", band)):
                if v is not None and getattr(self.model.cfg, f"use_{k}", False):
                    lab[k] = torch.full((len(texts),), float(v), device=self.device)
            if getattr(self.model.cfg, "use_music", False):
                from . import music as MU
                vals = {"sing": None if sing is None else float(bool(sing)), "bpm": MU.bpm_norm(bpm),
                        "bpb": MU.bpb_norm(bpb) if bpm else None, "key": MU.key_norm(key)}
                for k, v in vals.items():
                    if v is not None:
                        lab[k] = torch.full((len(texts),), float(v), device=self.device)
                gv = MU.genre_vector([genre] if isinstance(genre, str) else genre) if genre else None
                if gv is not None:
                    lab["genre"] = torch.tensor([gv] * len(texts), device=self.device)
            if lab:
                kw["labels"] = lab
            # previous line's text (meaning / reply context); the LM also reads the line itself
            if self.text_lm is not None:
                pt = prev_text if isinstance(prev_text, list) else [prev_text] * len(texts)
                # in its own fp16, as in training (train.py runs it outside autocast): under bf16 autocast every matmul
                # re-cast the frozen weights each call (2026-10-24: ~2x slower) and fed features training never saw
                with torch.autocast("cuda", enabled=False):
                    kw["lm"] = self.text_lm(texts, pt)
            fix = self._voice_lang_fix(speaker, voice_own, langs, lang_fix) if voice_own is not None and spk is not None else None
            self.model.global_cond.spk_fix = fix
            try:
                res = self.model.synthesize(tokens, x_len, spk=spk, style=sty, volume=vol, speed=spd, lang=lng,
                                            pick_codes=use_picker, **kw)
            finally:
                self.model.global_cond.spk_fix = None
        if durations_only:  # -> [{"word", "ms"}] for the new line, the duration model's own choice
            from .codec import frame_rate
            fps = frame_rate(self.model.cfg) / max(1, int(getattr(self.model.cfg, "patch", 1)))
            d, wt = res[0].float().cpu(), kw["word_tok"][0].cpu()
            return ([{"word": w, "ms": round(float(d[wt == k].sum()) * 1000.0 / fps)} for k, w in enumerate(line_words)]
                    + [{"gap": m, "after": a, "ms": round(float(d[wt == len(line_words) + g].sum()) * 1000.0 / fps)}
                       for g, (m, a) in enumerate(zip(gap_marks, gap_after))])
        if use_picker:  # DAC codes sampled by the code picker
            z, y_len, codes = res
            with torch.no_grad():
                audio = self.dac.decode(self.dac.quantizer.from_codes(codes)[0])[:, 0]
        else:  # snap the flow's continuous output to the nearest codes
            z, y_len = res
            if fast_codec and self.kvae and FAST_CODEC.exists():  # the fast codec: raw latents (B, 64, T) -> 44.1 kHz
                with torch.no_grad():
                    audio = fast_decoder(z.device)(self.latent.denormalize(z.float()).transpose(1, 2))
            else:
                dec = kvae_to_audio if self.kvae else latents_to_audio
                audio = dec(self.dac, self.latent.denormalize(z.float()))
        spf = KVAE_HOP * SAMPLE_RATE / KVAE_SR if self.kvae else HOP  # output samples per latent frame
        out = [a[: int(int(n) * spf)].float().cpu().numpy() for a, n in zip(audio, y_len)]
        if mastering:
            # no volume label: the model picks a level; master it like a typical voice
            rel = (volume_to_rel_db(0.05 if whispering else (volume if volume is not None else self.volume_normal),
                                    self.volume_map) if self.volume_map else None)
            out = [master(w, loudness_db, dynamics, rel, self.data_level_db) for w in out]
        return out
