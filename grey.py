"""Grey from the command line: text to speech, voice cloning from a recording, voice to voice.
Everything it writes is marked as AI-generated (a label in the WAV file + an inaudible watermark: dacts/marking.py).

    python grey.py download v16plus                                  # a model -> models/ (python grey.py download --list)
    python grey.py speak "Hello there." -o hello.wav                 # Grey's own voice
    python grey.py speak "Hallo!" --voice lib_lena -o hallo.wav      # a built-in voice (python grey.py voices)
    python grey.py speak "Hi, it's me." --ref me.wav -o me_hi.wav    # your own voice from a recording (10-30 s)
    python grey.py vtv input.wav --voice vctk_p225 -o heron.wav      # the same words and timing, another voice
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
MODELS = ROOT / "models"


def pick_model(arg):
    if arg:
        p = Path(arg)
        return p if p.exists() else MODELS / (arg if arg.endswith(".pt") else f"{arg}.pt")
    for name in ("v16plus", "v15plus", "v15xl-343k", "v14xl-beta", "v13xl-v", "smol-v145"):
        if (MODELS / f"{name}.pt").exists():
            return MODELS / f"{name}.pt"
    sys.exit("No model in models/. Get one first:  python grey.py download v16plus")


def load(args):
    import torch
    from dacts.infer import Synthesizer
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    path = pick_model(args.model)
    print(f"loading {path.name} on {dev}...", flush=True)
    s = Synthesizer.from_checkpoint(path, device=dev)
    lib = MODELS / "library_voices.pt"  # (the made-up voices: Lena, Jonas, Daan -- rows made for each model file)
    if lib.exists():
        d = torch.load(lib, map_location="cpu", weights_only=True)
        for vid, vec in (d["vectors"].get(path.name) or {}).items():
            s.set_voice(vid, vec)
    return s, dev


def reference(s, dev, ref, ref_text=None, lang=None):
    """A recording -> a voice (its table row) + the stretch Grey continues from (the prompt)."""
    import torch
    from dacts.spkemb import load_reference
    wav = load_reference(str(ref), 44100)
    s.add_voice("ref", wav)
    if ref_text:
        return s.make_prompt(wav[: int(20 * 44100)], ref_text, lang=lang)
    try:
        from dacts.auto_prompt import auto_prompt
        p = auto_prompt(s, wav if torch.is_tensor(wav) else torch.as_tensor(wav), dev)
        if p is None:
            print("(no clear speech found in the reference for a prompt: using the voice only)")
        return p
    except Exception as e:  # noqa: BLE001 (no speech recognizer installed: the voice alone still works)
        print(f"(couldn't transcribe the reference: {e}; pass --ref-text for a closer match)")
        return None


def cmd_speak(a):
    import torch
    from dacts.marking import write_ai_wav
    s, dev = load(a)
    prompt = reference(s, dev, a.ref, a.ref_text, None if a.lang == "auto" else a.lang) if a.ref else None
    speaker = "ref" if a.ref else (a.voice if a.voice else None)
    if a.seed is not None:
        torch.manual_seed(a.seed)
    kw = dict(lang=a.lang, steps=a.steps, guidance=a.guidance, temperature=a.temperature, prompt=prompt)
    if getattr(s, "voice_lang", None) is not None and speaker:
        kw["lang_fix"] = "steer"  # (a voice speaking another language gets that language's accent)
    if a.speed is not None:
        kw["speed"] = a.speed
    w = s(a.text, speaker=speaker, **kw)[0]
    write_ai_wav(a.out, w, 44100)
    print(f"wrote {a.out} ({len(w) / 44100:.1f} s, marked as AI-generated)")


def cmd_vtv(a):
    import soundfile as sf
    import torch
    from dacts.marking import write_ai_wav
    s, dev = load(a)
    if getattr(s.model, "vtv_in", None) is None:
        sys.exit("This model has no voice to voice: use Grey v14.5 or newer")
    prompt = reference(s, dev, a.ref, None) if a.ref else None
    w, sr = sf.read(a.input, dtype="float32", always_2d=True)
    import torchaudio.functional as AF
    src = AF.resample(torch.as_tensor(w.mean(1)), sr, 44100) if sr != 44100 else torch.as_tensor(w.mean(1))
    if a.seed is not None:
        torch.manual_seed(a.seed)
    target = "ref" if a.ref else a.voice
    lang = a.lang if a.lang != "auto" else ((getattr(s, "speaker_langs", None) or {}).get(target) or "en")  # (the target voice's own)
    out = s.convert(src, speaker=target, prompt=prompt, lang=lang, steps=a.steps)
    write_ai_wav(a.out, out, 44100)
    print(f"wrote {a.out} (marked as AI-generated)")


def cmd_voices(a):
    s, _ = load(a)
    names = sorted(k for k in s.speakers if not str(k).startswith(("spare", "_")))
    print(f"{len(names)} voices:", ", ".join(names))


def cmd_download(a):
    import download_model
    download_model.main(a.name, a.list)


def main():
    ap = argparse.ArgumentParser(description="Grey text-to-speech")
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", help="a model file or name in models/ (default: the newest you have)")
    common.add_argument("--device", help="cuda or cpu (default: cuda if there is one)")
    common.add_argument("--voice", help="a built-in voice id (see: grey.py voices)")
    common.add_argument("--ref", help="a recording of the voice to use (10-30 s of one person; only voices you may use)")
    common.add_argument("--lang", default="auto", help="en, de, nl, fr, es, pl, uk, ja, zh, ko or auto")
    common.add_argument("--steps", type=int, default=8)
    common.add_argument("--seed", type=int)
    common.add_argument("-o", "--out", default="grey.wav")
    p = sub.add_parser("speak", parents=[common], help="text to speech")
    p.add_argument("text")
    p.add_argument("--ref-text", help="what the reference recording says (else transcribed automatically)")
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--speed", type=float, help="0 slow .. 1 fast")
    p.set_defaults(fn=cmd_speak)
    p = sub.add_parser("vtv", parents=[common], help="voice to voice: a recording said again in another voice")
    p.add_argument("input")
    p.set_defaults(fn=cmd_vtv)
    p = sub.add_parser("voices", parents=[common], help="list a model's built-in voices")
    p.set_defaults(fn=cmd_voices)
    p = sub.add_parser("download", help="download a model from thisvoiceisai.com")
    p.add_argument("name", nargs="?", default="v16plus")
    p.add_argument("--list", action="store_true")
    p.set_defaults(fn=cmd_download)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
