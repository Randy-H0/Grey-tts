# Changelog

Grey's model versions, newest first. Every model is downloadable: `python grey.py download --list`.

## Speech editing — November 2026 (code, works with v16+)
- ✂️ **Change words in a real recording** (`grey.py edit`, `dacts.edit.edit_speech`): the new sentence is compared with
  what the recording says; for each change Grey says a short phrase again (the new words plus a neighbour or two, in the
  recording's own timing and voice) and the result is joined into the original at real word gaps, with the recording's
  own background under it. English only for now; the first run downloads a forced aligner (hubert-large, 1.3 GB).

## v16+ — November 2026
- 🇺🇦 **Ukrainian** joins: 10 languages.
- More Dutch, Polish and Spanish voices.
- Language fix: a voice speaking another language takes on that language's accent instead of its own.

## v15+ / v15 — November 2026
- 🇫🇷 **French** and 🇰🇷 **Korean**: 9 languages.
- v15+: the **Swift-RM** rhythm model built in, for more natural timing, and closer clones (Seed-TTS English voice match 0.65).

## v14.5 — November 2026
- 🔁 **Voice to voice**: say something, hear it back in another voice with the same words and timing.
- Big jump in voice cloning (Seed-TTS English voice match 0.53 → 0.62).
- **v14.5 Smol**: a small, fast model distilled from it (runs in a web browser).

## v13 XL — October 2026
- 🇩🇪 **German** joins: 7 languages.
- 🎙️ **Voice cloning** from a 10-30 s recording.

## v12 and earlier — September / October 2026
- v12: six languages (🇳🇱 Dutch, 🇵🇱 Polish and 🇨🇳 Chinese join).
- v10 / v6: the first models: English, Spanish and Japanese.
