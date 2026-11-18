# Contributing to Grey

Thanks for wanting to help! 🦜 Bug reports, fixes, new language front-ends, faster inference and better docs are all welcome.

## Before you start

- **Small fix?** Just open a pull request.
- **Bigger change** (a new feature, a new language, a training change)? Open an issue first so we can talk it through.
- Grey is **non-commercial** (see [LICENSE](LICENSE)). By contributing you agree your contribution is shared under the same terms.

## Setting up

```bash
git clone https://github.com/Randy-H0/Grey-tts.git
cd Grey-tts
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e third_party/kvae-audio
python grey.py download v16plus
python grey.py speak "It works!" -o test.wav
```

## Pull requests

- Keep each pull request about one thing, and say what it changes and how you checked it.
- Changes to the sound: attach a before/after pair (same text, same `--seed`).
- Changes to the text front-end (`dacts/text.py`, `dacts/numerals.py`): include a few example sentences in the description.
- Don't remove the AI marking (`dacts/marking.py`); every output must keep it.
- No model weights, datasets or recordings of real people in the repository.

## Voices and data

Only add voices or data you have the right to use, with their licence. Never contribute recordings of someone without their consent.
