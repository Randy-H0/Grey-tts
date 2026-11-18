# Grey

Grey is a multilingual text-to-speech model: natural speech in ten languages (English, German, Dutch, French, Spanish,
Polish, Ukrainian, Japanese, Chinese, Korean), voice cloning from a short recording, voice to voice, and controls for
delivery (style, mood, assertiveness, speed, whisper), timing and emphasis. It generates audio as continuous latents
(flow matching) on top of the KVAE-Audio codec at 44.1 kHz.

This repository holds the model code, inference and training. The ready-to-use desktop app (Grey Studio, Windows) and
the website live at [thisvoiceisai.com](https://thisvoiceisai.com) ([download](https://thisvoiceisai.com/download/)).

> Everything Grey generates is AI-generated speech. This code marks its output as such (a label in the WAV file and an
> inaudible AudioSeal watermark, `dacts/marking.py`), as the EU AI Act asks of AI systems that make synthetic audio.
> Keep the marks in, and say it's AI when you share it.

## Quick start

Python 3.12 and an NVIDIA GPU are recommended (CPU works, slowly).

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e third_party/kvae-audio
python grey.py download v16plus
python grey.py speak "Hello! This is Grey." -o hello.wav
```

More:

```bash
python grey.py download --list                                   # every model and what it can do
python grey.py voices                                            # the built-in voices of a model
python grey.py speak "Guten Tag!" --voice lib_lena -o hallo.wav  # a built-in voice
python grey.py speak "Hi, it's me." --ref me.wav -o me.wav       # your own voice from a 10-30 s recording
python grey.py vtv input.wav --voice vctk_p225 -o out.wav        # voice to voice (v14.5 and newer)
```

In Python:

```python
from dacts.infer import Synthesizer
from dacts.marking import write_ai_wav

grey = Synthesizer.from_checkpoint("models/v16plus.pt", device="cuda")
wav = grey("Hello there.", speaker="expresso_ex01", lang="en", steps=8, guidance=3.0)[0]   # 44.1 kHz float32
write_ai_wav("hello.wav", wav, 44100)
```

## Models

| Model | Languages | Own voices | Voice to voice |
|---|---|---|---|
| v16+ | 10 (+ Ukrainian) | yes | yes |
| v15+ / v15 | 9 | yes | yes |
| v14.5 | 7 | yes | yes |
| v13 XL | 7 | yes | no |
| v14.5 Smol / v13 Smol | 7 | no | no |
| v12 | 6 | no | no |
| v10 / v6 | 3 | no | no |

`python grey.py download <id>` fetches a model (resumable, checked by SHA-256). The downloadable files keep only the
built-in voices of the voice table.

## Layout

```
dacts/            the model package
  infer.py          Synthesizer: text to speech, cloning (add_voice / make_prompt), voice to voice (convert)
  model.py          the model (text encoder, duration / rhythm model, flow-matching decoder, voice conditioning)
  text.py           text front-end for the ten languages (numbers, stress, scripts); numerals.py
  codec.py          KVAE-Audio latents <-> audio; fast_codec.py, fast_encoder.py (distilled students)
  data.py, fastdata.py   training data loading
  marking.py        AI label + watermark for every output
  ...               labels (style, emotion, delivery), voice embeddings, LoRA, voice to voice (vtv.py), ...
train.py          training (and fine-tuning / LoRA: lora_rank=..., only_speakers=...)
preprocess.py     data preparation: audio -> clips; transcribe.py, encode_kvae.py, add_labels.py,
                  label_clean_gpu.py, relabel_band.py, extract_spkemb.py, fix_whisper_labels.py
grow.py, grow_v13xl.py   growing a model's voice table / languages for new data
configs/          small data files the front-end and labels read
tools/            export_voicelang.py (the language-fix table for a model)
third_party/      kvae-audio (MIT), rmvpe pitch tracker (MIT), DNSMOS quality model (CC BY 4.0)
grey.py           command line;  download_model.py   model downloads
```

### Training on your own data

```bash
python preprocess.py --help          # audio -> clips, per speaker
python transcribe.py --help          # text for each clip
python encode_kvae.py --help         # clips -> codec latents
python add_labels.py --help          # style / speed / loudness labels;  label_clean_gpu.py: quality labels
python train.py --data data/my_set --out runs/my_run --init models/v16plus.pt lora_rank=16
```

`train.py` takes any setting of `dacts/config.py` as `key=value` (`--resume` continues a run in `--out`). A LoRA
fine-tune of a few speakers (`lora_rank=16`, `grad_ckpt=True`) fits a 6-8 GB GPU.

## License

Free for personal, non-commercial use (see [LICENSE](LICENSE)). Part of the data the models learned from (Emilia,
CC BY-NC 4.0) and some of the voices only allow non-commercial use. Only clone voices that are yours or that you have
permission for; never use Grey to impersonate or mislead. Credits for the speech data and voices:
https://thisvoiceisai.com/terms.html#credits. Third-party code keeps its own licence (third_party/*/LICENSE).

Made in the Netherlands. Contact: support@thisvoiceisai.com
