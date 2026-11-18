"""Singing labels (owner 2026-10-18): sung or spoken, tempo, beats per bar, key, genre. All are optional and dropped
in training like every other label (cfg.label_drop / label_drop_all), so any of them -- or all -- can be left out.
Values are mapped to 0..1 for the scalar label MLPs in GlobalCond; genre is a multi-hot vector over GENRES."""
import math

GENRES = ("pop", "rock", "folk", "hiphop", "indie", "electronic", "alternative", "singersongwriter", "world", "dance",
          "rnb", "soul", "funk", "latin", "blues", "country", "reggae", "chanson", "jazz", "easylistening", "metal", "punk")
# Jamendo tags -> one or more of GENRES (compounds split, near-synonyms merged); tags not listed are ignored
_TAG = {**{g: (g,) for g in GENRES},
        "rap": ("hiphop",), "popfolk": ("pop", "folk"), "poprock": ("pop", "rock"), "electropop": ("electronic", "pop"),
        "dancepop": ("dance", "pop"), "indiepop": ("indie", "pop"), "indierock": ("indie", "rock"),
        "ethnicrock": ("world", "rock"), "folkrock": ("folk", "rock"), "punkrock": ("punk", "rock"), "rocknroll": ("rock",),
        "hardrock": ("rock",), "psychedelic": ("rock",), "progressive": ("rock",), "grunge": ("rock", "alternative"),
        "house": ("dance", "electronic"), "club": ("dance", "electronic"), "techno": ("dance", "electronic"),
        "trance": ("dance", "electronic"), "edm": ("dance", "electronic"), "triphop": ("electronic",),
        "ethno": ("world",), "celtic": ("world", "folk"), "african": ("world",), "ska": ("reggae",), "dub": ("reggae",),
        "chansonfrancaise": ("chanson",), "fusion": ("jazz",), "lounge": ("jazz", "easylistening"), "swing": ("jazz",),
        "chillout": ("easylistening",), "lofi": ("easylistening",), "rhythmandblues": ("rnb",), "heavymetal": ("metal",),
        "bossanova": ("latin", "jazz"), "reggaeton": ("latin", "dance"), "salsa": ("latin",), "tango": ("latin",)}
KEYS = [f"{n} {m}" for m in ("major", "minor") for n in "C C# D D# E F F# G G# A A# B".split()]
BPM_LO, BPM_HI = 60.0, 200.0


def genre_vector(tags):
    """Jamendo genre tags -> multi-hot list over GENRES, or None (no usable genre: label left out)."""
    v = [0.0] * len(GENRES)
    for t in tags or ():
        for g in _TAG.get(str(t).lower().replace(" ", "").replace("-", ""), ()):
            v[GENRES.index(g)] = 1.0
    return v if any(v) else None


def bpm_norm(bpm):
    """tempo -> 0..1 on a log scale (60..200 BPM; folded into 70..180 at labelling)."""
    if bpm is None:
        return None
    return min(1.0, max(0.0, math.log(float(bpm) / BPM_LO) / math.log(BPM_HI / BPM_LO)))


def key_norm(key):
    """'F# minor' (or an index 0..23) -> 0..1. Keys are 1/23 apart (x1000 in the label embedding: well separated)."""
    if key is None:
        return None
    i = key if isinstance(key, int) else KEYS.index(str(key).strip().replace("♯", "#"))
    return i / (len(KEYS) - 1)


def bpb_norm(beats_per_bar):
    """beats per bar 2..7 -> 0..1 (4/4 = 0.4)."""
    if beats_per_bar is None:
        return None
    return min(1.0, max(0.0, (float(beats_per_bar) - 2) / 5))
