"""Text -> token ids. English: normalized characters. Japanese: katakana + pitch-accent marks.
Spanish: English characters + á é í ó ú ñ ü ¿ ¡ (numbers spelled in Spanish).

Vocabulary layout is append-only, so older checkpoints (English-only) keep their ids:
  0 PAD, 1 BLANK, English characters, then Japanese symbols (katakana, long vowel, devoicing mark,
  accent marks). Models know their own vocab size (read from the checkpoint).

Japanese (no IPA): pyopenjtalk gives the reading in katakana and full-context labels; from the
labels we insert  ↑ pitch rise, ↓ pitch fall (accent nucleus), | accent-phrase boundary  after
the mora where they happen (same rules as ESPnet's prosody symbols, but on kana).
"""
import json
import re
from pathlib import Path
import unicodedata

PAD = 0
BLANK = 1
_CHARS = " abcdefghijklmnopqrstuvwxyz0123456789'.,!?-:;\"()"
_JA = "".join(chr(c) for c in range(0x30A1, 0x30F7)) + "ー’↑↓|"  # ァ..ヶ, long vowel, devoiced, marks
_VOCAB = {c: i + 2 for i, c in enumerate(_CHARS)}
_ES = "áéíóúñü¿¡"
_VOCAB.update({c: len(_CHARS) + 2 + i for i, c in enumerate(_JA)})
_VOCAB.update({c: len(_CHARS) + 2 + len(_JA) + i for i, c in enumerate(_ES)})
# Dutch / Polish letters (appended: older checkpoints keep their ids). Chinese = pinyin letters + tone digits 1-5
# (digits never occur in normalised English / Spanish text: numbers are spelled out there).
_NLPL = "ąćęłńśźżëïèàöêç"
_VOCAB.update({c: len(_CHARS) + 2 + len(_JA) + len(_ES) + i for i, c in enumerate(_NLPL)})
VOCAB_SIZE_EN = len(_CHARS) + 2          # vocab of English-only checkpoints
VOCAB_SIZE_JA = VOCAB_SIZE_EN + len(_JA)  # vocab of English + Japanese checkpoints
VOCAB_SIZE_ES = VOCAB_SIZE_JA + len(_ES)  # vocab of en+ja+es checkpoints (v4 .. v10)
VOCAB_SIZE_NLPL = VOCAB_SIZE_ES + len(_NLPL)  # + Dutch / Polish letters (v11 .. v12)
# German letters, all German-only symbols (owner: own characters, like Dutch / Polish got): ä ß, plus German
# ö / ü as their own symbols (U+E010 / U+E011 inside the model; normalize_nlpl maps them for lang="de") rather than
# sharing Dutch ö / Spanish ü. Appended so older checkpoints keep their ids.
_DE_OE, _DE_UE = "", ""
_DE = "äß" + _DE_OE + _DE_UE
_VOCAB.update({c: VOCAB_SIZE_NLPL + i for i, c in enumerate(_DE)})
VOCAB_SIZE_DE = VOCAB_SIZE_NLPL + len(_DE)  # + German (2026-10-20)
# French (2026-11-08, you: "add french and Korean"): the accented letters no earlier language had (é è ê ë à ç ï ü are
# shared with Spanish / Dutch already). Korean: Hangul syllables split into their conjoining jamo -- 19 initials, 21
# vowels, 27 finals (U+1100.., U+1161.., U+11A8..): 67 symbols instead of 11,172 syllables. Appended (older checkpoints
# keep their ids; fit_vocab maps the French letters to their plain letter for them, Korean has no older equivalent).
_FR = "âîôûùÿœæ"
_KO = ("".join(chr(c) for c in range(0x1100, 0x1113)) + "".join(chr(c) for c in range(0x1161, 0x1176))
       + "".join(chr(c) for c in range(0x11A8, 0x11C3)))
_VOCAB.update({c: VOCAB_SIZE_DE + i for i, c in enumerate(_FR)})
_VOCAB.update({c: VOCAB_SIZE_DE + len(_FR) + i for i, c in enumerate(_KO)})
VOCAB_SIZE_FRKO = VOCAB_SIZE_DE + len(_FR) + len(_KO)  # (v15 XL .. v15+)
# Ukrainian (you 2026-11-16: "can we also add ukrainian?"): the 33 letters of its Cyrillic alphabet, appended (older
# checkpoints keep their ids; they have no Cyrillic, fit_vocab gives them BLANK). The apostrophe is the shared "'".
_UK = "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя"
_VOCAB.update({c: VOCAB_SIZE_FRKO + i for i, c in enumerate(_UK)})
VOCAB_SIZE = VOCAB_SIZE_FRKO + len(_UK)
# a model trained before a letter existed reads it as its nearest old letter (same length: emphasis marks stay aligned)
_FALLBACK = {"ä": "a", "ß": "s", _DE_OE: "ö", _DE_UE: "ü", "â": "a", "î": "i", "ô": "o", "û": "u", "ù": "u", "ÿ": "y",
             "œ": "o", "æ": "a"}


def fit_vocab(ids: list[int], n_vocab: int) -> list[int]:
    """Map ids the model has no embedding for (letters added after it was trained) to their fallback letter."""
    if all(i < n_vocab for i in ids):
        return ids
    inv = {v: k for k, v in _VOCAB.items()}
    return [i if i < n_vocab else _VOCAB.get(_FALLBACK.get(inv.get(i, ""), ""), BLANK) for i in ids]
LANGS = {"en": 0, "ja": 1, "es": 2, "nl": 3, "pl": 4, "zh": 5, "de": 6, "fr": 7, "ko": 8, "uk": 9}  # (uk: models with cfg.n_langs >= 10)  # (v14.5 XL has 8 slots: ko
# needs a model grown to cfg.n_langs >= 9 -- the French / Korean continuation)

_REPLACE = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "..."}


def _num(m: re.Match) -> str:
    from num2words import num2words

    s = m.group(0)
    if "." in s:
        return num2words(float(s))
    n = int(s)
    if 1100 <= n <= 2099 and len(s) == 4:
        return num2words(n, to="year")
    return num2words(n)


def expand_numbers(text: str) -> str:
    """'21st' -> 'twenty-first', '1990' -> 'nineteen ninety', '$5' -> 'five dollars', '75%' -> ..."""
    from num2words import num2words

    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)  # 1,000 -> 1000
    text = re.sub(r"\$(\d+)\.(\d\d)\b", lambda m: f"{m.group(1)} dollars and {int(m.group(2))} cents", text)
    text = re.sub(r"\$(\d+)", lambda m: f"{m.group(1)} dollars", text)
    text = re.sub(r"(\d+)(st|nd|rd|th)\b", lambda m: num2words(int(m.group(1)), to="ordinal"), text)
    text = re.sub(r"\d+\.\d+|\d+", _num, text)
    return text.replace("%", " percent").replace("&", " and ")


_LETTERS = dict(zip("abcdefghijklmnopqrstuvwxyz", "ay bee see dee ee eff jee aitch eye jay kay el em en oh pee cue ar ess tee "
                                                      "you vee double-you ex why zee".split(" ")))
# acronyms said as words (everything else in capitals is spelled out letter by letter)
_WORDLIKE = {"NASA", "NATO", "UNESCO", "UNICEF", "LASER", "RADAR", "SCUBA", "AIDS", "COVID", "OPEC", "FIFA", "IKEA", "LEGO",
             "ASAP", "GIF", "JPEG", "PIN", "SIM", "WIFI", "YOLO", "FOMO", "SWAT", "NAFTA", "ISIS", "OK", "ROM", "RAM", "LAN"}
_ALWAYS_SPELL = {"UK", "EU", "UN", "TV", "PC", "AI", "ID", "DJ", "CEO", "FAQ", "USA", "BBC", "CNN", "DIY", "PS", "VR"}
_WORDS = {}


def _common_words() -> set:
    """Lowercase words common in our English transcripts (configs/en_words.txt): an all-caps token that is one of
    these is a shouted word, not an acronym."""
    if "w" not in _WORDS:
        f = Path(__file__).resolve().parent.parent / "configs" / "en_words.txt"
        _WORDS["w"] = set(f.read_text(encoding="utf-8").split()) if f.exists() else set()
    return _WORDS["w"]


_ACRO = re.compile(r"\b([A-Z]{2,6})(s|'s)?\b")
_CAPS_RUN = re.compile(r"\b[A-Z]{2,}\b(?:[\s,!?.'-]+\b[A-Z]{2,}\b)+")


def expand_acronyms(text: str) -> str:
    """'The FBI uses GPUs' -> 'The eff-bee-eye uses jee-pee-you's'. Word-like ones (NASA) stay words, and runs of
    capitalised words ('THIS IS SO COOL') are shouting, not acronyms."""
    shout = [(m.start(), m.end()) for m in _CAPS_RUN.finditer(text)]

    def sub(m):
        w = m.group(1)
        if any(a <= m.start() < b for a, b in shout) or w in _WORDLIKE:
            return m.group(0)
        if w == "US" and not re.search(r"(?i)\bthe\s+$", text[: m.start()]):  # "help US" (shouted) vs "the US"
            return m.group(0)
        if w not in _ALWAYS_SPELL and w != "US" and w.lower() in _common_words():  # one shouted word ("NO", "STOP") is a word
            return m.group(0)
        return "-".join(_LETTERS[c] for c in m.group(1).lower()) + (m.group(2) or "")
    return _ACRO.sub(sub, text)


def normalize(text: str) -> str:
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    text = re.sub(r"(\d)\s*([ap])\.?\s?m\.?(?=\W|$)", lambda m: f"{m.group(1)} {_LETTERS[m.group(2).lower()]} em", text, flags=re.I)
    text = re.sub(r"\b((?:[A-Z]\.){2,})(?!\w)", lambda m: m.group(1).replace(".", ""), text)  # U.S. -> US (then spelled)
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", lambda m: m.group(1) + " " + ("o'clock" if m.group(2) == "00" else ("oh " + m.group(2)[1] if m.group(2)[0] == "0" else m.group(2))), text)  # 7:30 -> 7 30
    text = expand_acronyms(text)
    text = expand_numbers(text)
    text = "".join(_ASCII_EXTRA.get(c, c) for c in text)  # (2026-11-16: Łódź -> lodz, Straße -> strasse, not odz / strae)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", text).strip()


_ASCII_EXTRA = {"ł": "l", "Ł": "L", "ß": "ss", "ẞ": "SS", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D", "æ": "ae", "Æ": "AE",
                "œ": "oe", "Œ": "OE", "þ": "th", "Þ": "Th", "ð": "d", "Ð": "D", "ı": "i"}


# languages detect_lang may answer with: French / Korean only once a model trained on them is what the site runs (until
# then a French / Korean-looking line is read as before -- an untrained language slot would sound broken)
ACTIVE_LANGS = {"en", "ja", "es", "nl", "pl", "zh", "de"}


def _detect_lang_rules(text: str) -> str:
    # (the old rules, 2026-10..11: the fallback when configs/langdet.json is missing; see _detect_lang below)
    """"ja" if the text has kana, "zh" if it has Chinese characters but no kana, "pl" if it has Polish letters,
    "es" if it has ñ ¿ ¡ or several Spanish accents, else "en" (Dutch can't be told from English by its letters:
    pick it explicitly)."""
    if any("぀" <= c <= "ヿ" for c in text):
        return "ja"
    if any("\u0400" <= c <= "\u04ff" for c in text):  # Cyrillic (only Ukrainian is trained)
        return "uk"
    if any("一" <= c <= "鿿" for c in text):
        return "zh"
    if any(c in "ąćęłńśźżĄĆĘŁŃŚŹŻ" for c in text):
        return "pl"
    if any("가" <= c <= "힣" or "ᄀ" <= c <= "ᇿ" or "㄰" <= c <= "㆏" for c in text):
        return "ko"
    if any(c in "äßÄ" for c in text):
        return "de"
    if any(c in "èêçàâîôûùœæÈÊÇÀÂÎÔÛŒ" for c in text):  # (letters Spanish / Polish / German never use; Dutch rarely)
        return "fr"
    t = text.lower()
    if any(c in t for c in "ñ¿¡") or sum(t.count(c) for c in "áéíóú") >= 2:
        return "es"
    # same Latin letters: decide by frequent function words (unambiguous ones only)
    words = re.findall(r"[a-zà-ÿ']+", t)
    if words:
        score = {k: sum(w in v for w in words) for k, v in _FUNC_WORDS.items()}
        best = max(score, key=score.get)
        if best != "en" and score[best] >= 2 and score[best] > score["en"]:
            return best
    return "en"


_FUNC_WORDS = {
    "en": set("the and to of you it that for on with this are be have not but what they he she my your so just "
              "do can will would there their them his her from at by an or if about when which who were been had "
              "has our me him how all".split()),
    "nl": set("de het een en van ik je jij niet dat dit deze ook als maar met voor zijn er nog wel naar dan om "
              "aan bij uit kan heb hebben wordt zo geen want toch even mijn jouw onze graag goed hij zij ze wat "
              "niets iets heel veel wij jullie hun mij jou hoe waar wanneer waarom omdat kunnen moet moeten zou "
              "zal wil willen gaat ga gaan komt komen heeft hebt worden werd ons zich hem nu hier daar dus nee ja "
              "zeer meer ben bent zit staat ligt weet denk zeg".split()),
    "es": set("el la los las de que y en un una es por para con no se lo del al como más mas pero su muy yo "
              "está esta estoy qué que porque también tambien hay ser fue sus esto eso nos mi tu".split()),
    "pl": set("nie się sie jest że ze do jak co ale tak już juz mi jestem mam być byc tylko czy na to ja ty on "
              "ona my wy oni od po za dla przez bardzo może moze teraz tutaj tam".split()),
    "de": set("der die das und ist nicht ich du er sie wir ihr es ein eine einen dem den des zu mit auf für fur "
              "von im sind war waren haben hat habe wird werden kann können konnen auch noch nur schon sehr "
              "aber oder wenn weil dass daß wie was wer wo warum mein dein sein unser euer kein keine doch mal "
              "jetzt hier dort heute gut ja nein bitte danke".split()),
    # French (2026-11-08): unambiguous function words only ("de", "en", "est" are shared with Dutch / Spanish)
    "fr": set("le les une des du et il elle je tu nous vous ils elles pas qui dans pour sur avec ce cette ces c'est "
              "j'ai suis sont était etait être etre avoir fait mais ou donc très tres aussi bien oui merci "
              "l'on qu'il n'est d'un d'une au aux leur leurs mon ton son ma ta sa mes tes ses notre votre".split()),
}


# ---------------------------------------------------------------- language detection (Solver 2026-11-17, you: "please
# get those fixed"): scripts counted, Latin languages by a letter n-gram table from our own transcripts
# (configs/langdet.json); short Dutch / French lines were 10-12 % right, now 58-92 %. notes_solver/langdet/REPORT.txt
LATIN = ("en", "es", "nl", "pl", "de", "fr")
_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")
_TABLE = {}
_WORD_CLIP = 8.0  # a word can put a language at most this far (in log-probability) behind the word's best language
_JA_ONLY = set("々図渋駅気円売読楽広県鉄済対戦労歩黒絵経変実単続総関発顔働込峠畑辻価仮伝沢桜悪圧営栄応覚観帰権検験雑歯収従処焼乗帯団弾遅庁聴"
               "転闘悩脳廃拝髪抜払辺豊毎満訳薬様頼覧両緑涙霊齢歴録姫巻")


def langs_of(n_langs: int) -> set:
    """The languages a model with cfg.n_langs language slots was trained on (8: no fr / ko; 9: + fr, ko; 10: + uk)."""
    out = {"en", "ja", "es", "nl", "pl", "zh", "de"}
    if n_langs >= 9:
        out |= {"fr", "ko"}
    if n_langs >= 10:
        out |= {"uk"}
    return out


def _words(text: str):
    return _WORD.findall(unicodedata.normalize("NFC", text).lower().replace("’", "'"))


def _grams(text: str):
    """(used to build the table) every letter piece of every word"""
    for w in _words(text):
        s = f"^{w}$"
        for n in (1, 2, 3):
            for i in range(len(s) - n + 1):
                yield s[i:i + n]


def _table(path=None):
    if "g" not in _TABLE:
        f = Path(path) if path else Path(__file__).resolve().parent.parent / "configs" / "langdet.json"
        if f.exists():
            t = json.loads(f.read_text(encoding="utf-8"))
            _TABLE["g"], _TABLE["langs"], _TABLE["bonus"] = t["grams"], t["langs"], t.get("bonus", {})
        else:
            _TABLE["g"] = None
    return _TABLE


def _script(text: str) -> str:
    """'ko' / 'ja' / 'zh' / 'uk' / 'latin' by counting letters of each script."""
    kana = han = hangul = cyr = latin = ja_only = 0
    for c in text:
        if ("぀" <= c <= "ヿ" and c not in "・ー゠") or "ｦ" <= c <= "ﾟ":
            kana += 1
        elif "一" <= c <= "鿿" or c == "々":
            han += 1
            ja_only += c in _JA_ONLY
        elif "가" <= c <= "힣" or "ᄀ" <= c <= "ᇿ" or "㄰" <= c <= "㆏":
            hangul += 1
        elif "Ѐ" <= c <= "ӿ":
            cyr += 1
        elif c.isalpha():
            latin += 1
    east = kana + han + hangul
    if east and latin <= 4 * east:  # (a Korean / Japanese / Chinese line with a few Latin words stays what it is)
        if hangul:
            return "ko"
        if kana >= 2 or (kana == 1 and han < 5) or ja_only:
            return "ja"
        return "zh"
    if cyr and 2 * cyr >= latin:
        return "uk"
    return "latin"


def _detect_lang(text: str, allowed=None, fallback=None, table_path=None) -> str:
    # (allowed None = every language a model on the site knows today: langs_of(9); no model reads uk yet)
    allowed = langs_of(9) if allowed is None else allowed
    fallback = fallback or _detect_lang_rules
    """The language of a line. allowed: a set of language codes to choose from (None = all)."""
    ok = (lambda l: allowed is None or l in allowed)
    sc = _script(text)
    if sc != "latin":
        if ok(sc):
            return sc
        if sc == "ja" and ok("zh") and not any("぀" <= c <= "ヿ" for c in text):
            return "zh"
        if sc == "zh" and ok("ja"):
            return "ja"
    t = _table(table_path)
    if t["g"] is None:
        got = fallback(text) if fallback else "en"
        return got if ok(got) else "en"
    langs, G = t["langs"], t["g"]
    idx = [i for i, l in enumerate(langs) if ok(l)]
    if not idx:
        return "en"
    tot, nw = [0.0] * len(langs), 0
    for w in _words(text):
        s, n = [0.0] * len(langs), 0
        p = f"^{w}$"
        for k in (1, 2, 3):
            for i in range(len(p) - k + 1):
                v = G.get(p[i:i + k])
                if v is not None:
                    n += 1
                    for j in idx:
                        s[j] += v[j]
        if n:
            nw += 1
            m = max(s[j] for j in idx)
            for j in idx:
                tot[j] += max(s[j] - m, -_WORD_CLIP)
    if not nw:
        return "en"
    return langs[max(idx, key=lambda j: tot[j] + t["bonus"].get(langs[j], 0.0) * nw)]


def detect_lang(text: str, allowed=None) -> str:
    """The language of a line among `allowed` (the speaking model's languages: langs_of(model.cfg.n_langs));
    without it, ACTIVE_LANGS (what every model on the site reads)."""
    return _detect_lang(text, ACTIVE_LANGS if allowed is None else allowed)

# ---------------------------------------------------------------- Spanish
def _num_es(m: re.Match) -> str:
    from num2words import num2words

    whole, _, frac = m.group(0).replace(".", ",").partition(",")
    s = num2words(int(whole), lang="es")
    return f"{s} coma {num2words(int(frac), lang='es')}" if frac else s


def normalize_es(text: str) -> str:
    """Like normalize(), but keeps Spanish letters / ¿ ¡ and spells numbers in Spanish."""
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    text = re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", text)  # 1.000 -> 1000
    text = re.sub(r"(\d+)\s*%", r"\1 por ciento", text).replace("&", " y ")
    text = re.sub(r"\d+(?:[.,]\d+)?", _num_es, text)
    text = unicodedata.normalize("NFC", text).lower()
    out = []
    for c in text:
        out.append(c if c in _ES else unicodedata.normalize("NFKD", c).encode("ascii", "ignore").decode())
    return re.sub(r"\s+", " ", "".join(out)).strip()


# ---------------------------------------------------------------- Japanese
_SMALL = set("ァィゥェォャュョヮ")
_JA_PUNCT = {"、": ",", "，": ",", ",": ",", "。": ".", "．": ".", ".": ".", "！": "!", "!": "!",
             "？": "?", "?": "?", "…": ".", "‥": "."}
_MORA_END = set("aiueoAIUEO") | {"N", "cl"}


def _feat(regex, label):
    m = re.search(regex, label)
    return int(m.group(1)) if m and m.group(1).lstrip("-").isdigit() else None


def _accent_marks(labels):
    """One entry per mora: '' or a mark ('↑', '↓', '|') placed after that mora."""
    marks = []
    for n, lab in enumerate(labels):
        p3 = re.search(r"\-(.*?)\+", lab).group(1)
        if p3 not in _MORA_END:
            continue
        a1 = _feat(r"/A:([0-9\-]+)\+", lab)
        a2 = _feat(r"\+(\d+)\+", lab)
        a3 = _feat(r"\+(\d+)/", lab)
        f1 = _feat(r"/F:(\d+)_", lab)
        nxt = labels[n + 1] if n + 1 < len(labels) else ""
        a2_next = _feat(r"\+(\d+)\+", nxt)
        p3_next = re.search(r"\-(.*?)\+", nxt).group(1) if nxt else "sil"
        mark = ""
        if a3 == 1 and (a2_next == 1 or p3_next in ("pau", "sil")):
            mark = "|"
        elif a1 == 0 and a2_next == (a2 or 0) + 1 and a2 != f1:
            mark = "↓"
        elif a2 == 1 and a2_next == 2:
            mark = "↑"
        marks.append(mark)
    return marks


def _moras(kana: str):
    out = []
    for c in kana:
        if out and (c in _SMALL or c == "’"):
            out[-1] += c
        else:
            out.append(c)
    return out


def japanese_to_kana(text: str, accent: bool = True) -> str:
    """'雨が降ってきた。' -> 'ア↓メガ|フッテキ’↓タ.' (katakana + accent marks + ASCII punctuation)."""
    import pyopenjtalk

    text = unicodedata.normalize("NFKC", text)
    njd = pyopenjtalk.run_frontend(text)
    orig_punct = [_JA_PUNCT[c] for c in text if c in _JA_PUNCT]
    pieces, kana = [], ""
    for w in njd:
        pron = w["pron"].replace("’", "’")
        if pron in _JA_PUNCT or w.get("pos") == "記号":
            pieces.append(("p", _JA_PUNCT.get(pron, ",")))
        else:
            pieces.append(("k", pron))
            kana += pron
    if len([p for p in pieces if p[0] == "p"]) == len(orig_punct):  # keep sentence ends (。 -> '.')
        it = iter(orig_punct)
        pieces = [(t, next(it)) if t == "p" else (t, v) for t, v in pieces]
    marks = _accent_marks(pyopenjtalk.make_label(njd)) if accent else []
    moras_all = _moras(kana)
    use_marks = accent and len(marks) == sum(1 for m in moras_all if m not in ("’",))
    out, mi = [], 0
    for t, v in pieces:
        if t == "p":
            out.append(v)
            continue
        for m in _moras(v):
            out.append(m)
            if use_marks and mi < len(marks):
                out.append(marks[mi])
            mi += 1
    s = "".join(out)
    return re.sub(r"\|(?=[.,!?]|$)", "", s)  # a phrase boundary before punctuation is implied


# Japanese lines with English words, each part in its own script: on only while the Synthesizer encodes a line
# (training data is encoded as before). Per THREAD (code review 2026-10-29: a module flag, so the website's warm-up
# thread could switch it off under a visitor's line): mixed_script(True / False)
_MIXED = __import__("threading").local()


def mixed_script(on: bool):
    _MIXED.on = bool(on)


_LATIN_RUN = re.compile(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'\- ]*[A-Za-zÀ-ÿ0-9]|[A-Za-zÀ-ÿ]")


def mixed_symbols(text: str) -> str:
    """A Japanese line with English / Spanish words in it: Japanese stretches -> kana (pyopenjtalk), Latin
    stretches stay letters (as in English / Spanish lines), instead of reading the whole line as Japanese."""
    out, pos = [], 0
    for m in _LATIN_RUN.finditer(text):
        if m.start() > pos and text[pos:m.start()].strip():
            out.append(japanese_to_kana(text[pos:m.start()]))
        w = m.group(0)
        out.append((normalize_es(w) if detect_lang(w) == "es" else normalize(w)).lower())
        pos = m.end()
    if text[pos:].strip():
        out.append(japanese_to_kana(text[pos:]))
    return " ".join(o.strip() for o in out if o.strip())


_PL_MONTHS = "stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|września|października|listopada|grudnia"


def _pl_ordinal(n: int, case: str) -> str:
    """Polish ordinal, masculine, in the genitive ('gen': trzydziestego pierwszego) or locative ('loc': dwudziestym).
    In years only the last part is ordinal (tysiąc dziewięćset dziewięćdziesiątego dziewiątego); round hundreds /
    thousands are one ordinal word (dwutysięcznego)."""
    from num2words import num2words
    end = {"gen": "ego", "loc": "ym"}[case]

    def inflect(w):  # pierwszy -> pierwszego / pierwszym; drugi -> drugiego / drugim
        if w.endswith("y"):
            return w[:-1] + end
        if w.endswith("i"):
            return w + "ego" if case == "gen" else w + "m"  # drugi -> drugiego / drugim, trzeci -> trzeciego
        return w
    if n % 100 and n >= 100:
        return f"{num2words(n - n % 100, lang='pl')} " + " ".join(inflect(w) for w in num2words(n % 100, lang="pl", to="ordinal").split())
    words = num2words(n, lang="pl", to="ordinal").split()
    return " ".join(words[:-1] + [inflect(words[-1])]) if n >= 100 else " ".join(inflect(w) for w in words)


def _pl_ordinals(text: str) -> str:
    """Dates and years read as Polish ordinals in the right case (cardinals everywhere else):
    31 grudnia -> trzydziestego pierwszego grudnia; w 1999 roku / w 1999 r. -> w ... dziewięćdziesiątym dziewiątym roku;
    (31 grudnia) 1999 (roku) / 1999 roku -> ... dziewięćdziesiątego dziewiątego (roku)."""
    text = re.sub(r"\b(\d{4})\s*r\.(?=\s|$|[,;:])", r"\1 roku", text)
    text = re.sub(rf"\b(\d{{1,2}})\.?(\s+(?:{_PL_MONTHS})\b)", lambda m: _pl_ordinal(int(m.group(1)), "gen") + m.group(2), text, flags=re.I)
    text = re.sub(r"\b(w|we)\s+(\d{3,4})(\s+roku)\b", lambda m: f"{m.group(1)} {_pl_ordinal(int(m.group(2)), 'loc')}{m.group(3)}", text, flags=re.I)
    text = re.sub(rf"((?:{_PL_MONTHS})\s+)(\d{{4}})\b", lambda m: m.group(1) + _pl_ordinal(int(m.group(2)), "gen"), text, flags=re.I)
    text = re.sub(r"\b(\d{3,4})(\s+roku)\b", lambda m: _pl_ordinal(int(m.group(1)), "gen") + m.group(2), text)
    return text


# Polish cardinals in the case their noun asks for (a tester 2026-10-26: "w 7 językach" was read "w siedem językach",
# like saying "I'm in I room"): (genitive = locative, dative, instrumental) of each part; 1 stays "jeden" in compounds
_PL_CASE_ONES = {2: ("dwóch", "dwóm", "dwoma"), 3: ("trzech", "trzem", "trzema"), 4: ("czterech", "czterem", "czterema"),
                 5: ("pięciu", "pięciu", "pięcioma"), 6: ("sześciu", "sześciu", "sześcioma"),
                 7: ("siedmiu", "siedmiu", "siedmioma"), 8: ("ośmiu", "ośmiu", "ośmioma"),
                 9: ("dziewięciu", "dziewięciu", "dziewięcioma")}
_PL_CASE_STEMS = {10: "dziesięci", 11: "jedenast", 12: "dwunast", 13: "trzynast", 14: "czternast", 15: "piętnast",
                  16: "szesnast", 17: "siedemnast", 18: "osiemnast", 19: "dziewiętnast", 20: "dwudziest",
                  30: "trzydziest", 40: "czterdziest", 50: "pięćdziesięci", 60: "sześćdziesięci", 70: "siedemdziesięci",
                  80: "osiemdziesięci", 90: "dziewięćdziesięci"}  # (+ "u" / "oma")
_PL_CASE_HUNDREDS = {100: ("stu", "stu", "stoma"), 200: ("dwustu", "dwustu", "dwustoma"), 300: ("trzystu", "trzystu", "trzystoma"),
                     400: ("czterystu", "czterystu", "czterystoma"), 500: ("pięciuset",) * 3, 600: ("sześciuset",) * 3,
                     700: ("siedmiuset",) * 3, 800: ("ośmiuset",) * 3, 900: ("dziewięciuset",) * 3}
_PL_CASE_IDX = {"gen": 0, "loc": 0, "dat": 1, "inst": 2}
_PL_THOUSAND = {"gen": ("tysiąca", "tysięcy"), "loc": ("tysiącu", "tysiącach"), "dat": ("tysiącowi", "tysiącom"),
                "inst": ("tysiącem", "tysiącami")}
_PL_GEN_PREP = {"od", "do", "bez", "dla", "około", "koło", "u", "według", "ponad", "powyżej", "poniżej", "sprzed",
                "spośród", "wśród", "podczas", "oprócz", "prócz", "obok", "zamiast", "wokół", "blisko", "z", "ze"}
_PL_DAT_PREP = {"dzięki", "ku", "przeciw", "przeciwko", "wbrew"}


def _pl_case_words(n: int, case: str) -> str | None:
    """2 <= n < 1 000 000 in gen / loc / dat / inst; None outside that (the nominative from num2words then)."""
    i = _PL_CASE_IDX[case]

    def under_1000(m: int) -> list:
        out = []
        if m >= 100:
            out.append(_PL_CASE_HUNDREDS[m // 100 * 100][i])
            m %= 100
        if 10 <= m < 20:
            out.append(_PL_CASE_STEMS[m] + ("oma" if i == 2 else "u"))
            return out
        if m >= 20:
            out.append(_PL_CASE_STEMS[m // 10 * 10] + ("oma" if i == 2 else "u"))
            m %= 10
        if m == 1:
            out.append("jeden")  # (in compounds: dwudziestu jeden)
        elif m:
            out.append(_PL_CASE_ONES[m][i])
        return out
    if not 2 <= n < 1_000_000:
        return None
    k, r = divmod(n, 1000)
    words = []
    if k == 1:
        words.append(_PL_THOUSAND[case][0])
    elif k:
        words += under_1000(k) + [_PL_THOUSAND[case][1] if case != "loc" else "tysiącach"]
    if r:
        words += under_1000(r)
    return " ".join(words)


def _pl_cases(text: str) -> str:
    """Numbers whose noun shows the case: '-ach / -ech' locative, '-ami / -mi' instrumental (adjectives in between are
    skipped); else the preposition before the number: genitive (od, do, bez, ...) or dative (dzięki, ku, ...)."""
    def sub(m):
        n = int(m.group(2))
        if n < 2:
            return m.group(0)
        case = None
        for w in re.findall(r"[^\W\d_]+", m.group(3) or "")[:3]:
            lw = w.lower()
            if lw.endswith(("ach", "ech")):
                case = "loc"
            elif len(lw) > 3 and lw.endswith(("ami", "mi")):
                case = "inst"
            elif lw.endswith(("ych", "ich")):
                continue  # (an adjective: look at the next word)
            break
        pre = (m.group(1) or "").strip().lower()
        if case is None and pre in _PL_GEN_PREP:
            case = "gen"
        elif case is None and pre in _PL_DAT_PREP:
            case = "dat"
        words = _pl_case_words(n, case) if case else None
        return f"{m.group(1) or ''}{words}" if words else m.group(0)
    return re.sub(r"(\b[^\W\d_]+\s+)?(?<![\d.,])\b(\d{1,6})\b(?![.,]\d)(?=((?:\s+[^\W\d_]+){0,3}))", sub, text)


def _say_num(digits: str, lang: str) -> str:
    """A number in words; a run too long for num2words (data has strings of 100+ digits) digit by digit."""
    from num2words import num2words
    if len(digits) > 15:
        return " ".join(num2words(int(d), lang=lang) for d in digits[:40])
    return num2words(int(digits), lang=lang)


def normalize_fr(text: str) -> str:
    """French: its letters kept (the shared accents + _FR), numbers spelled in French, « » as quotes."""
    from num2words import num2words
    for a, b in {**_REPLACE, "«": '"', "»": '"', " ": " ", " ": " "}.items():
        text = text.replace(a, b)
    text = re.sub(r"(\d+)\s*%", r"\1 pour cent", text)
    text = re.sub(r"\b(\d+)(?:er|re|ère|e|ème|eme)\b", lambda m: num2words(int(m.group(1)), lang="fr", to="ordinal"), text)
    text = re.sub(r"\d+", lambda m: _say_num(m.group(0), "fr"), text)
    text = unicodedata.normalize("NFC", text).lower()
    keep = set(_NLPL) | set(_ES) | set(_FR) | {"ä", "ö", "ü"}
    out = [c if c in keep else unicodedata.normalize("NFKD", c).encode("ascii", "ignore").decode() for c in text]
    return re.sub(r"\s+", " ", "".join(out)).strip()


def normalize_ko(text: str) -> str:
    """Korean: numbers in Sino-Korean words (num2words ko), then every Hangul syllable split into its conjoining jamo
    (NFD); Latin letters (loanwords, names) kept lowercase, other scripts dropped."""
    from num2words import num2words
    for a, b in {**_REPLACE, " ": " "}.items():
        text = text.replace(a, b)
    text = re.sub(r"(\d+)\s*%", lambda m: _say_num(m.group(1), "ko") + " 퍼센트", text)
    text = re.sub(r"\d+", lambda m: _say_num(m.group(0), "ko"), text)
    text = unicodedata.normalize("NFD", text).lower()  # (syllables -> jamo; the decomposition is exact for Hangul)
    ok = set(_KO) | set(_CHARS)
    out = []
    for c in text:
        if c in ok:
            out.append(c)
        elif "㄰" <= c <= "㆏":  # (lone compatibility jamo: their conjoining form)
            out.append(unicodedata.normalize("NFKD", c)[:1] if unicodedata.normalize("NFKD", c)[:1] in ok else " ")
        elif c in "，。！？":
            out.append({"，": ",", "。": ".", "！": "!", "？": "?"}[c])
        else:  # (2026-11-16: a dropped symbol between words left them glued, test@example -> testexample)
            out.append(" ")
    return re.sub(r"\s+", " ", "".join(out)).strip()


# Russian-only letters a Ukrainian line may still contain (quotes, names): their nearest Ukrainian letter
_UK_FROM_RU = {"ы": "и", "э": "е", "ё": "йо", "ъ": "'"}


def normalize_uk(text: str) -> str:
    """Ukrainian: Cyrillic letters kept (lowercase), the apostrophe (ʼ ’ ') as "'", numbers spelled in Ukrainian
    (num2words uk), % as "відсотків"; Latin letters (names, brands) kept lowercase."""
    for a, b in {**_REPLACE, "ʼ": "'", "`": "'", "´": "'", "«": '"', "»": '"', "\u00a0": " "}.items():
        text = text.replace(a, b)
    def _pct(m):  # 1 / 21 відсоток, 2-4 / 22-24 відсотки, else відсотків
        n = int(m.group(1))
        w = "відсоток" if n % 10 == 1 and n % 100 != 11 else ("відсотки" if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14 else "відсотків")
        return _say_num(m.group(1), "uk") + " " + w
    text = re.sub(r"(\d+)\s*%", _pct, text)
    text = re.sub(r"\d+", lambda m: _say_num(m.group(0), "uk"), text)
    text = unicodedata.normalize("NFC", text).lower()
    ok = set(_UK) | set(_CHARS)
    out = []
    for c in text:
        if c in ok:
            out.append(c)
        elif c in _UK_FROM_RU:
            out.append(_UK_FROM_RU[c])
        elif unicodedata.category(c) == "Mn":  # (stress marks: за́мок -> замок, 2026-11-16)
            continue
        else:
            d = unicodedata.normalize("NFKD", c).encode("ascii", "ignore").decode()
            out.append(d if d else " ")
    return re.sub(r"\s+", " ", "".join(out)).strip()


def normalize_nlpl(text: str, lang: str) -> str:
    """Dutch / Polish / German: keep the language's letters, spell numbers in that language."""
    from num2words import num2words
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    text = re.sub(r"(\d+)\s*%", r"\1 prozent" if lang == "de" else r"\1 procent", text)
    if lang == "pl":
        text = _pl_cases(_pl_ordinals(text))
    elif lang == "nl":  # 31e / 1ste / 2de -> eenendertigste / eerste / tweede
        text = re.sub(r"\b(\d+)(?:e|ste|de)\b", lambda m: num2words(int(m.group(1)), lang="nl", to="ordinal"), text)
    text = re.sub(r"\d+", lambda m: num2words(int(m.group(0)), lang=lang), text)
    text = unicodedata.normalize("NFC", text).lower()
    if lang == "de":  # German ö / ü: their own symbols
        text = text.replace("ö", _DE_OE).replace("ü", _DE_UE).replace("Ö", _DE_OE).replace("Ü", _DE_UE)
    keep = set(_NLPL) | set(_ES) | set(_DE)
    out = [c if c in keep else unicodedata.normalize("NFKD", c).encode("ascii", "ignore").decode() for c in text]
    return re.sub(r"\s+", " ", "".join(out)).strip()


_ZH_PUNCT = {"，": ",", "。": ".", "！": "!", "？": "?", "、": ",", "；": ";", "：": ":", "（": "(", "）": ")",
             "“": '"', "”": '"', "《": '"', "》": '"', "…": "..."}


def chinese_to_pinyin(text: str) -> str:
    """'你好，世界。' -> 'ni3 hao3, shi4 jie4.' (tone 5 = neutral); Latin words and numbers pass through
    (numbers are first read the Chinese way)."""
    from pypinyin import Style, lazy_pinyin
    text = re.sub(r"\d+", lambda m: _zh_number(int(m.group(0))), text)
    out = []
    for p in lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True, errors=lambda x: list(x)):
        p = _ZH_PUNCT.get(p, p)
        out.append(p.replace("ü", "v").lower())
    s_ = " ".join(out)
    s_ = re.sub(r"\s+([,.!?;:)\"])", r"\1", s_)
    return re.sub(r"\s+", " ", s_).strip()


def _zh_number(n: int) -> str:
    d = "零一二三四五六七八九"
    if n < 10:
        return d[n]
    if n < 100:
        return ("" if n // 10 == 1 else d[n // 10]) + "十" + (d[n % 10] if n % 10 else "")
    return "".join(d[int(c)] for c in str(n))  # longer numbers digit by digit


def to_symbols(text: str, lang: str = "en") -> str:
    # (you 2026-11-16 "please get those fixed"): numbers, money, times, dates, ranges and phone numbers the way each
    # language says them, before the language's own normalizer (dacts/numerals.py)
    from .numerals import prenum
    s = _to_symbols(prenum(text, lang if lang in LANGS else "en"), lang)
    # a symbol the model has no token for, between two letters, would be dropped and glue the words ("km/h" -> "kmh",
    # "3/4" -> "dreivier"): it becomes a space instead (2026-11-16, the Solver's text audit)
    return re.sub(r"(?<=\S)[^\s](?=\S)", lambda m: m.group(0) if m.group(0) in _VOCAB or m.group(0) in "’" else " ", s) \
        if any(c not in _VOCAB and not c.isspace() for c in s) else s


def _to_symbols(text: str, lang: str = "en") -> str:
    if lang == "ja":
        if getattr(_MIXED, "on", False) and _LATIN_RUN.search(text) and len(re.findall(r"[A-Za-z]", text)) >= 2:
            return mixed_symbols(text)
        return japanese_to_kana(text)
    if lang == "zh":
        return chinese_to_pinyin(text)
    if lang in ("nl", "pl", "de"):
        return normalize_nlpl(text, lang)
    if lang == "fr":
        return normalize_fr(text)
    if lang == "ko":
        return normalize_ko(text)
    if lang == "uk":
        return normalize_uk(text)
    return normalize_es(text) if lang == "es" else normalize(text)


def encode(text: str, add_blank: bool = True, lang: str = "en") -> list[int]:
    ids = [_VOCAB[c] for c in to_symbols(text, lang) if c in _VOCAB]
    if add_blank:
        out = [BLANK] * (2 * len(ids) + 1)
        out[1::2] = ids
        return out
    return ids


# ---------------------------------------------------------------- word emphasis
_SEP = {" ", "|"} | set(".,!?-:;\"()¿¡")  # word boundaries (Japanese: accent-phrase marks)
_INV = {i: c for c, i in _VOCAB.items()}


def word_ids(ids: list[int]) -> list[int]:
    """Token ids -> word index per token (-1: separators, padding, blanks between words)."""
    out, w, in_word = [], -1, False
    for i in ids:
        c = _INV.get(i)
        if i == BLANK or c is None:
            out.append(None)
            continue
        if c in _SEP:
            in_word = False
            out.append(-1)
        else:
            if not in_word:
                w, in_word = w + 1, True
            out.append(w)
    for k, v in enumerate(out):  # blanks inside a word belong to it
        if v is None:
            prev = next((out[j] for j in range(k - 1, -1, -1) if out[j] is not None), -1)
            nxt = next((out[j] for j in range(k + 1, len(out)) if out[j] is not None), -1)
            out[k] = prev if prev == nxt and prev >= 0 else -2
    return [-1 if v in (None, -2) else v for v in out]


_PAUSE = set(".,!?:;")  # the marks a reader pauses at (hyphens, quotes and brackets don't count)


def gap_ids(ids: list[int], wid: list[int] | None = None) -> tuple[list[int], list[str], list[int]]:
    """The pauses between words: every run of non-word tokens (blanks, spaces, punctuation) that holds a pause mark
    and follows a word. -> (gap index per token or -1, each gap's marks, the word each gap follows)."""
    wid = word_ids(ids) if wid is None else wid
    out, marks, after = [-1] * len(ids), [], []
    k, i = -1, 0
    while i < len(ids):
        if wid[i] >= 0:
            k, i = wid[i], i + 1
            continue
        j = i
        while j < len(ids) and wid[j] < 0:
            j += 1
        m = "".join(c for c in (_INV.get(t) for t in ids[i:j]) if c in _PAUSE)
        if m and k >= 0:
            out[i:j] = [len(marks)] * (j - i)
            marks.append(m)
            after.append(k)
        i = j
    return out, marks, after


_MARK = re.compile(r"\[([^\]]+)\]\s*:\s*([0-9]*\.?[0-9]+)")


def strip_marks(text: str) -> str:
    """'a [good]:2.0 day' -> 'a good day'."""
    return _MARK.sub(lambda m: m.group(1), text)


_VOWEL_CHARS = set("aeiouyAEIOUYáéíóúàèìòùäëïöüâêîôûąęǫãõåøæœ") | {chr(c) for c in range(0x3041, 0x3097)} |     {chr(c) for c in range(0x30A1, 0x30FB)} | {"ー"}


def vowel_flags(ids: list[int]) -> list[bool]:
    """Per token: does it carry a vowel (Latin / pinyin vowels, kana morae, the long mark)? Sung words hold these."""
    return [_INV.get(i, "") in _VOWEL_CHARS for i in ids]


def song_words(text: str, lang: str | None = None) -> list[str]:
    """The words a melody is written for, matching word_ids() one to one: the plain split when it agrees (keeps the
    spelling the composer learned), else the tokenizer's own words (Chinese: one per character; Japanese: phrases)."""
    lang = lang or detect_lang(text)
    ids = encode(text, False, lang); wid = word_ids(ids); out: dict[int, str] = {}
    for i, w in zip(ids, wid):
        if w is not None and w >= 0:
            out[w] = out.get(w, "") + _INV.get(i, "")
    split = text.split()
    return split if len(split) == len(out) else [out[k] for k in sorted(out)]


def _carry_marks(src: list[int], vals: list, dst: list[int]) -> list:
    """Emphasis values of tokens `src` -> the same text read as `dst` (a different reading of it): aligned tokens keep
    theirs; a changed stretch takes its marked value when that is one value; letters added inside a marked word (between
    two tokens of the same value) take it too. Separators never carry one."""
    import difflib
    sep = {_VOCAB[c] for c in _SEP if c in _VOCAB}
    out = [None] * len(dst)
    added = []
    for op, a0, a1, b0, b1 in difflib.SequenceMatcher(None, src, dst, autojunk=False).get_opcodes():
        if op == "equal":
            out[b0:b1] = vals[a0:a1]
        elif op == "replace":
            vs = [vals[j] for j in range(a0, a1) if src[j] not in sep]
            if vs and vs[0] is not None and all(v == vs[0] for v in vs):
                out[b0:b1] = [vs[0]] * (b1 - b0)
        elif op == "insert":
            added += range(b0, b1)
    for k in added:  # inside a marked word: both neighbours (no separator between) carry the same value
        a = next((j for j in range(k - 1, -1, -1) if out[j] is not None or dst[j] in sep), None)
        b = next((j for j in range(k + 1, len(dst)) if out[j] is not None or dst[j] in sep), None)
        if a is not None and b is not None and dst[a] not in sep and dst[b] not in sep and out[a] == out[b]:
            out[k] = out[a]
    return [None if t in sep else v for t, v in zip(dst, out)]


def encode_marked(text: str, add_blank: bool = True, lang: str = "en"):
    """Text with emphasis marks '[word]:2.0' (0 de-emphasised .. 1 normal .. 2 emphasised) -> (ids, values)
    where values[k] is the emphasis of token k or None (unspecified)."""
    syms, vals, pos = [], [], 0
    for m in list(_MARK.finditer(text)) + [None]:
        seg = text[pos: m.start() if m else len(text)]
        # (marks go up to 3 on the site: the encoder gets <= 2, model.sample stretches + pauses the rest)
        parts = [(seg, None)] + ([(m.group(1), min(3.0, max(0.0, float(m.group(2)))))] if m else [])
        for t, v in parts:
            if not t.strip():
                if t and syms and syms[-1] != " " and lang != "ja":
                    syms.append(" "), vals.append(None)
                continue
            s = to_symbols(t, lang)
            if lang != "ja" and t[:1].isspace() and syms and syms[-1] != " ":
                syms.append(" "), vals.append(None)
            for c in s:
                syms.append(c), vals.append(v if c not in _SEP else None)
            if lang != "ja" and t[-1:].isspace():
                syms.append(" "), vals.append(None)
        pos = m.end() if m else len(text)
    while syms and syms[-1] == " ":
        syms.pop(), vals.pop()
    keep = [(_VOCAB[c], v) for c, v in zip(syms, vals) if c in _VOCAB]
    ids, vs = [k for k, _ in keep], [v for _, v in keep]
    # the tokens are the PLAIN text's own; the marks only bring their values (you 2026-10-31: "a lot of words are just
    # not unstressed or stressed"). Each marked span read on its own changed the reading: Chinese syllables lost their
    # spaces (words ran together), a Japanese kanji got its standalone reading ('[公]:2[園]:2' = オーヤケエン, not
    # コーエン), '[Dr.]:2' lost 'doctor'. Values move over by aligning the two readings.
    plain = [_VOCAB[c] for c in to_symbols(strip_marks(text), lang) if c in _VOCAB]
    if plain != ids:
        ids, vs = plain, _carry_marks(ids, vs, plain)
    if not add_blank:
        return ids, vs
    out_i, out_v = [BLANK] * (2 * len(ids) + 1), [None] * (2 * len(ids) + 1)
    out_i[1::2], out_v[1::2] = ids, vs
    for k in range(0, len(out_v), 2):  # a blank between two tokens of the same marked span gets its value
        a = out_v[k - 1] if k > 0 else None
        b = out_v[k + 1] if k + 1 < len(out_v) else None
        out_v[k] = a if a is not None and a == b else None
    return out_i, out_v
