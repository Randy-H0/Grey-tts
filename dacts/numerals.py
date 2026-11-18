"""Numbers, money, times and dates read the way each language says them (you 2026-11-16: "please get those fixed"; the
Solver's text audit found e.g. de "1.000 Euro" -> "eins.null euro", fr "10 000" -> "dix zéro", es "3,05" -> "tres coma
cinco", zh "150元" read digit by digit, ko "2시" -> "이시", en "555-1234" -> "five hundred and fifty-five-twelve...").

prenum(text, lang) runs FIRST inside each language's normalizer (dacts/text.py). It turns the locale-specific forms into
words or into plain digits the normalizer's own number step already reads right:
  phone-like digit runs (555-1234, 03-1234-5678, +31 6 1234 5678) -> digit by digit
  thousands separators (en/zh/ja/ko "1,000"; de/nl/es/pl/uk "1.000"; fr/pl/uk/de "10 000") -> plain digits
  money (€5, 5 €, $3.50, 3,50 €, ¥, £, ₴/грн, ₩) -> "<number> <currency>" (+ cents)
  decimals ("3,05" / "3.05") -> whole + comma word + digits one by one
  percent -> the language's word (zh 百分之 in front)
  times (14:30, fr 14h30) -> the language's clock words; zh 点/分, ja 時/分, ko 시/분 (native hours)
  years 1100-1999 in German / Dutch -> neunzehnhundert... / negentienhonderd...
  German ordinal dates "am 3. Mai" -> "am dritten Mai"
  ranges "3-4" -> "3 to 4" in the language
  Chinese numbers -> Chinese numerals (一百五十, years digit by digit before 年), Korean native counters (두 시, 세 개)
Plain integers left over are read by the normalizer as before."""
import re

from num2words import num2words

_DIGIT_WORD = {}


def _dw(d: str, lang: str) -> str:
    """One digit as a word."""
    if lang == "zh":
        return "零一二三四五六七八九"[int(d)]
    if lang == "ja":
        return "ゼロ イチ ニ サン ヨン ゴ ロク ナナ ハチ キュウ".split()[int(d)]
    if lang == "ko":
        return "공 일 이 삼 사 오 육 칠 팔 구".split()[int(d)]
    k = (d, lang)
    if k not in _DIGIT_WORD:
        _DIGIT_WORD[k] = num2words(int(d), lang=lang)
    return _DIGIT_WORD[k]


def _digits(s: str, lang: str) -> str:
    sep = "" if lang in ("zh",) else " "
    return sep.join(_dw(c, lang) for c in s if c.isdigit())


def _card(n: int, lang: str) -> str:
    if lang == "zh":
        return zh_number(n)
    if lang == "ja":
        return str(n)  # (pyopenjtalk reads plain digits well)
    w = num2words(n, lang=lang)
    return w.replace(" ", "") if lang == "nl" else w  # (Dutch writes numbers as one word)


# ---------------------------------------------------------------- Chinese
def zh_number(n: int) -> str:
    """Chinese reading of an integer: 150 -> 一百五十, 1005 -> 一千零五, 20000 -> 两万 (二 in final position)."""
    d = "零一二三四五六七八九"
    if n == 0:
        return "零"

    def under_10000(m: int, lead: bool) -> str:
        out, zero = "", False
        for unit, name in ((1000, "千"), (100, "百"), (10, "十"), (1, "")):
            q = m // unit % 10
            if q:
                if zero:
                    out += "零"
                    zero = False
                if unit == 10 and q == 1 and not out and lead:
                    out += "十"  # 十五, not 一十五 (at the very start)
                else:
                    out += d[q] + name
            elif out:
                zero = True
        return out
    parts, i = [], 0
    for big in ("", "万", "亿", "万亿"):
        chunk = n // (10000 ** i) % 10000
        if i and n < 10000 ** i:
            break
        parts.append((chunk, big))
        i += 1
    out = ""
    for k, (chunk, big) in enumerate(reversed(parts)):
        lead = k == 0
        if chunk:
            if out and chunk < 1000:
                out += "零"
            out += under_10000(chunk, lead) + big
    return out.rstrip("零") or "零"


# ---------------------------------------------------------------- Korean native numbers (before counters)
_KO_NATIVE_1 = ["", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉"]
_KO_NATIVE_10 = ["", "열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔"]
_KO_COUNTERS = "시간|시|개(?!월)|명|살|마리|잔|권|장|병|번째|번|켤레|송이|그루"


def ko_native(n: int) -> str:
    if not 1 <= n <= 99:
        return num2words(n, lang="ko")
    t, o = divmod(n, 10)
    if t == 2 and o == 0:
        return "스무"
    return _KO_NATIVE_10[t] + _KO_NATIVE_1[o]


# ---------------------------------------------------------------- words per language
_COMMA = {"en": "point", "de": "Komma", "nl": "komma", "es": "coma", "fr": "virgule", "pl": "przecinek", "uk": "кома",
          "ko": "점", "zh": "点", "ja": "テン"}
_PCT = {"en": "percent", "de": "Prozent", "nl": "procent", "es": "por ciento", "fr": "pour cent", "pl": "procent",
        "ko": "퍼센트"}  # (uk: its own plural rule in normalize_uk; zh: 百分之 in front; ja: pyopenjtalk reads % itself)
_TO = {"en": "to", "de": "bis", "nl": "tot", "es": "a", "fr": "à", "pl": "-", "uk": "-", "ko": "-", "zh": "到", "ja": "から"}
_CUR = {  # symbol -> word per language
    "€": {"en": "euros", "de": "Euro", "nl": "euro", "es": "euros", "fr": "euros", "pl": "euro", "uk": "євро", "ko": "유로", "zh": "欧元", "ja": "ユーロ"},
    "$": {"en": "dollars", "de": "Dollar", "nl": "dollar", "es": "dólares", "fr": "dollars", "pl": "dolarów", "uk": "доларів", "ko": "달러", "zh": "美元", "ja": "ドル"},
    "£": {"en": "pounds", "de": "Pfund", "nl": "pond", "es": "libras", "fr": "livres", "pl": "funtów", "uk": "фунтів", "ko": "파운드", "zh": "英镑", "ja": "ポンド"},
    "¥": {"en": "yen", "de": "Yen", "nl": "yen", "es": "yenes", "fr": "yens", "pl": "jenów", "uk": "єн", "ko": "엔", "zh": "元", "ja": "円"},
    "₴": {"en": "hryvnias", "de": "Hrywnja", "nl": "grivna", "es": "grivnas", "fr": "hryvnias", "pl": "hrywien", "uk": "гривень", "ko": "흐리우냐", "zh": "格里夫纳", "ja": "フリヴニャ"},
    "₩": {"en": "won", "de": "Won", "nl": "won", "es": "wones", "fr": "wons", "pl": "wonów", "uk": "вон", "ko": "원", "zh": "韩元", "ja": "ウォン"},
}
_CENT_JOIN = {"en": "and", "es": "con", "fr": "", "de": "", "nl": "", "pl": "", "uk": ""}
_MONTHS_DE = "Januar|Jänner|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember"
_DE_ORD_NOUN = _MONTHS_DE + "|Jahrhundert|Jahrtausend|Platz|Stock|Stockwerk|Klasse|Etage|Liga|Advent|Mal|Geburtstag|Jahrestag|Kapitel|Runde|Spieltag|Weltkrieg"
_DE_DAT = r"am|vom|zum|im|beim|seit dem|ab dem|bis zum|nach dem|vor dem|an dem|von dem|zu dem|in dem|dem|den"
_DECIMAL_DOT = {"en", "zh", "ja", "ko"}
# Polish clock hours, feminine ordinal (locative / genitive "-ej": o czternastej, do piątej; the usual way to say a time)
_PL_HOURS = ["zero", "pierwszej", "drugiej", "trzeciej", "czwartej", "piątej", "szóstej", "siódmej", "ósmej", "dziewiątej",
             "dziesiątej", "jedenastej", "dwunastej", "trzynastej", "czternastej", "piętnastej", "szesnastej",
             "siedemnastej", "osiemnastej", "dziewiętnastej", "dwudziestej", "dwudziestej pierwszej",
             "dwudziestej drugiej", "dwudziestej trzeciej"]
_UK_MONTHS = "січня|лютого|березня|квітня|травня|червня|липня|серпня|вересня|жовтня|листопада|грудня"


def _uk_ord(n: int, case: str) -> str:
    """Ukrainian ordinal, masculine, genitive ('gen': двадцять четвертого) or locative ('loc': першому); only the last
    word is ordinal (тисяча дев'ятсот дев'яносто першого)."""
    words = num2words(n, lang="uk", to="ordinal").split()
    last = words[-1]
    if case == "nom":  # (по 2022 -> по дві тисячі двадцять другий)
        pass
    elif last.endswith("ій"):  # третій -> третього / третьому
        last = last[:-2] + ("ього" if case == "gen" else "ьому")
    elif last.endswith(("ий", "ой")):  # перший -> першого / першому
        last = last[:-2] + ("ого" if case == "gen" else "ому")
    out = " ".join(words[:-1] + [last])
    return re.sub(r"^одна тисяча", "тисяча", out)


def _uk_dates(text: str) -> str:
    """24 серпня -> двадцять четвертого серпня; 1991 року / 1991 р. -> ... першого року; у/в 1991 році -> ... першому році."""
    text = re.sub(r"\b(\d{4})\s*р\.(?=\s|$|[,;:])", r"\1 року", text)
    text = re.sub(rf"(?<!\d)(\d{{1,2}})(\s+(?:{_UK_MONTHS}))", lambda m: _uk_ord(int(m.group(1)), "gen") + m.group(2), text)
    text = re.sub(r"(?<!\d)(\d{3,4})(\s+року)", lambda m: _uk_ord(int(m.group(1)), "gen") + m.group(2), text)
    text = re.sub(r"(?<!\d)(\d{3,4})(\s+році)", lambda m: _uk_ord(int(m.group(1)), "loc") + m.group(2), text)
    # years with no "року / році" after them (news text: "У 1991", "з 2014 по 2022"): the case from the preposition
    yr = r"(?<!\d)(1[0-9]\d\d|20\d\d)(?![\d.,]\d|\d)"
    text = re.sub(rf"(?i)\b(у|в)\s+{yr}", lambda m: f"{m.group(1)} {_uk_ord(int(m.group(2)), 'loc')}", text)
    text = re.sub(rf"(?i)\b(з|із|до|від|після|кінця|початку|протягом|середини)\s+{yr}", lambda m: f"{m.group(1)} {_uk_ord(int(m.group(2)), 'gen')}", text)
    text = re.sub(rf"(?i)\b(по|на)\s+{yr}", lambda m: f"{m.group(1)} {_uk_ord(int(m.group(2)), 'nom')}", text)
    return text


def _num_words(s: str, lang: str) -> str:
    """'3.05' / '3,05' (already in this language's decimal form) -> words; an integer -> cardinal."""
    sep = "." if lang in _DECIMAL_DOT else ","
    whole, _, frac = s.partition(sep)
    if not frac:
        return _card(int(whole), lang)
    if lang == "ja":
        return s  # (pyopenjtalk: 3.5 -> サンテンゴ)
    joiner = "" if lang == "zh" else " "
    return joiner.join([_card(int(whole), lang), _COMMA[lang], _digits(frac, lang)])


_MONTHS = {
    "de": "Januar Februar März April Mai Juni Juli August September Oktober November Dezember".split(),
    "nl": "januari februari maart april mei juni juli augustus september oktober november december".split(),
    "fr": "janvier février mars avril mai juin juillet août septembre octobre novembre décembre".split(),
    "es": "enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre".split(),
    "pl": "stycznia lutego marca kwietnia maja czerwca lipca sierpnia września października listopada grudnia".split(),
    "uk": _UK_MONTHS.split("|"),
}
_POINT = {"en": "point", "de": "Punkt", "nl": "punt", "fr": "point", "es": "punto", "pl": "kropka", "uk": "крапка",
          "ko": "점", "zh": "点", "ja": "テン"}
_MINUS = {"en": "minus", "de": "minus", "nl": "min", "es": "menos", "fr": "moins", "pl": "minus", "uk": "мінус",
          "ko": "마이너스", "zh": "负", "ja": "マイナス"}
_FRAC = {
    "en": {"1/2": "one half", "1/3": "one third", "1/4": "one quarter", "3/4": "three quarters", "2/3": "two thirds"},
    "de": {"1/2": "ein halb", "1/3": "ein Drittel", "1/4": "ein Viertel", "3/4": "drei Viertel", "2/3": "zwei Drittel"},
    "nl": {"1/2": "een half", "1/3": "een derde", "1/4": "een kwart", "3/4": "drie kwart", "2/3": "twee derde"},
    "fr": {"1/2": "un demi", "1/3": "un tiers", "1/4": "un quart", "3/4": "trois quarts", "2/3": "deux tiers"},
    "es": {"1/2": "un medio", "1/3": "un tercio", "1/4": "un cuarto", "3/4": "tres cuartos", "2/3": "dos tercios"},
    "pl": {"1/2": "jedna druga", "1/3": "jedna trzecia", "1/4": "jedna czwarta", "3/4": "trzy czwarte", "2/3": "dwie trzecie"},
    "uk": {"1/2": "одна друга", "1/3": "одна третя", "1/4": "одна четверта", "3/4": "три четвертих", "2/3": "дві третіх"},
}
_HALF = {"en": " and a half", "de": "einhalb", "nl": " en een half", "fr": " et demi", "es": " y medio", "pl": " i pół",
         "uk": " з половиною"}
_SQ = {"en": ("square meters", "square kilometers"), "de": ("Quadratmeter", "Quadratkilometer"),
       "nl": ("vierkante meter", "vierkante kilometer"), "fr": ("mètres carrés", "kilomètres carrés"),
       "es": ("metros cuadrados", "kilómetros cuadrados"), "pl": ("metrów kwadratowych", "kilometrów kwadratowych"),
       "uk": ("квадратних метрів", "квадратних кілометрів")}
# letters that follow a number as an ordinal ending (kept glued): en 21st, nl 31e / 1ste / 2de, fr 1er / 2e / 3ème
_ORD_SUFFIX = {"en": r"st|nd|rd|th|s", "de": r"er|ern|ers|erjahre|erjahren", "nl": r"e|ste|de|en", "fr": r"er|re|ère|e|ème|eme|es|ers|ères|nd|nde|d|de|h|h\d\d"}
# German: feminine nouns (ein -> eine); common ones + endings
_DE_FEM_END = ("ung", "heit", "keit", "schaft", "ion", "tät", "ik", "ei", "ur", "enz", "anz")
_DE_FEM = set("stunde minute sekunde woche nacht frau stadt zeit sache seite frage person tür tasse flasche million milliarde "
              "nummer straße strasse firma familie geschichte schule klasse idee mark krone".split())
_FR_FEM = set("minute heure seconde semaine journée année personne fois page femme fille chose ville voiture maison "
              "place partie question chambre porte table phrase lettre langue chanson photo vidéo étoile équipe "
              "personnes minutes heures secondes semaines années pages femmes filles choses villes voitures maisons "
              "places parties questions chambres portes tables phrases lettres langues chansons photos vidéos étoiles "
              "équipes victoire victoires classe classes école écoles".split())
_PL_FEM = set("godziny minuty sekundy osoby kobiety strony książki córki siostry dziewczyny matki noce złotówki "
              "godzina minuta sekunda osoba kobieta strona".split())
_UK_FEM = set("години хвилини секунди особи сторінки жінки дівчини книжки доби ночі тисячі гривні "
              "година хвилина секунда особа сторінка жінка дівчина книжка доба ніч тисяча гривня".split())
_FW = str.maketrans("０１２３４５６７８９" + "".join(chr(c) for c in range(0xFF21, 0xFF3B)) + "".join(chr(c) for c in range(0xFF41, 0xFF5B)),
                    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
_SUPSUB = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉", "01234567890123456789")
_ONE_CUR = {  # the currency word after exactly 1
    "en": {"dollars": "dollar", "euros": "euro", "pounds": "pound", "hryvnias": "hryvnia"},
    "es": {"euros": "euro", "dólares": "dólar", "libras": "libra", "yenes": "yen", "grivnas": "grivna", "wones": "won"},
    "fr": {"euros": "euro", "dollars": "dollar", "livres": "livre", "yens": "yen", "hryvnias": "hryvnia", "wons": "won"},
    "pl": {"dolarów": "dolar", "funtów": "funt", "jenów": "jen", "wonów": "won"},
    "uk": {"доларів": "долар", "фунтів": "фунт", "єн": "єна", "вон": "вона"},
}
_CENT = {"de": "Cent", "nl": "cent", "fr": "centimes", "es": "céntimos", "pl": "groszy", "uk": "копійок"}
_ES_SKIP = set("a y o u e de del en por para con que es son está están fue era se lo la el los las al".split())
_ES_FEM_O = set("mano manos foto fotos moto motos radio radios".split())
_ES_NO_SHORT = r"número|nº|n\.º|capítulo|página|artículo|el|la"
_DE_DAT_PREP = r"mit|nach|seit|von|zu|bei|aus|vor|in"
_DE_TIME_WORD = set("jahr woche monat tag stunde minute sekunde nacht".split())
_PL_FEM_ONE = set("osoba godzina minuta sekunda kobieta strona książka córka siostra dziewczyna noc złotówka".split())
_UK_NEUT = set("яблуко місце вікно слово місто село око серце дерево поле море".split())
_LIST_WORD = r"und|en|et|y|i|oraz|та|і|and|or|oder|of|ou|o|lub|або"
_CLOCK_CUE = {
    "de": r"um|gegen|ab|bis|von|seit|nach|vor|zwischen|und", "nl": r"om|tegen|tot|vanaf|van|na|voor|rond|tussen|en",
    "fr": r"à|a|vers|de|jusqu'à|dès|après|avant|entre|et", "es": r"a las|a la|desde las|hasta las|de las|sobre las|las|la|entre las|y las",
}
_DEG = {"en": "degrees", "de": "Grad", "nl": "graden", "fr": "degrés", "es": "grados", "pl": "stopni", "uk": "градусів",
        "ko": "도", "zh": "度", "ja": "度"}
_CELSIUS = {"en": "Celsius", "de": "Celsius", "nl": "Celsius", "fr": "Celsius", "es": "Celsius", "pl": "Celsjusza",
            "uk": "Цельсія", "ko": "", "zh": "", "ja": ""}
_NUMERO = {"uk": "номер", "pl": "numer", "de": "Nummer", "nl": "nummer", "fr": "numéro", "es": "número", "en": "number",
           "ko": "번", "zh": "号", "ja": "番"}


def _half(n: str, lang: str) -> str:
    """'1' + ½ -> 'one and a half' / German 'eineinhalb' (one word) / ..."""
    if lang == "de":
        return ("ein" if n == "1" else num2words(int(n), lang="de")) + "einhalb"
    return n + _HALF[lang]


def _uk_plural(n: int, forms) -> str:
    """forms = (1, 2-4, 5+) e.g. ("гривня", "гривні", "гривень")."""
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return forms[1]
    return forms[2]


def _fr_fem(w: str) -> str:
    """French feminine for hours: un -> une, vingt et un -> vingt et une."""
    return re.sub(r"\bun$", "une", w)


def _pl_nom_hour(h: int) -> str:
    """o czternastej -> (Jest) czternasta."""
    out = []
    for w in _PL_HOURS[h].split():
        out.append("druga" if w == "drugiej" else (w[:-2] + "a" if w.endswith("ej") else w))
    return " ".join(out)


def prenum(text: str, lang: str) -> str:
    if not re.search(r"\d", text):
        return text
    cjk = lang in ("zh", "ja", "ko")
    # 0a. full-width digits / separators between digits, superscripts and subscripts -> ASCII digits
    text = re.sub(r"(?<=[０-９])，(?=[０-９]{3})", ",", text)
    text = text.translate(_FW)
    if lang == "ja":  # (１，５００円; in Chinese "，" is the sentence comma: 2:30，150元)
        text = re.sub(r"(?<=\d)[，](?=\d{3})", ",", text)
    text = re.sub(r"(?<=\d)[．](?=\d)", ".", text)
    if lang in _SQ:  # m² / km² -> the language's square meters
        text = re.sub(r"(\d)\s?(k?)m(?:²|\^2)", lambda m: f"{m.group(1)} {_SQ[lang][1 if m.group(2) else 0]}", text)
    text = text.translate(_SUPSUB)
    # 0b. ½ ¼ ¾ (1½ -> 1 and a half)
    if lang in _HALF:
        text = re.sub(r"(\d+)\s?½", lambda m: _half(m.group(1), lang), text)
    text = text.replace("½", " 1/2").replace("¼", " 1/4").replace("¾", " 3/4")
    if not cjk and lang != "en":  # 2€50 -> 2,50 €
        text = re.sub(r"(\d+)\s?€\s?(\d\d)(?!\d)", r"\1,\2 €", text)
    if lang == "en":  # the 1990s -> nineteen nineties
        text = re.sub(r"(?<![\d.,])(1[1-9]\d0|20[0-9]0)s\b", lambda m: (lambda w: w[:-1] + "ies" if w.endswith("y") else w + "s")(
            num2words(int(m.group(1)), to="year")), text)
    if lang == "de":  # 3 Mio. € -> drei Millionen Euro
        text = re.sub(r"(\d)\s?(Mio|Mrd|Tsd)\.?(\s?€)?", lambda m: m.group(1) + " " + {"Mio": "Millionen", "Mrd": "Milliarden", "Tsd": "tausend"}[m.group(2)]
                      + (" Euro" if m.group(3) else ""), text)
    # 0c. a digit glued to a letter (3D, A4, MP3, 4K, 10kg, 5G): a space between, except ordinal endings
    if not cjk:
        suf = _ORD_SUFFIX.get(lang)
        text = re.sub(r"(?<=\d)(?=[^\W\d_])" + (rf"(?!(?:{suf})\b)" if suf else ""), " ", text)
        text = re.sub(r"(?<=[^\W\d_])(?=\d)", " ", text)
    # 1. numeric dates (not English: m/d vs d/m can't be known) -> "<day> <month> <year>" in the language; the
    #    language's own ordinal / date steps below then read them
    def date(y, mo, d):
        y, mo, d = int(y), int(mo), int(d)
        if not (1 <= mo <= 12 and 1 <= d <= 31):
            return None
        if cjk:
            return f"{y}年{mo}月{d}日" if lang != "ko" else f"{y}년 {mo}월 {d}일"
        mn = _MONTHS[lang][mo - 1]
        if lang == "de":
            return f"{d}. {mn} {y}"
        if lang == "es":
            return f"{d} de {mn} de {y}"
        if lang == "fr":
            return f"{'1er' if d == 1 else d} {mn} {y}"
        if lang == "pl":
            return f"{d} {mn} {y} roku"
        if lang == "uk":
            return f"{d} {mn} {y} року"
        return f"{d} {mn} {y}"
    if cjk:  # 2024.05.12 / 2024/5/12
        text = re.sub(r"(?<![\d.,/-])(\d{4})[./](\d{1,2})[./](\d{1,2})(?!\d|[./]\d)", lambda m: date(m.group(1), m.group(2), m.group(3)) or m.group(0), text)
    if lang != "en":
        text = re.sub(r"(?<![\d.,/-])(\d{4})-(\d{1,2})-(\d{1,2})(?!\d|[.,/-]\d)", lambda m: date(m.group(1), m.group(2), m.group(3)) or m.group(0), text)
        text = re.sub(r"(?<![\d.,/-])(\d{1,2})([./-])(\d{1,2})\2(\d{4})(?!\d|[.,/-]\d)(?:\s*(?:r\.|року|roku))?",
                      lambda m: date(m.group(4), m.group(3), m.group(1)) or m.group(0), text)
    # 2. thousands separators -> plain digits (a 1-3 digit head, then groups of exactly 3)
    if lang in ("en", "zh", "ja", "ko"):
        text = re.sub(r"(?<![\d.,])[1-9]\d{0,2}(?:,\d{3})+(?!\d|,\d)", lambda m: m.group(0).replace(",", ""), text)
    else:
        text = re.sub(r"(?<![\d.,])[1-9]\d{0,2}(?:\.\d{3})+(?!\d|\.\d)", lambda m: m.group(0).replace(".", ""), text)
        text = re.sub(r"(?<![\d.,])[1-9]\d{0,2}(?:[ \u00a0\u202f\u2009]\d{3})+(?!\d|[.,]\d{3})",
                      lambda m: re.sub(r"\D", "", m.group(0)), text)
    # 2a'. comma languages: "3.5" / "0.125" with a point is a decimal too (a thousands group needs a non-zero head + 3 digits)
    if lang not in _DECIMAL_DOT:
        text = re.sub(r"(?<![\d.,])(\d+)\.(\d{1,2}|\d{4,})(?![\d.]|,\d)(?!\s?(?:Uhr|uur|h)\b)", r"\1,\2", text)  # (not 7.30 Uhr)
        text = re.sub(r"(?<![\d.,])(0)\.(\d{3})(?![\d.]|,\d)", r"\1,\2", text)
    # 2b. three or more dot groups (versions 2.0.1, addresses 192.168.1.1): each group a number, "point" between
    text = re.sub(r"(?<![\d.,])\d+(?:\.\d+){2,}(?!\d|\.\d)", lambda m: f" {_POINT.get(lang, 'point')} ".join(
        _card(int(g), lang) if lang != "ja" else g for g in m.group(0).split(".")), text)
    # 2c. a minus sign before a number (-5°C)
    text = re.sub(r"(?<![\w)\d])[-−](?=\d)", lambda m: _MINUS.get(lang, "minus") + ("" if lang in ("zh", "ja") else " "), text)
    # 2d. common fractions (1/2, 1/3, 1/4, 3/4, 2/3); zh / ja / ko: any N/M the regular way
    if lang in ("zh", "ja", "ko"):
        fmt = {"zh": "{m}分之{n}", "ja": "{m}分の{n}", "ko": "{m}분의 {n}"}[lang]
        text = re.sub(r"(?<![\d/.,])(\d{1,2})/(\d{1,2})(?![\d/])", lambda m: fmt.format(n=m.group(1), m=m.group(2)), text)
    elif lang in _FRAC:
        if lang in _HALF:  # 3 1/2 -> three and a half
            text = re.sub(r"(?<![\d/.,])(\d+)\s+1/2(?![\d/])", lambda m: _half(m.group(1), lang), text)
        if lang == "es":  # 1/2 litro -> medio litro, 1/2 hora -> media hora
            text = re.sub(r"(?<![\d/.,])1/2(\s+)([^\W\d_]+)", lambda m: ("media" if m.group(2).lower().endswith("a") else "medio") + m.group(1) + m.group(2), text)
        if lang == "de":  # 1/2 Liter -> ein halber Liter, 1/2 Stunde -> eine halbe Stunde
            text = re.sub(r"(?<![\d/.,])1/2(\s+)([A-ZÄÖÜ][^\W\d_]*)", lambda m: ("eine halbe" if m.group(2).lower() in _DE_FEM or m.group(2).lower().endswith(_DE_FEM_END)
                          else "ein halber") + m.group(1) + m.group(2), text)
        text = re.sub(r"(?<![\d/.,])([123])/([234])(?![\d/])", lambda m: _FRAC[lang].get(m.group(1) + "/" + m.group(2), m.group(0)), text)
    # 3. phone numbers and long digit runs -> digit by digit
    plus_w = {"en": "plus", "de": "plus", "nl": "plus", "es": "más", "fr": "plus", "pl": "plus", "uk": "плюс"}

    def phone(s):
        n = sum(c.isdigit() for c in s)
        groups = [g for g in re.split(r"[^\d]+", s) if g]
        glue = "、" if lang in ("ja", "zh") else ", "
        lead = (plus_w.get(lang, "") + " ") if s.lstrip().startswith("+") and plus_w.get(lang) else ""
        return lead + glue.join(_digits(g, lang) for g in groups) if 7 <= n <= 15 else s
    # 3a. starts with +, ( or 0: a phone whatever the group sizes
    text = re.sub(r"(?<![\d.,+A-Za-z])(?:\+\d|\(\d|0\d)[\d\s\-()./]{5,20}\d(?![\d])", lambda m: phone(m.group(0)), text)
    if lang == "zh":  # after 电话 / 热线 / 拨打 ...: digit by digit (10086, 110)
        text = re.sub(r"(热线|电话|拨打|号码|手机|呼叫|致电|拨)(是|为|：|:)?\s?(\d{3,})", lambda m: m.group(1) + (m.group(2) or "") + _digits(m.group(3), "zh"), text)
    # 3b. three or more dash / space groups of 2-4 digits, 7+ digits in all
    text = re.sub(r"(?<![\d.,])\d{2,4}(?:[\-\s]\d{2,4}){2,}(?!\d)", lambda m: phone(m.group(0)), text)
    # 3c. English 555-1234 only after a cue word; Korean 1588-1234 (4-4) is a phone
    if lang == "en":
        text = re.sub(r"(?i)\b(call|phone|tel|telephone|dial|number|fax|text)(\W{1,3}(?:me\s+(?:at|on)\s+)?)(\d{3})-(\d{4})(?!\d)",
                      lambda m: m.group(1) + m.group(2) + _digits(m.group(3), "en") + ", " + _digits(m.group(4), "en"), text)
    if lang == "ko":
        text = re.sub(r"(?<![\d.,])(\d{4})-(\d{4})(?!\d)", lambda m: _digits(m.group(1), "ko") + ", " + _digits(m.group(2), "ko"), text)
    # 3d. unbroken long runs: 9+ digits, or a leading 0 and 7+ (mobile numbers, IDs)
    text = re.sub(r"(?<![\d.,])(?:\d{9,}|0\d{6,})(?![\d.,]\d)", lambda m: _digits(m.group(0), lang), text)
    dec = r"\d+(?:\.\d+)?" if lang in _DECIMAL_DOT else r"\d+(?:,\d+)?"
    # 4. money: symbol before or after the amount
    if lang == "en":
        text = re.sub(r"\$(\d+(?:\.\d+)?)\s?(thousand|million|billion|trillion)\b", r"\1 \2 dollars", text)
        text = re.sub(r"\$(\d+)\.00(?!\d)", r"$\1", text)  # ($100.00 -> $100)
        text = re.sub(r"\$1\.(\d\d)(?!\d)", lambda m: f"one dollar and {int(m.group(1))} cent{'s' if int(m.group(1)) != 1 else ''}", text)
        text = re.sub(r"\$1(?!\d|[.,]\d)", "one dollar", text)
        text = re.sub(r"\$0\.(\d\d)(?!\d)", lambda m: f"{int(m.group(1))} cent{'s' if int(m.group(1)) != 1 else ''}", text)
        # $5.5 / $1.125 (not cents): a decimal amount of dollars (expand_numbers only knows $x.yy)
        text = re.sub(r"\$(\d+\.(?:\d|\d{3,}))(?!\d)", lambda m: _num_words(m.group(1), "en") + " dollars", text)
        def en_money(m):
            amt, big, word = m.group(2) or m.group(4), m.group(3) or "", _CUR[m.group(1) or m.group(5)]["en"]
            whole, _, frac = amt.partition(".")
            if whole == "0" and frac and len(frac) == 2 and not big:
                return f"{int(frac)} cent{'s' if int(frac) != 1 else ''}"
            if whole == "1" and not big:
                word = _ONE_CUR["en"].get(word, word)
            if frac and len(frac) == 2 and not big:  # €3.50 -> three euros fifty
                return f"{whole} {word} {int(frac)}"
            return f"{amt}{big} {word}"
        text = re.sub(r"([€£¥₴₩])\s?(\d+(?:\.\d+)?)(\s?(?:thousand|million|billion|trillion)\b)?|(\d+(?:\.\d+)?)\s?([€£¥₴₩])", en_money, text)
    else:
        syms = "".join(re.escape(k) for k in _CUR)

        def money(m):
            sym = m.group("s1") or m.group("s2")
            amt = m.group("a1") or m.group("a2")
            word = _CUR[sym].get(lang, "")
            sep = "." if lang in _DECIMAL_DOT else ","
            whole, _, frac = amt.partition(sep)
            if cjk:
                return f"{amt}{word}"
            if lang == "uk" and sym == "₴":
                word = _uk_plural(int(whole), ("гривня", "гривні", "гривень")) if not frac else "гривні"
            if whole == "1" and not (frac and int(frac)):
                word = _ONE_CUR.get(lang, {}).get(word, word)
            if whole == "0" and frac and len(frac) == 2 and int(frac) and lang in _CENT:  # 0,50 € -> fünfzig Cent
                return f"{int(frac)} {_CENT[lang]}"
            if frac and len(frac) == 2:
                if not int(frac):  # (5,00 € -> fünf Euro)
                    return f"{whole} {word}"
                j = _CENT_JOIN.get(lang, "")
                return f"{whole} {word}{' ' + j if j else ''} {int(frac)}"
            return f"{amt} {word}"
        text = re.sub(rf"(?P<s1>[{syms}])\s?(?P<a1>{dec})|(?P<a2>{dec})\s?(?P<s2>[{syms}])", money, text)
    if not cjk and lang != "en":  # the currency WORD after an amount with cents (19,99 Euro) and 2€50
        cw = r"Euro|euro|euros|Dollar|dollar|dollars|dólares|złotych|złote|zł|євро|доларів|гривень|гривні|грн"
        text = re.sub(rf"(?<![\d,])(\d+),(\d\d)\s?({cw})\b", lambda m: f"{m.group(1)} {m.group(3)}" + (f" {int(m.group(2))}" if int(m.group(2)) else ""), text)
        text = re.sub(r"(\d+)\s?€\s?(\d\d)\b", lambda m: f"{m.group(1)} {_CUR['€'][lang]} {int(m.group(2))}", text)
    if lang == "uk":
        text = re.sub(r"(\d+,\d+)\s?%", r"\1 відсотка", text)  # (a fraction of a percent: genitive singular)
        text = re.sub(r"(\d+(?:,\d+)?)\s?грн\.?", lambda m: m.group(1) + " " + (
            _uk_plural(int(m.group(1)), ("гривня", "гривні", "гривень")) if "," not in m.group(1) else "гривні"), text)
        # гривня is feminine: 1 / 21 -> одна / двадцять одна, 2 -> дві
        text = re.sub(r"(?<![\d,])(\d+)(\s+гривн[яі])\b", lambda m: re.sub(r"\bодин$", "одна", re.sub(r"\bдва$", "дві",
                      num2words(int(m.group(1)), lang="uk"))).replace("одна тисяча", "тисяча", 1) + m.group(2), text)
        text = _uk_dates(text)
    if lang == "pl":  # złoty: 1 złoty, 2-4 złote, else złotych (+ grosze); "od 500-1000" -> "od 500 do 1000" (cases follow)
        def zl(m):
            n = int(m.group(1))
            form = "złoty" if n == 1 else ("złote" if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14 else "złotych")
            return f"{m.group(1)} {form}" + (f" {int(m.group(2))}" if m.group(2) and int(m.group(2)) else "")
        text = re.sub(r"(?<![\d,])(\d+)(?:,(\d{2}))?\s?(?:zł|PLN)(?!\w)", zl, text)
        text = re.sub(r"(?i)\b(od)\s+(\d+)\s?[-–]\s?(\d+)", r"\1 \2 do \3", text)
    if lang in ("fr", "es"):
        text = re.sub(r"\b[nN][°º]\s?(?=\d)", "numéro " if lang == "fr" else "número ", text)
    if lang == "ko":
        text = re.sub(r"(?<!\d)6\s?월", "유월", text)
        text = re.sub(r"(?<!\d)10\s?월", "시월", text)
        text = re.sub(rf"(?<![\d.,])(\d{{1,2}})\s?[~～]\s?(\d{{1,2}})\s?({_KO_COUNTERS})",
                      lambda m: f"{ko_native(int(m.group(1)))} {m.group(3)}에서 {ko_native(int(m.group(2)))} {m.group(3)}", text)
    if lang == "ko":  # 3-4일 -> 삼사 일 (Sino counters)
        text = re.sub(r"(?<![\d.,])(\d)\s?[-–]\s?(\d)\s?(일|월|년|분|층|번|주)", lambda m: _dw(m.group(1), "ko") + _dw(m.group(2), "ko") + " " + m.group(3), text)
    # 5. percent, degrees, №
    if lang == "zh":
        text = re.sub(rf"({dec})\s?[%％]", r"百分之\1", text)
    elif lang in _PCT:
        text = re.sub(rf"({dec})\s?[%％]", lambda m: f"{m.group(1)} {_PCT[lang]}", text)
    if lang == "zh":
        text = re.sub(rf"(-?{dec})\s?°\s?C", r"摄氏\1度", text)
    text = re.sub(rf"({dec})\s?°\s?([CF])?\b", lambda m: f"{m.group(1)} {_DEG.get(lang, 'degrees')}"
                  + (f" {_CELSIUS.get(lang, '')}" if m.group(2) == "C" and _CELSIUS.get(lang) else (" Fahrenheit" if m.group(2) == "F" else "")), text)
    text = re.sub(r"№\s?(?=\d)", (_NUMERO.get(lang, "number") + " "), text)
    # 6. ordinals written with º / ª (Spanish), 1re / 1ère (French)
    if lang == "es":
        text = re.sub(r"(\d+)\.?\s?([ºª°])", lambda m: (lambda w: w[:-1] + "a" if m.group(2) == "ª" and w.endswith("o") else w)(
            num2words(int(m.group(1)), lang="es", to="ordinal")), text)
    if lang == "fr":
        text = re.sub(r"\b1(?:re|ère)\b", "première", text)
    # 7. clocks (only with a time cue; zh / ja / ko / pl / uk: HH:MM alone)
    def clock(h, mi, pre=""):
        h, mi = int(h), int(mi)
        if lang == "de":
            return f"{'ein' if h == 1 else _card(h, lang)} Uhr" + (f" {_card(mi, lang)}" if mi else "")
        if lang == "nl":
            return f"{_card(h, lang)} uur" + (f" {_card(mi, lang)}" if mi else "")
        if lang == "fr":
            return f"{_fr_fem(_card(h, lang))} heure{'s' if h > 1 else ''}" + (f" {_card(mi, lang)}" if mi else "")
        if lang == "es":
            return f"{_card(h, lang)}" + (f" y {_card(mi, lang)}" if mi else " en punto")
        if lang == "pl":
            hw = _PL_HOURS[h] if re.search(r"(?i)\b(o|do|od|po|przed|koło|około|między)\s*$", pre) else _pl_nom_hour(h)
            return hw + (f" {_card(mi, lang)}" if mi else "")
        if lang == "uk":
            return f"{_card(h, lang)}" + (f" {_card(mi, lang)}" if mi else " нуль нуль")
        if lang == "zh":
            return f"{'两' if h == 2 else zh_number(h)}点" + (f"{zh_number(mi)}分" if mi else "")
        if lang == "ja":
            return f"{h}時" + (f"{mi}分" if mi else "")
        if lang == "ko":
            return f"{ko_native(h) if 1 <= h <= 12 else num2words(h, lang='ko')} 시" + (f" {num2words(mi, lang='ko')} 분" if mi else "")
        return None
    if lang != "en":
        hhmm = r"([01]?\d|2[0-3])[:：]([0-5]\d)(?![\d:])"
        if lang in ("de", "nl", "fr", "es"):
            # with a unit after it (Uhr / uur / h, also "7.30 Uhr") ...
            unit = {"de": "Uhr", "nl": "uur", "fr": "h", "es": "h"}[lang]
            text = re.sub(rf"(?<![\d.,:])([01]?\d|2[0-3])[:.：]([0-5]\d)\s?{unit}\b", lambda m: clock(m.group(1), m.group(2)), text)
            # ... or after a time word (um / om / à / a las ...)
            text = re.sub(rf"(?i)(\b(?:{_CLOCK_CUE[lang]})\s+)(?<![\d.,:]){hhmm}", lambda m: m.group(1) + clock(m.group(2), m.group(3)), text)
            if lang == "fr":
                text = re.sub(r"(?<![\d.,:])([01]?\d|2[0-3])\s?h(?:\s?([0-5]\d))?\b", lambda m: clock(m.group(1), m.group(2) or "0"), text)
            if lang == "de":
                text = re.sub(r"(?<![\d.,])1\s+Uhr(?!\w)", "ein Uhr", text)
        else:
            text = re.sub(rf"(?<![\d.,:]){hhmm}", lambda m: clock(m.group(1), m.group(2), text[max(0, m.start() - 12):m.start()]) or m.group(0), text)
    # 8. German ordinal dates / nouns: "am 3. Mai" -> "am dritten Mai", "des 20. Jahrhunderts" -> "des zwanzigsten ..."
    if lang == "de":
        def de_ord(m):
            pre, n, noun = m.group(1) or "", int(m.group(2)), m.group(3)
            w = num2words(n, lang="de", to="ordinal")
            art = pre.strip().lower()
            if art in ("des", "dem", "den", "am", "im", "zum", "vom", "beim") or art in _DE_DAT.split("|"):
                w += "n"
            return f"{pre}{w} {noun}"
        text = re.sub(rf"(?i:\b((?:{_DE_DAT}|des|der|die|das)\s+))?(?<![\d.,])(\d{{1,2}})\.\s+((?:{_DE_ORD_NOUN})(?:e?s)?\b)", de_ord, text)
    # 10. ranges "3-4 Stunden" -> "3 bis 4 Stunden": increasing, a word after it (not scores "2-1"); not Polish / Ukrainian
    if _TO.get(lang) not in (None, "-"):
        text = re.sub(r"(?<![\d\-/.,])(\d{1,4})\s?[-–]\s?(\d{1,4})(?=\s*[^\W\d_])",
                      lambda m: f"{m.group(1)} {_TO[lang]} {m.group(2)}" if int(m.group(2)) > int(m.group(1)) else m.group(0), text)
    # 9. years 1100-1999 in German / Dutch (neunzehnhundertneunzig); 2000+ is a plain cardinal there
    if lang in ("de", "nl"):
        text = re.sub(r"(?<![\d.,])(1[1-9]\d\d)(?![\d]|[.,]\d)",
                      lambda m: num2words(int(m.group(1)), lang=lang, to="year").replace(" ", ""), text)
    # 11. Korean native numbers before counters (2시 -> 두 시, 3개 -> 세 개, 20살 -> 스무 살; 1번째 -> 첫 번째)
    if lang == "ko":
        text = re.sub(r"(?<![\d.,])1\s?번째", "첫 번째", text)
        text = re.sub(rf"(?<![\d.,제])(\d{{1,2}})\s?({_KO_COUNTERS})(?![가-힣]{{2,}})",
                      lambda m: (f"{ko_native(int(m.group(1)))} {m.group(2)}" if not (m.group(2) == "시" and int(m.group(1)) > 12)
                                 else f"{num2words(int(m.group(1)), lang='ko')} 시"), text)
    # 11b. numbers before a noun that need the noun's gender / a short form
    nxt = r"(\s+)([^\W\d_][\w'-]*)"
    if lang == "de":
        def de_one(m):
            w = m.group(3).lower()
            return ("eine" if w in _DE_FEM or w.endswith(_DE_FEM_END) else "ein") + m.group(2) + m.group(3)
        def de_dat(m):
            prep, w = m.group(1), m.group(4).lower()
            if prep.lower() in ("in", "vor") and w not in _DE_TIME_WORD:
                return m.group(0)
            fem = w in _DE_FEM or w.endswith(_DE_FEM_END)
            return f"{prep}{m.group(2)}{'einer' if fem else 'einem'}{m.group(3)}{m.group(4)}"
        text = re.sub(rf"(?i)\b({_DE_DAT_PREP})(\s+)1(\s+)([A-ZÄÖÜ][^\W\d_]*)", de_dat, text)
        text = re.sub(rf"(?<![\d.,])(1){nxt}", lambda m: de_one(m) if m.group(3)[:1].isupper() else m.group(0), text)
    if lang == "es":
        def es_fem(w):
            w = w.lower()
            return w in _ES_FEM_O or (w.endswith(("a", "as")) and not w.endswith(("ma", "mas", "pa", "pas", "día", "días")))

        def es_range(m):  # "de 200 a 300 euros": both numbers take the noun's gender
            fem = es_fem(m.group(5))
            return es_words(int(m.group(1)), fem) + m.group(2) + es_words(int(m.group(3)), fem) + m.group(4) + m.group(5)

        def es_words(n, fem):
            words = num2words(n, lang="es")
            if n % 10 == 1 and n % 100 != 11:
                words = re.sub(r"veintiuno$", "veintiuna" if fem else "veintiún", words)
                words = re.sub(r"uno$", "una" if fem else "un", words)
            if fem:
                words = re.sub(r"(cientos)\b", "cientas", words).replace("quinientos", "quinientas")
            return words
        text = re.sub(r"(?<![\d.,])(\d{1,6})(\s+(?:a|y|o|-)\s+)(\d{1,6})(\s+)([^\W\d_][\w'-]*)", es_range, text)

        def es_num(m):
            n, w = int(m.group(1)), m.group(3)
            if w.lower() in _ES_SKIP or len(w) == 1 or re.search(rf"(?i)\b(?:{_ES_NO_SHORT})\s*$", text[max(0, m.start() - 12):m.start()]):
                return m.group(0)
            fem = es_fem(w)
            words = num2words(n, lang="es")
            if n % 10 == 1 and n % 100 != 11:
                words = re.sub(r"veintiuno$", "veintiuna" if fem else "veintiún", words)
                words = re.sub(r"uno$", "una" if fem else "un", words)
            if fem:
                words = re.sub(r"(cientos)\b", "cientas", words).replace("quinientos", "quinientas")
            return words + m.group(2) + w
        text = re.sub(rf"(?<![\d.,])(\d{{1,6}}){nxt}", es_num, text)
    if lang == "fr":
        text = re.sub(rf"(?<![\d.,])(\d{{1,6}}){nxt}", lambda m: (re.sub(r"\bun$", "une", num2words(int(m.group(1)), lang="fr"))
                      if m.group(3).lower() in _FR_FEM and int(m.group(1)) % 10 == 1 and int(m.group(1)) % 100 != 11 else m.group(1)) + m.group(2) + m.group(3), text)
    if lang == "pl":
        text = re.sub(rf"(?<![\d.,])(\d{{1,6}}){nxt}", lambda m: (re.sub(r"\bdwa$", "dwie", num2words(int(m.group(1)), lang="pl"))
                      if m.group(3).lower() in _PL_FEM and int(m.group(1)) % 10 == 2 and int(m.group(1)) % 100 != 12 else m.group(1)) + m.group(2) + m.group(3), text)
    if lang == "pl":
        text = re.sub(rf"(?<![\d.,])1{nxt}", lambda m: ("jedna" if m.group(2).lower() in _PL_FEM_ONE else "1") + m.group(1) + m.group(2), text)
    if lang == "uk":
        def uk_fem(m):
            n, w = int(m.group(1)), m.group(3).lower()
            if w in _UK_NEUT and n % 10 == 1 and n % 100 != 11:
                return re.sub(r"\bодин$", "одне", num2words(n, lang="uk")) + m.group(2) + m.group(3)
            if w in _UK_FEM and n % 100 not in (11, 12) and n % 10 in (1, 2):
                return re.sub(r"\bодин$", "одна", re.sub(r"\bдва$", "дві", num2words(n, lang="uk"))) + m.group(2) + m.group(3)
            return m.group(0)
        text = re.sub(rf"(?<![\d.,])(\d{{1,6}}){nxt}", uk_fem, text)
    # 12. decimals -> words (not lists "1,2 und 3" / "1,2,3"; English "$3.50" stays for expand_numbers)
    sep = r"\." if lang in _DECIMAL_DOT else ","
    text = re.sub(rf"(?<![$\d{sep}])(?<!\d,)(\d+{sep}\d+)(?!{sep}?\d)(?!\s+(?:{_LIST_WORD})\s+\d)",
                  lambda m: _num_words(m.group(1), lang), text)
    # 13. Chinese: years digit by digit before 年, 2 before a measure word = 两, other integers as Chinese numerals
    if lang == "zh":
        text = re.sub(r"(\d{4})\s?年", lambda m: _digits(m.group(1), "zh") + "年", text)
        text = re.sub(r"(?<![\d.])2(?=\s?[点个位次本张只件条天])", "两", text)
        text = re.sub(r"\d+", lambda m: zh_number(int(m.group(0))) if len(m.group(0)) <= 16 else _digits(m.group(0), "zh"), text)
    return text
