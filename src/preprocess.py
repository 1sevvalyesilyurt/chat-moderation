"""Text normalization for noisy chat messages (Technique area B: regex normalization).

Two modes:
  - "full"  : used by the lexicon matcher, TF-IDF and word2vec models
  - "light" : used by the (cased) multilingual BERT models
"""
import re
import unicodedata

# --- Turkish character folding -------------------------------------------
# The Jigsaw data is English, but a gaming chat may contain Turkish letters
# (e.g. "salağğ", "İdiot"). We fold them to ASCII so that the same word is
# always written the same way. We translate BEFORE lower(), because Python's
# "İ".lower() gives "i" + a combining dot (two characters), not plain "i".
TURKISH_MAP = str.maketrans({
    "ç": "c", "Ç": "c",
    "ğ": "g", "Ğ": "g",
    "ı": "i", "I": "i", "İ": "i",
    "ö": "o", "Ö": "o",
    "ş": "s", "Ş": "s",
    "ü": "u", "Ü": "u",
})


def fold_turkish(text: str) -> str:
    """Map Turkish-specific letters to ASCII (ç->c, ğ->g, ı/İ->i, ö->o, ş->s, ü->u)."""
    return text.translate(TURKISH_MAP)


# --- Regex patterns ------------------------------------------------------
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")          # Jigsaw has many IP addresses
LEET_MAP = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"}
# a leet character counts only if it touches a letter (so "2020" stays "2020")
LEET_RE = re.compile(r"(?<=[a-z])[013457@$]|[013457@$](?=[a-z])")
# "!" and "|" only between two letters ("sh!t" -> "shit", but "hello!" stays)
LEET_INNER_RE = re.compile(r"(?<=[a-z])[!|](?=[a-z])")
REPEAT_RE = re.compile(r"(.)\1{2,}")                      # 3+ same chars -> 2
NON_ALNUM_RE = re.compile(r"[^a-z0-9' ]+")
SPACES_RE = re.compile(r"\s+")


def replace_leetspeak(text: str) -> str:
    """f4ck -> fack, sh1t -> shit, @ss -> ass, a$$ -> ass (text must be lowercase)."""
    for _ in range(3):  # repeat: "a55" needs two passes
        new = LEET_RE.sub(lambda m: LEET_MAP[m.group(0)], text)
        new = LEET_INNER_RE.sub(lambda m: "i" if m.group(0) == "!" else "l", new)
        if new == text:
            break
        text = new
    return text


def collapse_repeats(text: str) -> str:
    """stuuuupid -> stuupid, !!!!! -> !! (keeps double letters like 'good')."""
    return REPEAT_RE.sub(r"\1\1", text)


def normalize_text(text: str, mode: str = "full") -> str:
    if not isinstance(text, str):
        text = ""
    text = unicodedata.normalize("NFKC", text)
    text = URL_RE.sub(" ", text)
    text = IP_RE.sub(" ", text)
    text = collapse_repeats(text)
    if mode == "light":                     # BERT: keep case and punctuation
        return SPACES_RE.sub(" ", text).strip()
    text = fold_turkish(text).lower()
    text = replace_leetspeak(text)
    text = collapse_repeats(text)            # again, leet may create new repeats
    text = NON_ALNUM_RE.sub(" ", text)
    return SPACES_RE.sub(" ", text).strip()


def tokenize(text: str) -> list:
    """Whitespace tokenizer on already-normalized text."""
    return text.split()
