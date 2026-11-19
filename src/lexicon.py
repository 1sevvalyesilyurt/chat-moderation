"""Rule-based baseline: edit-distance lexicon matcher (Technique area B).

1) build_lexicon(): learn a list of offensive words from the TRAINING split only
   (words that appear often and almost only in offensive messages).
2) LexiconMatcher: a token is a "hit" if it equals a lexicon word or is within
   Levenshtein (minimum edit) distance 1 of it  ->  "fack" matches "fuck".
   A message is offensive if it has at least `min_hits` hits.
"""
from collections import Counter
from functools import lru_cache

from src.preprocess import normalize_text, tokenize


def levenshtein(a: str, b: str) -> int:
    """Classic dynamic-programming minimum edit distance (insert/delete/substitute = 1)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1,                 # deletion
                           cur[j - 1] + 1,              # insertion
                           prev[j - 1] + (ca != cb)))   # substitution
        prev = cur
    return prev[-1]


def within_one_edit(a: str, b: str) -> bool:
    """Fast check of levenshtein(a, b) <= 1 in linear time (same result, much faster)."""
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la > lb:
        a, b, la, lb = b, a, lb, la
    i = j = edits = 0
    while i < la and j < lb:
        if a[i] != b[j]:
            edits += 1
            if edits > 1:
                return False
            if la == lb:
                i += 1          # substitution
            j += 1              # (insertion into a) or substitution
        else:
            i += 1
            j += 1
    return edits + (lb - j) <= 1


def build_lexicon(texts, labels, top_n=200, min_count=20, min_precision=0.8, min_len=3):
    """Words with P(offensive | word) >= min_precision, sorted by frequency in offensive messages."""
    off_df, all_df = Counter(), Counter()
    for text, y in zip(texts, labels):
        words = set(tokenize(normalize_text(text)))
        all_df.update(words)
        if y == 1:
            off_df.update(words)
    candidates = [(w, c) for w, c in off_df.items()
                  if c >= min_count and len(w) >= min_len and c / all_df[w] >= min_precision]
    candidates.sort(key=lambda x: -x[1])
    return [w for w, _ in candidates[:top_n]]


class LexiconMatcher:
    def __init__(self, lexicon, min_hits=1, fuzzy_min_len=4):
        self.lexicon = set(lexicon)
        self.by_len = {}
        for w in self.lexicon:
            self.by_len.setdefault(len(w), []).append(w)
        self.min_hits = min_hits
        self.fuzzy_min_len = fuzzy_min_len      # short words: exact match only (too many false hits)
        self._match = lru_cache(maxsize=200_000)(self._match_token)

    def _match_token(self, tok: str):
        if tok in self.lexicon:
            return tok
        if len(tok) < self.fuzzy_min_len:
            return None
        for L in (len(tok) - 1, len(tok), len(tok) + 1):
            for w in self.by_len.get(L, []):
                if within_one_edit(tok, w):
                    return w
        return None

    def hits(self, text: str):
        """List of (token, matched_lexicon_word)."""
        out = []
        for tok in tokenize(normalize_text(text)):
            m = self._match(tok)
            if m is not None:
                out.append((tok, m))
        return out

    def predict_proba(self, text: str) -> float:
        """Heuristic confidence: 0 hits -> 0.0, 1 hit -> 0.5, 2 -> 0.75, ... (not a calibrated probability)."""
        return 1.0 - 0.5 ** len(self.hits(text))

    def predict(self, text: str) -> int:
        return int(len(self.hits(text)) >= self.min_hits)
