"""Run with:  python tests/test_lexicon.py"""
import os
import random
import string
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.lexicon import LexiconMatcher, build_lexicon, levenshtein, within_one_edit  # noqa: E402

# 1) fast check must agree with the full DP edit distance
random.seed(0)
for _ in range(20000):
    a = "".join(random.choices("abc", k=random.randint(0, 6)))
    b = "".join(random.choices("abc", k=random.randint(0, 6)))
    assert within_one_edit(a, b) == (levenshtein(a, b) <= 1), (a, b)
assert levenshtein("kitten", "sitting") == 3
print("OK  within_one_edit agrees with levenshtein on 20000 random pairs")

# 2) lexicon building on a toy dataset
texts = ["you idiot"] * 30 + ["you are nice"] * 30 + ["idiot moron"] * 25
labels = [1] * 30 + [0] * 30 + [1] * 25
lex = build_lexicon(texts, labels, min_count=20)
assert lex == ["idiot", "moron"], lex
print("OK  build_lexicon ->", lex)

# 3) matching with leetspeak / repeated letters / typos
m = LexiconMatcher(["idiot", "moron"], min_hits=1)
for msg, expected in [("You 1d10t!!", 1), ("what a m0r0nnn", 1), ("idiiot", 1),
                      ("have a nice day", 0), ("ID", 0)]:
    got = m.predict(msg)
    assert got == expected, (msg, got, m.hits(msg))
    print(f"OK  {msg!r:20} -> {got}  hits={m.hits(msg)}")
print("ALL TESTS PASSED")
