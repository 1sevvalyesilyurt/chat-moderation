"""Run with:  python tests/test_preprocess.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import normalize_text, fold_turkish  # noqa: E402

CASES = [
    ("You are SOOOO STUPIIIID!!!", "you are soo stupiid"),
    ("f4ck you, sh1t h3ad", "fack you shit head"),
    ("@ss and a$$", "ass and ass"),
    ("sh!t, hello!", "shit hello"),
    ("Year 2020 was ok", "year 2020 was ok"),
    ("İstanbul'da ŞAŞIRDIM çok", "istanbul'da sasirdim cok"),
    ("see http://example.com from 192.168.0.1", "see from"),
]

ok = True
for raw, expected in CASES:
    got = normalize_text(raw)
    status = "OK  " if got == expected else "FAIL"
    ok &= got == expected
    print(f"{status} {raw!r:40} -> {got!r}  (expected {expected!r})")

print("light:", repr(normalize_text("You are SOOOO STUPIIIID!!!", mode="light")))
print("turkish fold:", fold_turkish("ığüşöçİ"))
print("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")
sys.exit(0 if ok else 1)
