"""Phase 2: rule-based lexicon matcher + TF-IDF/Logistic Regression baseline.

- Everything is fitted on train.csv, hyper-parameters are chosen on val.csv.
- test.csv is used exactly once at the end for each model.
Run: python -m src.train_classical
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.lexicon import LexiconMatcher, build_lexicon
from src.metrics import best_threshold, evaluate, macro_f1, save_test_result
from src.preprocess import normalize_text

DATA, MODELS = "data/processed", "models"
os.makedirs(MODELS, exist_ok=True)

train = pd.read_csv(f"{DATA}/train.csv")
val = pd.read_csv(f"{DATA}/val.csv")
test = pd.read_csv(f"{DATA}/test.csv")
print("rows:", len(train), len(val), len(test))

# ---------------------------------------------------------------- 1) Lexicon
print("\n=== Edit-distance lexicon matcher ===")
full_lex = build_lexicon(train.text, train.label, top_n=1000)
best = None
for top_n in (50, 100, 200, 400):
    for min_hits in (1, 2):
        m = LexiconMatcher(full_lex[:top_n], min_hits=min_hits)
        f1 = macro_f1(val.label, [m.predict(t) for t in val.text])
        print(f"top_n={top_n:4d} min_hits={min_hits}  val macro-F1={f1:.4f}")
        if best is None or f1 > best[0]:
            best = (f1, top_n, min_hits)
_, top_n, min_hits = best
lexicon = full_lex[:top_n]
print(f"chosen on val: top_n={top_n}, min_hits={min_hits}")
with open(f"{MODELS}/lexicon.json", "w") as f:
    json.dump({"words": lexicon, "min_hits": min_hits}, f, indent=1)

matcher = LexiconMatcher(lexicon, min_hits=min_hits)
res = evaluate(test.label, [matcher.predict(t) for t in test.text])
res["chosen_on_val"] = {"top_n": top_n, "min_hits": min_hits, "val_macro_f1": round(best[0], 4)}
save_test_result("lexicon_edit_distance", res)
print("TEST macro-F1:", res["macro_f1"])

# ---------------------------------------------------------------- 2) TF-IDF + LR
print("\n=== TF-IDF + Logistic Regression ===")
Xtr = [normalize_text(t) for t in train.text]
Xva = [normalize_text(t) for t in val.text]
Xte = [normalize_text(t) for t in test.text]

best = None
for C in (0.5, 1.0, 4.0):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000,
                                  sublinear_tf=True)),
        ("lr", LogisticRegression(C=C, class_weight="balanced", max_iter=2000)),
    ])
    pipe.fit(Xtr, train.label)
    thr, f1 = best_threshold(val.label, pipe.predict_proba(Xva)[:, 1])
    print(f"C={C:<4} best threshold={thr:.2f}  val macro-F1={f1:.4f}")
    if best is None or f1 > best[0]:
        best = (f1, C, thr, pipe)
val_f1, C, thr, pipe = best
print(f"chosen on val: C={C}, threshold={thr:.2f}")
joblib.dump({"pipeline": pipe, "threshold": thr}, f"{MODELS}/tfidf_lr.joblib")

proba = pipe.predict_proba(Xte)[:, 1]
res = evaluate(test.label, (proba >= thr).astype(int))
res["chosen_on_val"] = {"C": C, "threshold": thr, "val_macro_f1": round(val_f1, 4)}
save_test_result("tfidf_logreg", res)
print("TEST macro-F1:", res["macro_f1"])
print(json.dumps({k: v for k, v in res.items() if k != "report"}, indent=2))

# a few test examples the model got wrong (useful for the "failure example" slide)
wrong = test.assign(proba=proba, pred=(proba >= thr).astype(int)).query("pred != label")
print("\nSome TF-IDF errors:")
for _, r in wrong.sample(min(5, len(wrong)), random_state=0).iterrows():
    print(f"label={r.label} proba={r.proba:.2f} | {str(r.text)[:120]!r}")
