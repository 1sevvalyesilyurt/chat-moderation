"""Re-evaluate the SAVED word2vec+FFNN model on test.csv (no retraining).

If this prints the same macro-F1 as the training log, the saved files and the
test split are the same ones that were used in training.
Run: python -m src.eval_w2v_ffnn
"""
import pandas as pd

from src.metrics import evaluate, save_test_result
from src.w2v_ffnn import W2VFFNNClassifier

test = pd.read_csv("data/processed/test.csv")
clf = W2VFFNNClassifier("models")
proba = [clf.predict_proba(t) for t in test.text]
pred = [int(p >= clf.threshold) for p in proba]
res = evaluate(test.label, pred)
res["threshold"] = clf.threshold
res["vocab_size"] = len(clf.vocab)
save_test_result("w2v_ffnn", res)
print("test rows:", len(test), "| offensive:", int(test.label.sum()))
print("w2v_ffnn TEST macro-F1 (re-evaluated from saved files):", res["macro_f1"])
print("confusion matrix:", res["confusion_matrix [[TN,FP],[FN,TP]]"])
