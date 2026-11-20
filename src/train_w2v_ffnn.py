"""Phase 4: train our own word2vec on the TRAINING messages, then a feedforward network on top.

- word2vec: gensim skip-gram, trained only on train.csv (no pretrained vectors)
- FFNN: trained with SGD (momentum) + backpropagation, early stopping on val macro-F1
- test.csv evaluated once at the end
Run: python -m src.train_w2v_ffnn
"""
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from torch import nn

from src.metrics import best_threshold, evaluate, save_test_result
from src.preprocess import normalize_text, tokenize
from src.w2v_ffnn import FFNN, average_vector

SEED, DIM, DATA, MODELS = 42, 100, "data/processed", "models"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
os.makedirs(MODELS, exist_ok=True)

train = pd.read_csv(f"{DATA}/train.csv")
val = pd.read_csv(f"{DATA}/val.csv")
test = pd.read_csv(f"{DATA}/test.csv")

# ------------------------------------------------------------ 1) word2vec
sentences = [tokenize(normalize_text(t)) for t in train.text]
print("training word2vec on", len(sentences), "train messages,",
      sum(map(len, sentences)), "tokens")
w2v = Word2Vec(sentences, vector_size=DIM, window=5, min_count=2, sg=1,
               negative=5, epochs=10, workers=1, seed=SEED)   # workers=1 -> reproducible
vocab = {w: i for i, w in enumerate(w2v.wv.index_to_key)}
vectors = w2v.wv.vectors.astype(np.float32)
print("vocabulary size:", len(vocab))
for probe in ("idiot", "stupid", "thanks", "article"):
    if probe in w2v.wv:
        print(f"  most similar to {probe!r}:", [w for w, _ in w2v.wv.most_similar(probe, topn=5)])
with open(f"{MODELS}/w2v_vocab.json", "w") as f:
    json.dump(vocab, f)
np.save(f"{MODELS}/w2v_vectors.npy", vectors)

# ------------------------------------------------------------ 2) features


def featurize(texts):
    return np.stack([average_vector(t, vocab, vectors) for t in texts])


Xtr, Xva, Xte = featurize(train.text), featurize(val.text), featurize(test.text)
mean, std = Xtr.mean(0), Xtr.std(0) + 1e-6          # standardize with TRAIN statistics only
Xtr, Xva, Xte = [(x - mean) / std for x in (Xtr, Xva, Xte)]
ytr = torch.tensor(train.label.values, dtype=torch.float32)

# ------------------------------------------------------------ 3) FFNN
arch = {"dim": DIM, "h1": 128, "h2": 64, "dropout": 0.3}
model = FFNN(**arch)
pos_weight = ((ytr == 0).sum() / (ytr == 1).sum()).float()   # class imbalance
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optim = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
Xtr_t, Xva_t = torch.from_numpy(Xtr), torch.from_numpy(Xva)

best, patience, bad = {"val_macro_f1": -1}, 5, 0
for epoch in range(1, 61):
    model.train()
    perm = torch.randperm(len(Xtr_t))
    total = 0.0
    for i in range(0, len(perm), 256):
        idx = perm[i:i + 256]
        loss = loss_fn(model(Xtr_t[idx]), ytr[idx])
        optim.zero_grad()
        loss.backward()                 # backpropagation
        optim.step()                    # SGD update
        total += loss.item() * len(idx)
    model.eval()
    with torch.no_grad():
        val_proba = torch.sigmoid(model(Xva_t)).numpy()
    thr, f1 = best_threshold(val.label, val_proba)
    print(f"epoch {epoch:2d} train loss {total / len(perm):.4f}  val macro-F1 {f1:.4f} (thr {thr:.2f})")
    if f1 > best["val_macro_f1"]:
        best = {"epoch": epoch, "threshold": thr, "val_macro_f1": round(f1, 4)}
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        bad = 0
    else:
        bad += 1
        if bad >= patience:
            print("early stopping")
            break

print("best on val:", best)
model.load_state_dict(best_state)
torch.save({"state_dict": best_state, "arch": arch, "feat_mean": mean.tolist(),
            "feat_std": std.tolist(), "threshold": best["threshold"]}, f"{MODELS}/ffnn.pt")

# ------------------------------------------------------------ 4) test, once
model.eval()
with torch.no_grad():
    proba = torch.sigmoid(model(torch.from_numpy(Xte))).numpy()
res = evaluate(test.label, (proba >= best["threshold"]).astype(int))
res["chosen_on_val"] = best
res["word2vec"] = {"dim": DIM, "sg": 1, "window": 5, "min_count": 2, "epochs": 10,
                   "vocab_size": len(vocab)}
save_test_result("w2v_ffnn", res)
print("TEST macro-F1:", res["macro_f1"])
print(json.dumps({k: v for k, v in res.items() if k != "report"}, indent=2))
