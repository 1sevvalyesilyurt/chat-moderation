"""Word2vec + feedforward network classifier (Technique area A + B embeddings).

Shared by training, benchmark and the app. At inference time gensim is NOT needed:
the word vectors are stored as a plain numpy matrix + vocabulary json.
"""
import json
import os

import numpy as np
import torch
from torch import nn

from src.preprocess import normalize_text, tokenize


class FFNN(nn.Module):
    """100-d averaged word2vec vector -> 128 -> 64 -> 1 logit (offensive)."""

    def __init__(self, dim=100, h1=128, h2=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def average_vector(text, vocab, vectors):
    """Mean of the word vectors of all known tokens; zero vector if no token is known."""
    ids = [vocab[t] for t in tokenize(normalize_text(text)) if t in vocab]
    if not ids:
        return np.zeros(vectors.shape[1], dtype=np.float32)
    return vectors[ids].mean(axis=0)


class W2VFFNNClassifier:
    def __init__(self, model_dir="models"):
        with open(os.path.join(model_dir, "w2v_vocab.json")) as f:
            self.vocab = json.load(f)
        self.vectors = np.load(os.path.join(model_dir, "w2v_vectors.npy"))
        ckpt = torch.load(os.path.join(model_dir, "ffnn.pt"), map_location="cpu")
        self.model = FFNN(**ckpt["arch"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.mean = np.array(ckpt["feat_mean"], dtype=np.float32)
        self.std = np.array(ckpt["feat_std"], dtype=np.float32)
        self.threshold = ckpt["threshold"]

    @torch.no_grad()
    def predict_proba(self, text: str) -> float:
        x = (average_vector(text, self.vocab, self.vectors) - self.mean) / self.std
        return float(torch.sigmoid(self.model(torch.from_numpy(x).unsqueeze(0)))[0])
