"""Unified moderation engine & post-processing used by app.py (R1, R5, R6).

Handles:
  1) Automatic download of non-BERT artifacts from Hugging Face Hub (so no weights live in git).
  2) End-to-end single-message moderation on CPU across all 5 models, with latency timing.
  3) Post-processing: masking offensive messages while preserving word spacing.
"""
import json
import os
import re
import shutil
import time

import joblib
from huggingface_hub import hf_hub_download

from src.bert_infer import BertClassifier, HF_REPO
from src.lexicon import LexiconMatcher
from src.preprocess import normalize_text
from src.w2v_ffnn import W2VFFNNClassifier

LATENCY_BUDGET_MS = 100.0
ARTIFACT_FILES = [
    "lexicon.json",
    "tfidf_lr.joblib",
    "w2v_vocab.json",
    "w2v_vectors.npy",
    "ffnn.pt",
]

MODEL_CHOICES = [
    "mBERT INT8 (Quantized Transformer – Default)",
    "mBERT FP32 (Fine-tuned Transformer)",
    "Word2Vec + FFNN (Own Neural Network)",
    "TF-IDF + Logistic Regression (Classical Baseline)",
    "Edit-Distance Lexicon (Rule Baseline)",
]
DEFAULT_MODEL = MODEL_CHOICES[0]


def mask_message(text: str) -> str:
    """Replace word characters with '*' while keeping spaces and punctuation."""
    return re.sub(r"\w", "*", text)


def ensure_artifacts(model_dir: str = "models", repo: str = HF_REPO) -> None:
    """Ensure classical & word2vec artifacts exist locally; otherwise fetch from HF Hub."""
    os.makedirs(model_dir, exist_ok=True)
    for fname in ARTIFACT_FILES:
        dst = os.path.join(model_dir, fname)
        if not os.path.exists(dst):
            cached = hf_hub_download(repo_id=repo, filename=f"artifacts/{fname}")
            shutil.copyfile(cached, dst)


class ModerationEngine:
    """Loads all 5 models on CPU once so switching models in the UI is instantaneous."""

    def __init__(self, model_dir: str = "models", repo: str = HF_REPO):
        ensure_artifacts(model_dir=model_dir, repo=repo)

        with open(os.path.join(model_dir, "lexicon.json")) as f:
            lex_cfg = json.load(f)
        self.lexicon = LexiconMatcher(lex_cfg["words"], min_hits=lex_cfg["min_hits"])
        self.lex_min_hits = int(lex_cfg["min_hits"])

        saved_tfidf = joblib.load(os.path.join(model_dir, "tfidf_lr.joblib"))
        self.tfidf_pipe = saved_tfidf["pipeline"]
        self.tfidf_thr = float(saved_tfidf["threshold"])

        self.w2v_ffnn = W2VFFNNClassifier(model_dir=model_dir)

        # Download mBERT once from HF Hub, then build dynamic INT8 copy from it
        self.mbert_fp32 = BertClassifier(quantized=False, repo=repo)
        self.mbert_int8 = BertClassifier(
            quantized=True,
            repo=repo,
            fp32_model=self.mbert_fp32.model,
            tokenizer=self.mbert_fp32.tok,
        )

        self.info = {
            MODEL_CHOICES[0]: {
                "area": "D · Efficient Models (Quantized Transformer)",
                "precision": self.mbert_int8.precision,
                "threshold": round(self.mbert_int8.threshold, 2),
                "preprocess": "light (URL/IP + repeat collapse)",
                "details": (f"CPU · engine={self.mbert_int8.engine} · "
                            f"{self.mbert_int8.quantized_layers} DynamicQuantizedLinear layers"),
            },
            MODEL_CHOICES[1]: {
                "area": "C · Transformer (Fine-tuned mBERT)",
                "precision": self.mbert_fp32.precision,
                "threshold": round(self.mbert_fp32.threshold, 2),
                "preprocess": "light (URL/IP + repeat collapse)",
                "details": "CPU · bert-base-multilingual-cased (177.9M params)",
            },
            MODEL_CHOICES[2]: {
                "area": "A · Own Neural Network (Word2Vec + FFNN)",
                "precision": "FP32 (PyTorch)",
                "threshold": round(self.w2v_ffnn.threshold, 2),
                "preprocess": "full (Turkish fold + leetspeak + repeats)",
                "details": f"CPU · own 100-d Word2Vec ({len(self.w2v_ffnn.vocab):,} words) + 128-64-1 FFNN",
            },
            MODEL_CHOICES[3]: {
                "area": "B · Classical NLP (TF-IDF + Logistic Regression)",
                "precision": "FP64 (scikit-learn)",
                "threshold": round(self.tfidf_thr, 2),
                "preprocess": "full (Turkish fold + leetspeak + repeats)",
                "details": "CPU · 1-2 gram TF-IDF (max 100k features) + balanced LR",
            },
            MODEL_CHOICES[4]: {
                "area": "B · Classical NLP (Rule Baseline)",
                "precision": "Rules / Edit-Distance <= 1",
                "threshold": round(1.0 - 0.5 ** self.lex_min_hits, 2),
                "preprocess": "full (Turkish fold + leetspeak + repeats)",
                "details": f"CPU · {len(self.lexicon.lexicon)} train-learned words, min_hits={self.lex_min_hits}",
            },
        }

    def moderate(self, text: str, model_name: str = DEFAULT_MODEL) -> dict:
        t0 = time.perf_counter()
        if model_name == MODEL_CHOICES[0]:
            p_off = self.mbert_int8.predict_proba(text)
            thr = self.mbert_int8.threshold
            norm_preview = normalize_text(text, mode="light")
        elif model_name == MODEL_CHOICES[1]:
            p_off = self.mbert_fp32.predict_proba(text)
            thr = self.mbert_fp32.threshold
            norm_preview = normalize_text(text, mode="light")
        elif model_name == MODEL_CHOICES[2]:
            p_off = self.w2v_ffnn.predict_proba(text)
            thr = self.w2v_ffnn.threshold
            norm_preview = normalize_text(text, mode="full")
        elif model_name == MODEL_CHOICES[3]:
            norm_preview = normalize_text(text, mode="full")
            p_off = float(self.tfidf_pipe.predict_proba([norm_preview])[0, 1])
            thr = self.tfidf_thr
        elif model_name == MODEL_CHOICES[4]:
            norm_preview = normalize_text(text, mode="full")
            p_off = self.lexicon.predict_proba(text)
            thr = 1.0 - 0.5 ** self.lex_min_hits
        else:
            raise ValueError(f"Unknown model: {model_name}")
        latency_ms = (time.perf_counter() - t0) * 1000.0

        is_offensive = p_off >= thr
        label = "OFFENSIVE" if is_offensive else "CLEAN"
        confidence = p_off if is_offensive else (1.0 - p_off)
        display_text = mask_message(text) if is_offensive else text
        return {
            "model": model_name,
            "label": label,
            "is_offensive": is_offensive,
            "p_offensive": round(float(p_off), 4),
            "confidence": round(float(confidence), 4),
            "threshold": round(float(thr), 2),
            "latency_ms": round(float(latency_ms), 2),
            "within_budget": latency_ms <= LATENCY_BUDGET_MS,
            "display_text": display_text,
            "normalized": norm_preview,
        }

    def compare_all(self, text: str) -> list:
        rows = []
        for name in MODEL_CHOICES:
            r = self.moderate(text, model_name=name)
            rows.append([
                name,
                r["label"],
                f"{r['confidence'] * 100:.1f}%",
                f"{r['p_offensive']:.3f}",
                f"{r['latency_ms']:.2f}",
                "Yes" if r["within_budget"] else "No",
            ])
        return rows
