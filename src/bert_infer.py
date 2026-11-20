"""Multilingual BERT inference on CPU: FP32 and dynamic-INT8 quantized (Technique areas C + D).

Used by BOTH the benchmark and the app, so the measured model == the model in the app.
The fine-tuned FP32 weights are downloaded from the Hugging Face Hub; the INT8 model is
created from them with PyTorch dynamic quantization (nn.Linear weights -> int8).
"""
import copy
import json
import os
import time

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.preprocess import normalize_text

HF_REPO = "1sevvalyesilyurt/mbert-chat-moderation-swe015"
MAX_LEN = 128


def _pick_quant_engine() -> str:
    engines = torch.backends.quantized.supported_engines
    for e in ("x86", "fbgemm", "qnnpack"):          # Intel/AMD -> x86/fbgemm, Apple Silicon -> qnnpack
        if e in engines:
            torch.backends.quantized.engine = e
            return e
    raise RuntimeError(f"No quantized engine available: {engines}")


def quantized_linear_modules(model) -> list:
    """Return all dynamically quantized Linear submodules (verifies quantization really ran)."""
    return [
        m for m in model.modules()
        if "quantized.dynamic" in type(m).__module__ and type(m).__name__ == "Linear"
    ]


def quantize_dynamic_int8(fp32_model):
    engine = _pick_quant_engine()
    q = torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(fp32_model), {torch.nn.Linear}, dtype=torch.qint8
    )
    q.eval()
    q_layers = quantized_linear_modules(q)
    if not q_layers:
        raise RuntimeError("Dynamic quantization produced 0 quantized Linear layers!")
    return q, engine, len(q_layers), repr(q_layers[0])


class BertClassifier:
    def __init__(self, quantized: bool, repo: str = HF_REPO, fp32_model=None, tokenizer=None):
        self.tok = tokenizer or AutoTokenizer.from_pretrained(repo)
        cfg_path = (os.path.join(repo, "moderation_config.json")
                    if os.path.isdir(repo)
                    else hf_hub_download(repo, "moderation_config.json"))
        with open(cfg_path) as f:
            self.config = json.load(f)
        self.threshold = float(self.config["threshold"])
        model = fp32_model or AutoModelForSequenceClassification.from_pretrained(repo)
        model.float().eval()
        self.precision, self.engine = "FP32", None
        self.quantized_layers, self.sample_quant_layer = 0, None
        if quantized:
            model, self.engine, self.quantized_layers, self.sample_quant_layer = quantize_dynamic_int8(model)
            self.precision = "INT8 (dynamic, Linear layers)"
        self.model = model

    @torch.no_grad()
    def predict_proba(self, text: str) -> float:
        enc = self.tok(normalize_text(text, mode="light"), truncation=True, max_length=MAX_LEN,
                       return_tensors="pt")
        return float(torch.softmax(self.model(**enc).logits, dim=-1)[0, 1])

    @torch.no_grad()
    def predict_proba_batch(self, texts, batch_size: int = 32, verbose: bool = False):
        """Fast CPU evaluation: sort by length so batches need little padding, then restore order."""
        texts = [normalize_text(t, mode="light") for t in texts]
        order = np.argsort([len(t) for t in texts])
        out = np.zeros(len(texts), dtype=np.float64)
        n_batches = (len(texts) + batch_size - 1) // batch_size
        t0 = time.perf_counter()
        for b, i in enumerate(range(0, len(texts), batch_size), 1):
            idx = order[i:i + batch_size]
            enc = self.tok([texts[j] for j in idx], truncation=True, max_length=MAX_LEN,
                           padding=True, return_tensors="pt")
            out[idx] = torch.softmax(self.model(**enc).logits, dim=-1)[:, 1].numpy()
            if verbose and (b % 25 == 0 or b == n_batches):
                el = time.perf_counter() - t0
                eta = el / b * (n_batches - b)
                print(f"  batch {b:3d}/{n_batches} ({min(i + batch_size, len(texts))}/{len(texts)}) "
                      f"elapsed {el:.0f}s | ETA {eta:.0f}s", flush=True)
        return out
