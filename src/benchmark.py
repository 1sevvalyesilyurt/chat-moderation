"""Phase 5: dynamic INT8 quantization + unified CPU benchmark for ALL models (R3, R4, Area D).

Measures every model on the EXACT SAME held-out test.csv and the EXACT SAME CPU:
  - macro-F1 (full 5,000-row test.csv)
  - model size on disk (MB, where 1 MB = 1024*1024 bytes)
  - end-to-end single-message latency (ms): 10 warmup + 200 timed runs (batch=1),
    including preprocessing and threshold decision.

Run on Colab CPU:  python -m src.benchmark
"""
import argparse
import gc
import json
import os
import platform
import tempfile
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers

from src.bert_infer import BertClassifier, HF_REPO, quantized_linear_modules
from src.lexicon import LexiconMatcher
from src.metrics import evaluate
from src.preprocess import normalize_text
from src.w2v_ffnn import W2VFFNNClassifier

LATENCY_BUDGET_MS = 100.0
WARMUP_RUNS = 10
BENCH_RUNS = 200
SEED = 42
MB = 1024 * 1024


def hardware_info() -> dict:
    cpu_model = platform.processor() or "unknown"
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    ram_gb = None
    if os.path.exists("/proc/meminfo"):
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    ram_gb = round(int(line.split()[1]) / (1024 * 1024), 2)
                    break
    return {
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "cpu_logical_cores": os.cpu_count(),
        "ram_gb": ram_gb,
        "torch_num_threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sklearn": sklearn.__version__,
    }


def files_size_mb(paths) -> float:
    return round(sum(os.path.getsize(p) for p in paths) / MB, 4)


def state_dict_size_mb(model) -> float:
    """Save state_dict to a temporary file in /tmp (so Drive is not filled) and measure MB."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        path = tmp.name
    try:
        torch.save(model.state_dict(), path)
        return round(os.path.getsize(path) / MB, 4)
    finally:
        if os.path.exists(path):
            os.remove(path)


def benchmark_latency(predict_one_fn, warmup_texts, bench_texts) -> dict:
    for t in warmup_texts:
        predict_one_fn(t)
    times_ms = []
    for t in bench_texts:
        t0 = time.perf_counter()
        predict_one_fn(t)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times_ms)
    return {
        "warmup_runs": len(warmup_texts),
        "timed_runs": len(bench_texts),
        "mean_ms": round(float(arr.mean()), 3),
        "median_ms": round(float(np.median(arr)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "min_ms": round(float(arr.min()), 3),
        "max_ms": round(float(arr.max()), 3),
    }


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main(args):
    hw = hardware_info()
    print("=== Hardware & Environment ===")
    print(json.dumps(hw, indent=2), flush=True)

    test = pd.read_csv(f"{args.data}/test.csv")
    texts = test.text.tolist()
    y_true = test.label.values
    print(f"\nHeld-out test set: {len(test)} rows ({int(y_true.sum())} offensive)", flush=True)

    # Fixed random sample from test.csv for warmup + latency benchmark (same texts for all models)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(texts), size=WARMUP_RUNS + BENCH_RUNS, replace=False)
    warmup_texts = [texts[i] for i in idx[:WARMUP_RUNS]]
    bench_texts = [texts[i] for i in idx[WARMUP_RUNS:]]
    avg_words = round(float(np.mean([len(str(t).split()) for t in bench_texts])), 1)
    print(f"Latency sample: {WARMUP_RUNS} warmup + {BENCH_RUNS} timed (avg {avg_words} words/msg)",
          flush=True)

    out_json = f"{args.results}/benchmark_results.json"
    out_preds = f"{args.results}/test_predictions.csv"
    report = {
        "hardware": hw,
        "latency_budget_ms": LATENCY_BUDGET_MS,
        "size_unit": "MiB (bytes / 1024^2)",
        "test_rows": len(test),
        "test_offensive": int(y_true.sum()),
        "latency_sample_avg_words": avg_words,
        "models": {},
    }
    preds_df = test[["id", "text", "label"]].copy()

    def record(key, role, precision, size_mb, size_note, lat, proba, threshold, extra=None):
        pred = (np.asarray(proba) >= threshold).astype(int)
        ev = evaluate(y_true, pred)
        entry = {
            "role": role,
            "precision": precision,
            "macro_f1": ev["macro_f1"],
            "accuracy": ev["accuracy"],
            "f1_offensive": ev["f1_offensive"],
            "confusion_matrix [[TN,FP],[FN,TP]]": ev["confusion_matrix [[TN,FP],[FN,TP]]"],
            "threshold": threshold,
            "size_mb": size_mb,
            "size_note": size_note,
            "latency": lat,
            "meets_budget_100ms": bool(lat["mean_ms"] <= LATENCY_BUDGET_MS),
        }
        if extra:
            entry.update(extra)
        report["models"][key] = entry
        preds_df[f"{key}_proba"] = np.round(proba, 4)
        preds_df[f"{key}_pred"] = pred
        save_json(report, out_json)
        preds_df.to_csv(out_preds, index=False)
        print(f"-> {key}: macro-F1={ev['macro_f1']:.4f} | size={size_mb:.4f} MB | "
              f"mean={lat['mean_ms']:.3f} ms (p95={lat['p95_ms']:.3f} ms) | "
              f"budget<=100ms: {entry['meets_budget_100ms']}", flush=True)

    # ------------------------------------------------- 1) Edit-distance Lexicon (Rule baseline)
    print("\n[1/5] Edit-distance lexicon matcher...", flush=True)
    lex_path = f"{args.models}/lexicon.json"
    with open(lex_path) as f:
        lex_cfg = json.load(f)
    matcher = LexiconMatcher(lex_cfg["words"], min_hits=lex_cfg["min_hits"])
    matcher._match.cache_clear()
    lat = benchmark_latency(lambda t: matcher.predict(t), warmup_texts, bench_texts)
    proba = [matcher.predict_proba(t) for t in texts]
    pred_lex = [matcher.predict(t) for t in texts]
    thr_lex = 1.0 - 0.5 ** lex_cfg["min_hits"]
    assert all((p >= thr_lex) == bool(y) for p, y in zip(proba, pred_lex))
    record("lexicon_edit_distance", "Rule-based baseline (extra, Area B)", "Rules / JSON",
           files_size_mb([lex_path]), "lexicon.json", lat, proba, thr_lex,
           extra={"lexicon_size": len(lex_cfg["words"]), "min_hits": lex_cfg["min_hits"]})

    # ------------------------------------------------- 2) TF-IDF + Logistic Regression (Model 1)
    print("\n[2/5] TF-IDF + Logistic Regression...", flush=True)
    tfidf_path = f"{args.models}/tfidf_lr.joblib"
    saved = joblib.load(tfidf_path)
    pipe, thr_tfidf = saved["pipeline"], float(saved["threshold"])
    lat = benchmark_latency(
        lambda t: int(pipe.predict_proba([normalize_text(t)])[0, 1] >= thr_tfidf),
        warmup_texts, bench_texts,
    )
    proba = pipe.predict_proba([normalize_text(t) for t in texts])[:, 1]
    record("tfidf_logreg", "1. Classical baseline (Area B)", "FP64 (sklearn)",
           files_size_mb([tfidf_path]), "tfidf_lr.joblib", lat, proba, thr_tfidf)
    del pipe, saved
    gc.collect()

    # ------------------------------------------------- 3) Word2Vec + FFNN (Model 2)
    print("\n[3/5] Own Word2Vec + FFNN...", flush=True)
    w2v_clf = W2VFFNNClassifier(args.models)
    w2v_files = [f"{args.models}/w2v_vocab.json",
                 f"{args.models}/w2v_vectors.npy",
                 f"{args.models}/ffnn.pt"]
    lat = benchmark_latency(
        lambda t: int(w2v_clf.predict_proba(t) >= w2v_clf.threshold),
        warmup_texts, bench_texts,
    )
    proba = [w2v_clf.predict_proba(t) for t in texts]
    record("w2v_ffnn", "2. Own neural network (Area A)", "FP32 (PyTorch)",
           files_size_mb(w2v_files), "w2v_vocab.json + w2v_vectors.npy + ffnn.pt",
           lat, proba, w2v_clf.threshold)
    del w2v_clf
    gc.collect()

    # ------------------------------------------------- 4 & 5) Load mBERT FP32 + create Dynamic INT8
    print(f"\n[4-5/5] Loading fine-tuned mBERT from {args.hf_repo} and quantizing to INT8...",
          flush=True)
    fp32_clf = BertClassifier(quantized=False, repo=args.hf_repo)
    int8_clf = BertClassifier(quantized=True, repo=args.hf_repo,
                              fp32_model=fp32_clf.model, tokenizer=fp32_clf.tok)
    assert len(quantized_linear_modules(fp32_clf.model)) == 0, "FP32 model was mutated!"
    print(f"  Quantization engine: {int8_clf.engine} | "
          f"quantized Linear layers: {int8_clf.quantized_layers}", flush=True)
    print(f"  Sample quantized layer: {int8_clf.sample_quant_layer}", flush=True)

    # Measure sizes & latencies back-to-back on the exact same 200 messages before heavy 5k eval
    fp32_mb = state_dict_size_mb(fp32_clf.model)
    int8_mb = state_dict_size_mb(int8_clf.model)
    print(f"  Disk state_dict size: FP32 = {fp32_mb:.2f} MB | INT8 = {int8_mb:.2f} MB "
          f"({(1 - int8_mb / fp32_mb) * 100:.1f}% smaller)", flush=True)

    print("  Timing FP32 (10 warmup + 200 single-message runs)...", flush=True)
    lat_fp32 = benchmark_latency(
        lambda t: int(fp32_clf.predict_proba(t) >= fp32_clf.threshold),
        warmup_texts, bench_texts,
    )
    print(f"  FP32 latency: mean={lat_fp32['mean_ms']:.2f} ms, "
          f"median={lat_fp32['median_ms']:.2f} ms, p95={lat_fp32['p95_ms']:.2f} ms", flush=True)

    print("  Timing INT8 (10 warmup + 200 single-message runs)...", flush=True)
    lat_int8 = benchmark_latency(
        lambda t: int(int8_clf.predict_proba(t) >= int8_clf.threshold),
        warmup_texts, bench_texts,
    )
    print(f"  INT8 latency: mean={lat_int8['mean_ms']:.2f} ms, "
          f"median={lat_int8['median_ms']:.2f} ms, p95={lat_int8['p95_ms']:.2f} ms "
          f"({lat_fp32['mean_ms'] / lat_int8['mean_ms']:.2f}x speedup)", flush=True)

    # Now evaluate full 5,000-row test.csv: INT8 first (priority), then FP32
    print("\n[4/5] Evaluating mBERT INT8 on all 5,000 test rows (CPU)...", flush=True)
    proba_int8 = int8_clf.predict_proba_batch(texts, batch_size=32, verbose=True)
    record("mbert_int8", "4. Quantized Transformer (Area D)", int8_clf.precision,
           int8_mb, "PyTorch state_dict (Linear=INT8, Embeddings=FP32; excl. tokenizer)",
           lat_int8, proba_int8, int8_clf.threshold,
           extra={"quant_engine": int8_clf.engine, "quantized_linear_layers": int8_clf.quantized_layers,
                  "hf_repo": args.hf_repo})
    del int8_clf
    gc.collect()

    print("\n[5/5] Evaluating mBERT FP32 on all 5,000 test rows (CPU)...", flush=True)
    proba_fp32 = fp32_clf.predict_proba_batch(texts, batch_size=32, verbose=True)
    record("mbert_fp32", "3. Fine-tuned Transformer (Area C)", fp32_clf.precision,
           fp32_mb, "PyTorch state_dict (all FP32; excl. tokenizer)",
           lat_fp32, proba_fp32, fp32_clf.threshold,
           extra={"hf_repo": args.hf_repo})

    # ------------------------------------------------- Summary table + failure examples
    print("\n" + "=" * 92)
    print("SUMMARY RESULTS TABLE (all measured on same held-out test.csv, n=5000, same CPU)")
    print("=" * 92)
    header = f"{'Model':24} | {'macro-F1':>8} | {'Size (MB)':>9} | {'Mean (ms)':>9} | {'p95 (ms)':>8} | {'<=100ms':>7}"
    print(header)
    print("-" * len(header))
    for k in ("lexicon_edit_distance", "tfidf_logreg", "w2v_ffnn", "mbert_fp32", "mbert_int8"):
        m = report["models"][k]
        print(f"{k:24} | {m['macro_f1']:8.4f} | {m['size_mb']:9.4f} | "
              f"{m['latency']['mean_ms']:9.3f} | {m['latency']['p95_ms']:8.3f} | "
              f"{str(m['meets_budget_100ms']):>7}")

    f32, i8 = report["models"]["mbert_fp32"], report["models"]["mbert_int8"]
    print("\nQuantization trade-off (FP32 -> Dynamic INT8):")
    print(f"  macro-F1 change : {i8['macro_f1'] - f32['macro_f1']:+.4f} ({f32['macro_f1']:.4f} -> {i8['macro_f1']:.4f})")
    print(f"  size reduction  : {(1 - i8['size_mb'] / f32['size_mb']) * 100:.1f}% ({f32['size_mb']:.2f} -> {i8['size_mb']:.2f} MB)")
    print(f"  speedup (mean)  : {f32['latency']['mean_ms'] / i8['latency']['mean_ms']:.2f}x "
          f"({f32['latency']['mean_ms']:.2f} -> {i8['latency']['mean_ms']:.2f} ms)")

    # Print real failure examples for the Conclusion slide
    print("\n=== Failure examples of quantized mBERT (mbert_int8) ===")
    wrong_i8 = preds_df[preds_df["mbert_int8_pred"] != preds_df["label"]]
    for lbl, title in [(0, "False Positive (clean flagged offensive)"),
                       (1, "False Negative (offensive missed)")]:
        sub = wrong_i8[wrong_i8["label"] == lbl]
        print(f"\n-- {title} (total {len(sub)}) --")
        for _, r in sub.sample(min(3, len(sub)), random_state=SEED).iterrows():
            print(f"  label={r.label} int8_p={r.mbert_int8_proba:.2f} fp32_p={r.mbert_fp32_proba:.2f} "
                  f"tfidf_p={r.tfidf_logreg_proba:.2f} | {str(r.text)[:140].replace(chr(10), ' ')!r}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--models", default="models")
    p.add_argument("--results", default="results")
    p.add_argument("--hf_repo", default=HF_REPO)
    main(p.parse_args())
