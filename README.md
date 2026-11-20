# Real-Time Chat Moderation (SWE015 – Use Case 4)

Interactive CPU chat moderation application that classifies live messages as **CLEAN** or **OFFENSIVE** with a confidence score, masks flagged messages, logs per-message response time, and lets the user switch between 5 models (default: **dynamic INT8 quantized multilingual BERT**).

- **Model weights & artifacts (Hugging Face Hub):** https://huggingface.co/1sevvalyesilyurt/mbert-chat-moderation-swe015
- **Dataset:** Kaggle *Jigsaw Toxic Comment Classification Challenge* (English; 50,000-row stratified subset, 80/10/10 train/val/test split, `seed=42`). Raw data and model weights are intentionally excluded from this repository.

---

## Quick Start (Fresh Environment)

Python 3.10+ is recommended. On first run, `app.py` automatically downloads all model weights and artifacts from Hugging Face Hub (~730 MB total) and builds the dynamic INT8 quantized mBERT model on CPU.

```bash
git clone https://github.com/1sevvalyesilyurt/chat-moderation.git
cd chat-moderation
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Single run command:**

```bash
python app.py
```

Then open the local URL printed in the terminal (default: `http://127.0.0.1:7860`).  
In Google Colab, run `python app.py --share` to generate a public Gradio link.

---

## Measured Results (Same Held-Out Test Set, n = 5,000, Same CPU)

All models were evaluated on the exact same held-out test split (`5,000` messages, `508` offensive = `10.16%`).  
**Hardware:** Google Colab CPU — AMD EPYC 7B12 (2 logical cores, `torch_num_threads=1`, 12.67 GB RAM), Linux x86_64, Python 3.13.18, PyTorch 2.14.1+cpu, transformers 5.22.1, scikit-learn 1.6.1.  
**Latency protocol:** 10 warmup runs + 200 timed single-message (`batch=1`) end-to-end runs on a fixed random sample from `test.csv` (`seed=42`, avg 75.6 words/message), including preprocessing. Pre-set latency budget: **≤ 100 ms/message on CPU**.

| Role | Model | Area | Precision | Test macro-F1 | Size (MB) | Mean Latency (ms) | p95 (ms) | Budget ≤100 ms |
|---|---|---|---|---:|---:|---:|---:|:---:|
| Rule baseline (extra) | Edit-distance lexicon (26 words, edit ≤ 1) | B | Rules / JSON | 0.6371 | 0.0003 | 0.188 | 0.675 | Yes |
| 1. Classical baseline | TF-IDF (1–2 gram) + Logistic Regression | B | FP64 (sklearn) | 0.8712 | 4.5561 | 1.016 | 1.998 | Yes |
| 2. Own neural network | Own Word2Vec (100-d) + FFNN (128-64-1) | A | FP32 (PyTorch) | 0.8558 | 16.0682 | 0.293 | 0.621 | Yes |
| 3. Transformer | Fine-tuned `bert-base-multilingual-cased` | C | FP32 (PyTorch) | 0.9020 | 678.5392 | 206.257 | 357.494 | No |
| **4. Quantized Transformer (App default)** | **Dynamic INT8 mBERT (74 Linear layers)** | **D** | **INT8 (dynamic, `x86`)** | **0.9032** | **433.9030** | **103.659** *(median 83.87)* | **202.807** | **No** *(mean)* |

- **Quantization trade-off (FP32 → Dynamic INT8):** `+0.0012` macro-F1 (`0.9020 → 0.9032`), `36.1%` disk size reduction (`678.54 → 433.90 MB`, embeddings stay FP32), `1.99×` mean CPU speedup (`206.26 → 103.66 ms`).

---

## Project Structure

```text
chat-moderation/
├── app.py                   # Gradio chat UI (default: dynamic INT8 mBERT)
├── requirements.txt         # Runtime dependencies
├── src/
│   ├── preprocess.py        # Area B: regex normalization (leetspeak, repeats, Turkish folding)
│   ├── prepare_data.py      # Label binarization + stratified 80/10/10 split
│   ├── lexicon.py           # Area B: edit-distance (Levenshtein <= 1) lexicon matcher
│   ├── train_classical.py   # Trains lexicon matcher + TF-IDF/Logistic Regression
│   ├── w2v_ffnn.py          # Area A: Word2Vec average-vector + PyTorch FFNN inference
│   ├── train_w2v_ffnn.py    # Trains Word2Vec (train split only) + FFNN (SGD + backprop)
│   ├── train_bert.py        # Area C: fine-tunes bert-base-multilingual-cased
│   ├── bert_infer.py        # Area C/D: CPU FP32 + dynamic INT8 mBERT inference
│   ├── benchmark.py         # Unified CPU benchmark (macro-F1, size MB, 200-run latency)
│   ├── moderation.py        # App engine, HF Hub artifact loader, offensive message masking
│   └── metrics.py           # Shared macro-F1 & threshold selection helpers
└── tests/
    ├── test_preprocess.py
    ├── test_lexicon.py
    └── test_app_smoke.py
```

---

## Reproducing Training & Benchmarks (Optional)

1. Download `train.csv` from the Kaggle *Jigsaw Toxic Comment Classification Challenge* into `data/raw/train.csv`.
2. Prepare splits: `python -m src.prepare_data --subset 50000`
3. Train classical models: `python -m src.train_classical`
4. Train own Word2Vec + FFNN (requires `pip install gensim`): `PYTHONHASHSEED=0 python -m src.train_w2v_ffnn`
5. Fine-tune mBERT (GPU recommended): `python -m src.train_bert --epochs 2`
6. Run unified CPU benchmark: `python -m src.benchmark`
