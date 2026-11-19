"""Phase 3: fine-tune multilingual BERT (bert-base-multilingual-cased) for clean/offensive.

- trained on train.csv, best epoch + decision threshold chosen on val.csv
- test.csv evaluated once at the end (FP32 model, GPU)
Run (Colab GPU): python -m src.train_bert --epochs 2
"""
import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.metrics import best_threshold, evaluate, save_test_result
from src.preprocess import normalize_text

MODEL_NAME = "bert-base-multilingual-cased"
MAX_LEN = 128


@torch.no_grad()
def predict_proba(model, tok, texts, device, batch_size=64):
    """Return P(offensive) for each text. Uses the same 'light' normalization as training."""
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        batch = [normalize_text(t, mode="light") for t in texts[i:i + batch_size]]
        enc = tok(batch, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
        logits = model(**enc).logits.float()
        out.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
    return np.array(out)


def main(a):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

    train = pd.read_csv(f"{a.data}/train.csv")
    val = pd.read_csv(f"{a.data}/val.csv")
    test = pd.read_csv(f"{a.data}/test.csv")
    if a.max_train and a.max_train < len(train):     # emergency option if GPU time is short
        train = train.sample(a.max_train, random_state=a.seed)
    print("rows:", len(train), len(val), len(test))

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

    texts = [normalize_text(t, mode="light") for t in train.text]
    labels = train.label.tolist()

    def collate(batch):
        b_texts, b_labels = zip(*batch)
        enc = tok(list(b_texts), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(b_labels)
        return enc

    loader = DataLoader(list(zip(texts, labels)), batch_size=a.batch_size, shuffle=True, collate_fn=collate)

    # class weights against imbalance (offensive is the minority class)
    w = compute_class_weight("balanced", classes=np.array([0, 1]), y=np.array(labels))
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float, device=device))
    print("class weights:", w.round(3))

    optim = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    total = len(loader) * a.epochs
    sched = get_linear_schedule_with_warmup(optim, int(0.1 * total), total)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = {"val_macro_f1": -1}
    for epoch in range(1, a.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            y = batch.pop("labels")
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), y)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            sched.step()
            running += loss.item()
            if step % 200 == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} loss {running / step:.4f} "
                      f"({time.time() - t0:.0f}s)")

        val_proba = predict_proba(model, tok, val.text.tolist(), device)
        thr, f1 = best_threshold(val.label, val_proba)
        print(f"== epoch {epoch}: val macro-F1 {f1:.4f} at threshold {thr:.2f} ({time.time() - t0:.0f}s)")
        if f1 > best["val_macro_f1"]:
            best = {"epoch": epoch, "threshold": thr, "val_macro_f1": round(f1, 4)}
            model.save_pretrained(a.out)
            tok.save_pretrained(a.out)
            print("   saved as best model ->", a.out)

    cfg = {"base_model": MODEL_NAME, "max_len": MAX_LEN, "lr": a.lr, "batch_size": a.batch_size,
           "epochs": a.epochs, "train_rows": len(train), "seed": a.seed, **best}
    with open(os.path.join(a.out, "moderation_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print("best:", cfg)

    # ---- final test evaluation, once, with the best checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(a.out).to(device)
    proba = predict_proba(model, tok, test.text.tolist(), device)
    res = evaluate(test.label, (proba >= best["threshold"]).astype(int))
    res["chosen_on_val"] = best
    res["evaluated_on"] = device
    save_test_result("mbert_fp32", res)
    print("TEST macro-F1:", res["macro_f1"])
    print(json.dumps({k: v for k, v in res.items() if k != "report"}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--out", default="models/mbert_fp32")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_train", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
