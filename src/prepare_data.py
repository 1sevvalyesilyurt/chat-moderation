"""Build the clean/offensive dataset and the train/val/test split.

Input : data/raw/train.csv  (Kaggle: jigsaw-toxic-comment-classification-challenge)
Output: data/processed/{train,val,test}.csv  + data/processed/split_stats.json

Label rule: offensive = 1 if ANY of the six Jigsaw labels is 1, otherwise clean = 0.
Run: python -m src.prepare_data --subset 50000
"""
import argparse
import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
SEED = 42


def main(raw_path: str, out_dir: str, subset: int) -> None:
    df = pd.read_csv(raw_path)
    n_raw = len(df)

    df["label"] = (df[LABELS].sum(axis=1) > 0).astype(int)
    df = df[["id", "comment_text", "label"]].rename(columns={"comment_text": "text"})
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""]
    df = df.drop_duplicates(subset="text")          # avoid same text in train and test
    n_clean = len(df)

    if subset and subset < len(df):                  # stratified random subset
        df, _ = train_test_split(df, train_size=subset, stratify=df["label"], random_state=SEED)

    train, temp = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=SEED)

    os.makedirs(out_dir, exist_ok=True)
    stats = {"raw_rows": n_raw, "rows_after_cleaning": n_clean, "subset_size": len(df),
             "seed": SEED, "label_rule": "offensive if any of " + ", ".join(LABELS)}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
        stats[name] = {"rows": len(part),
                       "offensive": int(part["label"].sum()),
                       "offensive_ratio": round(float(part["label"].mean()), 4)}

    with open(os.path.join(out_dir, "split_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/raw/train.csv")
    p.add_argument("--out", default="data/processed")
    p.add_argument("--subset", type=int, default=50000)
    a = p.parse_args()
    main(a.raw, a.out, a.subset)
