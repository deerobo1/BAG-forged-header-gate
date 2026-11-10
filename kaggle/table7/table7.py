"""LitCVit Table VII baselines on Kaggle: XGBoost on the flattened 1,600 bytes and a 2D CNN on the 40 x 40 image.

Reads the private dataset deetyam/litcvit-flows (kaggle/pack_flows.py) and runs
each dataset under two splits:
  random  stratified 80/20 over all packed flows of each class
  time    within each class and capture, the earliest 80% of flows (by first
          packet) train and the rest test
MCFP runs twice: mcfp counts every flow with the infected host, mcfp_strict only
flows the host starts, minus classes left with under 100 flows (datasets.md
section 4). Every other dataset keeps LitCVit's classes.

The paper gives no settings for these baselines, so ours are:
  xgb  300 trees, depth 8, learning rate 0.1, GPU hist, balanced sample weights
  cnn  conv 5x5 x32, pool, conv 5x5 x64, pool, dense 1024, dropout 0.5;
       Adam 1e-3, batch 256, 30 epochs, class-weighted cross-entropy
Seeds 0-2. Writes /kaggle/working/table7.csv (one row per run, appended as runs
finish) and preds/<dataset>_<split>_<model>_<seed>.npz (test rows of the
dataset's CSV, true class, class probabilities).

Push with:  kaggle kernels push -p kaggle/table7
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, recall_score

DATA = next(Path("/kaggle/input").rglob("classes.csv")).parent
OUT = Path("/kaggle/working")
RUNS = ["5gad2022", "mcfp", "mcfp_strict", "ciciot2022", "tii_ssrc23", "iot23", "ustc"]
SPLITS = ["random", "time"]
SEEDS = [0, 1, 2]
MIN_STRICT = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load(run):
    """Flow metadata (with its CSV row) and images [N, 40, 40] for one run."""
    name = run.removesuffix("_strict")
    meta = pd.read_csv(DATA / f"{name}.csv", keep_default_na=False)
    meta["row"] = np.arange(len(meta))
    X = np.load(DATA / f"{name}.npy")
    if run.endswith("_strict"):
        keep = meta["strict"].to_numpy() == 1
        sizes = meta[keep]["class"].value_counts()
        keep &= meta["class"].isin(sizes[sizes >= MIN_STRICT].index).to_numpy()
        meta, X = meta[keep].reset_index(drop=True), X[keep]
    return meta, X


def split(meta, kind, rng):
    """Train and test positions in meta."""
    train = np.zeros(len(meta), bool)
    ts = meta["ts_first"].to_numpy()
    groups = meta.groupby("class").indices if kind == "random" else meta.groupby(["class", "capture"]).indices
    for idx in groups.values():
        idx = rng.permutation(idx) if kind == "random" else idx[np.argsort(ts[idx], kind="stable")]
        train[idx[:int(round(0.8 * len(idx)))]] = True
    return np.flatnonzero(train), np.flatnonzero(~train)


def class_weights(y, k):
    counts = np.bincount(y, minlength=k)
    return len(y) / (k * np.maximum(counts, 1))


def fit_xgb(X_train, y_train, X_test, k, seed):
    model = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, tree_method="hist",
                              device=DEVICE, random_state=seed)
    model.fit(X_train.reshape(len(X_train), -1), y_train,
              sample_weight=class_weights(y_train, k)[y_train])
    return model.predict_proba(X_test.reshape(len(X_test), -1))


class CNN(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 10 * 10, 1024), nn.ReLU(), nn.Dropout(0.5), nn.Linear(1024, k))

    def forward(self, x):
        return self.net(x.unsqueeze(1).float() / 255)


def fit_cnn(X_train, y_train, X_test, k, seed, epochs=30, batch=256):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = CNN(k).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    weight = torch.tensor(class_weights(y_train, k), dtype=torch.float32, device=DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    X, y = torch.from_numpy(X_train).to(DEVICE), torch.from_numpy(y_train).to(DEVICE)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X), generator=gen).to(DEVICE)
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            loss = loss_fn(model(X[b]), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    X_test = torch.from_numpy(X_test).to(DEVICE)
    with torch.no_grad():
        return torch.cat([torch.softmax(model(X_test[i:i + 2048]), 1).cpu()
                          for i in range(0, len(X_test), 2048)]).numpy()


MODELS = {"xgb": fit_xgb, "cnn": fit_cnn}


def main():
    (OUT / "preds").mkdir(exist_ok=True)
    results = []
    print(f"data in {DATA}, device {DEVICE}", flush=True)
    for run in RUNS:
        meta, X = load(run)
        classes = sorted(meta["class"].unique())
        y = meta["class"].map({c: i for i, c in enumerate(classes)}).to_numpy()
        print(f"\n{run}: {len(meta):,} flows, classes {dict(meta['class'].value_counts().sort_index())}", flush=True)
        for kind in SPLITS:
            for seed in SEEDS:
                train, test = split(meta, kind, np.random.default_rng(seed))
                for name, fit in MODELS.items():
                    start = time.time()
                    proba = fit(X[train], y[train], X[test], len(classes), seed)
                    pred = proba.argmax(1)
                    recall = recall_score(y[test], pred, labels=range(len(classes)), average=None, zero_division=0)
                    row = {"dataset": run, "split": kind, "model": name, "seed": seed,
                           "train": len(train), "test": len(test),
                           "accuracy": round(100 * accuracy_score(y[test], pred), 2),
                           "macro_f1": round(100 * f1_score(y[test], pred, average="macro"), 2),
                           "seconds": round(time.time() - start),
                           "recall": json.dumps({c: round(100 * r, 1) for c, r in zip(classes, recall)})}
                    results.append(row)
                    pd.DataFrame(results).to_csv(OUT / "table7.csv", index=False)
                    np.savez_compressed(OUT / "preds" / f"{run}_{kind}_{name}_{seed}.npz",
                                        row=meta["row"].to_numpy()[test], y=y[test],
                                        proba=proba.astype(np.float16), classes=np.array(classes))
                    print(f"  {kind:6s} seed {seed} {name}: acc {row['accuracy']:6.2f}  "
                          f"macro-F1 {row['macro_f1']:6.2f}  ({row['seconds']}s)", flush=True)

    table = pd.DataFrame(results).groupby(["dataset", "split", "model"])[["accuracy", "macro_f1"]]
    summary = table.mean().round(2).join(table.std().round(2), rsuffix="_sd")
    summary.to_csv(OUT / "table7_summary.csv")
    print("\nMean over seeds (sd):")
    print(summary.to_string())


if __name__ == "__main__":
    main()
