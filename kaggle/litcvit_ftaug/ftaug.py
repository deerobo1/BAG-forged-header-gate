"""Shortcut fields randomised in fine-tuning as well: our LitCVit and the shortcut-aware encoder.

The SA job (kaggle/litcvit_sa/sa.py) showed that pretraining with the shortcut fields
randomised makes the frozen packet tokens far less sensitive to them, but a classifier
fine-tuned on clean flows still learns the residue (kaggle/litcvit/sa_probe.py, plan.txt
Progress Oct 30). Here the classifier never sees a clean training flow:
litcvit.finetune(..., train_copies=COPIES) caches COPIES copies of every training flow with
its shortcut fields set to fresh random bytes, and each step takes every flow from one of
them. Models, random split, seed 0:
  litcvit-sa-ftaug  the SA job's encoder_sa_random_0.pt
  litcvit-ftaug     our LitCVit's encoder (version 2's encoder_random_0.pt): is SA
                    pretraining needed once fine-tuning is randomised?
Otherwise fine-tuned and tested as in sa.py (clean, "random" and "zero" test flows, with the
same perturbed bytes), so rows compare one to one with results/litcvit_sa.csv, plus the two
tests with the payload zeroed (litcvit.perturb), which run after those and don't change them.

Needs deetyam/litcvit-flows and deetyam/litcvit-encoder, which holds litcvit.py (imported
from there) and both encoders:
  kaggle datasets version -p litcvit-encoder -m "..."   (from C:\\data\\CNS_proj\\kaggle)
Writes /kaggle/working/ftaug.csv (columns of sa.csv) and preds/<dataset>_random_<model>_0.npz
(row, y, classes, and logits per condition). About 5 h on Kaggle's GPU: 2.3 h of fine-tuning
per model, as in the SA job, plus about 15 min caching the copies.

Push with:  kaggle kernels push -p kaggle/litcvit_ftaug
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score

FOUND = next(Path("/kaggle/input").rglob("litcvit.py"), None)
if FOUND is None:  # local test: model code from the repo, encoders from the downloaded outputs
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "litcvit"))
    KAGGLE = Path("C:/data/CNS_proj/kaggle")
    ENCODERS = {"litcvit-sa-ftaug": KAGGLE / "litcvit_output_sa" / "encoder_sa_random_0.pt",
                "litcvit-ftaug": KAGGLE / "litcvit_output_random" / "encoder_random_0.pt"}
else:
    sys.path.insert(0, str(FOUND.parent))
    ENCODERS = {"litcvit-sa-ftaug": FOUND.parent / "encoder_sa_random_0.pt",
                "litcvit-ftaug": FOUND.parent / "encoder_random_0.pt"}
import litcvit as L  # noqa: E402

SPLIT, SEED = "random", 0
TESTS = L.PERTURBATIONS  # stops at once if the dataset still holds an older litcvit.py
COPIES = 16  # USTC-TFC2016's copies take ~2.6 GB of GPU memory


def record(results, run, model, out, seconds):
    rows, y, logits, classes, n_train = out
    for condition, z in logits.items():
        pred = z.argmax(1)
        recall = recall_score(y, pred, labels=range(len(classes)), average=None, zero_division=0)
        results.append({"dataset": run, "split": SPLIT, "model": model, "condition": condition, "seed": SEED,
                        "train": n_train, "test": len(y),
                        "accuracy": round(100 * accuracy_score(y, pred), 2),
                        "macro_f1": round(100 * f1_score(y, pred, average="macro"), 2),
                        "weighted_f1": round(100 * f1_score(y, pred, average="weighted"), 2),
                        "copies": COPIES, "seconds": seconds,
                        "recall": json.dumps({c: round(100 * r, 1) for c, r in zip(classes, recall)})})
        print(f"{run:12s} {model:16s} {condition:16s}: acc {results[-1]['accuracy']:6.2f}  "
              f"macro-F1 {results[-1]['macro_f1']:6.2f}", flush=True)
    pd.DataFrame(results).to_csv(L.OUT / "ftaug.csv", index=False)
    np.savez_compressed(L.OUT / "preds" / f"{run}_{SPLIT}_{model}_{SEED}.npz", row=rows, y=y,
                        classes=np.array(classes), **{c: z.astype(np.float32) for c, z in logits.items()})


def main():
    (L.OUT / "preds").mkdir(exist_ok=True)
    print(f"data in {L.DATA}, encoders {list(ENCODERS.values())}, device {L.DEVICE}", flush=True)
    results = []
    for model, path in ENCODERS.items():
        encoder = L.Encoder().to(L.DEVICE)
        encoder.load_state_dict(torch.load(path, map_location=L.DEVICE, weights_only=True))
        for run in L.RUNS:
            start = time.time()
            out = L.finetune(encoder, run, SPLIT, SEED, tests=TESTS, train_copies=COPIES)
            record(results, run, model, out, round(time.time() - start))


if __name__ == "__main__":
    main()
