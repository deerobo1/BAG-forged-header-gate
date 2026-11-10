"""Shortcut-aware contrastive pretraining for LitCVit, against our LitCVit, with header-perturbed tests.

LitCVit does not say how it forms positive pairs. Ours (litcvit.augment) differ by masked
bytes and a dropped packet, so the encoder is free to keep the header fields that identify
a host or a clock, the shortcuts our audit found (TTL, IP ID, TCP window, timestamps, ...).
Shortcut-aware pairs (litcvit.augment_sa) also set those fields (litcvit.shortcut_fields)
to random bytes, independently in each view, so matching the two views pays only for what
the rest of the flow says. Fine-tuning keeps the stem and byte block frozen, so the
classifier sees only what the pretrained encoder kept.

Models, for SPLIT and seed 0, both fine-tuned exactly as in litcvit.py on the unchanged images:
  litcvit-sa  pretrained here with shortcut-aware pairs, otherwise as litcvit.py
  litcvit     our LitCVit: its pretrained encoder_<SPLIT>_0.pt, re-fine-tuned to test it the same way
Each is tested on the clean test flows and on the same flows with the shortcut fields set to
random bytes ("random") or zeroed ("zero"), a stand-in for traffic from other hosts or times.

Needs deetyam/litcvit-flows and deetyam/litcvit-encoder, which holds litcvit.py (imported
from there) and the encoders:
  kaggle datasets version -p litcvit-encoder -m "..."   (from C:\\data\\CNS_proj\\kaggle)
Writes /kaggle/working/sa.csv (columns of litcvit.csv plus condition), encoder_sa_<SPLIT>_0.pt
and preds/<dataset>_<SPLIT>_<model>_0.npz (row, y, classes, and logits per condition).

Push with:  kaggle kernels push -p kaggle/litcvit_sa
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
if FOUND is None:  # local test: model code from the repo, encoders from the downloaded output
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "litcvit"))
    ENCODERS = Path("C:/data/CNS_proj/kaggle/litcvit_output_random")
else:
    sys.path.insert(0, str(FOUND.parent))
    ENCODERS = FOUND.parent
import litcvit as L  # noqa: E402

SPLIT, SEED = "random", 0
TESTS = ("random", "zero")


def record(results, run, model, out, seconds, pretrain):
    rows, y, logits, classes, n_train = out
    for condition, z in logits.items():
        pred = z.argmax(1)
        recall = recall_score(y, pred, labels=range(len(classes)), average=None, zero_division=0)
        results.append({"dataset": run, "split": SPLIT, "model": model, "condition": condition, "seed": SEED,
                        "train": n_train, "test": len(y),
                        "accuracy": round(100 * accuracy_score(y, pred), 2),
                        "macro_f1": round(100 * f1_score(y, pred, average="macro"), 2),
                        "weighted_f1": round(100 * f1_score(y, pred, average="weighted"), 2),
                        **pretrain, "seconds": seconds,
                        "recall": json.dumps({c: round(100 * r, 1) for c, r in zip(classes, recall)})})
        print(f"{run:12s} {model:10s} {condition:6s}: acc {results[-1]['accuracy']:6.2f}  "
              f"macro-F1 {results[-1]['macro_f1']:6.2f}", flush=True)
    pd.DataFrame(results).to_csv(L.OUT / "sa.csv", index=False)
    np.savez_compressed(L.OUT / "preds" / f"{run}_{SPLIT}_{model}_{SEED}.npz", row=rows, y=y,
                        classes=np.array(classes), **{c: z.astype(np.float32) for c, z in logits.items()})


def main():
    (L.OUT / "preds").mkdir(exist_ok=True)
    print(f"data in {L.DATA}, encoders in {ENCODERS}, device {L.DEVICE}", flush=True)
    baseline = L.Encoder().to(L.DEVICE)
    baseline.load_state_dict(torch.load(ENCODERS / f"encoder_{SPLIT}_{SEED}.pt", map_location=L.DEVICE,
                                        weights_only=True))
    start = time.time()
    sa, epochs = L.pretrain(SPLIT, SEED, view=L.augment_sa)
    torch.save(sa.state_dict(), L.OUT / f"encoder_sa_{SPLIT}_{SEED}.pt")
    pretrain = {"pretrain_epochs": epochs, "pretrain_hours": round((time.time() - start) / 3600, 2)}

    results = []
    for model, encoder, info in (("litcvit-sa", sa, pretrain), ("litcvit", baseline, {})):
        for run in L.RUNS:
            start = time.time()
            out = L.finetune(encoder, run, SPLIT, SEED, tests=TESTS)
            record(results, run, model, out, round(time.time() - start), info)


if __name__ == "__main__":
    main()
