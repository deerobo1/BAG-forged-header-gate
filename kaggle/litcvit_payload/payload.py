"""What LitCVit itself reads: our LitCVit and the shortcut-aware encoder tested without the payload.

The header vs payload study (baselines/header_payload.py) uses a random forest, so it shows
what the bytes allow, not what LitCVit uses. And on TLS attack flows the shortcut-aware model
keeps most of its recall with the shortcut fields randomised (kaggle/litcvit/tls_flows.py),
but those tests can't tell whether it reads the payload (the TLS handshake is plaintext) or
only lengths and flags. sa.py didn't save its fine-tuned classifiers, so this job fine-tunes
the same two encoders again, on clean flows as sa.py did (random split, seed 0):
  litcvit     our LitCVit (version 2's encoder_random_0.pt)
  litcvit-sa  the SA job's encoder_sa_random_0.pt
and tests them on clean flows and on every litcvit.perturb condition:
  random, zero      the shortcut fields randomised or zeroed, the same bytes as sa.py
  nopayload         payload zeroed, header kept
  random_nopayload  payload zeroed and the shortcut fields randomised, which leaves what any
                    observer of encrypted traffic sees: versions, lengths, protocol, TCP flags

Needs deetyam/litcvit-flows and deetyam/litcvit-encoder, which holds litcvit.py (imported
from there) and both encoders:
  kaggle datasets version -p litcvit-encoder -m "..."   (from C:\\data\\CNS_proj\\kaggle)
Writes /kaggle/working/payload.csv (columns of sa.csv) and preds/<dataset>_random_<model>_0.npz
(row, y, classes, and logits per condition). About 4.5 h on Kaggle's GPU (2.25 h per model, as
in the SA job).

Push with:  kaggle kernels push -p kaggle/litcvit_payload
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
    ENCODERS = {"litcvit": KAGGLE / "litcvit_output_random" / "encoder_random_0.pt",
                "litcvit-sa": KAGGLE / "litcvit_output_sa" / "encoder_sa_random_0.pt"}
else:
    sys.path.insert(0, str(FOUND.parent))
    ENCODERS = {"litcvit": FOUND.parent / "encoder_random_0.pt",
                "litcvit-sa": FOUND.parent / "encoder_sa_random_0.pt"}
import litcvit as L  # noqa: E402

SPLIT, SEED = "random", 0
TESTS = L.PERTURBATIONS  # stops at once if the dataset still holds an older litcvit.py


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
                        "seconds": seconds,
                        "recall": json.dumps({c: round(100 * r, 1) for c, r in zip(classes, recall)})})
        print(f"{run:12s} {model:10s} {condition:16s}: acc {results[-1]['accuracy']:6.2f}  "
              f"macro-F1 {results[-1]['macro_f1']:6.2f}", flush=True)
    pd.DataFrame(results).to_csv(L.OUT / "payload.csv", index=False)
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
            out = L.finetune(encoder, run, SPLIT, SEED, tests=TESTS)
            record(results, run, model, out, round(time.time() - start))


if __name__ == "__main__":
    main()
