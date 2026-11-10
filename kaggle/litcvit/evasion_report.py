"""Scores for the evasion test from the saved test logits of kaggle/litcvit_evasion.

Only attack flows that had a donor count, since the others stay as recorded. For each run,
model, condition and class, on all such flows and on the TLS ones:
  recall_clean, recall  flows given their own class, as recorded and attacked
  benign_clean, benign  flows called Benign, as recorded and attacked
  evaded                of the flows called malicious as recorded, the share called Benign when
                        attacked: the attacker's success

  python kaggle/litcvit/evasion_report.py [folder with the job's preds, default below]

Runs on CPU in seconds. Writes results/litcvit_evasion.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import litcvit

ROOT = Path(__file__).resolve().parent.parent.parent
PREDS = Path("C:/data/CNS_proj/kaggle/litcvit_output_evasion/preds")
RUNS = ("mcfp_strict", "ustc")
MODELS = ("xgb", "litcvit", "litcvit-sa", "litcvit-ftaug", "litcvit-sa-ftaug")
BENIGN = {"Benign", "Normal"}
MIN_FLOWS = 20


def score(run, model, condition, subset, name, m, y, clean, pred, benign):
    called_clean, called = benign[clean[m]], benign[pred[m]]
    detected = ~called_clean
    return {"dataset": run, "model": model, "condition": condition, "subset": subset, "class": name, "flows": int(m.sum()),
            "recall_clean": round(100 * (clean[m] == y[m]).mean(), 1), "recall": round(100 * (pred[m] == y[m]).mean(), 1),
            "benign_clean": round(100 * called_clean.mean(), 1), "benign": round(100 * called.mean(), 1),
            "evaded": round(100 * called[detected].mean(), 1) if detected.any() else float("nan")}


def main(preds=PREDS, out=ROOT / "results" / "litcvit_evasion.csv"):
    rows = []
    for run in RUNS:
        tls_all = pd.read_csv(litcvit.DATA / f"{run.removesuffix('_strict')}.csv", usecols=["tls"])["tls"].to_numpy()
        for model in MODELS:
            path = preds / f"{run}_{model}.npz"
            if not path.exists():
                print(f"{path.name}: not found, skipped")
                continue
            f = np.load(path, allow_pickle=False)
            y, classes = f["y"], f["classes"].astype(str)
            benign = np.isin(classes, list(BENIGN))
            tls = tls_all[f["row"]].astype(bool)
            clean = f["clean"].argmax(1)
            for condition in (c for c in f.files if c.startswith("mimic")):
                pred = f[condition].argmax(1)
                attacked = f["attacked_other" if condition.endswith("other") else "attacked_same"]
                for subset, mask in (("all", attacked), ("tls", attacked & tls)):
                    if mask.sum():
                        rows.append(score(run, model, condition, subset, "all attack", mask, y, clean, pred, benign))
                    for c in np.unique(y[mask]):
                        m = mask & (y == c)
                        if m.sum() >= MIN_FLOWS:
                            rows.append(score(run, model, condition, subset, classes[c], m, y, clean, pred, benign))
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(df[(df["class"] == "all attack")].pivot_table(index=["dataset", "subset", "condition"], columns="model",
                                                        values="evaded").to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1])) if len(sys.argv) > 1 else main()
