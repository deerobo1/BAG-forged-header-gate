"""Calibration and abstention for LitCVit from saved test logits (plan steps 10 and the paper's Section V-D).

Temperature scaling (Guo et al., 2017) is fitted by NLL on one half of each run's test
flows and scored on the other half. On the random split the halves are a random split
per class (seed 0); on the time split the earlier half of each (class, capture) fits T
and the later half is scored, as the split itself was made. Abstention withholds the
flows with the lowest calibrated confidence (max softmax) for review.

Logits come from the Kaggle runs: version 2 (random split) and version 3 (time split) of
deetyam/litcvit-reimplementation, and the SA job (deetyam/litcvit-shortcut-aware-pretraining),
whose two models were also tested with the shortcut fields randomised or zeroed. For
those, T is fitted on the clean calibration half and applied to every condition, as it
would be when traffic changes after deployment.

  python kaggle/litcvit/calibrate.py

Runs on CPU in about a minute. Writes results/litcvit_calibration.csv (one row per job,
run, model and condition) and results/litcvit_abstention.csv (accuracy of the kept flows
against the share withheld, 0-30% in 1% steps).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax
from sklearn.metrics import f1_score

import litcvit

ROOT = Path(__file__).resolve().parent.parent.parent
KAGGLE = Path("C:/data/CNS_proj/kaggle")
JOBS = {"v2": ("random", KAGGLE / "litcvit_output_random" / "preds", ["litcvit"]),
        "v3": ("time", KAGGLE / "litcvit_output_time" / "preds", ["litcvit"]),
        "sa": ("random", KAGGLE / "litcvit_output_sa" / "preds", ["litcvit", "litcvit-sa"])}
WITHHELD = [1, 5, 10]  # % of flows withheld, for the summary table
BINS = 15


def halves(meta, rows, y, kind):
    """Positions (into rows) of the calibration and evaluation halves of one test set."""
    m = meta.set_index("row").loc[rows]
    cal = np.zeros(len(rows), bool)
    rng = np.random.default_rng(0)
    capture = m["capture"].to_numpy() if kind == "time" else 0
    for idx in pd.DataFrame({"y": y, "capture": capture}).groupby(["y", "capture"]).indices.values():
        idx = rng.permutation(idx) if kind == "random" else idx[np.argsort(m["ts_first"].to_numpy()[idx], kind="stable")]
        cal[idx[:len(idx) // 2]] = True
    return np.flatnonzero(cal), np.flatnonzero(~cal)


def nll(z, y, t):
    return -log_softmax(z / t, axis=1)[np.arange(len(y)), y].mean()


def fit_temperature(z, y):
    return float(np.exp(minimize_scalar(lambda s: nll(z, y, np.exp(s)), bounds=(-3, 3), method="bounded").x))


def ece(conf, correct):
    """Expected calibration error over BINS equal-width confidence bins."""
    bins = np.minimum((conf * BINS).astype(int), BINS - 1)
    return sum(abs(correct[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean()
               for b in range(BINS) if (bins == b).any())


def abstain(conf, correct, share):
    """Accuracy of the kept flows and the share of errors withheld, withholding the least confident share %."""
    order = np.argsort(conf, kind="stable")
    k = int(round(share / 100 * len(conf)))
    kept, held = order[k:], order[:k]
    errors = max(1, (~correct).sum())
    return 100 * correct[kept].mean(), 100 * (~correct[held]).sum() / errors


def main():
    rows, curves = [], []
    for job, (kind, preds, models) in JOBS.items():
        for run in litcvit.RUNS:
            meta, _ = litcvit.load(run)
            for model in models:
                f = np.load(preds / f"{run}_{kind}_{model}_0.npz")
                conditions = {c: f[c] for c in ("clean", "random", "zero") if c in f.files} or {"clean": f["logits"]}
                y = f["y"]
                cal, ev = halves(meta, f["row"], y, kind)
                t = fit_temperature(conditions["clean"][cal], y[cal])
                for condition, z in conditions.items():
                    z, ye = z[ev].astype(np.float64), y[ev]
                    p1, pt = np.exp(log_softmax(z, axis=1)), np.exp(log_softmax(z / t, axis=1))
                    pred, conf = pt.argmax(1), pt.max(1)
                    correct = pred == ye
                    row = {"job": job, "dataset": run, "split": kind, "model": model, "condition": condition,
                           "calibrate": len(cal), "evaluate": len(ev), "temperature": round(t, 3),
                           "accuracy": round(100 * correct.mean(), 2),
                           "macro_f1": round(100 * f1_score(ye, pred, average="macro"), 2),
                           "ece_before": round(100 * ece(p1.max(1), p1.argmax(1) == ye), 2),
                           "ece_after": round(100 * ece(conf, correct), 2),
                           "nll_before": round(nll(z, ye, 1.0), 4), "nll_after": round(nll(z, ye, t), 4),
                           "mean_confidence": round(100 * conf.mean(), 2)}
                    for share in WITHHELD:
                        acc, caught = abstain(conf, correct, share)
                        row[f"acc_withhold{share}"], row[f"errors_caught{share}"] = round(acc, 2), round(caught, 1)
                    rows.append(row)
                    for share in range(31):
                        acc, caught = abstain(conf, correct, share)
                        curves.append({"job": job, "dataset": run, "split": kind, "model": model,
                                       "condition": condition, "withheld": share,
                                       "accuracy": round(acc, 3), "errors_caught": round(caught, 2)})
                    print(f"{job} {run:11s} {model:10s} {condition:6s} T {t:5.2f}  acc {row['accuracy']:6.2f}  "
                          f"ECE {row['ece_before']:5.2f} -> {row['ece_after']:5.2f}  withhold 5%: acc "
                          f"{row['acc_withhold5']:6.2f}, errors caught {row['errors_caught5']:5.1f}%", flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results" / "litcvit_calibration.csv", index=False)
    pd.DataFrame(curves).to_csv(ROOT / "results" / "litcvit_abstention.csv", index=False)
    print("wrote results/litcvit_calibration.csv and results/litcvit_abstention.csv")


if __name__ == "__main__":
    main()
