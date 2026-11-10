"""LitCVit on TLS flows, from saved test logits: recall per class, model and test condition.

TLS attack flows are the only flows in the six datasets where a model could be detecting
malware inside encryption, and almost all of them are in MCFP and USTC-TFC2016 (TII-SSRC-23
has 2 in its test set; the others have none). This asks whether recall on them survives once
the header fields that identify a host or a clock change (litcvit.perturb).

Logits come from the Kaggle jobs: version 2 (random split) and version 3 (time split) of
deetyam/litcvit-reimplementation, the SA job, and, once downloaded, the fine-tuning and
payload jobs (kaggle/litcvit_ftaug, kaggle/litcvit_payload). Jobs not downloaded are skipped.

  python kaggle/litcvit/tls_flows.py

Runs on CPU in seconds. Writes results/litcvit_tls.csv: one row per job, run, model, condition
and class with at least MIN_FLOWS TLS test flows.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import litcvit

ROOT = Path(__file__).resolve().parent.parent.parent
KAGGLE = Path("C:/data/CNS_proj/kaggle")
JOBS = {"v2": ("random", KAGGLE / "litcvit_output_random" / "preds", ["litcvit"]),
        "v3": ("time", KAGGLE / "litcvit_output_time" / "preds", ["litcvit"]),
        "sa": ("random", KAGGLE / "litcvit_output_sa" / "preds", ["litcvit", "litcvit-sa"]),
        "ftaug": ("random", KAGGLE / "litcvit_output_ftaug" / "preds", ["litcvit-ftaug", "litcvit-sa-ftaug"]),
        "payload": ("random", KAGGLE / "litcvit_output_payload" / "preds", ["litcvit", "litcvit-sa"])}
BENIGN = {"Benign", "Normal"}
MIN_FLOWS = 20


def main():
    rows = []
    for job, (kind, preds, models) in JOBS.items():
        if not preds.exists():
            print(f"{job}: not downloaded, skipped")
            continue
        for run in litcvit.RUNS:
            tls = pd.read_csv(litcvit.DATA / f"{run.removesuffix('_strict')}.csv", usecols=["tls"])["tls"].to_numpy()
            for model in models:
                f = np.load(preds / f"{run}_{kind}_{model}_0.npz", allow_pickle=False)
                y, classes = f["y"], f["classes"].astype(str)
                is_tls = tls[f["row"]].astype(bool)
                benign = np.isin(classes, list(BENIGN))
                conditions = {c: f[c] for c in f.files if c not in ("row", "y", "classes", "logits")} or {"clean": f["logits"]}
                for condition, z in conditions.items():
                    pred = z.argmax(1)
                    for c in np.unique(y[is_tls]):
                        m = is_tls & (y == c)
                        if m.sum() < MIN_FLOWS:
                            continue
                        values, counts = np.unique(pred[m], return_counts=True)
                        rows.append({"job": job, "dataset": run, "split": kind, "model": model, "condition": condition,
                                     "class": classes[c], "flows": int(m.sum()),
                                     "recall": round(100 * (pred[m] == c).mean(), 1),
                                     "predicted_benign": round(100 * benign[pred[m]].mean(), 1),
                                     "top_prediction": classes[values[counts.argmax()]]})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "litcvit_tls.csv", index=False)
    print(df[df["job"] == "sa"].pivot_table(index=["dataset", "class", "flows"], columns=["model", "condition"],
                                            values="recall").to_string())
    print("wrote results/litcvit_tls.csv")


if __name__ == "__main__":
    main()
