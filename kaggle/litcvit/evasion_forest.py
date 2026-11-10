"""The evasion test on random forests, on the laptop: a quick first answer and a check on the attack.

Forests are fitted on the training flows of a run, as recorded and with the shortcut fields
zeroed (litcvit.shortcut_fields), and tested on the attacked flows of kaggle/litcvit_evasion.
The attack changes only shortcut fields, so the second forest must give the same predictions on
clean and attacked flows; the script stops if it doesn't. Also scores the single-byte rule
"first packet's TTL is 32 means Benign" on USTC-TFC2016.

  python kaggle/litcvit/evasion_forest.py

Runs in about 5 minutes. Writes results/forest_evasion.csv and results/ustc_ttl_rule.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "kaggle" / "litcvit_evasion"))
import evasion as E  # noqa: E402

L, M = E.L, E.M


def flat(z):
    return np.asarray(z).reshape(len(z), -1)


def zero_fields(z):
    return flat(L.perturb(torch.from_numpy(np.asarray(z).reshape(len(z), 40, 40)), "zero", None))


def scores(rows, run, name, pred_clean, pred, classes, y, attacked, condition):
    benign = np.isin(classes, list(E.BENIGN))
    for c in np.unique(y[attacked]):
        m = attacked & (y == c)
        rows.append({"dataset": run, "model": name, "condition": condition, "class": classes[c], "flows": int(m.sum()),
                     "recall_clean": round(100 * (pred_clean[m] == c).mean(), 1),
                     "recall": round(100 * (pred[m] == c).mean(), 1),
                     "benign_clean": round(100 * benign[pred_clean[m]].mean(), 1),
                     "benign": round(100 * benign[pred[m]].mean(), 1)})


def ttl_rule(run="ustc"):
    meta, X = L.load(run)
    first = X.reshape(len(X), L.PACKETS, L.ROW)[:, 0]
    ipv4 = first[:, 0] >> 4 == 4
    rule = ipv4 & (first[:, 8] == 32)
    rows = [{"class": c, "flows": int((meta["class"] == c).sum()), "ttl32%": round(100 * rule[(meta["class"] == c).to_numpy()].mean(), 2)}
            for c in sorted(meta["class"].unique())]
    pd.DataFrame(rows).to_csv(ROOT / "results" / "ustc_ttl_rule.csv", index=False)
    benign = meta["class"].isin(E.BENIGN).to_numpy()
    print(pd.DataFrame(rows).to_string())
    print(f"rule 'TTL 32 means Benign': accuracy {100 * (rule == benign).mean():.2f}%, "
          f"benign caught {100 * rule[benign].mean():.2f}%, malware passed as benign {100 * rule[~benign].mean():.2f}%")


def main():
    ttl_rule()
    rows = []
    for run in E.RUNS:
        meta, X = L.load(run)
        classes = np.array(sorted(meta["class"].unique()))
        y_all = meta["class"].map({c: i for i, c in enumerate(classes)}).to_numpy()
        train, test = L.split(meta, E.SPLIT, np.random.default_rng(E.SEED))
        attacks, same, far = E.attacked_images(run)
        y = y_all[test]
        for name, prepare in (("rf", flat), ("rf-no-fields", zero_fields)):
            model = RandomForestClassifier(100, class_weight="balanced", n_jobs=-1, random_state=E.SEED)
            model.fit(prepare(X[train]), y_all[train])
            clean = model.predict(prepare(X[test]))
            for condition, images in attacks.items():
                pred = model.predict(prepare(images))
                if name == "rf-no-fields" and not np.array_equal(pred, clean):
                    raise SystemExit(f"{run} {condition}: the attack changed something other than the shortcut fields")
                scores(rows, run, name, clean, pred, classes, y, far if condition.endswith("other") else same, condition)
            print(f"{run} {name}: clean accuracy {100 * (clean == y).mean():.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "forest_evasion.csv", index=False)
    print(df[(df["model"] == "rf")].to_string())
    print("wrote results/forest_evasion.csv and results/ustc_ttl_rule.csv")


if __name__ == "__main__":
    main()
