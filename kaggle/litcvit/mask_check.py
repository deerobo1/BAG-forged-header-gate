"""Does litcvit.shortcut_fields remove the same signal as the audit's fixed-offset _min input?

A random forest (as in baselines/header_payload.py), random split seed 0 as in litcvit.py,
on the full image, on it with shortcut_fields zeroed, and on it with every header byte
zeroed except IP version/IHL, total length, protocol and TCP flags (header_payload.KEPT).
Also reports how many header bytes per packet the mask covers.

  python kaggle/litcvit/mask_check.py [run ...]        (default: iot23 5gad2022 ustc)

Runs on CPU in a few minutes. Writes results/litcvit_mask_check.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

import litcvit

ROOT = Path(__file__).resolve().parent.parent.parent
KEPT = [0, 2, 3, 9, 33]  # as in baselines/header_payload.py


def main():
    rows = []
    for run in sys.argv[1:] or ["iot23", "5gad2022", "ustc"]:
        meta, X = litcvit.load(run)
        classes = sorted(meta["class"].unique())
        y = meta["class"].map({c: i for i, c in enumerate(classes)}).to_numpy()
        train, test = litcvit.split(meta, "random", np.random.default_rng(0))
        fields = torch.cat([litcvit.shortcut_fields(torch.from_numpy(X[i:i + 10000]))
                            for i in range(0, len(X), 10000)]).numpy()
        images = X.reshape(len(X), litcvit.PACKETS, litcvit.ROW)
        present = images[..., :litcvit.HEADER].any(-1)
        ours, fixed = images.copy(), images.copy()
        ours[fields] = 0
        fixed[:, :, np.setdiff1d(np.arange(litcvit.HEADER), KEPT)] = 0
        for name, Z in (("full", images), ("shortcut fields zeroed", ours), ("fixed-offset _min", fixed)):
            Z = Z.reshape(len(Z), -1)
            rf = RandomForestClassifier(100, class_weight="balanced", n_jobs=-1, random_state=0)
            pred = rf.fit(Z[train], y[train]).predict(Z[test])
            rows.append({"dataset": run, "input": name,
                         "masked_bytes_per_packet": round(float(fields.sum(-1)[present].mean()), 1),
                         "macro_f1": round(100 * f1_score(y[test], pred, average="macro"), 2)})
            print(f"{run:10s} {name:24s} macro-F1 {rows[-1]['macro_f1']:6.2f}", flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results" / "litcvit_mask_check.csv", index=False)


if __name__ == "__main__":
    main()
