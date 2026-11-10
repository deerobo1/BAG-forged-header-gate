"""The gate's first real test, and the models the remaining attacks need. Seed 0, one job.

In order of what matters, so that stopping at the weekly quota loses the least:
  time     MobileNetV2 with ce and mask on every run, time split: the test of the gate whose bar plan.txt
           set before any such result (kaggle/litcvit/gate_report.py scores it). The same rows as
           kaggle/litcvit_bench's, which waits for the Nov 7 quota reset
  gated    light.GatedNet with losses gated and views, MobileNetV2, the three smallest runs, random split:
           can one network replace the gate's two?
  oneside  ShuffleNet, then MobileNetV2, on every run, random split. kaggle/litcvit_oneside saved no weights
           for them, and kaggle/litcvit/limits.py needs weights to attack them

Needs deetyam/litcvit-flows and deetyam/litcvit-encoder, which holds litcvit.py, mimic.py and light.py
(imported from there):
  kaggle datasets version -p litcvit-encoder -m "..."   (from C:\\data\\CNS_proj\\kaggle)
Writes /kaggle/working/gate_test.csv and preds/<run>_time_<model>[_mask].npz or preds/<run>_<model>_<loss>.npz,
with their .pt weights. About 2.2 h on Kaggle's GPU.

Push with:  kaggle kernels push -p kaggle/litcvit_gate
"""

import sys
from pathlib import Path

import pandas as pd

FOUND = next(Path("/kaggle/input").rglob("light.py"), None)
sys.path.insert(0, str(FOUND.parent if FOUND else Path(__file__).resolve().parent.parent / "litcvit"))
import light as T  # noqa: E402

L = T.L
SEED = 0
SMALL = ["5gad2022", "mcfp", "mcfp_strict"]  # the smallest runs, which the bar for the single network tries
PLAN = [("time", None, "mobilenetv2", ("ce", "mask")),
        ("random", SMALL, "mobilenetv2", ("gated", "views")),
        ("random", None, "shufflenet", ("oneside",)),
        ("random", None, "mobilenetv2", ("oneside",))]


def path_of(run, split, model, loss):
    if split == "time":
        return L.OUT / "preds" / f"{run}_time_{model}{'' if loss == 'ce' else '_' + loss}.npz"
    return L.OUT / "preds" / f"{run}_{model}_{loss}.npz"


def main(plan=PLAN, runs=None, epochs=T.EPOCHS):
    (L.OUT / "preds").mkdir(exist_ok=True)
    print(f"data in {L.DATA}, device {L.DEVICE}", flush=True)
    results, prepared = [], {}
    for split, only, model, losses in plan:
        for run in [r for r in runs or L.RUNS if only is None or r in only]:
            if (run, split) not in prepared:
                prepared[run, split] = T.test_sets(run, split, SEED)
            data = prepared[run, split]
            for loss in losses:
                rows = T.run_one(data, model, loss, SEED, path_of(run, split, model, loss), epochs=epochs,
                                 log=lambda m: print(m, flush=True))
                results += [{"dataset": run, "split": split, **r} for r in rows]
                pd.DataFrame(results).to_csv(L.OUT / "gate_test.csv", index=False)
                by = {r["condition"]: r for r in rows}
                print(f"{run:12s} {split:6s} {model:12s} {loss:7s} {rows[0]['seconds']:5d}s  macro-F1 clean {by['clean']['macro_f1']:6.2f}  "
                      f"random {by['random']['macro_f1']:6.2f}  mimic {by['mimic_stack']['macro_f1']:6.2f}  "
                      f"evaded {by['mimic_stack'].get('evaded')}", flush=True)


if __name__ == "__main__":
    main()
