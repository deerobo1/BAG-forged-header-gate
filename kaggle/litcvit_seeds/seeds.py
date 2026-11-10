"""New seeds for what the paper will claim: the gate's bar on new splits, and how far the scores move.

Retraining with the same seed (kaggle/litcvit_ablate) measures little, since most rows come back byte-identical. A new
seed changes the random split, the attack's donors and the weights. MobileNetV2, random split, seeds 1 and 2, in order
of what matters, so that a stopped job loses the least:
  ce, mask       the gate's two models: the bar in plan.txt for new seeds (kaggle/litcvit/gate_report.py scores it)
  both, oneside  the two losses the gate is compared with
  cfnce          does the contrastive term's gain on MCFP and CICIoT survive a new seed?

Runs on Kaggle, or on Colab through kaggle/colab/run_on_colab.ipynb, which lays out the same folders. Rows already in
seeds.csv aren't repeated, so a stopped session can simply be run again.

Needs deetyam/litcvit-flows and deetyam/litcvit-encoder, which holds litcvit.py, mimic.py and light.py (imported from
there). Writes /kaggle/working/seeds.csv and preds/<run>_mobilenetv2_<loss>_seed<seed>.npz with their .pt weights. About
5.6 h on Kaggle's GPU: 20, 21, 41, 43 and 42 minutes per seed.

Push with:  kaggle kernels push -p kaggle/litcvit_seeds
"""

import sys
from pathlib import Path

import pandas as pd

FOUND = next(Path("/kaggle/input").rglob("light.py"), None)
sys.path.insert(0, str(FOUND.parent if FOUND else Path(__file__).resolve().parent.parent / "litcvit"))
import light as T  # noqa: E402

L = T.L
SEEDS = (1, 2)
MODEL = "mobilenetv2"
STAGES = [("ce", "mask"), ("both", "oneside"), ("cfnce",)]


def main(seeds=SEEDS, stages=STAGES, runs=None, epochs=T.EPOCHS):
    (L.OUT / "preds").mkdir(parents=True, exist_ok=True)
    print(f"data in {L.DATA}, device {L.DEVICE}", flush=True)
    table = L.OUT / "seeds.csv"
    results = pd.read_csv(table).to_dict("records") if table.exists() else []
    done = {(r["dataset"], r["seed"], r["loss"]) for r in results}
    prepared = {}
    for losses in stages:
        for seed in seeds:
            for run in runs or L.RUNS:
                for loss in [l for l in losses if (run, seed, l) not in done]:
                    if (run, seed) not in prepared:
                        prepared[run, seed] = T.test_sets(run, "random", seed)
                    rows = T.run_one(prepared[run, seed], MODEL, loss, seed, L.OUT / "preds" / f"{run}_{MODEL}_{loss}_seed{seed}.npz",
                                     epochs=epochs, log=lambda m: print(m, flush=True))
                    results += [{"dataset": run, "split": "random", **r} for r in rows]
                    partial = table.with_suffix(".tmp")
                    pd.DataFrame(results).to_csv(partial, index=False)
                    partial.replace(table)
                    by = {r["condition"]: r for r in rows}
                    print(f"{run:12s} seed {seed} {loss:7s} {rows[0]['seconds']:5d}s  macro-F1 clean {by['clean']['macro_f1']:6.2f}  "
                          f"random {by['random']['macro_f1']:6.2f}  mimic {by['mimic_stack']['macro_f1']:6.2f}  "
                          f"evaded {by['mimic_stack'].get('evaded')}", flush=True)


if __name__ == "__main__":
    main()
