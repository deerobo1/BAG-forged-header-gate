"""Do other lightweight vision models score like LitCVit? Eight small models and a Vision Mamba.

Each model (kaggle/litcvit/light.py) trains with plain cross-entropy on every run, random split,
seed 0, and is tested on flows as recorded, with the shortcut fields randomised or zeroed, with
the payload zeroed, and under the header mimicry attack. Two more kinds of model also run on the
time split, where the attack is tested as well; MobileNetV2's rows there come from
kaggle/litcvit_gate. No pretraining and 30 epochs, against LitCVit's 100 and 500, so a model that
matches it here needs far less. Vision Mamba alone gets at least VIM_STEPS steps.

Runs on Kaggle, or on Colab through kaggle/colab/run_on_colab.ipynb, which lays out the same
folders. Rows already in bench.csv aren't repeated, so a stopped session can simply be run again.

Needs deetyam/litcvit-flows and deetyam/litcvit-encoder, which holds litcvit.py, mimic.py and
light.py (imported from there):
  kaggle datasets version -p litcvit-encoder -m "..."   (from C:\\data\\CNS_proj\\kaggle)
Writes /kaggle/working/bench.csv (one row per run, split, model and condition) and
preds/<run>_<split>_<model>.npz. About 4 h on Kaggle's GPU.

Push with:  kaggle kernels push -p kaggle/litcvit_bench
"""

import sys
import time
from pathlib import Path

import pandas as pd

FOUND = next(Path("/kaggle/input").rglob("light.py"), None)
sys.path.insert(0, str(FOUND.parent if FOUND else Path(__file__).resolve().parent.parent / "litcvit"))
import light as T  # noqa: E402

L = T.L
SEED = 0
PLAN = {"random": T.MODELS, "time": ["mobilevit", "vim"]}
VIM_STEPS = 2000  # Vision Mamba starts slowly, and 30 epochs give the small runs under 900 steps


def main(plan=PLAN, runs=None):
    (L.OUT / "preds").mkdir(parents=True, exist_ok=True)
    print(f"data in {L.DATA}, device {L.DEVICE}", flush=True)
    table = L.OUT / "bench.csv"
    results = pd.read_csv(table).to_dict("records") if table.exists() else []
    done = {(r["dataset"], r["split"], r["model"]) for r in results}
    for split, models in plan.items():
        for run in runs or L.RUNS:
            todo = [m for m in models if (run, split, m) not in done]
            if not todo:
                continue
            start = time.time()
            data = T.test_sets(run, split, SEED)
            print(f"{run} {split}: {len(data['y_train']):,} training flows, {time.time() - start:.0f}s to prepare", flush=True)
            for model in todo:
                name = model
                rows = T.run_one(data, model, "ce", SEED, L.OUT / "preds" / f"{run}_{split}_{name}.npz",
                                 log=lambda m: print(m, flush=True), min_steps=VIM_STEPS if model == "vim" else 0)
                results += [{"dataset": run, "split": split, **r} for r in rows]
                partial = table.with_suffix(".tmp")
                pd.DataFrame(results).to_csv(partial, index=False)
                partial.replace(table)
                by = {r["condition"]: r for r in rows}
                print(f"{run:12s} {split:6s} {name:17s} {rows[0]['seconds']:5d}s  {rows[0]['params'] / 1e6:5.2f}M params  "
                      f"acc {by['clean']['accuracy']:6.2f}  macro-F1 {by['clean']['macro_f1']:6.2f}  "
                      f"random {by['random']['macro_f1']:6.2f}  no payload {by['nopayload']['macro_f1']:6.2f}", flush=True)


if __name__ == "__main__":
    main()
