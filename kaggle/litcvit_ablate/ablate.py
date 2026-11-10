"""What in cfnce does the work? An ablation, and a stronger attack, on all seven runs.

kaggle/litcvit_loss found cfnce keeps nearly all of plain training's clean score and most of
masking's resistance. It can't say why, since swap trains only on counterfactual flows:
  both    cross-entropy on flows as recorded and on counterfactual flows, no contrastive term.
          If this matches cfnce, the result comes from the augmentation, not from a new loss
  cfnce1  cfnce with the donor counted as an ordinary negative: does weighting it matter?
  cfnce, ce, mask  retrained so they can also meet the two new tests (light.test_sets):
  mimic_all  the attack plus new initial sequence numbers, which any stack may choose
  isn        every flow with new initial sequence numbers and nothing else changed, which shows
             whether a model reads their absolute values at all
Random split, seed 0, as kaggle/litcvit_loss.

Needs deetyam/litcvit-flows and deetyam/litcvit-encoder, which holds litcvit.py, mimic.py and
light.py (imported from there):
  kaggle datasets version -p litcvit-encoder -m "..."   (from C:\\data\\CNS_proj\\kaggle)
Writes /kaggle/working/ablate.csv (columns of loss.csv), preds/<run>_<model>_<loss>.npz and their
.pt weights. About 3.5 h on Kaggle's GPU.

Push with:  kaggle kernels push -p kaggle/litcvit_ablate
"""

import sys
import time
from pathlib import Path

import pandas as pd

FOUND = next(Path("/kaggle/input").rglob("light.py"), None)
sys.path.insert(0, str(FOUND.parent if FOUND else Path(__file__).resolve().parent.parent / "litcvit"))
import light as T  # noqa: E402

L = T.L
SPLIT, SEED = "random", 0
PLAN = {"mobilenetv2": ["ce", "mask", "both", "cfnce", "cfnce1"], "shufflenet": ["both", "cfnce", "cfnce1"]}


def main(plan=PLAN, runs=None, epochs=T.EPOCHS):
    (L.OUT / "preds").mkdir(exist_ok=True)
    print(f"data in {L.DATA}, device {L.DEVICE}", flush=True)
    results = []
    for run in runs or L.RUNS:
        start = time.time()
        data = T.test_sets(run, SPLIT, SEED)
        print(f"{run}: {len(data['y_train']):,} training flows, {time.time() - start:.0f}s to prepare", flush=True)
        for model, losses in plan.items():
            for loss in losses:
                rows = T.run_one(data, model, loss, SEED, L.OUT / "preds" / f"{run}_{model}_{loss}.npz", epochs=epochs,
                                 log=lambda m: print(m, flush=True))
                results += [{"dataset": run, "split": SPLIT, **r} for r in rows]
                pd.DataFrame(results).to_csv(L.OUT / "ablate.csv", index=False)
                by = {r["condition"]: r for r in rows}
                print(f"{run:12s} {model:12s} {loss:7s} {rows[0]['seconds']:5d}s  macro-F1 clean {by['clean']['macro_f1']:6.2f}  "
                      f"isn {by['isn']['macro_f1']:6.2f}  mimic {by['mimic_stack']['macro_f1']:6.2f}  all {by['mimic_all']['macro_f1']:6.2f}  "
                      f"evaded {by['mimic_stack'].get('evaded')} / {by['mimic_all'].get('evaded')}", flush=True)


if __name__ == "__main__":
    main()
