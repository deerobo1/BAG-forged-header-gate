"""Can a loss stop a traffic classifier reading the header? Five losses on two small models, all seven runs.

The losses are in kaggle/litcvit/light.py: ce, mask, rand, swap and cfnce, the last being our
candidate. A flow takes the real header fields of a flow of another class but keeps its content
and label, and counterfactual_nce holds it nearer the flow it took its content from than the
flow it took its header from. Every model is tested on flows as recorded, with the shortcut fields
randomised or zeroed, with the payload zeroed, and under the header mimicry attack (mimic.py).
Random split, seed 0, as the jobs it is compared with.

Needs deetyam/litcvit-flows and deetyam/litcvit-encoder, which holds litcvit.py, mimic.py and
light.py (imported from there):
  kaggle datasets version -p litcvit-encoder -m "..."   (from C:\\data\\CNS_proj\\kaggle)
Writes /kaggle/working/loss.csv (one row per run, model, loss and condition) and
preds/<run>_<model>_<loss>.npz. About 2.5 h on Kaggle's GPU.

Push with:  kaggle kernels push -p kaggle/litcvit_loss
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
MODELS = ["shufflenet", "mobilenetv2"]
LOSSES = ["ce", "mask", "rand", "swap", "cfnce"]


def main(models=MODELS, losses=LOSSES, runs=None, name="loss", epochs=T.EPOCHS):
    (L.OUT / "preds").mkdir(exist_ok=True)
    print(f"data in {L.DATA}, device {L.DEVICE}", flush=True)
    results = []
    for run in runs or L.RUNS:
        start = time.time()
        data = T.test_sets(run, SPLIT, SEED)
        print(f"{run}: {len(data['y_train']):,} training flows, {len(data['y_test']):,} test flows, "
              f"{ {k: int(v.sum()) for k, v in data['attacked'].items()} } attacked, {time.time() - start:.0f}s", flush=True)
        for model in models:
            for loss in losses:
                rows = T.run_one(data, model, loss, SEED, L.OUT / "preds" / f"{run}_{model}_{loss}.npz", epochs=epochs,
                                 log=lambda m: print(m, flush=True))
                results += [{"dataset": run, "split": SPLIT, **r} for r in rows]
                pd.DataFrame(results).to_csv(L.OUT / f"{name}.csv", index=False)
                by = {r["condition"]: r for r in rows}
                print(f"{run:12s} {model:12s} {loss:6s} {rows[0]['seconds']:5d}s  macro-F1 clean {by['clean']['macro_f1']:6.2f}  "
                      f"random {by['random']['macro_f1']:6.2f}  no payload {by['nopayload']['macro_f1']:6.2f}  "
                      f"mimic {by['mimic_stack']['macro_f1']:6.2f}  evaded {by['mimic_stack'].get('evaded')}", flush=True)


if __name__ == "__main__":
    main()
