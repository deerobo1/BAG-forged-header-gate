"""The one-sided loss on every run: two small models, then LitCVit itself.

A header may convict but never acquit (kaggle/litcvit/light.py): malware flows also train with
real benign header fields, keeping their own label, while benign flows are never changed. An
attacker can forge a benign header and has no reason to forge a malicious one.
  shufflenet, mobilenetv2  as in kaggle/litcvit_loss/loss.py, whose rows these extend
  litcvit                  version 2's pretrained encoder, fine-tuned as in litcvit.py with
                           litcvit.finetune(..., forged_copies=COPIES), so its rows compare with
                           results/litcvit_payload.csv and the evasion job's
All are tested as in loss.py, on the same flows and attacks. Random split, seed 0.

Needs deetyam/litcvit-flows and deetyam/litcvit-encoder, which holds litcvit.py, mimic.py, light.py
and the encoder:
  kaggle datasets version -p litcvit-encoder -m "..."   (from C:\\data\\CNS_proj\\kaggle)
Writes /kaggle/working/oneside.csv (columns of loss.csv), preds/<run>_<model>_oneside.npz and
models/<run>_litcvit_oneside.pt. About 3.5 h on Kaggle's GPU: 40 min for the small models, and
2.8 h for LitCVit (fine-tuning takes 2.25 h, plus a quarter for the forged flows and their copies).

Push with:  kaggle kernels push -p kaggle/litcvit_oneside
"""

import sys
import time
from pathlib import Path

import pandas as pd
import torch

FOUND = next(Path("/kaggle/input").rglob("light.py"), None)
if FOUND is None:  # local test: model code from the repo, encoder from the downloaded output
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "litcvit"))
    ENCODER = Path("C:/data/CNS_proj/kaggle/litcvit_output_random/encoder_random_0.pt")
else:
    sys.path.insert(0, str(FOUND.parent))
    ENCODER = FOUND.parent / "encoder_random_0.pt"
import light as T  # noqa: E402

L = T.L
SPLIT, SEED = "random", 0
MODELS = ["shufflenet", "mobilenetv2"]
COPIES = 8  # USTC-TFC2016's copies take ~1.2 GB of GPU memory


def show(run, model, rows):
    by = {r["condition"]: r for r in rows}
    print(f"{run:12s} {model:12s} oneside {rows[0]['seconds']:5d}s  macro-F1 clean {by['clean']['macro_f1']:6.2f}  "
          f"random {by['random']['macro_f1']:6.2f}  no payload {by['nopayload']['macro_f1']:6.2f}  "
          f"mimic {by['mimic_stack']['macro_f1']:6.2f}  evaded {by['mimic_stack'].get('evaded')}", flush=True)


def main(models=MODELS, runs=None, epochs=T.EPOCHS, copies=COPIES):
    (L.OUT / "preds").mkdir(exist_ok=True)
    (L.OUT / "models").mkdir(exist_ok=True)
    print(f"data in {L.DATA}, device {L.DEVICE}", flush=True)
    results, prepared = [], {}

    def save(run, rows):
        results.extend({"dataset": run, "split": SPLIT, **r} for r in rows)
        pd.DataFrame(results).to_csv(L.OUT / "oneside.csv", index=False)

    for run in runs or L.RUNS:
        data = prepared[run] = T.test_sets(run, SPLIT, SEED)
        for model in models:
            rows = T.run_one(data, model, "oneside", SEED, L.OUT / "preds" / f"{run}_{model}_oneside.npz", epochs=epochs,
                             log=lambda m: print(m, flush=True))
            save(run, rows)
            show(run, model, rows)

    encoder = L.Encoder().to(L.DEVICE)
    encoder.load_state_dict(torch.load(ENCODER, map_location=L.DEVICE, weights_only=True))
    params = sum(p.numel() for p in encoder.parameters())
    for run, data in prepared.items():
        start = time.time()
        attacks = {name: images for name, images in data["conditions"].items() if name != "clean"}
        _, _, logits, _, _ = L.finetune(encoder, run, SPLIT, SEED, attacks=attacks, forged_copies=copies,
                                        save=L.OUT / "models" / f"{run}_litcvit_oneside.pt")
        rows = T.report(data, "litcvit", "oneside", SEED, params, 0, round(time.time() - start), logits,
                        L.OUT / "preds" / f"{run}_litcvit_oneside.npz")
        save(run, rows)
        show(run, "litcvit", rows)


if __name__ == "__main__":
    main()
