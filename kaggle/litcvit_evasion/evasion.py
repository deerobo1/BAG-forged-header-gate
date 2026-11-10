"""Evasion test: can malware pass as benign by changing only what its own TCP/IP stack sends?

kaggle/litcvit/mimic.py gives attack flows the header fields of a real benign flow; payload,
lengths, TCP flags and sequence numbers don't change, and checksums stay valid. Conditions:
  clean             test flows as recorded
  mimic_ttl         only the TTL, copied from a benign flow of the same network
  mimic_window      only the TCP window
  mimic_ttl_window  both
  mimic_stack       TTL, window, IP type of service, identification, don't-fragment bit, TCP options
  mimic_stack_other the same from benign flows of the other dataset: an attacker who knows
                    nothing of the network the model was trained on
Same-network donors are benign test flows, which the model never trained on.

Models, random split, seed 0, on the two runs with TLS malware flows:
  litcvit, litcvit-sa                as in kaggle/litcvit_sa/sa.py (fine-tuned on clean flows)
  litcvit-ftaug, litcvit-sa-ftaug    as in kaggle/litcvit_ftaug/ftaug.py (fine-tuned on randomised fields)
  xgb                                as in kaggle/table7/table7.py
so we can see whether the fields matter to LitCVit alone, and whether the defences hold.

Needs deetyam/litcvit-flows and deetyam/litcvit-encoder, which holds litcvit.py and mimic.py
(imported from there) and both encoders:
  kaggle datasets version -p litcvit-encoder -m "..."   (from C:\\data\\CNS_proj\\kaggle)
Writes /kaggle/working/evasion.csv (one row per run, model and condition, on all test flows),
preds/<run>_<model>.npz (row, y, classes, attacked_same, attacked_other, and logits per condition)
and models/<run>_<model>.pt (fine-tuned encoder and classifier). kaggle/litcvit/evasion_report.py
turns the predictions into scores. About 4.5 h on Kaggle's GPU (1 h per LitCVit model).

Push with:  kaggle kernels push -p kaggle/litcvit_evasion
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score

FOUND = next(Path("/kaggle/input").rglob("mimic.py"), None)
if FOUND is None:  # local test: model code from the repo, encoders from the downloaded outputs
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "litcvit"))
    KAGGLE = Path("C:/data/CNS_proj/kaggle")
    PLAIN, SA = KAGGLE / "litcvit_output_random" / "encoder_random_0.pt", KAGGLE / "litcvit_output_sa" / "encoder_sa_random_0.pt"
else:
    sys.path.insert(0, str(FOUND.parent))
    PLAIN, SA = FOUND.parent / "encoder_random_0.pt", FOUND.parent / "encoder_sa_random_0.pt"
import litcvit as L  # noqa: E402
import mimic as M  # noqa: E402

SPLIT, SEED = "random", 0
RUNS = {"mcfp_strict": "ustc", "ustc": "mcfp"}  # run -> the dataset lending its benign flows to mimic_stack_other
MODELS = {"litcvit": (PLAIN, 0), "litcvit-sa": (SA, 0), "litcvit-ftaug": (PLAIN, 16), "litcvit-sa-ftaug": (SA, 16)}
FIELDS = {"mimic_ttl": ("ttl",), "mimic_window": ("window",), "mimic_ttl_window": ("ttl", "window"), "mimic_stack": M.STACK}
BENIGN = {"Benign", "Normal"}


def attacked_images(run):
    """Test flows under each condition, with the masks of attack flows that had a donor."""
    meta, X = L.load(run)
    _, test = L.split(meta, SPLIT, np.random.default_rng(SEED))
    images = X[test]
    benign = meta["class"].isin(BENIGN).to_numpy()[test]
    rng = np.random.default_rng(SEED)

    def donated(donors):
        chosen = M.choose_donors(images, donors, rng)
        chosen[benign] = -1
        return chosen

    same = donated(images[benign])
    attacks = {name: M.apply_mimicry(images, images[benign], same, fields) for name, fields in FIELDS.items()}
    other_meta, other_X = L.load(RUNS[run])
    other = other_X[other_meta["class"].isin(BENIGN).to_numpy()]
    far = donated(other)
    attacks["mimic_stack_other"] = M.apply_mimicry(images, other, far, M.STACK)
    print(f"{run}: {len(test):,} test flows, {int((~benign).sum()):,} attack, {int((same >= 0).sum()):,} with a "
          f"same-network donor, {int((far >= 0).sum()):,} with another network's", flush=True)
    return attacks, same >= 0, far >= 0


def xgb_logits(run, attacks):
    import xgboost as xgb
    meta, X = L.load(run)
    classes = sorted(meta["class"].unique())
    y = meta["class"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    train, test = L.split(meta, SPLIT, np.random.default_rng(SEED))
    model = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, tree_method="hist",
                              device=L.DEVICE, random_state=SEED)
    model.fit(X[train].reshape(len(train), -1), y[train], sample_weight=L.class_weights(y[train], len(classes))[y[train]])
    proba = {"clean": model.predict_proba(X[test].reshape(len(test), -1))}
    proba.update({name: model.predict_proba(z.reshape(len(z), -1)) for name, z in attacks.items()})
    return (meta["row"].to_numpy()[test], y[test], {k: np.log(np.maximum(v, 1e-9)) for k, v in proba.items()},
            classes, len(train))


def record(results, run, model, out, seconds, same, far):
    rows, y, logits, classes, n_train = out
    for condition, z in logits.items():
        pred = z.argmax(1)
        recall = recall_score(y, pred, labels=range(len(classes)), average=None, zero_division=0)
        results.append({"dataset": run, "split": SPLIT, "model": model, "condition": condition, "seed": SEED,
                        "train": n_train, "test": len(y),
                        "accuracy": round(100 * accuracy_score(y, pred), 2),
                        "macro_f1": round(100 * f1_score(y, pred, average="macro"), 2), "seconds": seconds,
                        "recall": json.dumps({c: round(100 * r, 1) for c, r in zip(classes, recall)})})
        print(f"{run:12s} {model:16s} {condition:18s}: acc {results[-1]['accuracy']:6.2f}  "
              f"macro-F1 {results[-1]['macro_f1']:6.2f}", flush=True)
    pd.DataFrame(results).to_csv(L.OUT / "evasion.csv", index=False)
    np.savez_compressed(L.OUT / "preds" / f"{run}_{model}.npz", row=rows, y=y, classes=np.array(classes),
                        attacked_same=same, attacked_other=far, **{c: z.astype(np.float32) for c, z in logits.items()})


def main():
    (L.OUT / "preds").mkdir(exist_ok=True)
    (L.OUT / "models").mkdir(exist_ok=True)
    print(f"data in {L.DATA}, device {L.DEVICE}", flush=True)
    results = []
    for run in RUNS:
        attacks, same, far = attacked_images(run)
        start = time.time()
        record(results, run, "xgb", xgb_logits(run, attacks), round(time.time() - start), same, far)
        for model, (path, copies) in MODELS.items():
            encoder = L.Encoder().to(L.DEVICE)
            encoder.load_state_dict(torch.load(path, map_location=L.DEVICE, weights_only=True))
            start = time.time()
            out = L.finetune(encoder, run, SPLIT, SEED, train_copies=copies, attacks=attacks,
                             save=L.OUT / "models" / f"{run}_{model}.pt")
            record(results, run, model, out, round(time.time() - start), same, far)


if __name__ == "__main__":
    main()
