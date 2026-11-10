"""Where shortcut-aware pretraining's invariance stops: frozen packet tokens or flow embedding.

Shortcut-aware pairs (litcvit.augment_sa) differ in the header fields that identify a host
or a clock, and NT-Xent asks only the pooled flow embedding to match across them. Fine-tuning
trains the packet and flow blocks on the frozen packet tokens, so if those tokens still carry
the fields, the classifier can learn them again. For our LitCVit and the shortcut-aware
encoder (random split, seed 0), on each run's test flows:

  shift     mean distance between a flow's features and those of the same flow with its
            shortcut fields randomised (litcvit.perturb), over the mean distance between
            two different test flows: 0 means the fields are ignored
  clean, random
            macro-F1 of a random forest fitted on clean training features, tested on clean
            and on randomised test features (fine-tuning on clean flows, as in sa.py)

  python kaggle/litcvit/sa_probe.py [run ...]        (default: 5gad2022 ciciot2022 iot23 ustc)

Runs on CPU (~5 min per run for 5GAD-2022, ~35 min for USTC-TFC2016). Training flows are
capped at TRAIN_CAP (seeded) to keep it there. Needs the packed dataset, encoder_random_0.pt
and the SA job's encoder_sa_random_0.pt under C:\\data\\CNS_proj\\kaggle. Writes
results/litcvit_sa_probe_seed0.csv, replacing the rows of the runs given.
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
KAGGLE = Path("C:/data/CNS_proj/kaggle")
ENCODERS = {"litcvit": KAGGLE / "litcvit_output_random" / "encoder_random_0.pt",
            "litcvit-sa": KAGGLE / "litcvit_output_sa" / "encoder_sa_random_0.pt"}
OUT = ROOT / "results" / "litcvit_sa_probe_seed0.csv"
TRAIN_CAP = 10_000


def features(encoder, X):
    """Packet tokens [N, 5 * d] and flow embeddings [N, d] of uint8 images (torch)."""
    with torch.no_grad():
        P = torch.cat([encoder.packet_tokens(X[i:i + 256]) for i in range(0, len(X), 256)])
        F = torch.cat([encoder.from_packets(P[i:i + 4096]) for i in range(0, len(P), 4096)])
    return {"packet tokens": P.flatten(1).numpy(), "flow embedding": F.numpy()}


def shift(clean, moved, rng):
    other = clean[rng.permutation(len(clean))]
    return np.linalg.norm(clean - moved, axis=1).mean() / np.linalg.norm(clean - other, axis=1).mean()


def main():
    runs = sys.argv[1:] or ["5gad2022", "ciciot2022", "iot23", "ustc"]
    encoders = {}
    for name, path in ENCODERS.items():
        encoders[name] = litcvit.Encoder()
        encoders[name].load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        encoders[name].eval()

    rows = []
    for run in runs:
        meta, X = litcvit.load(run)
        y = meta["class"].map({c: i for i, c in enumerate(sorted(meta["class"].unique()))}).to_numpy()
        rng = np.random.default_rng(0)
        train, test = litcvit.split(meta, "random", rng)
        train = np.sort(rng.permutation(train)[:TRAIN_CAP])
        X = torch.from_numpy(X)
        gen = torch.Generator().manual_seed(0)
        moved = torch.cat([litcvit.perturb(X[test[i:i + 1024]], "random", gen) for i in range(0, len(test), 1024)])
        for name, encoder in encoders.items():
            fit, clean, perturbed = features(encoder, X[train]), features(encoder, X[test]), features(encoder, moved)
            for level in fit:
                rf = RandomForestClassifier(100, class_weight="balanced", n_jobs=-1, random_state=0)
                rf.fit(fit[level], y[train])
                score = lambda Z: round(100 * f1_score(y[test], rf.predict(Z), average="macro"), 2)
                rows.append({"dataset": run, "encoder": name, "features": level, "train": len(train),
                             "test": len(test), "shift": round(shift(clean[level], perturbed[level], rng), 3),
                             "clean": score(clean[level]), "random": score(perturbed[level])})
                r = rows[-1]
                print(f"{run:11s} {name:10s} {level:14s} shift {r['shift']:.3f}  macro-F1 clean {r['clean']:6.2f}"
                      f"  random {r['random']:6.2f}", flush=True)

    df = pd.DataFrame(rows)
    if OUT.exists():
        old = pd.read_csv(OUT)
        df = pd.concat([old[~old["dataset"].isin(runs)], df])
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
