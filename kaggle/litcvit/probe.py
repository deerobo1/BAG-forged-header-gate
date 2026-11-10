"""What LitCVit's frozen encoder keeps: a random forest on its features (Gate 2 diagnosis).

Fine-tuning keeps the stem and byte block frozen, so their packet tokens are all
the fine-tuned layers ever see. A random forest (as in baselines/header_payload.py)
on those tokens, random split seed 0 as in litcvit.py, shows how much of the label
survives them. Inputs per run: the raw 1,600 bytes; packet tokens and the flow
embedding from the pretrained encoder; the same from an untrained encoder (seed 0).

  python kaggle/litcvit/probe.py [run ...]        (default: 5gad2022)

Runs on CPU (~3 min per encoder for 5GAD-2022, ~30 min for USTC-TFC2016). Needs the
packed dataset in C:\\data\\CNS_proj\\kaggle\\litcvit-flows and the Kaggle run's
encoder_random_0.pt in C:\\data\\CNS_proj\\kaggle\\litcvit_output_random. Writes
results/litcvit_probe_seed0.csv, replacing the rows of the runs given.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score

import litcvit

ROOT = Path(__file__).resolve().parent.parent.parent
ENCODER = Path("C:/data/CNS_proj/kaggle/litcvit_output_random/encoder_random_0.pt")
OUT = ROOT / "results" / "litcvit_probe_seed0.csv"


def features(encoder, X):
    """Packet tokens [N, 5 * d] and flow embeddings [N, d]."""
    with torch.no_grad():
        P = torch.cat([encoder.packet_tokens(torch.from_numpy(X[i:i + 256])) for i in range(0, len(X), 256)])
        F = torch.cat([encoder.from_packets(P[i:i + 4096]) for i in range(0, len(P), 4096)])
    return P.flatten(1).numpy(), F.numpy()


def main():
    runs = sys.argv[1:] or ["5gad2022"]
    pretrained = litcvit.Encoder()
    pretrained.load_state_dict(torch.load(ENCODER, map_location="cpu", weights_only=True))
    torch.manual_seed(0)
    encoders = {"pretrained": pretrained.eval(), "untrained": litcvit.Encoder().eval()}

    rows = []
    for run in runs:
        meta, X = litcvit.load(run)
        classes = sorted(meta["class"].unique())
        y = meta["class"].map({c: i for i, c in enumerate(classes)}).to_numpy()
        train, test = litcvit.split(meta, "random", np.random.default_rng(0))
        inputs = {("raw bytes", "-"): X.reshape(len(X), -1)}
        for name, encoder in encoders.items():
            inputs[("packet tokens", name)], inputs[("flow embedding", name)] = features(encoder, X)
        for (inp, enc), Z in inputs.items():
            rf = RandomForestClassifier(100, class_weight="balanced", n_jobs=-1, random_state=0)
            pred = rf.fit(Z[train], y[train]).predict(Z[test])
            recall = recall_score(y[test], pred, labels=range(len(classes)), average=None, zero_division=0)
            rows.append({"dataset": run, "input": inp, "encoder": enc,
                         "accuracy": round(100 * accuracy_score(y[test], pred), 2),
                         "macro_f1": round(100 * f1_score(y[test], pred, average="macro"), 2),
                         "recall": json.dumps({c: round(100 * r, 1) for c, r in zip(classes, recall)})})
            print(f"{run:12s} {inp:15s} {enc:10s} acc {rows[-1]['accuracy']:6.2f}  "
                  f"macro-F1 {rows[-1]['macro_f1']:6.2f}", flush=True)

    df = pd.DataFrame(rows)
    if OUT.exists():
        old = pd.read_csv(OUT)
        df = pd.concat([old[~old["dataset"].isin(runs)], df])
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
