"""Seconds per training epoch of MobileNetV2 on this laptop's CPU, per run, to size the sprint."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kaggle" / "litcvit"))
import numpy as np, torch
import light as T
L = T.L
import os; torch.set_num_threads(int(os.environ.get("THREADS", "12")))
for run in sys.argv[1:]:
    meta, X = L.load(run)
    train, _ = L.split(meta, "random", np.random.default_rng(0))
    classes = sorted(meta["class"].unique())
    y = meta["class"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    net = T.FlowNet("mobilenetv2", len(classes))
    start = time.time()
    T.fit(net, X[train], y[train], "ce", 0, epochs=1)
    print(run, len(train), "train flows", round(time.time() - start), "s/epoch", flush=True)
