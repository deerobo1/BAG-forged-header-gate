"""Which malware flows still pass as Benign under the attack, and could any model catch them?

Tests a guess in plan.txt: that the counterfactual and one-sided losses stay weak on IoT-23 because
its malware flows hold nothing but their headers. From saved logits (kaggle/litcvit_loss and, once
downloaded, kaggle/litcvit_oneside; more folders may be given as arguments), for each run, model and
loss, on attacked flows called malicious as recorded:
  flows                       how many
  bare                        share with no payload byte in any kept packet
  copies                      share that, once attacked, match a benign test flow of the same run on
                              every byte but sequence and acknowledgement numbers and checksums.
                              A model that doesn't read those has no way to tell the two apart
  evaded, _bare, _payload     share called Benign under mimic_stack: all, without and with payload
  evaded_copies, _rest        the same among copies and the others
Also benign_recall on clean flows, since a model can avoid being evaded by calling everything malware.

  python kaggle/litcvit/evaded_flows.py [more folders of predictions]

Writes results/litcvit_evaded_flows.csv. Takes minutes on CPU, since it rebuilds the attacked flows.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import light as T

L, M = T.L, T.M
ROOT = Path(__file__).resolve().parent.parent.parent
KAGGLE = Path("C:/data/CNS_proj/kaggle")
PREDS = [KAGGLE / "litcvit_output_loss" / "preds", KAGGLE / "litcvit_output_oneside" / "preds"]
LOSSES = ("ce", "mask", "cfnce", "oneside")
CONDITION = "mimic_stack"


def has_payload(X):
    """Per flow, whether any byte follows the TCP header in any kept packet. Only for flows mimicry can change."""
    X = X.reshape(len(X), M.PACKETS, M.ROW)
    ihl, doff, _, _ = M.layout(X)
    return ((X != 0) & (np.arange(M.ROW) >= (ihl + doff)[..., None])).any((1, 2))


def fingerprint(X):
    """Each flow's bytes without sequence and acknowledgement numbers and checksums, as a hashable value."""
    X = X.reshape(len(X), M.PACKETS, M.ROW).copy()
    ihl, _, _, _ = M.layout(X)
    X[..., 10:12] = 0
    for offset in (*range(4, 12), 16, 17):
        np.put_along_axis(X, (ihl + offset)[..., None], 0, -1)
    return [row.tobytes() for row in X]


def share(mask, within):
    return round(100 * mask[within].mean(), 1) if within.any() else float("nan")


def scores(data, path, benign_flow):
    f = np.load(path, allow_pickle=False)
    y, classes = f["y"], f["classes"].astype(str)
    assert np.array_equal(f["row"], data["rows"]) and np.array_equal(f["attacked_" + CONDITION], data["attacked"][CONDITION]), path
    benign = np.isin(classes, list(T.BENIGN))
    clean, attacked = f["clean"].argmax(1), f[CONDITION].argmax(1)
    m = f["attacked_" + CONDITION] & ~benign[clean]
    bare, copies = ~data["payload"], data["copies"]
    called = benign[attacked]
    return {"flows": int(m.sum()), "bare": share(bare, m), "copies": share(copies, m), "evaded": share(called, m),
            "evaded_bare": share(called, m & bare), "evaded_payload": share(called, m & ~bare),
            "evaded_copies": share(called, m & copies), "evaded_rest": share(called, m & ~copies),
            "benign_recall": share(clean == y, benign_flow)}


def main(folders=PREDS, out=ROOT / "results" / "litcvit_evaded_flows.csv"):
    rows = []
    for run in L.RUNS:
        paths = {p.stem: p for folder in folders for p in sorted(Path(folder).glob(f"{run}_*.npz"))}
        named = {name: name.removeprefix(run + "_").split("_") for name in paths}  # mcfp's pattern also finds mcfp_strict's files
        paths = {name: p for name, p in paths.items() if len(named[name]) == 2 and named[name][1] in LOSSES}
        if not paths:
            continue
        data = T.test_sets(run, "random", 0)
        y_benign = np.isin(np.array(data["classes"])[data["y_test"]], list(T.BENIGN))
        images, attacked = data["conditions"]["clean"], data["conditions"][CONDITION]
        known = set(fingerprint(images[y_benign]))
        data["payload"] = has_payload(images)
        data["copies"] = np.array([flow in known for flow in fingerprint(attacked)]) & data["attacked"][CONDITION]
        for name, path in paths.items():
            model, loss = name.removeprefix(run + "_").rsplit("_", 1)
            rows.append({"dataset": run, "model": model, "loss": loss, **scores(data, path, y_benign)})
            print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main(PREDS + sys.argv[1:], ROOT / "results" / "litcvit_evaded_flows.csv")
