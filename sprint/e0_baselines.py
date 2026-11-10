"""E0: the baselines on the full attack suite. Saved weights and XGBoost, on CPU.

  plain     MobileNetV2 ce (F) alone              gate      F gated by the mask model B (decision form)
  xgb       XGBoost on the 1,600 bytes            xgb_gate  XGBoost gated by B
  gate_k41  the gate, and a TCP flow showing fewer than 41 payload bytes is never cleared (near_empty.py's rule)

  python sprint/e0_baselines.py random 0 [runs...]

Writes sprint/results/e0_<split>_<seed>.json and .csv, adding to rows already there.
"""

import sys
import time

import numpy as np

import common as C

L = C.L


def xgb_fn(s, seed):
    """XGBoost as in kaggle/litcvit_evasion/evasion.py, trained on first use (not at all if its logits are cached)."""
    model = []

    def fn(x):
        if not model:
            import xgboost as xgb
            model.append(xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, tree_method="hist", device=L.DEVICE, random_state=seed))
            y = s["y_train"]
            model[0].fit(s["X_train"].reshape(len(y), -1), y, sample_weight=L.class_weights(y, len(s["classes"]))[y])
        return np.log(np.maximum(model[0].predict_proba(np.asarray(x).reshape(len(x), -1)), 1e-9))
    return fn


def members(run, split, seed, s):
    k = len(s["classes"])
    key = f"{run}_{split}_{seed}"
    F = C.logits(s, key + "_ce", C.flownet(C.existing(run, split, seed, "ce"), k, False))
    B = C.logits(s, key + "_mask", C.flownet(C.existing(run, split, seed, "mask"), k, True), blind=True)
    return F, B


def main(split="random", seed=0, runs=None):
    name = f"e0_{split}_{seed}"
    path = C.RESULTS / f"{name}.json"
    rows = C.json.loads(path.read_text()) if path.exists() else []
    for run in runs or L.RUNS:
        if any(r["run"] == run for r in rows):
            continue
        start = time.time()
        s = C.suite(run, split, seed)
        print(f"{run}: suite ready, {time.time() - start:.0f}s", flush=True)
        F, B = members(run, split, seed, s)
        X = C.logits(s, f"{run}_{split}_{seed}_xgb", xgb_fn(s, seed))
        for method, args in {"plain": ([F], ()), "gate": ([F, B], (B,)), "xgb": ([X], ()), "xgb_gate": ([X, B], (B,))}.items():
            rows.append({"run": run, "split": split, "seed": seed, **C.score(s, *args, name=method)})
        rows.append({"run": run, "split": split, "seed": seed, **C.score(s, [F, B], (B,), rule_k=41, name="gate_k41")})
        for r in rows[-5:]:
            print({k: r[k] for k in ("method", "clean_f1", "benign_recall", "certified", "evaded_a", "evaded_b", "evaded_c",
                                      "evaded_d", "evaded_e", "evaded_f", "evaded_g", "evaded_u") if k in r}, flush=True)
        C.save(rows, name)
        print(f"{run}: done, {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3:] or None)
