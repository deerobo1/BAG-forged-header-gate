"""1C: a third, header-blind view of each flow: packet lengths and flags of the five packets, read by XGBoost.

Per packet: present, IP version, protocol, IP header length, IP total length, TCP data offset, TCP flags, payload
length; plus the payload bytes shown in all. None of these is a shortcut field, so the view is as blind to the header
attacks (a), (b) as B is (common.logits checks it on (a)). Direction is left out: inferring it needs sequence numbers,
which the attacker sets. Timing isn't in the images (00_inventory.md).

Variants: C (training flows as recorded), C_A, C_S, C_AS (malware flows plus attacked copies, train.bank's families).
Methods scored: C alone, the gate (F, B), the gate with C (cleared only if F, B and C all say Benign), and the gate
with the 41-byte rule, all from the same cached suite.

  python sprint/c1_length_view.py <split> <seed> [runs...]

Writes sprint/results/c1_<split>_<seed>.json and .csv.
"""

import sys
import time

import numpy as np

import common as C
import e0_baselines as E0
import train as TR

L = C.L
VARIANTS = {"C": "", "C_A": "A", "C_S": "S", "C_AS": "AS"}


def features(X):
    """[n, 5, 320] uint8 -> [n, 41] lengths and flags."""
    x = np.asarray(X).reshape(len(X), L.PACKETS, L.ROW).astype(np.int64)
    present = x[..., :L.HEADER].any(-1)
    version = x[..., 0] >> 4
    ipv4 = version == 4
    ihl = np.where(ipv4, (x[..., 0] & 15) * 4, 40)
    proto = np.where(ipv4, x[..., 9], x[..., 6])
    total = np.where(ipv4, x[..., 2] << 8 | x[..., 3], (x[..., 4] << 8 | x[..., 5]) + 40)
    at = lambda off: np.take_along_axis(x, np.clip(ihl + off, 0, L.HEADER - 1)[..., None], -1)[..., 0]
    tcp = proto == 6
    doff = np.where(tcp, (at(12) >> 4) * 4, np.where(proto == 17, 8, 0))
    flags = np.where(tcp, at(13), 0)
    payload = np.clip(total - ihl - doff, 0, None) * present
    kept = C.A.payload_kept(X)
    cols = [present, version, proto, ihl, total, doff, flags, payload]
    return np.concatenate([np.stack(cols, -1).reshape(len(x), -1), kept[:, None]], 1).astype(np.float32)


def length_fn(s, run, split, seed, families):
    model = []

    def fn(x):
        if not model:
            import xgboost as xgb
            X, y = s["X_train"], s["y_train"]
            if families:
                aug = TR.bank(run, split, seed, s)
                X = np.concatenate([X, *[aug[f] for f in families]])
                y = np.concatenate([y, *[y[aug["index"]] for _ in families]])
            m = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, tree_method="hist", device=L.DEVICE, random_state=seed)
            m.fit(features(X), y, sample_weight=L.class_weights(y, len(s["classes"]))[y])
            model.append(m)
        return np.log(np.maximum(model[0].predict_proba(features(x)), 1e-9))
    return fn


def main(split="random", seed=0, runs=None):
    name = f"c1_{split}_{seed}"
    path = C.RESULTS / f"{name}.json"
    rows = C.json.loads(path.read_text()) if path.exists() else []
    for run in runs or L.RUNS:
        if any(r["run"] == run for r in rows):
            continue
        start = time.time()
        s = C.suite(run, split, seed)
        F, B = E0.members(run, split, seed, s)
        base = {"run": run, "split": split, "seed": seed}
        rows.append({**base, **C.score(s, [F, B], (B,), name="gate")})
        rows.append({**base, **C.score(s, [F, B], (B,), rule_k=41, name="gate_k41")})
        for variant, families in VARIANTS.items():
            V = C.logits(s, f"{run}_{split}_{seed}_{variant}", length_fn(s, run, split, seed, families), blind=True)
            rows.append({**base, **C.score(s, [V], (V,), name=variant)})
            rows.append({**base, **C.score(s, [F, B, V], (B, V), name="gate+" + variant)})
        for r in rows[-10:]:
            print({k: r[k] for k in ("method", "clean_f1", "benign_recall", "certified", "evaded_c", "evaded_d", "evaded_e",
                                      "evaded_f", "evaded_g", "evaded_u") if k in r}, flush=True)
        C.save(rows, name)
        print(f"{run}: done, {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3:] or None)
