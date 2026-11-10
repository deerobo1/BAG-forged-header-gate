"""Why the length view (C_A) lets the adaptive attacker (g) through on 5GAD. No training; reads the cached suite and the
Kaggle job's saved logits.

  python sprint/diag_5gad.py [run] [job] [variant]        defaults: 5gad2022 seed0 C_A

The settings of (g) aren't saved, so they are replayed from its seed: common.adaptive draws them from
default_rng(1000 + seed) in a fixed order, and add_at/resegment draw nothing. The replay is checked by reproducing
the job's evaded_g. Writes sprint/results/diag_<run>_<job>.json.
"""

import json
import sys

import numpy as np
import pandas as pd

import c1_length_view as C1
import common as C

JOBS = {"seed0": ("random", 0), "seed1_light": ("random", 1), "seed2_light": ("random", 2), "time": ("time", 0)}
FIELDS = ["header", "op", "copies", "position", "size"]
OPS = ["add", "split", "add+split"]


def replay(n, seed, unseen=False):
    """The (g) settings, [draws, n, 5], in the order common.adaptive draws them."""
    rng = np.random.default_rng(1000 + seed)
    out = []
    for kind in ([False] * C.DRAWS_G + [True] * C.DRAWS_G):
        s = np.stack([rng.integers(0, 2, n), rng.integers(0, 3, n),
                      rng.choice(C.UNSEEN_COPIES if kind else C.COPIES, n), rng.integers(0, len(C.POSITIONS), n),
                      rng.choice(C.UNSEEN_SIZES if kind else C.SIZES, n)], -1)
        out.append(s)
    return np.stack(out[:C.DRAWS_G]), np.stack(out[C.DRAWS_G:])


def main(run="5gad2022", job="seed0", variant="C_A"):
    split, seed = JOBS[job]
    method = "gate+" + variant
    s = C.suite(run, split, seed)
    logits = C.KAGGLE / f"sprint_output_{job}" / "logits"
    F, B, V = [dict(np.load(logits / f"{run}_{split}_{seed}_{k}.npz")) for k in ("ce", "mask", variant)]
    y, benign, sample, classes = s["y"], s["benign"], s["sample"], s["classes"]
    clean = C.combine([F["clean"], B["clean"], V["clean"]], benign)
    detected = ~benign[clean[sample]]
    settings, _ = replay(len(sample), seed)

    reached = (s["g"] != s["clean"][sample][None]).any((2, 3))
    says_benign = {k: benign[m["g"].argmax(-1)] for k, m in (("F", F), ("B", B), ("C", V))}
    cleared = says_benign["F"] & says_benign["B"] & says_benign["C"] & reached
    flow_evaded = cleared.any(0) & detected & reached.any(0)
    rate = 100 * flow_evaded[detected & reached.any(0)].mean()
    expected = pd.read_csv(C.ROOT / "sprint" / "results" / "verdicts_runs.csv").query(
        "run == @run and setting == @job and method == @method").evaded_g.iloc[0]
    print(f"{run} {job}: replayed evaded_g {rate:.2f}, job reported {expected:.2f}")
    if abs(rate - expected) > 0.01:
        raise SystemExit("replay doesn't match the job: suite or logits differ")

    report = {"run": run, "job": job, "variant": variant, "evaded_g": round(rate, 2)}
    # per (draw, flow) attempt on detected malware: who says Benign
    ok = reached & detected[None]
    report["attempt_share_benign"] = {k: round(100 * v[ok].mean(), 2) for k, v in says_benign.items()}
    report["attempt_cleared"] = round(100 * cleared[ok].mean(), 2)
    # which setting values clear flows (rate per value, over attempts)
    flat = settings[ok]
    hit = cleared[ok]
    by = {}
    for i, f in enumerate(FIELDS):
        vals = flat[:, i]
        by[f] = {str(OPS[v] if f == "op" else C.POSITIONS[v] if f == "position" else v): round(100 * hit[vals == v].mean(), 2)
                 for v in np.unique(vals)}
    report["cleared_by_setting"] = by
    # by class
    cls = y[sample]
    report["evaded_by_class"] = {str(classes[c]): {"flows": int(((cls == c) & detected).sum()),
                                                   "evaded": round(100 * flow_evaded[(cls == c) & detected].mean(), 2)}
                                 for c in np.unique(cls[detected])}
    # are cleared attacked flows identical, in C's features, to benign training flows?
    feats_tr = C1.features(s["X_train"])
    ben_tr = benign[s["y_train"]]
    keys = lambda f: [r.tobytes() for r in f]
    ben_keys, mal_keys = set(keys(feats_tr[ben_tr])), set(keys(feats_tr[~ben_tr]))
    d, f = np.nonzero(cleared & detected[None])
    got = C1.features(s["g"][d, f])
    k = keys(got)
    report["cleared_attempts"] = len(k)
    report["cleared_features_match_benign_train"] = round(100 * np.mean([x in ben_keys for x in k]), 2) if k else None
    report["cleared_features_match_malware_train"] = round(100 * np.mean([x in mal_keys for x in k]), 2) if k else None
    # the same for the clean malware flows and for clean benign test flows, for scale
    clean_mal = C1.features(s["clean"][sample][detected])
    report["clean_malware_match_benign_train"] = round(100 * np.mean([x in ben_keys for x in keys(clean_mal)]), 2)
    report["benign_train_flows"], report["benign_unique_feature_rows"] = int(ben_tr.sum()), len(ben_keys)
    # the most common cleared feature rows, per packet (present, proto, total length, flags, payload)
    rows = pd.DataFrame(got[:, :40].reshape(-1, 5, 8)[:, :, [0, 2, 4, 6, 7]].reshape(len(got), -1).astype(int))
    top = rows.astype(str).agg(" ".join, axis=1).value_counts().head(5)
    report["top_cleared_rows"] = {"columns": "per packet: present proto total flags payload", **top.to_dict()}
    ben_rows = pd.DataFrame(feats_tr[ben_tr][:, :40].reshape(-1, 5, 8)[:, :, [0, 2, 4, 6, 7]].reshape(int(ben_tr.sum()), -1).astype(int))
    report["top_benign_train_rows"] = ben_rows.astype(str).agg(" ".join, axis=1).value_counts().head(5).to_dict()

    print(json.dumps(report, indent=1))
    (C.RESULTS / f"diag_{run}_{job}_{variant}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
