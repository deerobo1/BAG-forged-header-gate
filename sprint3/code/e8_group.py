"""E8: the benign-host-disjoint split, against the random split, with models cheap enough to train on CPU.

Split "group": benign flows grouped by source address; hosts shuffled (default_rng(seed)) and sent to test until at least
20% of benign flows are there; malware 80/20 at random per class (a source-disjoint malware split is impossible:
results/group_feasibility.csv). Split "random": litcvit.split's random 80/20, as everywhere else.

Models: F' XGBoost on the 1,600 bytes (300 trees, depth 8, as sprint/e0_baselines.xgb_fn); B' the same on B's input
(bytes in M zeroed); C_A (s3.c_model). Methods: F', B', gate F'+B', gate F'+B'+C_A.
Attacks on malware test flows: (a) donor copy, (b) best of 8 arbitrary, (m) best of 8 M-max, (g) adaptive K=16.

  python sprint3/code/e8_group.py [seed ...]    -> sprint3/results/e8_group.(json|csv)
"""

import sys
import time

import numpy as np

import s3

C, L, A, M = s3.C, s3.L, s3.A, s3.M


def group_split(meta, rng):
    benign = meta["class"].isin(L.BENIGN).to_numpy()
    test = np.zeros(len(meta), bool)
    hosts = meta.loc[benign, "src"].astype(str)
    order = rng.permutation(hosts.unique())
    counts = hosts.value_counts()
    need, got, chosen = 0.2 * benign.sum(), 0, set()
    for h in order:
        if got >= need:
            break
        chosen.add(h)
        got += counts[h]
    test[benign] = hosts.isin(chosen).to_numpy()
    for c in meta.loc[~benign, "class"].unique():
        idx = np.flatnonzero((meta["class"] == c).to_numpy())
        test[rng.permutation(idx)[:int(round(0.2 * len(idx)))]] = True
    return np.flatnonzero(~test), np.flatnonzero(test)


def setting(run, split, seed):
    meta, X = L.load(run)
    classes = sorted(meta["class"].unique())
    y = meta["class"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    if split == "group":
        train, test = group_split(meta, np.random.default_rng(seed))
    else:
        train, test = L.split(meta, "random", np.random.default_rng(seed))
    X = X.reshape(len(X), L.PACKETS, L.ROW)
    benign = np.isin(classes, list(L.BENIGN))
    s = {"clean": X[test], "y": y[test], "classes": np.array(classes), "benign": benign, "X_train": X[train], "y_train": y[train]}
    malware = ~benign[s["y"]]
    rng = np.random.default_rng(seed)
    donors = s["clean"][~malware]
    chosen = M.choose_donors(s["clean"], donors, rng)
    chosen[~malware] = -1
    s["a"] = M.apply_mimicry(s["clean"], donors, chosen, M.STACK)
    sample = np.random.default_rng(1).permutation(np.flatnonzero(malware))[:C.SAMPLE]
    s["sample"] = sample
    Xs = s["clean"][sample]
    s["b"] = np.stack([A.arbitrary(Xs, np.random.default_rng(100 + d), options=True) for d in range(8)])
    r = np.random.default_rng(3000 + seed)
    s["m"] = np.stack([s3.m_max(Xs, r) for _ in range(8)])
    r = np.random.default_rng(1000 + seed)
    s["g"] = np.stack([C.adaptive(Xs, s["a"][sample], r) for _ in range(16)])
    return s


def xgb(s, seed, masked, run, split):
    import xgboost
    path = s3.CACHE / "models" / f"{run}_{split}_{seed}_xgb{'_B' if masked else ''}.json"
    m = xgboost.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, tree_method="hist", random_state=seed, n_jobs=s3.os.cpu_count())
    prep = (lambda x: s3.b_input(x)) if masked else (lambda x: np.asarray(x).reshape(len(x), -1))
    if path.exists():
        m.load_model(path)
    else:
        y = s["y_train"]
        m.fit(prep(s["X_train"]), y, sample_weight=L.class_weights(y, len(s["classes"]))[y])
        path.parent.mkdir(parents=True, exist_ok=True)
        m.save_model(path)
    k = len(s["classes"])

    def fn(x):
        p = m.predict_proba(prep(x))
        if p.shape[1] < k:
            full = np.full((len(p), k), 1e-9)
            full[:, m.classes_] = p
            p = full
        return np.log(np.maximum(p, 1e-9))
    return fn


def main(seeds):
    path = s3.RESULTS / "e8_group.json"
    rows = s3.json.loads(path.read_text()) if path.exists() else []
    for seed in seeds:
        for run in s3.RUNS:
            for split in ("group", "random") if seed == 0 else ("group",):  # the random counterpart (not pre-registered): seed 0 only
                base = {"run": run, "split": split, "seed": seed}
                if any(all(r[k] == v for k, v in base.items()) for r in rows):
                    continue
                start = time.time()
                s = setting(run, split, seed)
                y, benign, sample, k = s["y"], s["benign"], s["sample"], len(s["classes"])
                malware = ~benign[y]
                fns = {"F'": xgb(s, seed, False, run, split), "B'": xgb(s, seed, True, run, split),
                       "C_A": s3.c_model(s, run, split, seed, "A")}
                Z = {n: {c: f(s[c]) if c in ("clean", "a") else f(s[c].reshape(-1, L.PACKETS, L.ROW)).reshape(len(s[c]), len(sample), -1)
                         for c in ("clean", "a", "b", "m", "g")} for n, f in fns.items()}
                reached = {"a": ((s["a"] != s["clean"]).any((1, 2)) & malware)[None]}
                for c in "bmg":
                    reached[c] = (s[c] != s["clean"][sample][None]).any((2, 3))
                for meth, mem in {"F'": ["F'"], "B'": ["B'"], "gate'": ["F'", "B'"], "gate'+C_A": ["F'", "B'", "C_A"]}.items():
                    clean = s3.answer([Z[m]["clean"] for m in mem], benign)
                    cl = benign[clean]
                    row = {**base, "method": meth, "n_test": len(y), "benign_test": int((~malware).sum()),
                           "clean_f1": round(s3.macro_f1(y, clean, k), 3), "benign_recall": round(100 * cl[~malware].mean(), 3)}
                    for c in "abmg":
                        if c == "a":
                            att = benign[s3.answer([Z[m]["a"] for m in mem], benign)][None]
                            res = s3.success(cl, att, reached["a"])
                        else:
                            att = np.stack([benign[s3.answer([Z[m][c][d] for m in mem], benign)] for d in range(len(s[c]))])
                            res = s3.success(cl[sample], att, reached[c])
                        row[f"cond_{c}"], row[f"uncond_{c}"] = res["cond"], res["uncond"]
                    rows.append(row)
                s3.save(rows, "e8_group")
                print(run, split, seed, f"{time.time() - start:.0f}s", [(r["method"], r["clean_f1"], r["benign_recall"]) for r in rows[-4:]], flush=True)


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]] or [0, 1, 2])
