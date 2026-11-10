"""E6: packet-view stress tests for C, and E7's C_full when called with --full (sprint3/PREREG.md).

C variants are retrained locally on CPU (s3.c_model; same recipe and augmentation bank as sprint/c1_length_view.py).
F and B answers on the old conditions come from the Kaggle logits; on new attacks they are computed here, lazily
(C first, then B, then F: exact for the gate). Attacks, all on malware only:
  old, full test set:  c add 2  d split 1  e a+c  f a+d
  old, the sample:     g adaptive K=16  u unseen parameters K=16
  new, the sample:     L / La   length mimicry, header as recorded / copied, K=8
                       H / Ha   chaff (wrong seq, TTL 1-4), header as recorded / copied, K=16
                       g64      adaptive K=64 over add, split, add+split, chaff, mimicry, each header choice
  E7 only:             s24      split size 24 (unseen)    a4  4 added copies (unseen), both on the full test set
Families per C variant (PREREG E6): seen, held-out or mixed.

  python sprint3/code/e6_packets.py [--full] [split seed ...]  -> sprint3/results/e6_packets (or e7_cfull) .json/.csv
"""

import sys
import time

import numpy as np

import s3

C, L, A, M = s3.C, s3.L, s3.A, s3.M
FAMILY = {"C_A": {"c": "seen", "e": "seen", "d": "held-out", "f": "held-out", "u": "held-out", "L": "held-out",
                  "La": "held-out", "H": "held-out", "Ha": "held-out", "g": "mixed", "g64": "mixed"},
          "C_S": {"d": "seen", "f": "seen", "c": "held-out", "e": "held-out", "L": "held-out", "La": "held-out",
                  "H": "held-out", "Ha": "held-out", "u": "held-out*", "g": "mixed", "g64": "mixed"},
          "C_fix2": {"L": "held-out", "La": "held-out", "H": "held-out", "Ha": "held-out", "s24": "held-out",
                     "a4": "held-out", "c": "seen", "d": "seen", "e": "seen", "f": "seen", "g": "seen", "u": "seen",
                     "g64": "mixed"},
          "C_full": {"L": "held-out", "La": "held-out", "H": "held-out", "Ha": "held-out", "s24": "held-out",
                     "a4": "held-out", "c": "seen", "d": "seen", "e": "seen", "f": "seen", "g": "seen", "u": "seen",
                     "g64": "mixed"}}
FULL_GATE = {("random", 0), ("time", 0)}  # gate F+B without C on the new attacks: these settings only (cost)


def g64(X, copied, donors, lengths, rng, draws=64):
    """Adaptive search: per flow and draw, a random header choice and operation among add, split, add+split, chaff, mimicry."""
    n = len(X)
    out = np.empty((draws,) + X.shape, X.dtype)
    for d in range(draws):
        header = rng.integers(0, 2, n).astype(bool)
        op = rng.integers(0, 5, n)
        base = np.where(header[:, None, None], copied, X)
        cur = base.copy()
        for o in range(3):
            idx = np.flatnonzero(op == o)
            if len(idx):
                cur[idx] = _one(_one(base[idx], 0, rng), 1, rng) if o == 2 else _one(base[idx], o, rng)
        idx = np.flatnonzero(op == 3)
        if len(idx):
            cur[idx] = s3.chaff(base[idx], lengths, rng)
        idx = np.flatnonzero(op == 4)
        if len(idx):
            cur[idx] = s3.mimicry(base[idx], donors, rng)[0]
        out[d] = cur
    return out


def _one(X, op, rng):
    """add (op 0) or split (op 1) with random parameters from the attacker's range (common.COPIES, POSITIONS, SIZES)."""
    out = X.copy()
    if op == 0:
        copies, pos = rng.choice(C.COPIES, len(X)), rng.integers(0, len(C.POSITIONS), len(X))
        for c in C.COPIES:
            for p in range(len(C.POSITIONS)):
                m = (copies == c) & (pos == p)
                if m.any():
                    out[m] = C.add_at(X[m], int(c), C.POSITIONS[p])
    else:
        size = rng.choice(C.SIZES, len(X))
        for z in C.SIZES:
            m = size == z
            if m.any():
                out[m] = A.resegment(X[m], int(z))
    return out


def full_bank(run, split, seed, s):
    """E7's augmentation: one copy per malware training flow per family over the whole range: added copies 1-3 at any
    position, and splits of any size in common.SIZES (1-32)."""
    path = s3.CACHE / "aug" / f"{run}_{split}_{seed}_full.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    X, y = s["X_train"], s["y_train"]
    bad = np.flatnonzero(~s["benign"][y])
    rng = np.random.default_rng(5000 + seed)
    out = {"index": bad, "A": _one(X[bad], 0, rng), "S": _one(X[bad], 1, rng)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return out


def fix2(s, run, split, seed, bank):
    """E7 fix 2: C_full fit on 80% of the training flows (validation = 20% per class, default_rng(10 + seed)), plus a
    benign-density flag: a flow C calls Benign is called malware when its standardised Euclidean distance to the nearest
    benign reference flow (the 80%) exceeds tau, the 99th percentile of validation benign flows' distances."""
    from sklearn.neighbors import NearestNeighbors
    y = s["y_train"]
    rng = np.random.default_rng(10 + seed)
    val = np.zeros(len(y), bool)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        val[rng.permutation(idx)[:int(round(0.2 * len(idx)))]] = True
    keep = np.flatnonzero(~val)
    sub = dict(s, X_train=s["X_train"][keep], y_train=y[keep])
    ok = ~val[bank["index"]]
    extra = (s3.C1.features(np.concatenate([bank["A"][ok], bank["S"][ok]])), np.concatenate([y[bank["index"][ok]]] * 2))
    Cn = s3.c_model(sub, run, split, seed, "", tag="C_full80", extra=extra)
    benign = s["benign"]
    ref = s3.C1.features(s["X_train"][keep][benign[y[keep]]])
    mu, sd = ref.mean(0), ref.std(0) + 1e-6
    nn = NearestNeighbors(n_neighbors=1).fit((ref - mu) / sd)
    dist = lambda x: nn.kneighbors((s3.C1.features(x) - mu) / sd)[0][:, 0]
    tau = float(np.percentile(dist(s["X_train"][val & benign[y]]), 99))

    def fn(x):
        z = Cn(x)
        far = benign[z.argmax(1)] & (dist(x) > tau)
        z[np.ix_(far, benign)] = -1e9  # flagged: its top malware class wins
        return z
    fn.tau = tau
    return fn


def main(settings, full=False):
    name = "e7_cfull" if full else "e6_packets"
    path = s3.RESULTS / f"{name}.json"
    rows = s3.json.loads(path.read_text()) if path.exists() else []
    for split, seed in settings:
        for run in s3.RUNS:
            base = {"run": run, "split": split, "seed": seed}
            if any(all(r[k] == v for k, v in base.items()) for r in rows):
                continue
            start = time.time()
            s = C.suite(run, split, seed)
            y, benign, sample, k = s["y"], s["benign"], s["sample"], len(s["classes"])
            malware = ~benign[y]
            KF, KB = s3.kaggle(run, split, seed, "ce"), s3.kaggle(run, split, seed, "mask")
            Fn, Bn = s3.nets(run, split, seed, k)
            if full:
                bank = full_bank(run, split, seed, s)
                extra = (s3.C1.features(np.concatenate([bank["A"], bank["S"]])), np.concatenate([s["y_train"][bank["index"]]] * 2))
                variants = {"C_full": s3.c_model(s, run, split, seed, "", tag="C_full", extra=extra),
                            "C_fix2": fix2(s, run, split, seed, bank)}
            else:
                variants = {"C_A": s3.c_model(s, run, split, seed, "A"), "C_S": s3.c_model(s, run, split, seed, "S")}
            rng = np.random.default_rng(6000 + seed)
            X, copied = s["clean"][sample], s["a"][sample]
            donors = s["clean"][benign[y]]
            lengths = s3.benign_lengths(s["X_train"][benign[s["y_train"]]])
            new = {}
            Lm = [s3.mimicry(X, donors, rng) for _ in range(8)]
            new["L"] = np.stack([m[0] for m in Lm])
            pad = np.stack([m[1] for m in Lm])
            pushed = np.stack([m[2] for m in Lm])
            new["La"] = np.stack([s3.mimicry(copied, donors, rng)[0] for _ in range(8)])
            new["H"] = np.stack([s3.chaff(X, lengths, rng) for _ in range(16)])
            new["Ha"] = np.stack([s3.chaff(copied, lengths, rng) for _ in range(16)])
            if (split, seed) in FULL_GATE:  # PREREG amendment 2: K=64 on seed 0 and time only (CPU)
                new["g64"] = g64(X, copied, donors, lengths, rng)
            old_full = {c: s[c] for c in "cdef"}
            if full:
                old_full["s24"] = A.resegment(s["clean"], 24, only=malware)
                rr = np.random.default_rng(7000 + seed)
                pos = rr.integers(0, len(C.POSITIONS), len(malware))
                a4 = s["clean"].copy()
                for p in range(len(C.POSITIONS)):
                    m = malware & (pos == p)
                    a4[m] = C.add_at(s["clean"][m], 4, C.POSITIONS[p])
                old_full["a4"] = a4
            fb = lambda z: s3.benign_of(z, benign)
            F_clean, B_clean = fb(KF["clean"]), fb(KB["clean"])
            shared = {}  # per new attack: C's answers per variant, and B-and-F cleared, computed once for every variant
            for c, V in new.items():
                flat = V.reshape(-1, L.PACKETS, L.ROW)
                Cv = {v: fb(Cn(flat)) for v, Cn in variants.items()}
                need = np.ones(len(flat), bool) if (split, seed) in FULL_GATE and c != "g64" else np.any(list(Cv.values()), 0)
                BF = np.zeros(len(flat), bool)
                BF[need] = s3.lazy_cleared([Bn, Fn], flat[need], benign)
                shared[c] = (Cv, BF)
            for vname, Cn in variants.items():
                C_clean = fb(Cn(s["clean"]))
                gate_clean = F_clean & B_clean
                gc_clean = gate_clean & C_clean
                common_cols = {"clean_f1_gate": round(s3.macro_f1(y, s3.answer([KF["clean"], KB["clean"]], benign), k), 3),
                               "clean_f1_gateC": round(s3.macro_f1(y, s3.answer([KF["clean"], KB["clean"], Cn(s["clean"])], benign), k), 3),
                               "benign_recall_gate": round(100 * gate_clean[~malware].mean(), 3),
                               "benign_recall_gateC": round(100 * gc_clean[~malware].mean(), 3)}
                results = {}
                for c, Xc in old_full.items():  # full test set, one version per flow
                    reached = ((Xc != s["clean"]).any((1, 2)) & malware)[None]
                    Cc = fb(Cn(Xc))[None]
                    if c in KF:
                        Fc, Bc = fb(KF[c])[None], fb(KB[c])[None]
                    else:
                        idx = np.flatnonzero(reached[0])
                        Fc, Bc = np.zeros_like(Cc), np.zeros_like(Cc)
                        okc = idx[Cc[0, idx]]
                        if len(okc):
                            Bc[0, okc] = fb(Bn(Xc[okc]))
                        okb = okc[Bc[0, okc]]
                        if len(okb):
                            Fc[0, okb] = fb(Fn(Xc[okb]))
                    results[c] = {"gate+C": s3.success(gc_clean, Fc & Bc & Cc, reached), "C": s3.success(C_clean, Cc, reached),
                                  "gate": s3.success(gate_clean, Fc & Bc, reached)}
                for c in ("g", "u"):
                    V = s[c]
                    reached = (V != X[None]).any((2, 3))
                    Cc = fb(Cn(V.reshape(-1, L.PACKETS, L.ROW))).reshape(len(V), -1)
                    Fc, Bc = fb(KF[c]), fb(KB[c])
                    results[c] = {"gate+C": s3.success(gc_clean[sample], Fc & Bc & Cc, reached),
                                  "C": s3.success(C_clean[sample], Cc, reached), "gate": s3.success(gate_clean[sample], Fc & Bc, reached)}
                for c, V in new.items():
                    reached = (V != X[None]).any((2, 3))
                    Cv, BF = shared[c]
                    Cc = Cv[vname].reshape(len(V), -1)
                    gc = (Cv[vname] & BF).reshape(len(V), -1)
                    results[c] = {"gate+C": s3.success(gc_clean[sample], gc, reached), "C": s3.success(C_clean[sample], Cc, reached)}
                    if (split, seed) in FULL_GATE and c != "g64":
                        results[c]["gate"] = s3.success(gate_clean[sample], BF.reshape(len(V), -1), reached)
                    if c == "L":
                        ev = gc & reached
                        first = np.where(ev.any(0), ev.argmax(0), -1)
                        ok = first >= 0
                        results[c]["cost"] = {
                            "pad_bytes_median_all": float(np.median(pad[0][pad[0] >= 0])) if (pad[0] >= 0).any() else float("nan"),
                            "pushed_bytes_median_all": float(np.median(pushed[0][pushed[0] >= 0])) if (pushed[0] >= 0).any() else float("nan"),
                            "pad_bytes_median_evaders": float(np.median(pad[first[ok], np.flatnonzero(ok)])) if ok.any() else float("nan"),
                            "pushed_bytes_median_evaders": float(np.median(pushed[first[ok], np.flatnonzero(ok)])) if ok.any() else float("nan"),
                            "extra_packets_median_evaders": float(np.median(np.ceil(pushed[first[ok], np.flatnonzero(ok)] / 1460))) if ok.any() else float("nan")}
                for c, res in results.items():
                    for meth, r in res.items():
                        if meth == "cost":
                            continue
                        rows.append({**base, "C_variant": vname, "attack": c, "family": FAMILY[vname].get(c, "-"),
                                     "method": meth, **r, **common_cols, **res.get("cost", {})})
            s3.save(rows, name)
            print(run, split, seed, f"{time.time() - start:.0f}s",
                  {f"{r['C_variant']}/{r['attack']}": r["cond"] for r in rows if r["method"] == "gate+C" and r["run"] == run
                   and r["split"] == split and r["seed"] == seed}, flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    full = "--full" in a
    a = [x for x in a if x != "--full"]
    main([(a[i], int(a[i + 1])) for i in range(0, len(a), 2)] if a else s3.SETTINGS, full)
