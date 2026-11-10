"""E3: forging the free TCP flag bits (PSH, URG, ECE, CWR, AE, reserved), which are outside M (sprint3/GUARANTEE.md).

Variants on the 2,000-flow malware sample, 8 draws each (draw 0 copies a donor's free bits when one with the same control
bits exists; draws 1-7 sample benign free-bit patterns per position, s3.flags_attack):
  f1     flags only            f1a    flags, then donor header copy (mimic_stack on the flagged flows)
  f1m    flags, then M-max (s3.m_max)
Scored: C (local C_A), B, and the gates F+B and F+B+C, both "donor only" (draw 0, the pre-registered attack) and best of 8.
Gates are evaluated lazily (s3.lazy_cleared), which is exact for them; F alone isn't scored.

  python sprint3/code/e3_flags.py [split seed ...]     -> sprint3/results/e3_flags.(json|csv)
"""

import sys
import time

import numpy as np

import s3

C, L, M = s3.C, s3.L, s3.M
DRAWS = 8


def main(settings):
    path = s3.RESULTS / "e3_flags.json"
    rows = s3.json.loads(path.read_text()) if path.exists() else []
    for split, seed in settings:
        for run in s3.RUNS:
            base = {"run": run, "split": split, "seed": seed}
            if any(all(r[k] == v for k, v in base.items()) for r in rows):
                continue
            start = time.time()
            s = C.suite(run, split, seed)
            y, benign, sample, k = s["y"], s["benign"], s["sample"], len(s["classes"])
            Fn, Bn = s3.nets(run, split, seed, k)
            Cn = s3.c_model(s, run, split, seed, "A")
            donors = s["clean"][benign[y]]
            table = s3.free_bit_table(s["X_train"][benign[s["y_train"]]])
            rng = np.random.default_rng(4000 + seed)
            X = s["clean"][sample]
            f1, donor_found = s3.flags_attack(X, donors, table, rng, DRAWS)
            f1a = np.stack([M.apply_mimicry(d, donors, M.choose_donors(d, donors, rng), M.STACK) for d in f1])
            f1m = np.stack([s3.m_max(d, rng) for d in f1])
            clean = {"B": s3.benign_of(Bn(X), benign), "C": s3.benign_of(Cn(X), benign)}
            clean["F"] = s3.benign_of(Fn(X), benign)
            clean["gate"] = clean["F"] & clean["B"]
            clean["gate+C"] = clean["gate"] & clean["C"]
            for name, V in (("f1", f1), ("f1a", f1a), ("f1m", f1m)):
                reached = (V != X[None]).any((2, 3))
                flat = V.reshape(-1, L.PACKETS, L.ROW)
                b_diff = int(sum((s3.b_input(v[r]) != s3.b_input(X[r])).any(1).sum() for v, r in zip(V, reached)))
                att = {"C": s3.benign_of(Cn(flat), benign).reshape(DRAWS, -1),
                       "B": s3.benign_of(Bn(flat), benign).reshape(DRAWS, -1)}
                att["gate"] = s3.lazy_cleared([Bn, Fn], flat, benign).reshape(DRAWS, -1)
                att["gate+C"] = s3.lazy_cleared([Cn, Bn, Fn], flat, benign).reshape(DRAWS, -1)
                for meth in ("C", "B", "gate", "gate+C"):
                    for mode, d in (("donor", slice(0, 1)), ("best8", slice(None))):
                        res = s3.success(clean[meth], att[meth][d], reached[d])
                        rows.append({**base, "attack": name, "method": meth, "mode": mode, **res,
                                     "flows_with_donor": int(donor_found.sum()), "b_input_differs": b_diff})
            s3.save(rows, "e3_flags")
            show = [r for r in rows[-24:] if r["mode"] == "best8"]
            print(run, split, seed, {f"{r['attack']}/{r['method']}": r["uncond"] for r in show}, f"{time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main([(a[i], int(a[i + 1])) for i in range(0, len(a), 2)] if a else s3.SETTINGS)
