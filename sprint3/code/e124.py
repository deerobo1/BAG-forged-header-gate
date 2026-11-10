"""E1 invariance assertion, E2 bound check, E4 masking versus gate (sprint3/PREREG.md), for every setting.

Old conditions (a), (b) use the Kaggle jobs' saved logits; the new M-max attack (m) is scored here with the saved F and B
weights. XGBoost, adv8 and the second plain model have no local weights, so their (m) entries are NaN.

  python sprint3/code/e124.py [split seed ...]      (default: all four settings, all 7 runs)

Writes sprint3/results/e1_invariance, e2_bound, e4_methods, e4_boot (.json and .csv), resuming per (run, split, seed).
"""

import sys
import time

import numpy as np

import s3

C, L = s3.C, s3.L
DRAWS = 8


def drawn_m(s, seed):
    rng = np.random.default_rng(3000 + seed)
    X = s["clean"][s["sample"]]
    return np.stack([s3.m_max(X, rng) for _ in range(DRAWS)])


def e1_rows(base, s, Xm):
    """Byte-level comparison of B's input and C's features, before and after each M-attack, on attacked malware."""
    malware = ~s["benign"][s["y"]]
    rows = []
    tests = {"a": (s["clean"], s["a"][None])} | {c: (s["clean"][s["sample"]], X) for c, X in (("b", s["b"]), ("m", Xm))}
    for attack, (clean, drawn) in tests.items():
        if attack == "a":
            clean, drawn = clean[malware], drawn[:, malware]
        cb, cc = s3.b_input(clean), s3.c_input(clean)
        changed = b_diff = c_diff = 0
        for X in drawn:
            ch = (X != clean).any((1, 2))
            changed += int(ch.sum())
            b_diff += int((s3.b_input(X[ch]) != cb[ch]).any(1).sum())
            c_diff += int((s3.c_input(X[ch]) != cc[ch]).any(1).sum())
        rows.append({**base, "attack": attack, "draws": len(drawn), "flows_changed": changed,
                     "b_input_differs": b_diff, "c_features_differ": c_diff, "pass": b_diff == 0 and c_diff == 0})
    return rows


def cleared_sets(s, z):
    """Per member logits dict -> bool cleared arrays for clean, a, b [8, n], m [8, n]."""
    benign = s["benign"]
    return {c: s3.benign_of(z[c], benign) for c in z}


def main(settings):
    names = ("e1_invariance", "e2_bound", "e4_methods", "e4_boot")
    done = {n: s3.json.loads((s3.RESULTS / f"{n}.json").read_text()) if (s3.RESULTS / f"{n}.json").exists() else [] for n in names}
    for split, seed in settings:
        for run in s3.RUNS:
            base = {"run": run, "split": split, "seed": seed}
            if any(all(r[k] == v for k, v in base.items()) for r in done["e4_methods"]):
                continue
            start = time.time()
            s = C.suite(run, split, seed)
            y, benign, sample, k = s["y"], s["benign"], s["sample"], len(s["classes"])
            malware = ~benign[y]
            K = {n: s3.kaggle(run, split, seed, n) for n in ("ce", "mask", "C_A", "xgb", "adv8", "ce_s1")}
            Fn, Bn = s3.nets(run, split, seed, k)
            Xm = drawn_m(s, seed)
            flat = Xm.reshape(-1, L.PACKETS, L.ROW)
            zFm = Fn(flat).reshape(DRAWS, len(sample), -1)
            zBm = Bn(flat).reshape(DRAWS, len(sample), -1)
            b_logit_gap = float(np.abs(zBm - K["mask"]["clean"][sample][None]).max())
            done["e1_invariance"] += [{**r, "b_logit_max_gap_m": b_logit_gap} for r in e1_rows(base, s, Xm)]
            e1_ok = all(r["pass"] for r in done["e1_invariance"][-3:])

            # logits per member and condition; invariant members' answers on (b) and (m) are their clean answers (E1)
            Z = {}
            for n, z in K.items():
                if z is None:
                    continue
                Z[n] = {"clean": z["clean"], "a": z["a"], "b": z["b"]}
            Z["ce"]["m"], Z["mask"]["m"] = zFm, zBm
            Z["C_A"]["m"] = np.repeat(K["C_A"]["clean"][sample][None], DRAWS, 0)
            reached = {"a": (s["a"] != s["clean"]).any((1, 2)) & malware,
                       "b": (s["b"] != s["clean"][sample][None]).any((2, 3)),
                       "m": (Xm != s["clean"][sample][None]).any((2, 3))}
            methods = {"F": ["ce"], "B": ["mask"], "C_A": ["C_A"], "gate": ["ce", "mask"], "gate+C_A": ["ce", "mask", "C_A"],
                       "xgb": ["xgb"], "xgb_gate": ["xgb", "mask"], "adv8": ["adv8"], "adv8_gate": ["adv8", "mask"],
                       "and_plain": ["ce", "ce_s1"]}
            invariant = {"gate": ["mask"], "gate+C_A": ["mask", "C_A"], "xgb_gate": ["mask"], "adv8_gate": ["mask"], "B": ["mask"]}
            preds = {}
            for meth, mem in methods.items():
                if any(m not in Z for m in mem):
                    continue
                row = {**base, "method": meth}
                clean = s3.answer([Z[m]["clean"] for m in mem], benign)
                preds[meth] = clean
                row["clean_f1"] = round(s3.macro_f1(y, clean, k), 3)
                row["benign_recall"] = round(100 * benign[clean][~malware].mean(), 3)
                row["detected"] = round(100 * (~benign[clean])[malware].mean(), 3)
                cl_clean = benign[clean]
                for c in ("a", "b", "m"):
                    if any(c not in Z[m] for m in mem):
                        row[f"uncond_{c}"] = row[f"cond_{c}"] = float("nan")
                        continue
                    if c == "a":
                        att = benign[s3.answer([Z[m]["a"] for m in mem], benign)][None]
                        r_, cc = reached["a"][None], cl_clean
                    else:
                        att = np.stack([benign[s3.answer([Z[m][c][d] for m in mem], benign)] for d in range(DRAWS)])
                        r_, cc = reached[c], cl_clean[sample]
                    res = s3.success(cc, att, r_)
                    row[f"uncond_{c}"], row[f"cond_{c}"] = res["uncond"], res["cond"]
                    if meth in invariant:  # E2: bound from the invariant members' answers as recorded
                        inv_clear = np.all([benign[Z[m]["clean"].argmax(1)] for m in invariant[meth]], 0)
                        inv_clear = inv_clear if c == "a" else inv_clear[sample]
                        hit = r_.any(0)
                        evade = (att & r_).any(0)
                        det = hit & ~cc
                        done["e2_bound"].append({**base, "gate": meth, "attack": c, "e1_pass": e1_ok,
                            "n_hit": int(hit.sum()), "cleared_after": int(evade[hit].sum()), "bound_uncond": int(inv_clear[hit].sum()),
                            "n_detected": int(det.sum()), "cleared_after_of_detected": int(evade[det].sum()),
                            "bound_cond": int(inv_clear[det].sum()),
                            "uncond_pct": res["uncond"], "bound_uncond_pct": round(100 * inv_clear[hit].mean(), 3) if hit.any() else float("nan"),
                            "cond_pct": res["cond"], "bound_cond_pct": round(100 * inv_clear[det].mean(), 3) if det.any() else float("nan"),
                            "violation": bool(evade[hit].sum() > inv_clear[hit].sum() or evade[det].sum() > inv_clear[det].sum())})
                hdr = [row[f"uncond_{c}"] for c in "abm" if not np.isnan(row[f"uncond_{c}"])]
                row["header_uncond_max"] = max(hdr) if hdr else float("nan")
                hdr = [row[f"cond_{c}"] for c in "abm" if not np.isnan(row[f"cond_{c}"])]
                row["header_cond_max"] = max(hdr) if hdr else float("nan")
                done["e4_methods"].append(row)
            boot = s3.boot_f1(y, [preds[m] for m in ("F", "B", "gate", "gate+C_A")], k)
            point = {m: s3.macro_f1(y, preds[m], k) for m in ("F", "B", "gate", "gate+C_A")}
            for g in ("gate", "gate+C_A"):
                gi = 2 if g == "gate" else 3
                for ref, ri, margin in (("B", 1, 0.0), ("F", 0, -2.0)):
                    d = boot[:, gi] - boot[:, ri]
                    done["e4_boot"].append({**base, "method": g, "vs": ref, "diff": round(point[g] - point[ref], 3),
                        "lo": round(float(np.percentile(d, 2.5)), 3), "hi": round(float(np.percentile(d, 97.5)), 3),
                        "h0_margin": margin, "p_one_sided": round(float((d <= margin).mean()), 4)})
            for n in names:
                s3.save(done[n], n)
            print(f"{run} {split} {seed}: E1 {'pass' if e1_ok else 'FAIL'}, {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main([(a[i], int(a[i + 1])) for i in range(0, len(a), 2)] if a else s3.SETTINGS)
