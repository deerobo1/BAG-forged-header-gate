"""1D(iii), 2A and 2B from the cached logits of the current gate's F and B (e0_baselines.py must have run). CPU, minutes.

1D(iii)  B overrules F only when B's malware-minus-Benign log mass exceeds t; t is the smallest margin keeping Benign
         recall within a budget of F alone, chosen on half the test flows and scored on the other half.
2A       abstain: both Benign -> clear, both malware -> alert, disagree -> defer to an analyst. Alerts and deferrals per
         day at 1% and 0.1% malware prevalence for an ASSUMED 100,000 flows per day (a parameter, not a measurement).
2B       conformal threshold on B's malware mass from held-out malware, alpha = 0.05: random split (random halves),
         time split static (latest 20% of B's own training malware: in-sample, so optimistic) and rolling (earliest
         half of the test period calibrates, the later half is scored).

  python sprint/analysis_saved.py

Writes sprint/results/d3_threshold, a2_abstain and b2_conformal (.json and .csv).
"""

import numpy as np
import torch

import common as C

L = C.L
BUDGETS = (0.5, 1.0, 2.0)
ALPHA, VOLUME, PREVALENCE = 0.05, 100_000, (0.01, 0.001)
SETTINGS = [("random", 0), ("random", 1), ("random", 2), ("time", 0)]


def masses(z, benign):
    lp = torch.log_softmax(torch.as_tensor(z, dtype=torch.float64), -1)
    return torch.logsumexp(lp[..., ~torch.as_tensor(benign)], -1).numpy(), torch.logsumexp(lp[..., torch.as_tensor(benign)], -1).numpy()


def margin(z, benign):
    bad, good = masses(z, benign)
    return bad - good


def thresholded(F, B, benign, t, c, d=slice(None)):
    """Benign-or-not answers: F's, but B flags where its margin exceeds t."""
    return benign[F[c][d].argmax(-1)] & ~(margin(B[c][d], benign) > t)


def evaded_on(s, cleared, half, c):
    """evaded (common.score's definition) on the flows in half, given cleared(condition, draw) -> bool per flow."""
    y, benign, sample = s["y"], s["benign"], s["sample"]
    malware = ~benign[y]
    clean = cleared("clean", slice(None))
    if c in C.FIXED:
        m = (s[c] != s["clean"]).any((1, 2)) & malware & ~clean & half
        return round(100 * cleared(c, slice(None))[m].mean(), 2) if m.any() else float("nan")
    reached = (s[c] != s["clean"][sample][None]).any((2, 3))
    hit = np.stack([cleared(c, d) & reached[d] for d in range(len(reached))]).any(0)
    m = reached.any(0) & ~clean[sample] & half[sample]
    return round(100 * hit[m].mean(), 2) if m.any() else float("nan")


def d3(s, F, B, run, split, seed):
    benign, y = s["benign"], s["y"]
    malware = ~benign[y]
    half = np.random.default_rng(7).random(len(y)) < 0.5  # True: scored; False: chooses t
    f_ok = benign[F["clean"].argmax(1)]
    mg = margin(B["clean"], benign)
    f_recall = lambda m: 100 * f_ok[~malware & m].mean()
    rows = []
    for budget in (*BUDGETS, None):
        if budget is None:
            t, label = None, "gate"
            cleared = lambda c, d: benign[C.combine([F[c][d], B[c][d]], benign)]
            certified = ~benign[B["clean"].argmax(1)]
        else:
            cal = ~malware & ~half
            allowed = int(f_ok[cal].sum() - np.ceil((f_recall(~half) - budget) / 100 * cal.sum()))  # benign flows B may flag
            v = np.sort(mg[cal & f_ok])[::-1]
            t = float(v[allowed]) if 0 <= allowed < len(v) else float(mg.min() - 1)
            label = f"threshold_{budget}"
            cleared = lambda c, d, t=t: thresholded(F, B, benign, t, c, d)
            certified = mg > t
        clean = cleared("clean", slice(None))
        row = {"run": run, "split": split, "seed": seed, "method": label, "t": t,
               "f_benign_recall": round(f_recall(half), 2), "benign_recall": round(100 * clean[~malware & half].mean(), 2),
               "certified": round(100 * certified[malware & half].mean(), 2)}
        for c in ("a", "b", "c", "d", "e", "f", "g"):
            row["evaded_" + c] = evaded_on(s, cleared, half, c)
        rows.append(row)
    return rows


def a2(s, F, B, run, split, seed):
    benign, y = s["benign"], s["y"]
    malware = ~benign[y]
    rows = []
    for c in ("clean", "a", "e", "f"):
        f_ok, b_ok = benign[F[c].argmax(1)], benign[B[c].argmax(1)]
        clear, alert = f_ok & b_ok, ~f_ok & ~b_ok
        defer = ~clear & ~alert
        changed = (s[c] != s["clean"]).any((1, 2)) if c != "clean" else np.ones(len(y), bool)
        mal = malware & changed
        row = {"run": run, "split": split, "seed": seed, "condition": c,
               "malware_cleared": round(100 * clear[mal].mean(), 2), "malware_deferred": round(100 * defer[mal].mean(), 2),
               "malware_alerted": round(100 * alert[mal].mean(), 2)}
        if c == "clean":
            decided = ~defer
            wrong = (clear & malware) | (alert & ~malware)
            row.update(coverage=round(100 * decided.mean(), 2), risk_decided=round(100 * wrong[decided].mean() if decided.any() else 0, 2),
                       benign_deferred=round(100 * defer[~malware].mean(), 2), benign_alerted=round(100 * alert[~malware].mean(), 2))
            gate_flag = ~clear  # the gate flags every flow not cleared
            for p in PREVALENCE:
                tag = str(p).replace("0.", "p")
                row[f"gate_alerts_{tag}"] = round(VOLUME * (p * gate_flag[malware].mean() + (1 - p) * gate_flag[~malware].mean()))
                row[f"gate_false_alerts_{tag}"] = round(VOLUME * (1 - p) * gate_flag[~malware].mean())
                row[f"abstain_alerts_{tag}"] = round(VOLUME * (p * alert[malware].mean() + (1 - p) * alert[~malware].mean()))
                row[f"abstain_false_alerts_{tag}"] = round(VOLUME * (1 - p) * alert[~malware].mean())
                row[f"abstain_deferrals_{tag}"] = round(VOLUME * (p * defer[malware].mean() + (1 - p) * defer[~malware].mean()))
        rows.append(row)
    return rows


def conformal(scores_cal, alpha=ALPHA):
    """The threshold, or NaN when there are too few calibration flows for one (fewer than 1/alpha - 1)."""
    s = np.sort(scores_cal)
    if len(s) < 1 / alpha - 1:
        return np.nan
    k = int(np.floor(alpha * (len(s) + 1)))
    return -np.inf if k < 1 else s[k - 1]


def b2(s, F, B, run, split, seed):
    benign, y = s["benign"], s["y"]
    malware = ~benign[y]
    score = masses(B["clean"], benign)[0]  # log malware mass
    f_ok = benign[F["clean"].argmax(1)]
    out = []

    def row(kind, thr, evaluate, n_cal):
        flagged = score >= thr
        ok = not np.isnan(thr)
        return {"run": run, "split": split, "seed": seed, "calibration": kind, "n_cal": int(n_cal), "threshold": float(thr),
                "n_eval_malware": int((malware & evaluate).sum()), "n_eval_benign": int((~malware & evaluate).sum()),
                "coverage": round(100 * flagged[malware & evaluate].mean(), 2) if ok else np.nan, "target": 100 * (1 - ALPHA),
                "benign_recall": round(100 * (f_ok & ~flagged)[~malware & evaluate].mean(), 2) if ok else np.nan,
                "argmax_certified": round(100 * (~benign[B["clean"].argmax(1)])[malware & evaluate].mean(), 2)}

    if split == "random":
        half = np.random.default_rng(11).random(len(y)) < 0.5
        out.append(row("random_halves", conformal(score[malware & ~half]), half, (malware & ~half).sum()))
    else:
        bad_train = np.flatnonzero(~benign[s["y_train"]])
        late = bad_train[s["ts_train"][bad_train] >= np.quantile(s["ts_train"][bad_train], 0.8)]
        net = C.flownet(C.existing(run, split, seed, "mask"), len(s["classes"]), True)
        cal = masses(net(s["X_train"][late]), benign)[0]
        order = s["ts_test"]
        later = order >= np.median(order)
        out.append(row("static_train_late20", conformal(cal), np.ones(len(y), bool), len(cal)))
        out.append(row("static_train_late20_laterhalf", conformal(cal), later, len(cal)))
        out.append(row("rolling_test_firsthalf", conformal(score[malware & ~later]), later, (malware & ~later).sum()))
    return out


def main():
    rows = {"d3_threshold": [], "a2_abstain": [], "b2_conformal": []}
    for split, seed in SETTINGS:
        for run in L.RUNS:
            key = f"{run}_{split}_{seed}"
            if not (C.CACHE / "logits" / f"{key}_mask.npz").exists():
                print(f"{key}: no cached logits, skipped")
                continue
            s = C.suite(run, split, seed)
            F = C.logits(s, key + "_ce", None)
            B = C.logits(s, key + "_mask", None, blind=True)
            rows["d3_threshold"] += d3(s, F, B, run, split, seed)
            rows["a2_abstain"] += a2(s, F, B, run, split, seed)
            rows["b2_conformal"] += b2(s, F, B, run, split, seed)
            print(f"{key}: done", flush=True)
    for name, r in rows.items():
        C.save(r, name)


if __name__ == "__main__":
    main()
