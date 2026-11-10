"""Shared pieces of sprint 3: settings, the saved Kaggle logits, the networks, new attacks, gates, bootstrap.

Builds on sprint/common.py: the same cached attack suite per setting (common.suite), so every method is scored on the same
attacked flows as in the earlier sprint. Logits of the old conditions come from the Kaggle jobs' saved files
(C:/data/CNS_proj/kaggle/sprint_output_<job>/logits); new attacks are scored here with the saved weights on CPU.

New attacks (sprint3/PREREG.md):
  m        M-max: attacks.arbitrary(options=True) then mimic.shift_sequence, best of 8 draws        (inside M)
  flags    the free TCP flag bits (PSH, URG, ECE, CWR, AE, reserved) from a donor with the same control bits (outside M)
  mimicry  the payload re-framed into a benign donor's per-packet payload lengths, zero padding      (packet attack)
  chaff    1-3 inserted packets with a wrong sequence number, TTL 1-4, random payload               (packet attack)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sprint"))
import common as C  # noqa: E402
import c1_length_view as C1  # noqa: E402

L, T, A, M = C.L, C.T, C.A, C.M
KAGGLE_OUT = Path("C:/data/CNS_proj/kaggle")
RESULTS = ROOT / "sprint3" / "results"
CACHE = Path("C:/data/CNS_proj/sprint3_cache")
RUNS = L.RUNS
SETTINGS = [("random", 0), ("random", 1), ("random", 2), ("time", 0)]
JOB = {("random", 0): "seed0", ("random", 1): "seed1_light", ("random", 2): "seed2_light", ("time", 0): "time"}
URG, PSH, ECE, CWR = 0x20, 0x08, 0x40, 0x80
FREE13 = PSH | URG | ECE | CWR  # free bits in byte 13; control bits are SYN 0x02, ACK 0x10, FIN 0x01, RST 0x04
CONTROL13 = 0x17
FREE12 = 0x0F  # AE and the three reserved bits, low nibble of byte 12
torch.set_num_threads(os.cpu_count())


# ---------------------------------------------------------------- settings and saved logits

def kaggle(run, split, seed, name):
    """Saved logits of one method on the old conditions, or None if that job didn't train it."""
    path = KAGGLE_OUT / f"sprint_output_{JOB[(split, seed)]}" / "logits" / f"{run}_{split}_{seed}_{name}.npz"
    if not path.exists():
        return None
    z = np.load(path)
    return {k: z[k] for k in z.files}


def nets(run, split, seed, k):
    """F and B as functions images -> logits (CPU)."""
    return (C.flownet(C.existing(run, split, seed, "ce"), k, False),
            C.flownet(C.existing(run, split, seed, "mask"), k, True))


def b_input(x):
    """What B reads: the images with the bytes in M zeroed (litcvit.perturb 'zero'), uint8 [n, 5, 320]."""
    x = torch.as_tensor(np.ascontiguousarray(np.asarray(x).reshape(len(x), L.PACKETS, L.ROW)))
    return torch.cat([L.perturb(x[i:i + 4096], "zero", None) for i in range(0, len(x), 4096)]).numpy().reshape(len(x), -1) if len(x) else x.numpy()


def c_input(x):
    return C1.features(x)


# ---------------------------------------------------------------- gates and scores

def benign_of(z, benign):
    return benign[np.asarray(z).argmax(-1)]


def answer(zs, benign):
    """The gate's multiclass answer (common.combine)."""
    return C.combine(list(zs), benign)


def lazy_cleared(fns, X, benign, batch=4096):
    """Bool per flow: every model in fns (cheapest first) calls it Benign. Later models only see flows the earlier ones
    cleared, so the answer is exact for the gate at a fraction of the cost."""
    X = np.asarray(X)
    ok = np.ones(len(X), bool)
    for fn in fns:
        idx = np.flatnonzero(ok)
        if not len(idx):
            break
        z = np.concatenate([fn(X[idx[i:i + batch]]) for i in range(0, len(idx), batch)])
        ok[idx] = benign_of(z, benign)
    return ok


def success(cleared_clean, cleared_attacked, reached):
    """Attack success over a drawn attack. cleared_clean [n]: the method clears the flow as recorded; cleared_attacked
    [K, n] and reached [K, n]: per draw. Returns conditional and unconditional percentages and counts."""
    hit = reached.any(0)
    evade = (cleared_attacked & reached).any(0)
    det = hit & ~cleared_clean
    pct = lambda a, m: round(100 * a[m].mean(), 3) if m.any() else float("nan")
    return {"cond": pct(evade, det), "uncond": pct(evade, hit), "n_hit": int(hit.sum()), "n_det": int(det.sum())}


def macro_f1(y, p, k):
    """Macro-F1 over the classes present in y or p (as sklearn's default), from counts."""
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp, fp, fn = np.diag(cm), cm.sum(0) - np.diag(cm), cm.sum(1) - np.diag(cm)
    present = (cm.sum(0) + cm.sum(1)) > 0
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    return 100 * f1[present].mean()


def boot_f1(y, preds, k, n=2000, seed=7):
    """Paired bootstrap of macro-F1 for several answers on the same flows: array [n, len(preds)]."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, len(preds)))
    for b in range(n):
        i = rng.integers(0, len(y), len(y))
        out[b] = [macro_f1(y[i], p[i], k) for p in preds]
    return out


def holm(pvals):
    """Holm-adjusted p-values, same order."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, (len(p) - r) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ---------------------------------------------------------------- new attacks

def m_max(X, rng):
    """One draw of M-max on flows [n, 5, 320]: random stack values with options, then new ISNs."""
    return M.shift_sequence(A.arbitrary(X, rng, options=True), rng)


def control_free(X):
    """Per packet: control bits (byte 13 & 0x17) and free bits (byte 13 & 0xE8, byte 12 & 0x0F), -1 where absent/non-TCP."""
    x = np.asarray(X).reshape(len(X), L.PACKETS, L.ROW)
    ihl, doff, flags, usable = M.layout(x)
    at = lambda off: np.take_along_axis(x, np.minimum(ihl + off, L.ROW - 1)[..., None], -1)[..., 0].astype(int)
    present = doff > 0
    ctrl = np.where(present, flags & CONTROL13, -1)
    free = np.where(present, (flags & FREE13) | (at(12) & FREE12) << 8, -1)
    return ctrl, free, ihl, doff, usable


def set_free(row, ihl, free):
    """One packet row with its free flag bits set to free (byte 13 bits in the low byte, byte 12 bits << 8); TCP checksum
    updated (RFC 1624), urgent pointer zeroed when URG becomes set."""
    new = row.copy()
    new[ihl + 13] = (int(row[ihl + 13]) & ~FREE13 & 0xFF) | (free & FREE13)
    new[ihl + 12] = (int(row[ihl + 12]) & 0xF0) | (free >> 8 & FREE12)
    if free & URG and not row[ihl + 13] & URG:
        new[ihl + 18:ihl + 20] = 0
    new[ihl + 16:ihl + 18] = M.updated_checksum(row[ihl + 16:ihl + 18], row[ihl + 12:ihl + 20], new[ihl + 12:ihl + 20])
    return new


def free_bit_table(X_benign):
    """Per position and control value: observed free-bit patterns of benign flows and their counts."""
    ctrl, free, _, _, usable = control_free(X_benign)
    table = {}
    for i in np.flatnonzero(usable):
        for k in range(L.PACKETS):
            if ctrl[i, k] >= 0:
                table.setdefault((k, int(ctrl[i, k])), []).append(int(free[i, k]))
    return {key: np.unique(v, return_counts=True) for key, v in table.items()}


def flags_attack(X, donors, table, rng, draws=8):
    """flags: K versions of each flow [K, n, 5, 320]. Draw 0 copies the free bits of a random donor with the same control
    bits at every position (if one exists, else a pattern drawn as below); draws 1.. take, per packet, a free-bit pattern
    drawn from the benign table for that position and control value (frequency-weighted). Usable TCP flows only."""
    X = np.asarray(X).reshape(len(X), L.PACKETS, L.ROW)
    ctrl, _, ihl, _, usable = control_free(X)
    d_ctrl, d_free, _, _, d_usable = control_free(donors)
    by_ctrl = {}
    for j in np.flatnonzero(d_usable):
        by_ctrl.setdefault(d_ctrl[j].tobytes(), []).append(j)
    out = np.repeat(X[None], draws, 0)
    donor_found = np.zeros(len(X), bool)
    for i in np.flatnonzero(usable):
        pool = by_ctrl.get(ctrl[i].tobytes())
        for d in range(draws):
            if d == 0 and pool:
                j = pool[rng.integers(len(pool))]
                pattern = d_free[j]
                donor_found[i] = True
            else:
                pattern = np.full(L.PACKETS, -1)
                for k in range(L.PACKETS):
                    if ctrl[i, k] >= 0 and (k, int(ctrl[i, k])) in table:
                        vals, counts = table[(k, int(ctrl[i, k]))]
                        pattern[k] = rng.choice(vals, p=counts / counts.sum())
            for k in range(L.PACKETS):
                if ctrl[i, k] >= 0 and pattern[k] >= 0:
                    out[d, i, k] = set_free(out[d, i, k], int(ihl[i, k]), int(pattern[k]))
    return out, donor_found


def _payload(row, ihl, doff):
    total = int(row[2]) << 8 | int(row[3])
    n = max(0, min(total - ihl - doff, L.ROW - L.HEADER))
    return row[L.HEADER:L.HEADER + n], max(0, total - ihl - doff)


def _with_payload(row, ihl, doff, data, length):
    """A packet row carrying `length` payload bytes, of which `data` are shown; total length, IP and TCP checksums follow."""
    new = row.copy()
    new[L.HEADER:] = 0
    shown = data[:L.ROW - L.HEADER]
    new[L.HEADER:L.HEADER + len(shown)] = shown
    new[2:4] = list(int(ihl + doff + length).to_bytes(2, "big"))
    new[10:12] = M.updated_checksum(row[10:12], row[:ihl], new[:ihl])
    # TCP checksum covers the payload; like attacks.resegment, update by the change we can see (header and shown bytes)
    old_cov = np.concatenate([row[ihl:ihl + doff], row[L.HEADER:], [0] * (len(row[L.HEADER:]) % 2)]).astype(np.uint8)
    new_cov = np.concatenate([new[ihl:ihl + doff], new[L.HEADER:], [0] * (len(new[L.HEADER:]) % 2)]).astype(np.uint8)
    new[ihl + 16:ihl + 18] = M.updated_checksum(row[ihl + 16:ihl + 18], old_cov, new_cov)
    return new


def mimicry(X, donors, rng):
    """(L) for usable TCP flows: copy a donor with the same control bits; the flow's payload stream (as far as the image
    shows it) is cut into the donor's per-packet payload lengths, zero padding where it runs out. Sequence numbers of later
    packets of the same side move with the bytes. Returns the flows, padding bytes added and bytes pushed out of the window
    per flow (-1: no donor)."""
    X = np.asarray(X).reshape(len(X), L.PACKETS, L.ROW)
    ctrl, _, ihl, doff, usable = control_free(X)
    d_ctrl, _, d_ihl, d_doff, d_usable = control_free(donors)
    donors = np.asarray(donors).reshape(len(donors), L.PACKETS, L.ROW)
    by_ctrl = {}
    for j in np.flatnonzero(d_usable):
        by_ctrl.setdefault(d_ctrl[j].tobytes(), []).append(j)
    out, pad, pushed = X.copy(), np.full(len(X), -1), np.full(len(X), -1)
    for i in np.flatnonzero(usable):
        pool = by_ctrl.get(ctrl[i].tobytes())
        if not pool:
            continue
        j = pool[rng.integers(len(pool))]
        sides = A.packet_sides(X[i], ihl[i], X[i, :, :][np.arange(L.PACKETS), ihl[i] + 13])
        streams, length_of = {0: [], 1: []}, {0: 0, 1: 0}
        for k in range(L.PACKETS):
            if sides[k] is None:
                continue
            data, n = _payload(X[i, k], int(ihl[i, k]), int(doff[i, k]))
            streams[sides[k]].append(data)
            length_of[sides[k]] += n
        streams = {s: np.concatenate(v) if v else np.zeros(0, np.uint8) for s, v in streams.items()}
        used, added, seq_move = {0: 0, 1: 0}, 0, {0: 0, 1: 0}
        for k in range(L.PACKETS):
            if sides[k] is None:
                continue
            s = sides[k]
            want = int.from_bytes(donors[j, k, 2:4].tobytes(), "big") - int(d_ihl[j, k]) - int(d_doff[j, k])
            want = max(0, want)
            take = streams[s][used[s]:used[s] + want]
            real = min(want, max(0, length_of[s] - used[s]))
            added += want - real
            _, orig_n = _payload(X[i, k], int(ihl[i, k]), int(doff[i, k]))
            row = _with_payload(X[i, k], int(ihl[i, k]), int(doff[i, k]), take, want)
            l4 = int(ihl[i, k])
            seq = int.from_bytes(row[l4 + 4:l4 + 8].tobytes(), "big")
            new_seq = (seq + seq_move[s]) % 2 ** 32
            fixed = row.copy()
            fixed[l4 + 4:l4 + 8] = list(new_seq.to_bytes(4, "big"))
            fixed[l4 + 16:l4 + 18] = M.updated_checksum(row[l4 + 16:l4 + 18], row[l4 + 4:l4 + 8], fixed[l4 + 4:l4 + 8])
            out[i, k] = fixed
            seq_move[s] += want - orig_n
            used[s] += want
        pad[i] = added
        pushed[i] = sum(max(0, length_of[s] - used[s]) for s in (0, 1))
    return out, pad, pushed


def chaff(X, lengths, rng):
    """(H) for usable TCP flows: 1-3 chaff packets inserted at a random position 1-4. Each copies the header of the packet
    before it, with a random sequence number (so the receiver drops it), TTL 1-4, random IP ID, a random payload whose
    length is drawn from `lengths` (benign payload lengths at that position, a list per position); checksums follow."""
    X = np.asarray(X).reshape(len(X), L.PACKETS, L.ROW)
    _, doff, _, usable = M.layout(X)
    ihl = (X[..., 0].astype(int) & 15) * 4
    out = X.copy()
    for i in np.flatnonzero(usable):
        present = np.flatnonzero(doff[i] > 0)
        pos = int(rng.integers(1, L.PACKETS))
        if pos > present[-1] + 1:
            pos = int(present[-1]) + 1
        copies = int(rng.integers(1, 4))
        src = X[i, pos - 1]
        l4, header = int(ihl[i, pos - 1]), int(doff[i, pos - 1])
        pieces = []
        for _ in range(copies):
            n = int(rng.choice(lengths[pos])) if len(lengths[pos]) else 0
            data = rng.integers(0, 256, n, dtype=np.uint8)
            row = _with_payload(src, l4, header, data, n)
            fake = row.copy()
            fake[l4 + 4:l4 + 8] = rng.integers(0, 256, 4, dtype=np.uint8)
            fake[8] = rng.integers(1, 5)
            fake[4:6] = rng.integers(0, 256, 2, dtype=np.uint8)
            fake[10:12] = M.updated_checksum(row[10:12], row[:l4], fake[:l4])
            fake[l4 + 16:l4 + 18] = M.updated_checksum(row[l4 + 16:l4 + 18], row[l4 + 4:l4 + 8], fake[l4 + 4:l4 + 8])
            pieces.append(fake)
        out[i] = np.stack([*X[i, :pos], *pieces, *X[i, pos:]][:L.PACKETS])
    return out


def benign_lengths(X_benign):
    """Payload lengths of benign flows per position (for chaff)."""
    x = np.asarray(X_benign).reshape(len(X_benign), L.PACKETS, L.ROW)
    ihl, doff, _, usable = M.layout(x)
    total = x[..., 2].astype(int) << 8 | x[..., 3]
    pay = np.clip(total - ihl - doff, 0, 1460)
    return [pay[usable & (doff[:, k] > 0), k] for k in range(L.PACKETS)]


# ---------------------------------------------------------------- local C models (XGBoost on the 41 features)

def c_model(s, run, split, seed, families, tag=None, extra=None):
    """A local, cached C: XGBoost (200 trees, depth 6, class-weighted) as in c1_length_view, trained on CPU on the training
    flows plus the named augmentation families from train.bank, plus `extra` (X, y) if given. Returns fn(images)->logits."""
    import xgboost as xgb
    import train as TR
    name = tag or ("C_" + families if families else "C")
    path = CACHE / "models" / f"{run}_{split}_{seed}_{name}.json"
    m = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, tree_method="hist", random_state=seed, n_jobs=os.cpu_count())
    if path.exists():
        m.load_model(path)
    else:
        X, y = s["X_train"], s["y_train"]
        if families:
            aug = TR.bank(run, split, seed, s)
            X = np.concatenate([X, *[aug[f] for f in families]])
            y = np.concatenate([y, *[y[aug["index"]] for _ in families]])
        feats = C1.features(X)
        if extra is not None:
            feats, y = np.concatenate([feats, extra[0]]), np.concatenate([y, extra[1]])
        m.fit(feats, y, sample_weight=L.class_weights(y, len(s["classes"]))[y])
        path.parent.mkdir(parents=True, exist_ok=True)
        m.save_model(path)
    k = len(s["classes"])

    def fn(x):
        p = m.predict_proba(C1.features(x))
        if p.shape[1] < k:  # a class absent from training
            full = np.full((len(p), k), 1e-9)
            full[:, m.classes_] = p
            p = full
        return np.log(np.maximum(p, 1e-9))
    return fn


def save(rows, name):
    import pandas as pd
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(rows, indent=1, default=float))
    pd.DataFrame(rows).to_csv(RESULTS / f"{name}.csv", index=False)
