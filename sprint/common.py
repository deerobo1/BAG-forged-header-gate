"""Shared pieces of the sprint: the attack suite (a)-(g), cached logits per model, gates and scores.

The suite for one run, split and seed is built once and cached in CACHE, so every method is scored on the same
attacked flows. Attacks touch malware test flows only, as in kaggle/litcvit/limits.py:
  a  mimic_stack (light.test_sets)            b  best of 8 attacks.arbitrary draws with options, on SAMPLE flows
  c  add_packets(copies=2)                     d  resegment(1)
  e  a + c                                     f  a + d
  g  adaptive: 16 random packet/payload settings per flow, on the same sample   u  the same, unseen parameters only
Logits of each model on every condition are cached as well, so a gate costs nothing once its members are scored.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "kaggle" / "litcvit"))
import attacks as A  # noqa: E402
import light as T  # noqa: E402
import mimic as M  # noqa: E402

L = T.L
torch.set_num_threads(os.cpu_count())
CACHE = Path(os.environ.get("SPRINT_CACHE", "C:/data/CNS_proj/sprint_cache"))  # suites, augmentation banks, logits
KAGGLE = Path(os.environ.get("SPRINT_WEIGHTS", "C:/data/CNS_proj/kaggle"))  # the existing jobs' saved weights
RESULTS = Path(os.environ.get("SPRINT_RESULTS", ROOT / "sprint" / "results"))
SCREEN = ["5gad2022", "mcfp_strict", "mcfp"]
DRAWS_B, DRAWS_G, SAMPLE = 8, 16, 2000
COPIES, POSITIONS, SIZES = (1, 2, 3), (None, 0, 1, 2, 3, 4), (1, 2, 4, 8, 16, 32)
UNSEEN_COPIES, UNSEEN_SIZES = (3,), (8, 16, 32)
FIXED = ("clean", "a", "c", "d", "e", "f")
DRAWN = {"b": DRAWS_B, "g": DRAWS_G, "u": DRAWS_G}


# ---------------------------------------------------------------- attacks

def add_at(X, copies, position):
    """attacks.add_packets on every usable TCP flow, repeating packet `position` (None: the first bare ACK, as there)."""
    if position is None:
        return A.add_packets(X, copies)
    out = np.asarray(X).reshape(len(X), M.PACKETS, M.ROW).copy()
    _, doff, _, usable = M.layout(out)
    for i in np.flatnonzero(usable & (doff[:, position] > 0)):
        out[i] = np.stack([*out[i, :position + 1], *[out[i, position]] * copies, *out[i, position + 1:]][:M.PACKETS])
    return out


def adaptive(plain, copied, rng, unseen=False):
    """One draw of the adaptive attacker on flows [n, 5, 320]: each flow gets its own random setting."""
    n = len(plain)
    header = rng.integers(0, 2, n)
    op = rng.integers(0, 3, n)  # 0 add, 1 split, 2 add then split
    copies = rng.choice(UNSEEN_COPIES if unseen else COPIES, n)
    position = rng.integers(0, len(POSITIONS), n)
    size = rng.choice(UNSEEN_SIZES if unseen else SIZES, n)
    out = plain.copy()
    settings = np.stack([header, op, copies, position, size], 1)
    for s in np.unique(settings, axis=0):
        idx = np.flatnonzero((settings == s).all(1))
        x = (copied if s[0] else plain)[idx]
        if s[1] in (0, 2):
            x = add_at(x, int(s[2]), POSITIONS[s[3]])
        if s[1] in (1, 2):
            x = A.resegment(x, int(s[4]))
        out[idx] = x
    return out


def suite(run, split="random", seed=0):
    """The cached attack suite: dict of arrays (see the module docstring), plus y, classes, sample and train data."""
    path = CACHE / "suite" / f"{run}_{split}_{seed}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files}
    data = T.test_sets(run, split, seed)
    y, classes = data["y_test"], np.array(data["classes"])
    benign = np.isin(classes, list(T.BENIGN))
    malware = ~benign[y]
    images = data["conditions"]["clean"].reshape(-1, L.PACKETS, L.ROW)
    copied = data["conditions"]["mimic_stack"].reshape(images.shape)
    out = {"clean": images, "a": copied, "c": A.add_packets(images, only=malware), "d": A.resegment(images, 1, only=malware),
           "e": A.add_packets(copied, only=malware), "f": A.resegment(copied, 1, only=malware)}
    sample = np.random.default_rng(1).permutation(np.flatnonzero(malware))[:SAMPLE]
    out["sample"] = sample
    out["b"] = np.stack([A.arbitrary(images[sample], np.random.default_rng(100 + d), options=True) for d in range(DRAWS_B)])
    rng = np.random.default_rng(1000 + seed)
    out["g"] = np.stack([adaptive(images[sample], copied[sample], rng) for _ in range(DRAWS_G)])
    out["u"] = np.stack([adaptive(images[sample], copied[sample], rng, unseen=True) for _ in range(DRAWS_G)])
    meta, X = L.load(run)
    train, test = L.split(meta, split, np.random.default_rng(seed))
    out.update(y=y, classes=classes, benign=benign, X_train=X[train].reshape(-1, L.PACKETS, L.ROW), y_train=data["y_train"],
               ts_train=meta["ts_first"].to_numpy()[train], ts_test=meta["ts_first"].to_numpy()[test])
    for c in FIXED[1:]:
        out["kept_" + c] = A.payload_kept(out[c])
    out["kept_clean"] = A.payload_kept(images)
    for c in DRAWN:
        out["kept_" + c] = np.stack([A.payload_kept(x) for x in out[c]])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return out


# ---------------------------------------------------------------- logits

def flownet(path, k, masked):
    net = T.FlowNet("mobilenetv2", k, masked=masked)
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.to(L.DEVICE).eval()
    return lambda x: T.logits_of(net, x, batch=512)


def existing(run, split, seed, loss):
    """Path of the current pipeline's MobileNetV2 weights for ce (F) or mask (B)."""
    if split == "time":
        return KAGGLE / "litcvit_output_gate" / "preds" / f"{run}_time_mobilenetv2{'_mask' if loss == 'mask' else ''}.pt"
    if seed == 0:
        return KAGGLE / "litcvit_output_ablate" / "preds" / f"{run}_mobilenetv2_{loss}.pt"
    return KAGGLE / "litcvit_output_seeds" / "preds" / f"{run}_mobilenetv2_{loss}_seed{seed}.pt"


def logits(s, key, fn, blind=False):
    """Logits of model fn on every condition of suite s, cached under key. A blind model reads no shortcut field, so
    its answers to (a) and (b) are those to the clean flows; that is checked on (a), not assumed."""
    path = CACHE / "logits" / f"{key}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    out = {c: fn(s[c]) for c in FIXED if not (blind and c == "a")}
    if blind:
        check = fn(s["a"])
        if not np.allclose(check, out["clean"], atol=1e-4):
            raise SystemExit(f"{key}: a header-blind model answered the header attack differently")
        out["a"] = out["clean"]
        out["b"] = np.repeat(out["clean"][s["sample"]][None], DRAWS_B, 0)
    else:
        out["b"] = fn(s["b"].reshape(-1, L.PACKETS, L.ROW)).reshape(DRAWS_B, len(s["sample"]), -1)
    for c in ("g", "u"):
        out[c] = fn(s[c].reshape(-1, L.PACKETS, L.ROW)).reshape(DRAWS_G, len(s["sample"]), -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return out


# ---------------------------------------------------------------- methods and scores

def combine(zs, benign, flag=None):
    """Answers of a gate: the first model's, unless it says Benign and a later one doesn't (then that one's), so a flow is
    cleared only if every model says Benign. flag: bool per flow forced not to be cleared (a rule), given the first
    model's top malware class."""
    pred = zs[0].argmax(-1)
    for z in zs[1:]:
        p = z.argmax(-1)
        pred = np.where(benign[pred] & ~benign[p], p, pred)
    if flag is not None:
        top_bad = np.where(benign, -np.inf, zs[0]).argmax(-1)
        pred = np.where(benign[pred] & flag, top_bad, pred)
    return pred


def thin(kept, k):
    return (kept >= 0) & (kept < k)


def score(s, members, blind=(), rule_k=0, name=None):
    """Scores of a method: members are logits dicts (first = base), blind the logits dicts of its header-blind members."""
    y, benign, sample = s["y"], s["benign"], s["sample"]
    malware = ~benign[y]
    answer = lambda c, d=slice(None): combine([m[c][d] for m in members], benign,
                                              thin(s["kept_" + c][d], rule_k) if rule_k else None)
    clean = answer("clean")
    row = {"method": name, "clean_f1": round(100 * f1_score(y, clean, average="macro"), 2),
           "benign_recall": round(100 * (~malware & benign[clean]).sum() / (~malware).sum(), 2),
           "detected": round(100 * (~benign[clean])[malware].mean(), 2)}
    if blind:
        row["certified"] = round(100 * np.any([~benign[b["clean"].argmax(1)] for b in blind], 0)[malware].mean(), 2)
    for c in FIXED[1:]:
        changed = (s[c] != s["clean"]).any((1, 2)) & malware
        m = changed & ~benign[clean]
        called = benign[answer(c)]
        row["evaded_" + c] = round(100 * called[m].mean(), 2) if m.any() else float("nan")
        row["uncond_" + c] = round(100 * called[changed].mean(), 2) if changed.any() else float("nan")
    for c in DRAWN:
        reached = (s[c] != s["clean"][sample][None]).any((2, 3))
        called = np.stack([benign[answer(c, d)] & reached[d] for d in range(len(reached))])
        hit = reached.any(0)
        m = hit & ~benign[clean[sample]]
        row["evaded_" + c] = round(100 * called.any(0)[m].mean(), 2) if m.any() else float("nan")
        row["uncond_" + c] = round(100 * called.any(0)[hit].mean(), 2) if hit.any() else float("nan")
    row["worst_packet"] = max(row[f"evaded_{c}"] for c in "cdefg" if not np.isnan(row[f"evaded_{c}"]))
    return row


def save(rows, name):
    """Raw JSON and a CSV summary under sprint/results."""
    import pandas as pd
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(rows, indent=1, default=float))
    pd.DataFrame(rows).to_csv(RESULTS / f"{name}.csv", index=False)
