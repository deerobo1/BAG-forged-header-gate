"""Training for the sprint's neural ideas, on CPU, with light.fit's protocol (AdamW, lr 1e-3, batch 256, 30 epochs, cosine
with warmup, balanced batches, class-weighted cross-entropy) and the same number of steps as a plain model on the run.

  B_A, B_S   1B: the mask model B, malware training flows plus one attacked copy each (family A: add, S: split)
  ce_s1      1D(i): a second plain model, training seed 1 on the same split
  adv8       1D(ii): one model, cross-entropy plus each malware flow's worst of 8 donor-forged headers
  nand       1A: F and B trained jointly with the noisy-AND loss; nand_wok: F's malware term on its worst of 8
  B_dro      2C: B with balanced draws over class x time quartile

  python sprint/train.py <idea> <run> [split] [seed]

Saves C:/data/CNS_proj/sprint_cache/models/<run>_<split>_<seed>_<idea>[_F|_B].pt and skips models already saved.
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import common as C

L, T, A = C.L, C.T, C.A
MODELS = C.Path(C.os.environ.get("SPRINT_MODELS", C.CACHE / "models"))
K_FORGED, AUX, B_FLAG = 8, 0.1, 0.1
SMOKE = bool(os.environ.get("SMOKE"))  # a few steps of one epoch, nothing saved: a test of the code
FORGES = ("adv8", "nand_wok")


# ---------------------------------------------------------------- augmentation bank

def bank(run, split, seed, s):
    """One attacked copy of each malware training flow per family, random parameters (seen ones only), cached."""
    path = C.CACHE / "aug" / f"{run}_{split}_{seed}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    X, y = s["X_train"], s["y_train"]
    bad = np.flatnonzero(~s["benign"][y])
    rng = np.random.default_rng(2000 + seed)
    out = {"index": bad}
    copies, position = rng.choice((1, 2), len(bad)), rng.integers(0, len(C.POSITIONS), len(bad))
    added = X[bad].copy()
    for c in (1, 2):
        for p in range(len(C.POSITIONS)):
            m = (copies == c) & (position == p)
            added[m] = C.add_at(X[bad][m], c, C.POSITIONS[p])
    size = rng.choice((1, 2, 4), len(bad))
    split_ = X[bad].copy()
    for z in (1, 2, 4):
        split_[size == z] = A.resegment(X[bad][size == z], z)
    out["A"], out["S"] = added, split_
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return out


# ---------------------------------------------------------------- losses

def log_benign(z, benign):
    """log of the softmax mass on benign classes, and on the others."""
    lp = F.log_softmax(z.float(), 1)
    return torch.logsumexp(lp[:, benign], 1), torch.logsumexp(lp[:, ~benign], 1)


def worst_forged(net, x, fields, y, benign, gen, k=K_FORGED):
    """Each malware flow of the batch, as recorded or with one of k random benign donors' header fields: whichever net
    finds most benign (chosen without gradient, in eval mode). Returns the flows and their positions in the batch."""
    tries = [x[~benign[y]]]
    for _ in range(k):
        forged, bad = T.forge(x, fields, y, benign, gen)
        tries.append(forged)
    if not len(bad):
        return x[:0], bad
    was = net.training
    net.eval()
    with torch.no_grad():
        scores = torch.stack([log_benign(net(t), benign)[0] for t in tries])
    net.train(was)
    pick = scores.argmax(0)
    return torch.stack(tries)[pick, torch.arange(len(bad))], bad


def plain_step(nets, x, y, fields, weight, gen, benign):
    return F.cross_entropy(nets[0](x), y, weight=weight)


def adv_step(nets, x, y, fields, weight, gen, benign):
    net = nets[0]
    worst, bad = worst_forged(net, x, fields, y, benign, gen)
    z = net(torch.cat([x, worst]))  # one pass, so batch norm sees both kinds together, as light's oneside loss
    value = F.cross_entropy(z[:len(x)], y, weight=weight)
    if len(bad):
        value = value + T.ALPHA / z.shape[1] * F.cross_entropy(z[len(x):], y[bad], weight=weight)
    return value


def nand_step(nets, x, y, fields, weight, gen, benign, worst=False):
    """Noisy-AND: clear = pF * pB, -log clear on benign flows, -log(1 - clear) on malware, each kind weighted 1/2; plus AUX
    cross-entropy on each model and B_FLAG on B's own malware mass. worst: F's malware term on its worst forged copy."""
    full, blind = nets
    is_bad = ~benign[y]
    zF, zB = full(x), blind(x)
    lF, _ = log_benign(zF, benign)
    lB, lB_bad = log_benign(zB, benign)
    if worst and is_bad.any():
        forged, bad = worst_forged(full, x, fields, y, benign, gen)
        if len(bad):
            lF = lF.clone()
            lF[bad] = torch.maximum(lF[bad], log_benign(full(forged), benign)[0])
    clear = (lF + lB).clamp(max=-1e-6)
    value = AUX * (F.cross_entropy(zF, y, weight=weight) + F.cross_entropy(zB, y, weight=weight))
    if (~is_bad).any():
        value = value - 0.5 * clear[~is_bad].mean()
    if is_bad.any():
        value = value - 0.5 * torch.log(-torch.expm1(clear[is_bad])).mean() - B_FLAG * lB_bad[is_bad].mean()
    return value


# ---------------------------------------------------------------- training

def fit(nets, step, X, y, groups, seed, steps, benign, forges=False, epochs=T.EPOCHS, log=print):
    """light.fit's loop for several networks sharing one optimizer; draws are balanced over groups."""
    device = L.DEVICE
    for n in nets:
        n.to(device)
    x_all = torch.from_numpy(np.ascontiguousarray(X)).to(device)
    y_all = torch.from_numpy(np.asarray(y, dtype=np.int64)).to(device)
    fields_all = T.stack_fields(x_all) if forges else None
    benign_t = torch.as_tensor(benign, device=device)
    weight = torch.tensor(L.class_weights(np.asarray(y), len(benign)), dtype=torch.float32, device=device)
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    params = [p for n in nets for p in n.parameters()]
    opt = torch.optim.AdamW(params, T.LR, weight_decay=L.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: L.lr_factor(i, steps * epochs))
    groups = torch.from_numpy(np.asarray(groups, dtype=np.int64)).to(device)
    amp = device == "cuda"  # light.fit's mixed precision on GPU; the losses take logits in float32
    scaler = torch.amp.GradScaler(enabled=amp)
    start = time.time()
    for epoch in range(epochs):
        for n in nets:
            n.train()
        total = 0.0
        for b in L.balanced_indices(groups, steps * T.BATCH, gen).view(steps, -1):
            with torch.autocast(device, dtype=torch.float16, enabled=amp):
                value = step(nets, x_all[b], y_all[b], None if fields_all is None else fields_all[b], weight, gen, benign_t)
            opt.zero_grad(set_to_none=True)
            scaler.scale(value).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += value.item()
        if (epoch + 1) % 5 == 0:
            log(f"    epoch {epoch + 1}: loss {total / steps:.4f}, {time.time() - start:.0f}s")
    return round(time.time() - start)


def train(idea, run, split="random", seed=0):
    key = f"{run}_{split}_{seed}_{idea}"
    names = ["F", "B"] if idea.startswith("nand") else [""]
    paths = [MODELS / f"{key}{'_' + n if n else ''}.pt" for n in names]
    if all(p.exists() for p in paths):
        print(f"{key}: already trained")
        return
    s = C.suite(run, split, seed)
    X, y, benign = s["X_train"], s["y_train"], s["benign"]
    k = len(s["classes"])
    steps = max(1, len(X) // T.BATCH)  # every idea gets the plain model's number of updates
    groups, train_seed = y, seed
    if idea in ("B_A", "B_S"):
        aug = bank(run, split, seed, s)
        X, y = np.concatenate([X, aug[idea[-1]]]), np.concatenate([y, y[aug["index"]]])
        groups = y
    if idea == "B_dro":
        quartile = np.zeros(len(y), int)
        for c in np.unique(y):
            m = y == c
            quartile[m] = np.minimum(4 * np.argsort(np.argsort(s["ts_train"][m], kind="stable")) // m.sum(), 3)
        groups = y * 4 + quartile
    if idea == "ce_s1":
        train_seed = 1
    masked = idea.startswith("B_")
    nets = [T.FlowNet("mobilenetv2", k, masked=True if n == "B" else masked) for n in names]
    step = {"adv8": adv_step, "nand": nand_step, "nand_wok": lambda *a: nand_step(*a, worst=True)}.get(idea, plain_step)
    print(f"{key}: {len(X)} training flows, {steps} steps per epoch", flush=True)
    if SMOKE:
        start = time.time()
        fit(nets, step, X, y, groups, train_seed, 3, benign, idea in FORGES, epochs=1)
        print(f"{key}: smoke test passed, {(time.time() - start) / 3:.2f}s per step")
        return
    seconds = fit(nets, step, X, y, groups, train_seed, steps, benign, idea in FORGES)
    MODELS.mkdir(parents=True, exist_ok=True)
    for net, path in zip(nets, paths):
        torch.save(net.state_dict(), path)
    print(f"{key}: trained in {seconds}s", flush=True)


if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2], *(sys.argv[3:4] or ["random"]), *(int(a) for a in sys.argv[4:5]))
