"""LitCVit reimplementation on Kaggle: contrastive pretraining, then fine-tuning on every dataset (plan step 6, Gate 2).

Follows the paper's Algorithms 1 and 2 and Table II; plan.txt (Model) lists where
the paper is silent and what we chose. Encoder, for a flow image of 5 packets x
320 bytes:
  stem    per packet: Conv1D 1 -> STEM_CHANNELS (kernel STEM_KERNEL; depthwise with
          a channel multiplier, as the input has one channel), BN, GELU; pointwise
          Conv1D -> d, BN, GELU. One token per byte, sinusoidal position encoding
          over the flow's 1,600 bytes
  byte    transformer block (LN, low-rank factorized attention, LN, FFN) within
          windows of BYTE_WINDOW bytes; depthwise conv; mean pool each packet into
          a packet token
  packet  block over the 5 packet tokens, each attending to its PACKET_WINDOW
          neighbours; depthwise conv
  flow    block with global attention over the packet tokens; mean pool
Factorized attention projects to rank r: softmax(Q K^T / tau) V with
Q, K, V = X W (d x r) and a learnable tau, then a linear map back to d. One block
per stage, so "the last two blocks" unfrozen in fine-tuning are packet and flow.

Pretraining: NT-Xent (tau 0.3) between two views of each flow (15% of bytes
zeroed, and with probability 0.5 one of packets 2-5 zeroed), two-layer MLP
projection head, AdamW 2e-3, weight decay 1e-4, batch 128, cosine schedule with 5%
warmup, PRETRAIN_EPOCHS epochs, class-balanced sampling. It uses only the training
flows of USTC-TFC2016 and CICIoT2022 (the paper's choice), so their test flows are
never seen.

Fine-tuning: three-layer MLP head (GELU, dropout 0.2), class-weighted cross-entropy
with class-balanced sampling, AdamW 2e-4, weight decay 1e-4, batch 64, cosine with
5% warmup, FINETUNE_EPOCHS epochs; encoder frozen for the first FREEZE_EPOCHS, then
the packet and flow blocks train. Stem and byte block stay frozen throughout, so
their packet tokens are computed once per flow.

Shortcut-aware variant (used by kaggle/litcvit_sa/sa.py, not by main() here):
shortcut_fields marks the header bytes that can identify a host or a clock (TTL, IP ID,
TCP window, sequence numbers, options with timestamps, checksums, ...); augment_sa makes
views with those bytes randomised, and finetune(..., tests=...) also scores test flows with
them randomised or zeroed, or with the payload zeroed (perturb). finetune(..., train_copies=K)
randomises them in the training flows too (used by kaggle/litcvit_ftaug/ftaug.py).

Splits and datasets are as in kaggle/table7/table7.py (load and split are copies;
keep them the same). Writes /kaggle/working/litcvit.csv (one row per run, appended
as runs finish), encoder_<split>_<seed>.pt and
preds/<dataset>_<split>_litcvit_<seed>.npz (test rows, true class, logits).

Push with:  kaggle kernels push -p kaggle/litcvit
"""

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.flop_counter import FlopCounterMode

DATA = next(Path("/kaggle/input").rglob("classes.csv"), Path("C:/data/CNS_proj/kaggle/litcvit-flows/classes.csv")).parent  # local copy for probe.py
OUT = Path("/kaggle/working")
RUNS = ["5gad2022", "mcfp", "mcfp_strict", "ciciot2022", "tii_ssrc23", "iot23", "ustc"]
SPLITS = ["time"]  # random ran as version 2 (one split per Kaggle run)
SEEDS = [0]
MIN_STRICT = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PACKETS, ROW, HEADER = 5, 320, 80
D, RANK, BYTE_WINDOW, PACKET_WINDOW = 192, 16, 5, 3
STEM_CHANNELS, STEM_KERNEL, FFN_RATIO, PROJ_DIM = 16, 5, 2, 128
PRETRAIN = ["ustc", "ciciot2022"]
PRETRAIN_EPOCHS, PRETRAIN_BATCH, PRETRAIN_LR, TAU = 100, 128, 2e-3, 0.3
PRETRAIN_HOURS = 5.0  # stop pretraining early past this, so a run fits Kaggle's 12-hour limit (~151 s/epoch on Kaggle's GPU)
MASK, DROP_PACKET = 0.15, 0.5
FINETUNE_EPOCHS, FREEZE_EPOCHS, FINETUNE_BATCH, FINETUNE_LR = 500, 100, 64, 2e-4
WEIGHT_DECAY, WARMUP = 1e-4, 0.05
BENIGN = ("Benign", "Normal")
FORGED_ALPHA, FORGED_SHARE = 1.0, 4  # one-sided loss: weight as in light.ALPHA; forged flows from 1 in 4 of a batch's malware, for speed


# ---------------------------------------------------------------- data (as in table7.py)

def load(run):
    """Flow metadata (with its CSV row) and images [N, 40, 40] for one run."""
    name = run.removesuffix("_strict")
    meta = pd.read_csv(DATA / f"{name}.csv", keep_default_na=False)
    meta["row"] = np.arange(len(meta))
    X = np.load(DATA / f"{name}.npy")
    if run.endswith("_strict"):
        keep = meta["strict"].to_numpy() == 1
        sizes = meta[keep]["class"].value_counts()
        keep &= meta["class"].isin(sizes[sizes >= MIN_STRICT].index).to_numpy()
        meta, X = meta[keep].reset_index(drop=True), X[keep]
    return meta, X


def split(meta, kind, rng):
    """Train and test positions in meta."""
    train = np.zeros(len(meta), bool)
    ts = meta["ts_first"].to_numpy()
    groups = meta.groupby("class").indices if kind == "random" else meta.groupby(["class", "capture"]).indices
    for idx in groups.values():
        idx = rng.permutation(idx) if kind == "random" else idx[np.argsort(ts[idx], kind="stable")]
        train[idx[:int(round(0.8 * len(idx)))]] = True
    return np.flatnonzero(train), np.flatnonzero(~train)


def class_weights(y, k):
    counts = np.bincount(y, minlength=k)
    return len(y) / (k * np.maximum(counts, 1))


# ---------------------------------------------------------------- model

def sinusoid(n, d):
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    freq = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
    return pe


class FactorizedAttention(nn.Module):
    def __init__(self, d, r):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * r)
        self.out = nn.Linear(r, d)
        self.log_tau = nn.Parameter(torch.tensor(math.log(math.sqrt(r))))

    def forward(self, x, mask=None):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        att = q @ k.transpose(-2, -1) / self.log_tau.exp()
        if mask is not None:
            att = att.masked_fill(~mask, float("-inf"))
        return self.out(att.softmax(-1) @ v)


class Block(nn.Module):
    def __init__(self, d, r):
        super().__init__()
        self.norm1, self.attn, self.norm2 = nn.LayerNorm(d), FactorizedAttention(d, r), nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, FFN_RATIO * d), nn.GELU(), nn.Linear(FFN_RATIO * d, d))

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        return x + self.ffn(self.norm2(x))


class Encoder(nn.Module):
    def __init__(self, d=D, r=RANK):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, STEM_CHANNELS, STEM_KERNEL, padding=STEM_KERNEL // 2),
            nn.BatchNorm1d(STEM_CHANNELS), nn.GELU(),
            nn.Conv1d(STEM_CHANNELS, d, 1), nn.BatchNorm1d(d), nn.GELU())
        self.register_buffer("pos", sinusoid(PACKETS * ROW, d))
        self.byte_block = Block(d, r)
        self.byte_mix = nn.Conv1d(d, d, 3, padding=1, groups=d)
        self.packet_block = Block(d, r)
        self.packet_mix = nn.Conv1d(d, d, 3, padding=1, groups=d)
        self.flow_block = Block(d, r)
        i = torch.arange(PACKETS)
        self.register_buffer("band", (i[:, None] - i[None, :]).abs() <= PACKET_WINDOW // 2)

    def upper(self):
        """The part fine-tuning unfreezes: packet and flow blocks."""
        return [self.packet_block, self.packet_mix, self.flow_block]

    def packet_tokens(self, x):
        """uint8 flow images [B, 40, 40] -> packet tokens [B, 5, d]."""
        B = len(x)
        e = self.stem(x.reshape(B * PACKETS, 1, ROW).float() / 255)
        e = e.transpose(1, 2).reshape(B, PACKETS * ROW, -1) + self.pos
        e = self.byte_block(e.reshape(-1, BYTE_WINDOW, e.shape[-1])).reshape(B * PACKETS, ROW, -1)
        e = e + self.byte_mix(e.transpose(1, 2)).transpose(1, 2)
        return e.mean(1).reshape(B, PACKETS, -1)

    def from_packets(self, p):
        """Packet tokens [B, 5, d] -> flow embedding [B, d]."""
        p = self.packet_block(p, self.band)
        p = p + self.packet_mix(p.transpose(1, 2)).transpose(1, 2)
        return self.flow_block(p).mean(1)

    def forward(self, x):
        return self.from_packets(self.packet_tokens(x))


def projection_head(d):
    return nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, PROJ_DIM))


def classifier_head(d, k):
    return nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(0.2),
                         nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(0.2), nn.Linear(d // 2, k))


def model_size(d, k=11):
    """Parameters (encoder, + projection head, + classifier) and multiply-adds per flow."""
    encoder = Encoder(d).eval()
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        encoder(torch.zeros(1, 40, 40, dtype=torch.uint8))
    count = lambda m: sum(p.numel() for p in m.parameters())
    return count(encoder), count(projection_head(d)), count(classifier_head(d, k)), counter.get_total_flops() / 2


# ---------------------------------------------------------------- training

def lr_factor(step, total):
    warm = max(1, int(WARMUP * total))
    if step < warm:
        return (step + 1) / warm
    return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def balanced_indices(groups, n, gen):
    """n draws with replacement, each group (class) equally likely."""
    weights = (1.0 / torch.bincount(groups).float())[groups]
    return torch.multinomial(weights, n, replacement=True, generator=gen)


def augment(x, gen):
    """A view of uint8 images [B, 40, 40]: 15% of bytes zeroed, sometimes one of packets 2-5 zeroed."""
    x = x.reshape(len(x), PACKETS, ROW).clone()
    x[torch.rand(x.shape, generator=gen, device=x.device) < MASK] = 0
    drop = torch.rand(len(x), generator=gen, device=x.device) < DROP_PACKET
    which = torch.randint(1, PACKETS, (len(x),), generator=gen, device=x.device)
    x[drop, which[drop]] = 0
    return x


IP4_FIELDS = [1, 4, 5, 6, 7, 8, 10, 11]  # TOS, IP ID, flags and fragment offset, TTL, checksum
IP6_FIELDS = [1, 2, 3, 7]  # traffic class and flow label (past the version nibble), hop limit


def shortcut_fields(x, sequence=True):
    """Header bytes that can identify a host or a clock: uint8 images [B, 40, 40] -> mask [B, 5, 320].

    IPv4: IP4_FIELDS and options; IPv6: IP6_FIELDS; TCP: sequence and acknowledgement
    numbers (unless sequence is False), window, checksum, urgent pointer and options
    (timestamps among them); UDP: checksum; ICMP and ICMPv6: checksum and the rest of the
    8-byte header. Kept: version, lengths, protocol, TCP data offset and flags, ICMP type and
    code, and the payload (addresses and ports are already zero). A transport header with TCP
    data offset under 20 or UDP length under 8 was cut short by the capture and is left alone.
    """
    h = x.reshape(len(x), PACKETS, ROW)[..., :HEADER].long()
    pos = torch.arange(HEADER, device=x.device)
    version = h[..., :1] >> 4
    ipv4 = version == 4
    l4 = torch.where(ipv4, (h[..., :1] & 15) * 4, 40)
    proto = torch.where(ipv4, h[..., 9:10], h[..., 6:7])
    at = lambda offset: h.gather(-1, (l4 + offset).clamp(max=HEADER - 1))
    span = lambda a, b: (pos >= l4 + a) & (pos < l4 + b)
    doff = (at(12) >> 4) * 4
    fields = torch.where(ipv4, torch.isin(pos, torch.tensor(IP4_FIELDS, device=x.device)) | (pos >= 20) & (pos < l4),
                         torch.isin(pos, torch.tensor(IP6_FIELDS, device=x.device)))
    tcp = span(14, 20) | span(20, HEADER) & (pos < l4 + doff)
    fields |= (proto == 6) & (doff >= 20) & (tcp | span(4, 12) if sequence else tcp)
    fields |= (proto == 17) & ((at(4) << 8 | at(5)) >= 8) & span(6, 8)
    fields |= ((proto == 1) | (proto == 58)) & span(2, 8)
    fields &= ipv4 | (version == 6)
    return torch.cat([fields, fields.new_zeros(*fields.shape[:-1], ROW - HEADER)], -1)


def randomize(x, fields, gen):
    """x [B, 5, 320] with the masked bytes replaced by uniform random bytes."""
    noise = torch.randint(0, 256, x.shape, generator=gen, device=x.device, dtype=torch.uint8)
    return torch.where(fields, noise, x)


def augment_sa(x, gen):
    """A shortcut-aware view: augment, then the shortcut fields of every packet the view keeps set to random bytes."""
    fields = shortcut_fields(x)
    v = augment(x, gen)
    return randomize(v, fields & (v != 0).any(-1, keepdim=True), gen)


PERTURBATIONS = ("random", "zero", "nopayload", "random_nopayload")  # in this order "random" draws the bytes sa.py did


def perturb(x, how, gen):
    """Test images uint8 [B, 40, 40] with their shortcut fields set to random bytes ("random") or zeroed ("zero"),
    or their payload zeroed and the header kept ("nopayload") or with those fields randomised ("random_nopayload")."""
    if how not in PERTURBATIONS:
        raise ValueError(f"unknown perturbation {how!r}")
    fields = shortcut_fields(x)
    x = x.reshape(len(x), PACKETS, ROW)
    if how == "zero":
        return x.masked_fill(fields, 0)
    if how.endswith("nopayload"):
        x = x.clone()
        x[..., HEADER:] = 0
    return randomize(x, fields, gen) if how.startswith("random") else x


def nt_xent(z, tau):
    """NT-Xent over 2n embeddings, where rows i and i + n are two views of one flow."""
    z = F.normalize(z.float(), dim=1)
    n = len(z) // 2
    sim = z @ z.T / tau
    sim.fill_diagonal_(float("-inf"))
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(n)]).to(z.device)
    return F.cross_entropy(sim, target)


def pretrain(kind, seed, view=augment):
    """Contrastive pretraining; view(x, gen) makes one view of a batch (augment, or augment_sa)."""
    images, groups = [], []
    for name in PRETRAIN:
        meta, X = load(name)
        train, _ = split(meta, kind, np.random.default_rng(seed))
        images.append(X[train])
        groups += [f"{name}/{c}" for c in meta["class"].to_numpy()[train]]
    X = torch.from_numpy(np.concatenate(images)).to(DEVICE)
    groups = torch.from_numpy(np.unique(groups, return_inverse=True)[1]).to(DEVICE)
    print(f"pretraining on {len(X):,} flows, {int(groups.max()) + 1} classes", flush=True)

    torch.manual_seed(seed)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    encoder, head = Encoder().to(DEVICE), projection_head(D).to(DEVICE)
    opt = torch.optim.AdamW([*encoder.parameters(), *head.parameters()], PRETRAIN_LR, weight_decay=WEIGHT_DECAY)
    steps = len(X) // PRETRAIN_BATCH
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, steps * PRETRAIN_EPOCHS))
    scaler = torch.amp.GradScaler(enabled=DEVICE == "cuda")
    start = time.time()
    for epoch in range(PRETRAIN_EPOCHS):
        encoder.train()
        total = torch.zeros((), device=DEVICE)
        for b in balanced_indices(groups, steps * PRETRAIN_BATCH, gen).view(steps, PRETRAIN_BATCH):
            x = X[b]
            with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
                loss = nt_xent(head(encoder(torch.cat([view(x, gen), view(x, gen)]))), TAU)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.detach()
        hours = (time.time() - start) / 3600
        print(f"  pretrain epoch {epoch + 1}: loss {total.item() / steps:.4f}, {hours:.2f} h", flush=True)
        if hours > PRETRAIN_HOURS:
            print(f"  stopping pretraining after {epoch + 1} epochs (time limit)", flush=True)
            break
    return encoder, epoch + 1


def cached_tokens(encoder, X):
    """Packet tokens of uint8 images (numpy or on DEVICE) through the frozen stem and byte block."""
    with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
        return torch.cat([encoder.packet_tokens(torch.as_tensor(X[i:i + 1024]).to(DEVICE)).float()
                          for i in range(0, len(X), 1024)])


def forged_tokens(encoder, bad, good, copies, gen):
    """Tokens [K, malware flows, 5, d] in half precision: uint8 malware flows [N, 5, 320], in K
    copies, each flow with the shortcut fields (sequence numbers apart) of a random benign flow."""
    in_chunks = lambda x: torch.cat([shortcut_fields(x[i:i + 1024], sequence=False) for i in range(0, len(x), 1024)])
    fields_bad, fields_good = in_chunks(bad), in_chunks(good)
    out = []
    for _ in range(copies):
        donors = torch.randint(len(good), (len(bad),), generator=gen, device=bad.device)
        flows = torch.cat([torch.where(fields_bad[i:i + 1024] & fields_good[donors[i:i + 1024]], good[donors[i:i + 1024]], bad[i:i + 1024])
                           for i in range(0, len(bad), 1024)])
        out.append(cached_tokens(encoder, flows).half())
    return torch.stack(out)


def finetune(pretrained, run, kind, seed, tests=(), train_copies=0, attacks=None, save=None, forged_copies=0):
    """Fine-tune on one run; logits on the test flows come back as {"clean": ..., how: ...} for each
    perturbation in tests (see perturb), applied to the test images only.
    With train_copies = K the classifier never sees a clean training flow: each step takes every
    flow from one of K copies with its shortcut fields set to random bytes (fresh in each copy).
    With forged_copies = K it trains the one-sided loss as well (see kaggle/litcvit/light.py): plain
    cross-entropy, plus cross-entropy on a share of the batch's malware flows taken from K copies
    that carry benign flows' header fields, with their own label.
    attacks: {name: uint8 images of the test flows, in test order}, scored the same way.
    save: path for the fine-tuned encoder and classifier, so later tests needn't retrain."""
    meta, X = load(run)
    classes = sorted(meta["class"].unique())
    y_all = meta["class"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    train, test = split(meta, kind, np.random.default_rng(seed))

    encoder = copy.deepcopy(pretrained).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    P = cached_tokens(encoder, X)
    if forged_copies:  # forged [K, malware training flows, 5, d]; column: position in X -> column
        forge_gen = torch.Generator(device=DEVICE).manual_seed(seed + 2)  # own generator: batches match K = 0
        is_bad = torch.from_numpy(~np.isin(np.array(classes)[y_all], BENIGN)).to(DEVICE)
        images = torch.from_numpy(X[train]).to(DEVICE).reshape(len(train), PACKETS, ROW)
        bad_train = is_bad[torch.from_numpy(train).to(DEVICE)]
        forged = forged_tokens(encoder, images[bad_train], images[~bad_train], forged_copies, forge_gen)
        column = torch.zeros(len(X), dtype=torch.long, device=DEVICE)
        column[torch.from_numpy(train).to(DEVICE)[bad_train]] = torch.arange(int(bad_train.sum()), device=DEVICE)
        del images
    if train_copies:  # copies [K, training flows, 5, d] in half precision; slot: position in X -> column
        copy_gen = torch.Generator(device=DEVICE).manual_seed(seed + 1)  # own generator: batches match K = 0
        images = torch.from_numpy(X[train]).to(DEVICE)
        copies = torch.stack([cached_tokens(encoder, torch.cat([perturb(images[i:i + 1024], "random", copy_gen)
                                                               for i in range(0, len(images), 1024)])).half()
                              for _ in range(train_copies)])
        slot = torch.zeros(len(X), dtype=torch.long, device=DEVICE)
        slot[torch.from_numpy(train).to(DEVICE)] = torch.arange(len(train), device=DEVICE)
        del images

    torch.manual_seed(seed)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    k = len(classes)
    head = classifier_head(D, k).to(DEVICE)
    upper = [p for m in encoder.upper() for p in m.parameters()]
    opt = torch.optim.AdamW([*upper, *head.parameters()], FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    steps = max(1, len(train) // FINETUNE_BATCH)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, steps * FINETUNE_EPOCHS))
    y = torch.from_numpy(y_all).to(DEVICE)
    tr = torch.from_numpy(train).to(DEVICE)
    weight = torch.tensor(class_weights(y_all[train], k), dtype=torch.float32, device=DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    start = time.time()
    for epoch in range(FINETUNE_EPOCHS):
        if epoch == FREEZE_EPOCHS:
            for p in upper:
                p.requires_grad_(True)
        head.train()
        total = torch.zeros((), device=DEVICE)
        for b in tr[balanced_indices(y[tr], steps * FINETUNE_BATCH, gen)].view(steps, -1):
            p = (copies[torch.randint(train_copies, (len(b),), generator=copy_gen, device=DEVICE), slot[b]].float()
                 if train_copies else P[b])
            loss = loss_fn(head(encoder.from_packets(p)), y[b])
            if forged_copies:
                m = b[is_bad[b]]
                m = m[:max(1, len(m) // FORGED_SHARE)]
                f = forged[torch.randint(forged_copies, (len(m),), generator=forge_gen, device=DEVICE), column[m]].float()
                loss = loss + FORGED_ALPHA / k * loss_fn(head(encoder.from_packets(f)), y[m])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.detach()
        if (epoch + 1) % 50 == 0 or epoch + 1 in (1, FINETUNE_EPOCHS):
            print(f"  {run} fine-tune epoch {epoch + 1}: loss {total.item() / steps:.4f}, "
                  f"{time.time() - start:.0f}s", flush=True)

    head.eval()
    tokens = {"clean": P[torch.from_numpy(test).to(DEVICE)]}
    test_gen = torch.Generator(device=DEVICE).manual_seed(seed)
    for how in tests:
        images = torch.from_numpy(X[test]).to(DEVICE)
        tokens[how] = cached_tokens(encoder, torch.cat([perturb(images[i:i + 1024], how, test_gen)
                                                        for i in range(0, len(images), 1024)]))
    for name, images in (attacks or {}).items():
        tokens[name] = cached_tokens(encoder, images)
    if save:
        torch.save({"encoder": encoder.state_dict(), "head": head.state_dict(), "classes": classes}, save)
    with torch.no_grad():
        logits = {how: torch.cat([head(encoder.from_packets(t[i:i + 4096])) for i in range(0, len(t), 4096)]).cpu().numpy()
                  for how, t in tokens.items()}
    return meta["row"].to_numpy()[test], y_all[test], logits, classes, len(train)


def main():
    (OUT / "preds").mkdir(exist_ok=True)
    print(f"data in {DATA}, device {DEVICE}", flush=True)
    for d in (128, 192):
        enc, proj, cls, macs = model_size(d)
        print(f"d = {d}: encoder {enc:,} parameters (+{proj:,} projection head, +{cls:,} classifier "
              f"for 11 classes), {macs / 1e6:.1f}M multiply-adds per flow", flush=True)

    results = []
    for kind in SPLITS:
        for seed in SEEDS:
            start = time.time()
            encoder, epochs = pretrain(kind, seed)
            torch.save(encoder.state_dict(), OUT / f"encoder_{kind}_{seed}.pt")
            pretrain_hours = round((time.time() - start) / 3600, 2)
            for run in RUNS:
                start = time.time()
                rows, y, logits, classes, n_train = finetune(encoder, run, kind, seed)
                logits = logits["clean"]
                pred = logits.argmax(1)
                recall = recall_score(y, pred, labels=range(len(classes)), average=None, zero_division=0)
                row = {"dataset": run, "split": kind, "model": "litcvit", "seed": seed,
                       "train": n_train, "test": len(y),
                       "accuracy": round(100 * accuracy_score(y, pred), 2),
                       "macro_f1": round(100 * f1_score(y, pred, average="macro"), 2),
                       "weighted_f1": round(100 * f1_score(y, pred, average="weighted"), 2),
                       "pretrain_epochs": epochs, "pretrain_hours": pretrain_hours,
                       "seconds": round(time.time() - start),
                       "recall": json.dumps({c: round(100 * r, 1) for c, r in zip(classes, recall)})}
                results.append(row)
                pd.DataFrame(results).to_csv(OUT / "litcvit.csv", index=False)
                np.savez_compressed(OUT / "preds" / f"{run}_{kind}_litcvit_{seed}.npz",
                                    row=rows, y=y, logits=logits.astype(np.float32), classes=np.array(classes))
                print(f"{run:12s} {kind:6s} seed {seed}: acc {row['accuracy']:6.2f}  macro-F1 {row['macro_f1']:6.2f}  "
                      f"weighted-F1 {row['weighted_f1']:6.2f}  ({row['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
