"""Lightweight vision models on flow images: one training protocol, several losses, the same tests.

Models (FlowNet wraps each): small convolutional and hybrid models from timm and torchvision, fed
the 40 x 40 image upsampled to 80 x 80 by repeating bytes, so their stride-2 stems start at one
token per byte; and VisionMamba, written here in plain PyTorch since mamba_ssm isn't on Kaggle.

Training follows LitCVit's fine-tuning (class-weighted cross-entropy, class-balanced batches, AdamW,
weight decay 1e-4, cosine with 5% warmup), with no pretraining, batch 256, learning rate 1e-3 and
EPOCHS epochs. Losses:
  ce      cross-entropy on flows as recorded
  mask    the shortcut fields (litcvit.shortcut_fields) zeroed in training and in every test, so
          they can't be read at all: the plain defence every other loss must beat
  rand    cross-entropy on flows with the shortcut fields set to fresh random bytes
  swap    cross-entropy on counterfactual flows: a flow keeps its content and label but takes the
          real header fields of a flow of another class in the batch (see counterfactual)
  both    cross-entropy on flows as recorded and on those counterfactual flows, for ablation
  cfnce   the same, plus counterfactual_nce; cfnce1 counts the donor as an ordinary negative
  oneside headers may convict but never acquit: cross-entropy on flows as recorded, plus on
          malware flows carrying a benign flow's real header fields, with their own label. An
          attacker can forge a benign header and has no reason to forge a malicious one, so
          fingerprints stay usable as evidence of malware (which mask gives up) while a benign
          fingerprint stops being evidence of innocence (which ce relies on)

Tests (test_sets): flows as recorded, the litcvit.perturb conditions, and the header mimicry attack
of mimic.py (donors from the same network on every run, and from another dataset on the runs in
OTHER), all on the same test flows. mimic_all adds new initial sequence numbers to the attack,
and isn gives every flow new ones and changes nothing else.
"""

import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.flop_counter import FlopCounterMode

import litcvit as L
import mimic as M

BENIGN = {"Benign", "Normal"}
EPOCHS, BATCH, LR = 30, 256, 1e-3
TAU, BETA, LAM = 0.2, 4.0, 0.5  # counterfactual_nce: temperature, donor weight, its share of the loss
ALPHA = 1.0  # oneside: total weight of the forged flows, as a multiple of the benign class's weight
OTHER = {"mcfp_strict": "ustc", "ustc": "mcfp"}  # runs under attack -> the dataset lending mimic_stack_other's donors
FIELDS = {"mimic_ttl": ("ttl",), "mimic_window": ("window",), "mimic_ttl_window": ("ttl", "window"), "mimic_stack": M.STACK}
TIMM = {"mobilenetv2": "mobilenetv2_050", "mobilenetv3": "mobilenetv3_small_100", "mobilenetv4": "mobilenetv4_conv_small",
        "ghostnet": "ghostnet_050", "mobilevit": "mobilevit_xxs", "edgenext": "edgenext_xx_small", "efficientvit": "efficientvit_b0"}
MODELS = [*TIMM, "shufflenet", "vim"]


# ---------------------------------------------------------------- Vision Mamba

def _scan(decay: torch.Tensor, drive: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    decays, drives, cs = decay.unbind(1), drive.unbind(1), c.unbind(1)  # slicing instead makes backward quadratic in length
    h = torch.zeros_like(decays[0])
    out = []
    for t in range(len(decays)):
        h = decays[t] * h + drives[t]
        out.append((h * cs[t].unsqueeze(1)).sum(-1))
    return torch.stack(out, 1)


try:
    _scan = torch.jit.script(_scan)  # the loop runs several times faster outside Python
except Exception:
    pass


class SelectiveSSM(nn.Module):
    """Mamba's selective state space layer (Gu and Dao, 2023) over a token sequence."""

    def __init__(self, d, state=4, rank=8):
        super().__init__()
        self.conv = nn.Conv1d(d, d, 4, padding=3, groups=d)
        self.params = nn.Linear(d, rank + 2 * state)
        self.dt = nn.Linear(rank, d)
        self.a_log = nn.Parameter(torch.log(torch.arange(1.0, state + 1)).repeat(d, 1))
        self.d = nn.Parameter(torch.ones(d))
        self.split = [rank, state, state]

    def forward(self, u):
        """[B, L, d] -> [B, L, d]; a causal scan from the first token to the last."""
        n = u.shape[1]
        u = F.silu(self.conv(u.transpose(1, 2))[..., :n].transpose(1, 2))
        delta, b, c = self.params(u).split(self.split, -1)
        delta = F.softplus(self.dt(delta))
        decay = torch.exp(-delta[..., None] * self.a_log.exp())
        drive = (delta * u)[..., None] * b[:, :, None]
        return _scan(decay, drive, c) + u * self.d


class BiMambaBlock(nn.Module):
    """Vision Mamba's block (Zhu et al., 2024): one scan forward and one backward over the tokens."""

    def __init__(self, d, expand=1, state=4):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.up = nn.Linear(d, expand * d)
        self.forward_scan, self.backward_scan = SelectiveSSM(expand * d, state), SelectiveSSM(expand * d, state)
        self.down = nn.Linear(expand * d, d)

    def forward(self, x):
        u = self.up(self.norm(x))
        return x + self.down(self.forward_scan(u) + self.backward_scan(u.flip(1)).flip(1))


class VisionMamba(nn.Module):
    """A small Vision Mamba on the 40 x 40 image: 5 x 5 patches (64 tokens, in packet order), mean pooled."""

    def __init__(self, d=96, depth=6, patch=5):
        super().__init__()
        self.patch = nn.Conv2d(1, d, patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, (40 // patch) ** 2, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.Sequential(*[BiMambaBlock(d) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        t = self.patch(x).flatten(2).transpose(1, 2) + self.pos
        return self.norm(self.blocks(t)).mean(1)


# ---------------------------------------------------------------- models

def backbone(name):
    """(module taking floats [B, 1, H, W] and returning [B, d], upsampling factor)."""
    if name == "vim":
        return VisionMamba(), 1
    if name == "shufflenet":
        import torchvision
        m = torchvision.models.shufflenet_v2_x0_5()
        m.conv1[0] = nn.Conv2d(1, m.conv1[0].out_channels, 3, 2, 1, bias=False)
        m.fc = nn.Identity()
        return m, 2
    import timm
    return timm.create_model(TIMM[name], pretrained=False, in_chans=1, num_classes=0), 2


class FlowNet(nn.Module):
    def __init__(self, name, k, masked=False):
        super().__init__()
        self.body, self.scale = backbone(name)
        self.masked = masked  # the mask loss: shortcut fields zeroed before every forward pass
        self.body.eval()
        with torch.no_grad():
            d = self.body(torch.zeros(2, 1, 40 * self.scale, 40 * self.scale)).shape[1]
        self.body.train()
        self.head = nn.Linear(d, k)
        self.projector = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 128))

    def embed(self, x):
        if self.masked:
            x = L.perturb(x, "zero", None)
        x = x.reshape(len(x), 1, 40, 40).float() / 255
        return self.body(F.interpolate(x, scale_factor=self.scale, mode="nearest") if self.scale > 1 else x)

    def forward(self, x):
        return self.head(self.embed(x))


def gate(full, content, benign):
    """A header may convict but never acquit. Benign classes score no higher than in content, a model that never
    reads the shortcut fields, and malware classes no lower. So a flow content calls malware can't be called
    Benign, whatever full makes of its fields: for its top malware class m and any benign class b,
    gate[m] >= content[m] > content[b] >= gate[b]."""
    return torch.where(benign, torch.minimum(full, content), torch.maximum(full, content))


class GatedNet(FlowNet):
    """One network reads each flow twice, as recorded and with its shortcut fields zeroed, and gate joins the answers."""

    def __init__(self, name, k, benign):
        super().__init__(name, k)
        self.register_buffer("benign", torch.as_tensor(np.asarray(benign)))

    def views(self, x, fields=None):
        """Logits of the flows as recorded and of the same flows zeroed, from one pass, so batch norm sees both kinds together."""
        x = x.reshape(len(x), L.PACKETS, L.ROW)
        zeroed = L.perturb(x, "zero", None) if fields is None else x.masked_fill(fields, 0)
        return self.head(self.embed(torch.cat([x, zeroed]))).chunk(2)

    def forward(self, x):
        return gate(*self.views(x), self.benign)


def size_of(net):
    """Parameters (without the projector, used only in training) and multiply-adds per flow."""
    params = sum(p.numel() for m in (net.body, net.head) for p in m.parameters())
    device = next(net.parameters()).device
    net.eval()
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        net(torch.zeros(1, 40, 40, dtype=torch.uint8, device=device))
    return params, counter.get_total_flops() // 2


# ---------------------------------------------------------------- losses

def stack_fields(x, chunk=4096):
    """Shortcut fields without sequence and acknowledgement numbers, whose differences carry the flow's behaviour."""
    return torch.cat([L.shortcut_fields(x[i:i + chunk], sequence=False) for i in range(0, len(x), chunk)])


def all_fields(x, chunk=4096):
    return torch.cat([L.shortcut_fields(x[i:i + chunk]) for i in range(0, len(x), chunk)])


def donors_of(y, gen):
    """For each flow of a batch, the position of a flow of another class (a few same-class ones may remain in tiny classes)."""
    donors = torch.randperm(len(y), generator=gen, device=y.device)
    for _ in range(8):
        same = y[donors] == y
        if not same.any():
            break
        donors[same] = torch.randint(len(y), (int(same.sum()),), generator=gen, device=y.device)
    return donors


def counterfactual(x, fields, donors):
    """x [B, 5, 320] with the donors' bytes wherever both flows have a shortcut field: real values, wrong flow."""
    return torch.where(fields & fields[donors], x[donors], x)


def counterfactual_nce(z_c, z_x, y, donors, tau=TAU, beta=BETA):
    """Content over header: a counterfactual flow must sit nearer the flow it took its content from
    than the flow it took its header fields from, and the same seen from the original flow.

    z_c, z_x: normalised embeddings of the counterfactual flows and of the originals. For anchor c_i
    the positive is x_i; negatives are flows of other classes, and the donor x_d(i), which shares
    its header fields with the anchor, counts beta times. An encoder that reads those fields scores
    the donor high and pays for it directly, which no random-byte view can do.
    """
    sim = z_c @ z_x.T / tau
    weight = (y[None] != y[:, None]).float()
    weight[torch.arange(len(y), device=y.device), donors] *= beta
    log_weight = weight.log()  # -inf where a flow of the same class is no negative
    pos = sim.diagonal()
    from_c = torch.logaddexp(pos, torch.logsumexp(sim + log_weight, 1)) - pos
    from_x = torch.logaddexp(pos, torch.logsumexp(sim + log_weight, 0)) - pos
    return (from_c.mean() + from_x.mean()) / 2


def forge(x, fields, y, benign, gen):
    """The batch's malware flows, each with the header fields of a random benign flow of the batch, and their positions.

    benign: bool per class. Empty if the batch lacks either kind of flow.
    """
    is_benign = benign[y]
    good, bad = is_benign.nonzero().squeeze(1), (~is_benign).nonzero().squeeze(1)
    if not len(good) or not len(bad):
        return x[:0], bad[:0]
    donors = good[torch.randint(len(good), (len(bad),), generator=gen, device=y.device)]
    return torch.where(fields[bad] & fields[donors], x[donors], x[bad]), bad


def step_loss(net, loss, x, y, fields, weight, gen, benign=None):
    """One batch's loss. fields: the batch's mask for this loss (None for ce and mask)."""
    if loss in ("ce", "mask"):
        return F.cross_entropy(net(x), y, weight=weight)
    if loss in ("gated", "views"):
        full, content = net.views(x, fields)
        first = gate(full, content, net.benign) if loss == "gated" else full  # gated alone lets full convict and leaves content idle
        return F.cross_entropy(first, y, weight=weight) + F.cross_entropy(content, y, weight=weight)
    if loss == "rand":
        return F.cross_entropy(net(L.randomize(x, fields, gen)), y, weight=weight)
    if loss == "oneside":
        forged, bad = forge(x, fields, y, benign, gen)
        logits = net(torch.cat([x, forged]))  # one pass, so batch norm sees both kinds together
        value = F.cross_entropy(logits[:len(x)], y, weight=weight)
        if len(bad):  # batches are class-balanced, so the benign class carries 1/k of the first term
            value = value + ALPHA / net.head.out_features * F.cross_entropy(logits[len(x):], y[bad], weight=weight)
        return value
    donors = donors_of(y, gen)
    c = counterfactual(x, fields, donors)
    if loss == "swap":
        return F.cross_entropy(net(c), y, weight=weight)
    if loss in ("both", "cfnce", "cfnce1"):
        e_x, e_c = net.embed(torch.cat([x, c])).chunk(2)  # one pass, so batch norm sees both kinds together
        ce = (F.cross_entropy(net.head(e_x), y, weight=weight) + F.cross_entropy(net.head(e_c), y, weight=weight)) / 2
        if loss == "both":
            return ce
        z_x, z_c = net.projector(e_x), net.projector(e_c)
        with torch.autocast(L.DEVICE, enabled=False):
            return ce + LAM * counterfactual_nce(F.normalize(z_c.float(), dim=1), F.normalize(z_x.float(), dim=1), y, donors,
                                                 beta=1.0 if loss == "cfnce1" else BETA)
    raise ValueError(f"unknown loss {loss!r}")


# ---------------------------------------------------------------- training and tests

def fit(net, images, y, loss, seed, epochs=EPOCHS, batch=BATCH, lr=LR, amp=True, log=None, benign=None, min_steps=0):
    """Train net in place on uint8 images [N, ...] with labels y (numpy). Returns the last epoch's mean loss.

    benign: bool per class, needed by the oneside loss. min_steps: more epochs if these would give fewer steps.
    """
    device = next(net.parameters()).device
    x_all = torch.from_numpy(np.asarray(images).reshape(len(images), L.PACKETS, L.ROW)).to(device)
    y_all = torch.from_numpy(np.asarray(y, dtype=np.int64).copy()).to(device)
    fields_all = {"rand": all_fields, "gated": all_fields, "views": all_fields, "swap": stack_fields, "both": stack_fields,
                  "cfnce": stack_fields, "cfnce1": stack_fields, "oneside": stack_fields}.get(loss, lambda x: None)(x_all)
    benign = None if benign is None else torch.as_tensor(np.asarray(benign), device=device)
    k = net.head.out_features
    weight = torch.tensor(L.class_weights(np.asarray(y), k), dtype=torch.float32, device=device)
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=L.WEIGHT_DECAY)
    steps = max(1, len(x_all) // batch)
    epochs = max(epochs, -(-min_steps // steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: L.lr_factor(s, steps * epochs))
    amp = amp and L.DEVICE == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp)
    start = time.time()
    for epoch in range(epochs):
        net.train()
        total = torch.zeros((), device=device)
        for b in L.balanced_indices(y_all, steps * batch, gen).view(steps, -1):
            fields = None if fields_all is None else fields_all[b]
            with torch.autocast(L.DEVICE, dtype=torch.float16, enabled=amp):
                value = step_loss(net, loss, x_all[b], y_all[b], fields, weight, gen, benign)
            opt.zero_grad(set_to_none=True)
            scaler.scale(value).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += value.detach()
        if log and (epoch + 1) % 10 == 0:
            log(f"    epoch {epoch + 1}: loss {total.item() / steps:.4f}, {time.time() - start:.0f}s")
    return total.item() / steps


def logits_of(net, images, batch=1024):
    device = next(net.parameters()).device
    net.eval()
    with torch.no_grad():
        return torch.cat([net(torch.as_tensor(np.asarray(images[i:i + batch])).to(device)).float().cpu()
                          for i in range(0, len(images), batch)]).numpy()


def test_sets(run, kind, seed, mimicry=True):
    """Everything fixed for one run and split: training flows, and the test flows under every condition."""
    meta, X = L.load(run)
    classes = sorted(meta["class"].unique())
    y = meta["class"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    train, test = L.split(meta, kind, np.random.default_rng(seed))
    images = X[test]
    conditions = {"clean": images}
    gen = torch.Generator().manual_seed(seed)
    tensor = torch.from_numpy(images)
    for how in L.PERTURBATIONS:
        conditions[how] = torch.cat([L.perturb(tensor[i:i + 4096], how, gen) for i in range(0, len(tensor), 4096)]).numpy()
    attacked = {}
    if mimicry:
        benign = np.isin(np.array(classes)[y[test]], list(BENIGN))
        rng = np.random.default_rng(seed)

        def donated(donors):
            chosen = M.choose_donors(images, donors, rng)
            chosen[benign] = -1
            return chosen

        same = donated(images[benign])
        for name, fields in FIELDS.items():
            conditions[name] = M.apply_mimicry(images, images[benign], same, fields)
            attacked[name] = same >= 0
        conditions["mimic_all"] = M.shift_sequence(conditions["mimic_stack"], rng, only=same >= 0)
        attacked["mimic_all"] = same >= 0
        conditions["isn"] = M.shift_sequence(images, rng)  # every flow: a new connection's numbers are arbitrary anyway
        if run in OTHER:
            other_meta, other_X = L.load(OTHER[run])
            other = other_X[other_meta["class"].isin(BENIGN).to_numpy()]
            far = donated(other)
            conditions["mimic_stack_other"] = M.apply_mimicry(images, other, far, M.STACK)
            attacked["mimic_stack_other"] = far >= 0
    return {"X_train": X[train], "y_train": y[train], "conditions": conditions, "y_test": y[test], "classes": classes,
            "rows": meta["row"].to_numpy()[test], "attacked": attacked}


def score(data, name, z):
    """Scores for one condition's logits; evasion only for attack conditions."""
    y, classes = data["y_test"], data["classes"]
    pred = z.argmax(1)
    recall = recall_score(y, pred, labels=range(len(classes)), average=None, zero_division=0)
    row = {"condition": name, "accuracy": round(100 * accuracy_score(y, pred), 2),
           "macro_f1": round(100 * f1_score(y, pred, average="macro"), 2),
           "recall": json.dumps({c: round(100 * r, 1) for c, r in zip(classes, recall)})}
    if name in data["attacked"]:
        benign = np.isin(classes, list(BENIGN))
        m = data["attacked"][name]
        row["attacked"] = int(m.sum())
        row["evaded"] = float("nan")
        if m.any():
            clean = data["clean_pred"][m]
            detected = ~benign[clean]
            row["detected_clean"] = round(100 * detected.mean(), 1)
            if detected.any():
                row["evaded"] = round(100 * benign[pred[m]][detected].mean(), 1)
    return row


def report(data, model, loss, seed, params, macs, seconds, logits, out):
    """Rows for one trained model's logits per condition, which are also saved to the path out."""
    data["clean_pred"] = logits["clean"].argmax(1)
    rows = [{"model": model, "loss": loss, "seed": seed, "params": params, "macs": macs, "train": len(data["y_train"]),
             "test": len(data["y_test"]), "seconds": seconds, **score(data, name, z)} for name, z in logits.items()]
    np.savez_compressed(out, row=data["rows"], y=data["y_test"], classes=np.array(data["classes"]),
                        **{"attacked_" + n: m for n, m in data["attacked"].items()},
                        **{c: z.astype(np.float32) for c, z in logits.items()})
    return rows


def run_one(data, model, loss, seed, out, log=print, epochs=EPOCHS, min_steps=0):
    """Train one model with one loss on one run and split; returns rows, and saves logits to the path out."""
    is_benign = np.isin(data["classes"], list(BENIGN))
    net = (GatedNet(model, len(data["classes"]), is_benign) if loss in ("gated", "views")
           else FlowNet(model, len(data["classes"]), masked=loss == "mask")).to(L.DEVICE)
    params, macs = size_of(net)
    start = time.time()
    fit(net, data["X_train"], data["y_train"], loss, seed, epochs=epochs, amp=model != "vim", log=log,
        benign=is_benign, min_steps=min_steps)
    seconds = round(time.time() - start)
    torch.save(net.state_dict(), out.with_suffix(".pt"))  # so later attacks needn't retrain
    logits = {name: logits_of(net, images) for name, images in data["conditions"].items()}
    return report(data, model, loss, seed, params, macs, seconds, logits, out)
