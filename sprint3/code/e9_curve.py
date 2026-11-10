"""E9: operating curve (F's benign threshold) for the gate versus masking alone, alerts per day, and the gate's cost.

Curve, per setting, from the Kaggle logits (no new inference): F says Benign iff its benign probability mass >= t.
  gate_t     F_t and B (B at its argmax)          gateC_t   F_t, B and C_A
  mask_t     B alone with the same threshold on its own benign mass
Axes: false alarms (1 - Benign recall) and unconditional attack success under (b) and (g) on the sample.
Alerts per day assume 100,000 flows a day (an assumption) at 1% and 0.1% malware.

Cost: parameters and multiply-adds (forward hooks on Conv2d and Linear), CPU latency per flow at batch 1 and 256
(12 threads, median of 5), for F, F+B and F+B+C, on 5GAD seed 0.

  python sprint3/code/e9_curve.py   -> sprint3/results/e9_curve.(json|csv), e9_cost.(json|csv)
"""

import time

import numpy as np
import torch

import s3

C, L = s3.C, s3.L
GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999)
FLOWS_PER_DAY, PREVALENCE = 100_000, (0.01, 0.001)


def mass(z, benign):
    z = np.asarray(z, np.float64)
    p = np.exp(z - z.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return p[..., benign].sum(-1)


def curve():
    rows = []
    for split, seed in s3.SETTINGS:
        for run in s3.RUNS:
            s = C.suite(run, split, seed)
            y, benign, sample = s["y"], s["benign"], s["sample"]
            malware = ~benign[y]
            F, B, CA = (s3.kaggle(run, split, seed, n) for n in ("ce", "mask", "C_A"))
            reach = {c: (s[c] != s["clean"][sample][None]).any((2, 3)) for c in ("b", "g")}
            Bc = {c: s3.benign_of(B[c], benign) for c in ("clean", "b", "g")}
            Cc = {c: s3.benign_of(CA[c], benign) for c in ("clean", "b", "g")}
            for t in GRID:
                Ft = {c: mass(F[c], benign) >= t for c in ("clean", "b", "g")}
                Bt = {c: mass(B[c], benign) >= t for c in ("clean", "b", "g")}
                for meth, cl in (("gate_t", {c: Ft[c] & Bc[c] for c in Ft}), ("gateC_t", {c: Ft[c] & Bc[c] & Cc[c] for c in Ft}),
                                 ("mask_t", Bt), ("F_t", Ft)):
                    far = 100 * cl["clean"][~malware].mean()
                    det = 100 * (~cl["clean"])[malware].mean()
                    row = {"run": run, "split": split, "seed": seed, "method": meth, "t": t,
                           "false_alarm": round(100 - far, 3), "detected": round(det, 3)}
                    for c in ("b", "g"):
                        row[f"uncond_{c}"] = s3.success(cl["clean"][sample], cl[c], reach[c])["uncond"]
                    for pi in PREVALENCE:
                        row[f"false_alerts_day_{pi}"] = round(FLOWS_PER_DAY * (1 - pi) * (100 - far) / 100, 1)
                        row[f"true_alerts_day_{pi}"] = round(FLOWS_PER_DAY * pi * det / 100, 1)
                    rows.append(row)
            print(run, split, seed, flush=True)
    s3.save(rows, "e9_curve")


def count_macs(net):
    macs, hooks = [0], []

    def hook(m, i, o):
        if isinstance(m, torch.nn.Conv2d):
            macs[0] += o.numel() // o.shape[0] * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, torch.nn.Linear):
            macs[0] += m.in_features * m.out_features
    for m in net.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            hooks.append(m.register_forward_hook(hook))
    with torch.no_grad():
        net(torch.zeros(1, L.PACKETS * L.ROW, dtype=torch.uint8))
    for h in hooks:
        h.remove()
    return macs[0]


def cost(run="5gad2022"):
    s = C.suite(run, "random", 0)
    k = len(s["classes"])
    nets = {}
    for loss, masked in (("ce", False), ("mask", True)):
        net = s3.T.FlowNet("mobilenetv2", k, masked=masked)
        net.load_state_dict(torch.load(C.existing(run, "random", 0, loss), map_location="cpu", weights_only=True))
        nets[loss] = net.eval()
    Cn = s3.c_model(s, run, "random", 0, "A")
    params = {n: sum(p.numel() for n2, p in net.named_parameters() if not n2.startswith("projector")) for n, net in nets.items()}
    macs = {n: count_macs(net) for n, net in nets.items()}
    X = s["clean"]
    fF = lambda x: s3.T.logits_of(nets["ce"], x, batch=len(x))
    fB = lambda x: s3.T.logits_of(nets["mask"], x, batch=len(x))
    methods = {"F": [fF], "F+B": [fF, fB], "F+B+C": [fF, fB, Cn]}
    rows = []
    for meth, fns in methods.items():
        for batch in (1, 256):
            xs = X[:batch]
            times = []
            for rep in range(6):
                t0 = time.perf_counter()
                for _ in range(20 if batch == 1 else 2):
                    for fn in fns:
                        fn(xs)
                times.append((time.perf_counter() - t0) / (20 if batch == 1 else 2) / batch)
            rows.append({"method": meth, "batch": batch, "ms_per_flow": round(1000 * float(np.median(times[1:])), 4),
                         "params": params["ce"] + (params["mask"] if "B" in meth else 0),
                         "macs": macs["ce"] + (macs["mask"] if "B" in meth else 0),
                         "c_trees": 200 * k if "C" in meth else 0, "threads": torch.get_num_threads()})
    for r in rows:
        ref = next(x for x in rows if x["method"] == "F" and x["batch"] == r["batch"])
        r["latency_ratio"] = round(r["ms_per_flow"] / ref["ms_per_flow"], 2)
        r["macs_ratio"] = round(r["macs"] / ref["macs"], 2)
        r["params_ratio"] = round(r["params"] / ref["params"], 2)
    s3.save(rows, "e9_cost")
    print(rows)


if __name__ == "__main__":
    curve()
    cost()
