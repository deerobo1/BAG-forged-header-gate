"""Scores of the fail-closed gate, from saved logits: no training, no GPU.

The content model (the mask loss) never reads the shortcut fields, so its answer on a flow is the
same whatever those fields hold. Gating another model with it (light.gate) gives a guarantee that
holds against any attacker who changes only those fields, however adaptive:
  certified  of the malware flows, the share the content model calls malicious: these stay detected
The guarantee covers detection only: such an attacker can still change which malware family is named.
It doesn't cover payload, packet lengths or added packets.

From kaggle/litcvit_gate and kaggle/litcvit_seeds, once downloaded: the bar plan.txt set for the gate on the time
split and on new seeds, and for one network (light.GatedNet) replacing the two. A GatedNet's answer on zeroed flows
is its content answer, since both of its views of them are the same.

Each row is a base model with no gate (none), or gated by the content model as a decision (decision:
the base model's answer, unless it says Benign and the content model doesn't) or on the logits
(logits: light.gate). Columns: macro-F1, Benign recall and malware detection on clean flows, and
macro-F1 and evaded under an attack (evaded: of the flows detected as recorded, the share called
Benign). The curve holds the content model to a margin, so it overrules the base model only where
its malware score beats its Benign score by more than the margin: fewer false alarms, a smaller
certified share. Margins come from half the test flows and are scored on the other half.

  python kaggle/litcvit/gate_report.py

Writes results/litcvit_gate.csv, litcvit_gate_wrapped.csv and litcvit_gate_curve.csv, and from
those jobs litcvit_gate_time.csv, litcvit_gate_seeds.csv and litcvit_gate_single.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent.parent
KAGGLE = Path("C:/data/CNS_proj/kaggle")
LOSS, EVASION = KAGGLE / "litcvit_output_loss" / "preds", KAGGLE / "litcvit_output_evasion" / "preds"
GATE_JOB, SEEDS_JOB = KAGGLE / "litcvit_output_gate" / "preds", KAGGLE / "litcvit_output_seeds" / "preds"
SMALL =("5gad2022", "mcfp", "mcfp_strict")
RUNS =["5gad2022", "mcfp", "mcfp_strict", "ciciot2022", "tii_ssrc23", "iot23", "ustc"]
MODELS = ("mobilenetv2", "shufflenet")
BASES = ("ce", "cfnce")
ATTACKS = ("mimic_ttl_window", "mimic_stack", "mimic_stack_other", "random")
BENIGN = {"Benign", "Normal"}
COSTS = (0.5, 1, 2, 5)  # false alarms the curve may add over the base model, in points of Benign recall


def gate_decision(base, content, benign):
    a, b = base.argmax(1), content.argmax(1)
    return np.where(benign[a] & ~benign[b], b, a)


def gate_logits(base, content, benign):
    return np.where(benign, np.minimum(base, content), np.maximum(base, content)).argmax(1)


def row_of(y, benign, clean, attacked, masks):
    """Scores of one method given its answers on clean flows and under each attack."""
    malware = ~benign[y]
    called = ~benign[clean]
    row = {"clean": round(100 * f1_score(y, clean, average="macro"), 1), "benign_recall": round(100 * (~called)[~malware].mean(), 1),
           "detected": round(100 * called[malware].mean(), 1)}
    for name, pred in attacked.items():
        m = masks[name] & malware & called
        row[name] = round(100 * f1_score(y, pred, average="macro"), 1)
        row["evaded_" + name] = round(100 * benign[pred[m]].mean(), 1) if m.any() else float("nan")
    return row


def scores(base, content, y, benign, masks, attacks):
    """Rows of a base model's logits per condition, with no gate and with each gate. The content model's clean logits serve every attack."""
    rows = []
    for how, pred in (("none", lambda z: z.argmax(1)), ("decision", lambda z: gate_decision(z, content, benign)),
                      ("logits", lambda z: gate_logits(z, content, benign))):
        rows.append({"gate": how, **row_of(y, benign, pred(base["clean"]), {a: pred(base[a]) for a in attacks if a in base}, masks)})
    return rows


def load_logits(f):
    return {name: f[name] for name in ("clean", *ATTACKS) if name in f.files}


def masks_of(f, y):
    masks = {name.removeprefix("attacked_"): f[name] for name in f.files if name.startswith("attacked_")}
    return masks | {"random": np.ones(len(y), bool)}


def margin(z, benign):
    return z[:, ~benign].max(1) - z[:, benign].max(1)


def curve(base, content, y, benign, model, run, loss):
    """Certified detection against false alarms, with the margin chosen on one half of the test flows and scored on the other."""
    half = np.random.default_rng(0).permutation(len(y)) % 2 == 0
    malware = ~benign[y]
    m = margin(content, benign)
    a = base.argmax(1)
    plain = 100 * benign[a][~half & ~malware].mean()
    rows = []
    for cost in COSTS:
        # the gate's false alarms grow as the margin falls: benign flows the base model clears and the content model overrules
        recall = lambda part, t: 100 * (benign[a] & ~(m > t))[part & ~malware].mean()
        options = np.quantile(m[half & ~malware], np.linspace(0, 1, 401))
        fits = [t for t in options if recall(half, t) >= recall(half, np.inf) - cost]
        t = float(min(fits)) if fits else float("inf")
        rows.append({"model": model, "dataset": run, "base": loss, "cost": cost, "margin": round(t, 2),
                     "benign_recall_none": round(plain, 1), "benign_recall": round(recall(~half, t), 1),
                     "certified": round(100 * (m > t)[~half & malware].mean(), 1)})
    return rows


def certified_share(content, y, benign):
    return round(100 * (~benign[content.argmax(1)])[~benign[y]].mean(), 1)


def gate_bar(paths_of, title):
    """The gate on the runs whose ce and mask logits exist, against the bar in plan.txt. Returns the rows, or None before any are in."""
    rows = []
    for run in RUNS:
        paths = paths_of(run)
        if not all(p.exists() for p in paths):
            continue
        ce, mask = (np.load(p, allow_pickle=False) for p in paths)
        assert np.array_equal(ce["row"], mask["row"]), run
        y, classes = ce["y"], ce["classes"].astype(str)
        benign = np.isin(classes, list(BENIGN))
        rows += [{"dataset": run, "certified": certified_share(mask["clean"], y, benign), **r}
                 for r in scores(load_logits(ce), mask["clean"], y, benign, masks_of(ce, y), ATTACKS)]
    if not rows:
        return None
    df = pd.DataFrame(rows)
    print(f"\n\n##### {title}")
    print(df[["dataset", "gate", "certified", "clean", "benign_recall", "detected", "mimic_stack", "evaded_mimic_stack", "evaded_random"]].to_string(index=False))
    plain, gated = df[df["gate"] == "none"].set_index("dataset"), df[df["gate"] == "logits"].set_index("dataset")
    n = len(gated)
    results = [("(1) certified at least 95", int((gated["certified"] >= 95).sum()), 5),
               ("(2) clean no more than 2 below ce", int((plain["clean"] - gated["clean"] <= 2).sum()), 5),
               ("(3) Benign recall at most 5 below ce's", int((plain["benign_recall"] - gated["benign_recall"] <= 5).sum()), 6)]
    for text, passed, need in results:
        print(f"  {'PASS' if passed >= need else 'miss'}  {text}: {passed} of {n} (needs {need})")
    print(f"  (2) read as within 2 either way: {int(((plain['clean'] - gated['clean']).abs() <= 2).sum())} of {n}")
    return df


def time_split(folder=GATE_JOB, out=ROOT / "results" / "litcvit_gate_time.csv"):
    df = gate_bar(lambda run: (folder / f"{run}_time_mobilenetv2.npz", folder / f"{run}_time_mobilenetv2_mask.npz"), "the gate on the time split")
    if df is not None:
        df.to_csv(out, index=False)
    return df


def new_seeds(folder=SEEDS_JOB, out=ROOT / "results" / "litcvit_gate_seeds.csv"):
    frames = []
    for seed in (1, 2):
        df = gate_bar(lambda run: (folder / f"{run}_mobilenetv2_ce_seed{seed}.npz", folder / f"{run}_mobilenetv2_mask_seed{seed}.npz"),
                      f"the gate on the random split, seed {seed}")
        if df is not None:
            frames.append(df.assign(seed=seed))
    if frames:
        pd.concat(frames).to_csv(out, index=False)
    return frames


def single_network(folder=GATE_JOB, two=LOSS, out=ROOT / "results" / "litcvit_gate_single.csv"):
    """One network against the gate's two separate models, MobileNetV2, random split, on the bar in plan.txt."""
    rows = []
    for run in SMALL:
        ce, mask = (np.load(two / f"{run}_mobilenetv2_{loss}.npz", allow_pickle=False) for loss in ("ce", "mask"))
        y = ce["y"]
        benign = np.isin(ce["classes"].astype(str), list(BENIGN))
        macro = lambda pred: round(100 * f1_score(y, pred, average="macro"), 1)
        pair = {"clean": macro(gate_logits(ce["clean"], mask["clean"], benign)), "certified": certified_share(mask["clean"], y, benign)}
        for loss in ("gated", "views"):
            path = folder / f"{run}_mobilenetv2_{loss}.npz"
            if not path.exists():
                continue
            f = np.load(path, allow_pickle=False)
            assert np.array_equal(f["row"], ce["row"]), (run, loss)
            rows.append({"dataset": run, "loss": loss, "clean": macro(f["clean"].argmax(1)), "certified": certified_share(f["zero"], y, benign),
                         "clean_two": pair["clean"], "certified_two": pair["certified"]})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print("\n\n##### one network or two")
    print(df.to_string(index=False))
    for loss, part in df.groupby("loss", sort=False):
        passed = int((((part["clean"] - part["clean_two"]).abs() <= 1) & ((part["certified"] - part["certified_two"]).abs() <= 1)).sum())
        print(f"  {'PASS' if passed >= 2 else 'miss'}  {loss}: within 1 of the two models on clean and certified: {passed} of {len(part)} (needs 2)")
    return df


def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 300)
    rows, wrapped, curves = [], [], []
    for model in MODELS:
        for run in RUNS:
            mask = np.load(LOSS / f"{run}_{model}_mask.npz", allow_pickle=False)
            y, classes = mask["y"], mask["classes"].astype(str)
            benign = np.isin(classes, list(BENIGN))
            content = mask["clean"]
            certified = round(100 * (~benign[content.argmax(1)])[~benign[y]].mean(), 1)
            masks = masks_of(mask, y)
            for loss in BASES:
                f = np.load(LOSS / f"{run}_{model}_{loss}.npz", allow_pickle=False)
                assert np.array_equal(f["row"], mask["row"]), (run, model, loss)
                base = load_logits(f)
                rows += [{"model": model, "dataset": run, "base": loss, "certified": certified, **r} for r in scores(base, content, y, benign, masks, ATTACKS)]
                curves += curve(base["clean"], content, y, benign, model, run, loss)
            for path in sorted(EVASION.glob(f"{run}_*.npz")):  # any detector can be wrapped, here those of the evasion job
                name = path.stem.removeprefix(run + "_")
                if name not in ("xgb", "litcvit", "litcvit-sa") or model != "shufflenet":
                    continue
                f = np.load(path, allow_pickle=False)
                assert np.array_equal(f["classes"], mask["classes"]) and np.array_equal(np.sort(f["row"]), np.sort(mask["row"])), (run, name)
                order = np.argsort(f["row"])[np.argsort(np.argsort(mask["row"]))]  # f's flows in the content model's order
                assert np.array_equal(f["row"][order], mask["row"]), (run, name)
                base = {k: f[k][order] for k in ("clean", *ATTACKS) if k in f.files}
                same = {"mimic_stack": f["attacked_same"][order], "mimic_ttl_window": f["attacked_same"][order],
                        "mimic_stack_other": f["attacked_other"][order]}
                wrapped += [{"dataset": run, "base": name, "content": model, "certified": certified, **r}
                            for r in scores(base, content, y, benign, same, tuple(same))]
    df, wrap, cur = pd.DataFrame(rows), pd.DataFrame(wrapped), pd.DataFrame(curves)
    df.to_csv(ROOT / "results" / "litcvit_gate.csv", index=False)
    wrap.to_csv(ROOT / "results" / "litcvit_gate_wrapped.csv", index=False)
    cur.to_csv(ROOT / "results" / "litcvit_gate_curve.csv", index=False)
    for column in ("clean", "benign_recall", "detected", "mimic_stack", "evaded_mimic_stack", "random", "evaded_random"):
        print(f"\n--- {column}")
        print(df.pivot_table(index=["model", "dataset"], columns=["base", "gate"], values=column, sort=False).to_string())
    print("\n--- certified (any change to the shortcut fields)")
    print(df.drop_duplicates(["model", "dataset"]).pivot_table(index="dataset", columns="model", values="certified", sort=False).to_string())
    print("\n--- other detectors wrapped by the ShuffleNet content model")
    print(wrap[["dataset", "base", "gate", "certified", "clean", "benign_recall", "detected", "mimic_stack", "evaded_mimic_stack", "evaded_mimic_stack_other"]].to_string(index=False))
    print("\n--- certified detection against false alarms (margin from half the test flows, scored on the other)")
    print(cur[cur["base"] == "ce"].pivot_table(index=["model", "dataset", "benign_recall_none"], columns="cost", values=["benign_recall", "certified"], sort=False).to_string())
    time_split()
    new_seeds()
    single_network()


if __name__ == "__main__":
    main()
