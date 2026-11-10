"""Score the loss comparison against the bars set in plan.txt before any results were in.

Reads the Kaggle jobs' output (kaggle/litcvit_loss and, once downloaded, kaggle/litcvit_oneside and
kaggle/litcvit_ablate), prints macro-F1 and evasion per run, model and loss, then each bar with its
pass count per model. Also scores the ce + mask ensemble that a reviewer would ask about: it answers
as ce does, except that a flow ce calls Benign and mask doesn't takes mask's answer. It needs two
models. The ablation retrains some rows of the loss job, so it is scored on its own rows only, and
LitCVit, which has no plain rows in these jobs, against results/litcvit_payload.csv and
results/litcvit_evasion.csv.

  python kaggle/litcvit/loss_report.py

Writes results/litcvit_loss.csv (all rows of all jobs, with a job column) and
results/litcvit_loss_ensemble.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent.parent
KAGGLE = Path("C:/data/CNS_proj/kaggle")
ABLATION = "ablate.csv"
JOBS = {"litcvit_output_loss": "loss.csv", "litcvit_output_oneside": "oneside.csv", "litcvit_output_ablate": ABLATION}
RUNS = ["5gad2022", "mcfp", "mcfp_strict", "ciciot2022", "tii_ssrc23", "iot23", "ustc"]
ATTACK_RUNS = ["mcfp_strict", "ustc"]  # the runs with TLS malware, where a forged header fits the threat model
BENIGN = {"Benign", "Normal"}


def load():
    frames = [pd.read_csv(KAGGLE / job / name).assign(job=name) for job, name in JOBS.items() if (KAGGLE / job / name).exists()]
    return pd.concat(frames, ignore_index=True)


def table(df, condition, value="macro_f1"):
    rows = df[df["condition"] == condition]
    return rows.pivot_table(index=["model", "dataset"], columns="loss", values=value, sort=False).round(1)


def show_bar(name, results):
    for text, passed, total, need in results:
        print(f"  {'PASS' if passed >= need else 'miss'}  {text}: {passed} of {total} (needs {need})")


def bars(df, model):
    """Each bar as written in plan.txt, for one model; returns nothing, prints the verdicts."""
    d = df[df["model"] == model]
    get = lambda loss, cond, value="macro_f1": d[(d["loss"] == loss) & (d["condition"] == cond)].set_index("dataset")[value]
    have = set(d["loss"])
    print(f"\n=== {model}")
    if {"ce", "mask", "rand", "swap", "cfnce"} <= have:
        clean, attack = get("cfnce", "clean"), get("cfnce", "mimic_stack")
        score = lambda loss: (get(loss, "clean") + get(loss, "mimic_stack")) / 2
        print(" cfnce:")
        show_bar("cfnce", [
            ("(1) beats ce by 5+ under mimic_stack", int((attack - get("ce", "mimic_stack") >= 5).sum()), len(clean), 4),
            ("(2) beats swap and rand on mean of clean and mimic_stack",
             int(((score("cfnce") > score("swap")) & (score("cfnce") > score("rand"))).sum()), len(clean), 4),
            ("(3) clean beats mask by 1+, and within 5 of its own clean under attack",
             int(((clean - get("mask", "clean") >= 1) & (clean - attack <= 5)).sum()), len(clean), 3)])
    if {"ce", "mask", "oneside"} <= have:
        clean = get("oneside", "clean")
        evaded = get("oneside", "mimic_stack", "evaded")
        evaded_ce, evaded_mask = get("ce", "mimic_stack", "evaded"), get("mask", "mimic_stack", "evaded")
        holds = lambda r: evaded[r] <= evaded_mask[r] + 5 and evaded[r] <= evaded_ce[r] - 20
        written = [r for r in ATTACK_RUNS if r in evaded.index and evaded_ce.get(r, 0) > 0]
        possible = [r for r in written if evaded_ce[r] >= 20]  # 20 below ce can't be met where ce is evaded less than that
        print(" oneside:")
        show_bar("oneside", [
            ("(1) clean within 1 of ce", int((get("ce", "clean") - clean <= 1).sum()), len(clean), 5),
            ("(2) clean beats mask by 1+", int((clean - get("mask", "clean") >= 1).sum()), len(clean), 3),
            ("(3) as written: evaded at most 5 above mask's and 20+ below ce's, runs with TLS malware where ce is evaded at all",
             sum(holds(r) for r in written), len(written), len(written)),
            ("(3) only the runs that can meet it, where ce is evaded 20+", sum(holds(r) for r in possible), len(possible), len(possible))])


def ablation(df):
    """The bar set in plan.txt before the job's results: does the contrastive term, or the donor's weight, do the work?"""
    d = df[df["job"] == ABLATION]
    if d.empty:
        return
    print("\n\n##### ablation")
    for condition, value in (("clean", "macro_f1"), ("mimic_stack", "macro_f1"), ("isn", "macro_f1"), ("mimic_all", "macro_f1"),
                             ("mimic_stack", "evaded"), ("mimic_all", "evaded")):
        print(f"\n--- {value} under {condition}")
        print(table(d, condition, value).to_string())
    for model in sorted(set(d["model"])):
        rows = d[d["model"] == model]
        get = lambda loss, cond, value="macro_f1", within=rows: within[(within["loss"] == loss) & (within["condition"] == cond)].set_index("dataset")[value]
        score = lambda loss: (get(loss, "clean") + get(loss, "mimic_stack")) / 2
        print(f"\n=== {model}")
        for other, what in (("both", "the contrastive term"), ("cfnce1", "the donor's weight")):
            gap = score("cfnce") - score(other)
            print(f" {what}, cfnce minus {other} on the mean of clean and mimic_stack: {gap.round(1).to_dict()}")
            show_bar(what, [("cfnce wins by 1+", int((gap >= 1).sum()), len(gap), 4),
                            ("cfnce doesn't lose by 1+", int((gap > -1).sum()), len(gap), len(gap) - 1)])
        for loss in sorted(set(rows["loss"]) & {"both", "cfnce", "cfnce1"}):
            rise = get(loss, "mimic_all", "evaded") - get(loss, "mimic_stack", "evaded")
            show_bar(loss, [(f"{loss}: evaded under mimic_all within 5 of mimic_stack", int((rise <= 5).sum()), len(rise), len(rise) - 1)])
        first = df[(df["job"] == "loss.csv") & (df["model"] == model)]
        for loss in sorted(set(rows["loss"]) & set(first["loss"])):
            again = (get(loss, "clean") - get(loss, "clean", within=first)).abs()
            print(f" same seed trained twice, {loss}: clean differs by {again.mean():.2f} on average, {again.max():.2f} at most")


def litcvit_rows(df):
    """LitCVit with the one-sided loss beside plain LitCVit, whose rows come from other jobs."""
    d = df[df["model"] == "litcvit"]
    if d.empty:
        return
    plain = pd.read_csv(ROOT / "results" / "litcvit_payload.csv")
    plain = plain[plain["model"] == "litcvit"]
    of = lambda rows, cond, value="macro_f1": rows[rows["condition"] == cond].set_index("dataset")[value]
    columns = {f"{cond} {name}": of(rows, cond) for cond in ("clean", "random", "nopayload") for name, rows in (("plain", plain), ("oneside", d))}
    columns |= {"mimic_stack oneside": of(d, "mimic_stack"), "mimic_stack_other oneside": of(d, "mimic_stack_other"),
                "evaded oneside": of(d, "mimic_stack", "evaded")}
    evasion = ROOT / "results" / "litcvit_evasion.csv"
    if evasion.exists():
        e = pd.read_csv(evasion)
        e = e[(e["model"] == "litcvit") & (e["subset"] == "all") & (e["class"] == "all attack") & (e["condition"] == "mimic_stack")]
        columns["evaded plain"] = e.set_index("dataset")["evaded"]
    print("\n--- LitCVit, plain and one-sided")
    print(pd.DataFrame(columns).reindex(RUNS).round(1).to_string())


def ensemble(df):
    """Scores of the fail-closed ce + mask ensemble, from the saved logits of the loss job."""
    preds = KAGGLE / "litcvit_output_loss" / "preds"
    rows = []
    for model in sorted(set(df["model"])):
        for run in RUNS:
            paths = [preds / f"{run}_{model}_{loss}.npz" for loss in ("ce", "mask")]
            if not all(p.exists() for p in paths):
                continue
            ce, mask = (np.load(p, allow_pickle=False) for p in paths)
            y, classes = ce["y"], ce["classes"].astype(str)
            benign = np.isin(classes, list(BENIGN))
            answer = lambda cond: np.where(benign[ce[cond].argmax(1)] & ~benign[mask[cond].argmax(1)], mask[cond].argmax(1), ce[cond].argmax(1))
            clean, attack = answer("clean"), answer("mimic_stack")
            m = ce["attacked_mimic_stack"]
            detected = ~benign[clean[m]]
            rows.append({"model": model, "dataset": run, "clean": round(100 * f1_score(y, clean, average="macro"), 1),
                         "mimic_stack": round(100 * f1_score(y, attack, average="macro"), 1),
                         "evaded": round(100 * benign[attack[m]][detected].mean(), 1) if detected.any() else float("nan")})
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 220)
    df = load()
    df.to_csv(ROOT / "results" / "litcvit_loss.csv", index=False)
    first = df[df["job"] != ABLATION]
    for condition, value in (("clean", "macro_f1"), ("random", "macro_f1"), ("mimic_stack", "macro_f1"), ("mimic_stack", "evaded"),
                             ("mimic_stack_other", "macro_f1")):
        print(f"\n--- {value} under {condition}")
        print(table(first, condition, value).to_string())
    for model in sorted(set(first["model"])):
        bars(first, model)
    both = ensemble(first)
    if len(both):
        both.to_csv(ROOT / "results" / "litcvit_loss_ensemble.csv", index=False)
        print("\n--- ce + mask, fail-closed (two models)")
        print(both.to_string())
    litcvit_rows(first)
    ablation(df)


if __name__ == "__main__":
    main()
