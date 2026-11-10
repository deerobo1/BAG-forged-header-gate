"""How far do the scores move with a new seed? Seed 0 from the Kaggle jobs, seeds 1 and 2 from kaggle/litcvit_seeds.

A new seed changes the random split, the attack's donors and the weights. MobileNetV2, random split. Prints, per run and
loss, the mean over the three seeds with the lowest and highest score, for macro-F1 on clean flows, under mimic_stack and
with the fields randomised, and for the share evaded. A retrain with seed 0 (kaggle/litcvit_ablate) counts towards the
range but not the mean. Then, for each seed, the numbers behind the bars for the contrastive term and the one-sided loss,
which were set for seed 0 alone and already scored there (kaggle/litcvit/loss_report.py), so these only show whether
their verdicts survive a new seed. Seeds 1 and 2 ran on Colab and seed 0 on Kaggle, so a gap between them could also come
from the platform. The gate's own bar for new seeds is scored by kaggle/litcvit/gate_report.py.

  python kaggle/litcvit/seed_report.py

Reads C:/data/CNS_proj/kaggle/litcvit_output_seeds/seeds.csv and results/litcvit_loss.csv; writes results/litcvit_seeds.csv.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
SEEDS = Path("C:/data/CNS_proj/kaggle/litcvit_output_seeds/seeds.csv")
RUNS = ["5gad2022", "mcfp", "mcfp_strict", "ciciot2022", "tii_ssrc23", "iot23", "ustc"]
MODEL = "mobilenetv2"
FIRST = {"ce": "loss.csv", "mask": "loss.csv", "both": "ablate.csv", "oneside": "oneside.csv", "cfnce": "loss.csv"}  # which job first trained seed 0
RETRAINED = {"ce", "mask", "cfnce"}  # and kaggle/litcvit_ablate trained these again
MEASURES = {"clean": ("clean", "macro_f1"), "mimic_stack": ("mimic_stack", "macro_f1"), "random": ("random", "macro_f1"), "evaded": ("mimic_stack", "evaded")}
LABELS = ("seed 0", "seed 1", "seed 2")
AGAIN = "seed 0 again"


def trainings(seeds=SEEDS, first=ROOT / "results" / "litcvit_loss.csv"):
    """All rows with a label per training."""
    old = pd.read_csv(first)
    old = old[(old["model"] == MODEL) & (old["split"] == "random")]
    frames = [pd.read_csv(seeds).assign(label=lambda d: "seed " + d["seed"].astype(str))]
    for loss, job in FIRST.items():
        frames.append(old[(old["loss"] == loss) & (old["job"] == job)].assign(label="seed 0"))
        if loss in RETRAINED:
            frames.append(old[(old["loss"] == loss) & (old["job"] == "ablate.csv")].assign(label=AGAIN))
    return pd.concat(frames, ignore_index=True)


def spread(d):
    """One row per run, loss and measure, with each training's value, the mean over the seeds and the range."""
    rows = []
    for measure, (condition, value) in MEASURES.items():
        by = d[d["condition"] == condition].pivot_table(index=["dataset", "loss"], columns="label", values=value)
        for (run, loss), scores in by.iterrows():
            seeds = scores.reindex(list(LABELS)).dropna()
            rows.append({"dataset": run, "loss": loss, "measure": measure, **scores.dropna().to_dict(), "mean": round(seeds.mean(), 1),
                         "low": round(scores.min(), 1), "high": round(scores.max(), 1)})
    return pd.DataFrame(rows)


def show(table, measure):
    t = table[table["measure"] == measure]
    cells = t.assign(cell=[f"{m:.1f} ({a:.1f} to {b:.1f})" for m, a, b in zip(t["mean"], t["low"], t["high"])])
    print(f"\n--- {measure}: mean over the seeds (lowest to highest)")
    print(cells.pivot(index="dataset", columns="loss", values="cell").reindex(index=RUNS, columns=list(FIRST)).fillna("-").to_string())


def verdicts(d, label):
    """The bars' numbers for one seed. The contrastive bar compares rows of one job, so seed 0's cfnce is the retrain beside both."""
    get = lambda loss, cond, value="macro_f1", at=label: d[(d["label"] == at) & (d["loss"] == loss) & (d["condition"] == cond)].set_index("dataset")[value].reindex(RUNS)
    score = lambda loss, at=label: (get(loss, "clean", at=at) + get(loss, "mimic_stack", at=at)) / 2
    gap = (score("cfnce", AGAIN if label == "seed 0" else label) - score("both")).dropna()
    print(f"\n=== {label}")
    if len(gap):
        print(f" cfnce minus both, mean of clean and mimic_stack: {gap.round(1).to_dict()}")
        print(f"  wins by 1+ on {int((gap >= 1).sum())} of {len(gap)}, loses by 1+ on {int((gap <= -1).sum())}")
    clean, evaded = get("oneside", "clean"), get("oneside", "mimic_stack", "evaded")
    done = clean.notna()
    print(f" oneside, of {int(done.sum())} runs: clean no more than 1 below ce on {int((get('ce', 'clean') - clean)[done].le(1).sum())}, "
          f"beats mask by 1+ on {int((clean - get('mask', 'clean'))[done].ge(1).sum())}; evaded on MCFP strict {evaded['mcfp_strict']} (ce {get('ce', 'mimic_stack', 'evaded')['mcfp_strict']}), "
          f"on USTC {evaded['ustc']} (ce {get('ce', 'mimic_stack', 'evaded')['ustc']})")


def main(seeds=SEEDS, out=ROOT / "results" / "litcvit_seeds.csv"):
    pd.set_option("display.width", 240)
    d = trainings(seeds)
    table = spread(d)
    table.to_csv(out, index=False)
    for measure in MEASURES:
        show(table, measure)
    for label in LABELS:
        verdicts(d, label)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
