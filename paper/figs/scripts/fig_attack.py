"""Fig.: header-attack success by run and model, (a) donor copy and (b) best of 8 random valid values.

Conditional success (of malware detected as recorded, share called Benign after the attack).
MobileNetV2 F, XGBoost and gate F+B: sprint3/results/e4_methods.csv, random split, seeds 0-2 (mean, min-max).
LitCVit: results/litcvit_limits.csv (mimic_stack, arbitrary_any), seed 0, the two runs with saved weights.
"""

import numpy as np

from style import BLUE, COL, GREEN, HATCH, LABEL, ORANGE, RES, RUNS, VERM, bars, csv, plt, save, spread


def main():
    e4 = csv("e4_methods.csv")
    r = e4[e4.split == "random"]
    lim = csv("litcvit_limits.csv", RES)
    lit = lim[(lim.model == "litcvit") & (lim.gate == "none")].set_index(["dataset", "condition"]).evaded
    fig, axes = plt.subplots(2, 1, figsize=(COL, 3.1), sharex=True)
    x = np.arange(len(RUNS))
    for ax, col, cond, title in [(axes[0], "cond_a", "mimic_stack", "(a) copy a benign donor's fields"),
                                 (axes[1], "cond_b", "arbitrary_any", "(b) best of 8 random valid values")]:
        for i, (m, c, h, lab) in enumerate([("F", BLUE, HATCH[0], "MobileNetV2 F"), ("xgb", ORANGE, HATCH[1], "XGBoost"),
                                            ("gate", GREEN, HATCH[3], "Gate F+B")]):
            mean, lo, hi = spread(r[r.method == m], "run", col)
            bars(ax, x + (i - 1.5) * 0.2, mean[RUNS].values, lo[RUNS].values, hi[RUNS].values, c, h, lab)
        lv = [lit.get((run, cond), np.nan) for run in RUNS]
        bars(ax, x + 1.5 * 0.2, np.array(lv), color=VERM, hatch=HATCH[2], label="LitCVit (2 runs, 1 seed)")
        ax.set_ylim(0, 105)
        ax.set_ylabel("Malware cleared (%)")
        ax.set_title(title, loc="left", pad=2)
    axes[1].set_xticks(x, [LABEL[r_] for r_ in RUNS])
    axes[0].legend(ncol=2, loc="upper right", bbox_to_anchor=(1.0, 1.55), columnspacing=0.8, handlelength=1.4)
    save(fig, "fig_attack")


if __name__ == "__main__":
    main()
