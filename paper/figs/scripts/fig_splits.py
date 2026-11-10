"""Fig.: header-attack success and Benign recall of the gate under the random, time and benign-host splits.

Random and time: gate F+B (MobileNetV2 members), sprint3/results/e4_methods.csv; random = seeds 0-2
(mean, min-max), time = seed 0 only (no bar). Benign-host (group): gate' with XGBoost members F', B',
sprint3/results/e8_group.csv, seeds 0-2. Success: worst of (a), (b), (m), unconditional.
"""

import numpy as np

from style import BLUE, COL, HATCH, LABEL, ORANGE, PURPLE, RUNS, bars, csv, plt, save, spread


def main():
    e4 = csv("e4_methods.csv")
    g = e4[e4.method == "gate"]
    e8 = csv("e8_group.csv")
    e8 = e8[(e8.split == "group") & (e8.method == "gate'")].copy()
    e8["header_uncond_max"] = e8[["uncond_a", "uncond_b", "uncond_m"]].max(axis=1)
    fig, axes = plt.subplots(2, 1, figsize=(COL, 2.75), sharex=True)
    x = np.arange(len(RUNS))
    for ax, col, ylab in [(axes[0], "header_uncond_max", "Header attack\ncleared (%)"), (axes[1], "benign_recall", "Benign recall (%)")]:
        mean, lo, hi = spread(g[g.split == "random"], "run", col)
        bars(ax, x - 0.26, mean[RUNS].values, lo[RUNS].values, hi[RUNS].values, BLUE, HATCH[0], "random, 3 seeds", 0.26)
        t = g[g.split == "time"].set_index("run")[col]
        bars(ax, x, t[RUNS].values, color=ORANGE, hatch=HATCH[1], label="time, 1 seed", width=0.26)
        mean, lo, hi = spread(e8, "run", col)
        bars(ax, x + 0.26, mean[RUNS].values, lo[RUNS].values, hi[RUNS].values, PURPLE, HATCH[2],
             "benign-host, 3 seeds (XGBoost members)", 0.26)
        ax.set_ylabel(ylab)
    axes[0].set_ylim(0, 35)
    axes[1].set_ylim(0, 105)
    axes[1].set_xticks(x, [LABEL[r_] for r_ in RUNS])
    axes[0].legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0), columnspacing=0.8, handlelength=1.4)
    save(fig, "fig_splits")


if __name__ == "__main__":
    main()
