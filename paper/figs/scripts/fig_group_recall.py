"""Fig.: Benign recall on the benign-host-disjoint split, per run and model (the L21 finding).

sprint3/results/e8_group.csv: group split, seeds 0-2 (mean, min-max) for F', B' and gate' (XGBoost members);
random-split counterpart of gate', seed 0 only (marker, no bar).
"""

import numpy as np

from style import BLACK, BLUE, COL, GREEN, HATCH, LABEL, RUNS, SKY, bars, csv, plt, save, spread


def main():
    e8 = csv("e8_group.csv")
    g = e8[e8.split == "group"]
    fig, ax = plt.subplots(figsize=(COL, 1.6))
    x = np.arange(len(RUNS))
    for i, (m, c, h, lab) in enumerate([("F'", BLUE, HATCH[0], "F$'$"), ("B'", SKY, HATCH[4], "B$'$ (masked)"),
                                        ("gate'", GREEN, HATCH[3], "Gate$'$")]):
        mean, lo, hi = spread(g[g.method == m], "run", "benign_recall")
        bars(ax, x + (i - 1) * 0.26, mean[RUNS].values, lo[RUNS].values, hi[RUNS].values, c, h, lab, width=0.26)
    rnd = e8[(e8.split == "random") & (e8.method == "gate'") & (e8.seed == 0)].set_index("run").benign_recall
    ax.plot(x + 0.26, rnd[RUNS].values, ls="none", marker="D", ms=3, color=BLACK, label="Gate$'$, random split (seed 0)")
    ax.set_ylim(0, 108)
    ax.set_ylabel("Benign recall (%)")
    ax.set_xticks(x, [LABEL[r_] for r_ in RUNS])
    ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0), columnspacing=0.8, handlelength=1.4)
    save(fig, "fig_group_recall")


if __name__ == "__main__":
    main()
