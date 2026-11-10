"""Fig.: clean macro-F1 of F, B (masking) and gate F+B per run.

sprint3/results/e4_methods.csv, random split, seeds 0-2: mean, error bars min-max over seeds.
"""

import numpy as np

from style import BLUE, COL, GREEN, HATCH, LABEL, RUNS, SKY, bars, csv, plt, save, spread


def main():
    e4 = csv("e4_methods.csv")
    r = e4[e4.split == "random"]
    fig, ax = plt.subplots(figsize=(COL, 1.55))
    x = np.arange(len(RUNS))
    for i, (m, c, h, lab) in enumerate([("F", BLUE, HATCH[0], "F (full image)"), ("B", SKY, HATCH[4], "B (masking)"),
                                        ("gate", GREEN, HATCH[3], "Gate F+B")]):
        mean, lo, hi = spread(r[r.method == m], "run", "clean_f1")
        bars(ax, x + (i - 1) * 0.26, mean[RUNS].values, lo[RUNS].values, hi[RUNS].values, c, h, lab, width=0.26)
    ax.set_ylim(40, 101)
    ax.set_ylabel("Clean macro-F1")
    ax.set_xticks(x, [LABEL[r_] for r_ in RUNS])
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), columnspacing=0.8, handlelength=1.4)
    save(fig, "fig_clean_f1")


if __name__ == "__main__":
    main()
