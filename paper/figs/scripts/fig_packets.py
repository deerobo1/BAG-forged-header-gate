"""Fig.: packet attacks (outside the guarantee) per run, with C off (gate F+B) and on (gate F+B+C).

sprint3/results/e6_packets.csv, C_A, random split, seeds 0-2: conditional success, mean, min-max over seeds.
(c) added packet (a family C_A was trained on), (d) split payload, (L) length copy, (H) chaff.
"""

import numpy as np

from style import GREEN, HATCH, LABEL, RUNS, TEXT, VERM, bars, csv, plt, save, spread


def main():
    d = csv("e6_packets.csv")
    d = d[(d.C_variant == "C_A") & (d.split == "random")]
    fig, axes = plt.subplots(1, 4, figsize=(TEXT, 1.55), sharey=True)
    x = np.arange(len(RUNS))
    for ax, (a, title) in zip(axes, [("c", "(c) added packet (trained on)"), ("d", "(d) split payload"),
                                     ("L", "(L) length copy"), ("H", "(H) chaff")]):
        for i, (m, col, h, lab) in enumerate([("gate", VERM, HATCH[1], "C off: gate F+B"), ("gate+C", GREEN, HATCH[3], "C on: gate F+B+C")]):
            mean, lo, hi = spread(d[(d.attack == a) & (d.method == m)], "run", "cond")
            bars(ax, x + (i - 0.5) * 0.36, mean[RUNS].values, lo[RUNS].values, hi[RUNS].values, col, h, lab, width=0.36)
        ax.set_title(title, loc="left", pad=2)
        ax.set_xticks(x, [LABEL[r_].replace("\n", " ") for r_ in RUNS], rotation=60, ha="right", rotation_mode="anchor")
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("Malware cleared (%)")
    axes[0].legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.1))
    fig.subplots_adjust(wspace=0.08)
    save(fig, "fig_packets")


if __name__ == "__main__":
    main()
