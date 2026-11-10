"""Fig.: operating curve, false-alarm rate against attack success under (b), as F's threshold t varies.

sprint3/results/e9_curve.csv, random split, seeds 0-2. Points are 3-seed means at
t = 0.5 ... 0.999; error bars are min-max over seeds of the success. Gate F_t+B, masking B_t, and F_t alone.
"""

import numpy as np

from style import BLUE, GREEN, LABEL, RUNS, SKY, TEXT, csv, plt, save


def main():
    c = csv("e9_curve.csv")
    c = c[c.split == "random"]
    fig, axes = plt.subplots(1, 7, figsize=(TEXT, 1.45), sharey=True)
    styles = [("F_t", BLUE, "o", ":", "F alone"), ("mask_t", SKY, "s", "--", "Masking B"), ("gate_t", GREEN, "^", "-", "Gate F+B")]
    for ax, run in zip(axes, RUNS):
        d = c[c.run == run]
        for m, col, mk, ls, lab in styles:
            g = d[d.method == m].groupby("t")
            fa, s, lo, hi = g.false_alarm.mean(), g.uncond_b.mean(), g.uncond_b.min(), g.uncond_b.max()
            ax.errorbar(fa, s, yerr=[s - lo, hi - s], color=col, marker=mk, ms=2.6, ls=ls, lw=0.9,
                        elinewidth=0.5, capsize=1, label=lab)
        ax.set_title(LABEL[run].replace("\n", " "), pad=2)
        ax.set_xlim(left=0)
        ax.set_xlabel("False alarms (%)")
        ax.set_ylim(-3, 103)
    axes[0].set_ylabel("Cleared under (b) (%)")
    axes[0].legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.12), columnspacing=1.0)
    fig.subplots_adjust(wspace=0.12)
    save(fig, "fig_operating")


if __name__ == "__main__":
    main()
