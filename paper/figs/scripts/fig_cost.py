"""Fig.: cost of F, gate F+B and gate F+B+C: parameters, multiply-adds, CPU latency per flow.

sprint3/results/e9_cost.csv, 5GAD-2022 seed 0, 12 CPU threads, median of 5 runs (one measurement, no bars).
C's 1,800 trees are not counted in parameters or multiply-adds.
"""

import numpy as np

from style import BLUE, COL, GREEN, HATCH, SKY, csv, plt, save


def main():
    c = csv("e9_cost.csv")
    models = ["F", "F+B", "F+B+C"]
    colors, hatches = [BLUE, SKY, GREEN], [HATCH[0], HATCH[4], HATCH[3]]
    q = lambda m, b, col: c[(c.method == m) & (c.batch == b)][col].iloc[0]
    panels = [("Params (M)", [q(m, 1, "params") / 1e6 for m in models]),
              ("MACs (M)", [q(m, 1, "macs") / 1e6 for m in models]),
              ("ms, batch 1", [q(m, 1, "ms_per_flow") for m in models]),
              ("ms, batch 256", [q(m, 256, "ms_per_flow") for m in models])]
    fig, axes = plt.subplots(1, 4, figsize=(COL, 1.5))
    for ax, (title, vals) in zip(axes, panels):
        ax.bar(np.arange(3), vals, 0.7, color=colors, edgecolor="black", linewidth=0.4, hatch=hatches)
        ax.set_title(title, pad=2, fontsize=8)
        ax.set_xticks(np.arange(3), models, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_ylim(0, max(vals) * 1.3)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    fig.subplots_adjust(wspace=0.6)
    save(fig, "fig_cost")


if __name__ == "__main__":
    main()
