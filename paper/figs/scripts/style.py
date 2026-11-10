"""Shared style and data helpers for the paper's figures.

Sizes are the IEEEtran journal print sizes (column 3.5 in, text 7.16 in), so 8 pt here is 8 pt in print.
Palette: Okabe-Ito (colour-blind safe); every series also differs in hatch or marker, so it reads in grey.
Error bars are the min-max over the three random-split seeds; one-seed values get no bar.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
R3 = ROOT / "sprint3" / "results"
RES = ROOT / "results"
OUT = ROOT / "paper" / "figs"

COL, TEXT = 3.5, 7.16
RUNS = ["5gad2022", "mcfp", "mcfp_strict", "ciciot2022", "tii_ssrc23", "iot23", "ustc"]
LABEL = {"5gad2022": "5GAD", "mcfp": "MCFP", "mcfp_strict": "MCFP\nstrict", "ciciot2022": "CICIoT",
         "tii_ssrc23": "TII", "iot23": "IoT-23", "ustc": "USTC"}
# Okabe-Ito
BLUE, ORANGE, GREEN, VERM, SKY, PURPLE, YELLOW, BLACK = ("#0072B2", "#E69F00", "#009E73", "#D55E00",
                                                         "#56B4E9", "#CC79A7", "#F0E442", "#000000")
HATCH = ["", "////", "....", "xxxx", "\\\\\\\\", "++"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "hatch.linewidth": 0.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def csv(name, base=R3):
    return pd.read_csv(base / name)


def spread(df, by, value):
    """Mean, min and max of `value` over seeds, per `by` group."""
    g = df.groupby(by)[value]
    return g.mean(), g.min(), g.max()


def bars(ax, x, mean, lo=None, hi=None, color=BLUE, hatch="", label=None, width=0.2):
    yerr = None if lo is None else [mean - lo, hi - mean]
    ax.bar(x, mean, width, color=color, edgecolor="black", linewidth=0.4, hatch=hatch, label=label,
           yerr=yerr, error_kw={"elinewidth": 0.6, "capsize": 1.2, "capthick": 0.6})


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")
