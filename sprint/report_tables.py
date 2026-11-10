"""Tables for sprint/REPORT.md and the paper, from the verdicts and the copied Kaggle results. No GPU, seconds.

  python sprint/verdicts.py && python sprint/report_tables.py

Writes sprint/tables/*.tex and prints the same tables as Markdown.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES, OUT = HERE / "results", HERE / "tables"
KAGGLE = RES / "kaggle"
NAMES = {"5gad2022": "5GAD", "ciciot2022": "CICIoT", "iot23": "IoT-23", "mcfp": "MCFP", "mcfp_strict": "MCFP$_{s}$",
         "tii_ssrc23": "TII", "ustc": "USTC"}
SEEDS = ["seed0", "seed1_light", "seed2_light"]
SETTINGS = {"seed0": "random_0", "seed1_light": "random_1", "seed2_light": "random_2", "time": "time_0"}


def latex(df, caption, label, path):
    cols = "l" + "r" * (df.shape[1] - 1)
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\setlength{\tabcolsep}{4pt}", rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             " & ".join(df.columns) + r" \\", r"\midrule"]
    lines += [" & ".join("--" if isinstance(v, float) and np.isnan(v) else f"{v:.1f}" if isinstance(v, float) else str(v)
                         for v in row) + r" \\" for row in df.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def markdown(df, title):
    print(f"\n### {title}\n")
    print("| " + " | ".join(df.columns) + " |")
    print("|" + "---|" * df.shape[1])
    for row in df.itertuples(index=False):
        print("| " + " | ".join("-" if isinstance(v, float) and np.isnan(v) else f"{v:.1f}" if isinstance(v, float) else
                                str(v).replace("$_{s}$", " strict") for v in row) + " |")


def results(kind):
    """All rows of <kind>_<split>_<seed>.csv over the four settings, with a setting column."""
    parts = []
    for job, tag in SETTINGS.items():
        f = KAGGLE / job / f"{kind}_{tag}.csv"
        if f.exists():
            parts.append(pd.read_csv(f).assign(setting=job))
    return pd.concat(parts)


def main_table(runs):
    """Gate vs gate + length view (add-trained): 3-seed mean on the random split, and the time split."""
    rows = []
    for run in NAMES:
        r = runs[(runs.run == run) & (runs.method == "gate+C_A")]
        rs, rt = r[r.setting.isin(SEEDS)], r[r.setting == "time"].iloc[0]
        rows.append({"Run": NAMES[run],
                     "Gate worst": rs.gate_worst_packet.mean(), "+C worst": rs.worst_packet.mean(),
                     "+C (u)": rs.evaded_u.mean(), "$\\Delta$F1": rs.d_f1.mean(), "$\\Delta$Ben.": rs.d_benign.mean(),
                     "Time: gate": rt.gate_worst_packet, "Time: +C": rt.worst_packet,
                     "Passes": f"{int(rs['pass'].sum())}/3, {'T' if rt['pass'] else '-'}"})
    return pd.DataFrame(rows)


def verdict_table(runs):
    """Runs passed per setting and the pre-registered verdict, for every method."""
    t = runs.pivot_table(index="method", columns="setting", values="pass", aggfunc="sum")
    t = t.reindex(columns=list(SETTINGS))
    mean = []
    for m in t.index:
        r = runs[(runs.method == m) & runs.setting.isin(SEEDS)]
        if r.setting.nunique() < 3:
            mean.append(np.nan)
            continue
        g = r.groupby("run")[["d_f1", "d_benign", "worst_packet"]].mean()
        mean.append(float(((g.d_f1 >= -2) & (g.worst_packet < 5) & (g.d_benign >= -5)).sum()))
    t["3-seed mean"] = mean

    def verdict(row):
        if not np.isnan(row["3-seed mean"]):  # finalists: 3-seed mean and the time split
            return "ADOPT" if row["3-seed mean"] >= 5 and row["time"] >= 5 else "MAYBE" if row["3-seed mean"] >= 3 else "DROP"
        best = row["seed0"]
        return "ADOPT" if best >= 5 and row["time"] >= 5 else "MAYBE" if best >= 3 else "DROP"
    t["Verdict"] = t.apply(verdict, axis=1)
    t = t.reset_index().rename(columns={"method": "Method", "seed0": "Seed 0", "seed1_light": "Seed 1",
                                        "seed2_light": "Seed 2", "time": "Time"})
    return t.sort_values(["Verdict", "Method"])


def baseline_table(runs):
    """Reviewer baselines vs the gate, seed 0: worst over runs of the 8-try header attack (b), and the F1 cost."""
    e0 = results("e0").query("setting == 'seed0'")
    tr = results("trained").query("setting == 'seed0'")
    both = pd.concat([e0, tr])
    rows = []
    for m, label in [("plain", "Plain"), ("and_plain", "AND of two plain"), ("adv8", "Adversarial best-of-8"),
                     ("gate", "Gate"), ("adv8_gate", "Adversarial + gate"), ("xgb", "XGBoost"), ("xgb_gate", "XGBoost + gate")]:
        r = both[both.method == m]
        if r.empty:
            continue
        d = runs[(runs.method == m) & (runs.setting == "seed0")]
        rows.append({"Method": label, "(a) max": r.evaded_a.max(), "(b) max": r.evaded_b.max(),
                     "$\\Delta$F1 min": d.d_f1.min() if len(d) else 0.0})
    return pd.DataFrame(rows)


def conformal_table():
    b = results("b2_conformal")
    rnd = b[b.setting.isin(SEEDS)].groupby("run").coverage.agg(["min", "max"])
    t = b[b.setting == "time"].pivot_table(index="run", columns="calibration", values="coverage", aggfunc="first")
    rows = []
    for run in NAMES:
        roll = t.loc[run, "rolling_test_firsthalf"]
        thr = b[(b.setting == "time") & (b.run == run) & (b.calibration == "rolling_test_firsthalf")].threshold.iloc[0]
        rows.append({"Run": NAMES[run], "Random min": rnd.loc[run, "min"], "Random max": rnd.loc[run, "max"],
                     "Time static": t.loc[run, "static_train_late20"],
                     "Time rolling": np.nan if np.isinf(thr) else roll})  # -inf: no malware to calibrate on
    return pd.DataFrame(rows)


def main():
    runs = pd.read_csv(RES / "verdicts_runs.csv")
    OUT.mkdir(exist_ok=True)
    m = main_table(runs)
    latex(m, "Malware let through (\\%) by the worst packet attack, (c)--(g), for the gate and the gate plus the length view "
             "C trained with added packets. Random split: mean of 3 seeds; (u) is the adaptive attacker with unseen "
             "parameters; $\\Delta$ is the change from the gate. Time: the split by time, seed 0.",
          "tab:length-view", OUT / "length_view.tex")
    markdown(m, "Gate vs gate + C_A")
    v = verdict_table(runs)
    markdown(v, "Runs passed (of 7) and verdicts")
    v.to_csv(RES / "verdicts_final.csv", index=False)
    b = baseline_table(runs)
    latex(b, "Reviewer baselines, random split, seed 0: the most malware let through (\\%) over the 7 runs by header copying "
             "(a) and by the best of 8 random header sets (b), and the largest drop in macro-F1 against the gate.",
          "tab:baselines", OUT / "baselines.tex")
    markdown(b, "Baselines, seed 0")
    c = conformal_table()
    latex(c, "Coverage (\\%) of malware by B's conformal threshold, target 95\\%. Random: min and max over 3 seeds. Time "
             "static: calibrated on the latest 20\\% of training malware; rolling: on the first half of the test period "
             "(-- where that half has no malware).", "tab:conformal", OUT / "conformal.tex")
    markdown(c, "Conformal coverage")


if __name__ == "__main__":
    main()
