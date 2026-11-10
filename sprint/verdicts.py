"""The pre-registered pass marks (01_preregistration.md), applied to the Kaggle jobs' results. No GPU, seconds.

Per run, versus the current gate on the same run, split and seed: clean macro-F1 >= gate - 2, and one of
  (i)   worst evaded over (c)-(g) < 5, Benign recall >= gate - 5
  (ii)  Benign recall >= gate + 1, evaded (a) and (b) each <= max(gate + 0.5, 1)
  (iii) CICIoT time split only: certified > 90
ADOPT on >= 5 of 7 runs, MAYBE on 3-4, DROP otherwise.

  python sprint/verdicts.py

Reads C:/data/CNS_proj/kaggle/sprint_output_<job>/results; writes sprint/results/verdicts_runs.csv and verdicts.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUTS = Path("C:/data/CNS_proj/kaggle")
RESULTS = Path(__file__).resolve().parent / "results"
JOBS = {"seed0": "random_0", "seed1_light": "random_1", "seed2_light": "random_2", "time": "time_0"}
PACKET = list("cdefg")
SEEN = {"gate_B_A": "ce", "gate_B_S": "df", "C_A": "ce", "C_S": "df", "gate+C_A": "ce", "gate+C_S": "df"}  # family A: c, e; S: d, f


def table(job, tag):
    folder = OUTPUTS / f"sprint_output_{job}" / "results"
    if not folder.exists():
        return None
    parts = [pd.read_csv(folder / f"{n}_{tag}.csv") for n in ("e0", "trained", "c1") if (folder / f"{n}_{tag}.csv").exists()]
    t = pd.concat(parts).drop_duplicates(["run", "method"]).reset_index(drop=True)
    t["setting"] = job
    return t


def marks(t):
    gate = t[t.method == "gate"].set_index("run")
    rows = []
    for _, r in t[t.method != "gate"].iterrows():
        g = gate.loc[r.run]
        worst = np.nanmax([r[f"evaded_{c}"] for c in PACKET])
        f1_ok = r.clean_f1 >= g.clean_f1 - 2
        i = worst < 5 and r.benign_recall >= g.benign_recall - 5
        ii = (r.benign_recall >= g.benign_recall + 1
              and all(r[f"evaded_{c}"] <= max(g[f"evaded_{c}"] + 0.5, 1) for c in "ab"))
        iii = r.setting == "time" and r.run == "ciciot2022" and r.get("certified", np.nan) > 90
        seen = SEEN.get(r.method)
        rows.append({"setting": r.setting, "run": r.run, "method": r.method, "clean_f1": r.clean_f1, "d_f1": round(r.clean_f1 - g.clean_f1, 2),
                     "benign_recall": r.benign_recall, "d_benign": round(r.benign_recall - g.benign_recall, 2),
                     "certified": r.get("certified", np.nan), "evaded_a": r.evaded_a, "evaded_b": r.evaded_b,
                     "worst_packet": worst, "gate_worst_packet": np.nanmax([g[f"evaded_{c}"] for c in PACKET]),
                     "worst_seen": np.nanmax([r[f"evaded_{c}"] for c in seen]) if seen else np.nan,
                     "worst_heldout": np.nanmax([r[f"evaded_{c}"] for c in "cdef" if c not in seen]) if seen else np.nan,
                     "evaded_g": r.evaded_g, "evaded_u": r.evaded_u, "f1_ok": f1_ok, "i": i, "ii": ii, "iii": iii,
                     "pass": bool(f1_ok and (i or ii or iii))})
    return pd.DataFrame(rows)


def main():
    runs = pd.concat([marks(t) for job, tag in JOBS.items() if (t := table(job, tag)) is not None])
    RESULTS.mkdir(exist_ok=True)
    runs.to_csv(RESULTS / "verdicts_runs.csv", index=False)
    summary = runs.groupby(["method", "setting"]).agg(passed=("pass", "sum"), runs=("pass", "size"), f1_ok=("f1_ok", "sum"),
                                                       by_i=("i", "sum"), by_ii=("ii", "sum"), d_f1_min=("d_f1", "min"),
                                                       d_benign_min=("d_benign", "min"), d_benign_mean=("d_benign", "mean"),
                                                       worst_packet_max=("worst_packet", "max"), evaded_g_max=("evaded_g", "max"),
                                                       evaded_b_max=("evaded_b", "max")).reset_index()
    summary["verdict"] = np.where(summary.passed >= 5, "ADOPT", np.where(summary.passed >= 3, "MAYBE", "DROP"))
    summary.round(2).to_csv(RESULTS / "verdicts.csv", index=False)
    pd.set_option("display.width", 250)
    print(summary.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
