"""Sources per class and the top source's share, per run: can a source-disjoint split exist?

  python sprint3/code/group_feasibility.py   -> sprint3/results/group_feasibility.csv
"""
import pandas as pd
from pathlib import Path

DATA = Path("C:/data/CNS_proj/kaggle/litcvit-flows")
OUT = Path(__file__).resolve().parents[1] / "results" / "group_feasibility.csv"
rows = []
for run in ["5gad2022", "ciciot2022", "iot23", "mcfp", "mcfp_strict", "tii_ssrc23", "ustc"]:
    m = pd.read_csv(DATA / f"{run.replace('_strict', '')}.csv", low_memory=False)
    if run == "mcfp_strict":
        m = m[m.strict == 1]
    for c, g in m.groupby("class"):
        share = g.src.value_counts(normalize=True)
        rows.append({"run": run, "class": c, "flows": len(g), "sources": g.src.nunique(),
                     "top_source_share": round(100 * share.iloc[0], 1),
                     "sources_for_80pct": int((share.cumsum() < 0.8).sum() + 1)})
pd.DataFrame(rows).to_csv(OUT, index=False)
print(pd.DataFrame(rows).to_string())
