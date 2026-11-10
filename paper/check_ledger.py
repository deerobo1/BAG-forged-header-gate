"""Check every number in the paper against its source file.

    python paper/check_ledger.py

Part A recomputes every cell of the generated tables the paper includes (g_n2, g_bound, g_packets),
with code written separately from make_paper_tables.py, and compares it with the printed cell.
Part B checks each result number quoted in main.tex: the quoted phrase must occur in main.tex, and the
value recomputed from its source must appear inside that phrase exactly as printed.
Exit code 1 on any mismatch.
"""

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R3, RES, SPR = ROOT / "sprint3" / "results", ROOT / "results", ROOT / "sprint" / "results" / "kaggle"
TAB = ROOT / "paper" / "tables"
TEXT = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
ORDER = ["5gad2022", "mcfp", "mcfp_strict", "ciciot2022", "tii_ssrc23", "iot23", "ustc"]
FOUR = ["iot23", "mcfp_strict", "tii_ssrc23", "mcfp"]
bad = []


def csv(name, base=R3):
    return pd.read_csv(base / name)


def r(v, d):
    return f"{v:.{d}f}"


# ------------------------------------------------------------------ Part A: tables
def table_numbers(name):
    body = (TAB / name).read_text(encoding="utf-8").split(r"\midrule")[1].split(r"\bottomrule")[0]
    rows = []
    for line in body.strip().splitlines():
        cells = [re.sub(r"\(([a-z]+)\)", "", c.replace("{,}", "")) for c in line.rstrip("\\ ").split("&")[1:]]
        rows.append([n for c in cells for n in re.findall(r"-?\d+(?:\.\d+)?", c)])
    return rows


def compare(table, expected):
    got = table_numbers(table)
    if len(got) != len(expected):
        bad.append(f"{table}: {len(got)} rows printed, {len(expected)} expected")
        return 0
    for i, (g, e) in enumerate(zip(got, expected)):
        if g != e:
            bad.append(f"{table} row {i + 1}: printed {g} expected {e}")
    return sum(map(len, expected))


cells = 0
n2 = csv("summary_n2.csv")
n2m = n2[n2.setting == "mean"].set_index("run")
e4 = csv("e4_methods.csv")
rnd = e4[e4.split == "random"]
gc = rnd[rnd.method == "gate+C_A"].groupby("run").header_uncond_max.mean()
cells += compare("g_n2.tex", [[r(n2m.loc[x, c], 1) for c in ["F", "B", "gate", "gate+C"]]
                              + [r(n2m.loc[x, "hdr_B"], 2), r(n2m.loc[x, "hdr_gate"], 2), r(gc[x], 2)]
                              + [r(n2m.loc[x, "benign_recall_F"], 1), r(n2m.loc[x, "benign_recall_gate"], 1)] for x in ORDER])
for line, x in zip((TAB / "g_n2.tex").read_text().split(r"\midrule")[1].strip().splitlines(), ORDER):
    if not line.rstrip("\\ ").endswith("yes" if str(n2m.loc[x, "pass"]) == "True" else "no"):
        bad.append(f"g_n2.tex {x}: pass column")

eb = csv("e2_bound.csv")
bmiss = eb[eb.gate == "B"].set_index(["run", "split", "seed", "attack"]).uncond_pct
exp = []
for g in ["gate", "gate+C_A", "xgb_gate", "adv8_gate"]:
    for a, x in eb[eb.gate == g].groupby("attack"):
        bound = bmiss.loc[[tuple(k) for k in x[["run", "split", "seed", "attack"]].values]].values
        exp.append([str(len(x)), str(int((x.uncond_pct.values > bound + 1e-9).sum())), r(x.uncond_pct.max(), 2), r(bound.max(), 2)])
cells += compare("g_bound.tex", exp)

e6 = csv("summary_e6_packets_C_A.csv")
m6 = e6[e6.split == "random"].groupby("run")[["held_d", "held_f", "held_u", "held_L", "held_La", "held_H", "held_Ha", "held_worst", "recall_drop"]].mean()
cells += compare("g_packets.tex", [[r(m6.loc[x, c], 1) for c in m6.columns[:-1]] + [r(m6.loc[x, "recall_drop"], 2)] for x in ORDER])

# ------------------------------------------------------------------ Part B: numbers in the text
claims = []


def claim(phrase, value, source):
    claims.append((phrase, value, source))


# audit (motivation)
hp = pd.concat([pd.read_csv(p).assign(run=p.name.split("_header")[0]) for p in RES.glob("*_header_payload_seed0.csv")])
hdr = hp[(hp.split == "random") & (hp.input == "header") & (hp.model == "rf")].macro_f1
claim("at least 92\\% macro-F1 on every dataset", f"{int(hdr.min())}", "results/*_header_payload_seed0.csv (header, random, rf)")
ttl = csv("ustc_ttl_rule.csv", RES)
mal = ttl[ttl["class"] != "Benign"]
err = (mal.flows * mal["ttl32%"] / 100).sum() + (ttl[ttl["class"] == "Benign"].flows * (100 - ttl[ttl["class"] == "Benign"]["ttl32%"]) / 100).sum()
claim("100.00\\% accuracy", f"{100 * (1 - err / ttl.flows.sum()):.2f}", "results/ustc_ttl_rule.csv")
tb = csv("tls_bytes.csv", RES)
claim("96.8\\% are plaintext", f"{tb[(tb.dataset == 'all') & (tb['class'] == 'all attack')]['plaintext%'].iloc[0]:.1f}", "results/tls_bytes.csv")
ci = csv("ciciot2022_header_payload_seed0.csv", RES).query("input == 'full' and model == 'rf'").set_index("split").macro_f1
claim("from 99.6 to 57.2 macro-F1", f"{ci['random']:.1f} to {ci['time']:.1f}", "results/ciciot2022_header_payload_seed0.csv")
yt = (ROOT / "results" / "gate1_yatc_compare.txt").read_text(encoding="utf-8")
alls = [list(map(int, l.split()[1:])) for l in yt.splitlines() if l.startswith("all ")]  # images exact loose none identical trailer packets other
matched = sum(a[1] + a[2] for a in alls)
claim("11{,}877 YaTC images", f"{matched:,}".replace(",", "{,}"), "results/gate1_yatc_compare.txt")
claim("6{,}857 of them byte for byte", f"{sum(a[4] for a in alls):,}".replace(",", "{,}"), "results/gate1_yatc_compare.txt")
claim("padding (4{,}579), packet count (300) or otherwise (141)",
      f"({sum(a[5] for a in alls):,}), packet count ({sum(a[6] for a in alls)}) or otherwise ({sum(a[7] for a in alls)})".replace(",", "{,}", 1),
      "results/gate1_yatc_compare.txt")

# attack
n1 = csv("summary_n1.csv").set_index("Run")
claim("at least 59\\% of the malware it detected", "59" if n1.drop("USTC")["F min seed"].min() >= 59 else "FAIL", "summary_n1.csv")
claim("at least 59\\% of the malware F detects", "59" if n1.drop("USTC")["F min seed"].min() >= 59 else "FAIL", "summary_n1.csv")
u = rnd[(rnd.method == "F") & (rnd.run == "ustc")].header_cond_max
claim("1.9--32.7\\%", f"{u.min():.1f}--{u.max():.1f}", "e4_methods.csv")
x6 = n1.drop("CICIoT")["XGBoost worst header, mean"]
claim("67.7--100\\% on six runs", f"{x6.min():.1f}--{x6.max():.0f}", "summary_n1.csv")
claim("and 3.6\\% on CICIoT2022", f"{n1.loc['CICIoT', 'XGBoost worst header, mean']:.1f}", "summary_n1.csv")
lim = csv("litcvit_limits.csv", RES)
la = lambda run, cond, model="litcvit": lim[(lim.dataset == run) & (lim.model == model) & (lim.gate == "none") & (lim.condition == cond)].evaded
claim("lets 52--71\\% through", f"{la('ustc', 'arbitrary_any').iloc[0]:.0f}--{la('mcfp_strict', 'arbitrary_any').iloc[0]:.0f}", "results/litcvit_limits.csv")
claim("71.2\\% through on MCFP strict and 52.3\\%", f"{la('mcfp_strict', 'arbitrary_any').iloc[0]:.1f}\\% through on MCFP strict and {la('ustc', 'arbitrary_any').iloc[0]:.1f}",
      "results/litcvit_limits.csv")
claim("0.0\\% and 0.1\\% under donor copying", f"{la('mcfp_strict', 'mimic_stack').iloc[0]:.1f}\\% and {la('ustc', 'mimic_stack').iloc[0]:.1f}", "results/litcvit_limits.csv")
gmax = rnd[rnd.method == "gate"][["cond_a", "cond_b"]].max().max()
claim("at most 0.94\\% in any seed", f"{gmax:.2f}", "e4_methods.csv gate cond_a/cond_b")

# realizability
f = csv("e5_forge.csv")
lv = csv("e5_live.csv")
claim("300 malware flows per run", f"{f.forged_flows.min()}" if f.forged_flows.nunique() == 1 else "FAIL", "e5_forge.csv")
errs = [c for c in f.columns if c.startswith(("own_", "dpkt_", "scapy_"))]
claim("no checksum error in 7{,}500 forged packets", f"{f.forged_packets.sum():,}".replace(",", "{,}") if f[errs].values.sum() == 0 else "FAIL", "e5_forge.csv")
claim("250 of 250 forged handshakes", f"{lv.handshake_ok.sum()} of {lv.sessions.sum()}", "e5_live.csv")
claim("and 243 exchanged data", f"{lv.echo_ok.sum()}", "e5_live.csv")
claim("the 7 failures", f"{lv.sessions.sum() - lv.echo_ok.sum()}", "e5_live.csv")

# guarantee
e1 = csv("e1_invariance.csv")
claim("649{,}966 attacked flows", f"{e1.flows_changed.sum():,}".replace(",", "{,}") if e1.b_input_differs.sum() == 0 else "FAIL", "e1_invariance.csv")
claim("Over 28 settings", f"{e1.groupby(['run', 'split', 'seed']).ngroups}", "e1_invariance.csv")
claim("all 336 pre-registered cells", f"{len(eb)}" if eb.violation.sum() == 0 else "FAIL", "e2_bound.csv")
claim("252 gate cells", f"{len(eb[eb.gate != 'B'])}", "e2_bound.csv")
bm = eb[eb.gate == "B"].groupby("attack").uncond_pct.max()
gm = eb[eb.gate == "gate"].groupby("attack").uncond_pct.max()
claim("reaches 30.34\\% under (b) and (m)", f"{bm['b']:.2f}" if bm["b"] == bm["m"] == gm["b"] else "FAIL", "e2_bound.csv (B and gate rows)")
claim("and 47.21\\% under (a)", f"{bm['a']:.2f}", "e2_bound.csv (B rows)")
claim("at most 0.25\\% through in every setting", f"{eb[eb.gate == 'gate+C_A'].uncond_pct.max():.2f}", "e2_bound.csv")

# gate vs masking
claim("(worst $-0.72$)", f"{n2m['gate-F'].min():.2f}", "summary_n2.csv")
claim("within 0.72 macro-F1", f"{abs(n2m['gate-F'].min()):.2f}", "summary_n2.csv")
gb = (n2m.gate - n2m.B)[["iot23", "ustc", "mcfp"]]
claim("recovers 9--51 points", f"{gb.min():.0f}--{gb.max():.0f}", "summary_n2.csv")
claim("52 points of macro-F1 on IoT-23, of which the gate recovers 51",
      f"{n2m.loc['iot23', 'F'] - n2m.loc['iot23', 'B']:.0f} points of macro-F1 on IoT-23, of which the gate recovers {n2m.loc['iot23', 'gate'] - n2m.loc['iot23', 'B']:.0f}", "summary_n2.csv")
claim("IoT-23 (47.5 for B against 98.4)", f"{n2m.loc['iot23', 'B']:.1f} for B against {n2m.loc['iot23', 'gate']:.1f}", "summary_n2.csv")
claim("USTC-TFC2016 (87.9 against 96.9)", f"{n2m.loc['ustc', 'B']:.1f} against {n2m.loc['ustc', 'gate']:.1f}", "summary_n2.csv")
claim("MCFP (76.4 against 90.2)", f"{n2m.loc['mcfp', 'B']:.1f} against {n2m.loc['mcfp', 'gate']:.1f}", "summary_n2.csv")
per = n2[n2.setting != "mean"].groupby("setting")["pass"].apply(lambda s: int((s.astype(str) == "True").sum()))
words = {3: "three", 4: "four"}
claim("passes on three, three and four runs, and on three on the time split",
      f"{words[per['random0']]}, {words[per['random1']]} and {words[per['random2']]} runs, and on {words[per['time0']]}", "summary_n2.csv")
claim("failed: three of seven runs", words[int((n2m['pass'].astype(str) == 'True').sum())], "summary_n2.csv")
holm = csv("summary_n2_holm.csv")
claim("$p_\\text{Holm}=0.056$", f"{holm[(holm.run == 'mcfp') & (holm.vs == 'B') & (holm.seed == 1)].p_holm.iloc[0]:.3f}", "summary_n2_holm.csv")
br = n2m.benign_recall_F - n2m.benign_recall_gate
claim("0.02 (USTC-TFC2016) to 3.86 (TII-SSRC-23)", f"{br['ustc']:.2f} (USTC-TFC2016) to {br['tii_ssrc23']:.2f}", "summary_n2.csv")
claim("costs up to 3.86 points of benign recall", f"{br.max():.2f}", "summary_n2.csv")
claim("2.05 on IoT-23, 1.32 on MCFP and at most 0.21", f"{br['iot23']:.2f} on IoT-23, {br['mcfp']:.2f} on MCFP and at most {br.drop(['iot23', 'mcfp', 'tii_ssrc23']).max():.2f}", "summary_n2.csv")
xg, xa = (rnd[rnd.method == m].groupby("run").clean_f1.mean() for m in ("xgb_gate", "xgb"))
claim("(92.8--99.9, against 93.5--100.0 alone)", f"{xg.min():.1f}--{xg.max():.1f}, against {xa.min():.1f}--{xa.max():.1f}", "e4_methods.csv")
claim("at most 1.50\\% through, so", f"{eb[eb.gate == 'xgb_gate'].uncond_pct.max():.2f}", "e2_bound.csv")
adv = e4[(e4.method == "adv8") & (e4.split == "random") & (e4.seed == 0)].set_index("run").cond_b
claim("63.6\\% (MCFP) and 73.9\\% (MCFP strict)", f"{adv['mcfp']:.1f}\\% (MCFP) and {adv['mcfp_strict']:.1f}", "e4_methods.csv adv8")
ap = e4[(e4.method == "and_plain") & (e4.split == "random") & (e4.seed == 0)].set_index("run").uncond_b
claim("86--100\\% on four of seven runs", f"{ap[ap > 80].min():.0f}--{ap.max():.0f}\\% on {words[int((ap > 80).sum())]}", "e4_methods.csv and_plain")

# packets
claim("failed: four of seven runs", words[int(((m6.held_worst < 10) & (m6.recall_drop <= 1)).sum())], "summary_e6_packets_C_A.csv")
claim("99.7\\% through on 5GAD-2022, 54.4\\% on MCFP and 47.2\\%",
      f"{m6.loc['5gad2022', 'held_worst']:.1f}\\% through on 5GAD-2022, {m6.loc['mcfp', 'held_worst']:.1f}\\% on MCFP and {m6.loc['mcfp_strict', 'held_worst']:.1f}", "summary_e6_packets_C_A.csv")
claim("at or below 0.5\\% everywhere", f"{m6[['held_d', 'held_f']].max().max():.1f}", "summary_e6_packets_C_A.csv")
f1x = csv("summary_e7_cfull_C_full.csv").query("split == 'random'").groupby("run").held_worst.mean()
claim("5GAD-2022 at 98.2\\% and MCFP at 26.8\\% and raised IoT-23 to 18.1\\%",
      f"{f1x['5gad2022']:.1f}\\% and MCFP at {f1x['mcfp']:.1f}\\% and raised IoT-23 to {f1x['iot23']:.1f}", "summary_e7_cfull_C_full.csv")
f2x = csv("summary_e7_cfull_C_fix2.csv").query("split == 'random'").groupby("run")[["held_worst", "recall_drop"]].mean()
claim("MCFP to 7.5\\%", f"{f2x.loc['mcfp', 'held_worst']:.1f}", "summary_e7_cfull_C_fix2.csv")
claim("more than 1 point", "1" if f2x.loc["mcfp", "recall_drop"] > 1 else "FAIL", "summary_e7_cfull_C_fix2.csv")
claim("5GAD-2022 at 60.6\\%", f"{f2x.loc['5gad2022', 'held_worst']:.1f}", "summary_e7_cfull_C_fix2.csv")
claim("at most 0.05 points", f"{csv('summary_e3.csv').excess_over_bound.max():.2f}", "summary_e3.csv")

# splits
claim("30.34\\% through on CICIoT2022's time split", f"{e4[(e4.split == 'time') & (e4.method == 'gate') & (e4.run == 'ciciot2022')].header_uncond_max.iloc[0]:.2f}", "e4_methods.csv")
claim("at most 1.13\\% on random splits", f"{rnd[rnd.method == 'gate'].header_uncond_max.max():.2f}", "e4_methods.csv")
e8 = csv("summary_e8_group.csv")
gg = e8[(e8.split == "group") & (e8.method == "gate'")].set_index("run")
claim("11.0\\% on IoT-23, 7.2\\% on MCFP strict, 31.9\\% on TII-SSRC-23 and 33.3\\%",
      "\\% on ".join([f"{gg.loc['iot23', 'benign_recall']:.1f}", f"IoT-23, {gg.loc['mcfp_strict', 'benign_recall']:.1f}",
                       f"MCFP strict, {gg.loc['tii_ssrc23', 'benign_recall']:.1f}"]) + f"\\% on TII-SSRC-23 and {gg.loc['mcfp', 'benign_recall']:.1f}", "summary_e8_group.csv")
four = e8[(e8.split == "group") & e8.run.isin(FOUR)].benign_recall
claim("6--49\\% across every model", f"{four.min():.0f}--{four.max():.0f}", "summary_e8_group.csv")
rr = e8[(e8.split == "random") & (e8.method == "gate'")].benign_recall
claim("94.7--100\\% on random splits", f"{rr.min():.1f}--{rr.max():.0f}", "summary_e8_group.csv")
ok = ["5gad2022", "ciciot2022", "ustc"]
claim("at most 0.32\\% through", f"{gg.loc[ok, 'header_uncond'].max():.2f}", "summary_e8_group.csv")
fg = e8[(e8.split == "group") & (e8.method == "F'")].set_index("run").header_uncond
claim("12.5--100\\% for F", f"{fg[ok].min():.1f}--{fg[ok].max():.0f}", "summary_e8_group.csv")

# operating point and cost
cv = csv("summary_e9_curve.csv")
t5 = cv[cv.t == 0.5].pivot_table(index="run", columns="method", values="false_alarm")
claim("6.91\\% of benign flows on TII-SSRC-23, 5.47\\% on IoT-23 and 3.67\\%",
      f"{t5.loc['tii_ssrc23', 'gate_t']:.2f}\\% of benign flows on TII-SSRC-23, {t5.loc['iot23', 'gate_t']:.2f}\\% on IoT-23 and {t5.loc['mcfp', 'gate_t']:.2f}", "summary_e9_curve.csv")
claim("3.05\\%, 3.42\\% and 2.35\\%", f"{t5.loc['tii_ssrc23', 'F_t']:.2f}\\%, {t5.loc['iot23', 'F_t']:.2f}\\% and {t5.loc['mcfp', 'F_t']:.2f}", "summary_e9_curve.csv")
claim("at or below 0.61\\%", f"{cv[cv.method == 'gate_t'].uncond_b.max():.2f}", "summary_e9_curve.csv")
c9 = csv("e9_cost.csv")
fb = c9[c9.method == "F+B"].latency_ratio
claim("1.75--2.01 times", f"{fb.min():.2f}--{fb.max():.2f}", "e9_cost.csv")
claim("0.70\\,M parameters", f"{c9.params.min() / 1e6:.2f}", "e9_cost.csv")
claim("1{,}800 trees", f"{int(c9.c_trees.max()):,}".replace(",", "{,}"), "e9_cost.csv")

# alternatives that failed
mb = lim[(lim.condition == "arbitrary_any") & (lim.gate == "none")].groupby("model").evaded.max()
claim("up to 73.4\\% and 86.7\\%", f"{mb['mobilenetv2-cfnce']:.1f}\\% and {mb['mobilenetv2-oneside']:.1f}", "results/litcvit_limits.csv")
vr = pd.read_csv(ROOT / "sprint" / "results" / "verdicts_runs.csv").query("setting == 'seed0'").set_index(["run", "method"])
claim("fell to 55.6 on 5GAD-2022, against 91.7 for F", f"{vr.loc[('5gad2022', 'nand_wok'), 'clean_f1']:.1f} on 5GAD-2022, against {vr.loc[('5gad2022', 'plain'), 'clean_f1']:.1f}",
      "sprint/results/verdicts_runs.csv")
claim("38.2\\% on IoT-23", f"{vr.loc[('iot23', 'nand_wok'), 'benign_recall']:.1f}", "sprint/results/verdicts_runs.csv")
cf = pd.read_csv(SPR / "time" / "b2_conformal_time_0.csv").query("run == 'ciciot2022' and calibration == 'static_train_late20'")
claim("61.1\\% on CICIoT2022", f"{cf.coverage.iloc[0]:.1f}", "sprint/results/kaggle/time/b2_conformal_time_0.csv")

# limitations
gs0, gs = csv("litcvit_gate.csv", RES), csv("litcvit_gate_seeds.csv", RES)
sel = lambda d: d[(d.dataset == "mcfp_strict")]
ev = list(sel(gs0)[(sel(gs0).model == "mobilenetv2") & (sel(gs0).base == "ce") & (sel(gs0).gate == "none")].evaded_mimic_stack_other) + \
    list(sel(gs)[sel(gs).gate == "none"].evaded_mimic_stack_other)
gev = list(sel(gs0)[(sel(gs0).model == "mobilenetv2") & (sel(gs0).base == "ce") & (sel(gs0).gate == "decision")].evaded_mimic_stack_other) + \
    list(sel(gs)[sel(gs).gate == "decision"].evaded_mimic_stack_other)
claim("43.8--45.4\\% of F's detections across three seeds and 0.0\\%",
      f"{min(ev):.1f}--{max(ev):.1f}\\% of F's detections across {words.get(len(ev), len(ev))} seeds and {max(gev):.1f}", "results/litcvit_gate*.csv")

for phrase, value, source in claims:
    if phrase not in TEXT:
        bad.append(f"text: phrase not found: '{phrase}' ({source})")
    elif value not in phrase:
        bad.append(f"text: '{phrase}' but source gives '{value}' ({source})")

if __name__ == "__main__":
    print(f"Part A: {cells} table cells recomputed from CSVs")
    print(f"Part B: {len(claims)} quoted numbers in the text")
    for b in bad:
        print("MISMATCH", b)
    print("mismatches:", len(bad))
    sys.exit(1 if bad else 0)
