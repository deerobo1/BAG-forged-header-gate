"""Write the paper's result tables as LaTeX from the CSVs in results/, so no number is typed by hand.

  python paper/make_tables.py

Writes to paper/tables/:
  header_payload.tex  random forest macro-F1 per input and split (baselines/header_payload.py)
  datasets.tex        classes, flows available and flows used, and TLS share of
                      benign and attack flows, from the packed Kaggle dataset
                      (kaggle/pack_flows.py; TLS share is over the packed sample)
  table7.tex          XGBoost and 2D CNN macro-F1, mean over seeds, random and
                      time split (results/table7.csv from kaggle/table7), next to
                      the scores LitCVit's Table VII reports for the same models
  litcvit.tex         our LitCVit (results/litcvit.csv from kaggle/litcvit) next to its
                      Table III; time-split columns appear once those rows are in
  tls_bytes.tex       how much of the TLS flows' images is ciphertext (results/tls_bytes.csv
                      from prep/audit_tls_bytes.py)
  litcvit_tls.tex     recall on TLS flows per model and test condition (results/litcvit_tls.csv
                      from kaggle/litcvit/tls_flows.py); the payload job's conditions appear
                      once its rows are in
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TABLES = Path(__file__).resolve().parent / "tables"
PACK = Path("C:/data/CNS_proj/kaggle/litcvit-flows")
BENIGN = {"Benign", "Normal"}
# CSV stem -> name in the paper, in the paper's dataset order
DATASETS = {"ustc": "USTC-TFC2016", "ciciot2022": "CICIoT2022", "5gad2022": "5GAD-2022",
            "mcfp": "MCFP", "iot23": "IoT-23", "tii_ssrc23": "TII-SSRC-23"}
INPUTS = {"full": "Full", "header": "Header", "payload": "Payload", "full_min": "Full$_{\\min}$",
          "header_min": "Header$_{\\min}$"}
# Table VII runs: MCFP also under the strict rule (flows the infected host starts)
RUNS = {"ustc": "USTC-TFC2016", "ciciot2022": "CICIoT2022", "5gad2022": "5GAD-2022", "mcfp": "MCFP",
        "mcfp_strict": "MCFP$_{s}$", "iot23": "IoT-23", "tii_ssrc23": "TII-SSRC-23"}
# Macro-F1 that LitCVit's Table VII reports for its own XGBoost and 2D CNN runs (three datasets only)
REPORTED_T7 = {"ustc": (92.70, 89.97), "ciciot2022": (91.39, 83.67), "mcfp": (63.54, 64.54)}
# Accuracy and F1 that LitCVit's Table III reports for itself
REPORTED_T3 = {"ustc": (98.37, 98.32), "ciciot2022": (99.12, 99.11), "5gad2022": (97.24, 97.25),
               "mcfp": (97.67, 97.65), "iot23": (98.52, 98.11), "tii_ssrc23": (98.63, 98.05)}
MIN_TLS_FLOWS = 20  # a class needs this many TLS flows for a row of its own
# TLS test flows in the recall table: run, class, name in the paper
TLS_ROWS = [("mcfp_strict", "Cobalt", "Cobalt"), ("mcfp_strict", "Dridex", "Dridex"),
            ("mcfp_strict", "Benign", "MCFP$_{s}$ Benign"), ("ustc", "Htbot", "Htbot"), ("ustc", "Shifu", "Shifu"),
            ("ustc", "Benign", "USTC Benign")]
TLS_MODELS = {"litcvit": "LitCVit", "litcvit-sa": "Shortcut-aware"}
TLS_CONDITIONS = {"clean": "Clean", "random": "Rand.", "zero": "Zero", "nopayload": "No pay.",
                  "random_nopayload": "Rand., no pay."}


def header_payload():
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Random-forest macro-F1 (\%) by input and split (seed 0). "
             r"Header$_{\min}$ keeps only IP version, total length, protocol and TCP flags of each packet.}",
             r"\label{tab:header-payload}", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{l" + "r" * len(INPUTS) + "}", r"\toprule",
             "Dataset / split & " + " & ".join(INPUTS.values()) + r" \\", r"\midrule"]
    for stem, name in DATASETS.items():
        df = pd.read_csv(ROOT / "results" / f"{stem}_header_payload_seed0.csv")
        df = df[df["model"] == "rf"].set_index(["split", "input"])["macro_f1"]
        for split in ("random", "time"):
            cells = [f"{df[split, i]:.1f}" for i in INPUTS]
            label = name if split == "random" else ""
            lines.append(f"{label} {split} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace[2pt]")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def datasets():
    counts = pd.read_csv(PACK / "classes.csv")
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Datasets as used here: LitCVit's classes, labelled flows available and used "
             r"(at most 10{,}000 per class), and the share of benign and attack flows with a TLS record "
             r"in their first five packets. Max attack is the most-encrypted attack class.}",
             r"\label{tab:datasets}", r"\setlength{\tabcolsep}{2pt}", r"\scriptsize",
             r"\begin{tabular}{lrrrrrl}", r"\toprule",
             r"Dataset & Classes & Available & Used & \multicolumn{2}{c}{TLS \%} & Max attack (TLS \%) \\",
             r"\cmidrule(lr){5-6}", r" & & & & Benign & Attack & \\", r"\midrule"]
    for stem, name in DATASETS.items():
        c = counts[counts["dataset"] == stem]
        meta = pd.read_csv(PACK / f"{stem}.csv", usecols=["class", "tls"], keep_default_na=False)
        benign = meta["class"].isin(BENIGN)
        per_class = meta[~benign].groupby("class")["tls"].mean() * 100
        top = per_class.idxmax()
        lines.append(f"{name} & {len(c)} & {thousands(c['flows_available'].sum())} & "
                     f"{thousands(c['flows_kept'].sum())} & {100 * meta.loc[benign, 'tls'].mean():.1f} & "
                     f"{100 * meta.loc[~benign, 'tls'].mean():.1f} & "
                     f"{top.replace('&', chr(92) + '&')} ({per_class[top]:.1f}) \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def table7():
    df = pd.read_csv(ROOT / "results" / "table7.csv")
    stats = df.groupby(["dataset", "split", "model"])["macro_f1"].agg(["mean", "std"])
    seeds = df["seed"].nunique()
    max_sd = stats.loc[(slice(None), "random"), "std"].max()
    lines = [r"\begin{table}[t]", r"\centering",
             rf"\caption{{Macro-F1 (\%) of the baselines in LitCVit's Table VII: as reported there "
             rf"(three datasets only) and in our runs, mean over {seeds} seeds. Standard deviations on the "
             rf"random split are at most {max_sd:.1f} points; on the time split XGBoost gives the same "
             rf"result for every seed. MCFP$_{{s}}$ keeps only flows the infected host starts "
             rf"(Benign, Cobalt, Dridex).}}",
             r"\label{tab:table7}", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{Reported} & \multicolumn{2}{c}{Random} & \multicolumn{2}{c}{Time} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Dataset & XGB & CNN & XGB & CNN & XGB & CNN \\", r"\midrule"]
    for stem, name in RUNS.items():
        reported = [f"{v:.1f}" for v in REPORTED_T7[stem]] if stem in REPORTED_T7 else ["--", "--"]
        ours = [f"{stats.loc[(stem, split, model), 'mean']:.1f}"
                for split in ("random", "time") for model in ("xgb", "cnn")]
        lines.append(f"{name} & " + " & ".join(reported + ours) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def litcvit():
    df = pd.read_csv(ROOT / "results" / "litcvit.csv")
    stats = df.groupby(["dataset", "split"])[["accuracy", "macro_f1"]].mean()
    splits = [s for s in ("random", "time") if s in set(df["split"])]
    seeds = df["seed"].nunique()
    cols = 2 + 2 * len(splits)
    lines = [r"\begin{table}[t]", r"\centering",
             rf"\caption{{LitCVit as reported (Table III there) and as reimplemented here "
             rf"({'seed 0' if seeds == 1 else f'mean over {seeds} seeds'}), in \%. The paper does not say "
             rf"which F1 it reports; ours is macro-F1. MCFP$_{{s}}$ as in Table~\ref{{tab:table7}}.}}",
             r"\label{tab:litcvit}", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{l" + "r" * cols + "}", r"\toprule",
             r" & \multicolumn{2}{c}{Reported}"
             + "".join(rf" & \multicolumn{{2}}{{c}}{{{s.capitalize()}}}" for s in splits) + r" \\",
             "".join(rf"\cmidrule(lr){{{i}-{i + 1}}}" for i in range(2, cols + 2, 2)),
             r"Dataset" + r" & Acc. & F1" * (1 + len(splits)) + r" \\", r"\midrule"]
    for stem, name in RUNS.items():
        cells = [f"{v:.1f}" for v in REPORTED_T3[stem]] if stem in REPORTED_T3 else ["--", "--"]
        for split in splits:
            cells += ([f"{v:.1f}" for v in stats.loc[(stem, split)]] if (stem, split) in stats.index
                      else ["--", "--"])
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def tls_bytes():
    df = pd.read_csv(ROOT / "results" / "tls_bytes.csv")
    attack = df[(df["dataset"] != "all") & ~df["class"].isin(BENIGN) & (df["flows"] >= MIN_TLS_FLOWS)]
    totals = df[df["dataset"] == "all"].sort_values("class")
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{What the first five packets of TLS flows hold: payload bytes kept in the image, and "
             r"how many are plaintext (handshake, record headers) or ciphertext. Totals cover all six datasets.}",
             r"\label{tab:tls-bytes}", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"TLS flows of & Flows & Bytes/flow & Plain \% & Cipher \% & \shortstack{Flows with\\cipher \%} \\",
             r"\midrule"]
    for frame in (attack, totals):
        for _, r in frame.iterrows():
            name = r["class"].capitalize() if r["dataset"] == "all" else r["class"]
            lines.append(f"{name} & {thousands(int(r['flows']))} & {r['bytes_per_flow']:.0f} & {r['plaintext%']:.1f} & "
                         f"{r['ciphertext%']:.1f} & {r['flows_with_ciphertext%']:.1f} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def litcvit_tls():
    df = pd.read_csv(ROOT / "results" / "litcvit_tls.csv")
    df = df[df["job"] == ("payload" if "payload" in set(df["job"]) else "sa")]
    stats = df.set_index(["dataset", "class", "model", "condition"])
    conditions = [c for c in TLS_CONDITIONS if c in set(df["condition"])]
    env = "table" if len(conditions) <= 3 else "table*"
    cols = len(TLS_MODELS) * len(conditions)
    lines = [rf"\begin{{{env}}}[t]", r"\centering",
             r"\caption{Recall (\%) on TLS test flows, random split, seed 0: flows as recorded (Clean), with the "
             r"shortcut fields set to random bytes (Rand.) or zeroed (Zero)"
             + (r", and with the payload zeroed (No pay.)" if "nopayload" in conditions else "")
             + r". MCFP classes use the strict label rule.}",
             r"\label{tab:litcvit-tls}", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lr" + "r" * cols + "}", r"\toprule",
             r" & " + "".join(rf" & \multicolumn{{{len(conditions)}}}{{c}}{{{m}}}" for m in TLS_MODELS.values()) + r" \\",
             "".join(rf"\cmidrule(lr){{{i}-{i + len(conditions) - 1}}}" for i in range(3, cols + 3, len(conditions))),
             r"Class & Flows" + "".join(f" & {TLS_CONDITIONS[c]}" for c in conditions) * len(TLS_MODELS) + r" \\",
             r"\midrule"]
    for run, name, label in TLS_ROWS:
        cells = [f"{stats.loc[(run, name, m, c), 'recall']:.1f}" for m in TLS_MODELS for c in conditions]
        flows = int(stats.loc[(run, name, next(iter(TLS_MODELS)), "clean"), "flows"])
        lines.append(f"{label} & {flows} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", rf"\end{{{env}}}"]
    return "\n".join(lines) + "\n"


def thousands(n):
    return f"{n:,}".replace(",", "{,}")


def main():
    TABLES.mkdir(exist_ok=True)
    for name, make in (("header_payload", header_payload), ("datasets", datasets), ("table7", table7),
                       ("litcvit", litcvit), ("tls_bytes", tls_bytes), ("litcvit_tls", litcvit_tls)):
        (TABLES / f"{name}.tex").write_text(make(), encoding="utf-8")
        print(f"wrote {TABLES / f'{name}.tex'}")


if __name__ == "__main__":
    main()
