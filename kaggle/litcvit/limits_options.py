"""Which values inside TCP options move a model, and could a working connection carry them? Saved weights, on CPU.

limits.py's arbitrary_options writes random bytes into MSS, window scale and timestamps. Those are well-formed, but random
timestamps need the attacker's own stack at both ends and most random scales break RFC 7323 (see attacks.arbitrary).
attacks.ordinary_options keeps to values any stack may choose and an unmodified receiver accepts. All on the sample of
malware flows that limits.py's arbitrary_any uses:

  arbitrary             no option values changed
  random_<kind>         arbitrary, with random bytes in one kind of option
  ordinary_<kind>       arbitrary, with ordinary values in one kind
  arbitrary_options     arbitrary, with random bytes in all three
  arbitrary_ordinary    arbitrary, with ordinary values in all three
  ordinary_only         ordinary values in all three, everything else as recorded
  arbitrary_any         arbitrary_options, a flow counting as evaded if any of 8 draws is called Benign
  ordinary_any          arbitrary_ordinary, likewise

All of these change only fields the content model never reads, and the script checks on the real flows that its logits
stay the same. evaded: of the flows the attack changed that the method detects as recorded, the share called Benign.

  python kaggle/litcvit/limits_options.py

Writes results/litcvit_limits_options.csv after each model, and skips models already in it. limits.py's arbitrary_any
rows from its first run also counted flows the attack can't change, such as UDP, which understates them; replace_any()
puts this file's rows in their place. Run it only when limits.py isn't writing.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import attacks as A
import light as T
import limits as LIM

L = T.L
ROOT = Path(__file__).resolve().parent.parent.parent
OUT, LIMITS = ROOT / "results" / "litcvit_limits_options.csv", ROOT / "results" / "litcvit_limits.csv"
KINDS = {"mss": A.MSS, "scale": A.SCALE, "stamps": A.STAMPS}
ORDER = ["mcfp_strict", "ustc", "5gad2022", "mcfp", "ciciot2022", "tii_ssrc23", "iot23"]  # the runs with plain LitCVit first


def attacks_on(x):
    """Each condition's images. The same generator seed for arbitrary keeps its own fields the same in all of them."""
    base = lambda: np.random.default_rng(0)
    ordinary = lambda **k: A.ordinary_options(x, np.random.default_rng(1), **k)
    out = {"arbitrary": A.arbitrary(x, base()), "arbitrary_options": A.arbitrary(x, base(), options=True),
           "arbitrary_ordinary": A.arbitrary(ordinary(), base()), "ordinary_only": ordinary()}
    for name, kind in KINDS.items():
        out[f"random_{name}"] = A.arbitrary(x, base(), options=(kind,))
        out[f"ordinary_{name}"] = A.arbitrary(ordinary(kinds=(kind,)), base())
    return out


def draws_on(x):
    """Several tries per flow. arbitrary_any's are limits.py's, so its rows can be replaced one for one."""
    return {"arbitrary_any": [A.arbitrary(x, np.random.default_rng(100 + d), options=True) for d in range(LIM.DRAWS)],
            "ordinary_any": [A.arbitrary(A.ordinary_options(x, np.random.default_rng(200 + d)), np.random.default_rng(300 + d))
                             for d in range(LIM.DRAWS)]}


def row_of(run, name, gate, condition, reached, called):
    return {"dataset": run, "model": name, "gate": gate, "condition": condition, "attacked": int(reached.sum()),
            "evaded": round(100 * called[reached].mean(), 1) if reached.any() else float("nan")}


def main(out=OUT):
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    done = {(r["dataset"], r["model"]) for r in rows}
    for run in ORDER:
        heavy_first = sorted(LIM.models_of(run).items(), key=lambda m: m[1][0] is LIM.light_model)
        models = [(name, m) for name, m in heavy_first if (run, name) not in done]
        if not models:
            continue
        data = T.test_sets(run, "random", 0, mimicry=False)
        y, classes = data["y_test"], np.array(data["classes"])
        benign = np.isin(classes, list(T.BENIGN))
        images = data["conditions"]["clean"].reshape(-1, L.PACKETS, L.ROW)
        sample = np.random.default_rng(1).permutation(np.flatnonzero(~benign[y]))[:LIM.SAMPLE]
        x = images[sample]
        attacks, draws = attacks_on(x), draws_on(x)
        changed = {c: (v != x).any((1, 2)) for c, v in attacks.items()}
        reached = {c: np.any([(v != x).any((1, 2)) for v in vs], 0) for c, vs in draws.items()}
        content_net = LIM.light_model(run, *LIM.CONTENT, len(classes))
        content = content_net(x)
        for c, v in [*attacks.items(), *((c, vs[0]) for c, vs in draws.items())]:
            assert np.abs(content_net(v) - content).max() < 1e-3, (run, c)  # what the guarantee rests on
        print(f"{run}: {len(x)} malware flows, content model unmoved by all {len(attacks) + len(draws)} conditions", flush=True)
        for name, (build, args) in models:
            net = build(*args, len(classes)) if build is LIM.light_model else build(*args)
            clean_logits = net(x)
            logits = {c: net(v) for c, v in attacks.items()}
            tries = {c: [net(v) for v in vs] for c, vs in draws.items()}
            for gate in ["none", "decision"] + (["logits"] if "mobilenetv2" in name else []):
                detected = ~benign[LIM.answers(clean_logits, content, gate, benign)]
                for c, z in logits.items():
                    rows.append(row_of(run, name, gate, c, changed[c] & detected, benign[LIM.answers(z, content, gate, benign)]))
                for c, zs in tries.items():
                    called = np.any([benign[LIM.answers(z, content, gate, benign)] for z in zs], 0)
                    rows.append(row_of(run, name, gate, c, reached[c] & detected, called))
            pd.DataFrame(rows).to_csv(out, index=False)
            shown = {r["condition"]: r["evaded"] for r in rows if (r["dataset"], r["model"], r["gate"]) == (run, name, "none")}
            print(f"  {name:20s} " + "  ".join(f"{c} {shown[c]}" for c in ("arbitrary", "arbitrary_options", "arbitrary_ordinary", "arbitrary_any", "ordinary_any")), flush=True)
    print(f"wrote {out}")


def replace_any(options=OUT, limits=LIMITS):
    """Put the corrected any-of-8 rows into limits.py's file."""
    old, new = pd.read_csv(limits), pd.read_csv(options)
    new = new[new["condition"] == "arbitrary_any"].set_index(["dataset", "model", "gate"])
    keys = pd.MultiIndex.from_frame(old[["dataset", "model", "gate"]])
    hit = (old["condition"] == "arbitrary_any").to_numpy() & keys.isin(new.index)
    old.loc[hit, "attacked"] = new.loc[keys[hit], "attacked"].to_numpy()
    old.loc[hit, "evaded"] = new.loc[keys[hit], "evaded"].to_numpy()
    old.to_csv(limits, index=False)
    print(f"replaced {hit.sum()} of {(old['condition'] == 'arbitrary_any').sum()} any-of-8 rows")


if __name__ == "__main__":
    main()
