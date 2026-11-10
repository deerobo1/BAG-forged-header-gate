"""Scores of the ideas trained by train.py, next to the current gate on the same cached suite.

  gate_B_A, gate_B_S   1B: F (the existing ce) gated by B trained with family A or S
  and_plain            1D(i): two plain models, cleared only if both say Benign (no header-blind member)
  adv8, adv8_gate      1D(ii): the adversarially trained model alone, and gated by the existing B
  nand, nand_wok       1A: the jointly trained F and B as a gate
  gate_B_dro           2C: F gated by the time-balanced B (time split)

  python sprint/score_trained.py <split> <seed> [runs...]

Writes sprint/results/trained_<split>_<seed>.json and .csv. Ideas not yet trained are skipped; rerun to add them.
"""

import sys

import common as C
import e0_baselines as E0
import train as TR

L = C.L
IDEAS = {"B_A": (True,), "B_S": (True,), "ce_s1": (False,), "adv8": (False,), "nand": (False, True), "nand_wok": (False, True),
         "B_dro": (True,)}  # idea -> masked flag of each saved network (F, B for the joint ones)


def trained(s, run, split, seed, idea):
    """Cached logits of an idea's networks, or None if it isn't trained yet."""
    key = f"{run}_{split}_{seed}_{idea}"
    names = ["_F", "_B"] if len(IDEAS[idea]) == 2 else [""]
    out = []
    for name, masked in zip(names, IDEAS[idea]):
        path = TR.MODELS / f"{key}{name}.pt"
        if not path.exists():
            return None
        out.append(C.logits(s, key + name, C.flownet(path, len(s["classes"]), masked), blind=masked))
    return out


def main(split="random", seed=0, runs=None):
    name = f"trained_{split}_{seed}"
    rows = []
    for run in runs or L.RUNS:
        s = C.suite(run, split, seed)
        F, B = E0.members(run, split, seed, s)
        base = {"run": run, "split": split, "seed": seed}
        rows.append({**base, **C.score(s, [F, B], (B,), name="gate")})
        rows.append({**base, **C.score(s, [F], (), name="plain")})
        for idea in IDEAS:
            nets = trained(s, run, split, seed, idea)
            if nets is None:
                continue
            if idea.startswith("B_"):
                methods = {f"gate_{idea}": ([F, nets[0]], (nets[0],))}
            elif idea == "ce_s1":
                methods = {"and_plain": ([F, nets[0]], ())}
            elif idea == "adv8":
                methods = {"adv8": ([nets[0]], ()), "adv8_gate": ([nets[0], B], (B,))}
            else:
                methods = {idea: (nets, (nets[1],)), idea + "_F": ([nets[0]], ())}
            for method, (members, blind) in methods.items():
                rows.append({**base, **C.score(s, members, blind, name=method)})
        print(f"{run}: {len(rows)} rows", flush=True)
    C.save(rows, name)
    import pandas as pd
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df[["run", "method", "clean_f1", "benign_recall", "certified", "evaded_a", "evaded_b", "evaded_c", "evaded_d",
              "evaded_e", "evaded_f", "evaded_g", "evaded_u"]].to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3:] or None)
