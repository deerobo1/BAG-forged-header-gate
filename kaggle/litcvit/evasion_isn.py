"""LitCVit barely falls to the header attack, yet collapses when the same fields are randomised. What does it read?

Scores the fine-tuned models saved by kaggle/litcvit_evasion (random split, seed 0) on CPU:
  clean, mimic_stack  as in that job, a check: must agree with its saved predictions
  isn                 every flow with new initial sequence numbers
  mimic_all           mimic_stack plus new initial sequence numbers
  stack_random        every shortcut field but sequence and acknowledgement numbers set to random bytes
  sequence_random     only sequence and acknowledgement numbers set to random bytes
  random              both, as in kaggle/litcvit_sa/sa.py
An attacker can do what the first four do. Random bytes show what a model reads, though no real stack
sends them. Evaded counts flows the attack changed: those with a donor for mimic_stack and mimic_all,
and every malware flow otherwise.

  python kaggle/litcvit/evasion_isn.py

Writes results/litcvit_isn.csv. About 15 min per model on USTC-TFC2016 on the laptop.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

import light as T

L = T.L
ROOT = Path(__file__).resolve().parent.parent.parent
JOB = Path("C:/data/CNS_proj/kaggle/litcvit_output_evasion")
RUNS, MODELS = ("mcfp_strict", "ustc"), ("litcvit", "litcvit-sa")
CONDITIONS = ("clean", "mimic_stack", "isn", "mimic_all", "stack_random", "sequence_random", "random")
BATCH = 128


def add_conditions(data, seed=0):
    """The conditions and attacked masks test_sets doesn't give; "random" stays as it made it."""
    images = data["conditions"]["clean"]
    flat = torch.from_numpy(images).reshape(len(images), L.PACKETS, L.ROW)
    every, stack = T.all_fields(flat), T.stack_fields(flat)
    gen = torch.Generator().manual_seed(seed)
    data["conditions"]["stack_random"] = L.randomize(flat, stack, gen).numpy()
    data["conditions"]["sequence_random"] = L.randomize(flat, every & ~stack, gen).numpy()
    malware = ~np.isin(np.array(data["classes"])[data["y_test"]], list(T.BENIGN))
    for name in ("isn", "stack_random", "sequence_random", "random"):
        data["attacked"][name] = malware


def logits_of(encoder, head, images):
    with torch.no_grad():
        return torch.cat([head(encoder.from_packets(encoder.packet_tokens(torch.as_tensor(images[i:i + BATCH]))))
                          for i in range(0, len(images), BATCH)]).numpy()


def main(runs=RUNS, models=MODELS, out=ROOT / "results" / "litcvit_isn.csv"):
    rows = []
    for run in runs:
        data = T.test_sets(run, "random", 0)
        add_conditions(data)
        for model in models:
            saved = np.load(JOB / "preds" / f"{run}_{model}.npz", allow_pickle=False)
            assert np.array_equal(saved["row"], data["rows"]) and np.array_equal(saved["attacked_same"], data["attacked"]["mimic_stack"]), run
            state = torch.load(JOB / "models" / f"{run}_{model}.pt", map_location="cpu", weights_only=True)
            encoder = L.Encoder().eval()
            encoder.load_state_dict(state["encoder"])
            head = L.classifier_head(L.D, len(state["classes"])).eval()
            head.load_state_dict(state["head"])
            logits = {}
            for name in CONDITIONS:
                logits[name] = logits_of(encoder, head, data["conditions"][name])
                if name == "clean":
                    data["clean_pred"] = logits[name].argmax(1)
                if name in saved.files:
                    agree = 100 * (logits[name].argmax(1) == saved[name].argmax(1)).mean()
                    assert agree > 99, (run, model, name, agree)
                    print(f"{run} {model} {name}: {agree:.2f}% of answers as on Kaggle", flush=True)
                rows.append({"dataset": run, "model": model, **T.score(data, name, logits[name])})
                print({k: v for k, v in rows[-1].items() if k != "recall"}, rows[-1]["recall"], flush=True)
                pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
