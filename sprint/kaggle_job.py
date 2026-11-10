"""The whole sprint for one setting on a Kaggle GPU: baselines, the length view, every trained idea on all 7 runs, scores.

  python kaggle_job.py seed0 | seed1 | seed2 | time       add _light (seed1_light) for the baselines, the length view and
                                                         the analyses only, with no neural training: under 1 h

Inputs (found anywhere under /kaggle/input): litcvit-flows (classes.csv and the .npy files), the sprint code (this file,
next to kaggle/litcvit), and the existing weights (folders litcvit_output_ablate, _seeds, _gate). Writes results/,
models/ and logits/ under /kaggle/working. About 5-6 h on a T4 per setting.
"""

import os
import shutil
import sys
import time
from pathlib import Path

INPUT, WORK = Path("/kaggle/input"), Path("/kaggle/working")
weights = next(INPUT.rglob("litcvit_output_ablate"), None)
if weights is None:
    raise SystemExit("attach the sprint-weights dataset: no litcvit_output_ablate folder under /kaggle/input")
os.environ.update(SPRINT_WEIGHTS=str(weights.parent), SPRINT_CACHE="/kaggle/temp/cache",
                  SPRINT_MODELS=str(WORK / "models"), SPRINT_RESULTS=str(WORK / "results"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True  # /kaggle/input is read-only

import torch  # noqa: E402

import analysis_saved  # noqa: E402
import c1_length_view  # noqa: E402
import common as C  # noqa: E402
import e0_baselines  # noqa: E402
import score_trained  # noqa: E402
import train  # noqa: E402

SETTINGS = {"seed0": ("random", 0), "seed1": ("random", 1), "seed2": ("random", 2), "time": ("time", 0)}
IDEAS = ["B_A", "B_S", "nand", "ce_s1", "adv8", "B_dro", "nand_wok"]  # longest last, so a cut loses the least


def main(job):
    light = job.endswith("_light")
    split, seed = SETTINGS[job.removesuffix("_light")]
    print(f"{job}: {split} split, seed {seed}, device {C.L.DEVICE}, data in {C.L.DATA}", flush=True)
    if C.L.DEVICE != "cuda":
        raise SystemExit("no GPU: Settings > Accelerator > GPU T4 x2 (or P100)")
    start = time.time()
    e0_baselines.main(split, seed)
    c1_length_view.main(split, seed)
    analysis_saved.SETTINGS = [(split, seed)]
    analysis_saved.main()
    for name in ("d3_threshold", "a2_abstain", "b2_conformal"):  # main saves under one name per analysis
        for ext in (".json", ".csv"):
            src = C.RESULTS / (name + ext)
            if src.exists():
                src.rename(C.RESULTS / f"{name}_{split}_{seed}{ext}")
    for idea in [] if light else IDEAS:
        if idea == "B_dro" and split != "time":
            continue
        for run in C.L.RUNS:
            print(f"=== {idea} {run} at {(time.time() - start) / 3600:.2f} h", flush=True)
            train.train(idea, run, split, seed)
        score_trained.main(split, seed)
        shutil.copytree(C.CACHE / "logits", WORK / "logits", dirs_exist_ok=True)
    shutil.copytree(C.CACHE / "logits", WORK / "logits", dirs_exist_ok=True)
    print(f"{job}: all done in {(time.time() - start) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main(sys.argv[1])
