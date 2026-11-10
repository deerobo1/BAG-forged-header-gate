"""CPU latency of LitCVit per flow on this machine, against the paper's numbers (plan step 12).

The paper gives 2.67 ms per flow on a GTX 1080, 5.32 ms on a multi-threaded i5-7400 and
8.70 ms single-threaded, at d = 128 (0.32M parameters, 115.55M multiply-adds). We time
the encoder and classifier head (11 classes, as USTC-TFC2016) on one flow at a time,
with one thread, two (the performance cores of a 12th-gen laptop i3; spreading one flow
over the efficiency cores too is slower) and all of PyTorch's default threads, at d = 128
and the d = 192 our results use; also throughput on batches of 32 (batches of 256 are
slower per flow on a laptop: they spill out of cache). Weights are untrained: the time does
not depend on them. Packet parsing and flow assembly are not timed (nor are they in the
paper, as far as it says).

  python kaggle/litcvit/latency.py

Close other programs and keep the laptop on mains power. Writes results/litcvit_latency.csv.
"""

import time
import winreg
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import litcvit

ROOT = Path(__file__).resolve().parent.parent.parent
WARMUP, REPEATS, BATCH = 20, 300, 32


def cpu_name():
    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
    return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()


def times(model, x, repeats):
    with torch.inference_mode():
        for _ in range(WARMUP):
            model(x)
        out = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(x)
            out.append(time.perf_counter() - start)
    return np.array(out)


def main():
    threads = torch.get_num_threads()
    gen = torch.Generator().manual_seed(0)
    rows = []
    for d in (128, 192):
        encoder, _, head, macs = litcvit.model_size(d)
        model = torch.nn.Sequential(litcvit.Encoder(d), litcvit.classifier_head(d, 11)).eval()
        one = torch.randint(0, 256, (1, 40, 40), generator=gen, dtype=torch.uint8)
        batch = torch.randint(0, 256, (BATCH, 40, 40), generator=gen, dtype=torch.uint8)
        for n in (1, 2, threads):
            torch.set_num_threads(n)
            single, batched = 1000 * times(model, one, REPEATS), times(model, batch, REPEATS // 30)
            rows.append({"cpu": cpu_name(), "d": d, "threads": n, "parameters": encoder + head,
                         "macs_millions": round(macs / 1e6, 2),
                         "ms_per_flow_median": round(float(np.median(single)), 2),
                         "ms_per_flow_p95": round(float(np.percentile(single, 95)), 2),
                         f"flows_per_s_batch{BATCH}": round(BATCH / float(np.median(batched)))})
            print(rows[-1], flush=True)
    torch.set_num_threads(threads)
    pd.DataFrame(rows).to_csv(ROOT / "results" / "litcvit_latency.csv", index=False)
    print("wrote results/litcvit_latency.csv")


if __name__ == "__main__":
    main()
