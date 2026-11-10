"""LitCVit fine-tuning ablation on 5GAD-2022 (Gate 2 follow-up, plan.txt Progress Oct 29).

Our LitCVit keeps the stem and byte block frozen in fine-tuning, and on 5GAD-2022 it
trails the paper by 7 accuracy points; kaggle/litcvit/probe.py shows the frozen packet
tokens lose what separates its attack classes. Each variant is fine-tuned as in
kaggle/litcvit/litcvit.py (random split, seed 0) except:
  full       pretrained encoder (version 2's encoder_random_0.pt); from FREEZE_EPOCHS on
             the whole encoder trains on the images, not only the packet and flow blocks.
             BatchNorm keeps its pretraining statistics, as in litcvit.py
  untrained  untrained encoder (seed 0, as in probe.py), fine-tuned exactly as litcvit.py
             does: does pretraining help at all?

Needs the private dataset deetyam/litcvit-encoder, which holds encoder_random_0.pt and a
copy of kaggle/litcvit/litcvit.py (imported from there):
  kaggle datasets create -p C:\\data\\CNS_proj\\kaggle\\litcvit-encoder
Writes /kaggle/working/ablation.csv (columns of litcvit.csv plus variant) and
preds/5gad2022_random_litcvit-<variant>_0.npz.

Push with:  kaggle kernels push -p kaggle/litcvit_ablation
"""

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score

FOUND = next(Path("/kaggle/input").rglob("encoder_random_0.pt"), None)
if FOUND is None:  # local test: model code from the repo, encoder from the downloaded output
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "litcvit"))
    ENCODER = Path("C:/data/CNS_proj/kaggle/litcvit_output_random/encoder_random_0.pt")
else:
    sys.path.insert(0, str(FOUND.parent))
    ENCODER = FOUND
import litcvit as L  # noqa: E402

RUN, KIND, SEED = "5gad2022", "random", 0
VARIANTS = ["untrained", "full"]
DEVICE = L.DEVICE


def finetune_full(pretrained, run, kind, seed):
    """litcvit.finetune, except that from FREEZE_EPOCHS on the whole encoder trains on the images."""
    meta, X = L.load(run)
    classes = sorted(meta["class"].unique())
    y_all = meta["class"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    train, test = L.split(meta, kind, np.random.default_rng(seed))

    encoder = copy.deepcopy(pretrained).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    images = torch.from_numpy(X).to(DEVICE)
    with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
        P = torch.cat([encoder.packet_tokens(images[i:i + 1024]).float() for i in range(0, len(X), 1024)])

    torch.manual_seed(seed)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    k = len(classes)
    head = L.classifier_head(L.D, k).to(DEVICE)
    opt = torch.optim.AdamW([*encoder.parameters(), *head.parameters()], L.FINETUNE_LR, weight_decay=L.WEIGHT_DECAY)
    steps = max(1, len(train) // L.FINETUNE_BATCH)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: L.lr_factor(s, steps * L.FINETUNE_EPOCHS))
    scaler = torch.amp.GradScaler(enabled=DEVICE == "cuda")
    y = torch.from_numpy(y_all).to(DEVICE)
    tr = torch.from_numpy(train).to(DEVICE)
    weight = torch.tensor(L.class_weights(y_all[train], k), dtype=torch.float32, device=DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    start = time.time()
    for epoch in range(L.FINETUNE_EPOCHS):
        frozen = epoch < L.FREEZE_EPOCHS
        if epoch == L.FREEZE_EPOCHS:
            for p in encoder.parameters():
                p.requires_grad_(True)
        head.train()
        total = torch.zeros((), device=DEVICE)
        for b in tr[L.balanced_indices(y[tr], steps * L.FINETUNE_BATCH, gen)].view(steps, -1):
            # frozen phase as in litcvit.py (fp32 on the cached tokens); then mixed precision on the images
            with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda" and not frozen):
                loss = loss_fn(head(encoder.from_packets(P[b]) if frozen else encoder(images[b])).float(), y[b])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.detach()
        if (epoch + 1) % 50 == 0 or epoch + 1 in (1, L.FINETUNE_EPOCHS):
            print(f"  {run} full fine-tune epoch {epoch + 1}: loss {total.item() / steps:.4f}, "
                  f"{time.time() - start:.0f}s", flush=True)

    head.eval()
    te = torch.from_numpy(test).to(DEVICE)
    with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
        logits = torch.cat([head(encoder(images[te[i:i + 1024]])).float() for i in range(0, len(te), 1024)])
    return meta["row"].to_numpy()[test], y_all[test], logits.cpu().numpy(), classes, len(train)


def main():
    (L.OUT / "preds").mkdir(exist_ok=True)
    print(f"data in {L.DATA}, encoder {ENCODER}, device {DEVICE}", flush=True)
    pretrained = L.Encoder().to(DEVICE)
    pretrained.load_state_dict(torch.load(ENCODER, map_location=DEVICE, weights_only=True))
    results = []
    for variant in VARIANTS:
        start = time.time()
        if variant == "full":
            rows, y, logits, classes, n_train = finetune_full(pretrained, RUN, KIND, SEED)
        else:
            torch.manual_seed(SEED)
            untrained = L.Encoder().to(DEVICE)
            rows, y, logits, classes, n_train = L.finetune(untrained, RUN, KIND, SEED)
            logits = logits["clean"]
        pred = logits.argmax(1)
        recall = recall_score(y, pred, labels=range(len(classes)), average=None, zero_division=0)
        row = {"dataset": RUN, "split": KIND, "model": "litcvit", "variant": variant, "seed": SEED,
               "train": n_train, "test": len(y),
               "accuracy": round(100 * accuracy_score(y, pred), 2),
               "macro_f1": round(100 * f1_score(y, pred, average="macro"), 2),
               "weighted_f1": round(100 * f1_score(y, pred, average="weighted"), 2),
               "seconds": round(time.time() - start),
               "recall": json.dumps({c: round(100 * r, 1) for c, r in zip(classes, recall)})}
        results.append(row)
        pd.DataFrame(results).to_csv(L.OUT / "ablation.csv", index=False)
        np.savez_compressed(L.OUT / "preds" / f"{RUN}_{KIND}_litcvit-{variant}_{SEED}.npz",
                            row=rows, y=y, logits=logits.astype(np.float32), classes=np.array(classes))
        print(f"{RUN} {variant:9s}: acc {row['accuracy']:6.2f}  macro-F1 {row['macro_f1']:6.2f}  "
              f"({row['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
