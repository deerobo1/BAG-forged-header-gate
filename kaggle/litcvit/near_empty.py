"""What does the gate do when a flow leaves the content model nothing to read? Saved weights, on CPU.

Split payload (attacks.resegment) leaves a flow a few payload bytes, and added packets can push them out of the five a
model reads. Such flows look alike to the content model, so its answer comes from what the training mix taught it about
empty flows, not from the flow. Where that answer is Benign, the gate rests on the other model alone.

A rule can choose that answer instead: a TCP flow showing fewer than k payload bytes isn't cleared, whatever the models
say. It reads only lengths, which the shortcut fields can't change, so the guarantee stands. MobileNetV2 ce and mask, the
decision gate, random split, seed 0. For each k: Benign recall on flows as recorded, the share of each kind of flow the
rule stops clearing, and evaded under each attack outside the guarantee, on the flows limits.py scores, so k = 0 gives its
rows. Also how each model alone answers those flows, and what each answers on a blank image.

  python kaggle/litcvit/near_empty.py

Writes results/litcvit_near_empty.csv after each run, and skips runs already in it.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import attacks as A
import light as T
import limits as LIM

L = T.L
ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "results" / "litcvit_near_empty.csv"
KEPT = (0, 1, 6, 41, 129)  # no rule; no payload at all; the most 1-byte pieces can show; the most 8-byte pieces can; more


def main(out=OUT):
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    done = {r["dataset"] for r in rows}
    for run in L.RUNS:
        if run in done:
            continue
        data = T.test_sets(run, "random", 0)
        y, classes = data["y_test"], np.array(data["classes"])
        benign = np.isin(classes, list(T.BENIGN))
        malware = ~benign[y]
        images = data["conditions"]["clean"].reshape(-1, L.PACKETS, L.ROW)
        copied = data["conditions"]["mimic_stack"].reshape(images.shape)
        attacks = {"add_packets": A.add_packets(images, only=malware), "resegment_1": A.resegment(images, 1, only=malware),
                   "resegment_8": A.resegment(images, 8, only=malware), "mimic_add": A.add_packets(copied, only=malware),
                   "mimic_resegment": A.resegment(copied, 1, only=malware)}
        plain = LIM.light_model(run, "mobilenetv2", "ce", len(classes))
        content = LIM.light_model(run, *LIM.CONTENT, len(classes))
        blank = np.zeros((1, L.PACKETS, L.ROW), images.dtype)
        said = {"dataset": run, "plain_on_blank": classes[plain(blank).argmax(1)][0], "content_on_blank": classes[content(blank).argmax(1)][0]}
        clean_plain, clean_content = plain(images), content(images)
        clean = LIM.answers(clean_plain, clean_content, "decision", benign)
        kept = A.payload_kept(images)
        thin = lambda shown, k: (shown >= 0) & (shown < k)
        for k in KEPT:
            rows.append({**said, "k": k, "condition": "clean", "benign_recall": round(100 * (benign[clean] & ~thin(kept, k))[~malware].mean(), 1),
                         "benign_thin": round(100 * thin(kept, k)[~malware].mean(), 1), "malware_thin": round(100 * thin(kept, k)[malware].mean(), 1)})
        for name, x in attacks.items():
            m = (x != images).any((1, 2)) & malware & ~benign[clean]
            z_plain, z_content = plain(x), content(x)
            called = benign[LIM.answers(z_plain, z_content, "decision", benign)]
            shown = A.payload_kept(x)
            alone = {"plain_alone": round(100 * benign[z_plain.argmax(1)][m].mean(), 1), "content_alone": round(100 * benign[z_content.argmax(1)][m].mean(), 1)} if m.any() else {}
            for k in KEPT:
                rows.append({**said, **alone, "k": k, "condition": name, "attacked": int(m.sum()),
                             "evaded": round(100 * (called & ~thin(shown, k))[m].mean(), 1) if m.any() else float("nan")})
        pd.DataFrame(rows).to_csv(out, index=False)
        table = pd.DataFrame([r for r in rows if r["dataset"] == run])
        print(f"\n{run}: on a blank image plain says {said['plain_on_blank']}, content says {said['content_on_blank']}")
        print(table[table["condition"] == "clean"][["k", "benign_thin", "malware_thin", "benign_recall"]].to_string(index=False))
        print(table[table["condition"] != "clean"].pivot_table(index="condition", columns="k", values="evaded", sort=False).to_string())
        print(table[(table["condition"] != "clean") & (table["k"] == 0)][["condition", "attacked", "plain_alone", "content_alone"]].to_string(index=False), flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
