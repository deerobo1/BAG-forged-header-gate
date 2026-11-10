"""Pack labelled flow images from all six datasets into one folder for a private Kaggle dataset.

Training runs on Kaggle (plan steps 5 and 6), so only labelled flows go up. For
each dataset this keeps the flows that get a LitCVit class (the rules are in
baselines/header_payload.py) and at most --cap flows per class, drawn at random
with a fixed seed. A random draw keeps every part of each capture, so the time
split still works. Writes, flat in --out:
  <name>.npy   uint8 [N, 40, 40], ordered by capture, then flow
  <name>.csv   one row per image: class, capture, then the pcap_to_flows.py
               columns (real addresses and ports, first-packet time, packets,
               tls, quic, tag)
MCFP also gets a strict column: 1 if the flow keeps its class when only
flows the infected host starts count (datasets.md section 4); Benign is always 1.

Example:
  python kaggle/pack_flows.py --out C:/data/CNS_proj/kaggle/litcvit-flows --kaggle-id <user>/litcvit-flows
  kaggle datasets create -p C:/data/CNS_proj/kaggle/litcvit-flows
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baselines"))
from header_payload import CLASSES, MCFP_MALWARE, load_index  # noqa: E402

FLOWS = Path("C:/data/CNS_proj/flows")
# output name -> (folder under --flows, class rule)
DATASETS = {"ustc": ("USTC-TFC2016", "ustc"), "ciciot2022": ("CICIoT2022", "ciciot"),
            "iot23": ("IoT-23", "iot23"), "5gad2022": ("5GAD-2022", "5gad"),
            "mcfp": ("MCFP", "mcfp"), "tii_ssrc23": ("TII-SSRC-23", "tii")}


def mcfp_strict(stem, row, label):
    if label == "Benign":
        return 1
    _, host = MCFP_MALWARE[stem.split("__")[0].removeprefix("CTU-Malware-Capture-Botnet-")]
    return int(row["src"] == host)


def pack(name, flows_dir, rule, cap, rng, out):
    captures = load_index(flows_dir, CLASSES[rule])
    picked = []  # [capture, flow] pairs
    counts = {}
    for label in sorted({x for _, labels, _ in captures for x in labels} - {None}):
        ids = np.concatenate([np.column_stack([np.full(len(k), c), k])
                              for c, (_, labels, _) in enumerate(captures)
                              for k in [np.flatnonzero(labels == label)]])
        counts[label] = (len(ids), min(len(ids), cap))
        if len(ids) > cap:
            ids = ids[rng.choice(len(ids), cap, replace=False)]
        picked.append(ids)
    picked = np.concatenate(picked)
    picked = picked[np.lexsort((picked[:, 1], picked[:, 0]))]

    images = np.empty((len(picked), 40, 40), np.uint8)
    with open(out / f"{name}.csv", "w", newline="") as fh:
        writer = None
        for c in np.unique(picked[:, 0]):
            stem, labels, _ = captures[c]
            rows_at = np.flatnonzero(picked[:, 0] == c)
            flows = picked[rows_at, 1]
            images[rows_at] = np.load(flows_dir / f"{stem}.npy", mmap_mode="r")[flows]
            with open(flows_dir / f"{stem}.csv", newline="") as f:
                rows = list(csv.DictReader(f))
            for i in flows:
                row = {"class": labels[i], "capture": stem, **rows[i]}
                if rule == "mcfp":
                    row["strict"] = mcfp_strict(stem, rows[i], labels[i])
                if writer is None:
                    writer = csv.DictWriter(fh, list(row))
                    writer.writeheader()
                writer.writerow(row)
    np.save(out / f"{name}.npy", images)
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flows", type=Path, default=FLOWS, help="folder holding one folder per dataset")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=10_000, help="most flows kept per class")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", help="comma-separated dataset names to pack (default all)")
    parser.add_argument("--kaggle-id", help="owner/slug: also write dataset-metadata.json for the Kaggle CLI")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    names = args.only.split(",") if args.only else list(DATASETS)
    summary = []
    for name in names:
        folder, rule = DATASETS[name]
        counts = pack(name, args.flows / folder, rule, args.cap, np.random.default_rng(args.seed), args.out)
        kept = sum(k for _, k in counts.values())
        print(f"{name}: {kept:,} flows kept, {(args.out / f'{name}.npy').stat().st_size / 2**20:,.0f} MB", flush=True)
        for label, (available, k) in counts.items():
            print(f"  {label:18s} {available:>9,} {k:>7,}")
            summary.append([name, label, available, k])
    with open(args.out / "classes.csv", "w", newline="") as f:
        csv.writer(f).writerows([["dataset", "class", "flows_available", "flows_kept"]] + summary)

    if args.kaggle_id:
        meta = {"id": args.kaggle_id, "title": "LitCVit flow images", "licenses": [{"name": "other"}],
                "description": "40 x 40 flow images (first 5 packets x 320 bytes, addresses and ports zeroed) "
                               "from USTC-TFC2016, CICIoT2022, IoT-23, 5GAD-2022, MCFP and TII-SSRC-23, "
                               "for a re-evaluation of LitCVit. Each source keeps its own licence."}
        (args.out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    sys.exit(main())
