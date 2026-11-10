"""Header-only vs payload-only baselines on flow images (plan step 5, early look at step 8).

Asks whether a classifier needs the payload at all, or whether the IP/TCP
header bytes LitCVit keeps in its image already give the label away.

Each class is split two ways, with the same number of flows per class:
  random  flows drawn at random from the whole class, then split 80/20
          (the paper never states its split; this is the usual one)
  time    within each capture, the earliest 80% of flows (by first packet)
          are trained on and the latest 20% tested on

Each split is run on five inputs cut from the 5 x 320-byte rows:
  full        all 1,600 bytes
  header      bytes 0-79 of each row (IP + TCP/UDP header, addresses and ports zeroed)
  payload     bytes 80-319 of each row
  full_min    full, with every header byte zeroed except IP version/IHL, total
              length, protocol and TCP flags
  header_min  header, zeroed the same way
The _min inputs drop the fields that can fingerprint the sending host (TTL,
IP ID, checksums, TCP window and options, ...) but keep what an observer of
encrypted traffic legitimately sees: packet sizes, protocol and TCP flags.

Models are a random forest and, optionally, sklearn's histogram gradient
boosting as a stand-in for XGBoost (paper Table VII; not installed here). The
boosting is slow on 1,600 features (about 4 minutes on 3,300 flows), so it is
off by default; run real XGBoost on Kaggle for Table VII. Reports accuracy,
macro-F1, per-class recall, the share of test images that also occur
byte-for-byte in training, and the header bytes the forest relies on most.

Classes follow LitCVit (--dataset):
  ustc   second part of the capture name ("Malware__Zeus" -> Zeus), with all
         Benign captures merged into one class
  iot23  from each flow's Zeek tag (iot23_flows.py): Benign; Muhstik = any
         malicious flow of scenario 3-1, the Muhstik capture; otherwise
         HeartBeat (any C&C-HeartBeat*), C&C (other C&C*), PortScan
         (PartOfAHorizontalPortScan*) or Okiru (Okiru*). Other labels (DDoS,
         Attack, FileDownload) are left out. The paper names these six
         classes but not how it built them.
  ciciot CICIoT2022 by folder: 3-Interactions -> Benign, 6-Attacks/1-Flood
         -> Flood, 6-Attacks/2-RTSP Brute Force/Hydra or /Nmap -> Hydra, Nmap
         (names as pcap_to_flows writes them with --root at the dataset folder)
  5gad   from each flow's tag (fivegad_flows.py): the attack folder, under
         LitCVit's short names (FakeAMFDelete -> AMFDel, ...), or Normal
  mcfp   by capture folder: CTU-Normal-* -> Benign; malware captures 341-1
         TrojanDownloader, 345-1 Cobalt, 346-1 Dridex, 347-1 BitcoinMiner,
         157-1 Botnet (prep/lists/mcfp.txt says why these), counting only
         flows with the capture's infected host: the 2018 captures also hold
         other lab machines' multicast traffic
  tii    TII-SSRC-23 by folder: benign/* -> Benign; malicious/bruteforce,
         dos, information-gathering, mirai-botnet -> Bruteforce, DoS, Recon,
         Mirai (LitCVit's five classes, the 26 attack subtypes merged)

Example:
  python baselines/header_payload.py C:/data/CNS_proj/flows/USTC-TFC2016 --csv results/ustc_header_payload.csv
"""

import argparse
import collections
import csv
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score

PACKETS, ROW, HEADER = 5, 320, 80
INPUTS = ["full", "header", "payload", "full_min", "header_min"]
KEPT = [0, 2, 3, 9, 33]  # IPv4 version/IHL, total length, protocol; TCP flags

# Header byte offsets, assuming a 20-byte IPv4 header followed by TCP (or UDP).
FIELDS = [(0, "IP version/IHL"), (1, "IP TOS"), (2, "IP total length"), (4, "IP ID"),
          (6, "IP flags/fragment"), (8, "IP TTL"), (9, "IP protocol"), (10, "IP checksum"),
          (12, "IP addresses (zeroed)"), (20, "ports (zeroed)"), (24, "TCP seq / UDP length"),
          (26, "TCP seq / UDP checksum"), (28, "TCP ack"), (32, "TCP data offset"),
          (33, "TCP flags"), (34, "TCP window"), (36, "TCP checksum"), (38, "TCP urgent"),
          (40, "TCP options"), (60, "TCP options (long)")]


def field_name(offset):
    return [name for start, name in FIELDS if start <= offset][-1]


def ustc_class(stem, row):
    parts = stem.split("__")
    return "Benign" if parts[0] == "Benign" else parts[1]


IOT23_PREFIXES = [("C&C-HeartBeat", "HeartBeat"), ("C&C", "C&C"),
                  ("PartOfAHorizontalPortScan", "PortScan"), ("Okiru", "Okiru")]


def iot23_class(stem, row):
    label, detailed = row["tag"].split("|", 1)
    if label.lower() == "benign":
        return "Benign"
    if stem.endswith("Capture-3-1"):
        return "Muhstik"
    return next((name for prefix, name in IOT23_PREFIXES if detailed.startswith(prefix)), None)


def ciciot_class(stem, row):
    parts = stem.split("__")
    if parts[0] == "3-Interactions":
        return "Benign"
    return "Flood" if parts[1] == "1-Flood" else parts[2]  # Hydra or Nmap


FIVEGAD_NAMES = {"FakeAMFDelete": "AMFDel", "FakeAMFInsert": "AMFIns", "AMFLookingForUDM": "AMFUDM",
                 "CrashNRF": "CrashNRF", "GetAllNFs": "GetAllNFs", "GetUserData": "GetUData",
                 "randomAMFInsert": "rndAMFIns", "randomDataDump": "rndDump"}


def fivegad_class(stem, row):
    tag = row["tag"].split("|", 1)[0]
    return "Normal" if tag == "Normal" else FIVEGAD_NAMES.get(tag)


# capture -> (family, infected host from the capture's README; 157-1's has none, 10.0.2.15 is its only VM)
MCFP_MALWARE = {"341-1": ("TrojanDownloader", "192.168.1.134"), "345-1": ("Cobalt", "192.168.1.121"),
                "346-1": ("Dridex", "192.168.1.122"), "347-1": ("BitcoinMiner", "192.168.1.126"),
                "157-1": ("Botnet", "10.0.2.15")}


def mcfp_class(stem, row):
    folder = stem.split("__")[0]
    if folder.startswith("CTU-Normal-"):
        return "Benign"
    family, host = MCFP_MALWARE.get(folder.removeprefix("CTU-Malware-Capture-Botnet-"), (None, None))
    return family if host in (row["src"], row["dst"]) else None


TII_NAMES = {"bruteforce": "Bruteforce", "dos": "DoS", "information-gathering": "Recon",
             "mirai-botnet": "Mirai"}


def tii_class(stem, row):
    kind, category = stem.split("__")[-3:-1]  # ...__pcap__malicious__dos__syn_tcp_dos
    return "Benign" if kind == "benign" else TII_NAMES[category]


CLASSES = {"ustc": ustc_class, "iot23": iot23_class, "ciciot": ciciot_class,
           "5gad": fivegad_class, "mcfp": mcfp_class, "tii": tii_class}


def load_index(flows_dir, class_of):
    """[(capture stem, class of each flow or None to leave it out, first-packet time of each flow)]"""
    captures = []
    for f in sorted(flows_dir.glob("*.csv")):
        if f.name.endswith(".labels.csv"):
            continue
        with open(f, newline="") as fh:
            rows = list(csv.DictReader(fh))
        labels = np.array([class_of(f.stem, r) for r in rows], dtype=object)
        captures.append((f.stem, labels, np.array([float(r["ts_first"]) for r in rows])))
    return captures


def make_splits(captures, n_train, n_test, rng):
    """{split: (train, test)}, each a list of (class, [[capture, flow], ...]) per class."""
    pools = collections.defaultdict(lambda: collections.defaultdict(list))
    for c, (_, labels, ts) in enumerate(captures):
        for label in sorted(set(labels) - {None}):
            idx = np.flatnonzero(labels == label)
            order = idx[np.argsort(ts[idx], kind="stable")]
            cut = int(round(0.8 * len(order)))
            for name, part in (("all", order), ("early", order[:cut]), ("late", order[cut:])):
                pools[label][name].append(np.column_stack([np.full(len(part), c), part]))

    splits = {"random": ([], []), "time": ([], [])}
    for label in sorted(pools):
        p = {name: np.concatenate(parts) for name, parts in pools[label].items()}
        tr, te = min(n_train, len(p["early"])), min(n_test, len(p["late"]))
        both = rng.choice(len(p["all"]), tr + te, replace=False)
        splits["random"][0].append((label, p["all"][both[:tr]]))
        splits["random"][1].append((label, p["all"][both[tr:]]))
        splits["time"][0].append((label, p["early"][rng.choice(len(p["early"]), tr, replace=False)]))
        splits["time"][1].append((label, p["late"][rng.choice(len(p["late"]), te, replace=False)]))
    return splits


def gather(flows_dir, captures, parts, classes):
    """Images [N, 5, 320] and class indices for a list of (class, ids) parts."""
    ids = np.concatenate([p for _, p in parts])
    y = np.concatenate([np.full(len(p), classes.index(label)) for label, p in parts])
    X = np.empty((len(ids), PACKETS, ROW), np.uint8)
    for c in np.unique(ids[:, 0]):
        rows = np.flatnonzero(ids[:, 0] == c)
        rows = rows[np.argsort(ids[rows, 1])]
        images = np.load(flows_dir / f"{captures[c][0]}.npy", mmap_mode="r")
        X[rows] = images[ids[rows, 1]].reshape(-1, PACKETS, ROW)
    return X, y


def features(X, inp):
    """Flatten [N, 5, 320] images into one of the INPUTS."""
    if inp.endswith("_min"):
        X = X.copy()
        X[:, :, np.setdiff1d(np.arange(HEADER), KEPT)] = 0
        inp = inp[:-len("_min")]
    cols = {"full": slice(0, ROW), "header": slice(0, HEADER), "payload": slice(HEADER, ROW)}[inp]
    return np.ascontiguousarray(X[:, :, cols].reshape(len(X), -1))


def seen_share(train, test):
    """Share of test rows that occur byte-for-byte in train."""
    seen = {r.tobytes() for r in train}
    return float(np.mean([r.tobytes() in seen for r in test]))


def make_model(name, seed):
    if name == "rf":
        return RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight="balanced",
                                      random_state=seed)
    return HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=seed)


def header_importance(model, inp):
    """Forest importance summed over the 5 packets, per header byte offset."""
    per_row = model.feature_importances_.reshape(PACKETS, -1).sum(axis=0)
    return per_row[:HEADER] if inp == "full" else per_row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("flows", type=Path, help="folder of .npy/.csv pairs from pcap_to_flows.py")
    parser.add_argument("--dataset", choices=sorted(CLASSES), default="ustc", help="how flows get their class")
    parser.add_argument("--train-per-class", type=int, default=4000)
    parser.add_argument("--test-per-class", type=int, default=1000)
    parser.add_argument("--models", default="rf", help="comma-separated: rf, hgb (slow)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path, help="write one row per run here")
    args = parser.parse_args(argv)

    captures = load_index(args.flows, CLASSES[args.dataset])
    available = collections.Counter(label for _, labels, _ in captures for label in labels if label)
    classes = sorted(available)
    splits = make_splits(captures, args.train_per_class, args.test_per_class,
                         np.random.default_rng(args.seed))
    print(f"{len(captures)} captures, {len(classes)} classes "
          f"(flows available, train, test per split):", flush=True)
    for i, c in enumerate(classes):
        print(f"  {c:12s} {available[c]:>9,} {len(splits['time'][0][i][1]):>7,} {len(splits['time'][1][i][1]):>7,}")

    results, importances = [], {}
    for split, (train_parts, test_parts) in splits.items():
        X_train, y_train = gather(args.flows, captures, train_parts, classes)
        X_test, y_test = gather(args.flows, captures, test_parts, classes)
        for inp in INPUTS:
            A, B = features(X_train, inp), features(X_test, inp)
            seen = seen_share(A, B)
            for name in args.models.split(","):
                start = time.time()
                model = make_model(name, args.seed).fit(A, y_train)
                pred = model.predict(B)
                recall = recall_score(y_test, pred, labels=range(len(classes)), average=None, zero_division=0)
                row = {"split": split, "input": inp, "model": name, "train": len(A), "test": len(B),
                       "seen_in_train%": round(100 * seen, 1),
                       "accuracy": round(100 * accuracy_score(y_test, pred), 2),
                       "macro_f1": round(100 * f1_score(y_test, pred, average="macro"), 2),
                       "seconds": round(time.time() - start)}
                row.update({f"recall:{c}": round(100 * r, 1) for c, r in zip(classes, recall)})
                results.append(row)
                print(f"{split:6s} {inp:10s} {name:3s}  acc {row['accuracy']:6.2f}  "
                      f"macro-F1 {row['macro_f1']:6.2f}  test seen in train {row['seen_in_train%']:5.1f}%  "
                      f"({row['seconds']}s)", flush=True)
                if name == "rf" and inp in ("full", "header"):
                    importances[split, inp] = header_importance(model, inp)
                if name == "rf" and inp.startswith("full"):
                    imp = model.feature_importances_.reshape(PACKETS, ROW)
                    row["rf_header_share%"] = round(100 * imp[:, :HEADER].sum() / imp.sum(), 1)

    print("\nPer-class recall (%)")
    keys = [(r["split"], r["input"], r["model"]) for r in results]
    print(f"{'class':12s}" + "".join(f"{'/'.join(k):>20s}" for k in keys))
    for c in classes:
        print(f"{c:12s}" + "".join(f"{r[f'recall:{c}']:>20}" for r in results))

    print("\nForest importance on the full image that falls on header bytes:")
    for r in results:
        if "rf_header_share%" in r:
            print(f"  {r['split']}/{r['input']}: {r['rf_header_share%']}%")

    print("\nHeader bytes the forest relies on most (importance summed over 5 packets)")
    for (split, inp), imp in importances.items():
        top = np.argsort(imp)[::-1][:8]
        total = imp.sum()
        print(f"  {split}/{inp}: " + ", ".join(f"{o} {field_name(o)} {100 * imp[o] / total:.0f}%" for o in top))

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        columns = list(dict.fromkeys(k for r in results for k in r))
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, columns)
            writer.writeheader()
            writer.writerows(results)


if __name__ == "__main__":
    sys.exit(main())
