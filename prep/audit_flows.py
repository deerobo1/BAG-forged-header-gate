"""Encryption audit of flow images made by pcap_to_flows.py (plan step 4).

For each group of captures, reports:
  flows       number of flow images
  tls%/quic%  flows with a TLS/DTLS record or QUIC long header in the first 5 packets
  syn%        flows whose first packet is a bare TCP SYN (the flow starts at its handshake)
  5pkt%       flows with all 5 packets present
  pay%        flows with at least --min-payload real payload bytes in the image
  hient%      of those, flows whose payload looks random (byte entropy at least
              --ratio x that of uniform random bytes of the same length)

Payload length comes from the IP length fields, so the zero padding isn't counted.
Captures are grouped by the first --depth parts of their name ("Malware__Zeus").

Example:
  python prep/audit_flows.py C:/data/CNS_proj/flows/USTC-TFC2016 --csv audit.csv
"""

import argparse
import collections
import csv
import sys
from pathlib import Path

import numpy as np

PACKETS, ROW, HEADER = 5, 320, 80
TCP = 6
COLUMNS = ["group", "flows", "tls%", "quic%", "syn%", "5pkt%", "pay%", "hient%"]


def payload_and_syn(image):
    """Real payload bytes of the first 5 packets, and whether packet 1 is a bare SYN."""
    parts, syn = [], False
    for k, row in enumerate(image.reshape(PACKETS, ROW)):
        h = row[:HEADER]
        if not h.any():
            break
        if h[0] >> 4 == 4:
            ihl, total, proto = (h[0] & 0x0F) * 4, int(h[2]) << 8 | h[3], h[9]
        else:  # IPv6: fixed 40-byte header, payload length excludes it
            ihl, total, proto = 40, 40 + (int(h[4]) << 8 | h[5]), h[6]
        l4 = (h[ihl + 12] >> 4) * 4 if proto == TCP else 8
        if k == 0 and proto == TCP:
            syn = h[ihl + 13] & 0x12 == 0x02
        parts.append(row[HEADER:HEADER + max(0, min(ROW - HEADER, total - ihl - l4))])
    return (np.concatenate(parts) if parts else np.zeros(0, np.uint8)), syn


def entropy(b):
    p = np.bincount(b, minlength=256)
    p = p[p > 0] / len(b)
    return float(-(p * np.log2(p)).sum())


_rng = np.random.default_rng(0)
_random_entropy = {}


def random_entropy(n):
    """Mean entropy of n uniform random bytes; below 8 for small n (e.g. ~6.6 at 128)."""
    if n not in _random_entropy:
        _random_entropy[n] = np.mean([entropy(_rng.integers(0, 256, n, dtype=np.uint8)) for _ in range(20)])
    return _random_entropy[n]


def audit(flows_dir, depth, min_payload, ratio, sample):
    groups = collections.defaultdict(list)
    for f in sorted(flows_dir.glob("*.csv")):
        groups["__".join(f.stem.split("__")[:depth])].append(f.stem)

    rng = np.random.default_rng(0)
    for group, stems in groups.items():
        n = tls = quic = syn = five = pay = hi = 0
        for stem in stems:
            images = np.load(flows_dir / f"{stem}.npy", mmap_mode="r")
            with open(flows_dir / f"{stem}.csv", newline="") as f:
                rows = list(csv.DictReader(f))
            idx = rng.permutation(len(rows))[:sample] if sample else range(len(rows))
            for i in idx:
                r = rows[i]
                data, starts = payload_and_syn(np.asarray(images[i]))
                n += 1
                tls += r["tls"] == "1"
                quic += r["quic"] == "1"
                syn += starts
                five += int(r["packets"]) >= PACKETS
                if len(data) >= min_payload:
                    pay += 1
                    hi += entropy(data) >= ratio * random_entropy(len(data))
        pct = lambda x, d=n: round(100 * x / d, 1) if d else 0.0
        yield [group, n, pct(tls), pct(quic), pct(syn), pct(five), pct(pay), pct(hi, pay)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("flows", type=Path, help="folder of .npy/.csv pairs from pcap_to_flows.py")
    parser.add_argument("--depth", type=int, default=2, help="name parts that form a group")
    parser.add_argument("--min-payload", type=int, default=128, help="bytes needed for the entropy test")
    parser.add_argument("--ratio", type=float, default=0.9, help="fraction of random-byte entropy that counts as random")
    parser.add_argument("--sample", type=int, default=0, help="flows per capture to check (0 = all)")
    parser.add_argument("--csv", type=Path, help="also write the table here")
    args = parser.parse_args(argv)

    table = []
    print(f"{COLUMNS[0]:28s}" + "".join(f"{c:>9s}" for c in COLUMNS[1:]))
    for row in audit(args.flows, args.depth, args.min_payload, args.ratio, args.sample):
        table.append(row)
        print(f"{row[0]:28s}" + "".join(f"{v:>9}" for v in row[1:]), flush=True)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            csv.writer(f).writerows([COLUMNS] + table)


if __name__ == "__main__":
    sys.exit(main())
