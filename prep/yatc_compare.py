"""Gate 1: compare our flow images with YaTC's released MFR images, byte by byte.

YaTC (github.com/NSSL-SJTU/YaTC) released USTC-TFC2016 and CICIoT2022 as
40 x 40 PNGs in the layout LitCVit says it follows. For each YaTC image this
finds the same flow among ours and reports where the bytes differ.

Matching: YaTC replaces IP addresses with random ones and zeroes ports, and
we zero both, so both sides have the addresses and ports blanked first (IPv4
bytes 12-23 of each packet row; IPv6 bytes 8-43). A YaTC image then matches
"exactly" when its whole first packet row equals one of ours, and "loosely"
when the first packet's IPv4 header fields other than addresses (and TCP
sequence and acknowledgement numbers) do; these ignore trailing bytes. IPv6
first packets match loosely on payload length, next header, hop limit and the
first 16 payload bytes. When
several of ours match, the one with the fewest differing bytes is compared.

Each matched pair is then:
  identical  every byte equal
  trailer    in each packet the differing bytes are one short run (at most
             24 bytes) that is zero in ours: YaTC keeps what follows the IP
             packet in the frame (Ethernet padding, an FCS), which we cut at
             the IP total length
  packets    different numbers of packets: YaTC's flows (SplitCap sessions)
             have no idle timeout, so they can join connections hours apart
  other      anything else
plus, per byte position, the share of pairs whose bytes disagree.

Example:
  python prep/yatc_compare.py C:/data/CNS_proj/raw/YaTC/YaTC_datasets/USTC-TFC2016_MFR C:/data/CNS_proj/flows/USTC-TFC2016
"""

import argparse
import collections
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROW = 320
HEADER = 80
MAX_CANDIDATES = 200
MAX_TRAILER = 24  # Ethernet padding of a 28-byte IP packet (18) plus a 4-byte FCS, rounded up


def blank_addresses(rows):
    """Zero IP addresses and ports in (n, 320) packet rows, in place."""
    v4 = rows[:, 0] >> 4 == 4
    v6 = rows[:, 0] >> 4 == 6
    rows[np.ix_(v4, np.arange(12, 24))] = 0
    rows[np.ix_(v6, np.arange(8, 44))] = 0
    return rows


def loose_key(row):
    if row[0] >> 4 == 4:  # IPv4 length, ID, flags, TTL, protocol, checksum (+ TCP seq and ack)
        return bytes(row[2:12]) + (bytes(row[24:32]) if row[9] == 6 else b"")
    return bytes(row[4:8]) + bytes(row[HEADER:HEADER + 16])


def pair_kind(differ, ours, theirs):
    if not differ.any():
        return "identical"
    if ours.any(axis=1).sum() != theirs.any(axis=1).sum():
        return "packets"
    spans = [np.ptp(np.flatnonzero(row)) + 1 for row in differ if row.any()]
    if max(spans) <= MAX_TRAILER and not ours[differ].any():
        return "trailer"
    return "other"


def load_ours(flows):
    """(first rows, [(file, index)]) for every flow in a folder of pcap_to_flows outputs."""
    firsts, where = [], []
    for npy in sorted(flows.glob("*.npy")):
        images = np.load(npy, mmap_mode="r")
        if not len(images):
            continue
        firsts.append(np.array(images.reshape(len(images), -1)[:, :ROW]))
        where += [(npy, i) for i in range(len(images))]
    return blank_addresses(np.concatenate(firsts)), where


def load_yatc(folder):
    paths = sorted(folder.glob("*/*/*.png"))
    images = np.stack([np.asarray(Image.open(p), dtype=np.uint8).reshape(-1) for p in paths])
    return paths, images


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("yatc", type=Path, help="a YaTC *_MFR folder (train/ and test/ of class folders)")
    parser.add_argument("flows", type=Path, help="our pcap_to_flows output folder for the same dataset")
    args = parser.parse_args(argv)

    paths, yatc = load_yatc(args.yatc)
    print(f"YaTC: {len(paths):,} images in {args.yatc.name}")
    ours_first, where = load_ours(args.flows)
    print(f"ours: {len(where):,} flows in {args.flows}")

    exact = collections.defaultdict(list)
    loose = collections.defaultdict(list)
    for i, row in enumerate(ours_first):
        exact[row.tobytes()].append(i)
        loose[loose_key(row)].append(i)

    per_class = collections.defaultdict(collections.Counter)
    diff_bytes = np.zeros(5 * ROW, dtype=np.int64)
    packet_diff = collections.Counter()
    pairs = 0
    cache = {}
    for path, image in zip(paths, yatc):
        cls = path.parent.name
        rows = blank_addresses(image.reshape(5, ROW).copy())
        hits = exact.get(rows[0].tobytes())
        kind = "exact" if hits else None
        if not hits:
            hits = loose.get(loose_key(rows[0]))
            kind = "loose" if hits else "none"
        per_class[cls][kind] += 1
        if not hits:
            continue
        best = None
        for i in hits[:MAX_CANDIDATES]:  # several of ours can share a first row: take the closest
            npy, j = where[i]
            if npy not in cache:
                cache[npy] = np.load(npy, mmap_mode="r")
            ours = blank_addresses(np.array(cache[npy][j]).reshape(5, ROW))
            differ = ours.reshape(-1) != rows.reshape(-1)
            if best is None or differ.sum() < best[0].sum():
                best = differ, ours
        differ, ours = best
        diff_bytes += differ
        packet_diff[int(rows.any(axis=1).sum()) - int(ours.any(axis=1).sum())] += 1
        per_class[cls][pair_kind(differ.reshape(5, ROW), ours, rows)] += 1
        pairs += 1

    columns = ["exact", "loose", "none", "identical", "trailer", "packets", "other"]
    print(f"\n{'class':24s} {'images':>7s}" + "".join(f" {c:>9s}" for c in columns))
    total = sum(per_class.values(), collections.Counter())
    for cls, c in sorted(per_class.items()) + [("all", total)]:
        print(f"{cls:24s} {c['exact'] + c['loose'] + c['none']:7d}" + "".join(f" {c[k]:9d}" for k in columns))
    if not pairs:
        return
    print(f"\nYaTC packets minus ours, over {pairs:,} matched pairs: "
          + ", ".join(f"{d:+d}: {k:,}" for d, k in sorted(packet_diff.items())))
    share = diff_bytes.reshape(5, ROW) / pairs
    print("byte positions (packet, offset) that differ in more than 5% of matched pairs:")
    for p in range(5):
        offsets = np.flatnonzero(share[p] > 0.05)
        if len(offsets):
            runs, start = [], offsets[0]
            for a, b in zip(offsets, np.append(offsets[1:], -1)):
                if b != a + 1:
                    runs.append(f"{start}-{a}" if a > start else f"{a}")
                    start = b
            print(f"  packet {p + 1}: {', '.join(runs)}  (max {100 * share[p].max():.0f}%)")


if __name__ == "__main__":
    sys.exit(main())
