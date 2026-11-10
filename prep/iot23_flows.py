"""Make IoT-23 flow images, labelled from Zeek's conn.log.labeled.

Each scenario folder holds one pcap and bro/conn.log.labeled. Some are huge
port scans (36-1 alone is about 13 million one-packet connections), so the
Zeek log is read twice: once to count connections per (label, detailed-label),
then to keep at most --cap of each, chosen by a hash of the Zeek uid so reruns
keep the same ones. Rarer labels are kept whole. pcap_to_flows then makes
images only for flows that start inside a kept connection, tagged
"<label>|<detailed-label>" (e.g. "Malicious|PartOfAHorizontalPortScan").

A flow matches a Zeek connection with the same protocol and endpoints (either
direction) when it starts between the connection's start and end, give or
take --tolerance seconds; the latest such start wins. ICMP is matched on
addresses only, because Zeek stores ICMP type and code in the port fields.

Writes to --out, per scenario:
  <scenario>.npy / .csv     as pcap_to_flows, with the tag column filled
  <scenario>.labels.csv     per tag: Zeek connections, kept, flows written

Scenarios already done, or still downloading, are skipped, so the command can
be rerun as downloads finish.

Example:
  python prep/iot23_flows.py C:/data/CNS_proj/raw/IoT-23 --out C:/data/CNS_proj/flows/IoT-23
"""

import argparse
import collections
import csv
import socket
import sys
import time
import zlib
from pathlib import Path

import pcap_to_flows as ptf

PROTOS = {"tcp": ptf.TCP, "udp": ptf.UDP, "icmp": ptf.ICMP}
LOG_NAME = "conn.log.labeled"


def read_log(path):
    """Yield (uid, start, end, proto, orig_h, orig_p, resp_h, resp_p, tag) per connection."""
    cols = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                if line.startswith("#fields"):  # the last three are space-separated in IoT-23
                    cols = {name: i for i, name in enumerate(line.split()[1:])}
                    cols.setdefault("detailed-label", cols.get("det_label"))  # 2023 re-releases, e.g. 34-1
                continue
            parts = line.split()
            if (cols is None or len(parts) != len(cols) - ("det_label" in cols)
                    or parts[cols["proto"]] not in PROTOS
                    or not line.endswith("\n")):  # last line of a partly downloaded log
                continue
                continue
            start = float(parts[cols["ts"]])
            duration = parts[cols["duration"]]
            yield (parts[cols["uid"]], start, start + (float(duration) if duration != "-" else 0.0),
                   PROTOS[parts[cols["proto"]]], parts[cols["id.orig_h"]], int(parts[cols["id.orig_p"]]),
                   parts[cols["id.resp_h"]], int(parts[cols["id.resp_p"]]),
                   f"{parts[cols['label']]}|{parts[cols['detailed-label']]}")


def flow_key(proto, src, sport, dst, dport):
    """Direction-free key, as pcap_to_flows builds it; ICMP and ICMPv6 match on addresses."""
    if proto in (ptf.ICMP, ptf.ICMPV6):
        proto, sport, dport = ptf.ICMP, 0, 0
    a, b = (src, sport), (dst, dport)
    return (proto, a, b) if a <= b else (proto, b, a)


def packed(ip):
    return socket.inet_pton(socket.AF_INET6 if ":" in ip else socket.AF_INET, ip)


def index_log(path, cap):
    """Kept connections by flow key, and per tag: (Zeek connections, kept)."""
    counts = collections.Counter(row[-1] for row in read_log(path))
    threshold = {tag: min(1.0, cap / n) * 2**32 for tag, n in counts.items()}
    index, kept = collections.defaultdict(list), collections.Counter()
    for uid, start, end, proto, oh, op, rh, rp, tag in read_log(path):
        if zlib.crc32(uid.encode()) < threshold[tag]:
            index[flow_key(proto, packed(oh), op, packed(rh), rp)].append((start, end, tag))
            kept[tag] += 1
    return index, {tag: (n, kept[tag]) for tag, n in counts.items()}


def make_select(index, tolerance):
    def select(ts, proto, src, sport, dst, dport):
        best = None
        for start, end, tag in index.get(flow_key(proto, src, sport, dst, dport), ()):
            if start - tolerance <= ts <= end + tolerance and (best is None or start > best[0]):
                best = (start, tag)
        return best[1] if best else None
    return select


def find_inputs(folder):
    """(pcap, log) of a scenario, or None while either is missing or still downloading."""
    pcaps = [p for p in folder.rglob("*") if p.is_file() and ptf.is_capture(p)]
    logs = list(folder.rglob(LOG_NAME))
    if len(pcaps) != 1 or len(logs) != 1:
        return None
    return pcaps[0], logs[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw", type=Path, help="folder of IoT-23 scenario folders")
    parser.add_argument("--out", type=Path, required=True, help="output folder")
    parser.add_argument("--cap", type=int, default=100_000,
                        help="Zeek connections kept per label per scenario (default 100,000)")
    parser.add_argument("--tolerance", type=float, default=0.5, help="seconds of slack when matching (default 0.5)")
    parser.add_argument("--timeout", type=float, default=120.0, help="flow timeout, as pcap_to_flows")
    parser.add_argument("--overwrite", action="store_true", help="redo scenarios that are already done")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    for folder in sorted(p for p in args.raw.iterdir() if p.is_dir()):
        stem = args.out / folder.name
        if (not args.overwrite and ptf.sibling(stem, ".npy").exists()
                and ptf.sibling(stem, ".labels.csv").exists()):
            print(f"{folder.name}: already done, skipped")
            continue
        inputs = find_inputs(folder)
        if inputs is None:
            print(f"{folder.name}: pcap or {LOG_NAME} missing or still downloading, skipped")
            continue
        pcap, log = inputs
        start = time.time()
        index, counts = index_log(log, args.cap)
        print(f"{folder.name}: {sum(n for n, _ in counts.values()):,} Zeek connections, "
              f"{sum(k for _, k in counts.values()):,} kept ({time.time() - start:.0f}s)", flush=True)
        packets, skipped, flows, tls, quic = ptf.process_capture(
            pcap, stem, args.timeout, select=make_select(index, args.tolerance))
        with open(ptf.sibling(stem, ".csv"), newline="") as f:
            written = collections.Counter(r["tag"] for r in csv.DictReader(f))
        with open(ptf.sibling(stem, ".labels.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tag", "zeek_connections", "kept", "flows"])
            for tag, (n, k) in sorted(counts.items(), key=lambda kv: -kv[1][0]):
                w.writerow([tag, n, k, written[tag]])
        print(f"{folder.name}: {packets:,} packets, {flows:,} flows written, {tls:,} TLS "
              f"({time.time() - start:.0f}s)", flush=True)
        for tag, (n, k) in sorted(counts.items(), key=lambda kv: -kv[1][0]):
            print(f"    {tag:45s} {n:>11,} conns {k:>9,} kept {written[tag]:>9,} flows")


if __name__ == "__main__":
    sys.exit(main())
