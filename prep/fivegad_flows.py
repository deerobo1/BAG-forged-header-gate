"""Make 5GAD-2022 flow images, labelled per attack folder or as Normal.

The repository (git clone with git-lfs) has Attacks/<attack>/ and
Normal-1UE/, Normal-2UE/. Every capture is an "allcap" file recorded on four
interfaces at once: lo (the 5G core's REST APIs on 127.0.0.x:8000, MongoDB),
upfgtp (user traffic leaving the UPF, raw IP, UE addresses 10.60.0.x), eno1
(the same user traffic after NAT, 192.168.70.x) and enp5s0 (the gNB link:
GTP-U on UDP 2152 and NGAP over SCTP, 192.168.9.x).

Attacks: each attack folder also has Attacks_<attack>.pcapng, which the
authors cut down to the attack packets. A flow of the folder's allcap file is
labelled with the attack when its endpoints appear in that file on a TCP
connection to an NF's REST port (8000); everything else in the attack
captures (MongoDB, NGAP, the host's own web traffic) is left out. The port
rule matters for FakeAMFDelete, whose Attacks_ file is a byte-identical copy
of its allcap file. The two PFCP attacks (automatedDropWithTimer,
automatedRedirectWithTimer) are single UDP flows and are skipped, as in
LitCVit's eight attack classes.

Normal: every flow, tagged "Normal|<interface>" with the interface guessed
from the addresses above, because the same user connection is recorded on
both upfgtp and eno1 (and inside the GTP-U tunnel on enp5s0).

Normal files may be prefixes (prep/lists/5gad.txt); LFS pointer files (not
downloaded) are skipped.

Example:
  python prep/fivegad_flows.py C:/data/CNS_proj/raw/5GAD-2022 --out C:/data/CNS_proj/flows/5GAD-2022
"""

import argparse
import collections
import csv
import sys
import time
from pathlib import Path

import pcap_to_flows as ptf
from iot23_flows import flow_key

SBI_PORT = 8000
SKIPPED_ATTACKS = {"automatedDropWithTimer", "automatedRedirectWithTimer"}
INTERFACES = [(bytes([127]), 1, "lo"), (bytes([10, 60]), 2, "upfgtp"),
              (bytes([192, 168, 70]), 3, "eno1"), (bytes([192, 168, 9]), 3, "enp5s0")]


def interface(src, dst):
    for prefix, n, name in INTERFACES:
        if src[:n] == prefix or dst[:n] == prefix:
            return name
    return "other"


def attack_keys(path):
    """Flow keys of the TCP connections to port 8000 in an Attacks_ capture."""
    keys = set()
    with ptf.open_capture(path) as f:
        for _, linktype, frame in ptf.read_packets(f):
            pkt = ptf.ip_packet(linktype, frame)
            parsed = ptf.parse_packet(pkt) if pkt is not None else None
            if parsed and parsed[0] == ptf.TCP and SBI_PORT in (parsed[2], parsed[4]):
                keys.add(flow_key(*parsed[:5]))
    return keys


def is_lfs_pointer(path):
    with open(path, "rb") as f:
        return f.read(24) == b"version https://git-lfs."


def jobs(raw):
    """Yield (output name, capture, select) for every capture to convert."""
    for folder in sorted((raw / "Attacks").iterdir()):
        if not folder.is_dir() or folder.name in SKIPPED_ATTACKS:
            continue
        keys = attack_keys(next(folder.glob("Attacks_*.pcapng")))
        name = folder.name
        yield (f"Attacks__{name}", next(folder.glob("allcap_*.pcapng")),
               lambda ts, proto, src, sport, dst, dport, keys=keys, name=name:
               name if flow_key(proto, src, sport, dst, dport) in keys else None)
    for folder in ("Normal-1UE", "Normal-2UE"):
        for path in sorted((raw / folder).glob("allcap_*.pcapng")):
            yield (f"{folder}__{path.stem}", path,
                   lambda ts, proto, src, sport, dst, dport: f"Normal|{interface(src, dst)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw", type=Path, help="the 5GAD git clone")
    parser.add_argument("--out", type=Path, required=True, help="output folder")
    parser.add_argument("--timeout", type=float, default=120.0, help="flow timeout, as pcap_to_flows")
    parser.add_argument("--overwrite", action="store_true", help="redo captures that are already done")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, path, select in jobs(args.raw):
        stem = args.out / name
        if is_lfs_pointer(path):
            print(f"{name}: not downloaded (LFS pointer), skipped")
            continue
        if not args.overwrite and ptf.sibling(stem, ".npy").exists() and ptf.sibling(stem, ".csv").exists():
            print(f"{name}: already done, skipped")
            continue
        start = time.time()
        packets, skipped, flows, tls, quic = ptf.process_capture(path, stem, args.timeout, select=select)
        with open(ptf.sibling(stem, ".csv"), newline="") as f:
            written = collections.Counter(r["tag"] for r in csv.DictReader(f))
        print(f"{name}: {packets:,} packets, {flows:,} flows, {tls:,} TLS, {quic:,} QUIC "
              f"({time.time() - start:.0f}s)  " + ", ".join(f"{t} {n:,}" for t, n in written.most_common()),
              flush=True)


if __name__ == "__main__":
    sys.exit(main())
