"""Turn pcap/pcapng captures into LitCVit-style 40 x 40 flow images.

A flow is a bidirectional 5-tuple session, split after --timeout seconds of
silence. Its first 5 packets become 5 x 320 bytes in YaTC's MFR layout:
80 bytes of IP + TCP/UDP header, then 240 bytes of payload, each truncated or
zero-padded. Missing packets are all zeros. The Ethernet header is dropped and
IP addresses and ports are zeroed in the image; the CSV keeps the real ones so
flows can be labelled and split by host or capture.

For each capture, writes to --out:
  <name>.npy  uint8 array [N, 40, 40]
  <name>.csv  one row per image (flow_id is the row in the .npy); the tag
              column is empty unless process_capture is given a select
              function (iot23_flows.py uses it to keep labelled flows only)

<name> is the capture's path relative to its input folder, with "/" replaced
by "__". Captures whose outputs already exist are skipped, so an interrupted
run can be restarted with the same command.

Example:
  python prep/pcap_to_flows.py C:/data/USTC-TFC2016 --out C:/data/flows/ustc
"""

import argparse
import bz2
import csv
import gzip
import ipaddress
import itertools
import lzma
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np

PACKETS = 5
HEADER_BYTES = 80
PAYLOAD_BYTES = 240
SIDE = 40  # 5 packets x 320 bytes = 1,600 = 40 x 40
EMPTY_ROW = bytes(HEADER_BYTES + PAYLOAD_BYTES)
SWEEP_EVERY = 100_000  # packets between scans for timed-out flows

CAPTURE_SUFFIXES = {".pcap", ".pcapng", ".cap"}
OPENERS = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open}

LINKTYPE_NULL, LINKTYPE_ETHERNET, LINKTYPE_RAW, LINKTYPE_LOOP = 0, 1, 101, 108
LINKTYPE_LINUX_SLL, LINKTYPE_IPV4, LINKTYPE_IPV6, LINKTYPE_LINUX_SLL2 = 113, 228, 229, 276
DLT_RAW = 12  # raw IP written unmapped, as 5GAD-2022's upfgtp interface is
IP_ETHERTYPES = (b"\x08\x00", b"\x86\xdd")
VLAN_ETHERTYPES = (b"\x81\x00", b"\x88\xa8")
PCAPNG_SHB = b"\x0a\x0d\x0d\x0a"

ICMP, TCP, UDP, ICMPV6 = 1, 6, 17, 58
TLS_VERSIONS = {b"\x03\x00", b"\x03\x01", b"\x03\x02", b"\x03\x03", b"\x03\x04", b"\xfe\xff", b"\xfe\xfd"}
QUIC_VERSIONS = {b"\x00\x00\x00\x01", b"\x6b\x33\x43\xcf"}

CSV_COLUMNS = ["flow_id", "ts_first", "proto", "src", "sport", "dst", "dport", "packets", "tls", "quic", "tag"]


# ---------------------------------------------------------------- reading

def open_capture(path):
    return OPENERS.get(path.suffix.lower(), open)(path, "rb")


def read_packets(f):
    """Yield (timestamp, linktype, frame) from a pcap or pcapng stream."""
    magic = f.read(4)
    if magic == PCAPNG_SHB:
        yield from _read_pcapng(f)
    else:
        yield from _read_pcap(f, magic)


def _read_pcap(f, magic):
    if magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"):
        endian = "<"
    elif magic in (b"\xa1\xb2\xc3\xd4", b"\xa1\xb2\x3c\x4d"):
        endian = ">"
    else:
        raise ValueError("not a pcap or pcapng file")
    scale = 1e-9 if magic in (b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d") else 1e-6
    linktype = struct.unpack(endian + "I", f.read(20)[16:20])[0] & 0xFFFF
    record = struct.Struct(endian + "IIII")
    while True:
        head = f.read(16)
        if len(head) < 16:
            return
        sec, frac, caplen, _ = record.unpack(head)
        frame = f.read(caplen)
        if len(frame) < caplen:
            return  # capture cut off mid-record
        yield sec + frac * scale, linktype, frame


def _read_pcapng(f):
    # Only Enhanced Packet Blocks carry packets we use; Simple and obsolete
    # Packet Blocks (no timestamp or rare) are skipped.
    endian = "<"
    interfaces = []  # (linktype, seconds per timestamp tick)
    block_type = PCAPNG_SHB
    while True:
        if block_type == PCAPNG_SHB:
            head = f.read(8)
            if len(head) < 8:
                return
            endian = "<" if head[4:8] == b"\x4d\x3c\x2b\x1a" else ">"
            f.read(struct.unpack(endian + "I", head[:4])[0] - 12)
            interfaces = []
        else:
            raw_length = f.read(4)
            if len(raw_length) < 4:
                return
            length = struct.unpack(endian + "I", raw_length)[0]
            body = f.read(length - 8)
            if len(body) < length - 8:
                return
            btype = struct.unpack(endian + "I", block_type)[0]
            if btype == 1:  # Interface Description Block
                linktype = struct.unpack(endian + "H", body[:2])[0]
                interfaces.append((linktype, _tsresol(body[8:-4], endian)))
            elif btype == 6:  # Enhanced Packet Block
                iface, high, low, caplen = struct.unpack(endian + "IIII", body[:16])
                if iface >= len(interfaces):
                    raise ValueError("packet refers to an undeclared interface")
                linktype, tick = interfaces[iface]
                yield ((high << 32) | low) * tick, linktype, body[20:20 + caplen]
        block_type = f.read(4)
        if len(block_type) < 4:
            return


def _tsresol(options, endian):
    """Seconds per timestamp tick from an Interface Description Block's options."""
    i = 0
    while i + 4 <= len(options):
        code, size = struct.unpack(endian + "HH", options[i:i + 4])
        if code == 0:
            break
        if code == 9 and size >= 1:  # if_tsresol
            value = options[i + 4]
            return 2.0 ** -(value & 0x7F) if value & 0x80 else 10.0 ** -value
        i += 4 + (size + 3) // 4 * 4
    return 1e-6


# ---------------------------------------------------------------- parsing

def ip_packet(linktype, frame):
    """Return the IP packet carried by a link-layer frame, or None."""
    if linktype == LINKTYPE_ETHERNET:
        offset, ethertype = 14, frame[12:14]
        while ethertype in VLAN_ETHERTYPES:
            ethertype = frame[offset + 2:offset + 4]
            offset += 4
        return frame[offset:] if ethertype in IP_ETHERTYPES else None
    if linktype in (LINKTYPE_RAW, LINKTYPE_IPV4, LINKTYPE_IPV6, DLT_RAW):
        return frame
    if linktype == LINKTYPE_LINUX_SLL:
        return frame[16:] if frame[14:16] in IP_ETHERTYPES else None
    if linktype == LINKTYPE_LINUX_SLL2:
        return frame[20:] if frame[0:2] in IP_ETHERTYPES else None
    if linktype in (LINKTYPE_NULL, LINKTYPE_LOOP):
        return frame[4:]
    return None


def parse_packet(pkt):
    """Split an IP packet into (proto, src, sport, dst, dport, pkt, l4, end).

    pkt is trimmed to the IP length (dropping link-layer padding), l4 is where
    the transport header starts and end is where the payload starts. The
    header covers TCP, UDP and the 8-byte ICMP/ICMPv6 header, as scapy (and so
    YaTC) splits them; other protocols get ports 0 and an IP-only header.
    Returns None for packets that can't belong to a flow: not IPv4/IPv6, too
    short, or a non-first IPv4 fragment. IPv6 extension headers are not walked.
    """
    if len(pkt) < 20:
        return None
    version = pkt[0] >> 4
    if version == 4:
        ihl = (pkt[0] & 0x0F) * 4
        if ihl < 20 or int.from_bytes(pkt[6:8], "big") & 0x1FFF:
            return None
        total = int.from_bytes(pkt[2:4], "big")
        if total >= ihl:  # 0 means segmentation offload: keep everything
            pkt = pkt[:total]
        proto, src, dst, l4 = pkt[9], pkt[12:16], pkt[16:20], ihl
    elif version == 6:
        if len(pkt) < 40:
            return None
        total = 40 + int.from_bytes(pkt[4:6], "big")
        if total > 40:
            pkt = pkt[:total]
        proto, src, dst, l4 = pkt[6], pkt[8:24], pkt[24:40], 40
    else:
        return None
    end = l4
    if proto == TCP and len(pkt) >= l4 + 20:
        end = l4 + max((pkt[l4 + 12] >> 4) * 4, 20)
    elif proto in (UDP, ICMP, ICMPV6) and len(pkt) >= l4 + 8:
        end = l4 + 8
    sport = dport = 0
    if proto in (TCP, UDP) and end > l4:
        sport = int.from_bytes(pkt[l4:l4 + 2], "big")
        dport = int.from_bytes(pkt[l4 + 2:l4 + 4], "big")
    return proto, src, sport, dst, dport, pkt, l4, end


def image_row(pkt, proto, l4, end):
    """One 320-byte image row: anonymised header (80 bytes), then payload (240)."""
    header = bytearray(pkt[:end])
    if pkt[0] >> 4 == 4:
        header[12:20] = bytes(8)
    else:
        header[8:40] = bytes(32)
    if proto in (TCP, UDP) and end > l4:
        header[l4:l4 + 4] = bytes(4)
    header = bytes(header[:HEADER_BYTES]).ljust(HEADER_BYTES, b"\0")
    return header + pkt[end:end + PAYLOAD_BYTES].ljust(PAYLOAD_BYTES, b"\0")


def looks_like_tls(payload):
    """TLS or DTLS record header: content type 20-23 and a known version."""
    return len(payload) >= 5 and 20 <= payload[0] <= 23 and payload[1:3] in TLS_VERSIONS


def looks_like_quic(payload):
    """QUIC long-header packet (Initial, Handshake, ...) with a known version."""
    return (len(payload) >= 5 and payload[0] & 0xC0 == 0xC0
            and (payload[1:5] in QUIC_VERSIONS or payload[1:4] == b"\xff\x00\x00"))


def ip_str(addr):
    return str(ipaddress.ip_address(addr))


# ---------------------------------------------------------------- flows

class Flow:
    __slots__ = ("proto", "src", "sport", "dst", "dport", "ts_first", "ts_last",
                 "rows", "tls", "quic", "emitted", "tag")

    def __init__(self, ts, proto, src, sport, dst, dport):
        self.proto, self.src, self.sport, self.dst, self.dport = proto, src, sport, dst, dport
        self.ts_first = self.ts_last = ts
        self.rows = []
        self.tls = self.quic = self.emitted = False
        self.tag = ""


def sibling(stem, ext):
    """stem + ext, without Path.with_suffix eating dotted names like 192.168.1.5."""
    return stem.with_name(stem.name + ext)


class FlowWriter:
    """Streams flow images to a raw temp file and metadata to a temp CSV."""

    def __init__(self, stem, limit=None):
        self.stem, self.limit = stem, limit
        self.raw_path = sibling(stem, ".u8.tmp")
        self.csv_path = sibling(stem, ".csv.tmp")
        self.raw = open(self.raw_path, "wb")
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv = csv.writer(self.csv_file)
        self.csv.writerow(CSV_COLUMNS)
        self.count = self.tls = self.quic = 0

    @property
    def full(self):
        return self.limit is not None and self.count >= self.limit

    def write(self, flow):
        flow.emitted = True
        if not self.full:
            self.raw.write(b"".join(flow.rows) + EMPTY_ROW * (PACKETS - len(flow.rows)))
            self.csv.writerow([self.count, f"{flow.ts_first:.6f}", flow.proto,
                               ip_str(flow.src), flow.sport, ip_str(flow.dst), flow.dport,
                               len(flow.rows), int(flow.tls), int(flow.quic), flow.tag])
            self.count += 1
            self.tls += flow.tls
            self.quic += flow.quic
        flow.rows = None  # the flow now only absorbs later packets with its key

    def close(self):
        """Finish both files and move them into place."""
        self.raw.close()
        self.csv_file.close()
        npy_tmp = sibling(self.stem, ".npy.tmp")
        with open(npy_tmp, "wb") as out, open(self.raw_path, "rb") as raw:
            np.lib.format.write_array_header_1_0(
                out, {"descr": "|u1", "fortran_order": False, "shape": (self.count, SIDE, SIDE)})
            shutil.copyfileobj(raw, out, 1 << 20)
        self.raw_path.unlink()
        self.csv_path.replace(sibling(self.stem, ".csv"))
        npy_tmp.replace(sibling(self.stem, ".npy"))

    def abort(self):
        self.raw.close()
        self.csv_file.close()
        self.raw_path.unlink(missing_ok=True)
        self.csv_path.unlink(missing_ok=True)


def process_capture(path, stem, timeout=120.0, max_active=1_000_000, max_flows=None, select=None):
    """Write <stem>.npy and <stem>.csv for one capture.

    select, if given, is called as select(ts, proto, src, sport, dst, dport)
    when a flow starts (addresses as packed bytes). It returns the flow's tag
    for the CSV, or None to leave the flow out.

    Returns (packets read, packets not usable, flows, TLS flows, QUIC flows).
    """
    active = {}  # key -> Flow, oldest first
    writer = FlowWriter(stem, max_flows)
    packets = skipped = 0
    try:
        with open_capture(path) as f:
            for ts, linktype, frame in read_packets(f):
                packets += 1
                pkt = ip_packet(linktype, frame)
                parsed = parse_packet(pkt) if pkt is not None else None
                if parsed is None:
                    skipped += 1
                    continue
                proto, src, sport, dst, dport, pkt, l4, end = parsed
                a, b = (src, sport), (dst, dport)
                key = (proto, a, b) if a <= b else (proto, b, a)

                flow = active.get(key)
                if flow is not None and timeout and ts - flow.ts_last > timeout:
                    if not flow.emitted:
                        writer.write(flow)
                    del active[key]
                    flow = None
                if flow is None:
                    flow = active[key] = Flow(ts, proto, src, sport, dst, dport)
                    if select is not None:
                        tag = select(ts, proto, src, sport, dst, dport)
                        if tag is None:  # left out: the flow only absorbs its packets
                            flow.emitted, flow.rows = True, None
                        else:
                            flow.tag = tag
                flow.ts_last = ts
                if not flow.emitted:
                    flow.rows.append(image_row(pkt, proto, l4, end))
                    payload = pkt[end:end + 5]
                    flow.tls = flow.tls or looks_like_tls(payload)
                    flow.quic = flow.quic or (proto == UDP and looks_like_quic(payload))
                    if len(flow.rows) == PACKETS:
                        writer.write(flow)

                if timeout and packets % SWEEP_EVERY == 0:
                    for old in [k for k, fl in active.items() if ts - fl.ts_last > timeout]:
                        if not active[old].emitted:
                            writer.write(active[old])
                        del active[old]
                if len(active) > max_active:  # e.g. a flood with spoofed sources
                    for old in list(itertools.islice(active, len(active) // 10)):
                        if not active[old].emitted:
                            writer.write(active[old])
                        del active[old]
                if writer.full:
                    break
        for flow in active.values():
            if not flow.emitted:
                writer.write(flow)
    except BaseException:
        writer.abort()
        raise
    writer.close()
    return packets, skipped, writer.count, writer.tls, writer.quic


# ---------------------------------------------------------------- CLI

def is_capture(path):
    suffixes = [s.lower() for s in path.suffixes[-2:]]
    if suffixes and suffixes[-1] in OPENERS:
        suffixes = suffixes[:-1]
    return bool(suffixes) and suffixes[-1] in CAPTURE_SUFFIXES


def find_captures(inputs):
    """Yield (capture path, folder its output name is relative to)."""
    for item in inputs:
        if item.is_dir():
            for path in sorted(item.rglob("*")):
                if path.is_file() and is_capture(path):
                    yield path, item
        else:
            yield item, item.parent


def output_name(path, root):
    rel = path.relative_to(root)
    while rel.suffix.lower() in OPENERS or rel.suffix.lower() in CAPTURE_SUFFIXES:
        rel = rel.with_suffix("")
    return "__".join(rel.parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path,
                        help="capture files, or folders searched recursively")
    parser.add_argument("--out", type=Path, required=True, help="output folder")
    parser.add_argument("--root", type=Path,
                        help="name outputs relative to this folder instead of each input folder")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="seconds of silence that end a flow; 0 disables (default 120)")
    parser.add_argument("--max-flows", type=int,
                        help="keep only the first N flows of each capture (for quick tests)")
    parser.add_argument("--max-active", type=int, default=1_000_000,
                        help="flows held in memory before the oldest are written out (default 1,000,000)")
    parser.add_argument("--overwrite", action="store_true", help="redo captures that are already done")
    args = parser.parse_args(argv)

    captures = list(find_captures(args.inputs))
    if not captures:
        sys.exit("no captures found")
    args.out.mkdir(parents=True, exist_ok=True)
    failed = 0
    for path, root in captures:
        stem = args.out / output_name(path, args.root or root)
        if not args.overwrite and sibling(stem, ".npy").exists() and sibling(stem, ".csv").exists():
            print(f"{stem.name}: already done, skipped")
            continue
        start = time.time()
        try:
            packets, skipped, flows, tls, quic = process_capture(
                path, stem, args.timeout, args.max_active, args.max_flows)
        except (ValueError, OSError, EOFError, struct.error) as err:
            print(f"{stem.name}: FAILED ({err})", file=sys.stderr)
            failed += 1
            continue
        print(f"{stem.name}: {packets:,} packets ({skipped:,} not usable), {flows:,} flows, "
              f"{tls:,} TLS, {quic:,} QUIC, {time.time() - start:.0f}s", flush=True)
    if failed:
        sys.exit(f"{failed} of {len(captures)} captures failed")


if __name__ == "__main__":
    main()
