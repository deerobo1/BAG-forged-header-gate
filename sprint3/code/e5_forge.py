"""E5 part 1: forge M fields in real pcaps and check that every forged packet is still valid (R1, sprint3/GUARANTEE.md).

For each run with raw pcaps (TII has none locally), TCP flows are rebuilt from malware captures and benign captures by
5-tuple (first 5 packets, both directions). Each malware flow takes a benign donor with the same flag sequence, as
mimic.py does, and gets the donor's TOS, IP ID, DF, TTL, window and (same-length) TCP options per packet, plus a new
ISN per side. Payload bytes are untouched. IP and TCP checksums are recomputed (in full when the capture holds the whole
packet, else by RFC 1624 from the old value).

Validation, per forged packet, by three independent checks (tshark isn't installable offline here, BLOCKERS.md):
  own   RFC 1071 IP header checksum and TCP checksum with the pseudo-header, plus length consistency
  dpkt  parses; recomputed checksums match
  scapy parses with a TCP layer and no stray padding; recomputed checksums match

  python sprint3/code/e5_forge.py   -> sprint3/results/e5_forge.(json|csv), e5_fields.csv, and value sets for e5_live.py
"""

import glob
import json
import lzma
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "prep"))
import pcap_to_flows as P  # noqa: E402

RAW = Path("C:/data/CNS_proj/raw")
OUT = Path("C:/data/CNS_proj/sprint3_cache/e5")
RESULTS = ROOT / "sprint3" / "results"
SOURCES = {
    "5gad2022": ("5GAD-2022/Attacks/*/Attacks_*.pcapng", "5GAD-2022/Normal-1UE/**/*.pcap*"),
    "ciciot2022": ("CICIoT2022/6-Attacks/**/*.pcap", "CICIoT2022/3-Interactions/**/*.pcap"),
    "iot23": ("IoT-23/CTU-IoT-Malware-Capture-*/**/*.pcap*", "IoT-23/CTU-Honeypot-Capture-*/**/*.pcap*"),
    "mcfp": ("MCFP/CTU-Malware-Capture-Botnet-*/*.pcap", "MCFP/CTU-Normal-*/*.pcap"),
    "ustc": ("USTC-TFC2016/USTC-TFC2016-master/Malware/*.pcap", "USTC-TFC2016/USTC-TFC2016-master/Benign/*.pcap"),
}
WANT_MALWARE, WANT_DONORS, PER_FILE_PACKETS, LIVE_SETS = 300, 3000, 300_000, 50


def csum(data):
    if len(data) % 2:
        data += b"\0"
    s = sum(struct.unpack(f"!{len(data) // 2}H", data))
    while s >> 16:
        s = (s & 0xFFFF) + (s >> 16)
    return ~s & 0xFFFF


def tcp_sum(pkt, ihl):
    seg = bytearray(pkt[ihl:])
    seg[16:18] = b"\0\0"
    pseudo = pkt[12:20] + struct.pack("!BBH", 0, 6, len(seg))
    return csum(bytes(pseudo + seg))


def incr(check, old, new):
    """RFC 1624 update of a 16-bit checksum when bytes old (even length) become new."""
    c = ~check & 0xFFFF
    for i in range(0, len(old), 2):
        c += (~int.from_bytes(old[i:i + 2], "big") & 0xFFFF) + int.from_bytes(new[i:i + 2], "big")
    while c >> 16:
        c = (c & 0xFFFF) + (c >> 16)
    return ~c & 0xFFFF


def flows_from(pattern, want, min_packets, rng):
    """TCP flows (lists of up to 5 (packet bytes, complete) in capture order) from the files matching pattern."""
    files = sorted(glob.glob(str(RAW / pattern), recursive=True))
    rng.shuffle(files)
    done, active = [], {}
    for path in files:
        try:
            with P.open_capture(Path(path)) as f:
                for n, (_, link, frame) in enumerate(P.read_packets(f)):
                    if n >= PER_FILE_PACKETS or len(done) >= want:
                        break
                    ip = P.ip_packet(link, frame)
                    if ip is None or len(ip) < 40 or ip[0] >> 4 != 4 or ip[9] != 6:
                        continue
                    parsed = P.parse_packet(ip)
                    if parsed is None:
                        continue
                    _, src, sport, dst, dport, pkt, l4, _ = parsed
                    total = int.from_bytes(pkt[2:4], "big")
                    key = tuple(sorted([(src, sport), (dst, dport)]))
                    flow = active.setdefault(key, [])
                    if len(flow) < 5:
                        flow.append((bytes(pkt), total == len(pkt) and total > 0))
                        if len(flow) == 5:
                            done.append(active.pop(key))
        except (OSError, ValueError, EOFError, lzma.LZMAError) as e:
            print("skip", path, e)
        if len(done) >= want:
            break
    done += [f for f in active.values() if len(f) >= min_packets][:max(0, want - len(done))]
    return done[:want]


def flags_of(flow):
    out = []
    for p, _ in flow:
        ihl = (p[0] & 15) * 4
        out.append((p[ihl + 13], (p[ihl + 12] >> 4) * 4))
    return out


def forge(flow, donor, rng):
    """Copy the donor's M fields onto the flow, packet by packet; new ISNs per side; checksums fixed."""
    sides, offsets, out, fields = {}, {}, [], []
    for (p, complete), (d, _) in zip(flow, donor):
        p, d = bytearray(p), bytes(d)
        ihl, dihl = (p[0] & 15) * 4, (d[0] & 15) * 4
        doff, ddoff = (p[ihl + 12] >> 4) * 4, (d[dihl + 12] >> 4) * 4
        old_ip, old_tcp = bytes(p[:ihl]), bytes(p[ihl:ihl + doff])
        side = sides.setdefault(bytes(p[12:16]), len(sides))
        offsets.setdefault(side, int(rng.integers(1, 2 ** 32)))
        offsets.setdefault(1 - side, int(rng.integers(1, 2 ** 32)))
        p[1] = d[1]                                      # TOS
        p[4:6] = d[4:6]                                  # IP ID
        p[6] = (p[6] & ~0x40 & 0xFF) | (d[6] & 0x40)    # DF
        p[8] = d[8]                                      # TTL
        p[ihl + 14:ihl + 16] = d[dihl + 14:dihl + 16]   # window
        same_opts = doff == ddoff and doff > 20
        if same_opts:
            p[ihl + 20:ihl + doff] = d[dihl + 20:dihl + ddoff]
        seq = (int.from_bytes(p[ihl + 4:ihl + 8], "big") + offsets[side]) % 2 ** 32
        p[ihl + 4:ihl + 8] = seq.to_bytes(4, "big")
        if p[ihl + 13] & 0x10:
            ack = (int.from_bytes(p[ihl + 8:ihl + 12], "big") + offsets[1 - side]) % 2 ** 32
            p[ihl + 8:ihl + 12] = ack.to_bytes(4, "big")
        p[10:12] = b"\0\0"
        p[10:12] = csum(bytes(p[:ihl])).to_bytes(2, "big")
        if complete:
            p[ihl + 16:ihl + 18] = tcp_sum(bytes(p), ihl).to_bytes(2, "big")
        else:
            old = int.from_bytes(p[ihl + 16:ihl + 18], "big")
            new_tcp = bytes(p[ihl:ihl + doff])
            o, n = bytearray(old_tcp), bytearray(new_tcp)
            o[16:18] = n[16:18] = b"\0\0"
            p[ihl + 16:ihl + 18] = incr(old, bytes(o), bytes(n)).to_bytes(2, "big")
        out.append((bytes(p), complete))
        fields.append({"ttl": d[8], "tos": d[1], "ipid": int.from_bytes(d[4:6], "big"), "df": bool(d[6] & 0x40),
                       "window": int.from_bytes(d[dihl + 14:dihl + 16], "big"),
                       "options": d[dihl + 20:dihl + ddoff].hex() if same_opts else None, "side": side,
                       "flags": p[ihl + 13]})
    return out, fields


def check(p, complete):
    """Three validators -> dict of booleans (True = problem)."""
    import dpkt
    from scapy.layers.inet import IP, TCP
    res = {}
    ihl = (p[0] & 15) * 4
    doff = (p[ihl + 12] >> 4) * 4
    total = int.from_bytes(p[2:4], "big")
    res["own_malformed"] = ihl < 20 or doff < 20 or ihl + doff > len(p) or (complete and total != len(p))
    res["own_ip_csum"] = csum(p[:ihl]) != 0
    res["own_tcp_csum"] = complete and tcp_sum(p, ihl) != int.from_bytes(p[ihl + 16:ihl + 18], "big")
    try:
        d = dpkt.ip.IP(p)
        ok = isinstance(d.data, dpkt.tcp.TCP)
        res["dpkt_malformed"] = not ok
        if ok:
            ip_sum, tcp_s = d.sum, d.data.sum
            d.sum = 0
            d.data.sum = 0
            raw = bytes(d)
            again = dpkt.ip.IP(raw)
            res["dpkt_ip_csum"] = again.sum != ip_sum
            res["dpkt_tcp_csum"] = complete and again.data.sum != tcp_s
    except (dpkt.UnpackError, ValueError, struct.error):
        res["dpkt_malformed"] = True
    try:
        s = IP(p)
        ok = s.haslayer(TCP) and not (complete and s.haslayer("Padding"))
        res["scapy_malformed"] = not ok
        if ok:
            ip_sum, tcp_s = s.chksum, s[TCP].chksum
            del s.chksum
            del s[TCP].chksum
            again = IP(bytes(s))
            res["scapy_ip_csum"] = again.chksum != ip_sum
            res["scapy_tcp_csum"] = complete and again[TCP].chksum != tcp_s
    except Exception:  # noqa: BLE001 - any scapy failure counts as malformed
        res["scapy_malformed"] = True
    return res


def write_pcap(path, packets):
    with open(path, "wb") as f:
        f.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 101))  # LINKTYPE_RAW
        for i, p in enumerate(packets):
            f.write(struct.pack("<IIII", i, 0, len(p), len(p)) + p)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, field_rows, live = [], [], {}
    for run, (bad_glob, good_glob) in SOURCES.items():
        rng = np.random.default_rng(11)
        malware = flows_from(bad_glob, WANT_MALWARE * 3, 3, rng)
        donors = flows_from(good_glob, WANT_DONORS, 3, rng)
        ctrl = lambda fl: tuple(f & 0x17 for f, _ in fl)  # SYN, ACK, FIN, RST: a SYN takes a SYN's fields
        by_flags, by_packet = {}, {}
        for j, d in enumerate(donors):
            by_flags.setdefault(ctrl(flags_of(d)), []).append(j)
            for (f, _), pkt in zip(flags_of(d), d):
                by_packet.setdefault(f & 0x17, []).append(pkt)
        forged_flows, forged_fields, orig_bad, per_packet = [], [], 0, 0
        for flow in malware:
            fl = flags_of(flow)
            pool = by_flags.get(ctrl(fl))
            if pool:
                best = max(pool, key=lambda j: sum(a[1] == b[1] for a, b in zip(fl, flags_of(donors[j]))) + rng.random() * 0.1)
                donor = donors[best]
            elif all((f & 0x17) in by_packet for f, _ in fl):  # no whole flow matches: a benign packet per position
                donor = [by_packet[f & 0x17][rng.integers(len(by_packet[f & 0x17]))] for f, _ in fl]
                per_packet += 1
            else:
                continue
            new, fields = forge(flow, donor, rng)
            forged_flows.append((flow, new))
            forged_fields.append(fields)
            if len(forged_flows) >= WANT_MALWARE:
                break
        totals = {}
        packets = 0
        for (orig, new) in forged_flows:
            for (po, comp), (pn, _) in zip(orig, new):
                packets += 1
                ihl = (po[0] & 15) * 4
                orig_bad += bool(comp and tcp_sum(po, ihl) != int.from_bytes(po[ihl + 16:ihl + 18], "big"))
                for k2, v in check(pn, comp).items():
                    totals[k2] = totals.get(k2, 0) + int(bool(v))
        complete = sum(c for (_, new) in forged_flows for _, c in new)
        rows.append({"run": run, "malware_flows_read": len(malware), "donor_flows": len(donors), "forged_flows": len(forged_flows),
                     "forged_packets": packets, "complete_packets": complete,
                     "original_tcp_csum_bad": orig_bad, "per_packet_donors": per_packet, **totals})
        write_pcap(OUT / f"{run}_forged.pcap", [p for _, new in forged_flows for p, _ in new])
        write_pcap(OUT / f"{run}_original.pcap", [p for orig, _ in forged_flows for p, _ in orig])
        for fields in forged_fields:
            for f in fields:
                field_rows.append({"run": run, **{k: v for k, v in f.items() if k != "options"},
                                   "options_len": len(f["options"]) // 2 if f["options"] else 0})
        pick = rng.permutation(len(forged_fields))[:LIVE_SETS]
        live[run] = [forged_fields[i] for i in pick]
        print(rows[-1], flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "e5_forge.json").write_text(json.dumps(rows, indent=1))
    import pandas as pd
    pd.DataFrame(rows).to_csv(RESULTS / "e5_forge.csv", index=False)
    pd.DataFrame(field_rows).to_csv(RESULTS / "e5_fields.csv", index=False)
    (OUT / "live_sets.json").write_text(json.dumps(live))


if __name__ == "__main__":
    main()
