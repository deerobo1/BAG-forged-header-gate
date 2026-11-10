"""Tests for pcap_to_flows on small hand-built captures. Run: pytest prep"""

import csv
import gzip
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pcap_to_flows as ptf  # noqa: E402

A, B, C = bytes([10, 0, 0, 1]), bytes([10, 0, 0, 2]), bytes([192, 168, 1, 9])
A6, B6 = bytes(15) + b"\x01", bytes(15) + b"\x02"
CLIENT_HELLO = b"\x16\x03\x01\x00\x40" + bytes(range(64))
ECHO = b"\x08\x00\x00\x00\x00\x01\x00\x01"  # ICMP echo request header


# ---------------------------------------------------------------- builders

def ipv4(src, dst, proto, l4):
    return struct.pack("!BBHHHBBH4s4s", 0x45, 0, 20 + len(l4), 1, 0, 64, proto, 0, src, dst) + l4


def ipv6(src, dst, proto, l4):
    return struct.pack("!IHBB16s16s", 6 << 28, len(l4), proto, 64, src, dst) + l4


def tcp(sport, dport, payload=b""):
    return struct.pack("!HHIIBBHHH", sport, dport, 1, 0, 5 << 4, 0x18, 65535, 0, 0) + payload


def udp(sport, dport, payload=b""):
    return struct.pack("!HHHH", sport, dport, 8 + len(payload), 0) + payload


def ether(ip, ethertype=b"\x08\x00", vlan=False, trailer=b""):
    tag = b"\x81\x00\x00\x05" if vlan else b""
    return b"\xaa" * 6 + b"\xbb" * 6 + tag + ethertype + ip + trailer


def write_pcap(path, frames, linktype=1):
    with open(path, "wb") as f:
        f.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, linktype))
        for ts, frame in frames:
            usec = round(ts * 1e6)
            f.write(struct.pack("<IIII", usec // 10**6, usec % 10**6, len(frame), len(frame)) + frame)


def write_pcapng(path, frames, linktype=1, nanoseconds=False):
    def block(btype, body):
        body += bytes(-len(body) % 4)
        return struct.pack("<II", btype, len(body) + 12) + body + struct.pack("<I", len(body) + 12)

    options = struct.pack("<HHB3x", 9, 1, 9) + struct.pack("<HH", 0, 0) if nanoseconds else b""
    ticks = 1e9 if nanoseconds else 1e6
    with open(path, "wb") as f:
        f.write(block(0x0A0D0D0A, struct.pack("<IHHq", 0x1A2B3C4D, 1, 0, -1)))
        f.write(block(1, struct.pack("<HHI", linktype, 0, 65535) + options))
        for ts, frame in frames:
            t = round(ts * ticks)
            f.write(block(6, struct.pack("<IIIII", 0, t >> 32, t & 0xFFFFFFFF, len(frame), len(frame)) + frame))


def run(tmp_path, frames, write=write_pcap, name="cap.pcap", **kwargs):
    cap = tmp_path / name
    write(cap, frames)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    stem = out / "cap"
    ptf.process_capture(cap, stem, **kwargs)
    with open(ptf.sibling(stem, ".csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    return np.load(ptf.sibling(stem, ".npy")), rows


# ---------------------------------------------------------------- tests

def test_bidirectional_flow_keeps_first_five_packets_in_mfr_layout(tmp_path):
    frames = [(0.0, ether(ipv4(A, B, 6, tcp(1234, 443, CLIENT_HELLO))))]
    frames += [(0.1 * i, ether(ipv4(B, A, 6, tcp(443, 1234, bytes([i]) * 10))) if i % 2 else
                ether(ipv4(A, B, 6, tcp(1234, 443, bytes([i]) * 10)))) for i in range(1, 7)]
    images, rows = run(tmp_path, frames)

    assert images.shape == (1, 40, 40) and images.dtype == np.uint8
    assert rows[0]["src"] == "10.0.0.1" and rows[0]["sport"] == "1234"
    assert rows[0]["dst"] == "10.0.0.2" and rows[0]["dport"] == "443"
    assert rows[0]["packets"] == "5" and rows[0]["tls"] == "1" and rows[0]["quic"] == "0"

    first = images[0].reshape(5, 320)[0]
    expected_header = bytearray(ipv4(A, B, 6, tcp(1234, 443))[:40])
    expected_header[2:4] = struct.pack("!H", 40 + len(CLIENT_HELLO))  # real total length
    expected_header[12:20] = bytes(8)  # addresses zeroed
    expected_header[20:24] = bytes(4)  # ports zeroed
    assert first[:40].tobytes() == bytes(expected_header)
    assert not first[40:80].any()
    assert first[80:80 + len(CLIENT_HELLO)].tobytes() == CLIENT_HELLO
    assert not first[80 + len(CLIENT_HELLO):].any()
    # Packet 2 is the reply, 5 is the last kept; 6 and 7 are dropped.
    assert images[0].reshape(5, 320)[1, 80:90].tobytes() == b"\x01" * 10
    assert images[0].reshape(5, 320)[4, 80:90].tobytes() == b"\x04" * 10


def test_short_flows_are_zero_padded_and_timeout_splits_a_key(tmp_path):
    query = ether(ipv4(A, C, 17, udp(5000, 53, b"dns?")))
    images, rows = run(tmp_path, [(0.0, query), (1.0, query), (500.0, query)], timeout=120)

    assert [r["packets"] for r in rows] == ["2", "1"]
    assert [float(r["ts_first"]) for r in rows] == [0.0, 500.0]
    assert not images[0].reshape(5, 320)[2:].any()
    assert images[0].reshape(5, 320)[0, 80:84].tobytes() == b"dns?"

    _, rows = run(tmp_path, [(0.0, query), (1.0, query), (500.0, query)], timeout=0)
    assert [r["packets"] for r in rows] == ["3"]


def test_pcap_and_pcapng_give_identical_output(tmp_path):
    frames = [
        (0.0, ether(ipv4(A, B, 6, tcp(1234, 80, b"GET /")), vlan=True)),
        (0.5, ether(ipv4(B, A, 6, tcp(80, 1234, b"HTTP/1.1")), trailer=b"\xee" * 6)),
        (0.7, ether(b"\x00" * 28, ethertype=b"\x08\x06")),  # ARP: not usable
        (1.0, ether(ipv4(A, C, 1, ECHO + b"ping"))),  # ICMP: no ports
    ]
    images_a, rows_a = run(tmp_path, frames, name="a.pcap")
    images_b, rows_b = run(tmp_path, frames, write=write_pcapng, name="b.pcapng")

    assert np.array_equal(images_a, images_b) and rows_a == rows_b
    assert [(r["proto"], r["packets"]) for r in rows_a] == [("6", "2"), ("1", "1")]
    reply = images_a[0].reshape(5, 320)[1]
    assert reply[80:88].tobytes() == b"HTTP/1.1" and not reply[88:].any()  # trailer dropped
    icmp = images_a[1].reshape(5, 320)[0]
    assert icmp[20:28].tobytes() == ECHO  # ICMP header kept whole, in the header block
    assert not icmp[28:80].any() and icmp[80:84].tobytes() == b"ping"


def test_ipv6_quic_and_nanosecond_pcapng(tmp_path):
    initial = b"\xc3\x00\x00\x00\x01" + bytes(30)
    frames = [(1.000000001, ether(ipv6(A6, B6, 17, udp(50000, 443, initial)), ethertype=b"\x86\xdd"))]
    images, rows = run(tmp_path, frames, write=write_pcapng, name="q.pcapng")

    assert rows[0]["quic"] == "1" and rows[0]["tls"] == "0"
    assert rows[0]["src"] == "::1" and rows[0]["dport"] == "443"
    assert rows[0]["ts_first"] == "1.000000"
    header = images[0].reshape(5, 320)[0, :80]
    assert not header[8:40].any() and not header[40:44].any()  # addresses, ports
    assert header[44:48].tobytes() == struct.pack("!HH", 8 + len(initial), 0)
    assert not header[48:].any()


def test_raw_ip_link_types(tmp_path):
    packet = ipv4(A, B, 17, udp(1000, 2000, b"x"))
    for linktype in (101, 12):  # LINKTYPE_RAW, and DLT_RAW as 5GAD-2022 writes it
        name = f"raw{linktype}.pcapng"
        images, rows = run(tmp_path, [(0.0, packet)], write=lambda p, f: write_pcapng(p, f, linktype), name=name)
        assert len(rows) == 1 and images[0].reshape(5, 320)[0, 80] == ord("x")


def test_cli_reads_gzip_names_outputs_and_skips_done(tmp_path, capsys):
    data = tmp_path / "data" / "Malware"
    data.mkdir(parents=True)
    plain = tmp_path / "plain.pcap"
    write_pcap(plain, [(0.0, ether(ipv4(A, B, 6, tcp(1, 2))))])
    with gzip.open(data / "Zeus.pcap.gz", "wb") as f:
        f.write(plain.read_bytes())
    plain.replace(data / "2018-05-09-192.168.100.103.pcap")
    (data / "notes.txt").write_text("not a capture")

    out = tmp_path / "flows"
    ptf.main([str(tmp_path / "data"), "--out", str(out)])
    assert sorted(p.name for p in out.iterdir()) == [
        "Malware__2018-05-09-192.168.100.103.csv", "Malware__2018-05-09-192.168.100.103.npy",
        "Malware__Zeus.csv", "Malware__Zeus.npy"]
    assert np.load(out / "Malware__Zeus.npy").shape == (1, 40, 40)

    ptf.main([str(tmp_path / "data"), "--out", str(out)])
    assert capsys.readouterr().out.count("already done") == 2


def test_max_flows_caps_output(tmp_path):
    frames = [(float(i), ether(ipv4(A, B, 17, udp(1000 + i, 53)))) for i in range(10)]
    images, rows = run(tmp_path, frames, max_flows=3)
    assert images.shape[0] == 3 and [r["sport"] for r in rows] == ["1000", "1001", "1002"]


def test_select_keeps_and_tags_chosen_flows_only(tmp_path):
    frames = [(0.0, ether(ipv4(A, B, 6, tcp(1000, 80, b"a")))),
              (0.5, ether(ipv4(A, C, 6, tcp(2000, 80, b"b")))),
              (1.0, ether(ipv4(B, A, 6, tcp(80, 1000, b"c")))),
              (1.5, ether(ipv4(C, A, 6, tcp(80, 2000, b"d"))))]
    calls = []

    def select(ts, proto, src, sport, dst, dport):
        calls.append((ts, src, sport))
        return "Malicious|C&C" if dst == B else None

    images, rows = run(tmp_path, frames, select=select)
    assert calls == [(0.0, A, 1000), (0.5, A, 2000)]  # once per flow, at its first packet
    assert [(r["dst"], r["packets"], r["tag"]) for r in rows] == [("10.0.0.2", "2", "Malicious|C&C")]
    assert images.shape[0] == 1 and images[0].reshape(5, 320)[1, 80:81].tobytes() == b"c"

    _, rows = run(tmp_path, frames)
    assert [r["tag"] for r in rows] == ["", ""]
