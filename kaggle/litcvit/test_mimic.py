"""Tests for header mimicry on hand-built packets with real checksums. Run: pytest kaggle/litcvit/test_mimic.py"""

import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import mimic as M  # noqa: E402

SRC, DST = bytes([10, 0, 0, 5]), bytes([93, 184, 216, 34])
SYN, SYN_ACK, ACK, PSH_ACK = 0x02, 0x12, 0x10, 0x18


def checksum(b):
    if len(b) % 2:
        b += b"\0"
    total = sum(struct.unpack(f"!{len(b) // 2}H", b))
    while total >> 16:
        total = (total & 0xFFFF) + (total >> 16)
    return total ^ 0xFFFF


def real_packet(ttl=128, tos=0, ipid=0x1234, df=True, window=8192, flags=SYN, options=b"", payload=b"", seq=7, ack=9):
    """IPv4 TCP packet with correct IP and TCP checksums, as sent on the wire."""
    doff = (20 + len(options)) // 4
    tcp = struct.pack("!HHIIBBHHH", 49152, 443, seq, ack, doff << 4, flags, window, 0, 0) + options
    total = 20 + len(tcp) + len(payload)
    frag = 0x4000 if df else 0
    ip = struct.pack("!BBHHHBBH", 0x45, tos, total, ipid, frag, ttl, 6, 0) + SRC + DST
    ip = ip[:10] + struct.pack("!H", checksum(ip)) + ip[12:]
    pseudo = SRC + DST + struct.pack("!BBH", 0, 6, len(tcp) + len(payload))
    c = checksum(pseudo + tcp + payload)
    return ip + tcp[:16] + struct.pack("!H", c) + tcp[18:] + payload


def row_of(packet):
    """The image row for a real packet: addresses and ports zeroed, header and payload padded."""
    ihl = (packet[0] & 15) * 4
    doff = (packet[ihl + 12] >> 4) * 4
    header = bytearray(packet[:ihl + doff])
    header[12:20] = bytes(8)
    header[ihl:ihl + 4] = bytes(4)
    data = bytes(header).ljust(M.HEADER, b"\0") + packet[ihl + doff:].ljust(M.ROW - M.HEADER, b"\0")
    return np.frombuffer(data, np.uint8).copy()


def restore(row, payload=b""):
    """Back from an image row to wire bytes, so the checksums can be recomputed in full."""
    ihl = (row[0] & 15) * 4
    doff = (row[ihl + 12] >> 4) * 4
    b = bytearray(row[:ihl + doff].tobytes())
    b[12:16], b[16:20] = SRC, DST
    b[ihl:ihl + 4] = struct.pack("!HH", 49152, 443)
    return bytes(b) + payload


def checksums_valid(packet):
    ihl = (packet[0] & 15) * 4
    pseudo = packet[12:20] + struct.pack("!BBH", 0, 6, len(packet) - ihl)
    return checksum(packet[:ihl]) == 0 and checksum(pseudo + packet[ihl:]) == 0


def flow(*packets):
    rows = [row_of(p) for p in packets] + [np.zeros(M.ROW, np.uint8)] * (M.PACKETS - len(packets))
    return np.stack(rows)[None]


TS_A = b"\x01\x01\x08\x0a" + bytes(range(100, 108))
TS_B = b"\x01\x01\x08\x0a" + bytes(range(200, 208))
NOP4 = b"\x01\x01\x01\x01"


def test_checksums_stay_valid_after_every_field_changes():
    payload = bytes(range(60))
    attack = real_packet(ttl=128, ipid=1, df=True, window=8192, flags=PSH_ACK, options=TS_A, payload=payload)
    donor = real_packet(ttl=32, tos=8, ipid=999, df=False, window=5792, flags=PSH_ACK, options=TS_B)
    assert checksums_valid(attack)
    new = M.patch_packet(row_of(attack), row_of(donor), M.STACK)
    assert checksums_valid(restore(new, payload))


def test_only_named_fields_change():
    attack, donor = row_of(real_packet(ttl=128, window=8192)), row_of(real_packet(ttl=32, tos=8, ipid=999, df=False, window=5792))
    new = M.patch_packet(attack, donor, ("ttl",))
    changed = set(np.flatnonzero(new != attack).tolist())
    assert 8 in changed and changed <= {8, 10, 11}  # TTL and the IP checksum; TCP doesn't cover TTL
    new = M.patch_packet(attack, donor, ("window",))
    changed = set(np.flatnonzero(new != attack).tolist())
    assert {34, 35} <= changed and changed <= {34, 35, 36, 37}  # window and the TCP checksum


def test_stack_keeps_payload_lengths_flags_and_sequence_numbers():
    attack = row_of(real_packet(flags=PSH_ACK, options=TS_A, payload=b"hello", seq=11, ack=22))
    donor = row_of(real_packet(ttl=32, flags=PSH_ACK, options=TS_B, seq=33, ack=44))
    new = M.patch_packet(attack, donor, M.STACK)
    for byte in [0, 2, 3, 9, 32, 33] + list(range(24, 32)):  # version, length, protocol, offset and flags, seq and ack
        assert new[byte] == attack[byte]
    assert np.array_equal(new[M.HEADER:], attack[M.HEADER:])
    assert new[40:52].tobytes() == TS_B


def test_options_copied_only_when_lengths_match():
    attack = row_of(real_packet(options=TS_A))
    donor = row_of(real_packet(ttl=32, options=NOP4))
    new = M.patch_packet(attack, donor, M.STACK)
    assert new[40:52].tobytes() == TS_A and new[8] == 32


def test_donors_need_the_same_flag_order_and_prefer_matching_lengths():
    a = flow(real_packet(flags=SYN, options=TS_A), real_packet(flags=SYN_ACK), real_packet(flags=ACK))
    wrong_order = flow(real_packet(flags=SYN), real_packet(flags=ACK))
    other_length = flow(real_packet(flags=SYN), real_packet(flags=SYN_ACK), real_packet(flags=ACK))
    same_length = flow(real_packet(flags=SYN, options=TS_B), real_packet(flags=SYN_ACK), real_packet(flags=ACK))
    donors = np.concatenate([wrong_order, other_length, same_length])
    chosen = M.choose_donors(a, donors, np.random.default_rng(0))
    assert chosen.tolist() == [2]
    assert M.choose_donors(a, wrong_order, np.random.default_rng(0)).tolist() == [-1]


def test_non_tcp_flows_stay_as_recorded():
    udp = row_of(real_packet())
    udp[9] = 17
    a = np.stack([udp] + [np.zeros(M.ROW, np.uint8)] * 4)[None]
    donors = flow(real_packet())
    chosen = M.choose_donors(a, donors, np.random.default_rng(0))
    assert chosen.tolist() == [-1]
    assert np.array_equal(M.apply_mimicry(a, donors, chosen, M.STACK), a)


def sequence_of(row):
    return struct.unpack("!II", row[24:32].tobytes())


def handshake():
    return (real_packet(flags=SYN, seq=1000, ack=0), real_packet(flags=SYN_ACK, seq=900000000, ack=1001),
            real_packet(flags=ACK, seq=1001, ack=900000001), real_packet(flags=PSH_ACK, seq=1001, ack=900000001, payload=b"hello"),
            real_packet(flags=ACK, seq=900000001, ack=1006))


def test_shift_keeps_each_sides_differences_and_the_acknowledgements_matching():
    packets = handshake()
    out = M.shift_sequence(flow(*packets), np.random.default_rng(0))[0]
    (s0, _), (s1, a1), (s2, a2), (s3, a3), (s4, a4) = [sequence_of(out[k]) for k in range(5)]
    assert s0 != 1000 and s1 != 900000000
    assert a1 == (s0 + 1) % 2 ** 32  # the server acknowledges the client's new number
    assert s2 == (s0 + 1) % 2 ** 32 and a2 == (s1 + 1) % 2 ** 32
    assert s3 == s2 and a3 == a2
    assert s4 == (s1 + 1) % 2 ** 32 and a4 == (s3 + 5) % 2 ** 32


def test_shift_keeps_checksums_valid_and_changes_nothing_else():
    packets = handshake()
    x = flow(*packets)
    out = M.shift_sequence(x, np.random.default_rng(0))[0]
    payloads = [b"", b"", b"", b"hello", b""]
    for k in range(5):
        assert checksums_valid(restore(out[k], payloads[k]))
        changed = set(np.flatnonzero(out[k] != x[0, k]).tolist())
        assert changed <= set(range(24, 32)) | {36, 37}  # sequence numbers and the TCP checksum


def test_shift_leaves_a_syns_empty_acknowledgement_and_a_resets_zero_sequence_alone():
    scan = (real_packet(flags=SYN, seq=5000, ack=0), real_packet(flags=0x14, seq=0, ack=5001))
    out = M.shift_sequence(flow(*scan), np.random.default_rng(0))[0]
    (s0, a0), (s1, a1) = sequence_of(out[0]), sequence_of(out[1])
    assert a0 == 0 and s1 == 0 and a1 == (s0 + 1) % 2 ** 32


def test_shift_can_be_limited_to_some_flows_and_skips_non_tcp():
    x = np.concatenate([flow(*handshake()), flow(*handshake())])
    out = M.shift_sequence(x, np.random.default_rng(0), only=np.array([True, False]))
    assert not np.array_equal(out[0], x[0]) and np.array_equal(out[1], x[1])
    udp = row_of(real_packet())
    udp[9] = 17
    a = np.stack([udp] + [np.zeros(M.ROW, np.uint8)] * 4)[None]
    assert np.array_equal(M.shift_sequence(a, np.random.default_rng(0)), a)


def test_apply_changes_every_packet_of_a_flow_and_not_the_input():
    a = flow(real_packet(ttl=128, flags=SYN), real_packet(ttl=64, flags=SYN_ACK))
    d = flow(real_packet(ttl=32, flags=SYN), real_packet(ttl=33, flags=SYN_ACK))
    before = a.copy()
    out = M.apply_mimicry(a, d, np.array([0]), ("ttl",))
    assert out[0, 0, 8] == 32 and out[0, 1, 8] == 33
    assert np.array_equal(a, before)
