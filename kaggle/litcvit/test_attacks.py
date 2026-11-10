"""Tests for attacks.py on hand-built packets. Run: pytest kaggle/litcvit/test_attacks.py"""

import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import attacks as A  # noqa: E402
import mimic as M  # noqa: E402
from test_mimic import ACK, NOP4, PSH_ACK, SYN, SYN_ACK, checksum, checksums_valid, flow, handshake, real_packet, restore  # noqa: E402


def test_arbitrary_values_stay_valid_packets_and_leave_everything_else_alone():
    x = flow(real_packet(flags=SYN, options=NOP4), real_packet(flags=PSH_ACK, payload=b"hello"))
    out = A.arbitrary(x, np.random.default_rng(0))[0]
    allowed = {1, 4, 5, 6, 8, 10, 11, 34, 35, 36, 37, 38, 39}
    assert set(np.flatnonzero(out[1] != x[0, 1]).tolist()) <= allowed
    for k, payload in enumerate((b"", b"hello")):
        assert checksums_valid(restore(out[k], payload))
        assert out[k, 8] >= A.MIN_TTL and out[k, 34:36].any()
        assert out[k, 33] == x[0, k, 33] and out[k, 2:4].tolist() == x[0, k, 2:4].tolist()
    assert not out[2:].any()
    assert out[0, 40:44].tolist() == list(NOP4)


def test_arbitrary_moves_every_free_field_across_flows():
    x = np.concatenate([flow(real_packet(flags=SYN))] * 64)
    out = A.arbitrary(x, np.random.default_rng(0))
    for offset in (1, 4, 5, 8, 34, 35, 38, 39):
        assert len(set(out[:, 0, offset].tolist())) > 8, offset
    assert len(set((out[:, 0, 6] & M.DF).tolist())) == 2


def test_arbitrary_keeps_the_urgent_pointer_when_it_means_something():
    x = flow(real_packet(flags=ACK | A.URG))
    assert A.arbitrary(x, np.random.default_rng(0))[0, 0, 38:40].tolist() == x[0, 0, 38:40].tolist()


OPTIONS = b"\x02\x04\x05\xb4" + b"\x01" + b"\x03\x03\x07" + b"\x08\x0a" + bytes(range(1, 9)) + b"\x01\x01"  # MSS, NOP, scale, timestamps, NOPs


def test_option_values_finds_the_values_of_well_formed_options_only():
    row = flow(real_packet(flags=SYN, options=OPTIONS))[0, 0]
    assert A.option_values(row, 20, 40) == [42, 43, 47, *range(50, 58)]
    broken = row.copy()
    broken[41] = 200  # an MSS option running past the header: nothing after it can be trusted
    assert A.option_values(broken, 20, 40) == []


def test_arbitrary_can_change_option_values_and_keeps_them_well_formed():
    x = flow(real_packet(flags=SYN, options=OPTIONS))
    out = A.arbitrary(x, np.random.default_rng(0), options=True)[0, 0]
    assert checksums_valid(restore(out))
    assert not np.array_equal(out[50:58], x[0, 0, 50:58]) and not np.array_equal(out[42:44], x[0, 0, 42:44])
    for offset in (40, 41, 44, 45, 46, 48, 49, 58, 59):
        assert out[offset] == x[0, 0, offset], offset
    assert np.array_equal(A.arbitrary(x, np.random.default_rng(0))[0, 0, 40:60], x[0, 0, 40:60])


def test_arbitrary_can_change_one_kind_of_option_value():
    x = flow(real_packet(flags=SYN, options=OPTIONS))
    plain = A.arbitrary(x, np.random.default_rng(0))[0, 0]
    out = A.arbitrary(x, np.random.default_rng(0), options=(A.SCALE,))[0, 0]
    assert checksums_valid(restore(out))
    assert out[47] != x[0, 0, 47] and np.array_equal(out[42:44], x[0, 0, 42:44]) and np.array_equal(out[50:58], x[0, 0, 50:58])
    assert np.array_equal(out[:36], plain[:36])  # the other fields don't depend on which options are chosen


def stamped(val, echo, syn=False):
    """Options with these timestamps: MSS and window scale as well if the packet opens its side."""
    stamps = b"\x08\x0a" + struct.pack("!II", val, echo)
    return b"\x02\x04\x05\xb4\x01\x03\x03\x07\x01\x01" + stamps if syn else b"\x01\x01" + stamps


def stamped_handshake():
    return (real_packet(flags=SYN, seq=1000, ack=0, options=stamped(5000, 0, syn=True)),
            real_packet(flags=SYN_ACK, seq=900000000, ack=1001, options=stamped(70000, 5000, syn=True)),
            real_packet(flags=ACK, seq=1001, ack=900000001, options=stamped(5010, 70000)),
            real_packet(flags=PSH_ACK, seq=1001, ack=900000001, options=stamped(5012, 70000), payload=b"hello"),
            real_packet(flags=ACK, seq=900000001, ack=1006, options=stamped(70020, 5012)))


def stamps_of(row, syn=False):
    return struct.unpack("!II", row[(52 if syn else 44):(60 if syn else 52)].tobytes())


def test_packet_sides_follow_the_sequence_numbers():
    x = flow(*handshake())
    ihl, _, flags, _ = M.layout(x)
    assert A.packet_sides(x[0], ihl[0], flags[0]) == [0, 1, 0, 0, 1]
    scan = flow(real_packet(flags=SYN, seq=5000, ack=0), real_packet(flags=0x14, seq=0, ack=5001))
    ihl, _, flags, _ = M.layout(scan)
    assert A.packet_sides(scan[0], ihl[0], flags[0]) == [0, 1, None, None, None]


def test_ordinary_options_keep_each_clock_and_echo_consistent():
    out = A.ordinary_options(flow(*stamped_handshake()), np.random.default_rng(0))[0]
    (a0, e0), (b1, e1) = stamps_of(out[0], syn=True), stamps_of(out[1], syn=True)
    (a2, e2), (a3, e3), (b4, e4) = stamps_of(out[2]), stamps_of(out[3]), stamps_of(out[4])
    assert a0 != 5000 and b1 != 70000
    assert e0 == 0  # nothing to echo yet
    assert e1 == a0 and e2 == b1 and e3 == b1 and e4 == a3
    assert (a2 - a0) % 2 ** 32 == 10 and (a3 - a0) % 2 ** 32 == 12 and (b4 - b1) % 2 ** 32 == 20


def test_ordinary_options_stay_in_range_on_each_side_and_change_nothing_else():
    x = np.concatenate([flow(*stamped_handshake())] * 64)
    out = A.ordinary_options(x, np.random.default_rng(0))
    mss = out[:, :2, 42].astype(int) * 256 + out[:, :2, 43]
    assert mss.min() >= A.MSS_RANGE[0] and mss.max() <= A.MSS_RANGE[1] and len(set(mss.ravel().tolist())) > 16
    assert out[:, :2, 47].max() <= A.MAX_SCALE and len(set(out[:, :2, 47].ravel().tolist())) > 8
    payloads = [b"", b"", b"", b"hello", b""]
    allowed = [{36, 37, 42, 43, 47, *range(52, 60)}] * 2 + [{36, 37, *range(44, 52)}] * 3
    for k in range(5):
        assert checksums_valid(restore(out[0, k], payloads[k]))
        assert set(np.flatnonzero(out[0, k] != x[0, k]).tolist()) <= allowed[k]


def test_ordinary_options_can_change_one_kind():
    x = flow(*stamped_handshake())
    out = A.ordinary_options(x, np.random.default_rng(0), kinds=(A.STAMPS,))[0]
    assert np.array_equal(out[0, 40:50], x[0, 0, 40:50]) and not np.array_equal(out[0, 52:56], x[0, 0, 52:56])
    out = A.ordinary_options(x, np.random.default_rng(0), kinds=(A.MSS,))[0]
    assert not np.array_equal(out[0, 42:44], x[0, 0, 42:44]) and np.array_equal(out[0, 44:60], x[0, 0, 44:60])


def test_add_packets_repeats_the_first_bare_ack_and_pushes_data_out():
    x = flow(*handshake())
    out = A.add_packets(x, copies=2)[0]
    assert [int(out[k, 33]) for k in range(5)] == [SYN, SYN_ACK, ACK, ACK, ACK]
    assert np.array_equal(out[3], x[0, 2]) and np.array_equal(out[4], x[0, 2])
    assert not out[:, M.HEADER:].any()


def test_add_packets_retransmits_the_first_packet_when_no_ack_is_bare():
    x = flow(real_packet(flags=PSH_ACK, payload=b"hello"), real_packet(flags=PSH_ACK, payload=b"world"))
    out = A.add_packets(x, copies=1)[0]
    assert np.array_equal(out[0], x[0, 0]) and np.array_equal(out[1], x[0, 0]) and np.array_equal(out[2], x[0, 1])


def test_resegment_shows_only_the_first_bytes_with_sequence_and_lengths_following():
    x = flow(real_packet(flags=SYN, seq=1000, ack=0), real_packet(flags=SYN_ACK, seq=5000, ack=1001),
             real_packet(flags=PSH_ACK, seq=1001, ack=5001, payload=b"hello there"))
    out = A.resegment(x, size=2)[0]
    assert np.array_equal(out[:2], x[0, :2])
    for n, k in enumerate((2, 3, 4)):
        assert out[k, M.HEADER:M.HEADER + 2].tobytes() == b"hello there"[2 * n:2 * n + 2]
        assert not out[k, M.HEADER + 2:].any()
        assert int.from_bytes(out[k, 24:28].tobytes(), "big") == 1001 + 2 * n
        assert int.from_bytes(out[k, 2:4].tobytes(), "big") == 42
        assert checksum(restore(out[k])[:20]) == 0
        assert not int(out[k, 33]) & A.PSH  # the data goes on past the five kept packets


def test_resegment_ends_a_short_payload_on_its_last_byte_and_keeps_what_follows():
    x = flow(real_packet(flags=PSH_ACK, seq=7, payload=b"abc"), real_packet(flags=ACK, seq=9000))
    out = A.resegment(x, size=2)[0]
    assert out[0, M.HEADER:M.HEADER + 3].tobytes() == b"ab\0" and not int(out[0, 33]) & A.PSH
    assert int.from_bytes(out[1, 2:4].tobytes(), "big") == 41 and out[1, M.HEADER] == ord("c") and int(out[1, 33]) == PSH_ACK
    assert int.from_bytes(out[2, 24:28].tobytes(), "big") == 9000


def test_payload_kept_counts_what_a_model_sees_from_the_lengths():
    x = flow(*handshake())
    assert A.payload_kept(x)[0] == 5
    assert A.payload_kept(A.resegment(x, size=1))[0] == 2
    assert A.payload_kept(flow(real_packet(flags=PSH_ACK, options=NOP4, payload=b"abc")))[0] == 3
    long = flow(real_packet(flags=PSH_ACK, payload=b"abc")).copy()
    long[0, 0, 2:4] = list((40 + 5000).to_bytes(2, "big"))
    assert A.payload_kept(long)[0] == M.ROW - M.HEADER
    y = flow(*handshake()).copy()
    y[0, :, 9] = 17
    assert A.payload_kept(y)[0] == -1


def test_attacks_skip_flows_outside_the_mask_or_without_tcp():
    x = np.concatenate([flow(*handshake()), flow(*handshake())])
    keep = np.array([True, False])
    for out in (A.resegment(x, only=keep), A.add_packets(x, only=keep), A.arbitrary(x, np.random.default_rng(0), only=keep)):
        assert np.array_equal(out[1], x[1]) and not np.array_equal(out[0], x[0])
    y = flow(*handshake()).copy()
    y[0, :, 9] = 17
    for out in (A.resegment(y), A.add_packets(y), A.arbitrary(y, np.random.default_rng(0))):
        assert np.array_equal(out, y)
    z = np.concatenate([flow(*stamped_handshake()), flow(*stamped_handshake())])
    out = A.ordinary_options(z, np.random.default_rng(0), only=keep)
    assert np.array_equal(out[1], z[1]) and not np.array_equal(out[0], z[0])
    z = flow(*stamped_handshake()).copy()
    z[0, :, 9] = 17
    assert np.array_equal(A.ordinary_options(z, np.random.default_rng(0)), z)


def test_attacks_do_not_change_their_input():
    x = flow(*stamped_handshake())
    before = x.copy()
    A.resegment(x), A.add_packets(x), A.arbitrary(x, np.random.default_rng(0)), A.ordinary_options(x, np.random.default_rng(0))
    assert np.array_equal(x, before)
