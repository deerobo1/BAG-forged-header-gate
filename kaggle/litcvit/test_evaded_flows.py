"""Tests for evaded_flows.py on hand-built packets. Run: pytest kaggle/litcvit/test_evaded_flows.py"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import evaded_flows as E  # noqa: E402
import mimic as M  # noqa: E402
from test_mimic import ACK, PSH_ACK, SYN, SYN_ACK, flow, handshake, real_packet  # noqa: E402


def bare(**fields):
    return flow(real_packet(flags=SYN, seq=1000, ack=0, **fields), real_packet(flags=SYN_ACK, seq=900000000, ack=1001, **fields),
                real_packet(flags=ACK, seq=1001, ack=900000001, **fields))


def talks(payload, **fields):
    return flow(real_packet(flags=SYN, **fields), real_packet(flags=SYN_ACK, **fields), real_packet(flags=PSH_ACK, payload=payload, **fields))


def test_payload_is_any_byte_after_the_tcp_header_whatever_its_length():
    options = flow(real_packet(flags=SYN, options=b"\x01\x01\x01\x01"))
    assert E.has_payload(np.concatenate([bare(), flow(*handshake()), options])).tolist() == [False, True, False]


def test_a_copy_may_differ_in_sequence_numbers_and_checksums_only():
    x = bare()
    shifted = M.shift_sequence(x, np.random.default_rng(0))
    assert not np.array_equal(shifted, x)
    assert E.fingerprint(shifted) == E.fingerprint(x)
    for other in (bare(window=1024), bare(ttl=64), bare(ipid=9), talks(b"hi")):
        assert E.fingerprint(other) != E.fingerprint(x)


def test_fingerprint_doesnt_change_the_flows_it_reads():
    x = bare()
    before = x.copy()
    E.fingerprint(x)
    assert np.array_equal(x, before)


def test_an_attacked_flow_with_no_payload_copies_its_donor_and_one_with_payload_does_not():
    malware = dict(ttl=64, window=1024, ipid=77)
    benign = dict(ttl=128, window=8192, ipid=5)
    chosen = np.array([0])
    assert E.fingerprint(M.apply_mimicry(bare(**malware), bare(**benign), chosen, M.STACK)) == E.fingerprint(bare(**benign))
    assert E.fingerprint(M.apply_mimicry(talks(b"bad", **malware), talks(b"good", **benign), chosen, M.STACK)) != E.fingerprint(talks(b"good", **benign))
