"""Tests for litcvit's shortcut-field mask and the views and test perturbations built on it. Run: pytest kaggle/litcvit"""

import struct
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import litcvit as L  # noqa: E402

PAYLOAD = bytes(range(1, 241))
TS_OPTIONS = b"\x01\x01\x08\x0a" + bytes(range(100, 108))  # NOP, NOP, timestamps (TSval, TSecr)


# ---------------------------------------------------------------- builders (addresses and ports zero, as in the images)

def ipv4(proto, l4, options=b""):
    ihl = 5 + len(options) // 4
    return struct.pack("!BBHHHBBH8x", 0x40 | ihl, 0x10, 99, 0x1234, 0x4000, 64, proto, 0xBEEF) + options + l4


def ipv6(proto, l4):
    return struct.pack("!IHBB32x", 6 << 28 | 0xA << 20 | 0x12345, len(l4), proto, 64) + l4


def tcp(options=b""):
    doff = (20 + len(options)) // 4
    return struct.pack("!4xIIBBHHH", 0x11111111, 0x22222222, doff << 4, 0x18, 0xFFFF, 0xCAFE, 7) + options


def udp():
    return struct.pack("!4xHH", 8 + len(PAYLOAD), 0xD00D)


def icmp():
    return struct.pack("!BBHHH", 8, 0, 0xABCD, 0x0102, 0x0304)


def image(*headers):
    """One flow image [1, 40, 40] with a row per header (and PAYLOAD), the rest empty."""
    rows = [h.ljust(L.HEADER, b"\0") + PAYLOAD for h in headers] + [bytes(L.ROW)] * (L.PACKETS - len(headers))
    return torch.from_numpy(np.frombuffer(b"".join(rows), np.uint8).reshape(1, 40, 40).copy())


def masked(x):
    """Masked offsets of each packet row."""
    return [torch.nonzero(r).flatten().tolist() for r in L.shortcut_fields(x)[0]]


IP4 = [1, 4, 5, 6, 7, 8, 10, 11]


# ---------------------------------------------------------------- tests

def test_ipv4_tcp_with_timestamps():
    tcp_fields = list(range(24, 32)) + list(range(34, 40)) + list(range(40, 52))
    assert masked(image(ipv4(6, tcp(TS_OPTIONS))))[0] == IP4 + tcp_fields


def test_ip_options_shift_udp():
    assert masked(image(ipv4(17, udp(), options=b"\x01\x01\x01\x00")))[0] == IP4 + [20, 21, 22, 23, 30, 31]


def test_icmp_keeps_type_and_code():
    assert masked(image(ipv4(1, icmp())))[0] == IP4 + [22, 23, 24, 25, 26, 27]


def test_ipv6_tcp():
    assert masked(image(ipv6(6, tcp())))[0] == [1, 2, 3, 7] + list(range(44, 52)) + list(range(54, 60))


def test_truncated_transport_and_empty_rows_left_alone():
    rows = masked(image(ipv4(6, b""), ipv4(17, bytes(8))))
    assert rows[0] == IP4 and rows[1] == IP4
    assert rows[2:] == [[], [], []]


def test_perturb_changes_only_masked_bytes():
    x = image(ipv4(6, tcp(TS_OPTIONS)), ipv6(17, udp()), ipv4(1, icmp()))
    fields = L.shortcut_fields(x)
    flat = x.reshape(1, L.PACKETS, L.ROW)
    gen = torch.Generator().manual_seed(0)
    zero, rand = L.perturb(x, "zero", gen), L.perturb(x, "random", gen)
    assert torch.equal(zero[~fields], flat[~fields]) and not zero[fields].any()
    assert torch.equal(rand[~fields], flat[~fields]) and not torch.equal(rand[fields], flat[fields])


def test_perturb_without_payload():
    x = image(ipv4(6, tcp(TS_OPTIONS)), ipv6(17, udp()), ipv4(1, icmp()))
    fields = L.shortcut_fields(x)
    flat = x.reshape(1, L.PACKETS, L.ROW)
    header = torch.zeros_like(fields)
    header[..., :L.HEADER] = True
    gen = torch.Generator().manual_seed(0)
    kept, rand = L.perturb(x, "nopayload", gen), L.perturb(x, "random_nopayload", gen)
    assert torch.equal(kept[header], flat[header]) and not kept[~header].any()
    assert torch.equal(rand[header & ~fields], flat[header & ~fields]) and not rand[~header].any()
    assert not torch.equal(rand[fields], flat[fields])
    assert flat[~header].any()  # the input itself keeps its payload


def test_random_test_draws_the_same_bytes_as_before():
    x = image(ipv4(6, tcp(TS_OPTIONS)), ipv4(17, udp()))
    new = L.perturb(x, "random", torch.Generator().manual_seed(0))
    old = L.randomize(x.reshape(1, L.PACKETS, L.ROW), L.shortcut_fields(x), torch.Generator().manual_seed(0))
    assert torch.equal(new, old)


def test_perturb_rejects_unknown_names():
    with pytest.raises(ValueError):
        L.perturb(image(ipv4(6, tcp())), "nopayloads", torch.Generator())


def test_augment_sa_only_adds_field_noise(monkeypatch):
    monkeypatch.setattr(L, "MASK", 0.0)
    monkeypatch.setattr(L, "DROP_PACKET", 0.0)
    x = image(ipv4(6, tcp(TS_OPTIONS)), ipv4(17, udp()))
    fields = L.shortcut_fields(x)
    v = L.augment_sa(x, torch.Generator().manual_seed(0))
    flat = x.reshape(1, L.PACKETS, L.ROW)
    assert torch.equal(v[~fields], flat[~fields])
    assert not v[0, 2:].any()  # empty packets stay empty
