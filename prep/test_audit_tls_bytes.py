"""Tests for audit_tls_bytes on hand-built flow images. Run: pytest prep/test_audit_tls_bytes.py"""

import struct

import numpy as np

from audit_tls_bytes import HEADER, PACKETS, ROW, sort_bytes


def packet(payload, real_length=None):
    """One image row: IPv4 and TCP headers (20 + 20 bytes), then the payload the image keeps."""
    n = len(payload) if real_length is None else real_length
    ip = struct.pack("!BBH", 0x45, 0, 40 + n) + bytes(5) + b"\x06" + bytes(10)
    tcp = bytes(12) + b"\x50\x18" + bytes(6)
    return (ip + tcp).ljust(HEADER, b"\0") + payload[:ROW - HEADER].ljust(ROW - HEADER, b"\0")


def record(kind, length):
    return bytes([kind, 3, 3]) + struct.pack("!H", length) + b"\xAA" * length


def image(*rows):
    rows = list(rows) + [bytes(ROW)] * (PACKETS - len(rows))
    return np.frombuffer(b"".join(rows), np.uint8)


def test_empty_packets_have_no_bytes():
    assert sort_bytes(image(packet(b""), packet(b""))) == (0, 0, 0)


def test_client_hello_is_plaintext():
    assert sort_bytes(image(packet(record(22, 100)))) == (105, 0, 0)


def test_application_data_body_is_ciphertext_and_its_header_is_not():
    assert sort_bytes(image(packet(record(23, 100)))) == (5, 100, 0)


def test_handshake_after_change_cipher_spec_is_ciphertext():
    assert sort_bytes(image(packet(record(20, 1) + record(22, 40)))) == (6 + 5, 40, 0)


def test_alert_is_ciphertext_unless_two_bytes_long():
    assert sort_bytes(image(packet(record(21, 2)))) == (7, 0, 0)
    assert sort_bytes(image(packet(record(21, 26)))) == (5, 26, 0)


def test_only_kept_bytes_count():
    assert sort_bytes(image(packet(record(23, 1000), 1005))) == (5, 235, 0)


def test_record_running_past_its_packet_continues_with_its_kind():
    certificate = record(22, 2000)
    first = packet(certificate[:1460], 1460)
    second = packet(certificate[1460:], 545)
    assert sort_bytes(image(first, second)) == (240 + 240, 0, 0)
    data = record(23, 2000)
    assert sort_bytes(image(packet(data[:1460], 1460), packet(data[1460:], 545))) == (5, 235 + 240, 0)


def test_payload_outside_any_record_is_unknown():
    assert sort_bytes(image(packet(b"\x01\x02\x03\x04\x05\x06\x07\x08"))) == (0, 0, 8)
    assert sort_bytes(image(packet(record(22, 10) + b"GET /"))) == (15, 0, 5)


def test_garbled_length_is_not_a_record():
    assert sort_bytes(image(packet(b"\x17\x03\x03\xFF\xFF" + b"\xAA" * 20))) == (0, 0, 25)
