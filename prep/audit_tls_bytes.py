"""How much of a TLS flow's image is ciphertext (plan step 8, gap 2).

A flow counts as TLS if one of its first 5 packets starts with a TLS record header, but 5
packets may hold little beyond the handshake. This walks the TLS records in the payload bytes
the image keeps (at most 240 per packet; real lengths come from the IP and TCP headers) and
sorts each byte:
  plaintext   record headers, handshake records before a change cipher spec (ClientHello,
              ServerHello, certificates in TLS 1.2), 2-byte alerts before it, change cipher spec
  ciphertext  bodies of application data records, of alerts longer than 2 bytes (encrypted, on
              a flow that starts mid-connection), and of handshake and alert records after a
              change cipher spec in the flow
  unknown     payload that neither starts a record nor continues one seen earlier

A packet that doesn't start a record continues the last record that ran past its packet.
Addresses and ports are zero in the images, so direction is unknown: one change cipher spec
marks the rest of the flow as encrypted, whichever side sent it. Plaintext therefore errs low
(a NewSessionTicket after it counts as ciphertext), which is the cautious way round here.

  python prep/audit_tls_bytes.py

Writes results/tls_bytes.csv from the packed Kaggle dataset (kaggle/pack_flows.py): one row per
dataset and class with TLS flows, plus a total for benign and attack flows over all datasets.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pcap_to_flows import TLS_VERSIONS

ROOT = Path(__file__).resolve().parent.parent
PACK = Path("C:/data/CNS_proj/kaggle/litcvit-flows")
DATASETS = ["ustc", "ciciot2022", "5gad2022", "mcfp", "iot23", "tii_ssrc23"]
BENIGN = {"Benign", "Normal"}
PACKETS, ROW, HEADER = 5, 320, 80
TCP, MAX_RECORD = 6, 18432  # 2^14 + 2048, the longest TLSCiphertext
CCS, ALERT, HANDSHAKE, APP_DATA = 20, 21, 22, 23
ALERT_LENGTH = 2  # a plaintext alert's body (level, description); longer ones are encrypted


def payload_of(row):
    """Payload the image keeps of one packet row, and the packet's real payload length."""
    h = row[:HEADER]
    if not h.any():
        return row[:0], 0
    if h[0] >> 4 == 4:
        ihl, total, proto = (int(h[0]) & 0x0F) * 4, int(h[2]) << 8 | int(h[3]), h[9]
    else:  # IPv6: fixed 40-byte header, payload length excludes it
        ihl, total, proto = 40, 40 + (int(h[4]) << 8 | int(h[5])), h[6]
    l4 = (int(h[ihl + 12]) >> 4) * 4 if proto == TCP else 8
    n = max(0, total - ihl - l4)
    return row[HEADER:HEADER + min(ROW - HEADER, n)], n


def record_at(p, pos):
    """(content type, header length, body length) of a TLS or DTLS record at pos, or None."""
    if pos + 5 > len(p) or not CCS <= p[pos] <= APP_DATA or bytes(p[pos + 1:pos + 3]) not in TLS_VERSIONS:
        return None
    hdr = 13 if p[pos + 1] == 0xFE else 5  # DTLS adds epoch and sequence number
    if pos + hdr > len(p):
        return None
    length = int(p[pos + hdr - 2]) << 8 | int(p[pos + hdr - 1])
    return (p[pos], hdr, length) if length <= MAX_RECORD else None


def sort_bytes(image):
    """(plaintext, ciphertext, unknown) payload bytes kept in one flow image."""
    plain = cipher = unknown = 0
    encrypted = False  # a change cipher spec has been seen
    left, left_cipher = 0, False  # bytes still to come of a record that ran past its packet, and their kind
    for row in image.reshape(PACKETS, ROW):
        p, n = payload_of(row)
        pos = 0
        if len(p) and record_at(p, 0) is None:
            if not left:
                unknown += len(p)
                continue
            pos = min(left, n)
            if left_cipher:
                cipher += min(pos, len(p))
            else:
                plain += min(pos, len(p))
            left -= pos
        while pos < len(p):
            record = record_at(p, pos)
            if record is None:
                unknown += len(p) - pos
                break
            kind, hdr, length = record
            is_cipher = (kind == APP_DATA or (kind == ALERT and length != ALERT_LENGTH)
                         or (encrypted and kind in (HANDSHAKE, ALERT)))
            end = pos + hdr + length
            plain += min(hdr, len(p) - pos)
            body = max(0, min(end, len(p)) - pos - hdr)
            if is_cipher:
                cipher += body
            else:
                plain += body
            if kind == CCS:
                encrypted = True
            if end > n:
                left, left_cipher = end - n, is_cipher
            pos = end
    return plain, cipher, unknown


def row_of(dataset, name, flows, plain, cipher, unknown, with_cipher):
    total = max(1, plain + cipher + unknown)
    return {"dataset": dataset, "class": name, "flows": flows, "payload_bytes": plain + cipher + unknown,
            "bytes_per_flow": round((plain + cipher + unknown) / max(1, flows), 1),
            "plaintext%": round(100 * plain / total, 1), "ciphertext%": round(100 * cipher / total, 1),
            "unknown%": round(100 * unknown / total, 1), "flows_with_ciphertext%": round(100 * with_cipher / max(1, flows), 1)}


def main():
    rows, totals = [], {"benign": np.zeros(5, int), "attack": np.zeros(5, int)}
    for dataset in DATASETS:
        meta = pd.read_csv(PACK / f"{dataset}.csv", usecols=["class", "tls"], keep_default_na=False)
        X = np.load(PACK / f"{dataset}.npy", mmap_mode="r")
        for name, group in meta[meta["tls"] == 1].groupby("class"):
            counts = np.array([sort_bytes(np.asarray(X[i])) for i in group.index])
            stats = np.array([len(group), *counts.sum(0), (counts[:, 1] > 0).sum()])
            totals["benign" if name in BENIGN else "attack"] += stats
            rows.append(row_of(dataset, name, *(int(v) for v in stats)))
    rows += [row_of("all", f"all {kind}", *(int(v) for v in stats)) for kind, stats in totals.items()]
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "tls_bytes.csv", index=False)
    print(df.to_string())
    print("wrote results/tls_bytes.csv")


if __name__ == "__main__":
    main()
