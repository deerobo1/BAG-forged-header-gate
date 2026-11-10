"""Header mimicry: give attack flows the header fields of a real benign flow (the paper's evasion test).

Threat model: the attacker controls both ends of a connection, as with malware and its own
command-and-control server, so they choose what both TCP/IP stacks send. They copy these
fields from a benign flow, packet by packet, and change nothing else:
  ttl      IP time to live (one system setting on most hosts)
  window   TCP window
  stack    those two, plus IP type of service, identification and don't-fragment bit, and TCP
           options when the two packets have the same TCP header length
Payload, packet lengths, TCP flags, packet order and sequence numbers stay as recorded, and the
IP and TCP checksums are updated to match, so each packet is one a real stack could send. A real
stack also picks its own initial sequence numbers, which shift_sequence adds.

A benign flow can donate only if its packets have the same TCP flags in the same order, so
that a SYN takes a SYN's fields. Among those, the donor is the flow with the most packets of
matching TCP header length, chosen at random between ties. An attack flow with no donor, or
with any packet that isn't IPv4 TCP, stays as recorded and is left out of the evasion scores.

Works on images [N, 5, 320] or [N, 40, 40] (addresses and ports zero); numpy only. The checksums
cover the addresses and ports, but an update needs only the bytes that change (RFC 1624).
"""

import numpy as np

PACKETS, ROW = 5, 320
HEADER = 80
TCP = 6
STACK = ("tos", "ipid", "df", "ttl", "window", "options")
DF, RST, ACK = 0x40, 0x04, 0x10
NEAR = 2 ** 20  # shift_sequence: two sequence numbers this close belong to one side of a connection


def layout(X):
    """Per packet IP header length, TCP header length and flags [N, 5] (0 for no packet), and a mask of usable flows."""
    X = np.asarray(X).reshape(len(X), PACKETS, ROW)
    present = X[..., :HEADER].any(-1)
    version4 = X[..., 0] >> 4 == 4
    ihl = (X[..., 0].astype(int) & 15) * 4
    ok = ~present | (version4 & (X[..., 9] == TCP) & (ihl >= 20))
    at = lambda offset: np.take_along_axis(X, np.minimum(ihl + offset, ROW - 1)[..., None], -1)[..., 0].astype(int)
    doff = (at(12) >> 4) * 4
    ok &= ~present | ((doff >= 20) & (ihl + doff <= HEADER))
    flags = at(13)
    usable = ok.all(1) & present.any(1)
    return np.where(present, ihl, 0), np.where(present, doff, 0), np.where(present, flags, 0), usable


def choose_donors(attack, donors, rng):
    """For each flow in attack, the position in donors of the flow that lends its fields, or -1."""
    a_ihl, a_doff, a_flags, a_ok = layout(attack)
    d_ihl, d_doff, d_flags, d_ok = layout(donors)
    groups = {}
    for i in np.flatnonzero(d_ok):
        groups.setdefault(d_flags[i].tobytes(), []).append(i)
    groups = {k: np.array(v) for k, v in groups.items()}
    chosen = np.full(len(attack), -1)
    for i in np.flatnonzero(a_ok):
        pool = groups.get(a_flags[i].tobytes())
        if pool is None:
            continue
        score = (d_doff[pool] == a_doff[i]).sum(1)
        chosen[i] = rng.choice(pool[score == score.max()])
    return chosen


def updated_checksum(check, old, new):
    """Internet checksum (2 bytes) after the even-length bytes it covers change from old to new (RFC 1624)."""
    words = lambda b: b[0::2].astype(np.int64) << 8 | b[1::2]
    c = (int(check[0]) << 8 | int(check[1])) ^ 0xFFFF
    c += int((words(old) ^ 0xFFFF).sum() + words(new).sum())
    while c >> 16:
        c = (c & 0xFFFF) + (c >> 16)
    c ^= 0xFFFF
    return c >> 8, c & 0xFF


def patch_packet(row, donor, fields):
    """One IPv4 TCP packet row with the named fields copied from the donor's packet at the same position."""
    new = row.copy()
    ihl, d_ihl = (int(row[0]) & 15) * 4, (int(donor[0]) & 15) * 4
    doff, d_doff = (int(row[ihl + 12]) >> 4) * 4, (int(donor[d_ihl + 12]) >> 4) * 4
    if "tos" in fields:
        new[1] = donor[1]
    if "ipid" in fields:
        new[4:6] = donor[4:6]
    if "df" in fields:
        new[6] = (int(row[6]) & ~DF) | (int(donor[6]) & DF)
    if "ttl" in fields:
        new[8] = donor[8]
    if "window" in fields:
        new[ihl + 14:ihl + 16] = donor[d_ihl + 14:d_ihl + 16]
    if "options" in fields and doff == d_doff:
        new[ihl + 20:ihl + doff] = donor[d_ihl + 20:d_ihl + d_doff]
    new[10:12] = updated_checksum(row[10:12], row[:ihl], new[:ihl])
    new[ihl + 16:ihl + 18] = updated_checksum(row[ihl + 16:ihl + 18], row[ihl:ihl + doff], new[ihl:ihl + doff])
    return new


def shift_sequence(X, rng, only=None):
    """A copy of images X [N, 5, 320] with new initial sequence numbers on each usable TCP flow
    (or those in the mask only), as any stack may choose: each side's sequence numbers, and the
    other side's acknowledgements of them, move by one random offset, so differences survive.

    Addresses are zero in the images, so a packet's side comes from its sequence number: the
    side whose last one is nearer. A flow's first packet opens one side, and a later packet
    opens the other if it is at least NEAR away. A reset with sequence number 0 keeps it, and
    the acknowledgement of a packet without the ACK flag means nothing and stays.
    """
    out = np.asarray(X).reshape(len(X), PACKETS, ROW).copy()
    ihl, doff, flags, usable = layout(out)
    distance = lambda a, b: min((a - b) % 2 ** 32, (b - a) % 2 ** 32)
    for i in np.flatnonzero(usable if only is None else usable & only):
        offsets = [int(v) for v in rng.integers(1, 2 ** 32, 2)]
        last = [None, None]
        for k in range(PACKETS):
            if not out[i, k, :HEADER].any():
                continue
            row, l4, flag = out[i, k], int(ihl[i, k]), int(flags[i, k])
            seq, ack = int.from_bytes(row[l4 + 4:l4 + 8].tobytes(), "big"), int.from_bytes(row[l4 + 8:l4 + 12].tobytes(), "big")
            if flag & RST and seq == 0:
                side = 0 if last[0] is None else 1
            else:
                if last[0] is None:
                    side = 0
                elif last[1] is None:
                    side = 0 if distance(seq, last[0]) < NEAR else 1
                else:
                    side = 0 if distance(seq, last[0]) <= distance(seq, last[1]) else 1
                last[side] = seq
            new_seq = seq if flag & RST and seq == 0 else (seq + offsets[side]) % 2 ** 32
            new_ack = (ack + offsets[1 - side]) % 2 ** 32 if flag & ACK else ack
            new = row.copy()
            new[l4 + 4:l4 + 8] = list(new_seq.to_bytes(4, "big"))
            new[l4 + 8:l4 + 12] = list(new_ack.to_bytes(4, "big"))
            header = int(doff[i, k])
            new[l4 + 16:l4 + 18] = updated_checksum(row[l4 + 16:l4 + 18], row[l4:l4 + header], new[l4:l4 + header])
            out[i, k] = new
    return out


def apply_mimicry(X, donors, chosen, fields):
    """A copy of images X [N, 5, 320], where each flow with chosen[i] >= 0 takes the named fields of donors[chosen[i]]."""
    out = np.asarray(X).reshape(len(X), PACKETS, ROW).copy()
    donors = np.asarray(donors).reshape(len(donors), PACKETS, ROW)
    for i in np.flatnonzero(chosen >= 0):
        for k in range(PACKETS):
            if out[i, k, :HEADER].any():
                out[i, k] = patch_packet(out[i, k], donors[chosen[i], k], fields)
    return out
