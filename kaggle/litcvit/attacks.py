"""Attacks past copying a benign flow's fields, under the threat model of mimic.py: the attacker holds both ends.

  arbitrary    random values, per packet, in the fields a stack may set as it likes. Needs no donor, and copying
               a benign flow is one choice among all of these. The gate's guarantee covers it
  add_packets  repeats a packet, so later packets leave the five a model reads
  resegment    sends the first data in tiny segments, so the packets a model reads hold only its first bytes
The last two change bytes the content model reads, so the guarantee doesn't cover them. They test its limits, and
those of any model that reads a flow's first packets. Each packet stays one a stack could send, except that resegment
can't compute a TCP checksum without the addresses and the whole payload; it moves the old one by the change it can see.

Works on images [N, 5, 320] or [N, 40, 40] (addresses and ports zero), like mimic.py; numpy only.
"""

import numpy as np

import mimic as M

PSH, URG = 0x08, 0x20
MIN_TTL = 32  # arbitrary: lower may not reach the other end
MSS, SCALE, STAMPS = 2, 3, 8  # TCP option kinds
MSS_RANGE, MAX_SCALE = (536, 1460), 14  # ordinary_options: the default MSS up to Ethernet's; the largest scale in RFC 7323


def option_spans(row, l4, doff):
    """(kind, start, end) of the value in each of a packet's well-formed MSS, window scale and timestamps options."""
    spans, at = [], l4 + 20
    while at < l4 + doff and row[at]:
        length = 1 if row[at] == 1 else int(row[min(at + 1, M.ROW - 1)])
        if row[at] != 1 and (length < 2 or at + length > l4 + doff):
            break
        if row[at] in (MSS, SCALE, STAMPS):
            spans.append((int(row[at]), at + 2, at + length))
        at += length
    return spans


def option_values(row, l4, doff, kinds=(MSS, SCALE, STAMPS)):
    """Positions of the bytes a stack may set as it likes in a packet's well-formed TCP options: MSS, window scale, timestamps."""
    return [p for kind, start, end in option_spans(row, l4, doff) if kind in kinds for p in range(start, end)]


def packet_sides(rows, ihl, flags):
    """The side of the connection that sent each packet of a flow [5, 320], None for no packet, by M.shift_sequence's rule:
    a packet belongs to the side whose last sequence number is nearer, and a reset with sequence number 0 to the other side."""
    distance = lambda a, b: min((a - b) % 2 ** 32, (b - a) % 2 ** 32)
    sides, last = [], [None, None]
    for k in range(M.PACKETS):
        if not rows[k, :M.HEADER].any():
            sides.append(None)
            continue
        seq = int.from_bytes(rows[k, int(ihl[k]) + 4:int(ihl[k]) + 8].tobytes(), "big")
        if flags[k] & M.RST and seq == 0:
            side = 0 if last[0] is None else 1
        else:
            if last[0] is None:
                side = 0
            elif last[1] is None:
                side = 0 if distance(seq, last[0]) < M.NEAR else 1
            else:
                side = 0 if distance(seq, last[0]) <= distance(seq, last[1]) else 1
            last[side] = seq
        sides.append(side)
    return sides


def ordinary_options(X, rng, only=None, kinds=(MSS, SCALE, STAMPS)):
    """A copy of images X whose usable TCP flows (or those in the mask only) hold new TCP option values that any stack may
    choose and an unmodified receiver accepts. Each side takes an MSS in MSS_RANGE, a window scale up to MAX_SCALE and one
    offset for its timestamp clock, which Linux already randomises per connection, so a clock keeps its pace and an echo
    still matches what it echoes. An empty echo stays zero. Checksums follow."""
    out = np.asarray(X).reshape(len(X), M.PACKETS, M.ROW).copy()
    ihl, doff, flags, usable = M.layout(out)
    number = lambda b: int.from_bytes(b.tobytes(), "big")
    four_bytes = lambda v: list((v % 2 ** 32).to_bytes(4, "big"))
    for i in np.flatnonzero(usable if only is None else usable & only):
        mss = [int(v) for v in rng.integers(MSS_RANGE[0], MSS_RANGE[1] + 1, 2)]
        scale = [int(v) for v in rng.integers(0, MAX_SCALE + 1, 2)]
        clock = [int(v) for v in rng.integers(1, 2 ** 32, 2)]
        for k, side in enumerate(packet_sides(out[i], ihl[i], flags[i])):
            if side is None:
                continue
            row, l4 = out[i, k], int(ihl[i, k])
            donor = row.copy()
            for kind, start, end in option_spans(row, l4, int(doff[i, k])):
                if kind not in kinds:
                    continue
                if kind == MSS and end - start == 2:
                    donor[start:end] = list(mss[side].to_bytes(2, "big"))
                elif kind == SCALE and end - start == 1:
                    donor[start] = scale[side]
                elif kind == STAMPS and end - start == 8:
                    donor[start:start + 4] = four_bytes(number(row[start:start + 4]) + clock[side])
                    if row[start + 4:end].any():
                        donor[start + 4:end] = four_bytes(number(row[start + 4:end]) + clock[1 - side])
            out[i, k] = M.patch_packet(row, donor, ("options",))
    return out


def arbitrary(X, rng, only=None, options=False):
    """A copy of images X whose usable TCP flows (or those in the mask only) hold random values in type of service,
    identification, don't-fragment bit, a TTL of at least MIN_TTL, a nonzero window and, without the URG flag, the
    urgent pointer. An unmodified receiver accepts all of these. With options set (True, or a tuple of kinds) the values
    in option_values too: well-formed, but a scale above MAX_SCALE breaks RFC 7323, though a receiver still uses 14
    (section 2.3), and an unmodified receiver drops timestamps that go backwards (section 5.3), so random timestamps
    need the attacker's own stack at both ends. Checksums follow."""
    kinds = (MSS, SCALE, STAMPS) if options is True else options
    out = np.asarray(X).reshape(len(X), M.PACKETS, M.ROW).copy()
    ihl, doff, flags, usable = M.layout(out)
    which = np.flatnonzero(usable if only is None else usable & only)
    noise = rng.integers(0, 256, (len(which), M.PACKETS, 6), dtype=np.uint8)
    ttl = rng.integers(MIN_TTL, 256, (len(which), M.PACKETS), dtype=np.uint8)
    window = rng.integers(1, 2 ** 16, (len(which), M.PACKETS))
    for n, i in enumerate(which):
        for k in range(M.PACKETS):
            if not doff[i, k]:
                continue
            row, l4 = out[i, k], int(ihl[i, k])
            donor = row.copy()
            donor[1], donor[4:6], donor[6], donor[8] = noise[n, k, 0], noise[n, k, 1:3], noise[n, k, 3], ttl[n, k]
            donor[l4 + 14], donor[l4 + 15] = int(window[n, k]) >> 8, int(window[n, k]) & 255
            if kinds:
                values = option_values(row, l4, int(doff[i, k]), kinds)
                donor[values] = rng.integers(0, 256, len(values), dtype=np.uint8)
            new = M.patch_packet(row, donor, M.STACK)
            if not flags[i, k] & URG:  # the pointer means nothing without the flag
                urgent = new.copy()
                urgent[l4 + 18:l4 + 20] = noise[n, k, 4:6]
                urgent[l4 + 16:l4 + 18] = M.updated_checksum(new[l4 + 16:l4 + 18], new[l4 + 18:l4 + 20], noise[n, k, 4:6])
                new = urgent
            out[i, k] = new
    return out


def has_data(row, ihl, doff):
    return int.from_bytes(row[2:4].tobytes(), "big") > ihl + doff


def payload_kept(X):
    """Payload bytes the images X show a model per usable TCP flow [N], -1 for other flows. Reads only the lengths."""
    x = np.asarray(X).reshape(len(X), M.PACKETS, M.ROW)
    ihl, doff, _, usable = M.layout(x)
    total = x[..., 2].astype(int) << 8 | x[..., 3]
    return np.where(usable, np.clip(total - ihl - doff, 0, M.ROW - M.HEADER).sum(1), -1)


def add_packets(X, copies=2, only=None):
    """A copy of images X where usable TCP flows (or those in the mask only) repeat one packet copies times straight
    after itself: the first ACK with no data, a duplicate ACK, or failing that the first packet, a retransmission."""
    out = np.asarray(X).reshape(len(X), M.PACKETS, M.ROW).copy()
    ihl, doff, flags, usable = M.layout(out)
    for i in np.flatnonzero(usable if only is None else usable & only):
        bare = [k for k in range(M.PACKETS) if doff[i, k] and flags[i, k] == M.ACK and not has_data(out[i, k], int(ihl[i, k]), int(doff[i, k]))]
        k = bare[0] if bare else 0
        out[i] = np.stack([*out[i, :k + 1], *[out[i, k]] * copies, *out[i, k + 1:]][:M.PACKETS])
    return out


def resegment(X, size=1, only=None):
    """A copy of images X where usable TCP flows (or those in the mask only) send their first segment with data in
    pieces of size bytes. Sequence number, total length and IP checksum follow; PUSH stays only on the piece that ends it."""
    out = np.asarray(X).reshape(len(X), M.PACKETS, M.ROW).copy()
    ihl, doff, _, usable = M.layout(out)
    for i in np.flatnonzero(usable if only is None else usable & only):
        first = [k for k in range(M.PACKETS) if doff[i, k] and has_data(out[i, k], int(ihl[i, k]), int(doff[i, k]))]
        if not first:
            continue
        k = first[0]
        row, l4, header = out[i, k].copy(), int(ihl[i, k]), int(doff[i, k])
        total = int.from_bytes(row[2:4].tobytes(), "big") - l4 - header
        seq = int.from_bytes(row[l4 + 4:l4 + 8].tobytes(), "big")
        pieces = []
        for n in range(min(M.PACKETS - k, -(-total // size))):
            piece, start = row.copy(), n * size
            taken = row[M.HEADER:][start:start + size]
            piece[M.HEADER:] = 0
            piece[M.HEADER:M.HEADER + len(taken)] = taken
            piece[2:4] = list((l4 + header + min(size, total - start)).to_bytes(2, "big"))
            piece[10:12] = M.updated_checksum(row[10:12], row[:l4], piece[:l4])
            piece[l4 + 4:l4 + 8] = list(((seq + start) % 2 ** 32).to_bytes(4, "big"))
            if start + size < total:
                piece[l4 + 13] = int(row[l4 + 13]) & ~PSH
            piece[l4 + 16:l4 + 18] = M.updated_checksum(row[l4 + 16:l4 + 18], row[l4 + 4:l4 + 8], piece[l4 + 4:l4 + 8])
            pieces.append(piece)
        kept = np.stack([*out[i, :k], *pieces, *out[i, k + 1:]][:M.PACKETS])
        out[i, :len(kept)] = kept
    return out
