# The architecture, explained

Figure: `paper/figs/architecture.pdf` (source `paper/figs/architecture.tex`), Fig. 1 of `main.pdf`.

## What each box is

**Flow image x (left).** One network flow, cut to its first 5 packets. Each packet becomes 320 bytes: bytes 0–79 are the IP and TCP/UDP header (zero-padded if shorter), bytes 80–319 are payload. The five packets stacked make 1,600 bytes, drawn as a 40×40 grey-scale image. In the drawing, each packet is 8 rows of 40 bytes. Its first 2 rows are the header, and the red cells are the header bytes an attacker can rewrite. Addresses and ports are zeroed before anything else happens, so no model sees them.

**The mutable set M(x) (bottom strip).** The bytes of one IPv4 + TCP header (here with 12 bytes of TCP options) that the sender is free to set:
- *IPv4:* type of service, identification, flags and fragment offset, TTL, header checksum.
- *TCP:* sequence and acknowledgement numbers, window, checksum, urgent pointer, option values.

Grey cells are header bytes that are not in M: version and header length, total length, protocol, TCP data offset and flags. Every payload byte is outside M too. Where M sits in a packet depends only on the grey cells (header length, protocol, data offset), so changing the red cells cannot move M.

**F (top branch).** A MobileNetV2 that reads the whole image. It is the accurate detector, and the one an attacker can fool by rewriting red cells.

**Zero every byte in M(x), then B (middle branch).** The same architecture, trained and tested on the image with every red cell set to zero. B never sees a byte the attacker controls.

**41 numbers, then C (bottom branch).** For each packet: present or not, IP version, protocol, IP header length, total length, TCP data offset, TCP flags, payload length; plus the total payload shown. XGBoost reads these. None of them is in M.

**Benign or Flag.** Each member names a class. "Benign" means it named a benign class; anything else is a flag.

**Unanimity gate.** The flow is cleared (Benign) only if F, B and C all say Benign. If any one flags it, the verdict is Flag.

**Shaded region, line styles, legend.**
- The blue region (the mask and B, with its solid arrows) is what the guarantee covers.
- Dashed paths (F, C) are empirical. Their accuracy and robustness are measured, not guaranteed.

**Inset (right).** The attacker writes TTL, IP ID, window, initial sequence number and option values into red cells. F reads the changed bytes. On the way to B they are overwritten with zeros, so B gets the same input as before the attack.

## Why B cannot be moved by an attacker who changes only M

1. B's input is "x with M(x) zeroed".
2. The attacker changes only bytes inside M(x), giving x′.
3. The positions of M are computed from version, header length, protocol and data offset, none of which is in M. So M(x′) = M(x).
4. Zeroing the same positions in x and x′ gives byte-identical images, because the two differ only at those positions.
5. B is a fixed, deterministic function of its input, so B(x′) = B(x).

The gate can clear x′ only if B(x′) is Benign, which means B(x) was already Benign. **So under any M-only attack, the share of malware the gate clears is at most the share B misses with no attack at all.** We checked step 4 on all 649,966 attacked flows (0 differing bytes) and the bound in 336 cells (0 violations).

## Why F stays in the gate

Zeroing M throws away real signal on some datasets. On IoT-23, B alone scores 47.5 macro-F1 against F's 99.1. On USTC-TFC2016 it is 87.9 against 96.9, and on MCFP 76.4 against 80.4. Keeping F and letting B veto only F's Benign verdicts gives back most of that: 98.4, 96.9 and 90.2 respectively. F can only *add* flags, so its weakness to forged fields can never clear a flow B flags. The price is false alarms: the gate flags whatever either member flags (up to 3.86 points of benign recall lost on TII-SSRC-23). On the other four runs B alone is as accurate as F, so there the gate buys nothing over masking. This is why the pre-registered gate-versus-masking test failed (3 of 7 runs).

## Why C is there, and what it does not cover

B's guarantee assumes the packets themselves don't change. An attacker can also add packets, split the payload differently or pad it, which changes what B sees. C reads only lengths and flags and is trained with copies of malware that have extra packets, as a cheap extra veto.

What C does not cover:
- **It is empirical.** It was tested only against our attack suite.
- **Its TCP flags are attacker-settable.** The free bits (PSH, URG, ECE, CWR, reserved) are outside M, so C is not invariant to them.
- **Chaff defeats it on three of seven runs:** 99.7% on 5GAD-2022, 54.4% on MCFP and 47.2% on MCFP strict. Chaff packets copy a real header but carry TTL 1–4 and a wrong sequence number, so the receiver drops them while the monitor counts them. TTL is in M, so B can't see it, and C sees a plausible length.
- **Length mimicry** makes C's 41 numbers exactly a benign donor's, so C alone lets 100% through by construction. F and B still catch most of it.

## Worked example (illustrative, not a measured flow)

Take a Dridex command-and-control flow from MCFP strict: SYN, SYN-ACK, ACK, a TLS ClientHello and the server's reply. As recorded, all three members flag it.

**The attack.** The attacker's own stack sends the same five packets with the M fields of a benign Windows host: TTL 128, the host's window and IP ID pattern, its TCP option values (same lengths) and a fresh initial sequence number. Checksums are recomputed, and payload, lengths and flags stay as they were.

1. **F** reads the new TTL, window and options. Those are exactly the bytes it learned to associate with benign traffic, so it now says **Benign**. On MCFP strict the worst header attack clears 100% of the malware F detects (`sprint3/results/summary_n1.csv`, 3-seed mean; Fig. 2 of the paper).
2. **B** zeroes every M byte. The image it gets is byte-for-byte the one it got before the attack, so it says what it said before: **Flag**.
3. **C** reads lengths and flags, which didn't change, so it says **Flag**.
4. **Gate:** not all three say Benign, so the verdict is **Flag**. Measured on MCFP strict, the two-member gate lets 0.04% of attacked malware through (3-seed mean, unconditional). That 0.04% is malware that B already missed with no attack.

## What the diagram does not show

- **Packet count and order.** Only the first five packets are read. Adding, removing or reordering packets changes the image, and the guarantee says nothing about that (chaff is the measured failure).
- **Packet sizes and payload.** Total length and payload bytes are outside M, so the guarantee doesn't cover changes to them. Length mimicry and re-segmentation are measured, not guaranteed.
- **Timing.** The images hold no timestamps, so no model reads timing. Delaying packets changes no answer, but that is because the detector is blind to timing, not because it is robust.
- **Option lengths and fragmentation.** Adding or removing options changes IHL or the data offset and moves M. Fragmenting changes the packets. Both are outside the attack model.
- **Training data.** Poisoning is not considered.
- **B's clean error.** The bound is B's miss rate, which the diagram can't show. On CICIoT2022's time split (one seed) it is 30.34% under attack (b).
