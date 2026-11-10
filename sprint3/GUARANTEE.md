# The guarantee, stated before it is tested (Nov 4, 2026)

## 1. Inputs, and what each model reads

- **Input.** Each flow is an image of its first 5 packets × 320 bytes, built by `kaggle/litcvit/litcvit.py: load`:
  - bytes 0–79 of each row are the IP and transport headers, zero-padded;
  - bytes 80–319 are payload;
  - the Ethernet header is gone, and addresses and ports are zero in every image.
- **F** (`ce`) reads the whole image.
- **B** (`mask`) reads the image with the bytes in **M(x)** set to zero before every forward pass (`light.FlowNet.embed` → `litcvit.perturb(x, "zero")` → `litcvit.shortcut_fields`).
- **C** (`sprint/c1_length_view.py: features`) reads 41 numbers per flow. Per packet: present, IP version, protocol, IHL, total length, TCP data offset, TCP flags, payload length. Plus the total payload shown.
- **The gate** clears a flow (calls it Benign) only if every member calls it Benign (`sprint/common.py: combine`).

**Where M sits.** M(x) is a set of byte positions. It is computed per packet from four values: IP version, IHL, protocol and TCP data offset (or UDP length). **None of these four is in M**, so an attacker who changes only bytes in M can't move M. This is the property the proof needs, and E1 checks it.

## 2. The mutable set M

**Columns**
- **Controls:** who sets the field. *Sender* is the host that emits the packet; under this threat model, the malware or its own server. *Path* means routers may change it. *Derived* means it is computed from other bytes.
- **Realizable** (our scale, see §5):
  - **free**: an ordinary socket API sets it;
  - **cost**: needs a custom or raw-socket stack, or breaks an optional mechanism;
  - **no**: the packet would stop working.

### IPv4

| Field (bytes) | Controls | Realizable | One-line justification |
|---|---|---|---|
| TOS: DSCP and ECN (1) | Sender; path may remark DSCP or set CE | free | RFC 2474 and RFC 3168. `IP_TOS` sets it. Receivers don't reject any value |
| Identification (4–5) | Sender | cost | RFC 6864 §4: for atomic datagrams the value has no meaning and may be anything. Linux picks it, so exact values need a raw socket |
| Flags and fragment offset (6–7) | Sender | DF free; MF or offset ≠ 0: **no** | RFC 791. DF via `IP_MTU_DISCOVER`. Setting MF or an offset turns the packet into a fragment that is never reassembled. The attacks change DF only |
| TTL (8) | Sender; path decrements it | free while ≥ path length | RFC 791. `IP_TTL` sets it. The attacks use TTL ≥ 32 (`attacks.MIN_TTL`) |
| Header checksum (10–11) | Derived | – | RFC 791, recomputed (RFC 1624). It is in M only because it follows the fields above |
| Options (20 … IHL), **same IHL** | Sender | cost | RFC 791. Option bytes may change. Their total length may not, because IHL isn't in M. Many paths drop IP options |

### TCP

| Field (bytes) | Controls | Realizable | One-line justification |
|---|---|---|---|
| Sequence and acknowledgement numbers (4–11) | Sender (ISN); ack is the peer's sequence | free (ISN) | RFC 9293 §3.4.1, RFC 6528: the ISN is chosen per connection. The later numbers follow from the payload lengths, which aren't in M |
| Window (14–15) | Sender (its receive buffer) | free in range; exact values per packet: cost | RFC 9293 §3.8.6. `SO_RCVBUF` and `TCP_WINDOW_CLAMP` set it |
| Checksum (16–17) | Derived | – | RFC 9293 §3.1, recomputed |
| Urgent pointer (18–19) | Sender | cost | RFC 9293: meaningful only with URG. The attacks change it only when URG is clear, where receivers ignore it |
| Options (20 … data offset), **same data offset** | Sender | MSS free (`TCP_MAXSEG`); window scale, timestamps and kind bytes: cost | RFC 9293, RFC 7323. Values may change, total length may not. Random timestamps fail PAWS (RFC 7323 §5.3) unless both stacks are the attacker's. A scale above 14 is clamped to 14 by receivers (§2.3) |

### UDP, ICMP, IPv6

| Field (bytes) | Controls | Realizable | One-line justification |
|---|---|---|---|
| UDP checksum (6–7) | Derived, or 0 (no checksum) | free | RFC 768 |
| ICMP and ICMPv6 checksum and the rest of the 8-byte header (2–7) | Sender: identifier and sequence for echo; derived checksum | free | RFC 792, RFC 4443 |
| IPv6 traffic class, flow label, hop limit | Sender | free | RFC 8200, RFC 6437 |

### Not in M (B and C read these)

- IP version and IHL
- total length
- protocol
- TCP data offset and **flags**
- UDP length
- ICMP type and code
- every payload byte
- which packets exist, and their order

Not in any image (so no model reads them):
- **addresses and ports**, which preprocessing zeroes;
- **timing**, since the images hold no timestamps.

## 3. Are TCP flags in M? No.

- **Control bits** SYN, ACK, FIN, RST define the connection's state machine (RFC 9293 §3.10). Changing one doesn't forge a field, it sends a different packet sequence. That belongs with the packet attacks, outside M.
- **The "free" flag bits** are another matter:
  - PSH: advisory, RFC 9293 §3.9.1;
  - URG: with its pointer;
  - ECE and CWR: RFC 3168, accepted after ECN negotiation, and tolerated otherwise;
  - AE (formerly NS) and the reserved bits: must be zero, receivers ignore them, middleboxes may drop.

  An attacker who controls both ends **can set these freely**. They are attacker-settable fields **outside M**, and both B and C read them.

**Consequences:**
1. The guarantee covers M only. It does **not** cover forged free flag bits, for either B or C.
2. C's claim to be invariant holds for M and nothing more. C is an **empirical add-on** for packet attacks. It is not a second invariant model with respect to flags.
3. Experiment E3 measures what forging the free bits does. If the gate's attack success rises above the bound for M-only attacks, the abstract must say that flags are outside the guarantee.

## 4. The statement

**Setting.**
- x is a flow image. The F, B and C weights are fixed, in eval mode, so each model is deterministic.
- An **M-attack** maps x to a flow x′ that differs from x only in bytes in M(x).
- Write B(x) = Benign when B's answer is a benign class, and the same for the others.
- The gate G clears x iff F(x), B(x) and C(x) are all Benign. The two-model gate leaves out C.

**Lemma (invariance).** For every M-attack, M(x′) = M(x), because version, IHL, protocol and data offset are unchanged.
- So B's input is unchanged: x′ with M zeroed equals x with M zeroed, byte for byte. Hence B(x′) = B(x).
- C's 41 features read only bytes outside M, so C(x′) = C(x). "Present" reads any header byte, but the version nibble of a present packet is never zero.

**Claim.** Let S be a set of malware flows. Then, for any attacker who applies one M-attack per flow, adaptive and with full knowledge of the weights:

1. **Unconditional.** The share of S that G clears after the attack is at most **u = the share of S that B (and C, for the three-model gate) call Benign as recorded**, i.e. the invariant members' miss rate.
2. **Conditional** (the paper's "let through"). Among flows of S that G flags as recorded, the share G clears after the attack is at most the share that G flags *only* because of F. This equals 1 − certified ÷ detected, where certified is the share of S flagged by B (or C).

*Proof.* G(x′) = Benign implies B(x′) = Benign, and C(x′) = Benign for the three-model gate. By the lemma that means B(x) = Benign (and C(x) = Benign). ∎

**Assumptions, all checked or stated:**
- (A1) **Same packets.** The monitor sees the packets the receiver acts on. Chaff that only the monitor sees is a packet attack, outside M.
- (A2) **Same images.** Flow assembly and the 5-packet, 320-byte cut don't depend on bytes in M. They depend on the 5-tuple and capture order, not on TTL, IP ID or sequence numbers.
- (A3) **M-only changes.** The attacker changes nothing outside M. In particular, option lengths stay the same and nothing is fragmented.
- (A4) **Deterministic models.** True for fixed weights in eval mode on one machine.

**What the claim does NOT cover:**
- packet count, packet sizes, packet order, segmentation;
- payload bytes;
- timing (no model reads it, so a delay-only attack can't change any answer; the claim adds nothing there);
- TCP flags, including the free bits;
- IHL or data-offset changes (adding or removing options);
- fragmentation;
- addresses and ports. They are zero in our images. A deployment that keeps them needs them in M or masked;
- the training data (poisoning);
- the clean error rate. u is whatever B and C miss without any attack.

## 5. Realizability scale used in this sprint (ours, not Alatawi's)

Alatawi 2026 is UNVERIFIED (NOVELTY_MATRIX.md), so its taxonomy isn't used. We use four levels:

| Level | Meaning | Checked by |
|---|---|---|
| R0 | Change in feature space or image space only | – |
| R1 | Every packet parses, lengths are consistent, checksums are correct | Rewritten real pcaps, reparsed by two independent parsers (E5) |
| R2 | An unmodified receiver stack accepts the packets | A live session between two network namespaces, with the forged values on the wire (E5) |
| R3 | The application exchange completes end to end, with the attacker controlling both ends | The same netns test, with data echoed back (E5) |

Planned placement of the attacks:
- (a), (b) and the M-max attack: R1 by construction (checksums recomputed), R2 and R3 only for the values E5 replays. (b)'s random timestamps and scales above 14 are "cost".
- add-packet and resegment: R1, except for the TCP checksum caveat that `attacks.py` documents.
- chaff (wrong sequence number, short TTL): R2 by design. The receiver discards it, which is the point.
- length mimicry with padding: R1. It needs both ends, since the padding changes the application byte stream.
- E5 replaces "planned" with what was measured.
