# Sprint 3 pre-registration

Written Nov 4, 2026, before any sprint-3 experiment ran, and committed before the first one. The commit hash is the timestamp. Pass marks may be tightened later, never loosened. Every later change goes in a dated amendment at the end.

## 0. What was already seen before this file was written

So nobody mistakes a re-scoring for a prediction, these numbers were known when the pass marks were set:

- **All of `sprint/REPORT.md`:** the gate and C results for seeds 0–2 and the time split, the reviewer baselines, conformal, and the 5GAD diagnosis.
- **`results/litcvit_seeds.csv`:** clean macro-F1 of F (`ce`) and B (`mask`), 3 seeds per run. B is at least 2 points below F only on IoT-23, USTC and MCFP. **So the brief's N2 pass mark (at least 5 of 7 runs) is expected to fail.** It is kept exactly as the brief wrote it.
- **`sprint/results/kaggle/*/e0_*.csv`:** clean F1 and header-attack success of plain F, the gate, XGBoost and XGBoost behind the gate.
- **`results/litcvit_isn.csv`, `results/litcvit_evasion.csv`:** LitCVit as rebuilt, on MCFP strict and USTC only.

## 1. Compute and scope

- **No GPU.** Colab quota is gone, and the brief forbids sending anything anywhere, so no Kaggle uploads either. CPU only: a laptop with 12 threads.
- **No neural training on 7 runs.** A MobileNetV2 costs about 17 CPU-hours over 7 runs (`sprint/00_inventory.md`). Every neural model here reuses saved weights:
  - random split: seeds 0, 1, 2;
  - time split: seed 0.
- **New training is limited to:**
  - XGBoost models (F′ on bytes, B′ on masked bytes, C variants), minutes each;
  - for the group split only, MobileNetV2 F and B on the three smallest runs (5GAD, MCFP strict, MCFP), seed 0, about 6 CPU-hours. They run only if the CPU is free after E1–E9.
- **Consequences, stated now:**
  - any number from adversarial training (`adv8`) or the AND of two plain models (`and_plain`) exists for seed 0 and the time split only;
  - every time-split neural number is one seed;
  - none of these may appear in an abstract as final (stop rule). BLOCKERS.md lists them.
- **Settings.** A "setting" is (run, split, seed):
  - 7 runs: 5GAD, MCFP, MCFP strict, CICIoT, TII, IoT-23, USTC;
  - random split, seeds 0, 1, 2;
  - time split, seed 0;
  - the group split (§E8), where stated.
- **Test-set rule.** Thresholds and density cut-offs are tuned on **validation** flows only. Validation is 20% of each class's training flows, drawn with `numpy.random.default_rng(10 + seed)`, and models that use it are fit on the other 80%. Each final number touches the test set once.

## 2. Definitions used by every experiment

- **Attack success, conditional** (the earlier sprints' "let through"): of attacked malware flows that the method detects as recorded, the share called Benign after the attack.
- **Attack success, unconditional:** of attacked malware flows, the share called Benign after the attack.
- Both are always saved. Where a pass mark names one, it says which.
- **Drawn attacks** (best of K): a flow counts as cleared if any of the K draws is called Benign. They run on the fixed sample of up to 2,000 malware test flows from `sprint/common.py: suite`.
- **Header attacks,** all inside M:
  - **(a)** `mimic_stack`: donor copy.
  - **(b)** best of 8 `attacks.arbitrary(options=True)` draws.
  - **(m) M-max, new:** best of 8 draws of (b) plus `mimic.shift_sequence`, a fresh ISN per side. Every attacker-settable TCP field in M changes at once.
- **"C"** means `C_A` (trained with added packets) unless stated otherwise. It is the earlier sprint's finalist.
- **Models used as written in `sprint/`:**
  - F = `ce`;
  - B = `mask`;
  - XGBoost = `xgb` (300 trees, depth 8);
  - XGBoost behind the gate = `xgb_gate`;
  - `adv8`, `adv8_gate`, `and_plain`.

## E1. Invariance assertion

- **Scope.** Every setting (28 = 7 runs × 4), every header attack (a), (b) and (m), and every attacked malware flow.
- **Check on B.** Compute B's input before and after the attack (`litcvit.perturb(x, "zero")`) and compare it **byte for byte**.
- **Check on C.** Compare C's 41 features the same way.
- **Pass:** zero differing bytes and zero differing features, everywhere.
- **Fail:** any difference is a leak or a bug. Report the flow, field and attack, and drop every guarantee claim for that branch until it is fixed.
- **Expected to differ:** the flag attack (E3), which is outside M. Its differences are reported. They show the check can fail.

## E2. Bound check

- **Scope.** Each setting, each M-attack (a), (b), (m), and each gate: F+B, F+B+C, and XGBoost+B.
- **Unconditional check:** measured success ≤ u, the share of the same attacked malware that every invariant member (B, plus C for the three-model gate) calls Benign as recorded.
- **Conditional check:** measured success ≤ 1 − (flagged by an invariant member ÷ flagged by the gate), both as recorded.
- **Tolerance zero.** The models are deterministic, and E1 guarantees identical inputs.
- **Pass:** no violation in any cell. Output: a table of measured vs bound.

## E3. Flag forging (outside M)

**The attack (f1).** It applies to each usable TCP malware flow in the sample.
- **Donor.** A benign donor is drawn from the same pool as (a), so the donor rule matches. It must have the same **control bits** (SYN, ACK, FIN, RST) at every packet position. Copy its **free bits** (PSH, URG, ECE, CWR, AE and reserved) packet by packet, and update the TCP checksum (RFC 1624).
- **Urgent pointer.** If URG becomes set, the urgent pointer is set to 0. That counts as cost-realizable.
- **No donor.** If no donor matches, try up to 8 free-bit patterns. Draw each from the per-position free-bit frequencies of the benign training flows, conditioned on the flow's own control bits. Keep the first draw that the evaluated method calls Benign; if none does, keep the last.

**Variants scored**
- f1: flags only.
- f1+a: flags, then donor header copy.
- f1+m: flags, then M-max. Best of 8 draws, as in (m).

**Methods scored:** F, B, C, gate F+B, and gate F+B+C.

**Pre-registered reading:**
- If gate F+B+C's unconditional success under f1+m exceeds its own (m) bound u by more than 1 point on any run, the abstract and GUARANTEE.md must say the guarantee doesn't cover flags. They say it anyway, but then the size of the hole is named.
- A smaller increase is reported as such.

## E4. Masking versus gate: the core test of N2

**Scope**
- **Settings:** all 7 runs; random split, seeds 0–2; time split, seed 0.
- **Methods:**
  - F alone; B alone (masking);
  - gate F+B; gate F+B+C;
  - XGBoost alone; XGBoost+B;
  - `adv8` and `adv8_gate` (seed 0 and time only);
  - `and_plain` (seed 0 and time only).
- **Header-attack success:** the maximum over (a), (b) and (m), **unconditional**.
  - Why unconditional: the conditional measure divides by each method's own detections, so B alone scores 0 by construction, and any detection the gate owes to F alone would count against it.
  - The conditional version is reported beside it.

**Pass mark for N2** (the brief's, unchanged). On at least 5 of 7 runs, using the 3-seed mean, gate F+B must meet all three:
- its clean macro-F1 is within 2 points of F's;
- its clean macro-F1 beats B alone by at least 2 points;
- its header-attack success is no higher than B alone's.

The time split and each seed are scored and reported separately, using the same rule.

**Statistics, per run**
- paired bootstrap of clean macro-F1, gate − F and gate − B: 2,000 resamples of test flows, seed 7;
- 95% percentile intervals;
- Holm correction across the 7 runs at α = 0.05 for the two one-sided tests: gate > B + 0 and gate > F − 2;
- the pass mark itself uses the point estimates; significance is reported beside it.

## E5. Realizability

**Runs.** These runs have raw pcaps: 5GAD, CICIoT, IoT-23, MCFP (which also covers MCFP strict), and USTC. **TII has no raw pcaps locally and is skipped.**

**Forging.** For each run:
- Take at least 200 malware TCP flows, reassembled from the raw pcaps by 5-tuple, first 5 packets.
- Pick a benign donor flow from the same dataset with the same flag sequence.
- Rewrite only M fields to the donor's values: TTL, TOS, IP ID, DF, window, TCP option values of the same length, and an ISN shift.
- Recompute the IP and TCP checksums. Leave payload bytes alone.

**Validation (R1)**
- Reparse with tshark if it can be installed offline. Otherwise use two independent parsers (scapy and dpkt), plus an RFC 1071 checksum check of our own.
- **Pass:** 0 malformed packets and 0 checksum errors.

**Live check (R2 and R3)**
- Set up two Linux network namespaces in WSL.
- A scapy client runs TCP sessions to an unmodified kernel echo server, using donor value sets. Draw 50 per run from the forged flows: TTL, TOS, IP ID, DF, window, MSS, window scale, timestamps.
- **Pass:** every session completes its handshake, data is echoed back intact, and a capture shows the forged values on the wire.

**Classification.** Every forged value is labelled realizable, realizable at a cost, or not realizable (GUARANTEE.md §2). Attack success is recomputed using only the first two classes.

**If a tool can't be installed offline,** that part is a BLOCKER, and it is reported as such.

## E6. Packet-view stress tests for C

**New attacks on the sample.** Each is best of K, applied to malware only.

**(L) Length mimicry**
- Pick a benign donor with the same control bits, as in E3.
- The malware's byte stream is re-framed into the donor's per-packet payload lengths, using pieces of its own payload. Where its data runs out inside the 5 packets, it is padded with zero bytes.
- Headers keep their M fields, and lengths and checksums follow.
- **Cost reported:** padding bytes added, and extra packets needed to carry the payload that has been pushed out of the window.

**(H) Chaff, NetMasquerade-style**
- Insert 1–3 chaff packets at random positions 1–4.
- Each copies the TCP header of the packet before it, with:
  - a random (wrong) sequence number;
  - TTL 1–4;
  - a random IP ID;
  - a random payload whose length is drawn from benign flows' payload lengths at that position.
- K = 16 draws.
- **Delay-only attacks** are not run. No model reads timing (GUARANTEE.md §4), so a delay changes no answer. This is stated as a structural fact, not as a measurement.

**(g64) Adaptive search, K = 64**
- The setting space of (g), extended with chaff (H) and mimicry (L) as further operations.
- K = 16 is (g) itself.

**(d) Held-out families**
- Seen and unseen are reported separately.
- **For C_A** (trained on added packets):
  - seen: (c), (e);
  - held-out: (d), (f), (u), (L), (H).
- **For C_S** (trained on splits):
  - seen: (d), (f);
  - held-out: (c), (e), (L), (H).
- (g) and (g64) mix families and are reported as "mixed".

**Retraining.** New attacks need C's answers on new inputs, so C_A and C_S are **retrained locally on CPU**, with the same recipe, seeds and augmentation bank (`sprint/train.py: bank`). Their clean metrics are reported beside the Kaggle ones. All scores on new attacks use the local models; scores on the old attacks keep the Kaggle logits.

**Pass mark for keeping C as a result** (the brief's). On at least 5 of 7 runs, using the 3-seed mean, gate F+B+C_A must have:
- worst-case **conditional** success over its held-out attacks under 10%;
- Benign recall no more than 1 point below gate F+B.

Otherwise C is described as an add-on driven by known limitations, not as a result.

## E7. Fix for 5GAD and MCFP on the random split: at most two tries

**Fix 1: `C_full`**
- C trained with augmentation over the attacker's whole range: copies 1–3, any position, split sizes 1–32. One augmented copy per malware flow, per family. This is the earlier sprint's suggested next step.
- **Its held-out set:**
  - (L) and (H);
  - split size 24 (unseen);
  - copies = 4 (unseen).

**Fix 2: only if Fix 1 fails**
- `C_full` plus a benign-density flag. A flow that every member calls Benign is flagged anyway when the distance from its C features to the nearest benign training flow is above τ.
- **Distance:** standardised Euclidean, on the 41 features, with k = 1.
- **τ:** set on validation benign flows, so that at most 1% of them are flagged.

**Pass mark for a fix**
- On 5GAD and on MCFP random, 3-seed mean, gate F+B+C_fix must have:
  - worst-case conditional success on its held-out set under 10%;
  - Benign recall no more than 1 point below gate F+B.
- On the other 5 runs, worst-case held-out success must be no more than 2 points worse than gate F+B+C_A.
- Every attempt is reported, failures included. If both fail, the two runs go in the limitations.

## E8. Statistics and splits

**Per setting**
- Bootstrap 95% intervals (2,000 resamples of test flows) for clean macro-F1, Benign recall and each attack success.
- **Over the random-split seeds:** mean, standard deviation, and the min–max range.
- Paired bootstrap and Holm correction for method comparisons, as in E4.

**Group split (new): benign-host-disjoint**
- Benign flows are grouped by source address. Hosts are shuffled (`default_rng(seed)`) and assigned to test until at least 20% of benign flows are in test. The rest go to training.
- Malware is split at random, 80/20 per class.
- **Why only half the split is grouped:** a true source-disjoint split for malware is impossible. Most malware classes come from one or two hosts. `sprint3/results/group_feasibility.csv` records each class's number of sources and its top-source share. It is written before any group-split model is trained.
- **Models on this split:**
  - XGBoost F′ and B′, gate F′+B′, and C_A: all 7 runs, seeds 0–2;
  - MobileNetV2 F and B: seed 0, three smallest runs, only if the CPU is free (§1). This is one seed and labelled not final.

## E9. Operating curve and cost

**Operating curve**
- F's threshold t on its benign probability mass: F says Benign iff the mass is ≥ t.
- Grid: t ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999}.
- Plotted for gate F_t+B, against B_t alone (masking only, the same threshold on B).
- **Axes:** false-alarm rate (1 − Benign recall) against unconditional success under (b) and under (g).
- **Alerts per day:** at an *assumed* 100,000 flows a day, with malware making up 1% and 0.1% of flows.

**Cost**
- Parameters and multiply-adds per flow, counted with forward hooks on Conv2d and Linear.
- CPU latency per flow at batch 1 and batch 256, with 12 threads, the median of 5 runs.
- Reported for F alone, F+B and F+B+C, with ratios to F alone.

## Order of runs

E1 → E2 → E4 (cached logits) → E3 → E6 → E7 → E9 → E8 group split → E5 → group-split neural models, if time allows.

## Amendment 1 (Nov 4, 2026, before any E5 result was scored)

- **E5 donor rule.** On IoT-23 and USTC, no benign donor flow had exactly the same TCP flag byte sequence as the malware flows read, so 0 flows were forged.
  - The donor now only needs the same **control bits** (SYN, ACK, FIN, RST) at each position. A SYN still takes a SYN's fields.
  - If no whole flow matches, each packet takes the M fields of a benign packet with the same control bits, drawn at random. These flows are counted in `per_packet_donors`.
  - The flags stay the malware's own, since they're outside M.
  - Pass marks are unchanged: 0 malformed, 0 checksum errors.
- **E5 source of "malware" flows.** Flows are taken from malware *captures* by 5-tuple, without per-flow labels. IoT-23's malware captures also hold benign flows. Realizability concerns packet validity, not labels, so this doesn't affect E5's pass mark.

## Amendment 2 (Nov 4, 2026, before E6 ran on any setting except the 5GAD seed-0 smoke test)

- **Seen before this amendment:** E6's smoke test, 5GAD, random split, seed 0 (`results/smoke/e6_packets.csv`).
- **Change:** CPU time. The adaptive K = 64 search (g64) runs on random seed 0 and the time split only, not seeds 1 and 2. g64 is a "mixed" family, so it isn't part of C's held-out pass mark. The pass mark and the held-out set are unchanged.

## Amendment 3 (Nov 5, 2026, before E7 ran)

- **Seen before this amendment:** E6 interim results, random seeds 0–2. C_A fails its pass mark on 5GAD, MCFP and MCFP strict, all because of chaff (H, Ha), which is also held-out for C_full.
- **Change: CPU time only.** Fix 2 is *computed* in the same pass as Fix 1 (`e6_packets.py --full`, variant `C_fix2`). It is *reported* as an attempt only if Fix 1 fails its pass mark, exactly as pre-registered.
- **Fix 2's design, as pre-registered:**
  - C_full refit on 80% of the training flows: validation is 20% per class, `default_rng(10 + seed)`.
  - k = 1 standardised Euclidean distance to the 80% benign reference.
  - τ = the 99th percentile of validation benign distances, so that at most 1% of validation benign flows are flagged.
  - Nothing is tuned on test flows.
