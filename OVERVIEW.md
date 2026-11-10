# Project overview (as of Nov 5, 2026)

Covers everything from the first LitCVit re-check through Sprint 3. Every number comes from `sprint3/REPORT.md` or the files in `results/` and `sprint3/results/`. The L-numbers refer to the claim ledger in `sprint3/REPORT.md` §9.

---

## 1. The project in one paragraph

We started by re-checking LitCVit, a lightweight vision transformer that turns network flows into images and claims to detect malicious *encrypted* traffic. The re-check showed that LitCVit-style models mostly read packet header fields, not encrypted content. It also showed that the sender sets those fields. That turned the project into a security question: can malware forge those fields to get labelled safe? It can, easily. Our contribution is a defence: a **gate** that only marks a flow safe if a second model, one that never sees the forgeable fields, agrees. The gate fully works against header forgery. It only partly works against attacks that add packets, and it costs more false alarms than we hoped. Along the way we found that the datasets let models memorise individual machines. That's a problem for every paper that uses them, LitCVit's included.

---

## 2. Phase 1: re-checking LitCVit (before Oct 31)

**Method**
- LitCVit released no code, so we rebuilt its whole pipeline for all 6 datasets: USTC-TFC2016, CICIoT2022, 5GAD-2022, MCFP, IoT-23 and TII-SSRC-23. MCFP also has a "strict" subset, which makes **7 test runs**.
- Each flow becomes a 40×40 byte image: the first 5 packets, 320 bytes each, with headers in bytes 0–79 and payload after that. Addresses and ports are zeroed.
- We checked the pipeline against YaTC's released data: our images match its 11,694 images **byte for byte**.
- We reimplemented LitCVit, including its self-supervised pretraining and fine-tuning, and ran it on Kaggle GPUs.

**Findings**
- **Reproduction:** our LitCVit comes within 2 points of the paper's reported accuracy on 4 of 6 datasets. **XGBoost beats it on all 6.**
- **Headers are enough:** header bytes alone give at least 92% macro-F1 on every dataset. The shortcuts are:
  - fields that identify the infected machine (IoT-23);
  - TCP clock bytes (5GAD);
  - lab traffic labelled as malware (MCFP);
  - on USTC, one rule ("the first packet's TTL is 32, so it's benign") gets **100.00%** accuracy. All 14 benign captures have TTL 32.
- **Hardly anything is encrypted:** 96.8% of the payload bytes the model sees in "encrypted" TLS attack flows are plaintext. Every Dridex and Shifu TLS flow ends at the ClientHello. When we randomise the header fields, LitCVit's perfect recall on these flows drops to 0, almost all to Benign. **Nothing in this project detects malware inside encryption, and the paper must not claim it.**
- **Random splits flatter the models:** splitting by time instead drops CICIoT2022 from 99.6 to 57.2 macro-F1.
- **Pretraining isn't needed:** a plain MobileNetV2 with no pretraining nearly matches LitCVit at 13M multiply-adds per flow, against LitCVit's 123–263M.

On Oct 31 we decided that a re-check alone wasn't novel enough. This phase became the paper's **motivation**.

---

## 3. Phase 2: looking for a new method (Oct 31 – Nov 2)

We tried three directions. Each had a pass mark set before its results came in.

**(a) Two new training losses: failed as a contribution**
- **Counterfactual contrastive loss (cfnce):** a malware flow is given a benign flow's header, keeps its own label, and a contrastive term pulls it back toward the original flow.
- **One-sided loss:** a header can make a flow look malicious but can never make it look safe. This rests on the idea that an attacker has no reason to fake a *malicious* header.
- Against simple header copying they looked great: on MCFP, the share of malware that slipped through went from 65% to 0.4%.
- They failed in three ways:
  - Our own pass mark for cfnce missed by one run (part 3: 2 of 7 runs, 3 needed).
  - The ablation showed the contrastive term adds nothing over plain data augmentation. It beat the no-contrastive version on only 2–3 of 7 runs, where 4 were needed. **So it's an augmentation, not a new loss.**
  - Random header values broke both losses: up to 71.6% let through. An attacker who tries 8 header sets per flow gets 73–87% through. LitCVit lets 52–71% through.
- Outcome: both losses are comparisons in the paper, not contributions.

**(b) A benchmark of small vision models**
- Eight small models plus a Vision Mamba, all using the same pipeline.
- Every one of them falls to header copying: on average, 36–67% of malware gets through.
- MobileViT has the best clean score (98.3). Vision Mamba is weak on MCFP (48.3).
- **Status:** 58 of 77 runs done. EdgeNeXt didn't train at all (flat loss), so it needs a fix or has to be dropped.
- This is a comparison, not novelty. Mamba for traffic is a crowded field.

**(c) The gate: this became the method** (Section 4)

---

## 4. The method: the gate

**Three models**
- **F**, the full model: a MobileNetV2 that reads the whole flow image.
- **B**, the header-blind model: the same network, but the bytes in **M** are zeroed before every forward pass. **M** is the set of header fields the sender can set:
  - IP: TOS, identification, DF, TTL, IPv4 options (same length);
  - TCP: sequence and acknowledgement numbers, window, urgent pointer, option values (same length);
  - checksums, which follow from the fields above;
  - IPv6 traffic class, flow label and hop limit;
  - UDP and ICMP checksum fields.

  Each field is justified by an RFC in `sprint3/GUARANTEE.md`.
- **C**, the packet-length model: XGBoost on 41 numbers per flow (packet lengths, TCP flags, header sizes), trained on copies of malware flows with packets added.

**The rule:** a flow is marked safe only if **every** model says it's safe. If any model flags it, it's flagged.

**The guarantee:** where the fields in M sit is fixed by bytes that aren't in M (IP version, header lengths, protocol). So an attacker who changes only M can't change anything B sees. B's answer can't move, and the gate can't be made to clear anything B flags. **Under any header-only attack, attack success is at most B's own miss rate.** The proof is one line. We present it as a property we state and test, not as a theorem.

**Why it beats just masking:** B alone is "masking", which an earlier paper (elShehaby and Matrawy) proposed. Masking loses accuracy on some datasets. The gate keeps F's accuracy and lets B veto only a safe verdict.

---

## 5. How we tested it

- **Pass marks set before any run**, committed to git first (`sprint3/PREREG.md`). Every later change was a numbered amendment, also committed before the experiment it changes. This is the main reason the results are believable.
- **7 runs × 3 seeds** on a random split, plus a **split by time** (one seed) and a **split where the benign machines differ between training and test** (3 seeds, XGBoost models).
- **Header attacks:**
  - (a) copy a real benign flow's header;
  - (b) the best of 8 random sets of valid header values;
  - (m) (b) plus new sequence numbers, so every field in M changes at once.

  Checksums are recomputed, and payload, lengths and packet count stay unchanged.
- **Packet attacks** (outside M): adding packets, splitting payloads, copying a benign flow's lengths, and **chaff**, meaning decoy packets the receiver throws away.
- **Can the attacks really be sent?**
  - We checked 7,500 forged packets with three independent packet checkers.
  - We ran 250 live connections against an unmodified Linux machine.
- **Metrics:**
  - macro-F1;
  - Benign recall, which measures false alarms;
  - attack success: the share of malware called safe after the attack, both of the malware the model caught before the attack and of all malware.
- Significance tests: paired bootstrap with Holm correction.
- **Compute:** Kaggle and Colab GPUs; the Sprint 3 analysis ran on the laptop's CPU.

---

## 6. What worked

| Result | Evidence |
|---|---|
| **The attack is real.** Forging only header fields gets ≥ 59% of the malware MobileNetV2 had caught called safe, on 6 of 7 runs and in every seed. USTC: 1.9–32.7%. XGBoost: ≥ 65% on 6 of 7. Our LitCVit: 82.4% (MCFP strict) and 58.3% (USTC) | L1–L4 |
| **The attack can be sent for real.** 0 errors in 7,500 forged packets. 250 of 250 live handshakes completed, and 243 exchanged data. All 7 failures come from copied TCP timestamps | L5, L6 |
| **The guarantee holds as tested.** B saw identical input before and after 649,966 forged flows, and the bound held in 336 of 336 checks | L7, L8 |
| **The three-model gate stops header attacks:** at most **0.25%** gets through on every run, seed and split | L9 |
| **Accuracy is kept:** the gate stays within 0.72 macro-F1 of F on all 7 runs. On MCFP it *beats* F by 8–12 points | L11 |
| **It beats masking where masking hurts** (IoT-23, USTC, MCFP): it recovers 9–51 points of macro-F1 | L12 |
| **It works around other models:** with XGBoost inside the gate, accuracy stays ≥ 92.8 and at most 1.5% gets through | L15 |
| **Simpler alternatives fail:** two plain models that must agree still let 86–100% through on 4 runs, and adversarial training lets 64–78% through, so the gain comes from B being header-blind | L16 |
| **Faking the TCP flags** (outside M) adds at most 0.05 points | L17 |
| **Split payloads are stopped:** C keeps them at ≤ 0.5% everywhere, even though it never trained on them | REPORT §6 |
| **The guarantee holds when benign machines differ between training and test,** wherever the model still recognises benign traffic: ≤ 0.32% gets through, against 12–100% for F alone | REPORT §6b |
| **It's cheap:** twice the parameters and compute (0.70M to 1.40M parameters, 13M to 26M multiply-adds), and 1.75–2× F's CPU time per flow | L20 |

---

## 7. What failed

1. **Gate vs masking (N2): pass mark failed, 3 of 7 runs** (5 needed). On 4 runs (5GAD, MCFP strict, CICIoT, TII), masking alone is as good as the gate. The gate's accuracy advantage is real only where masking hurts. We can't claim it in general.
2. **The packet-length model (N3): pass mark failed, 4 of 7.** Chaff gets **47–100%** through on 5GAD, MCFP and MCFP strict. Chaff copies a real header with a random payload and TTL 1–4. TTL is in M, so B can't see it, and C sees a believable length. Both fixes we set in advance failed:
   - Fix 1 (training on the attacker's full range): 5GAD stays at 98%, and IoT-23 gets 16 points worse.
   - Fix 2 (adding a flag for unusual packet patterns): MCFP drops to 7.5%, a near miss, but it adds more than 1 point of false alarms and leaves 5GAD at 61%.

   **C goes in the paper as a partial add-on and a limitation, not a result.**
3. **Two models aren't enough:** gate F+B lets 30.3% through on CICIoT's time split, because B misses that much there. It's only one seed, but it's why we need all three models.
4. **More false alarms than hoped:** Benign recall is up to **3.86 points** lower than F's (TII: false alarms 6.9% against 3.1%). At an assumed 100,000 flows a day, that's about 6,800 false alerts a day on TII. The earlier "at most 1 point" claim is wrong.
5. **The two new losses** lost to stronger attackers (Section 3).
6. **Ideas we dropped:**
   - a joint loss for both models (it collapsed);
   - time-balanced training;
   - tuning B's threshold;
   - a rule never to clear a flow with fewer than 41 payload bytes (35–38 points of false alarms on USTC);
   - conformal coverage under the time split (CICIoT: 61% against a 95% target).
7. **5GAD** is the weakest run against packet attacks throughout. Its malware differs from benign traffic by one packet length, and two tricks hide it.

**The biggest surprise (L21):** when the benign machines in the test set never appeared in training, Benign recall falls to **7–34%** on 4 of 7 runs (IoT-23 11%, MCFP strict 7%, TII 32%, MCFP 34%), for every model. The models learn *which machines* are benign, not what benign behaviour looks like. Every accuracy number on these datasets is an upper bound, LitCVit's paper included. A split where the *malware* machines differ isn't possible: most malware classes come from one or two machines.

---

## 8. Novelty: what's ours and what isn't

Details in `sprint3/REPORT.md`.

| Claim | Verdict | Closest prior work |
|---|---|---|
| **N1:** forging the shortcut fields clears malware | **Narrowed.** Forging headers isn't new: FlowPaint (2026) does header-only forging with checksum repair, against censorship classifiers. Adversarial Pre-Padding forges TCP fields together with inserted bytes. Ours: an RFC-backed field set, malware detectors (LitCVit included), and a check that the attacks can really be sent | FlowPaint, Adversarial Pre-Padding, Wang et al., Yu et al. |
| **N2:** the gate and its guarantee | **Narrowed.** The argument "a model that can't see a field can't be fooled through it" goes back to Incer et al. (2018). Masking is elShehaby and Matrawy's. Ours: keeping the full model and letting a header-blind model veto only a safe verdict, on raw bytes, with the guarantee checked flow by flow | Incer et al., elShehaby & Matrawy |
| **N3:** the packet-length model | **Close to dead as a headline.** CertTA (USENIX Security 2025) already *certifies* robustness to packet insertion. Ours is empirical and partial | CertTA, BARS |
| **N4:** evaluation lessons | **Narrowed.** Comparing splits isn't new. Reporting *attack success* across random, time and benign-host splits is, as is the host-memorisation finding | Yu et al. |

**Honest framing:** this is a security measurement paper with a simple, model-agnostic defence. The contribution is the evidence (7 runs, 3 splits, checks that the attacks can be sent, a checked guarantee, failures reported honestly), not a clever new mechanism. Write "to our knowledge", never "first".

---

## 9. What we must not claim

- detection inside encryption;
- a guarantee for anything outside M (packet count, sizes, payload, timing, flags);
- that C makes the detector robust to packet attacks;
- being first to forge header fields;
- "50–100% on seven datasets" (there are six datasets, and USTC is 1.9–32.7%);
- LitCVit results beyond the 2 runs that have them;
- any time-split number as final (one seed);
- that it works on unseen machines, or on real traffic.

---

## 10. What's left before Nov 10

1. **Rewrite the paper around the gate.** `paper/main.tex` still has the earlier LitCVit re-check framing. Every claim should map to a line in the claim ledger (`sprint3/REPORT.md` §9). Draft abstracts for a security venue and a vision venue are in §11 of that report, and the LaTeX tables are in `sprint3/tables/`.
2. **The small-model benchmark:** finish the last 19 runs or report it as partial, and fix or drop EdgeNeXt.
3. **Sudarshan:** please send your department and email for the author block.
4. **After Nov 10, for the conference version:** more seeds on the time split, a neural model on the benign-host split, a comparison with CertTA, and a wider literature review.

This project is private: please don't share this file or the results outside the team.
