# Sprint pre-registration

Written Nov 3, 2026, 09:00 IST, before any sprint experiment ran (only `sprint/timing.py`, a speed test, has run). Committed to git before the first run; the commit hash is the timestamp. Pass marks may be tightened later, never loosened.

## Compute and scope

- **No GPU.** Colab's GPU quota is used up (user, Nov 3), so the brief's "6 GPU hours" are not available. Laptop CPU only: 12 threads, one job at a time (two at once is 2.7x slower each).
- Measured: ~11 ms per flow per epoch for MobileNetV2 training. One 30-epoch model costs ~42 min (5GAD), ~62 min (MCFP strict), ~80 min (MCFP); all 7 runs ~17 h.
- **CPU budget: 45 hours**, Nov 3 09:00 to Nov 5 night, leaving Nov 6-10 for writing. Nothing is started that the queue cannot finish within this budget.
- **Screening set for trained ideas: 5GAD, MCFP strict, MCFP, random split, seed 0.** These are the three cheapest runs and the three with the worst packet-attack results in the notes (98%, 89%, 62%). No new idea can be screened on 7 runs on CPU.
- **Consequence, stated in advance:** a trained idea screened on 3 runs can pass on at most 3 of 7 runs, so its verdict can be at best **MAYBE** unless a GPU finalist job runs. The finalist job is prepared in `sprint/colab/` for when GPU quota returns.
- Ideas that need no new neural training (1C with a tree model, 1D(iii), 2A, 2B) use the existing F and B weights, so they run on **all 7 runs, seeds 0-2 (random split), and seed 0 on the time split**, and can reach ADOPT.

| Queue (in order) | Est. CPU h |
|---|---|
| E0: attack suite (a)-(g) on the current gate, plain `ce`, and gate-wrapped XGBoost, seed 0, 7 runs | 2 |
| 1C: length view (XGBoost), 4 variants, 7 runs x 3 seeds + time | 1.5 |
| 1D(iii), 2A, 2B from saved logits | 0.5 |
| 1B: B trained with add-family or split-family augmentation, 3 runs | 7 |
| 1A: noisy-AND joint loss, 3 runs | 6.5 |
| 1D(i): second `ce` with another initialisation, 3 runs | 3 |
| 1D(ii): adversarial training, best-of-8 donors in the loop, 3 runs | 9 |
| 2C: time-block-balanced B, CICIoT time split | 2 |
| E0 for seeds 1, 2 and the time split | 3 |
| 1A worst-of-K, 3 runs, **only if 1A shows signal** | 12 |
| Total | ~46 |

Tier 3 (Vision Mamba, EdgeNeXt) is out of budget on CPU and is not attempted unless GPU returns.

## Attack suite (same everywhere)

Applied to malware test flows only. Benign flows stay as recorded.

- **(a)** `mimic_stack`: copy a same-network benign donor's TOS, IP ID, DF, TTL, window, options (`light.test_sets`).
- **(b)** best-of-8 random valid header sets, options included (`attacks.arbitrary`, seeds 100-107), on a fixed sample of up to 2,000 malware flows (sample seed 1, as in `limits.py`). A flow evades if any draw is called Benign.
- **(c)** `add_packets(copies=2)`. **(d)** `resegment(1)`. **(e)** (a) + (c). **(f)** (a) + (d). All exactly as in `limits.py`.
- **(g) adaptive packet/payload attacker:** K = 16 random settings per flow, on the same sample as (b); the flow evades if any setting is called Benign. Each setting draws:
  - header: as recorded, or copied as in (a);
  - operation: add, split, or add then split;
  - copies: 1-3;
  - position: the first bare ACK (as `add_packets`), or any present packet 0-4;
  - split size: 1, 2, 4, 8, 16 or 32 bytes.

  "Packet size" in the brief is read as the split size (which sets the size of the new packets) and as the choice of which packet (and so which size) is repeated. Inserting packets with new payload would change the application byte stream and is not modelled.
- **Unseen parameters:** (g-unseen) is (g) restricted to copies = 3 and split sizes 8, 16, 32, none of which appear in any training augmentation below.

## Metrics, per run

- Clean macro-F1 of the method's multiclass answer.
- Benign recall on clean flows (false-alarm cost).
- **Evaded** per attack, the notes' definition: of malware flows the attack changed that the method detects as recorded, the share called Benign. The unconditional share (of all attacked malware) is also saved.
- **Certified:** share of malware the header-blind views flag (B, or B or C for 1C).
- Worst-case packet/payload evaded = max over (c)-(g).

## Leakage rule

Augmentation families:
- **A (add):** copies 1-2, random present position.
- **S (split):** sizes 1, 2, 4.

A model trained on A is "seen" on (c), (e) and "held-out" on (d), (f), and vice versa. (g) mixes both families and is reported for each. (g-unseen) tests parameters outside both. Seen and held-out are reported separately. A model trained on donor-copied headers (1D(ii), 1A worst-of-K) is seen on (a) and held-out on (b), which uses random values, not donors.

## Pass marks (per run, versus the current gate on the same run, split and seed)

**PASS on a run** if clean macro-F1 >= gate - 2.0, AND at least one of:
- **(i)** worst-case evaded over (c)-(g) < 5.0%, with Benign recall >= gate - 5.0. For ideas trained on an attack family, (i) must hold on the held-out attacks as well as the seen ones; passing on seen attacks only counts as MAYBE.
- **(ii)** Benign recall >= gate + 1.0, with evaded on (a) and (b) each <= max(gate's value + 0.5, 1.0).
- **(iii)** (time split, CICIoT only) certified > 90%.

**Verdict:**
- **ADOPT:** passes on >= 5 of 7 runs on the random split. For finalists, also mean over 3 seeds, and on the time split, it must pass on >= 5 of 7 runs.
- **MAYBE:** passes on 3-4 runs, or only on held-out-free (seen) attacks.
- **DROP:** otherwise.

**Screening signal** (decides whether to continue a tier-1 idea): passes on >= 2 of the 3 screening runs, or for 1A, Benign recall >= gate + 1.0 on >= 2 of 3 with no pass mark broken by more than 2 points. No signal means stop that idea.

## Ideas: fixed design choices

- **Baselines:** the current gate (F = `ce`, B = `mask`, decision form); plain `ce`; XGBoost (300 trees, depth 8, as in `evasion.py`) alone and gated by B. All three are scored on the full suite.
- **1A noisy-AND:**
  - F (full view) and B (masked view) are trained jointly from scratch: 30 epochs, batch 256, lr 1e-3, class-balanced batches.
  - pF and pB are the softmax mass on benign classes; clear = pF * pB.
  - Loss: -log(clear) averaged over benign flows, plus -log(1 - clear) averaged over malware flows (the two halves weighted equally), plus 0.1 x class-weighted CE on each of F and B, plus 0.1 x (-log(1 - pB)) on malware (B's own flag term).
  - Weights are fixed in advance; no tuning (no budget).
  - Worst-of-K variant: pF on malware is the max over the recorded flow and K = 8 donor-forged copies (`light.forge` with 8 donors). The worst copy is chosen without gradient, then trained with gradient. **Deviation:** a hard max instead of log-sum-exp smoothing, which would need gradients through all 8 copies (8x the cost).
  - Decision: cleared iff F's and B's argmax are both Benign.
- **1B:** B (masked) trained with family A or family S.
  - Each malware training flow gets one precomputed attacked copy with random parameters, added as an extra training flow of its class. Benign flows are unchanged.
  - F stays the existing `ce`.
  - **Deviation from the brief ("train both F and B"):** a flow is flagged whenever B flags it, and under (e)/(f) the copied header already fools F. So only B can stop those attacks, and training F too would double the cost. A GPU finalist trains both.
- **1C length view C:**
  - Per packet k = 0-4: present, protocol, IP header length, IP total length, TCP data offset, TCP flags, payload length; plus total payload shown.
  - No direction: the only way to infer it is sequence numbers, which are attacker-controlled. No timing, which is not available (inventory).
  - Model: XGBoost (200 trees, depth 6), class-weighted.
  - Variants: C-plain, C-A, C-S, C-AS (augmented as in 1B).
  - Gate (F, B, C): cleared iff F, B and C all say Benign.
  - Compared with the hard rule "never clear a TCP flow with < 41 payload bytes" on the same runs, including IoT-23 and USTC.
- **1D(i):** a second `ce` model, same split (seed 0), training seed 1. Gate = cleared iff both say Benign.
- **1D(ii):** single model; class-weighted CE on recorded flows, plus CE on each malware flow's worst of 8 donor-forged copies (max benign probability, chosen without gradient), weighted like `oneside` (ALPHA/k).
- **1D(iii):** B overrules F only when B's malware-minus-benign log-probability margin exceeds t. t is chosen on half of the test flows (fixed seed) for a Benign-recall budget of 0.5, 1, 2 points over F alone, and scored on the other half (as `gate_report`'s curve).
- **2A abstain:**
  - Outcomes: both Benign → clear; both malware → alert; disagree → defer.
  - Report coverage, risk on decided flows, and alerts and deferrals at 1% and 0.1% malware prevalence for an **assumed** 100,000 flows per day. The volume is a parameter, not a measurement.
- **2B conformal:** B's score is its malware probability mass; the threshold is the floor(alpha (n+1))-th lowest calibration malware score, alpha = 0.05.
  - Random split: calibrate on a random half of the test malware, evaluate on the other half.
  - Time split, static: calibrate on the latest 20% (by `ts_first`) of B's training malware flows. This is in-sample and therefore optimistic, and is stated as such.
  - Time split, rolling: calibrate on the earliest half of the test period, evaluate on the later half.
  - Report coverage vs 95% and the Benign recall cost.
- **2C:** B on the CICIoT time split trained with time-block-balanced sampling: groups = class x time quartile of `ts_first` within the training flows, equal draw probability per group. Pass mark (iii).

## Reading of ambiguities

- "Attack success rate" uses the notes' conditional definition (above) so numbers compare with the paper; the unconditional one is saved too.
- "Adds no more than 5 points of false alarms" is measured against the current gate, as the brief says, and is also reported against plain `ce`.
- "Equal attack success" in (ii) means within 0.5 points, or both at or under 1%.
- Seeds: seed s means the random split, donors and initialisation of the existing seed-s job.

## Amendment 1 (Nov 3, 2026, 10:30 IST, before any trained idea has run)

The user will run the sprint on a Kaggle GPU instead of the laptop. Scope widens; **pass marks, attack suite, metrics and design choices are unchanged.**

- Every idea runs on **all 7 runs**, on the random split seeds 0, 1, 2 and on the time split (seed 0), through `sprint/kaggle_job.py`, one notebook run per setting.
- So the "screen on 3 runs, at best MAYBE" cap no longer applies: the verdicts use the 7-run rule directly. The finalist step is folded in, since every idea gets all settings.
- 1A worst-of-K runs regardless of 1A's screen (it is now affordable). It runs last in each job, so a time-out cuts it first.
- 2C (time-balanced B) runs on all 7 runs of the time split; its pass mark is still (iii) on CICIoT.
- On GPU, training uses `light.fit`'s mixed precision, as the existing F and B did.
- Seen before this amendment: E0 baseline rows for 5GAD, MCFP and MCFP strict, seed 0 (the current gate, plain, XGBoost; no new idea).
