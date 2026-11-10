# Sprint inventory (Nov 3, 2026)

What exists before the sprint, read from the repo and `C:\data\CNS_proj`. Nothing here is a new result.

## Code

| What | Where |
|---|---|
| Flow loading, splits, class weights, balanced batches | `kaggle/litcvit/litcvit.py`: `load`, `split` (random: 80/20 per class; time: first 80% of each class-and-capture by `ts_first`), `class_weights`, `balanced_indices` |
| Attacker-controllable fields ("shortcut fields") | `litcvit.shortcut_fields`. IPv4: TOS, IP ID, flags and fragment offset, TTL, checksum, options. IPv6: traffic class, flow label, hop limit. TCP: sequence and ack numbers, window, checksum, urgent pointer, options. UDP: checksum. ICMP: checksum and rest of header. Addresses and ports are already zero in every image. **Lengths, protocol, TCP data offset and flags, and payload are not in the set**, so the content model reads them. |
| Models, losses, training, test conditions | `kaggle/litcvit/light.py`: `FlowNet` (MobileNetV2 = timm `mobilenetv2_050`, 40x40 image upsampled to 80x80), losses `ce`, `mask`, `both`, `cfnce`, `oneside`, `gated`, `views`; `fit` (AdamW, lr 1e-3, batch 256, 30 epochs, cosine with warmup, class-balanced batches, class-weighted CE); `test_sets`; `score` |
| The gate | `light.gate` (logit form), `gate_report.gate_decision` (decision form: base answer unless base says Benign and content model does not) |
| Header attacks | `mimic.py`: copy of a benign donor's TOS, IP ID, DF, TTL, window, options (`mimic_stack`); `mimic_all` adds new ISNs. `attacks.py`: `arbitrary` (random valid values), `ordinary_options` |
| Packet and payload attacks | `attacks.py`: `add_packets(copies=2)` (repeats the first bare ACK, else packet 0), `resegment(size)` (first data segment in `size`-byte pieces), `payload_kept` |
| Attack evaluation on saved weights | `limits.py` (suite (a)-(f) plus `arbitrary_any`, best-of-8 on 2,000 sampled malware flows), `near_empty.py` (the k-byte rule; IoT-23 and USTC not yet scored), `gate_report.py` (pass marks for new seeds and time split) |
| XGBoost | `kaggle/litcvit_evasion/evasion.py: xgb_logits` (300 trees, depth 8, lr 0.1, hist, class weights, on the flattened 1,600 bytes). Not installed locally until this sprint; installed xgboost 3.2.0. |

## Data

- **Flow images**, all 7 runs: `C:\data\CNS_proj\kaggle\litcvit-flows\<dataset>.npy` (uint8, N x 40 x 40 = first 5 packets x 320 bytes, header in the first 80 bytes of each row) with `<dataset>.csv` (class, capture, `ts_first`, proto, packets, tls, quic, strict for MCFP).
- Run sizes (flows / train flows on the random split, seed 0): 5GAD 9,307 / 7,445; MCFP 18,493 / 14,795; MCFP strict 13,871 / 11,096; CICIoT 23,339 / 18,671; TII 40,819 / 32,655; IoT-23 49,035 / 39,228; USTC 105,464 / 84,371.
- **Per-packet information available without re-parsing:** for the 5 packets in the image, IP total length, protocol, TCP data offset and flags, and payload bytes (all outside the shortcut fields). Packet direction can be inferred from sequence numbers (`attacks.packet_sides`), but those are attacker-controllable fields.
- **Per-packet timing: not available.** The flow CSVs keep only `ts_first`. Raw pcaps exist for 5 of 7 datasets (`C:\data\CNS_proj\raw`: 5GAD, CICIoT, IoT-23, MCFP, USTC; about 15 GB). **TII has no raw pcaps locally.** A timing view would need `prep/pcap_to_flows.py` extended and rerun over ~15 GB and would cover at most 6 of 7 runs. **Not done in this sprint** (cost and missing TII); the length view in 1C uses the 5 packets in the image instead.

## Saved models and logits (reusable on CPU, no training)

| Models (MobileNetV2) | Split / seed | Folder |
|---|---|---|
| `ce`, `mask`, `both`, `cfnce`, `cfnce1` | random, seed 0, all 7 runs | `litcvit_output_ablate/preds/<run>_mobilenetv2_<loss>.pt` |
| `oneside`, `gated`, `views` | random, seed 0 | `litcvit_output_gate/preds` |
| `ce`, `mask` (the gate's F and B) | **time split**, seed 0, all 7 runs | `litcvit_output_gate/preds/<run>_time_mobilenetv2[_mask].pt` |
| `ce`, `mask`, `both`, `oneside` | random, seeds 1 and 2 (USTC seed 2 `both`/`oneside`, IoT-23 seed 2 `oneside` missing; no `cfnce`) | `litcvit_output_seeds/preds/<run>_mobilenetv2_<loss>_seed<s>.pt` |
| 9 lightweight backbones, `ce`, random seed 0 | benchmark, 58-ish of 77 rows before Colab ran out | `litcvit_output_bench` |

So the current gate (F = `ce`, B = `mask`) can be re-scored on any new attack for **3 seeds on the random split and seed 0 on the time split, on all 7 runs, with no training.**

## Compute measured today

- Laptop: 12 CPU threads, **no GPU** (torch 2.8.0+cpu). Colab GPU quota is used up (user, Nov 3). The brief's "about 6 GPU hours" is therefore not available now.
- MobileNetV2, one training epoch on CPU, 12 threads: 83 s on 5GAD (7,445 flows), 159 s on MCFP (14,795) = about 11 ms per flow per epoch. Two jobs at 6 threads each: 225 s per epoch each, so jobs run one at a time.
- Full 30-epoch model: 5GAD ~42 min, MCFP strict ~62 min, MCFP ~80 min, CICIoT ~1.7 h, TII ~3.0 h, IoT-23 ~3.6 h, USTC ~7.8 h. **All 7 runs = ~17 CPU-hours per model.**

## Missing, and what it rules out

- Per-packet timing (see above): 1C runs as a length-and-flags view of the first 5 packets, not a timing view.
- GPU: 3 seeds on 7 runs for new trained methods cannot run on the laptop. Screening uses the three smallest runs, which are also the three where the packet attacks are worst (5GAD, MCFP strict, MCFP). Finalists get a ready-to-run Colab job (`sprint/colab/`) for when GPU quota returns.
- EdgeNeXt (3B) and Vision Mamba (3A) need GPU at any useful scale. Tier 3 is out of budget unless GPU returns.
