# Public datasets for the encrypted traffic project

Checked 2026-10-24; section 4 updated 2026-10-26 from the LitCVit PDF and our first download, and 2026-10-27 with the first header-vs-payload baseline. USTC-TFC2016 is downloaded and processed (in `C:\data\CNS_proj`); nothing else is downloaded yet. Everything else below comes from dataset pages, repository READMEs and paper metadata (Crossref, Semantic Scholar).

LitCVit is now the lead paper (see plan.txt), so section 4 matters most. Sections 1–3 are for the mmTraffic fallback.

---

## 1. mmTraffic's six datasets: original sources

| Dataset | Official source | Format, size | License / citation | Notes |
|---|---|---|---|---|
| ISCXVPN2016 | https://www.unb.ca/cic/datasets/vpn.html (direct link to cicresearch.ca, no form) | pcap + CSV, ~28 GB | "Publicly available for researchers"; cite Draper-Gil et al., ICISSP 2016 | 7 categories, VPN and non-VPN |
| ISCX-Tor-2016 | https://www.unb.ca/cic/datasets/tor.html (http://cicresearch.ca/CICDataset/ISCX-Tor-NonTor-2017/) | pcap + CSV, ~22 GB | Same as above; cite Lashkari et al., ICISSP 2017 | Workstation pcap (non-Tor) + gateway pcap (Tor) |
| USTC-TFC-2016 | https://github.com/davidyslu/USTC-TFC2016 (old name yungshenglu redirects; original: echowei/DeepTraffic) | pcap / .7z; 0.29 GB zip, 4.0 GB extracted (Windows `tar -xf` opens the .7z files) | MPL-2.0; cite Wang et al., ICOIN 2017 | 10 benign + 10 malware. Benign part is largely unencrypted (SMB, MySQL, FTP). Malware part comes from CTU captures |
| CSTNet-TLS1.3 | ET-BERT repo, `datasets/CSTNET-TLS 1.3/readme.md` → Google Drive (https://github.com/linwhitehat/ET-BERT) | Anonymised flows | Cite ET-BERT (Lin et al., WWW 2022) | **Handshake and Client Hello removed in the released copy** (per the Sweet Danger paper), so "TLS header present" claims will never hold on this dataset |
| CrossPlatform Android / iOS | **None.** https://recon.meddle.mobi/cross-market.html says "no longer available"; Tsinghua backup link (https://cloud.tsinghua.edu.cn/f/b00ce0c960114b59b948/) returns "Link does not exist" | — | Cite FlowPrint (van Ede et al., NDSS 2020) and Ren et al. | Only processed copies survive (section 2). Android class count differs by copy: 209 / 212 / 215 |

## 2. Already-processed copies

| Copy | Contents | Use |
|---|---|---|
| mmTraffic BaiduNetdisk (code `v5u6`, link in https://github.com/lgzhangzlg/Multimodal-Reasoning-with-LLM-for-Encrypted-Traffic-Interpretation-A-Benchmark) | All six datasets; README says each sample carries the 5-field Claude annotation (class, traits, evidence, description, notes) | Reproduction; only source of the annotations. Needs a Baidu account; slow without membership |
| NetBench (https://github.com/WM-JayLab/NetBench → Google Drive folder `NetBench_Dataset`, with `Flow-Level/` and `Packet-Level/`) | ISCX-VPN, ISCX-Tor, USTC-TFC, CrossPlatform Android (66,346 flows) and iOS (34,912 flows), DoHBrw-2020, CIC-IoT-2023 | **Best non-Baidu source for CrossPlatform.** File format inside not yet checked |
| Lens (https://huggingface.co/datasets/Charles59/lens-network-traffic) | Same sources as NetBench, pre-split train/val/test, as tshark-style text with hex payloads; CC-BY-NC-4.0 | Quick checks; must convert text back to bytes for the 10 x 160 format |
| Debunk / Pcap-Encoder (https://huggingface.co/datasets/rigcor7/Debunk_Traffic_Representation, 41 GB, MIT) | ISCX-VPN, USTC-TFC, CSTNet TLS-120 with per-packet **and per-flow (leakage-free)** splits | Settles Gap 5 / plan step 2. Code: https://github.com/SmartData-Polito/Debunk_Traffic_Representation |
| YaTC (https://github.com/NSSL-SJTU/YaTC, Google Drive) | ISCXVPN, ISCX-Tor, USTC-TFC, CICIoT2022 as "MFR" byte matrices (image-like) | Ready vision baseline for Gap 6 |
| NetMamba (https://github.com/wangtz19/NetMamba) | README links a Google Drive of "processed datasets"; the visible folder only showed paper subfolders | Not confirmed useful |

## 3. Extra datasets for the mmTraffic gaps

**Cross-dataset tests (plan step 7)**

- **VNAT**, MIT Lincoln Lab: https://www.ll.mit.edu/r-d/datasets/vpnnonvpn-network-application-traffic-dataset-vnat
  - 36 GB of pcap, direct download.
  - 10 apps, VPN and non-VPN: Vimeo, Netflix, YouTube, Zoiper, Skype, SSH, RDP, SFTP, RSYNC, SCP.
  - Overlaps ISCXVPN2016 on Skype, YouTube/Vimeo, SFTP and VoIP.
  - Its paper (arXiv 2205.05628) covers uncertainty quantification, which is related work for Gap 3.
- **CipherSpectrum**, UNSW: https://cgi.cse.unsw.edu.au/~cspectrum/
  - 120k TLS 1.3 sessions, 40 domains x 3 cipher suites, captured Jan–Mar 2024.
  - pcap, CC-BY-NC-4.0, Google Form with no approval step.
  - Partner for CSTNet-TLS1.3; check which domains the two share.
  - Cite the SoK by Wickramasinghe et al., IEEE S&P 2025.

**Not independent: don't use as held-out test sets**

- CIC-Darknet2020 (https://www.unb.ca/cic/datasets/darknet2020.html) is ISCXVPN2016 + ISCX-Tor-2016 merged.
- CTU-13 / MCFP overlap with USTC-TFC's malware half.

**Wrong format or hard to get**

- CESNET-TLS22, QUIC22, TLS-Year22 (https://cesnet.github.io/cesnet-datazoo/): only sizes, directions and timings of the first 30 packets, no payload bytes. Can't build the 10 x 160 matrix.
- NUDT MobileTraffic (https://github.com/Abby-ZS/NUDT_MobileTraffic): 350 Android apps, 293 GB, by emailing the authors only.
- MIRAGE-2019 (https://traffic.comics.unina.it/mirage/): JSON per-packet features; payload bytes not confirmed; CC-BY-NC-ND.

---

## 4. LitCVit datasets

**Paper:** Mehr Un Nisa, Adnan Noor Mian, Mubashir Husain Rehmani, "LitCVit: A Lightweight Self-Supervised Contrastive Vision Transformer for Encrypted Malicious Traffic Detection", IEEE TIFS vol. 21, pp. 3960–3972, 2026. DOI 10.1109/TIFS.2026.3683528, IEEE Xplore document 11480891.

**Confirmed from the PDF** (Section VI-A, Table III, Figs. 4 and 5):
- Six datasets. The text says "four" once, but every table and figure uses six.
- **Pretraining (unlabeled):** USTC-TFC2016 and CICIoT2022 only. **Fine-tuning and testing:** all six. The other four are "unseen" only in pretraining; each is still fine-tuned on its own labels.
- **No flow counts, split ratio or split method** anywhere, although the abstract criticises per-packet splits and claims "leakage-free evaluation".
- **Input:** first 5 packets x 320 bytes as a 40 x 40 image, Ethernet header removed, IPs and ports anonymised. The paper says this follows YaTC.
- Only USTC-TFC2016 is filtered to encrypted flows ("we retain only complete encrypted traffic flows").

| Ref | Dataset | Classes LitCVit uses (from Figs. 4 and 5) |
|---|---|---|
| 19 | USTC-TFC2016 | 11: Benign (merged) + Cridex, Geodo, Htbot, Miuref, Neris, Nsis-ay, Shifu, Tinba, Virut, Zeus |
| 41 | CICIoT2022 | 4: Benign, Flood, Hydra, Nmap |
| 42 | 5GAD-2022 | 9: Normal + AMFDel, AMFIns, AMFUDM, CrashNRF, GetAllNFs, GetUData, rndAMFIns, rndDump. The text says "10 attack categories" in one place and "eight attacks" in another |
| 43 | MCFP | 5: Benign (600 CTU-Normal pcaps), Botnet, Cobalt, Dridex, TrojanDownloader. BitcoinMiner is named in the text but isn't in the figures; the chosen captures aren't listed |
| 44 | IoT-23 | 6: benign, C&C, HeartBeat, Muhstik, Okiru, PortScan. The chosen scenarios aren't listed |
| 45 | TII-SSRC-23 | 5: Benign, Bruteforce, DoS, Recon, Mirai (its 26 attack subtypes merged) |

Sources (USTC-TFC2016 is in section 1):

| Ref | Dataset | What it is | Format, size | Download | License / citation |
|---|---|---|---|---|---|
| 41 | **CICIoT2022** | IoT profiling: Audio / Camera / Home-Automation devices; Power, Idle, Interactions, Scenarios, Active, Attacks (Flood, RTSP brute force) | pcap | https://www.unb.ca/cic/datasets/iotdataset-2022.html → http://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2022/. **As of 2026-10-26 this URL shows CIC's download form** (name, email, organisation) instead of the file list. After the form, the root lists 1-Power, 2-Idle, 3-Interactions, 4-Scenarios, 6-Attacks (.tar.gz each), 5-Active/, CSV/, Device List.xlsx, Readme.txt, Z-Wave and Zigbee experiments. We take 6-Attacks (Flood, Hydra, Nmap), 3-Interactions (benign), Readme.txt and Device List.xlsx | Cite Dadkhah et al., PST 2022 |
| 42 | **5GAD-2022** (Idaho National Lab) | Simulated 5G core (free5GC): normal (1 UE, 2 UE: YouTube, web, FTP, Samba, Teams) + 10 attacks (recon, NF reconfiguration, DoS) | pcapng, 4 interfaces | https://github.com/IdahoLabResearch/5GAD (clone with git-lfs; repo archived) | MIT; cite Coldwell et al., IEEE Globecom Workshops 2022. Their prep scripts need ≥96 GB RAM |
| 43 | **Stratosphere MCFP** | Hundreds of long-running malware captures (CTU-Malware-Capture-Botnet-N) | pcap + zipped malware binary (password `infected`) | https://www.stratosphereips.org/datasets-malware → https://mcfp.felk.cvut.cz/publicDatasets/ | Not stated (CTU-13 is CC-BY). **The zips contain live malware: only use the pcaps, or work in a VM** |
| 44 | **IoT-23** | 20 malicious (Mirai, Okiru, Torii, Hajime, …) + 3 benign IoT captures | pcap + Zeek `conn.log.labeled`; full archive 20 GB (about 60 GB of pcaps once extracted), small 8.7 GB (no pcaps) | **Use the individual scenarios** (see below): https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/. Each folder has one `.pcap` and `bro/conn.log.labeled` | Cite Garcia et al. 2020, Zenodo 10.5281/zenodo.4743746 |
| 45 | **TII-SSRC-23** | 8 types / 32 subtypes. Benign: audio, background, text, video. Malicious: brute force, DoS, information gathering, Mirai | pcap + CSV, 27.5 GB | https://www.kaggle.com/datasets/daniaherzalla/tii-ssrc-23 (Kaggle account) | Not stated in paper; cite Herzalla et al., IEEE Access 2023 (arXiv 2310.10661) |

**IoT-23 subset to download.** The paper doesn't list its scenarios, so we take every scenario whose pcap is 1 GB or less. That's 13 folders and about 3 GB of pcaps, plus their `conn.log.labeled` files (1.5–1.7 GB each for 7-1 and 36-1). Put each one in `C:\data\CNS_proj\raw\IoT-23\<scenario>\`.
- Malware: CTU-IoT-Malware-Capture-1-1, 3-1 (Muhstik), 7-1, 8-1, 9-1, 20-1, 21-1, 34-1, 36-1 (Okiru), 42-1.
- Benign: CTU-Honeypot-Capture-4-1, 5-1, 7-1.
- From each folder, download the `.pcap` and `bro/conn.log.labeled`. Skip any `.zip`: those hold the live malware binaries.
- We left out the big scans and floods: 17-1, 33-1, 35-1, 39-1, 43-1, 44-1, 48-1, 49-1, 52-1 and 60-1, which are 1.2–21 GB each.
- This subset should cover all six classes (benign, C&C, HeartBeat, Muhstik, Okiru, PortScan). HeartBeat is the least certain, and we'll confirm it from the label files.

**IoT-23 download and labelling** (2026-10-27):
- **Download.** The CTU server gave about 130 KB/s per connection at first, then capped us at about 300 KB/s in total, whatever the number of connections. `prep/fetch.py` downloads in parallel and resumes after interruptions. The file list and order are in `prep/lists/iot23.txt`.
- **Big scans are partial.** For 7-1, 9-1 and 36-1 (6.5 GB together, about 7 hours at that rate) we take the first 15% of each pcap and the first 30% of its label file. The label file is written roughly in time order, so 30% leaves margin to cover the pcap's 15%. Rerunning the list without the limits extends them to full files.
- **Labels.** `prep/iot23_flows.py` labels each flow from `conn.log.labeled`. To keep the multi-million-connection scans small, it keeps at most 100,000 Zeek connections per label per scenario, chosen by a hash of the Zeek uid. Rarer labels are kept whole.
- **Flows vs Zeek connections.** A flow matches a Zeek connection with the same endpoints that it starts inside. We write fewer flows than Zeek has connections: Zeek logs each unanswered SYN retry from the same source port as a new connection, and our flow (same endpoints, 120 s timeout) merges them. For example, 8-1's 8,222 C&C connections cover exactly 2,056 endpoint pairs, giving 2,056 flows. Every flow in 8-1, 20-1 and 42-1 matched a Zeek connection; we checked this against an unfiltered conversion.
- **TLS is almost absent:** 0 TLS flows in 3-1, 8-1, 20-1 and 21-1, 2 in 42-1 and 1 in honeypot 7-1.

Labels per scenario (Zeek connections). For the three partial scenarios, the counts cover the first 30% of the label file:

| Scenario | Benign | Labels on malicious connections |
|---|---|---|
| 1-1 (Hide and Seek) | 469,275 | PartOfAHorizontalPortScan 539,465; C&C 8 |
| 3-1 (Muhstik) | 4,536 | PartOfAHorizontalPortScan 145,597; Attack 5,962; C&C 8 |
| 7-1 (partial) | 3,781 | Okiru 3,478,012; DDoS 39,584; C&C-HeartBeat 2,666 |
| 8-1 | 2,181 | C&C 8,222 (all unanswered SYNs) |
| 9-1 (partial) | 8,697 | PartOfAHorizontalPortScan 1,930,325 |
| 20-1, 21-1 (Torii) | 3,193; 3,272 | C&C-Torii 16; 14 |
| 34-1 | 1,923 | DDoS 14,394; C&C 6,706; PartOfAHorizontalPortScan 122 |
| 36-1 (partial) | 631 | Okiru 4,092,132; C&C-HeartBeat 4,708; Okiru-Attack 3 |
| 42-1 | 4,420 | FileDownload 3; C&C-FileDownload 3 |
| Honeypots 4-1, 5-1, 7-1 (Somfy-01) | 452; 1,374; 130 | none |

Notes on the table:
- **All six LitCVit classes are present.** HeartBeat comes only from 7-1 and 36-1.
- **Partial scenarios are covered.** For 7-1, 9-1 and 36-1, the downloaded part of the label file runs about 3 hours past the end of the downloaded part of the pcap, so no flow there lacks a label.
- **34-1's label file was re-released in 2023** with the column renamed from `detailed-label` to `det_label`; `iot23_flows.py` accepts both names. 1-1 also has a 2023 file, `CTU-IoT-Malware-1-1.conn.log.labeled`, which is just a 7,008-connection sample of the full log, so we don't use it.
- **TLS is almost absent:** 2 TLS flows across all ten malware scenarios (in 42-1), and 138 in the honeypots.

Flows per class after labelling: Benign 113,433, C&C 6,159, HeartBeat 2,876, Muhstik 56,553, Okiru 107,126, PortScan 85,286.

Header vs payload test (random forest, seed 0; results in `results/iot23_header_payload_seed0.csv`). Macro-F1:

| Input | Random split | Time split |
|---|---|---|
| full | 99.4 | 98.3 |
| header | 99.4 | 99.4 |
| payload | 28.1 | 25.5 |
| full_min | 65.7 | 62.6 |
| header_min | 65.5 | 62.5 |

What it shows:
- **The header alone does as well as the full image,** and both match LitCVit's 98.11 F1. The payload alone scores 28.
- **The accuracy comes from fields that identify the device.** Zeroing them costs 34 points: recall drops for C&C from 99% to 26%, for HeartBeat from 100% to 45%, and for PortScan from 100% to 39%. Okiru and Muhstik stay at 99–100%. The forest puts 87% of its importance on header bytes. Its single most-used byte is in the TCP options (23% of all importance), followed by TCP window, TTL and the IP don't-fragment flag. These fields identify the infected device's TCP/IP stack.
- **The malicious flows have nothing else to offer.** These shares carry no payload in their first 5 packets: HeartBeat 100%, Okiru 100%, PortScan 98%, Muhstik 89%, C&C 84%. With the device fields zeroed, 91% of sampled flows are byte-identical to a flow of another class.
- **About three quarters of the Benign class is probably mislabelled.** In 1-1, 85,059 of 92,508 benign flows (92%) are one repeated single-packet UDP message (payload starts `51ff8559…`), sent to many different ports. The same capture labels that exact message C&C when it goes to port 2407. Scenario 1-1 supplies 82% of our Benign class, so about 75% of it is very likely the bot's own peer-to-peer traffic. We haven't confirmed this by decoding the protocol.
- **Splitting by time barely matters here:** 98.3 against 99.4.

**Our class rule for LitCVit's six IoT-23 classes.** The paper names the classes but doesn't say how it built them, and "Muhstik" is a malware name, not an IoT-23 label. So we use:
- Benign: any benign flow.
- Muhstik: every malicious flow of 3-1, the only Muhstik capture.
- Otherwise by detailed label: HeartBeat for C&C-HeartBeat*, C&C for other C&C*, PortScan for PartOfAHorizontalPortScan*, Okiru for Okiru*.
- Everything else (DDoS, Attack, FileDownload) is left out.

The rule is in `baselines/header_payload.py`. One consequence: Muhstik is mostly port scans from one infected device, while PortScan comes from other devices, so the two classes may be separable only by device. That's the same shortcut as on USTC-TFC2016, here built into the class list.

**CICIoT2022** (downloaded 2026-10-27 through CIC's form; in `raw\CICIoT2022`):
- **Files:** `6-Attacks.tar.gz` is 420 MB (5.5 GB unpacked) and `3-Interactions.tar.gz` is 73 MB (85 MB unpacked). Also `Readme.txt` and `Device List.xlsx`, which gives device names and MAC addresses but no IP addresses. Both archives contain macOS `._*` and `.DS_Store` files, which we leave out when extracting.
- **Classes,** by folder (the rule is in `baselines/header_payload.py`):
  - Benign: `3-Interactions`. The paper doesn't say which benign captures it used; this is our choice.
  - Flood: `6-Attacks/1-Flood`, LOIC HTTP, TCP and UDP floods on 13 devices.
  - Hydra and Nmap: `6-Attacks/2-RTSP Brute Force/Hydra` and `/Nmap`.
- **Conversion:** we keep the first 20,000 flows of each attack capture (`pcap_to_flows.py --max-flows 20000`), because the three Philips Hue TCP floods are 660–760 MB each. Interaction captures are kept whole.

| Class | Captures | Flows | TLS |
|---|---|---|---|
| Benign (Interactions) | 864 | 3,213 | 37.9% |
| Flood | 79 | 259,495 (20,000 per capture at most) | 0% |
| Hydra | 6 | 126 | 0% |
| Nmap | 11 | 12,005 | 0% |

- **CIC captured the attacks on the attacking machine** (per its readme). In every attack capture, at least 95% of flows are between one target device and `192.168.137.5`, which is the same attacker in all of them.
- **All TLS is benign:** 38% of benign flows use TLS and no attack flow does.
- **Hydra is tiny:** Hydra sends its guesses over a few long connections, so its 6 captures give only 126 flows. Its recall is measured on 24 test flows.

Header vs payload test (random forest, seed 0; results in `results/ciciot2022_header_payload_seed0.csv`). Macro-F1:

| Input | Random split | Time split |
|---|---|---|
| full | 99.6 | 57.2 |
| header | 99.3 | 59.7 |
| payload | 77.6 | 46.4 |
| full_min | 98.1 | 54.1 |
| header_min | 95.5 | 57.2 |

What it shows:
- **On a random split, a forest beats LitCVit.** It gets 99.6 against LitCVit's 99.11 F1, and 99.3 from header bytes alone. Its top header bytes are TTL, IP checksum, TCP window and TCP options, which identify the machine as on USTC-TFC2016.
- **The Nmap class is mostly a port scan.** In each Nmap capture, about the first 80% of flows are Nmap's TCP port scan: 2-packet SYN/RST flows to many ports. Only the last part is the RTSP brute force on port 554, as 5-packet flows. The time split tests on that last part, and Nmap recall falls from 99% to 22%. Of those flows, 62% are predicted as Hydra (the same attack done with another tool) and 12% as Benign. So on a random split, "Nmap brute force" is recognised mainly by its port scan.

**5GAD-2022** (downloaded 2026-10-27; a git clone in `raw\5GAD-2022`):
- **Files:** the attacks are small (135 MB, `git lfs pull --include "Attacks/**"`). The normal traffic is 35.5 GB: Normal-1UE is 6 files covering about 20 hours, and Normal-2UE is 15 files. We take the first 128 MB of each 1UE file and 48 MB of each 2UE file (`prep/lists/5gad.txt`), so every part of the capture period is represented. GitHub gives about 1.5 MB/s here with 8 connections.
- **Every capture is recorded on four interfaces at once** ("allcap"):
  - `lo`: the 5G core's REST APIs on `127.0.0.x:8000`, plus MongoDB.
  - `upfgtp`: user traffic leaving the UPF, as raw IP (link type 12, which `pcap_to_flows.py` now reads), with the UE at `10.60.0.x`.
  - `eno1`: the same user traffic after NAT, as `192.168.70.x`.
  - `enp5s0`: the gNB link, with GTP-U on UDP 2152 and NGAP over SCTP.
  - So each user connection appears twice as a flow (upfgtp and eno1). `prep/fivegad_flows.py` tags Normal flows with the interface, guessed from the addresses.
- **The attacks are plaintext `curl` calls to the core's REST API.** Each attack repeats one HTTP/1.1 request about 400 times, from `127.0.0.1` to an NF on port 8000 (e.g. `GET /nnrf-disc/v1/nf-instances?...`). An attack's type is written in its request line. No attack flow uses TLS.
- **Labels.** Each attack folder has an `Attacks_<name>.pcapng` that the authors cut down to the attack packets. We label a flow of the folder's allcap file with the attack when its endpoints appear in that file on a TCP connection to port 8000. Everything else in the attack captures (MongoDB, NGAP, the host's own web traffic) is left out. Two problems with the released labels:
  - `Attacks_FakeAMFDelete.pcapng` is byte-identical to its allcap file, so it isn't filtered at all. The port rule keeps its 800 REST connections. Half of them are the `PUT` that re-inserts the fake AMF, which is byte for byte the same request as FakeAMFInsert. So AMFDel and AMFIns overlap by construction, which fits LitCVit reporting AMFIns as its weakest 5GAD class (95%).
  - randomAMFInsert likewise holds a `PUT` and a `DELETE` per round.
- **Skipped:** the two PFCP attacks (automatedDropWithTimer, automatedRedirectWithTimer) are one UDP flow each. LitCVit's eight attack classes leave them out too.
- **The authors' own `Data_prep.py` works per packet** (payload hex only), and its Normal-2UE loop re-reads the last Normal-1UE file instead of the 2UE files (it uses `file` where it means `pcap`).

| Class | Flows | Notes |
|---|---|---|
| AMFDel, rndAMFIns | 800, 799 | a `PUT` and a `DELETE` per round |
| AMFIns, AMFUDM, CrashNRF, GetAllNFs, GetUData, rndDump | 397–398 each | |
| Normal | 5,322 | 2,577 eno1 + 1,975 upfgtp (mostly the same connections twice), 338 lo, 185 enp5s0, 247 other; 60% TLS |

Header vs payload test (random forest, seed 0; results in `results/5gad2022_header_payload_seed0.csv`). Macro-F1:

| Input | Random split | Time split |
|---|---|---|
| full | 100.0 | 97.5 |
| header | 100.0 | 92.7 |
| payload | 93.6 | 92.4 |
| full_min | 93.5 | 92.4 |
| header_min | 75.2 | 75.3 |

What it shows:
- **Header bytes alone give 100%, and they do it by reading the clock.** The top header bytes are 45 and 49, the second byte of the TCP timestamp option's TSval and TSecr. That byte steps about once a minute. Each attack was recorded once, a few minutes apart, so each has its own set of values: AMFUDM 15–17, CrashNRF 17–19, AMFDel 19–24, AMFIns 27–29, GetAllNFs 29–31, rndAMFIns 33–38, rndDump 47–49, GetUData 73–75 (client side). Zeroing the timestamps with the other machine fields (full_min) costs 6.5 points.
- **Without the clock, the payload does the work, because it's plaintext.** The request line names the attack. What's left is the built-in overlap: AMFDel recall is 57%, because its `PUT` half is the same request as AMFIns.
- **Nothing here is encrypted.** All attack flows are HTTP/1.1 on loopback. LitCVit's 97.25 F1 on 5GAD says nothing about encrypted traffic.
- **Every attack is a single capture,** so only a time split is possible. Under it, full still gets 97.5.

**MCFP** (downloaded 2026-10-27/28; `prep/lists/mcfp.txt`, into `raw\MCFP`; flows in `flows\MCFP`):
- **The paper names families, not captures:** BitcoinMiner, Botnet, Cobalt, Dridex and Trojan Downloader, plus "600 benign PCAP files from the CTU-Normal series". Its figures show five classes: Benign, Botnet, Cobalt, Dridex and TrojanDownloader.
- **Our choice of captures.** Stratosphere's malware list has one capture per named family in the 2018 batch: 341-1 Trojan Downloader, 345-1 Cobalt (the only Cobalt capture), 346-1 Dridex and 347-1 BitCoinMiner. "Botnet" isn't a family. The only capture the list labels with a generic botnet is 157-1, "Linux Botnet (Unknown)", so we use it. 341-1, 345-1 and 157-1 are taken whole; 346-1 and 347-1 as their first 64 MB.
- **Benign:** the CTU-Normal captures that have a full pcap: 7, 12 and 20–32. Of the others, 13 is empty and 4/5/6/10 have DNS-only or no pcaps. We take the first 32 MB of each. Fifteen captures can't make "600 PCAP files"; the paper doesn't say how it got that number.
- **The 2018 captures aren't one machine's traffic.** They were recorded on a shared lab segment. In the first 64 MB of 341-1 and 345-1 to 347-1, about 90% of flows are other machines' IPv6 multicast (MLD reports from the same link-local addresses in every capture), IGMP and SSDP. Each one-packet report becomes its own flow after the 120 s timeout, which gives 215,000–332,000 flows per prefix. In the first 64 MB:
  - 341-1: no traffic at all from its infected host (`192.168.1.134`).
  - 345-1 Cobalt: 485 flows, almost all TLS on 443.
  - 346-1 Dridex: 5,605 flows.
  - 347-1 BitcoinMiner: 1,355 flows (ICMP, stratum on port 3333).
  - So 341-1 and 345-1 were downloaded whole (2026-10-28): 903,335 and 682,652 flows. 341-1's infected host appears later in the capture (1,022 flows). 345-1 still gives 485 Cobalt flows: all of them were in the first 64 MB.
- **Labels:** a malware flow gets the family only if it involves the infected host named in the capture's README (`baselines/header_payload.py`, `mcfp_class`). Labelling every flow of the capture would mostly label lab background traffic, which differs by recording date. The paper doesn't say what it did.
- **"Botnet" is too small:** 157-1 is complete (73 MB, 606k packets) but has only 26 flows, because its C&C runs over a few long TCP connections on port 23031.
- The `.zip` files in each folder are live malware and are never downloaded.

Flows per class (2026-10-28). "With host" is the current rule (`mcfp_class`): the infected host is one end of the flow. "Started by host" counts only flows whose first packet the infected host sent.

| Class | Capture | With host | Started by host | TLS among started |
|---|---|---|---|---|
| Benign | 15 CTU-Normal | 29,741 | n/a | 17.0% of all (69% of flows are DNS) |
| TrojanDownloader | 341-1 | 1,022 | 16 | 8 |
| Cobalt | 345-1 | 485 | 483 | 476 (98.6%) |
| Dridex | 346-1 | 5,605 | 3,388 | 2,320 (68.5%) |
| BitcoinMiner | 347-1 | 1,355 | 6 | 0 (stratum on port 3333) |
| Botnet | 157-1 | 26 | 24 | 1 |

- **Even the infected host's flows are mostly not the malware.** The "with host" rule still picks up two kinds of lab traffic:
  - **A port scan from another lab machine.** `192.168.1.149` scans the infected host in 341-1 and 347-1: 1,004 and 1,005 two-packet SYN/RST flows, about 1,000 ports each. So 98% of TrojanDownloader and 74% of BitcoinMiner are the same inbound scan. `.149` also sends multicast and broadcast traffic in all four 2018 captures.
  - **ICMP Redirects from the lab gateway** (`192.168.1.2`, type 5): 2,215 of Dridex's flows (40%) and 344 of BitcoinMiner's (25%).
- **Counting only flows the infected host starts leaves three usable classes.** Benign, Cobalt (483) and Dridex (3,388) remain. TrojanDownloader (16), BitcoinMiner (6) and Botnet (24) are too small, even with 341-1 and 157-1 taken whole.
- **Cobalt and Dridex are real encrypted malware traffic,** the first we've found in the six datasets. Cobalt's C2 is TLS on 443. Dridex uses TLS on 443, plus 580 flows each to three non-standard ports (4743, 4143, 3889).

Header vs payload test, "with host" rule (random forest, seed 0; results in `results/mcfp_header_payload_seed0.csv`):

| Input | Random split: accuracy / macro-F1 | Time split: accuracy / macro-F1 |
|---|---|---|
| full | 99.5 / 92.3 | 99.4 / 88.3 |
| header | 99.5 / 92.3 | 97.9 / 83.7 |
| payload | 82.5 / 71.3 | 88.3 / 78.2 |
| full_min | 91.4 / 76.2 | 98.3 / 86.2 |
| header_min | 91.3 / 76.0 | 98.2 / 85.3 |

What it shows:
- **Accuracy beats LitCVit's 97.67; macro-F1 is lower** mainly because Botnet has 5 test flows (recall 40%).
- **The forest tells TrojanDownloader from BitcoinMiner by the machine that answers the scan.** Its top header bytes are IP ID, IP checksum and TTL. With those fields zeroed (full_min), BitcoinMiner recall falls from 99.6 to 23.2 on the random split: its scan flows become identical to TrojanDownloader's.
- **These numbers mostly measure the lab, not the malware.** Under this rule, TrojanDownloader is 98% lab traffic (the scan), BitcoinMiner 99.6% (scan and redirects) and Dridex 40% (redirects). The audit should use the "started by host" rule.

**TII-SSRC-23** (converted on Kaggle 2026-10-28; outputs in `flows\TII-SSRC-23`, audit in `flows\TII-SSRC-23_audit.csv`):
- **Conversion:** `kaggle/tii_ssrc23_flows.ipynb` (built by `kaggle/build_tii_notebook.py`) carries the same `pcap_to_flows.py` as `prep/`, checked against the copy in its output. It keeps the first 20,000 flows of each pcap, as for CICIoT2022.
- **Files:** 32 pcaps (27.5 GB), listed in `layout.txt`. Six are benign: audio, background, text, and video over HTTP, RTP and UDP. The other 26 are attack subtypes in four folders.
- **Classes,** by folder (`baselines/header_payload.py`, `tii_class`):
  - Benign: all six benign pcaps.
  - Bruteforce: DNS, FTP, HTTP, SSH and Telnet.
  - DoS: 12 floods.
  - Recon: `information-gathering`.
  - Mirai: 8 captures.

| Class | Captures | Flows | TLS |
|---|---|---|---|
| Benign | 6 | 819 | 16.6% (136 flows, all audio and text) |
| Bruteforce | 5 | 28,072 | 2 flows |
| DoS | 12 | 200,008 (20,000 per capture at most) | 0 |
| Recon | 1 | 20,000 | 8 flows |
| Mirai | 8 | 61,627 | 31 flows |

- **Benign is tiny.** Its 2.3 GB of pcaps are a few long audio and video streams, so with our 120 s idle timeout they give only 819 flows. The test set has 164 of them. A tool that cuts long flows into fixed-length pieces, as CICFlowMeter does, would give far more. LitCVit doesn't say what it did.
- **One attacker, one victim, and the same two machines in benign traffic.** Every brute-force, DoS and recon flow runs between `192.168.1.70` and `192.168.1.90`, and the benign video also streams from `.70` to `.90`. Unlike IoT-23 and CICIoT2022, machine fields can't separate benign from attack here. Mirai runs on other hosts (`.177`, `.244`, `.59`).
- **The 20,000-flow cap keeps about 20 s of each TCP/UDP flood.** `icmp_dos` (1 GB) is only 6 flows, because ICMP has no ports, and `mac_dos` (0.1 MB) is 2.
- **Nothing on the attack side is encrypted.** TLS appears only in benign audio and text flows (65–69% of those). The random-looking payloads in Mirai ACK (99% of flows with payload) and UDP-plain (97%) are flood filler.

Header vs payload test (random forest, seed 0; results in `results/tii_ssrc23_header_payload_seed0.csv`). Macro-F1:

| Input | Random split | Time split |
|---|---|---|
| full | 99.3 | 99.4 |
| header | 99.3 | 99.2 |
| payload | 88.7 | 90.9 |
| full_min | 98.6 | 98.9 |
| header_min | 98.0 | 98.3 |

What it shows:
- **A forest beats LitCVit.** It gets 99.3 against LitCVit's 98.05 F1 (YaTC: 93.58 accuracy).
- **Packet sizes and flags are enough.** header_min keeps 5 bytes per packet (IP version/IHL, total length, protocol, TCP flags) and still gets 98.0. Zeroing the machine fields costs 1.3 points here, against 33.9 on IoT-23 and 8.8 on USTC-TFC2016 (header to header_min, random split). The classes differ in what the packets are: SYN, ACK or FIN floods, DNS queries, scans. Lengths and flags show that directly.
- **Most test flows are copies of training flows.** With the per-packet counters zeroed (IP ID, checksums, sequence numbers), 96% of header_min test inputs occur byte-for-byte in training. Flood flows differ only in those counters. On this dataset a random split comes close to testing on the training data, whatever the model.
- **Payload alone misses Mirai:** recall is 52% on the random split and 59% on the time split. Most Mirai flood flows carry no payload, or random filler.
- **The time split changes nothing** (99.4 against 99.3). Each attack capture is one tool running steadily, and after the cap a DoS capture covers only about 20 s.

**USTC-TFC2016 encryption audit** (2026-10-26, all 562,415 flows, `prep/audit_flows.py`). Columns:
- tls%: share of flows with a TLS record in their first 5 packets.
- syn%: share of flows whose first packet is a bare TCP SYN.
- 5pkt%: share of flows with all 5 packets.
- pay%: share of flows with at least 128 payload bytes.
- hient%: of those, the share whose payload is close to random bytes.

| Class | Flows | TLS% | SYN% | 5pkt% | pay% | hient% |
|---|---|---|---|---|---|---|
| Benign: BitTorrent | 7,517 | 99.8 | 0.0 | 0.0 | 67.8 | 100.0 |
| Benign: Facetime | 6,000 | 0.0 | 0.0 | 0.0 | 100.0 | 56.7 |
| Benign: FTP | 101,037 | 0.0 | 18.6 | 27.6 | 11.3 | 54.6 |
| Benign: Gmail | 8,629 | 59.2 | 7.4 | 16.3 | 74.7 | 79.3 |
| Benign: MySQL | 86,089 | 0.0 | 8.1 | 8.1 | 8.2 | 0.0 |
| Benign: Outlook | 7,524 | 99.3 | 0.1 | 0.0 | 99.1 | 99.8 |
| Benign: Skype | 6,321 | 96.3 | 0.0 | 0.0 | 96.3 | 13.0 |
| Benign: SMB (2 captures) | 38,937 | 0.0 | 0.6 | 80.8 | 86.1 | 92.6 |
| Benign: Weibo (4 captures) | 39,950 | 0.0 | 0.0 | 98.3 | 99.8 | 99.9 |
| Benign: WorldOfWarcraft | 7,883 | 0.0 | 98.9 | 99.6 | 0.7 | 0.0 |
| Cridex | 61,699 | 0.0 | 97.3 | 51.2 | 51.2 | 0.0 |
| Geodo | 50,464 | 0.0 | 80.7 | 25.8 | 25.9 | 0.0 |
| Htbot | 8,093 | 7.0 | 57.9 | 68.7 | 40.4 | 2.6 |
| Miuref | 14,706 | 0.0 | 91.3 | 41.6 | 41.7 | 0.0 |
| Neris | 39,365 | 0.1 | 83.9 | 51.4 | 25.5 | 0.0 |
| Nsis-ay | 7,371 | 0.0 | 5.0 | 30.1 | 86.5 | 57.1 |
| Shifu | 11,245 | 3.1 | 3.6 | 7.2 | 31.2 | 0.0 |
| Tinba | 10,448 | 0.0 | 0.0 | 0.2 | 87.7 | 0.0 |
| Virut | 37,383 | 0.0 | 87.7 | 86.5 | 11.7 | 0.0 |
| Zeus | 11,754 | 0.0 | 49.2 | 51.6 | 5.5 | 0.0 |

What it shows:
- **Malware payloads are mostly plaintext.** In their first 5 packets we found HTTP `POST`/`GET` command-and-control, DNS lookups of random-looking (DGA) domains (`*.xyz`, `zapto.org`), NetBIOS and SSDP. TLS appears in at most 7% of any malware class.
- **The paper's filter can't work as written.** A literal "complete encrypted flows only" filter would remove 8 of the 10 malware classes.
- **Benign TLS flows are fragments.** The benign TLS apps are 1–2-packet pieces of longer sessions on port 443, with no handshake.
- **The header may give the label away.** Six malware classes mostly start with a TCP SYN; benign flows almost never do. The header-only test in plan step 8 checks whether the model uses this.

**Header vs payload baseline** (2026-10-27, `baselines/header_payload.py`). Setup:
- Random forest, 100 trees, seed 0.
- 4,000 training and 1,000 test flows per class, 11 classes with Benign merged as in the paper.
- All flows, not the paper's unstated encrypted-only subset.
- Results in `results/ustc_header_payload_seed0.csv`.

Inputs:
- full: all 1,600 bytes.
- header / payload: only the 80 header bytes or only the 240 payload bytes of each packet.
- full_min / header_min: the same, but with every header byte zeroed except IP version, IP length, protocol and TCP flags. This removes the fields that identify the sending machine (TTL, IP ID, checksums, TCP sequence numbers, window and options) and keeps what anyone watching encrypted traffic can see.

Splits:
- random: a random 80/20 split of each class.
- time: within each capture, the earliest 80% of flows train and the latest 20% test. Most classes come from a single capture, so a per-capture split isn't possible.

| Input | Random split: acc / macro-F1 | Time split: acc / macro-F1 |
|---|---|---|
| full | 96.8 / 96.8 | 97.3 / 97.3 |
| header | 94.9 / 94.9 | 87.7 / 87.1 |
| payload | 70.1 / 72.5 | 71.8 / 73.7 |
| full_min | 88.0 / 88.1 | 88.8 / 88.9 |
| header_min | 85.9 / 86.1 | 87.2 / 87.6 |

What it shows:
- **A plain forest is already near LitCVit.** It reaches 96.8 macro-F1. The paper's Table VII gives its own RF 84.72 and its XGBoost 92.70; LitCVit gets 98.32. Our data isn't the paper's, so this isn't like-for-like, but the paper's tree baselines look weak.
- **The payload adds little.** Packet sizes, protocol and TCP flags alone (header_min) reach 86–87%. Adding all 1,200 payload bytes on top (full_min) gains only about 2 points.
- **The last 9 points come from fields that identify the machine.** The forest's top header bytes are TCP window, TCP options, IP length, TTL, IP ID and IP checksum. The checksum still depends on the original IP addresses, because the image zeroes the addresses without recomputing it.
- **The reason is flows with no payload.** In their first 5 packets, these share of flows carry no payload: 88% of Virut, 73% of Geodo and Neris, 58% of Miuref, and 49% of Cridex and Zeus. Once the machine-identifying fields are zeroed, 31% of sampled flows become byte-identical to a flow from another class. Cridex recall drops from 97% to 50%, and Miuref from 96% to 55%. For these flows, only the infected machine's TCP/IP stack tells the families apart.
- **A random split flatters header fields that change during a capture.** Under the time split, header-only falls from 94.9 to 87.7, mostly on Tinba (recall 99.7 to 31.3) and Geodo (93 to 72). The full image doesn't drop. The likely cause is counters such as IP ID that rise over a capture; we haven't checked this yet.
- **Limits:** this shows what the bytes allow a forest to do, not what LitCVit uses; that needs plan step 8 run on LitCVit itself. Only one seed so far. Field names assume a 20-byte IPv4 header.

**Gate 1: our images vs YaTC's** (2026-10-27, `prep/yatc_compare.py`, output in `results/gate1_yatc_compare.txt`). YaTC's Google Drive archive (`YaTC_datasets.zip`, 15.9 MB, in `raw\YaTC`) has 40 x 40 PNGs for USTC-TFC2016 (4,882 flows, 20 apps/families) and CICIoT2022 (10,950 flows, 10 device/attack classes), plus ISCXVPN2016 and ISCX-Tor. For each YaTC image we find the same flow among ours and compare all 1,600 bytes, with addresses and ports blanked on both sides.

| | YaTC images | Found among ours | Identical | Only trailing bytes differ | Packet count differs | Other |
|---|---|---|---|---|---|---|
| USTC-TFC2016 | 4,882 | 4,882 | 2,068 | 2,656 | 158 | 0 |
| CICIoT2022, Attacks + Interactions | 6,812 | 6,812 | 4,789 | 1,923 | 100 | 0 |

(CICIoT2022's Idle and Power classes, 4,138 images, are captures we didn't convert.)

What it shows:
- **The layout matches.** Every YaTC image of a dataset we converted is one of our flows: first 5 packets, Ethernet header removed, 80 header bytes then 240 payload bytes, zero-padded. The gate passes.
- **YaTC keeps what follows the IP packet in the frame**: the 4-byte Ethernet FCS on several USTC benign captures (checked: on `FTP.pcap` every frame ends with a valid CRC-32 of the frame, which covers the MAC and IP addresses), and Ethernet padding on short packets. We cut each packet at its IP total length. Some padding holds stale memory, such as pieces of earlier packets (`GET /a`, `EHLO r`, TCP options), on Neris, Virut and Nsis-ay. In YaTC's images these bytes are noise that differs between captures, and so between classes. The FCS is another field that still encodes the original addresses.
- **YaTC's flows have no idle timeout.** They come from SplitCap sessions, which join every packet with the same 5-tuple. One Cridex image joins two connections 53 hours apart. Our flows end after 120 s of silence. This affects 3% of USTC and 1.5% of CICIoT2022 images.
- **Addresses differ.** YaTC zeroes ports in both datasets. Its USTC images carry random IP addresses (the checksum isn't recomputed), but its CICIoT2022 images keep the real ones (e.g. `192.168.137.40`). LitCVit says it anonymises IPs; we zero them.
- **Decision for the replication run:** keep our cleaner layout as the default. If a LitCVit number won't reproduce, a YaTC-compatible option (keep trailers, no timeout) is the first thing to try.

**Baselines** (all retrained by the authors on the six datasets):
- Table III: AppScanner, Kitsune, 2D-CNN, PEAN, PERT, ET-BERT, YaTC, NetMamba, Pcap-Encoder. YaTC is the best baseline on all six.
- Table VII (lightweight models): RF and XGBoost on the flattened 1,600 bytes; 2D-CNN, GRU, BiLSTM on the flow image.
- Ref 9 (Pcap-Encoder) is The Sweet Danger of Sugar (SIGCOMM 2025), the per-packet-leakage paper; its per-flow splits are listed in section 2.

**What this means for the plan**
- **Gate 1 check:** done 2026-10-27; our layout matches YaTC's MFR images (see Gate 1 above).
- **The "encrypted" gap is real:** the datasets are mostly IoT, 5G-core and DoS/brute-force attack traffic, only USTC-TFC2016 is filtered to encrypted flows, and the IP/TCP headers stay in the image. The S&P 2025 SoK, which LitCVit itself cites, shows that "encrypted" classifiers are often trained on plaintext traffic. Measure the share of TLS/QUIC flows in each set.
- **No code:** the PDF has no code link either, and none was found on GitHub, arXiv or the authors' DBLP. The first author's only other listed traffic paper is a LightGBM WiFi IDS in Wireless Networks, 2024.
- **Checklist status:** done; see plan.txt.

---

## 5. Changes to make in plan.txt

1. **Risk table:** the CrossPlatform fallback is "NetBench Drive or Baidu copy", not "official site". *Done 2026-10-26.*
2. **Leakage gap (now LitCVit gap 1; also mmTraffic gap 5):** cite The Sweet Danger of Sugar (SIGCOMM 2025, arXiv 2507.16438). ET-BERT on TLS-120 falls from 96.8 to 15.4 F1 when TCP sequence numbers and timestamps are randomised.
3. **Claim checker:** flag flows that aren't actually encrypted (USTC-TFC benign; most LitCVit datasets), and expect no handshake bytes in CSTNet-TLS1.3.

## 6. Suggested download order

LitCVit (lead). Everything goes in `C:\data\CNS_proj\raw\<dataset>\`, and outputs go in `C:\data\CNS_proj\flows\<dataset>\`.
1. USTC-TFC2016: **done 2026-10-26**. Still to fetch: YaTC's MFR copy of USTC-TFC and CICIoT2022 (Google Drive), for the Gate 1 comparison.
2. CICIoT2022 (CIC form) and the IoT-23 scenario subset above.
3. 5GAD-2022 (git-lfs) and MCFP (pcaps only). TII-SSRC-23 is already on Kaggle: preprocess it in a Kaggle notebook instead of downloading 27.5 GB.

mmTraffic (only if we fall back):
1. Baidu copy: confirm the annotations are included.
2. ISCXVPN2016 + USTC-TFC from their official sites; rebuild the 10 x 160 format and compare with the Baidu copy.
3. Debunk per-flow splits.
4. NetBench Drive (CrossPlatform).
5. VNAT + CipherSpectrum (cross-dataset).

---

## Search log

| Lookup | Result |
|---|---|
| arXiv 2604.08140, mmTraffic GitHub README | Six datasets, Baidu-only download, annotations included, MIT vs Apache-2.0 conflict, README paths reference missing `DTLR_model/` |
| UNB CIC pages (VPN, Tor, Darknet2020, IDS2017, DoHBrw-2020, IoT2022, AndMal2017) | All direct-download from cicresearch.ca, no form |
| ET-BERT repo + GitHub API | CSTNet data is a Google Drive link in `datasets/CSTNET-TLS 1.3/readme.md` |
| recon.meddle.mobi, ETA-Resource, Tsinghua backup | CrossPlatform original and backup both dead |
| NetBench GitHub + Drive, Lens HF | Processed CrossPlatform survives |
| Sweet Danger paper (arXiv 2507.16438) + its HF dataset | Per-packet split leakage; per-flow splits released |
| VNAT, CipherSpectrum, CESNET DataZoo, NUDT, MIRAGE | See section 3 |
| Crossref works/10.1109/tifs.2026.3683528 | LitCVit reference list (46 refs); refs 41–45 = datasets |
| Semantic Scholar API (same DOI) | LitCVit abstract; no open PDF, references elided |
| OpenAlex (same DOI) | 504 timeout |
| IoT-23, MCFP, 5GAD, TII-SSRC-23 pages | See section 4 |
| Web search for LitCVit code / author pages | Nothing found |
| LitCVit PDF (IEEE Xplore copy, added 2026-10-26) | Confirms six datasets (refs 19, 41–45), classes used, pretraining on USTC-TFC + CICIoT2022 only; no counts, splits or code |
| GitHub API, repo 185182119 (2026-10-26) | USTC-TFC2016 repo renamed to davidyslu/USTC-TFC2016; 21 files, 0.39 GB, no git-lfs |
| cicresearch.ca CICIoT2022 URL (2026-10-26) | Now redirects to CIC's download form |
| IoT-23 README and IndividualScenarios listing (2026-10-26) | Per-scenario pcap sizes (2 MB to 21 GB, about 60 GB total); each folder has the pcap, `bro/conn.log.labeled` and a README with no label summary |
| Other malicious-traffic sources found along the way, not cited by LitCVit | CIC-IDS2017, CIC-AndMal2017, CIC-DoHBrw-2020 (malicious DoH tunnels), CTU-13, malware-traffic-analysis.net |
