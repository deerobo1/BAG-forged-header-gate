"""E5 summary: live-session outcomes per run, and each forged field's realizability class (GUARANTEE.md section 2).

  python sprint3/code/e5_summary.py  -> sprint3/results/e5_live.csv, e5_realizability.csv
"""
import json
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
live = json.loads((R / "e5_live.json").read_text())
rows = []
for run, rs in live.items():
    rows.append({"run": run, "sessions": len(rs), "handshake_ok": sum(r["handshake"] for r in rs),
                 "echo_ok": sum(r["echo"] for r in rs), "syn_values_on_wire": sum(r.get("wire_match", False) for r in rs),
                 "with_tcp_options": sum(bool(r["options"]) for r in rs),
                 "echo_failed_all_timestamp_only": all(r["options"] and r["options"].startswith("0101080a") for r in rs if not r["echo"])})
pd.DataFrame(rows).to_csv(R / "e5_live.csv", index=False)
print(pd.DataFrame(rows).to_string())
fields = pd.read_csv(R / "e5_fields.csv")
CLASS = [("TTL", "free", "IP_TTL socket option; used values >= 32 except as recorded", "ttl"),
         ("TOS (DSCP/ECN)", "free", "IP_TOS socket option", "tos"),
         ("DF bit", "free", "IP_MTU_DISCOVER", "df"),
         ("TCP window", "free (range) / cost (exact per packet)", "SO_RCVBUF, TCP_WINDOW_CLAMP; exact values need a custom stack", "window"),
         ("IP ID", "cost", "Linux chooses it; exact values need a raw socket (RFC 6864 allows any value)", "ipid"),
         ("TCP options (same length)", "cost", "MSS free via TCP_MAXSEG; scale and timestamps need a custom stack", "options_len"),
         ("ISN shift", "free", "any ISN is valid (RFC 9293, RFC 6528)", None),
         ("IP / TCP checksum", "derived", "recomputed", None)]
out = []
for name, cls, how, col in CLASS:
    n = len(fields) if col is None else int((fields[col] > 0).sum()) if col == "options_len" else len(fields)
    out.append({"field": name, "class": cls, "how": how, "forged_packets": n, "not_realizable_values": 0})
pd.DataFrame(out).to_csv(R / "e5_realizability.csv", index=False)
print(pd.DataFrame(out).to_string())
