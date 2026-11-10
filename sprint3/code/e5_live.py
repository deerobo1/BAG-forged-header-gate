"""E5 part 2 (R2, R3): do forged M values survive a live TCP session with an unmodified Linux receiver?

Runs inside WSL as root (PYTHONPATH=/opt/s3py for scapy). Two network namespaces joined by a veth pair: s3b runs an
unmodified kernel TCP echo server and a capture; s3a runs a scapy client whose kernel RSTs are dropped by iptables.
For each value set from e5_forge.py (a forged flow's client-side M values per packet: TTL, TOS, IP ID, DF, window,
TCP options), the client opens a connection with the SYN's values, ACKs the SYN-ACK and sends data with the next
packets' values, and expects the data echoed back. The capture at the server is then checked for the forged values.

  wsl -u root -e sh -c "PYTHONPATH=/opt/s3py python3 sprint3/code/e5_live.py <live_sets.json> <out.json>"
"""

import json
import subprocess
import sys
import time

A_NS, B_NS, A_IP, B_IP, PORT = "s3a", "s3b", "10.77.0.1", "10.77.0.2", 9000
SERVER = r"""
import socket, threading
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("0.0.0.0", %d)); s.listen(64)
def serve(c):
    c.settimeout(5)
    try:
        d = c.recv(4096)
        if d: c.sendall(d)
    except Exception: pass
    c.close()
while True:
    c, _ = s.accept(); threading.Thread(target=serve, args=(c,), daemon=True).start()
""" % PORT


def sh(cmd, check=True):
    return subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)


def setup():
    for ns in (A_NS, B_NS):
        sh(f"ip netns del {ns}", check=False)
        sh(f"ip netns add {ns}")
    sh(f"ip link add va type veth peer name vb")
    sh(f"ip link set va netns {A_NS}; ip link set vb netns {B_NS}")
    sh(f"ip -n {A_NS} addr add {A_IP}/24 dev va; ip -n {A_NS} link set va up; ip -n {A_NS} link set lo up")
    sh(f"ip -n {B_NS} addr add {B_IP}/24 dev vb; ip -n {B_NS} link set vb up; ip -n {B_NS} link set lo up")
    sh(f"ip netns exec {A_NS} iptables -A OUTPUT -p tcp --tcp-flags RST RST -j DROP")
    for ns, dev in ((A_NS, "va"), (B_NS, "vb")):  # real checksums on the wire, no offload
        sh(f"ip netns exec {ns} ethtool -K {dev} tx off rx off", check=False)


def teardown(procs):
    for p in procs:
        p.terminate()
    for ns in (A_NS, B_NS):
        sh(f"ip netns del {ns}", check=False)


def options_of(hexstr):
    """Raw TCP option bytes -> scapy option list (MSS, WScale, SAckOK, Timestamp, NOP, EOL; others raw)."""
    if not hexstr:
        return []
    b, out, i = bytes.fromhex(hexstr), [], 0
    while i < len(b):
        kind = b[i]
        if kind == 0:
            out.append(("EOL", None))
            i += 1
            continue
        if kind == 1:
            out.append(("NOP", None))
            i += 1
            continue
        if i + 1 >= len(b) or b[i + 1] < 2:
            break
        n, val = b[i + 1], b[i + 2:i + b[i + 1]]
        if kind == 2 and n == 4:
            out.append(("MSS", int.from_bytes(val, "big")))
        elif kind == 3 and n == 3:
            out.append(("WScale", val[0]))
        elif kind == 4 and n == 2:
            out.append(("SAckOK", b""))
        elif kind == 8 and n == 10:
            out.append(("Timestamp", (int.from_bytes(val[:4], "big"), int.from_bytes(val[4:], "big"))))
        else:
            out.append((kind, val))
        i += n
    return out


CLIENT = r"""
import json, sys, random
from scapy.all import IP, TCP, Raw, sr1, send, AsyncSniffer, conf
conf.verb = 0
sys.path.insert(0, sys.argv[3])
from e5_live import options_of
sets, out = json.load(open(sys.argv[1])), []
for i, fields in enumerate(sets):
    mine = [f for f in fields if f["side"] == 0] or fields
    f0, f1, f2 = mine[0], mine[min(1, len(mine) - 1)], mine[min(2, len(mine) - 1)]
    sport, isn = int(sys.argv[4]) + i, random.randrange(2 ** 32)
    ip = lambda f: IP(src="%s", dst="%s", ttl=f["ttl"], tos=f["tos"], id=f["ipid"], flags="DF" if f["df"] else 0)
    syn = ip(f0) / TCP(sport=sport, dport=%d, flags="S", seq=isn, window=f0["window"], options=options_of(f0["options"]))
    r = {"set": i, "ttl": f0["ttl"], "tos": f0["tos"], "ipid": f0["ipid"], "df": f0["df"], "window": f0["window"],
         "options": f0["options"], "handshake": False, "echo": False}
    sa = sr1(syn, timeout=2)
    if sa is not None and sa.haslayer(TCP) and sa[TCP].flags & 0x12 == 0x12:
        r["handshake"] = True
        ts = [o for o in options_of(f1["options"]) if o[0] == "Timestamp"]
        srv_ts = [o[1][0] for o in sa[TCP].options if o[0] == "Timestamp"]
        def opts(f):
            o = options_of(f["options"])
            if srv_ts:  # keep the echo honest: TSecr is the server's last TSval, TSval the donor's
                o = [("Timestamp", (v[0], srv_ts[0])) if k == "Timestamp" else (k, v) for k, v in o]
            return [x for x in o if x[0] not in ("MSS", "WScale", "SAckOK")]
        ack = sa[TCP].seq + 1
        send(ip(f1) / TCP(sport=sport, dport=%d, flags="A", seq=isn + 1, ack=ack, window=f1["window"], options=opts(f1)))
        data = b"s3-live-%%d" %% i
        sniff = AsyncSniffer(filter="tcp and src port %d and dst port %%d" %% sport, timeout=3)
        sniff.start()
        send(ip(f2) / TCP(sport=sport, dport=%d, flags="PA", seq=isn + 1, ack=ack, window=f2["window"], options=opts(f2)) / Raw(data))
        got = sniff.join() or sniff.results
        r["echo"] = any(p.haslayer(Raw) and bytes(p[Raw].load) == data for p in (sniff.results or []))
        send(IP(src="%s", dst="%s") / TCP(sport=sport, dport=%d, flags="FA", seq=isn + 1 + len(data), ack=ack + len(data)))
    out.append(r)
json.dump(out, open(sys.argv[2], "w"))
""" % (A_IP, B_IP, PORT, PORT, PORT, PORT, A_IP, B_IP, PORT)


def on_wire(capture, results, base):
    """Compare each client's SYN as captured at the server with the values it was meant to carry."""
    from scapy.all import IP, TCP, rdpcap
    syns = {p[TCP].sport: p for p in rdpcap(capture) if p.haslayer(TCP) and p[TCP].flags & 0x02 and p[IP].src == A_IP}
    for r in results:
        p = syns.get(base + r["set"])
        r["syn_captured"] = p is not None
        if p is not None:
            r["wire_match"] = (p[IP].ttl == r["ttl"] and p[IP].tos == r["tos"] and p[IP].id == r["ipid"]
                               and bool(p[IP].flags & 2) == r["df"] and p[TCP].window == r["window"]
                               and bytes(p[TCP])[20:p[TCP].dataofs * 4].hex() == (r["options"] or ""))
    return results


def main(sets_path, out_path):
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    live = json.load(open(sets_path))
    setup()
    srv = subprocess.Popen(["ip", "netns", "exec", B_NS, "python3", "-c", SERVER])
    cap = f"/tmp/s3_live.pcap"
    dump = subprocess.Popen(["ip", "netns", "exec", B_NS, "tcpdump", "-i", "vb", "-w", cap, "-U", "tcp"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)
    results = {}
    try:
        for run, sets in live.items():
            src, dst = f"/tmp/s3_sets_{run}.json", f"/tmp/s3_res_{run}.json"
            json.dump(sets, open(src, "w"))
            open("/tmp/s3_client.py", "w").write(CLIENT)
            sh(f"ip netns exec {A_NS} env PYTHONPATH={os.environ.get('PYTHONPATH', '')} python3 /tmp/s3_client.py {src} {dst} {here} {20000 + 1000 * list(live).index(run)}")
            results[run] = json.load(open(dst))
        time.sleep(1.5)
    finally:
        dump.terminate()
        time.sleep(0.5)
        teardown([srv])
    for run in results:
        results[run] = on_wire(cap, [dict(r) for r in results[run]], 20000 + 1000 * list(live).index(run))
    json.dump(results, open(out_path, "w"), indent=1)
    for run, rs in results.items():
        print(run, len(rs), "handshake", sum(r["handshake"] for r in rs), "echo", sum(r["echo"] for r in rs),
              "wire_match", sum(r.get("wire_match", False) for r in rs))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
