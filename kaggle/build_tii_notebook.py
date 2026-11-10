"""Build kaggle/tii_ssrc23_flows.ipynb, which converts TII-SSRC-23 on Kaggle.

TII-SSRC-23 is 27.5 GB of pcaps on Kaggle (daniaherzalla/tii-ssrc-23), too
much to download here, so the conversion runs in a Kaggle notebook with the
dataset attached. The notebook carries its own copy of prep/pcap_to_flows.py;
rerun this script after changing that file.

  python kaggle/build_tii_notebook.py
"""

from pathlib import Path

import nbformat

HERE = Path(__file__).parent
CONVERTER = (HERE.parent / "prep" / "pcap_to_flows.py").read_text(encoding="utf-8")

INTRO = """# TII-SSRC-23 to LitCVit flow images

1. Attach the dataset **daniaherzalla/tii-ssrc-23** (Add Input, top right). No GPU and no internet needed.
2. Run all cells (or Save Version, then Save & Run All, so it runs in the background; it can take a few hours).
3. Output: `/kaggle/working/TII-SSRC-23/` with one `.npy` (flow images) and `.csv` (flow metadata) per pcap,
   named after the pcap's path, plus `TII-SSRC-23.zip` with all of them and `layout.txt` listing the input pcaps.
   Download the zip and put its contents in `C:\\data\\CNS_proj\\flows\\TII-SSRC-23`.

Each capture keeps at most `MAX_FLOWS` flows (its first ones), as for the CICIoT2022 attack captures.
The cell below is a copy of `prep/pcap_to_flows.py`, written by `kaggle/build_tii_notebook.py`."""

LAYOUT = """import os
from pathlib import Path

INPUT = Path("/kaggle/input")
OUT = Path("/kaggle/working/TII-SSRC-23")
MAX_FLOWS = 20_000
WORKERS = os.cpu_count()

import pcap_to_flows as ptf

captures = sorted(p for p in INPUT.rglob("*") if p.is_file() and ptf.is_capture(p))
lines = [f"{p.stat().st_size / 2**20:10,.1f} MB  {p.relative_to(INPUT)}" for p in captures]
Path("/kaggle/working/layout.txt").write_text("\\n".join(lines) + "\\n")
print(f"{len(captures)} captures, {sum(p.stat().st_size for p in captures) / 2**30:.1f} GB")
print("\\n".join(lines[:60]))"""

CONVERT = """import time
from concurrent.futures import ProcessPoolExecutor, as_completed

OUT.mkdir(parents=True, exist_ok=True)

def convert(path):
    stem = OUT / ptf.output_name(path, INPUT)
    if ptf.sibling(stem, ".npy").exists() and ptf.sibling(stem, ".csv").exists():
        return f"{stem.name}: already done"
    start = time.time()
    try:
        packets, skipped, flows, tls, quic = ptf.process_capture(path, stem, max_flows=MAX_FLOWS)
    except Exception as err:
        return f"{stem.name}: FAILED ({err})"
    return (f"{stem.name}: {packets:,} packets ({skipped:,} not usable), {flows:,} flows, "
            f"{tls:,} TLS, {quic:,} QUIC, {time.time() - start:.0f}s")

# largest first, so the long ones start early
with ProcessPoolExecutor(WORKERS) as pool:
    jobs = [pool.submit(convert, p) for p in sorted(captures, key=lambda p: -p.stat().st_size)]
    for job in as_completed(jobs):
        print(job.result(), flush=True)"""

PACK = """import shutil

shutil.copy("/kaggle/working/layout.txt", OUT / "layout.txt")
shutil.make_archive("/kaggle/working/TII-SSRC-23", "zip", OUT)
print(f"{(Path('/kaggle/working/TII-SSRC-23.zip').stat().st_size) / 2**20:,.0f} MB zip")"""

nb = nbformat.v4.new_notebook()
nb.cells = [
    nbformat.v4.new_markdown_cell(INTRO),
    nbformat.v4.new_code_cell("%%writefile pcap_to_flows.py\n" + CONVERTER),
    nbformat.v4.new_code_cell(LAYOUT),
    nbformat.v4.new_code_cell(CONVERT),
    nbformat.v4.new_code_cell(PACK),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbformat.write(nb, HERE / "tii_ssrc23_flows.ipynb")
print(f"wrote {HERE / 'tii_ssrc23_flows.ipynb'}")
