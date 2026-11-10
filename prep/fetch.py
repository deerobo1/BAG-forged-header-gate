"""Download files over HTTP with several ranged connections at once, resumably.

Some dataset servers are slow per connection: the CTU server that hosts
IoT-23 and MCFP sent 130 KB/s per connection from here on a good run, and
8 connections got about 1 MB/s (on 2026-10-27 it later capped the total at
about 300 KB/s however many connections were used). Each file is cut into
--chunk-mb byte ranges, and --workers connections fetch ranges in list order.
Finished ranges are listed in <file>.part.done, so stopping and rerunning
the same command resumes; finished files are skipped.

The URL list has one URL per line, optionally followed by a size limit in
MB: only that much of the start of the file is fetched (rounded down to whole
chunks). Removing the limit later extends the file instead of starting over.
Blank lines and # comments are ignored. Each file is saved under --out at
its URL path relative to --base.

Example:
  python prep/fetch.py prep/lists/iot23.txt --out C:/data/CNS_proj/raw/IoT-23 \\
      --base https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/
"""

import argparse
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def read_list(path):
    """[(url, size limit in MB or None)]"""
    lines = (line.split("#", 1)[0].split() for line in open(path, encoding="utf-8"))
    return [(parts[0], int(parts[1]) if len(parts) > 1 else None) for parts in lines if parts]


def remote_size(url):
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
        return int(r.headers["Content-Length"])


def fetch_range(url, start, end, retries=6):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status != 206:
                    raise IOError(f"HTTP {r.status} instead of 206: server ignores ranges")
                data = r.read()
            if len(data) != end - start + 1:
                raise IOError(f"got {len(data)} of {end - start + 1} bytes")
            return data
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(5 * 2 ** attempt)


class Target:
    """One file being downloaded into <path>.part, with its finished ranges in <path>.part.done."""

    def __init__(self, url, path, size, chunk):
        self.url, self.path, self.size, self.chunk = url, path, size, chunk
        self.part = path.with_name(path.name + ".part")
        self.done_file = path.with_name(path.name + ".part.done")
        self.lock = threading.Lock()
        done = set()
        if self.part.exists() and self.done_file.exists() and self.part.stat().st_size == size:
            done = {int(x) for x in self.done_file.read_text().split()}
        elif not self.part.exists() and path.exists() and path.stat().st_size % chunk == 0:
            # a finished shorter download (an earlier size limit): extend it
            done = set(range(path.stat().st_size // chunk))
            path.replace(self.part)
            with open(self.part, "r+b") as f:
                f.truncate(size)
            self.done_file.write_text("".join(f"{i}\n" for i in sorted(done)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.part, "wb") as f:
                f.truncate(size)
            self.done_file.write_text("")
        self.todo = [i for i in range(-(-size // chunk)) if i not in done]
        self.left = len(self.todo)
        if not self.todo:  # an earlier run stopped just before renaming
            self.part.replace(path)
            self.done_file.unlink()

    def ranges(self):
        for i in self.todo:
            yield i, i * self.chunk, min(self.size, (i + 1) * self.chunk) - 1

    def store(self, i, start, data):
        """Write one range; returns True when that was the file's last one."""
        with self.lock:
            with open(self.part, "r+b") as f:
                f.seek(start)
                f.write(data)
            with open(self.done_file, "a") as f:
                f.write(f"{i}\n")
            self.left -= 1
            if self.left == 0:
                self.part.replace(self.path)
                self.done_file.unlink()
                return True
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("urls", type=Path, help="text file with one URL per line")
    parser.add_argument("--out", type=Path, required=True, help="folder to save into")
    parser.add_argument("--base", required=True, help="URL prefix that maps to --out")
    parser.add_argument("--workers", type=int, default=8, help="parallel connections (default 8)")
    parser.add_argument("--chunk-mb", type=int, default=8, help="range size in MB (default 8)")
    args = parser.parse_args(argv)

    targets, total, chunk = [], 0, args.chunk_mb << 20
    for url, limit_mb in read_list(args.urls):
        if not url.startswith(args.base):
            sys.exit(f"{url} is not under --base")
        path = args.out / urllib.parse.unquote(url[len(args.base):])
        size = remote_size(url)
        if limit_mb is not None:
            size = min(size, max(chunk, (limit_mb << 20) // chunk * chunk))
        if path.exists() and path.stat().st_size >= size:  # done, or longer than a limit asks for
            continue
        if size == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            continue
        target = Target(url, path, size, chunk)
        if target.left:
            targets.append(target)
            total += size
    remaining = sum(min(t.size, t.left * t.chunk) for t in targets)
    print(f"{len(targets)} files to fetch, {remaining / 2**20:,.0f} MB left of {total / 2**20:,.0f} MB", flush=True)
    if not targets:
        return

    got, failed, lock, stop = [0], [], threading.Lock(), threading.Event()

    def work(target, i, start, end):
        try:
            data = fetch_range(target.url, start, end)
        except Exception as err:
            failed.append(f"{target.path.name} bytes {start}-{end}: {err}")
            return
        with lock:
            got[0] += len(data)
        if target.store(i, start, data):
            print(f"done: {target.path.relative_to(args.out)} ({target.size / 2**20:,.1f} MB)", flush=True)

    def report():
        start, last = time.time(), 0
        while not stop.wait(60):
            rate = (got[0] - last) / 60
            last = got[0]
            eta = (remaining - got[0]) / rate / 60 if rate else float("inf")
            print(f"[{(time.time() - start) / 60:.0f} min] {got[0] / 2**20:,.0f} of {remaining / 2**20:,.0f} MB, "
                  f"{rate / 2**10:,.0f} KB/s, about {eta:.0f} min left", flush=True)

    threading.Thread(target=report, daemon=True).start()
    with ThreadPoolExecutor(args.workers) as pool:
        for t in targets:
            for i, start, end in t.ranges():
                pool.submit(work, t, i, start, end)
    stop.set()
    for line in failed:
        print(f"FAILED {line}", file=sys.stderr)
    if failed:
        sys.exit(f"{len(failed)} ranges failed; rerun the same command to retry them")
    print("all files complete", flush=True)


if __name__ == "__main__":
    sys.exit(main())
