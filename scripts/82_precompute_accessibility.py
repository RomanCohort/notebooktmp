"""Precompute ViennaRNA per-position unpaired probability for every mRNA window.

Why
---
The relation model compresses a 50-nt mRNA window into a handful of soft
prompts.  A seed match alone is not sufficient for binding -- the site also has
to be accessible (not buried in a stem).  This cache gives the resampler a
per-position openness signal so it can learn to prefer accessible positions.

What is stored
--------------
For each unique window, 50 float16 values in [0, 1]:
    unpaired[i] = 1 - (sum_j P(i pairs j) + sum_j P(j pairs i))
computed from the base-pair probability matrix of the isolated-window
ensemble.  All 50 positions are kept -- the hypothesis is positional ("is the
seed region at 2-8 open?"), so pooling would destroy exactly the signal we want.

Known limitation
----------------
Isolated-window folding: no miRNA-mRNA cofold, no transcript context.  This is
a proxy feature.  Always pair it with a `shuf_acc` ablation when evaluating.

Usage
-----
    python scripts/82_precompute_accessibility.py --dry-run
    python scripts/82_precompute_accessibility.py --jobs 12
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import numpy as np

WIN = 50
DTYPE = np.float16


def read_unique_windows(data_path: Path) -> list[str]:
    """All distinct mRNA windows, first-seen order, across every split.

    Dedup is by exact sequence, which is correct even when one sequence maps to
    several genomic coordinates: structure is a function of sequence only.
    Deduping across train and test also makes the 57 known cross-split window
    pairs get one consistent value instead of two.
    """
    seen: set[str] = set()
    out: list[str] = []
    with data_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            w = (rec.get("mrna_window") or "").upper()
            if len(w) != WIN or w in seen:
                continue
            seen.add(w)
            out.append(w)
    return out


def compute_unpaired(seq: str) -> np.ndarray:
    """Per-position unpaired probability from the equilibrium ensemble."""
    import RNA

    fc = RNA.fold_compound(seq)
    fc.pf()
    bpp = np.asarray(fc.bpp(), dtype=np.float64)
    # bpp is (L+1)x(L+1) with a zero row/col 0 (1-based SWIG indexing).
    b = bpp[1:, 1:]
    # Row sums and column sums differ (the matrix is not symmetric by much for
    # long stems), so both sides must be added before subtracting from 1.
    paired = b.sum(axis=1) + b.sum(axis=0)
    return np.clip(1.0 - paired, 0.0, 1.0)


def _worker(chunk: list[str]) -> np.ndarray:
    return np.stack([compute_unpaired(s) for s in chunk]).astype(DTYPE)


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path, bufsize: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/processed/mirbench.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data/processed/_mirbench_acc.f16"))
    ap.add_argument("--idx", default=str(ROOT / "data/processed/_mirbench_acc.pt"))
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=512,
                    help="windows per work unit (tuned for IPC overhead)")
    ap.add_argument("--save-every", type=int, default=20000)
    ap.add_argument("--min-free-gb", type=float, default=2.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    idx = Path(args.idx)
    state_path = out.with_suffix(out.suffix + ".state.json")

    t0 = time.time()
    seqs = read_unique_windows(Path(args.data))
    n = len(seqs)
    total_bytes = n * WIN * np.dtype(DTYPE).itemsize
    print(f"[data] {n:,} unique windows of {WIN} nt  "
          f"({time.time()-t0:.0f}s scan)", flush=True)
    print(f"[data] memmap {total_bytes / 2**20:.1f} MiB", flush=True)

    if args.dry_run:
        sample = seqs[:5]
        t1 = time.time()
        for s in sample:
            compute_unpaired(s)
        per = (time.time() - t1) / len(sample)
        print(f"[dry] {per*1000:.2f} ms/window single-threaded", flush=True)
        print(f"[dry] serial {n*per/60:.1f} min, "
              f"{args.jobs} jobs ~{n*per/60/args.jobs*1.4:.1f} min", flush=True)
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(out.parent).free / 2**30
    if not out.exists() and free_gb < total_bytes / 2**30 + args.min_free_gb:
        raise RuntimeError(
            f"Not enough disk: need {total_bytes/2**30:.2f} GiB + "
            f"{args.min_free_gb:.1f} GiB reserve, have {free_gb:.1f} GiB")

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        start = int(state["next_index"])
        print(f"[resume] window {start:,}/{n:,}", flush=True)
    else:
        start = 0
        atomic_json(state_path, dict(next_index=0, total=n, dim=WIN,
                                     dtype="float16", rna_version=_rna_version()))

    mm = np.memmap(out, dtype=DTYPE, mode="r+" if out.exists() else "w+",
                   shape=(n, WIN))

    chunks = [(i, seqs[i:i + args.chunk]) for i in range(start, n, args.chunk)]
    done = start
    t0 = time.time()
    with mp.Pool(args.jobs) as pool:
        for (base, _), arr in zip(chunks, pool.imap(_worker, [c for _, c in chunks])):
            mm[base:base + len(arr)] = arr
            done = base + len(arr)
            if done % args.save_every < args.chunk or done == n:
                mm.flush()
                atomic_json(state_path, dict(next_index=done, total=n, dim=WIN,
                                             dtype="float16",
                                             rna_version=_rna_version()))
                rate = (done - start) / max(time.time() - t0, 1e-9)
                print(f"  [acc] {done:,}/{n:,}  {rate:.0f}/s  "
                      f"eta {(n-done)/max(rate,1e-9)/60:.1f} min", flush=True)

    mm.flush()
    del mm
    if done != n:
        raise RuntimeError(f"Expected {n:,} windows, wrote {done:,}")

    blob = "\n".join(seqs).encode("utf-8")
    import torch
    torch.save(dict(
        windows=blob, n=n, dim=WIN, dtype="float16",
        rna_version=_rna_version(), mmap=str(out),
        sha256_f16=sha256_file(out),
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
    ), idx)
    print(f"[save] {out}  {out.stat().st_size/2**20:.1f} MiB", flush=True)
    print(f"[save] {idx}  windows={n:,}  "
          f"sha256={sha256_file(out)[:16]}…", flush=True)
    state_path.unlink(missing_ok=True)
    return 0


def _rna_version() -> str:
    try:
        import RNA
        return str(RNA.__version__)
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
