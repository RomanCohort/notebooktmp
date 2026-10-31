"""Encode the unique sequences in RNAQA v3 into one resumable memmap cache.

This is the Drive-free Colab path. The 68 GiB cache is generated locally on the
Colab runtime instead of being uploaded as a notebook payload.

Usage:
    python scripts/80_encode_v3_hidden.py --data data/processed/rnaqa_v3_l234.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DIM = 768
MAX_NT = 1024


def read_unique(data_path: Path) -> tuple[list[str], list[str]]:
    seen: set[str] = set()
    uids: list[str] = []
    seqs: list[str] = []
    with data_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            uid = record.get("uid")
            seq = (record.get("sequence") or "").upper().replace("T", "U")
            if uid and uid not in seen and 4 <= len(seq) <= MAX_NT:
                seen.add(uid)
                uids.append(uid)
                seqs.append(seq)
    return uids, seqs


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/processed/rnaqa_v3_l234.jsonl"))
    ap.add_argument("--encoder", default=str(ROOT / "models/ernierna"))
    ap.add_argument("--out", default=str(ROOT / "data/processed/_rnaqa_v3_hidden.f16"))
    ap.add_argument("--idx", default=str(ROOT / "data/processed/_rnaqa_v3_index.pt"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--save-every", type=int, default=256)
    ap.add_argument("--min-free-gb", type=float, default=8.0)
    args = ap.parse_args()

    out = Path(args.out)
    idx = Path(args.idx)
    state_path = out.with_suffix(out.suffix + ".state.json")
    uids, seqs = read_unique(Path(args.data))
    lens = [len(s) for s in seqs]
    total = sum(lens)
    total_bytes = total * DIM * 2
    print(f"[data] {len(seqs):,} sequences / {total:,} positions", flush=True)
    print(f"[data] memmap {total_bytes / 2**30:.2f} GiB", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(out.parent).free / 2**30
    if not out.exists() and free_gb < total_bytes / 2**30 + args.min_free_gb:
        raise RuntimeError(
            f"Not enough disk: need {total_bytes/2**30:.1f} GiB + "
            f"{args.min_free_gb:.1f} GiB reserve, have {free_gb:.1f} GiB"
        )

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        start = int(state["next_index"])
        pos = int(state["positions"])
        kept_u = list(state["kept_uids"])
        kept_l = [int(x) for x in state["kept_lens"]]
        print(f"[resume] sequence {start:,}/{len(seqs):,} position {pos:,}", flush=True)
    else:
        start, pos, kept_u, kept_l = 0, 0, [], []
        atomic_json(state_path, dict(
            next_index=0, positions=0, kept_uids=[], kept_lens=[],
            total_positions=total, dim=DIM,
        ))

    mm = np.memmap(out, dtype=np.float16, mode="r+" if out.exists() else "w+",
                   shape=(total, DIM))
    from rnaqwen.models.encoder import RnaEncoder

    enc = RnaEncoder(args.encoder, dtype=torch.float32).to(DEV).eval()
    t0 = time.time()
    with torch.no_grad():
        for base in range(start, len(seqs), args.batch):
            chunk = seqs[base:base + args.batch]
            out_chunk = enc(chunk)
            hidden = out_chunk.hidden.float()
            mask = out_chunk.mask.squeeze(-1) > 0
            if mask.shape[1] != hidden.shape[1]:
                mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
            for j, seq in enumerate(chunk):
                hh = hidden[j][mask[j]]
                if hh.shape[0] == len(seq) + 2:
                    hh = hh[1:-1]
                if hh.shape[0] != len(seq):
                    raise RuntimeError(
                        f"Encoder length mismatch for {uids[base+j]}: "
                        f"expected {len(seq)}, got {hh.shape[0]}"
                    )
                mm[pos:pos + len(seq)] = hh.half().cpu().numpy()
                pos += len(seq)
                kept_u.append(uids[base + j])
                kept_l.append(len(seq))
            del out_chunk, hidden, mask

            done = min(base + len(chunk), len(seqs))
            elapsed = max(time.time() - t0, 1e-9)
            rate = (done - start) / elapsed
            print(
                f"  [enc] {done:,}/{len(seqs):,}  {rate:.1f}/s  "
                f"positions {pos:,}/{total:,}  "
                f"eta {(len(seqs)-done)/max(rate,1e-9)/60:.1f} min",
                flush=True,
            )
            if done - start >= args.save_every or done == len(seqs):
                mm.flush()
                atomic_json(state_path, dict(
                    next_index=done, positions=pos,
                    kept_uids=kept_u, kept_lens=kept_l,
                    total_positions=total, dim=DIM,
                ))

    mm.flush()
    if pos != total:
        raise RuntimeError(f"Expected {total:,} positions, wrote {pos:,}")

    lens_t = torch.tensor(kept_l, dtype=torch.long)
    offsets = torch.cumsum(
        torch.cat([torch.zeros(1, dtype=torch.long), lens_t]), 0
    )[:-1]
    torch.save(dict(
        lens=lens_t, offsets=offsets, uids=kept_u,
        n_pos=pos, dim=DIM, mmap=str(out),
    ), idx)
    print(f"[save] {out}  {out.stat().st_size/2**30:.2f} GiB", flush=True)
    print(f"[save] {idx}  sequences={len(kept_u):,}", flush=True)
    state_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
