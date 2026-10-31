"""Stream RNAQA v3 sequences through ERNIE + the trained FSQ into uint16 codes.

This avoids storing the 68 GiB ERNIE hidden-state cache on disk. The resulting
code cache is about 95 MiB for 47.5M nucleotide positions and is used by
50_routeC_e2e.py with --codes / --codes-idx.

The FSQ encoder is frozen in this path. The LLM adapter and code embedding table
remain trainable.
"""
from __future__ import annotations

import argparse
import hashlib
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/processed/rnaqa_v3_l234.jsonl"))
    ap.add_argument("--encoder", default=str(ROOT / "models/ernierna"))
    ap.add_argument("--fsq", default=str(ROOT / "checkpoints/fsq_stream/tokenizer.pt"))
    ap.add_argument("--out", default=str(ROOT / "data/processed/_rnaqa_v3_codes.u16"))
    ap.add_argument("--idx", default=str(ROOT / "data/processed/_rnaqa_v3_codes_index.pt"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--pool", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--restart", action="store_true",
                    help="discard an unfinished state and start from sequence 0")
    ap.add_argument("--sort-by-length", action="store_true",
                    help="sort sequences by length to reduce padding")
    ap.add_argument("--min-free-gb", type=float, default=2.0)
    args = ap.parse_args()

    data_path = Path(args.data)
    fsq_path = Path(args.fsq)
    out = Path(args.out)
    idx = Path(args.idx)
    state_path = out.with_suffix(out.suffix + ".state.json")
    data_sha = sha256_file(data_path)
    fsq_sha = sha256_file(fsq_path)
    print(f"[hash] dataset {data_sha}", flush=True)
    print(f"[hash] fsq     {fsq_sha}", flush=True)
    uids, seqs = read_unique(data_path)
    if args.limit:
        uids, seqs = uids[:args.limit], seqs[:args.limit]
    if args.sort_by_length:
        order = sorted(range(len(seqs)), key=lambda i: (len(seqs[i]), uids[i]))
        uids = [uids[i] for i in order]
        seqs = [seqs[i] for i in order]
    if args.pool < 1:
        raise ValueError("--pool must be >= 1")
    lens = [len(s) // args.pool for s in seqs]
    total = sum(lens)
    need_gb = total * 2 / 2**30
    print(f"[data] {len(seqs):,} sequences / {total:,} positions", flush=True)
    print(f"[data] pool={args.pool}  uint16 cache {need_gb:.3f} GiB", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    if args.restart:
        for p in (out, state_path, idx):
            p.unlink(missing_ok=True)
    free_gb = shutil.disk_usage(out.parent).free / 2**30
    if not out.exists() and free_gb < need_gb + args.min_free_gb:
        raise RuntimeError(
            f"Need {need_gb + args.min_free_gb:.1f} GiB, have {free_gb:.1f} GiB"
        )

    ck = torch.load(fsq_path, map_location="cpu", weights_only=False)
    latent = int(ck["latent"])
    levels = int(ck["levels"])
    vocab = levels ** latent
    if vocab > 65535:
        raise RuntimeError(f"VOCAB={vocab} does not fit uint16")
    fsq_w = ck["state"]["enc.weight"].float().to(DEV)
    fsq_b = ck["state"]["enc.bias"].float().to(DEV)
    print(f"[fsq] latent={latent} levels={levels} vocab={vocab}", flush=True)

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if bool(state.get("sort_by_length", False)) != bool(args.sort_by_length):
            raise RuntimeError(
                "state was created with a different --sort-by-length setting; "
                "rerun with --restart"
            )
        start = int(state["next_index"])
        pos = int(state["positions"])
        kept_u = list(uids[:start])
        kept_l = [int(x) for x in lens[:start]]
        if pos != sum(kept_l):
            raise RuntimeError(
                f"state position mismatch: {pos:,} != sum(lens[:{start}]) {sum(kept_l):,}"
            )
        print(f"[resume] sequence {start:,}/{len(seqs):,} position {pos:,}", flush=True)
    else:
        start, pos, kept_u, kept_l = 0, 0, [], []
        atomic_json(state_path, dict(
            next_index=0, positions=0,
            total_positions=total, vocab=vocab,
            dataset=str(data_path), dataset_sha256=data_sha,
            fsq=str(fsq_path), fsq_sha256=fsq_sha,
            sort_by_length=bool(args.sort_by_length),
        ))

    mm = np.memmap(out, dtype=np.uint16, mode="r+" if out.exists() else "w+",
                   shape=(total,))
    from rnaqwen.models.encoder import RnaEncoder

    enc = RnaEncoder(args.encoder, dtype=torch.float32).to(DEV).eval()
    t0 = time.time()
    with torch.no_grad():
        base = start
        while base < len(seqs):
            max_len = len(seqs[base])
            if max_len <= 256:
                batch_size = max(1, args.batch * 4)
            elif max_len <= 512:
                batch_size = max(1, args.batch * 2)
            elif max_len <= 768:
                batch_size = max(1, args.batch)
            else:
                batch_size = max(1, args.batch // 2)
            chunk = seqs[base:base + batch_size]
            encoded = enc(chunk)
            hidden = encoded.hidden.float()
            mask = encoded.mask.squeeze(-1) > 0
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
                ncode = len(seq) // args.pool
                pooled = hh[:ncode * args.pool].reshape(
                    ncode, args.pool, hh.shape[-1]
                ).mean(1)
                digits = torch.round(
                    (torch.tanh(pooled @ fsq_w.T + fsq_b) + 1)
                    / 2 * (levels - 1)
                ).long()
                codes = torch.zeros(ncode, dtype=torch.long, device=DEV)
                for d in range(latent):
                    codes = codes * levels + digits[:, d]
                mm[pos:pos + ncode] = codes.cpu().numpy().astype(np.uint16)
                pos += ncode
                kept_u.append(uids[base + j])
                kept_l.append(ncode)
            del encoded, hidden, mask

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
                    total_positions=total, vocab=vocab,
                    dataset=str(data_path), dataset_sha256=data_sha,
                    fsq=str(fsq_path), fsq_sha256=fsq_sha,
                    sort_by_length=bool(args.sort_by_length),
                ))
            base = done

    mm.flush()
    if pos != total:
        raise RuntimeError(f"Expected {total:,} positions, wrote {pos:,}")

    lens_t = torch.tensor(kept_l, dtype=torch.long)
    offsets = torch.cumsum(
        torch.cat([torch.zeros(1, dtype=torch.long), lens_t]), 0
    )[:-1]
    torch.save(dict(
        lens=lens_t, offsets=offsets, uids=kept_u,
        n_pos=pos, vocab=vocab, latent=latent, levels=levels, pool=args.pool,
        codes=str(out), dataset=str(data_path), dataset_sha256=data_sha,
        sort_by_length=bool(args.sort_by_length),
        fsq=str(fsq_path), fsq_sha256=fsq_sha,
    ), idx)
    print(f"[save] {out}  {out.stat().st_size/2**20:.1f} MiB", flush=True)
    print(f"[save] {idx}  sequences={len(kept_u):,}", flush=True)
    state_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
