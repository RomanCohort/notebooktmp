"""Stream RNAQA v3 sequences through ERNIE + the trained FSQ into uint16 codes.

This avoids storing the 68 GiB ERNIE hidden-state cache on disk. The resulting
code cache is about 95 MiB for 47.5M nucleotide positions and is used by
50_routeC_e2e.py with --codes / --codes-idx.

The FSQ encoder is frozen in this path. The LLM adapter and code embedding table
remain trainable.
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
    ap.add_argument("--fsq", default=str(ROOT / "checkpoints/fsq_stream/tokenizer.pt"))
    ap.add_argument("--out", default=str(ROOT / "data/processed/_rnaqa_v3_codes.u16"))
    ap.add_argument("--idx", default=str(ROOT / "data/processed/_rnaqa_v3_codes_index.pt"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--save-every", type=int, default=256)
    ap.add_argument("--min-free-gb", type=float, default=2.0)
    args = ap.parse_args()

    out = Path(args.out)
    idx = Path(args.idx)
    state_path = out.with_suffix(out.suffix + ".state.json")
    uids, seqs = read_unique(Path(args.data))
    lens = [len(s) for s in seqs]
    total = sum(lens)
    need_gb = total * 2 / 2**30
    print(f"[data] {len(seqs):,} sequences / {total:,} positions", flush=True)
    print(f"[data] uint16 cache {need_gb:.3f} GiB", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(out.parent).free / 2**30
    if not out.exists() and free_gb < need_gb + args.min_free_gb:
        raise RuntimeError(
            f"Need {need_gb + args.min_free_gb:.1f} GiB, have {free_gb:.1f} GiB"
        )

    ck = torch.load(args.fsq, map_location="cpu", weights_only=False)
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
        start = int(state["next_index"])
        pos = int(state["positions"])
        kept_u = list(state["kept_uids"])
        kept_l = [int(x) for x in state["kept_lens"]]
        print(f"[resume] sequence {start:,}/{len(seqs):,} position {pos:,}", flush=True)
    else:
        start, pos, kept_u, kept_l = 0, 0, [], []
        atomic_json(state_path, dict(
            next_index=0, positions=0, kept_uids=[], kept_lens=[],
            total_positions=total, vocab=vocab,
        ))

    mm = np.memmap(out, dtype=np.uint16, mode="r+" if out.exists() else "w+",
                   shape=(total,))
    from rnaqwen.models.encoder import RnaEncoder

    enc = RnaEncoder(args.encoder, dtype=torch.float32).to(DEV).eval()
    t0 = time.time()
    with torch.no_grad():
        for base in range(start, len(seqs), args.batch):
            chunk = seqs[base:base + args.batch]
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
                digits = torch.round(
                    (torch.tanh(hh @ fsq_w.T + fsq_b) + 1)
                    / 2 * (levels - 1)
                ).long()
                codes = torch.zeros(digits.shape[0], dtype=torch.long, device=DEV)
                for d in range(latent):
                    codes = codes * levels + digits[:, d]
                mm[pos:pos + len(seq)] = codes.cpu().numpy().astype(np.uint16)
                pos += len(seq)
                kept_u.append(uids[base + j])
                kept_l.append(len(seq))
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
                    kept_uids=kept_u, kept_lens=kept_l,
                    total_positions=total, vocab=vocab,
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
        n_pos=pos, vocab=vocab, latent=latent, levels=levels,
        codes=str(out),
    ), idx)
    print(f"[save] {out}  {out.stat().st_size/2**20:.1f} MiB", flush=True)
    print(f"[save] {idx}  sequences={len(kept_u):,}", flush=True)
    state_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
