"""Build listwise ranking groups: 1 positive + K-1 same-miRNA negatives.

Why this task
-------------
The binary yes/no task is seed-dominated: a k-mer baseline reaches AUROC 0.631
and the 35B model only 0.645.  A listwise task asks a sharper question --
given a miRNA and K windows for that same miRNA, which one is the bound site?
Random chance is 1/K, and the negatives are same-miRNA so the model cannot
solve it by learning "this miRNA's seed".

Coverage (verified on the real data): 100.0% of train positives and 99.6% of
test positives have at least 3 same-miRNA negatives, so K=4 is safe.

Usage
-----
    python scripts/54_build_ranking_groups.py --dry-run
    python scripts/54_build_ranking_groups.py --k 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def collect(path: Path):
    """positives[split] -> list of rows; negatives[split][mirna] -> list of rows."""
    positives: dict[str, list[dict]] = defaultdict(list)
    negatives: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    n_rows = n_pos = n_neg = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("kind") != "mirna_target":
                continue
            n_rows += 1
            split = r.get("split")
            mi = (r.get("mirna") or "").upper().replace("U", "T")
            if not mi:
                continue
            row = {"mirna": mi, "window": (r.get("mrna_window") or "").upper(),
                   "region": r.get("region") or "",
                   "source": r.get("source_dataset") or ""}
            if r.get("label") == 1:
                positives[split].append(row)
                n_pos += 1
            else:
                negatives[split][mi].append(row)
                n_neg += 1
    return positives, negatives, n_rows, n_pos, n_neg


def build_groups(positives, neg_by_mi, k: int, max_groups: int, seed: int):
    rng = random.Random(seed)
    rows = list(positives)
    rng.shuffle(rows)
    groups = []
    skipped = 0
    for r in rows:
        pool = neg_by_mi.get(r["mirna"], [])
        if len(pool) < k - 1:
            skipped += 1
            continue
        picked = rng.sample(pool, k - 1)
        groups.append({
            "mirna": r["mirna"],
            "windows": [r["window"]] + [p["window"] for p in picked],
            "regions": [r["region"]] + [p["region"] for p in picked],
            "sources": [r["source"]] + [p["source"] for p in picked],
        })
        if max_groups and len(groups) >= max_groups:
            break
    return groups, skipped


def write_groups(path: Path, groups: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for g in groups:
            fh.write(json.dumps(g, ensure_ascii=False) + "\n")
    tmp.replace(path)
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 23), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/processed/mirbench.jsonl"))
    ap.add_argument("--out-train", default=str(ROOT / "data/processed/mirbench_rank_train.jsonl"))
    ap.add_argument("--out-test", default=str(ROOT / "data/processed/mirbench_rank_test.jsonl"))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-groups", type=int, default=200_000)
    ap.add_argument("--max-test-groups", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    positives, negatives, n_rows, n_pos, n_neg = collect(Path(args.data))
    print(f"[data] target rows {n_rows:,}  pos {n_pos:,}  neg {n_neg:,}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    if args.dry_run:
        for split in ("train", "test"):
            ps = positives.get(split, [])
            total = len(ps)
            cov = sum(1 for r in ps
                      if len(negatives[split].get(r["mirna"], [])) >= args.k - 1)
            print(f"[{split}] positives {total:,}  covered by >={args.k-1} "
                  f"negatives {cov:,} ({100*cov/max(total,1):.1f}%)", flush=True)
        print(f"[dry] k={args.k} max_groups={args.max_groups} "
              f"max_test={args.max_test_groups}", flush=True)
        return 0

    for split, out, limit in (("train", args.out_train, args.max_groups),
                              ("test", args.out_test, args.max_test_groups)):
        t1 = time.time()
        groups, skipped = build_groups(positives.get(split, []),
                                       negatives.get(split, {}),
                                       args.k, limit, args.seed)
        if not groups:
            raise SystemExit(f"{split} 一组都没生成 —— 检查 k 或数据路径")
        sha = write_groups(Path(out), groups)
        print(f"[{split}] groups {len(groups):,}  k={args.k}  skipped {skipped:,}  "
              f"{Path(out).stat().st_size/2**20:.1f} MiB  {time.time()-t1:.0f}s",
              flush=True)
        print(f"[{split}] sha256 {sha}", flush=True)
    print(f"[done] {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
