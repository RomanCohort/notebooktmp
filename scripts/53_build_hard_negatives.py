"""Build seed-identical cross-miRNA decoys: hard negatives for relation training.

Why
---
The model scores 0.62 AUROC, but a k-mer logistic regression already reaches
0.63 and seed-match alone reaches 0.61.  The negatives MiRBench ships are
window-matched but *seed-incompatible*: only 0.21% of real negatives carry a
7-mer seed match, versus 24% of positives.  So "does this window contain a seed
complement" almost perfectly separates the classes and the model can coast on it.

A decoy removes that shortcut: keep the window, keep the seed, swap the miRNA
for a different one whose 7-mer seed is identical.  The only way to get such a
pair right is to look at pairing geometry, 3' supplementary contacts, and
accessibility -- exactly the features this upgrade adds.

Same-family miRNAs (mir-17 vs mir-20) are excluded: they share a seed by
definition and very likely do bind, so using them as negatives would inject
label noise.  Family comes from the raw TSV's `noncodingRNA_fam`, joined
through the uid's `(source_dataset, split, n)` triple.

Usage
-----
    python scripts/53_build_hard_negatives.py --dry-run
    python scripts/53_build_hard_negatives.py --max-decoys 300000 --seed 7
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")
SPLIT_MAP = {"train": "train", "test": "test", "leftout": "test"}
SEED_START, SEED_END = 1, 8          # 0-based [1:8] == positions 2..8, 7 nt


def rc(s: str) -> str:
    return s.translate(COMP)[::-1]


def seed_of(mirna: str) -> str:
    m = mirna.upper().replace("U", "T")
    return rc(m[SEED_START:SEED_END]) if len(m) >= SEED_END else ""


def hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y) if len(a) == len(b) else 99


def resolve_raw_dir(data_path: Path, explicit: str) -> Path | None:
    """Locate the raw tsv.gz cache.

    Local runs keep them in <repo>/tmp/mirbench_raw; the Colab notebook downloads
    them next to mirbench.jsonl (DATA_DIR/mirbench_raw).  Getting this wrong
    silently yields zero decoys because every family lookup misses, so the
    caller must fail loudly instead of continuing.
    """
    if explicit:
        p = Path(explicit)
        return p if p.is_dir() else None
    for cand in (data_path.parent / "mirbench_raw",
                 ROOT / "tmp/mirbench_raw",
                 ROOT / "data/processed/mirbench_raw"):
        if cand.is_dir() and any(cand.glob("*.tsv.gz")):
            return cand
    return None


def raw_path_for(raw_dir: Path, source: str, split: str) -> Path | None:
    cands = list(raw_dir.glob(f"{source}_*.tsv.gz"))
    want = f"_{split}.tsv.gz"
    exact = [p for p in cands if p.name.endswith(want)]
    if exact:
        return exact[0]
    # leftout rows live in the same file as test for Manakov, and the full
    # dataset file is never part of the build.
    other = [p for p in cands if "full_dataset" not in p.name]
    return other[0] if len(other) == 1 else None


def load_families(raw_dir: Path, sources: set[str]) -> tuple[dict[tuple[str, str, int], str],
                                                            dict[str, str]]:
    """(source, split, n) -> family, and miRNA sequence -> family."""
    key2fam: dict[tuple[str, str, int], str] = {}
    seq2fam: dict[str, str] = {}
    for source in sorted(sources):
        files = [p for p in raw_dir.glob(f"{source}_*.tsv.gz")
                 if "full_dataset" not in p.name]
        for p in files:
            split_raw = p.name[len(source) + 1:-len(".tsv.gz")]
            split = SPLIT_MAP.get(split_raw, split_raw)
            n = 0
            with gzip.open(p, "rt", encoding="utf-8", errors="replace",
                           newline="") as fh:
                for row in csv.DictReader(fh, delimiter="\t"):
                    fam = (row.get("noncodingRNA_fam") or "unknown").strip()
                    seq = (row.get("noncodingRNA") or "").upper().replace("U", "T")
                    key2fam[(source, split, n)] = fam
                    if fam and fam != "unknown":
                        seq2fam.setdefault(seq, fam)
                    n += 1
            print(f"  [fam] {p.name}  {n:,} rows -> split {split}", flush=True)
    return key2fam, seq2fam


def parse_uid(uid: str) -> tuple[str, str, int] | None:
    parts = uid.split(":")
    if len(parts) != 4 or parts[0] != "mb":
        return None
    try:
        return parts[1], parts[2], int(parts[3])
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/processed/mirbench.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data/processed/mirbench_hardneg.jsonl"))
    ap.add_argument("--max-decoys", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seed-mismatch", type=int, default=1,
                    help="允许的 seed 错配位数；0 表示必须完全相同")
    ap.add_argument("--max-per-mirna", type=int, default=0,
                    help="单个 decoy miRNA 的配额上限；0 表示不限制")
    ap.add_argument("--raw-dir", default="",
                    help="原始 tsv.gz 目录；留空则自动探测")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    data = Path(args.data)

    # ---- pass 1: positives, windows, seed->miRNA map ----------------------
    positives: set[tuple[str, str]] = set()      # (mirna, window) label 1 anywhere
    seen_pairs: set[tuple[str, str]] = set()     # any (mirna, window) row
    train_positives: list[dict] = []
    seed2mirna: dict[str, set[str]] = defaultdict(set)
    mirna_seed: dict[str, str] = {}
    test_windows: set[str] = set()
    sources: set[str] = set()
    n_rows = n_target = n_region = 0

    with data.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            n_rows += 1
            kind = r.get("kind")
            src = r.get("source_dataset") or ""
            sources.add(src)
            mi = (r.get("mirna") or "").upper().replace("U", "T")
            win = (r.get("mrna_window") or "").upper()
            if kind == "mirna_region":
                n_region += 1
                continue
            n_target += 1
            if r.get("split") == "test":
                test_windows.add(win)
            positives.add((mi, win))
            seen_pairs.add((mi, win))
            if not mi:
                continue
            if mi not in mirna_seed:
                mirna_seed[mi] = seed_of(mi)
            seed2mirna[mirna_seed[mi]].add(mi)
            if r.get("split") == "train" and r.get("label") == 1:
                train_positives.append(r)

    print(f"[data] rows {n_rows:,}  target {n_target:,}  region {n_region:,}  "
          f"positives {len(positives):,}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"[seed] {len(seed2mirna):,} seeds -> {len(mirna_seed):,} miRNAs",
          flush=True)

    # ---- families ---------------------------------------------------------
    raw_dir = resolve_raw_dir(data, args.raw_dir)
    if raw_dir is None:
        raise SystemExit(
            "找不到原始 tsv.gz（noncodingRNA_fam 列只在那里）。"
            "用 --raw-dir 指定，或先跑 build_mirbench 下载。"
            f"已尝试: {data.parent / 'mirbench_raw'}, {ROOT / 'tmp/mirbench_raw'}")
    print(f"[fam] raw dir {raw_dir}", flush=True)
    key2fam, seq2fam = load_families(raw_dir, sources)
    if not seq2fam:
        raise SystemExit(
            f"{raw_dir} 里没读到任何 miRNA 家族标注 —— 路径对了但文件不对，"
            "拒绝生成一个全是 unknown 家族的 decoy 集合")
    print(f"[fam] {len(key2fam):,} raw rows, {len(seq2fam):,} miRNAs named",
          flush=True)

    def family_of(row: dict, mi: str) -> str:
        parsed = parse_uid(row.get("uid") or "")
        if parsed and parsed in key2fam:
            f = key2fam[parsed]
            if f and f != "unknown":
                return f
        return seq2fam.get(mi, "unknown")

    # ---- pass 2: build decoy candidates -----------------------------------
    rng = random.Random(args.seed)
    cands: list[tuple[dict, str, str]] = []   # (source row, decoy miRNA, family)
    zero = 0
    fam_excluded = 0
    pos_conflict = 0
    tol = args.seed_mismatch

    # Precompute, once, every seed within `tol` mismatches of each seed.  Doing
    # this per positive row instead is ~1.6 billion comparisons (1.25M rows x
    # 1,334 seeds) and takes hours; the neighbourhood map is ~1,334 x 22 lookups.
    seed_list = sorted(seed2mirna)
    if tol == 0:
        near_map = {s: [s] for s in seed_list}
    else:
        def neighbours(s: str) -> list[str]:
            out = []
            for t in seed_list:
                if hamming(t, s) <= tol:
                    out.append(t)
            return out
        near_map = {s: neighbours(s) for s in seed_list}
        print(f"[seed] neighbourhoods: mean {sum(map(len, near_map.values()))/len(near_map):.1f} "
              f"seeds within {tol} mismatch", flush=True)

    for row in train_positives:
        mi = (row.get("mirna") or "").upper().replace("U", "T")
        win = (row.get("mrna_window") or "").upper()
        sd = mirna_seed.get(mi, "")
        if not sd:
            zero += 1
            continue
        fam_a = family_of(row, mi)
        near = near_map.get(sd, [])
        options = [b for s in near for b in seed2mirna[s] if b != mi]
        if not options:
            zero += 1
            continue
        kept = []
        for b in options:
            fam_b = seq2fam.get(b, "unknown")
            # A miRNA with no family annotation cannot be proven to be outside
            # fam_a, so it must not be used as a decoy: otherwise 30% of the
            # pool lands on `unknown` and the shortcut comes back.
            if fam_b == "unknown":
                fam_excluded += 1
                continue
            if fam_a == fam_b:
                fam_excluded += 1
                continue
            if (b, win) in positives:
                pos_conflict += 1
                continue
            kept.append(b)
        if not kept:
            zero += 1
            continue
        b = rng.choice(sorted(kept))
        cands.append((row, b, seq2fam.get(b, "unknown")))

    print(f"[decoy] eligible positives {len(train_positives):,}  "
          f"with candidates {len(cands):,}  zero-candidate {zero:,} "
          f"({100*zero/max(len(train_positives),1):.1f}%)", flush=True)
    print(f"[decoy] family-excluded {fam_excluded:,}  "
          f"positive-conflict {pos_conflict:,}", flush=True)

    # one decoy per window, and never a window that also appears in test:
    # a train decoy built on a test window would leak test information into
    # training and inflate the AUROC we are trying to measure.
    by_window: dict[str, list[tuple[dict, str, str]]] = defaultdict(list)
    leak = 0
    for row, b, fam in cands:
        win = (row.get("mrna_window") or "").upper()
        if win in test_windows:
            leak += 1
            continue
        by_window[win].append((row, b, fam))
    print(f"[decoy] distinct windows {len(by_window):,}", flush=True)
    print(f"[leak] dropped {leak:,} decoys whose window also appears in test",
          flush=True)

    # ---- balanced sampling -------------------------------------------------
    # Naive uniform sampling concentrates on whichever decoy miRNAs happen to
    # have many same-seed, different-family partners: the first build put 34%
    # of all decoys on one family.  That teaches "mir-515 never binds", which is
    # just a new shortcut.  Round-robin over decoy miRNAs instead, so each one
    # contributes an equal share, and prefer the miRNA with the fewest picks so
    # far whenever a window offers several.
    per_mi: dict[str, int] = defaultdict(int)
    picks: list[tuple[dict, str, str]] = []
    remaining = list(by_window.values())
    rng.shuffle(remaining)
    cap = args.max_per_mirna
    while remaining and len(picks) < args.max_decoys:
        keep = []
        for options in remaining:
            if len(picks) >= args.max_decoys:
                break
            usable = [o for o in options
                      if cap <= 0 or per_mi[o[1]] < cap]
            if not usable:
                continue
            best = min(usable, key=lambda o: per_mi[o[1]])
            picks.append(best)
            per_mi[best[1]] += 1
            rest = [o for o in options if o[1] != best[1]]
            if rest:
                keep.append(rest)
        remaining = keep
    picked = picks
    print(f"[decoy] picked {len(picked):,} across {len(per_mi):,} decoy miRNAs",
          flush=True)

    # Tier A: real negatives that already carry a >=6-mer seed
    tier_a = 0
    with data.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("kind") != "mirna_target" or r.get("split") != "train":
                continue
            if r.get("label") != 0:
                continue
            mi = (r.get("mirna") or "").upper().replace("U", "T")
            win = (r.get("mrna_window") or "").upper()
            s7 = seed_of(mi)
            if s7 and s7 in win:
                tier_a += 1
                continue
            s6 = seed_of(mi[SEED_START - 1:SEED_END]) if len(mi) >= SEED_END else ""
            if s6 and s6 in win:
                tier_a += 1
    print(f"[tierA] real train negatives with >=6-mer seed: {tier_a:,}", flush=True)

    n = min(args.max_decoys, len(picked))
    if n < args.max_decoys:
        print(f"[warn] only {len(picked):,} candidates < requested {args.max_decoys:,}",
              flush=True)
    # 写出空文件是这里最容易犯的错：文件存在，notebook 下一格就跳过生成，
    # 训练则拿着 0 条困难负样本继续跑，指标看起来正常但实验已经废了。
    if n == 0:
        raise SystemExit(
            "0 个 decoy —— 上游有问题（家族路径 / seed 映射 / 候选过滤），"
            "拒绝写出空文件")
    rng.shuffle(picked)
    picked = picked[:n]

    if args.dry_run:
        print(f"[dry] would write {n:,} decoys  ({time.time()-t0:.0f}s)", flush=True)
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row, b, fam in picked:
            rec = dict(row)
            rec["uid"] = f"hn:{row['uid']}:{hashlib.sha1(b.encode()).hexdigest()[:8]}"
            rec["mirna"] = b
            rec["label"] = 0
            rec["answer"] = "否"
            rec["kind"] = "mirna_target"
            rec["hard"] = "decoy"
            rec["decoy_family"] = fam
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(out)
    h = hashlib.sha256()
    with out.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 23), b""):
            h.update(chunk)
    print(f"[save] {out}  {n:,} rows  {out.stat().st_size/2**20:.1f} MiB  "
          f"{(time.time()-t0)/60:.1f} min", flush=True)
    print(f"[sha256] {h.hexdigest()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
