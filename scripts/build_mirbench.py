"""miRBench -> RNA-RNA 关系判别训练数据。

miRBench (Zenodo 20540907, CC-BY-4.0) 提供三个 AGO2 实验数据集，
每条是 (mRNA 窗口 50nt, miRNA 21-23nt, label 0/1)：

    AGO2_CLASH_Hejret2023       CLASH，有 train/test
    AGO2_eCLIP_Manakov2022      eCLIP，主力（train 528 MB）
    AGO2_eCLIP_Klimentova2022   eCLIP，仅 test

负样本由 miRBench 的偏差校正方法生成（不是简单随机），所以正负都可信。

本脚本把三个数据集转成两种任务：

  mirna_target   二分类：这个 miRNA 与这个 mRNA 窗口是否有实验验证的靶向关系
  mirna_region   四分类：结合位点位于哪个区域（3'UTR / CDS / 5'UTR / intron）

输出 jsonl，每行一条：

    {"uid":..., "kind":"mirna_target", "split":"train",
     "mirna":"...", "mrna_window":"...", "answer":"Yes"|"No",
     "label":1, "region":"three_prime_utr", "mirna_name":"hsa-miR-..."}

用法
    python scripts/build_mirbench.py --out data/processed/mirbench.jsonl
    python scripts/build_mirbench.py --datasets AGO2_CLASH_Hejret2023 --keep-gz-dir tmp
"""
from __future__ import annotations

import argparse
import gzip
import json
import urllib.request
from collections import Counter
from pathlib import Path

ZENODO = "20540907"
BASE = f"https://zenodo.org/records/{ZENODO}/files"

DATASETS = {
    "AGO2_CLASH_Hejret2023": ["train", "test"],
    "AGO2_eCLIP_Manakov2022": ["train", "test", "leftout"],
    "AGO2_eCLIP_Klimentova2022": ["test"],
}

# 问题模板：中文为默认（Qwen 原生中文强，且 ACGU 与中文天然区分），
# 英文保留做对照。答案标签在两种语言下都保持简短：是/否、3'UTR/CDS/...
PROMPTS = {
    "zh": {
        "target": "这条 miRNA 与这段 mRNA 窗口之间是否存在经实验验证的靶向关系？"
                  "只回答「是」或「否」。",
        "region": "这条 miRNA 的结合位点位于哪个区域？"
                  "只回答「3'UTR」「5'UTR」「CDS」「Intron」之一。",
        "yes": "是", "no": "否",
    },
    "en": {
        "target": "Does this miRNA have an experimentally validated target site "
                  "in this mRNA window? Answer Yes or No.",
        "region": "In which region is this miRNA binding site located? "
                  "Answer with one of 3'UTR, 5'UTR, CDS, Intron.",
        "yes": "Yes", "no": "No",
    },
}

REGION_MAP = {
    "three_prime_utr": "3'UTR",
    "five_prime_utr": "5'UTR",
    "exon": "CDS",
    "intron": "Intron",
}


def fetch(name: str, split: str, cache: Path) -> Path:
    dst = cache / f"{name}_{split}.tsv.gz"
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    url = f"{BASE}/{name}_{split}.tsv.gz?download=1"
    part = dst.with_suffix(".part")
    print(f"[dl] {name} {split} ...", flush=True)
    with urllib.request.urlopen(url, timeout=120) as r, part.open("wb") as fh:
        total = int(r.headers.get("Content-Length", 0))
        got = 0
        while True:
            chunk = r.read(4 << 20)
            if not chunk:
                break
            fh.write(chunk)
            got += len(chunk)
            if total and got % (32 << 20) < (4 << 20):
                print(f"     {got/2**20:.0f}/{total/2**20:.0f} MB", flush=True)
    part.replace(dst)
    return dst


def parse(path: Path):
    with gzip.open(path, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        idx = {k: i for i, k in enumerate(header)}
        need = ["gene", "noncodingRNA", "label"]
        for k in need:
            if k not in idx:
                raise SystemExit(f"{path.name} 缺列 {k}；实际列 {header}")
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            yield {
                "gene": parts[idx["gene"]],
                "mirna": parts[idx["noncodingRNA"]],
                "label": int(parts[idx["label"]]),
                "name": parts[idx["noncodingRNA_name"]] if "noncodingRNA_name" in idx else "",
                "feature": parts[idx["feature"]] if "feature" in idx else "",
                "chr": parts[idx["chr"]] if "chr" in idx else "",
                "start": parts[idx["start"]] if "start" in idx else "",
                "end": parts[idx["end"]] if "end" in idx else "",
            }


def find_region(feature: str) -> str | None:
    """feature 可能是 'exon' / 'three_prime_utr' / 'exon,intron' 等复合值。
    取最具体的单一区域；复合值不猜，直接跳过（region 任务的标签必须干净）。"""
    if not feature or "," in feature:
        return None
    return REGION_MAP.get(feature)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/mirbench.jsonl")
    ap.add_argument("--datasets", default="", help="逗号分隔；空=全部")
    ap.add_argument("--cache", default="tmp/mirbench_raw")
    ap.add_argument("--tasks", default="target,region",
                    help="target=二分类, region=四分类")
    ap.add_argument("--lang", choices=["zh", "en"], default="zh",
                    help="提问语言。zh 为默认（Qwen 原生中文强，ACGU 与中文天然区分）")
    args = ap.parse_args()
    P = PROMPTS[args.lang]
    print(f"[lang] {args.lang}", flush=True)

    want = [d.strip() for d in args.datasets.split(",") if d.strip()] or list(DATASETS)
    tasks = {t.strip() for t in args.tasks.split(",")}
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    region_stats = Counter()
    label_stats = Counter()

    with out_path.open("w", encoding="utf-8") as out:
        for ds in want:
            if ds not in DATASETS:
                raise SystemExit(f"未知数据集 {ds}；可选 {list(DATASETS)}")
            for split in DATASETS[ds]:
                gz = fetch(ds, split, cache)
                n = 0
                for r in parse(gz):
                    uid = f"mb:{ds}:{split}:{n}"
                    base = {
                        "uid": uid,
                        "mirna": r["mirna"],
                        "mrna_window": r["gene"],
                        "split": "test" if split in ("test", "leftout") else "train",
                        "label": r["label"],
                        "region": r["feature"],
                        "mirna_name": r["name"],
                        "source_dataset": ds,
                    }
                    if "target" in tasks:
                        out.write(json.dumps({
                            **base,
                            "kind": "mirna_target",
                            "question": P["target"],
                            "answer": P["yes"] if r["label"] == 1 else P["no"],
                        }, ensure_ascii=False) + "\n")
                        stats["mirna_target"] += 1
                        label_stats[(base["split"], r["label"])] += 1
                    if "region" in tasks and r["label"] == 1:
                        reg = find_region(r["feature"])
                        if reg:
                            out.write(json.dumps({
                                **base,
                                "kind": "mirna_region",
                                "question": P["region"],
                                "answer": reg,   # 区域用缩写，中英通用
                            }, ensure_ascii=False) + "\n")
                            stats["mirna_region"] += 1
                            region_stats[reg] += 1
                    n += 1
                print(f"[ok] {ds}/{split}: {n:,} 行", flush=True)

    print(f"\n[out] {out_path}")
    print("任务分布:", dict(stats))
    print("region 分布:", dict(region_stats))
    print("target 标签分布:", dict(label_stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
