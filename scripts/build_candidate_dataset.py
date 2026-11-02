"""方案 C：把「猜家族名」改成「从候选里选」——生产可用的判别式数据集。

为什么换形态
------------
原数据是按家族切分的（train 1198 家族 / test 483 家族，交集 0），
模型必须「背」出没见过的家族名才能答对 —— 这在结构上不可能，
所以 val CE 在 step 1000 就到顶、再训也不改善。

生产要的是：给一条 RNA + 一个问题，能给出有用答案。
把答案空间从「1947 个自由文本家族名」换成「N 个候选里选一个」：
  · 正确家族名直接出现在 prompt 里，模型不需要背
  · 对没见过的家族也有效（只要候选里有它）
  · 输出是一个序号（A/B/C/D），token 极短，易学、易评

输出格式（一行一条 JSONL）
    {
      "uid": ..., "sequence": ..., "split": ...,
      "kind": "candidate_family",
      "question": "Which of these Rfam families does this RNA belong to?\nA) tRNA\nB) 5S_rRNA\nC) U6\nD) RF01960",
      "answer": "A",
      "gold_family": "RF00005",
      "candidates": ["RF00005", "RF00177", ...]
    }

用法
    python scripts/_build_candidate_dataset.py --out data/processed/rnaqa_v3_cand.jsonl \
        --n-choices 4 --min-seqs 5
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed/rnaqa_v3_l234.jsonl")
    ap.add_argument("--out", default="data/processed/rnaqa_v3_cand.jsonl")
    ap.add_argument("--n-choices", type=int, default=4, help="每题候选数（含正确项）")
    ap.add_argument("--min-seqs", type=int, default=5,
                    help="进候选池的家族至少要有这么多条不同序列")
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--struct-ratio", type=float, default=0.3,
                    help="结构题相对候选题的保留比例（0=不要结构题）")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # 1) 第一遍：收集每个家族的序列（uid -> seq）与家族样本数
    fam_seqs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    rows = []
    with (ROOT / args.src).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append(r)
            if r.get("template") != "l2_family_acc":
                continue
            fam = r["answer"]
            if fam == "unassigned":
                continue
            key = (fam, r["uid"])
            if key in seen:
                continue
            seen.add(key)
            fam_seqs[fam].append((r["uid"], r.get("sequence") or ""))

    pool = [f for f, v in fam_seqs.items() if len(v) >= args.min_seqs]
    pool.sort()
    print(f"[pool] 家族总数 {len(fam_seqs)}，>= {args.min_seqs} 条序列的 {len(pool)} 个")

    # 2) 第二遍：为每条 l2 样本生成候选题
    #    同一个 uid 只取一次（原来 l2_family_acc/name/rna_type 三份重复）
    by_uid: dict[str, dict] = {}
    for r in rows:
        if r.get("template") != "l2_family_acc":
            continue
        fam = r["answer"]
        if fam not in pool:
            continue
        if r["uid"] not in by_uid:
            by_uid[r["uid"]] = r

    out = []
    skipped = 0
    for uid, r in by_uid.items():
        fam = r["answer"]
        distract = [f for f in rng.sample(pool, min(len(pool), args.n_choices * 3))
                    if f != fam][: args.n_choices - 1]
        if len(distract) < args.n_choices - 1:
            skipped += 1
            continue
        cands = distract + [fam]
        rng.shuffle(cands)
        letters = "ABCDEFGH"[: len(cands)]
        correct = letters[cands.index(fam)]
        opts = "\n".join(f"{letters[i]}) {c}" for i, c in enumerate(cands))
        q = ("Which of these Rfam families does this RNA belong to?\n"
             + opts + "\nAnswer with the letter only.")
        # **按序列随机划分，不用原始 split。** 原始数据是按家族切的
        # （train/test 家族交集 0），沿用的话模型永远学不会「见过的家族选对」。
        # 候选式任务的全部价值就在于此，所以这里必须让同一家族同时出现在
        # train 和 test 里。
        rr = rng.random()
        new_split = "test" if rr < 0.15 else ("val" if rr < 0.23 else "train")
        out.append({
            "uid": uid,
            "sequence": r.get("sequence") or "",
            "split": new_split,
            "kind": "candidate_family",
            "question": q,
            "answer": correct,
            "gold_family": fam,
            "candidates": cands,
        })

    # 3) 结构类问答（l3_*）按比例降采样。
    #    它们占原始数据的 77%，如果全留会把候选题淹没，而它们本身是
    #    数值回归（0.70 / -26.9），模型极难学准 —— 留一部分当辅助信号即可。
    struct_rows = [r for r in rows
                   if str(r.get("template", "")).startswith("l3_")]
    n_cand = len(out)
    keep_struct = min(len(struct_rows), max(n_cand, 1) * args.struct_ratio)
    struct_rows = rng.sample(struct_rows, int(keep_struct))
    n_struct = 0
    for r in struct_rows:
        rr = rng.random()
        new_split = "test" if rr < 0.15 else ("val" if rr < 0.23 else "train")
        out.append({
            "uid": r["uid"], "sequence": r.get("sequence") or "",
            "split": new_split,
            "kind": r.get("kind", "struct"),
            "question": r["question"], "answer": r["answer"],
        })
        n_struct += 1

    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    sp = Counter(r["split"] for r in out)
    kinds = Counter(r["kind"] for r in out)
    print(f"[out] {outp}")
    print(f"  候选题 {len(by_uid) - skipped:,} 条（跳过 {skipped}）  结构题 {n_struct:,} 条")
    print(f"  split 分布: {dict(sp)}")
    print(f"  kind 分布: {dict(kinds)}")
    print(f"  每题候选数: {args.n_choices}，随机基线准确率 {100/args.n_choices:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
