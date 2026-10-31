"""ROUGE —— 自实现，口径锁定。

为什么要自己写
--------------
论文没给实现细节，而 ROUGE 的不同实现（Google `rouge_score` / pltrdy `rouge` /
原版 ROUGE-1.5.5 perl）在同一份文本上能差好几个点。复现里**口径漂移 = 无法解释的差值**，
所以：

  1. 这里给出一个自实现，口径写死在代码里；
  2. `scripts/03_eval_selfcheck.py` 拿它跟 `rouge_score`（若装了）对数，
     要求逐样本一致（1e-9）；不一致就说明口径错了，不许往下走。

口径（默认 `protocol="rouge_score"`）
------------------------------------
  * 分词：`re.findall(r"[a-z0-9]+", text.lower())`（对齐 rouge_score 的 `_split_into_words`）
  * ROUGE-N：n-gram 计数，F = 2PR/(P+R)
  * ROUGE-L：LCS，`F = (1+β²)PR/(R+β²P)`，**默认 β=1.0**（rouge_score 口径）
    —— 另提供 `beta=1.2`（ROUGE-1.5.5 / pltrdy 默认）备用，报告里要写明用的哪个
  * 多参考：逐参考算 F 后取 **max**（与 rouge_score 的 `use_stemmer=False` 行为一致）
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """rouge_score 口径的分词：小写 + 只保留字母数字（丢弃标点）。"""
    return _WORD_RE.findall(text.lower())


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _pr_f(count_overlap: int, count_hyp: int, count_ref: int, beta: float = 1.0) -> tuple[float, float, float]:
    if count_hyp == 0 or count_ref == 0 or count_overlap == 0:
        return 0.0, 0.0, 0.0
    p = count_overlap / count_hyp
    r = count_overlap / count_ref
    f = (1 + beta * beta) * p * r / (r + beta * beta * p) if (r + beta * beta * p) > 0 else 0.0
    return p, r, f


def rouge_n(hypothesis: str, references: str | Sequence[str], n: int = 1, beta: float = 1.0) -> tuple[float, float, float]:
    """返回 (precision, recall, f)。多参考取 F 最大的那个参考。"""
    refs = [references] if isinstance(references, str) else list(references)
    hyp_tokens = tokenize(hypothesis)
    hyp_grams = _ngrams(hyp_tokens, n)
    best = (0.0, 0.0, 0.0)
    for ref in refs:
        ref_grams = _ngrams(tokenize(ref), n)
        overlap = sum((hyp_grams & ref_grams).values())
        prf = _pr_f(overlap, sum(hyp_grams.values()), sum(ref_grams.values()), beta)
        if prf[2] > best[2]:
            best = prf
    return best


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """经典 O(len(a)*len(b)) LCS 长度，滚动数组省内存。"""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l(
    hypothesis: str,
    references: str | Sequence[str],
    beta: float = 1.0,
) -> tuple[float, float, float]:
    """LCS 版 ROUGE-L。默认 β=1.0（F1）；ROUGE-1.5.5 传统口径是 β=1.2。"""
    refs = [references] if isinstance(references, str) else list(references)
    hyp_tokens = tokenize(hypothesis)
    best = (0.0, 0.0, 0.0)
    for ref in refs:
        ref_tokens = tokenize(ref)
        lcs = _lcs_length(hyp_tokens, ref_tokens)
        prf = _pr_f(lcs, len(hyp_tokens), len(ref_tokens), beta)
        if prf[2] > best[2]:
            best = prf
    return best


def rouge_1(hypothesis: str, references: str | Sequence[str], beta: float = 1.0) -> tuple[float, float, float]:
    return rouge_n(hypothesis, references, 1, beta)


def rouge_2(hypothesis: str, references: str | Sequence[str], beta: float = 1.0) -> tuple[float, float, float]:
    return rouge_n(hypothesis, references, 2, beta)


@dataclass
class RougeScores:
    """论文表里的三个数（都是 F-measure）。"""

    rouge1: float
    rouge2: float
    rougeL: float
    n: int = 0
    beta: float = 1.0
    protocol: str = "rouge_score"

    def as_dict(self) -> dict:
        return asdict(self)


def rouge_scores(
    hypotheses: Iterable[str],
    references: Iterable[str],
    beta: float = 1.0,
    protocol: str = "rouge_score",
) -> RougeScores:
    """对整个数据集算平均 ROUGE-1/2/L（F-measure）。"""
    hyps = list(hypotheses)
    refs = list(references)
    if len(hyps) != len(refs):
        raise ValueError(f"hypotheses/references 长度不一致: {len(hyps)} vs {len(refs)}")
    if not hyps:
        return RougeScores(0.0, 0.0, 0.0, 0, beta, protocol)

    acc = [0.0, 0.0, 0.0]
    for h, r in zip(hyps, refs):
        acc[0] += rouge_1(h, r, beta)[2]
        acc[1] += rouge_2(h, r, beta)[2]
        acc[2] += rouge_l(h, r, beta)[2]
    n = len(hyps)
    return RougeScores(acc[0] / n, acc[1] / n, acc[2] / n, n, beta, protocol)


def rouge_scores_rouge_score(
    hypotheses: Sequence[str], references: Sequence[str], use_stemmer: bool = False
) -> RougeScores:
    """可选后端：直接调 Google `rouge_score`（装了才可用），用于对数自检。"""
    from rouge_score import rouge_scorer  # 局部 import：没装就不影响主路径

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=use_stemmer)
    acc = [0.0, 0.0, 0.0]
    for h, r in zip(hypotheses, references):
        s = scorer.score(r, h)  # rouge_score 的签名是 (target, prediction)
        acc[0] += s["rouge1"].fmeasure
        acc[1] += s["rouge2"].fmeasure
        acc[2] += s["rougeL"].fmeasure
    n = max(len(hypotheses), 1)
    return RougeScores(acc[0] / n, acc[1] / n, acc[2] / n, len(hypotheses), 1.0, "rouge_score-lib")
