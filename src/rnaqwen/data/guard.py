"""验收闸门：QA 里不许出现序列原文。

PLAN §3 P0 完成判据写得很直白：
    "能产出 1,000 条格式正确的 QA，且 QA 里**不出现序列原文**（出现就白训）"

PLAN §5 风险登记也点了名：
    "QA 里复述了序列 → 指标虚高 → 造数据时就用置零检测验收"

所以这个模块是**造数据时的 CI**，不是事后检查。两道闸：

  1. **精确泄漏**：序列（或其长度 >= min_match 的连续子串）出现在文本里。
     做 U/T 归一化 —— 生成器可能把 U 写成 T。
  2. **k-mer 泄漏**：文本里出现 >= k 个连续核苷酸 token 的片段，
     即便它不是 reference 序列的子串（例如模型自己编了一段 ACGU 串）。
     这一条挡的是"答案里堆碱基字母"这种作弊。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

_NT_RE = re.compile(r"[ACGTUacgtu]{6,}")          # 连续 >=6 个核苷酸字母的串
_CLEAN_RE = re.compile(r"[^ACGUacgu]")

# ⚠️ 判据必须是"**独立成词的**核苷酸串"，不能是"把文本里所有非 ACGU 字符删掉剩下的字母"。
#
# 踩过的坑（真踩了）：老写法用 `normalize_sequence(text)` 过滤整段文本再找 k-mer。
# 但**英文散文里全是 a/c/g/u 这四个字母** ——
#   "The expression levels of the immune checkpoints ..." 过滤完只剩
#   "UCCCAAACAACACACAACCC"（20 个字符），而它照样能撞上 1022 nt 序列里的某个 12-mer。
# 实测 29,370 组 QA 上报了 53 处"泄漏"，逐条查下来**全是假阳性**：
#   过滤后的文本只有 20~30 字符（原句 160 字符），命中的都是 AAACAACAACGG 这种低复杂度串，
#   而且**没有一条**能在文本里找到独立成词的核苷酸串。
# 概率上算得通：过滤后 ~20 个字母 ≈ 10 个 12-mer，撞 1011 个候选 ≈ 0.06%，
# 乘 5.8 万条文本 ≈ 35 次 —— 又是生日悖论。
_NT_RUN_RE = re.compile(r"(?<![A-Za-z0-9])[ACGTUacgtu]{6,}(?![A-Za-z0-9])")


def normalize_sequence(seq: str, t_to_u: bool = True) -> str:
    """只留字母并大写；默认把 T 归一成 U（RNA 语境）。

    只对**序列本身**用，别拿去"清洗"自然语言文本（见上面的注释）。
    """
    s = _CLEAN_RE.sub("", seq).upper()
    return s.replace("T", "U") if t_to_u else s


def extract_nt_runs(text: str, min_len: int = 6) -> list[str]:
    """从文本里抠出**独立成词**的核苷酸串（前后都不能是字母数字）。

    这是判断"模型有没有吐碱基串"的正确做法：`ACGUACGU` 是一个词，会被抓到；
    而 "can act as a competing" 里的字母不会被误当成序列。
    """
    if not text:
        return []
    return [m.group(0).upper().replace("T", "U") for m in _NT_RUN_RE.finditer(text)
            if len(m.group(0)) >= min_len]


def hamming_free_kmers(seq: str, k: int) -> set[str]:
    return {seq[i : i + k] for i in range(max(0, len(seq) - k + 1))}


@dataclass
class LeakReport:
    uid: str = ""
    exact_hits: list[str] = field(default_factory=list)      # 出现在文本里的序列子串
    ngram_hits: list[str] = field(default_factory=list)      # 文本里的长核苷酸串
    checked_fields: list[str] = field(default_factory=list)
    ok: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


def find_sequence_leak(
    text: str,
    sequence: str,
    min_match: int = 12,
    kmer: int = 12,
) -> tuple[list[str], list[str]]:
    """返回 (精确命中子串, 长核苷酸串命中)。

    判据只作用于**独立成词的核苷酸串**（`extract_nt_runs`），不再对整段散文做字母过滤 ——
    理由见文件上方关于假阳性的注释。
    """
    if not text or not sequence:
        return [], []
    seq = normalize_sequence(sequence)
    runs = extract_nt_runs(text, min_len=min(6, min_match))

    exact: list[str] = []
    if len(seq) >= min_match:
        kmers = hamming_free_kmers(seq, min_match)
        for run in runs:
            for i in range(max(0, len(run) - min_match + 1)):
                hit = run[i : i + min_match]
                if hit in kmers:
                    exact.append(hit)

    # 文本里独立成词的长核苷酸串 —— 即便不是 reference 的子串，也同样是"在吐碱基"
    ngram_hits = [g for g in runs if len(g) >= kmer]

    return sorted(set(exact)), sorted(set(ngram_hits))


def check_record_qa(
    uid: str,
    qa_pairs: list[dict],
    sequence: str,
    min_match: int = 12,
    kmer: int = 12,
) -> LeakReport:
    """对一个记录的全部 QA 对做泄漏检查（问题 + 答案都查）。"""
    rep = LeakReport(uid=uid)
    for pair in qa_pairs:
        for fld in ("question", "answer"):
            text = pair.get(fld, "") or ""
            rep.checked_fields.append(f"{uid}:{fld}")
            exact, ngrams = find_sequence_leak(text, sequence, min_match=min_match, kmer=kmer)
            rep.exact_hits.extend(exact)
            rep.ngram_hits.extend(ngrams)
    rep.exact_hits = sorted(set(rep.exact_hits))
    rep.ngram_hits = sorted(set(rep.ngram_hits))
    rep.ok = not rep.exact_hits and not rep.ngram_hits
    return rep


# ---------------------------------------------------------------- 另一个闸门
def zero_encoder_probe(encoder, llm) -> dict:
    """PLAN §3 P1 完成判据：置零编码器后，对齐指标必须明显下降。

    这里只给接口占位（真正的实现在 rnaqwen.models 里），
    因为它是 P1 的事；放在这里是为了让"验收"这件事集中在一处。
    """
    raise NotImplementedError("P1 阶段实现在 rnaqwen.models.alignment")
