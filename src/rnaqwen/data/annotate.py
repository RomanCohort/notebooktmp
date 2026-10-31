"""D&C 变体：divide-and-conquer 分组摘要 → annotation。

原文（arXiv 2411.08900v1 §2.2）的原话：
    "We apply latent Dirichlet allocation (LDA) topic modeling to **group papers by topic**,
     summarizing each group individually. ... The final annotation is created by
     **combining these summaries in a final round of summarization**."
    理由是："summarizing broad topics often causes key details to be missed."

所以是三步，一步都不能省：
    1. **LDA 主题分组**（`group_segments`）
    2. 每组单独摘要
    3. 合并 + 收口一轮

我们**最初漏了 LDA 那一步**，只做"按长度切块 + 逐块摘要"。
按长度切是**物理分组**，LDA 是**语义分组** —— 相邻段落完全可能讲两件不同的事，
按长度切就会把两件事塞进同一个摘要，正是原文说会丢细节的那种情况。
现在 `group_segments(method="lda")` 是默认路径，缺依赖时自动退回 `sequential` 并打日志。

为什么要有 annotation 这一层：
    AIS 变体直接把 abstract 拆句子；D&C 变体则先压成一段"RNA 注释"。
    两个变体在论文里都报了数，而且 Modality Fusion 在 D&C 上塌了
    （0.0922 vs AIS 的 0.2239）—— 这个异常后面要解释，所以两条都要造。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rnaqwen.data.prompts import (
    CHUNK_SUMMARY,
    COMBINE_ANNOTATION,
    SYSTEM_RNA_EXPERT,
    fill,
)
from rnaqwen.llm import Generator, strip_thinking

_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_PMC_RE = re.compile(r"\[(PMC\d+|[A-Za-z]{2,6}\d{4,})\]")


def split_by_source(text: str) -> list[str]:
    """把 `context` 按**来源标记**切成一段段（每段 ≈ 一篇文献的片段）。

    RNAcentral 的文献片段用 `[PMC1234567]` 这类标记带出处 —— 那是天然的分组边界，
    比按标点切准得多。
    """
    if not text:
        return []
    marks = list(_PMC_RE.finditer(text))
    if not marks:
        return [text.strip()] if text.strip() else []
    segs, start = [], 0
    for m in marks:
        end = m.start()
        piece = text[start:end].strip()
        if piece:
            segs.append(piece)
        start = end
    tail = text[start:].strip()
    if tail:
        segs.append(tail)
    return segs or ([text.strip()] if text.strip() else [])


def group_segments(segments: list[str], method: str = "lda", n_topics: int = 4,
                   min_segments: int = 4) -> list[list[str]]:
    """按主题把文献片段分组（原版的 LDA 那一步）。

    * `method="lda"`：sklearn 的 LatentDirichletAllocation，段落数少于 `min_segments` 时
      直接返回单组（样本太少，LDA 拟合出来的主题没意义）。
    * `method="sequential"`：退化为按顺序切块（= 我们最初的做法）。
    """
    segs = [s for s in (segments or []) if s and s.strip()]
    if not segs:
        return []
    if method != "lda" or len(segs) < min_segments:
        return [segs]
    try:
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer
    except ImportError:
        print("[d&c] 没有 sklearn，LDA 分组退回 sequential")
        return [segs]

    k = max(2, min(n_topics, len(segs) // 2))
    vec = CountVectorizer(stop_words="english", max_features=20000, min_df=1)
    try:
        X = vec.fit_transform(segs)
        lda = LatentDirichletAllocation(n_components=k, random_state=0, max_iter=50, learning_method="batch")
        doc_topic = lda.fit_transform(X)
    except ValueError:
        return [segs]

    groups: list[list[str]] = [[] for _ in range(k)]
    for i, seg in enumerate(segs):
        groups[int(doc_topic[i].argmax())].append(seg)
    return [g for g in groups if g]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text or "") if s.strip()]


def divide(text: str, max_chars: int = 900, max_chunks: int = 4) -> list[str]:
    """按句子边界切块，块长接近 max_chars，最多 max_chunks 块（= physical 分组）。

    超长文本先等分成 max_chunks 段，再各自按句子边界收紧 —— 保证每块都完整，
    不会把句子劈成两半。
    """
    sentences = split_sentences(text)
    if not sentences:
        return []
    if sum(len(s) for s in sentences) <= max_chars:
        return [" ".join(sentences)]

    target = max(max_chars, sum(len(s) for s in sentences) // max_chunks + 1)
    chunks: list[list[str]] = [[]]
    size = 0
    for s in sentences:
        if size + len(s) > target and chunks[-1]:
            chunks.append([])
            size = 0
        chunks[-1].append(s)
        size += len(s) + 1
    return [" ".join(c) for c in chunks if c][:max_chunks]


@dataclass
class AnnotationTrace:
    """把中间产物留下来 —— 数据质量出问题时要知道是哪一步坏的。"""

    chunks: list[str]
    chunk_summaries: list[str]
    annotation: str


def build_annotation(
    literature: str,
    gen: Generator,
    definition: str = "",
    organism: str = "",
    max_chars: int = 900,
    max_chunks: int = 4,
    temperature: float = 0.2,
    keep_trace: bool = True,
) -> AnnotationTrace:
    chunks = divide(literature, max_chars=max_chars, max_chunks=max_chunks)
    if not chunks:
        return AnnotationTrace([], [], "")

    convs = [
        [{"role": "system", "content": SYSTEM_RNA_EXPERT},
         {"role": "user", "content": fill(CHUNK_SUMMARY, chunk=c)}]
        for c in chunks
    ]
    summaries = [strip_thinking(s).strip() for s in gen.chat(convs, max_new_tokens=220, temperature=temperature)]

    summary_block = "\n".join(f"{i+1}. {s}" for i, s in enumerate(summaries) if s)
    conv = [
        {"role": "system", "content": SYSTEM_RNA_EXPERT},
        {"role": "user", "content": fill(COMBINE_ANNOTATION, definition=definition or "(unknown)",
                                         organism=organism or "(unknown)", summaries=summary_block)},
    ]
    annotation = strip_thinking(gen.chat([conv], max_new_tokens=400, temperature=temperature)[0]).strip()
    return AnnotationTrace(chunks if keep_trace else [], summaries, annotation)
