"""指令数据：把 abstract / annotation 拆成 QA 对。

两个变体（论文都报了数，所以两条都要造）：

    AIS  —— abstract → sentence → QA
    D&C  —— abstract → 分组摘要 → annotation → QA

闸门：生成的 QA 立刻过 `rnaqwen.data.guard`，出现碱基串就整条丢弃
（PLAN §3 P0 的完成判据）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rnaqwen.data.guard import check_record_qa
from rnaqwen.data.prompts import QA_FROM_TEXT, SYSTEM_RNA_EXPERT, fill
from rnaqwen.llm import Generator, extract_json, strip_thinking


@dataclass
class QaBuildResult:
    pairs: list[dict] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)   # 被闸门拦下的
    raw: str = ""
    error: str = ""


def _normalize_pairs(obj) -> list[dict]:
    """模型可能返回 [{"question","answer"}]、{"qa":[...]}、或 [["q","a"]]，都归一。"""
    items = obj
    if isinstance(obj, dict):
        for key in ("qa", "pairs", "data", "items", "questions"):
            if key in obj and isinstance(obj[key], list):
                items = obj[key]
                break
        else:
            items = [obj]
    out: list[dict] = []
    for it in items if isinstance(items, list) else []:
        if isinstance(it, dict):
            q = it.get("question") or it.get("q") or it.get("Q")
            a = it.get("answer") or it.get("a") or it.get("A")
            if q and a:
                out.append({"question": str(q).strip(), "answer": str(a).strip()})
        elif isinstance(it, (list, tuple)) and len(it) >= 2:
            out.append({"question": str(it[0]).strip(), "answer": str(it[1]).strip()})
    return out


def generate_qa(
    text: str,
    gen: Generator,
    n: int = 3,
    source: str = "annotation",
    uid: str = "",
    sequence: str = "",
    temperature: float = 0.6,
    max_new_tokens: int = 900,
    enforce_no_sequence: bool = True,
    min_answer_chars: int = 20,
) -> QaBuildResult:
    """从一段文本生成 n 组 QA；不合法/带碱基串的直接丢。"""
    res = QaBuildResult()
    if not text or not text.strip():
        res.error = "空文本"
        return res

    prompt = fill(QA_FROM_TEXT, text=text.strip(), n=n)
    out = gen.chat(
        [[{"role": "system", "content": SYSTEM_RNA_EXPERT}, {"role": "user", "content": prompt}]],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )[0]
    res.raw = strip_thinking(out)

    try:
        obj = extract_json(out)
    except ValueError as exc:
        res.error = str(exc)
        return res

    pairs = []
    for p in _normalize_pairs(obj):
        p["source"] = source
        p["generator"] = gen.name
        p["kind"] = "factoid"
        if len(p["answer"]) >= min_answer_chars and len(p["question"]) >= 8:
            pairs.append(p)
        else:
            res.dropped.append({**p, "reason": "too_short"})

    if enforce_no_sequence and sequence:
        rep = check_record_qa(uid or "record", pairs, sequence)
        if not rep.ok:
            bad = set(rep.exact_hits) | set(rep.ngram_hits)
            kept = []
            for p in pairs:
                blob = f"{p['question']} {p['answer']}"
                if any(h in blob.upper().replace("T", "U") for h in bad):
                    res.dropped.append({**p, "reason": "sequence_leak"})
                else:
                    kept.append(p)
            pairs = kept

    res.pairs = pairs
    return res
