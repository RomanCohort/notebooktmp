"""RNA-QA 记录结构 + JSONL 读写。

一条记录 = 一个 RNA + 它的文献文本 + 生成的注释 + 由它派生的 QA 对。

字段命名对照原论文：
    `sequence`     —— RNA 序列（ACGU）
    `literature`   —— abstract 原文（AIS 变体的来源）
    `annotation`   —— divide-and-conquer 分组摘要（D&C 变体的来源）
    `qa`           —— 指令数据（由 literature / annotation 拆成问答）
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Iterable, Iterator


@dataclass
class QaPair:
    question: str
    answer: str
    source: str = ""          # "abstract" | "annotation"
    generator: str = ""       # 生成器标识（本地 Qwen 版本）
    kind: str = "factoid"     # 自由标签：factoid / summary / reasoning

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class RnaQaRecord:
    uid: str
    sequence: str = ""
    organism: str = ""
    rna_type: str = ""
    definition: str = ""
    literature: str = ""            # abstract
    annotation: str = ""            # D&C 摘要
    qa: list[dict] = field(default_factory=list)
    pmid: str = ""
    nuc_accession: str = ""
    provenance: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RnaQaRecord":
        keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in keys})


def write_jsonl(path: Path, records: Iterable[RnaQaRecord | dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            d = rec.as_dict() if isinstance(rec, RnaQaRecord) else rec
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[RnaQaRecord]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield RnaQaRecord.from_dict(json.loads(line))


def append_jsonl(path: Path, records: Iterable[RnaQaRecord | dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            d = rec.as_dict() if isinstance(rec, RnaQaRecord) else rec
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
    return n
