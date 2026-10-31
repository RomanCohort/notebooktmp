"""Semantic Precision / Recall / F1 —— 论文的第二套指标。

论文口径（PLAN §0）：用三种嵌入算余弦相似度
    S_BERT（BERT）· S_Pub（PubMedBERT）· S_GPT（OpenAI text-embedding-3-large）

本实现的口径（写死、可对数）
----------------------------
  * **句子级** max-cosine：
      P = mean_i  max_j  cos(gen_i, ref_j)      （gen 有几句就平均几句）
      R = mean_j  max_i  cos(gen_j, ref_i)
      F1 = 2PR/(P+R)
  * 句子切分：换行 + `.!?` 结尾切分，丢弃空句
  * 嵌入：句向量，L2 归一化后点积（= 余弦）
  * **不做** BERTScore 的 baseline rescaling —— 论文报的 0.86 这种量级
    是原始余弦，不是 rescale 之后的数

三种嵌入对应 `--embedder` 的 `bert` / `pubmedbert` / `gpt`。
`gpt` 那条走 OpenAI 兼容接口（本机可以是任何 OpenAI 协议的网关）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Protocol, Sequence

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

DEFAULT_EMBEDDERS: dict[str, str] = {
    "bert": "google-bert/bert-base-uncased",
    "pubmedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "gpt": "openai:text-embedding-3-large",
}

QWEN3_EMBEDDING_LOCAL = "models/Qwen3-Embedding-8B"
QWEN3_EMBEDDING_TASK = (
    "Given a description of an RNA molecule, retrieve passages that describe the same RNA"
)


def split_sentences(text: str, min_len: int = 2) -> list[str]:
    """句子切分；丢掉长度 < min_len 的碎片。"""
    if not text:
        return []
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text)]
    return [p for p in parts if len(p) >= min_len]


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, sentences: Sequence[str]) -> list[list[float]]: ...


@dataclass
class HFEmbedder:
    """句向量：最后一层 hidden state 池化后 L2 归一化。

    `pooling` 三选一：
      * `mean`  —— mask 加权的平均池化（BERT / PubMedBERT 用这个，BERTScore 的常见做法）
      * `cls`   —— 取 [CLS]
      * `last`  —— 取**最后一个非 padding token**（Qwen3-Embedding 的规定做法）

    `instruction`：Qwen3-Embedding 这类模型建议给**查询**加指令前缀
    （`Instruct: {task}\nQuery: {text}`），文档侧不加。留空则不加。
    """

    model_id: str
    name: str = ""
    pooling: str = "mean"
    device: str | None = None
    batch_size: int = 64
    max_length: int = 512
    instruction: str = ""

    def __post_init__(self) -> None:
        self.name = self.name or self.model_id

    def _lazy(self):
        if getattr(self, "_model", None) is None:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._torch = torch
            # last-token pooling 必须让 padding 在左边，否则"最后一个非 pad token"不是句尾
            self._tok = AutoTokenizer.from_pretrained(
                self.model_id, padding_side="left" if self.pooling == "last" else "right"
            )
            self._model = AutoModel.from_pretrained(self.model_id)
            self._model.eval()
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self.device)
            self.dim = int(self._model.config.hidden_size)
        return self._model

    def _prepare(self, sentences: Sequence[str]) -> list[str]:
        if not self.instruction:
            return list(sentences)
        prefix = f"Instruct: {self.instruction}\nQuery: "
        return [prefix + s for s in sentences]

    def encode(self, sentences: Sequence[str]) -> list[list[float]]:
        model = self._lazy()
        torch = self._torch
        texts = self._prepare(sentences)
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            if not batch:
                continue
            enc = self._tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"]
            if self.pooling == "cls":
                vec = hidden[:, 0]
            elif self.pooling == "last":
                # 左 padding ⇒ 每行最后一个位置就是句尾
                idx = torch.full((hidden.shape[0],), hidden.shape[1] - 1, device=hidden.device)
                vec = hidden[torch.arange(hidden.shape[0], device=hidden.device), idx]
            else:
                m = mask.unsqueeze(-1).to(hidden.dtype)
                vec = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-6)
            vec = torch.nn.functional.normalize(vec, p=2, dim=-1)
            out.extend(vec.float().cpu().tolist())
        return out


@dataclass
class OpenAICompatibleEmbedder:
    """OpenAI 协议的无状态嵌入客户端（不依赖 openai 包，用标准库发请求）。"""

    model_id: str = "text-embedding-3-large"
    name: str = "text-embedding-3-large"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    batch_size: int = 64
    timeout: int = 60

    def encode(self, sentences: Sequence[str]) -> list[list[float]]:
        import json
        import math
        import urllib.request

        out: list[list[float]] = []
        for i in range(0, len(sentences), self.batch_size):
            batch = list(sentences[i : i + self.batch_size])
            if not batch:
                continue
            body = json.dumps({"model": self.model_id, "input": batch}).encode()
            req = urllib.request.Request(
                f"{self.base_url.rstrip('/')}/embeddings",
                data=body,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode())
            for item in payload["data"]:
                v = item["embedding"]
                n = math.sqrt(sum(x * x for x in v)) or 1.0
                out.append([x / n for x in v])
        return out


def build_embedder(spec: str, **kwargs) -> Embedder:
    """`bert` / `pubmedbert` / `gpt` / `qwen3emb`，或任意 HF repo id / `openai:<model>`。

    `qwen3emb` = 本地 Qwen3-Embedding-8B（见下），可用环境变量 `QWEN3_EMBEDDING_PATH` 指到别处。
    想手动配池化/指令，直接用 HFEmbedder：
        HFEmbedder("models/Qwen3-Embedding-8B", pooling="last", instruction="...")
    """
    if spec == "qwen3emb":
        # 原版第三列 S_GPT 用 OpenAI text-embedding-3-large（闭源、要 key）。
        # 这里换成**同量级的开源旗舰** Qwen3-Embedding-8B：本机可跑、离线可复现。
        # 它不是同一个模型 ⇒ 这一列**不能**拿来跟原报数逐位对齐，报告里必须写明。
        import os

        path = os.environ.get("QWEN3_EMBEDDING_PATH", QWEN3_EMBEDDING_LOCAL)
        if not Path(path).exists():
            raise FileNotFoundError(
                f"本地没有 Qwen3-Embedding 权重：{path}\n"
                f"下载：python scripts/00b_download_modelscope.py --repo Qwen/Qwen3-Embedding-8B --dest {path}"
            )
        kwargs.setdefault("pooling", "last")
        kwargs.setdefault("instruction", QWEN3_EMBEDDING_TASK)
        return HFEmbedder(model_id=path, name="qwen3-embedding-8b", **kwargs)
    if spec in DEFAULT_EMBEDDERS:
        spec = DEFAULT_EMBEDDERS[spec]
    if spec.startswith("openai:"):
        return OpenAICompatibleEmbedder(model_id=spec.split(":", 1)[1], name=spec, **kwargs)
    kwargs.setdefault("name", spec)
    return HFEmbedder(model_id=spec, **kwargs)


def _cos_matrix(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[sum(x * y for x, y in zip(va, vb)) for vb in b] for va in a]


@dataclass
class SemanticScores:
    precision: float
    recall: float
    f1: float
    embedder: str
    n: int = 0
    n_sent_hyp: int = 0
    n_sent_ref: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def semantic_scores(
    hypotheses: Iterable[str],
    references: Iterable[str],
    embedder: Embedder,
) -> SemanticScores:
    """数据集级 semantic P/R/F1（逐样本算完再平均）。"""
    hyps = [split_sentences(h) for h in hypotheses]
    refs = [split_sentences(r) for r in references]
    if len(hyps) != len(refs):
        raise ValueError(f"hypotheses/references 长度不一致: {len(hyps)} vs {len(refs)}")

    flat_h = [s for doc in hyps for s in doc]
    flat_r = [s for doc in refs for s in doc]
    vecs_h = embedder.encode(flat_h) if flat_h else []
    vecs_r = embedder.encode(flat_r) if flat_r else []

    ih = ir = 0
    p_acc = r_acc = f_acc = 0.0
    counted = 0
    for doc_h, doc_r in zip(hyps, refs):
        vh = vecs_h[ih : ih + len(doc_h)]
        vr = vecs_r[ir : ir + len(doc_r)]
        ih += len(doc_h)
        ir += len(doc_r)
        if not vh or not vr:
            continue
        m = _cos_matrix(vh, vr)
        p = sum(max(row) for row in m) / len(vh)
        r = sum(max(m[i][j] for i in range(len(vh))) for j in range(len(vr))) / len(vr)
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        p_acc += p
        r_acc += r
        f_acc += f
        counted += 1

    n = max(counted, 1)
    return SemanticScores(
        precision=p_acc / n,
        recall=r_acc / n,
        f1=f_acc / n,
        embedder=embedder.name,
        n=counted,
        n_sent_hyp=len(flat_h),
        n_sent_ref=len(flat_r),
    )
