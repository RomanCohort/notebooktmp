"""数据源客户端：NCBI Entrez（eutils）。

为什么不是 RNAcentral
---------------------
PLAN §3 P0 原定源是 **RNAcentral**。实测本机到 EBI 全线不可达：

    rnacentral.org/api/v1/...   SSLError: Max retries exceeded
    ftp.ebi.ac.uk/...           ReadTimeout
    europepmc.org/...           SSLError

而 NCBI eutils 通（实测 esearch/efetch/elink 全部 200）。所以最小版本改用：

    PubMed（摘要） + elink → nuccore（序列）

拿到的是同样是 **(RNA 序列, 文献文本)** 对，管线结构不变。
差异要写进报告：RNAcentral 的 RNA type / 描述字段在 NCBI 侧要用
`gbseq_definition` 之类替代，字段可用性不完全等价。

速率：NCBI 无 key 限 3 req/s，这里串行 + sleep，别被打回。
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class Entrez:
    """最小 Entrez 客户端：限速 + 重试 + JSON/XML 原样返回。"""

    def __init__(self, tool: str = "rnaqwen-repro", email: str = "repro@localhost", api_key: str | None = None,
                 min_interval: float = 0.34, retries: int = 3, timeout: int = 30):
        self.tool = tool
        self.email = email
        self.api_key = api_key
        self.min_interval = min_interval if api_key is None else 0.11
        self.retries = retries
        self.timeout = timeout
        self._last = 0.0

    def _get(self, endpoint: str, params: dict[str, Any]) -> str:
        params = {**params, "tool": self.tool, "email": self.email}
        if self.api_key:
            params["api_key"] = self.api_key
        url = f"{EUTILS}/{endpoint}?{urllib.parse.urlencode(params)}"
        last: Exception | None = None
        for attempt in range(self.retries):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                    body = resp.read().decode("utf-8", "replace")
                self._last = time.time()
                return body
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Entrez {endpoint} 失败: {last}") from last

    # ---- 基本操作 ----
    def esearch(self, db: str, term: str, retmax: int = 20, retstart: int = 0) -> list[str]:
        body = self._get("esearch.fcgi", {"db": db, "term": term, "retmode": "json",
                                          "retmax": retmax, "retstart": retstart, "sort": "relevance"})
        return json.loads(body)["esearchresult"].get("idlist", [])

    def efetch(self, db: str, ids: Iterable[str], retmode: str = "xml", **extra) -> str:
        ids = list(ids)
        if not ids:
            return ""
        return self._get("efetch.fcgi", {"db": db, "id": ",".join(ids), "retmode": retmode, **extra})

    def elink(self, dbfrom: str, db: str, ids: Iterable[str]) -> dict[str, list[str]]:
        body = self._get("elink.fcgi", {"dbfrom": dbfrom, "db": db, "retmode": "json",
                                        "id": ",".join(ids)})
        out: dict[str, list[str]] = {}
        for ls in json.loads(body).get("linksets", []):
            src = None
            for idobj in ls.get("ids", []):
                src = str(idobj)
            links = [str(x) for lsd in ls.get("linksetdbs", []) for x in lsd.get("links", [])]
            if src is not None:
                out[src] = links
        return out


# ---------------------------------------------------------------- 解析
@dataclass
class LiteratureRecord:
    """一条 (序列, 文献) 记录 —— RNA-QA 的原子单位。"""

    uid: str                      # PMID
    title: str = ""
    abstract: str = ""
    journal: str = ""
    year: str = ""
    organism: str = ""
    nucleotide_ids: list[str] = field(default_factory=list)
    sequences: list[str] = field(default_factory=list)
    definitions: list[str] = field(default_factory=list)

    @property
    def primary_sequence(self) -> str:
        return max(self.sequences, key=len) if self.sequences else ""

    def as_dict(self) -> dict:
        d = asdict(self)
        d["primary_sequence"] = self.primary_sequence
        return d


def parse_pubmed_xml(xml: str) -> dict[str, dict]:
    """从 PubmedArticleSet XML 抽 PMID / title / abstract / journal / year / organism。"""
    import re

    out: dict[str, dict] = {}
    for art in re.findall(r"<PubmedArticle>.*?</PubmedArticle>", xml, flags=re.S):
        m = re.search(r"<PMID[^>]*>(\d+)</PMID>", art)
        if not m:
            continue
        pmid = m.group(1)
        title = re.search(r"<ArticleTitle[^>]*>(.*?)</ArticleTitle>", art, flags=re.S)
        journal = re.search(r"<Title>(.*?)</Title>", art, flags=re.S)
        year = re.search(r"<PubDate>.*?<Year>(\d{4})</Year>", art, flags=re.S) or re.search(
            r"<PubMedPubDate[^>]*>.*?<Year>(\d{4})</Year>", art, flags=re.S
        )
        # abstract 可能分多段 (AbstractText 带 Label)，全部拼起来
        chunks = re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", art, flags=re.S)
        organism = re.search(r'<Organism[^>]*>(.*?)</Organism>', art, flags=re.S)
        clean = lambda s: re.sub(r"<[^>]+>", "", s).strip() if s else ""  # noqa: E731
        out[pmid] = {
            "title": clean(title.group(1) if title else ""),
            "abstract": " ".join(clean(c) for c in chunks).strip(),
            "journal": clean(journal.group(1) if journal else ""),
            "year": year.group(1) if year else "",
            "organism": clean(organism.group(1) if organism else ""),
        }
    return out


def parse_fasta(text: str) -> list[tuple[str, str, str]]:
    """返回 [(accession, definition, sequence)]。"""
    records: list[tuple[str, str, str]] = []
    acc = defn = None
    buf: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if acc is not None:
                records.append((acc, defn or "", "".join(buf)))
            head = line[1:]
            parts = head.split(None, 1)
            acc = parts[0]
            defn = parts[1] if len(parts) > 1 else ""
            buf = []
        else:
            buf.append(line)
    if acc is not None:
        records.append((acc, defn or "", "".join(buf)))
    return records


@dataclass
class GenBankRecord:
    """GenBank flat file 里的关键字段。

    为什么用 GenBank 而不是 FASTA + elink：
      elink 传入**多个** uid 时会合并成一个 linkset，不告诉你哪条链接属于哪条序列
      （实测：5 个 id 的全部 pubmed 链接挤在一个 linkset 里）。
      而 GenBank 记录里本身就带 `PUBMED` 字段 —— 一次 efetch 同时拿到序列和 PMID，
      一一对应，没有歧义，也省掉一半请求。
    """

    accession: str = ""
    definition: str = ""
    organism: str = ""
    sequence: str = ""
    pubmeds: list[str] = field(default_factory=list)
    molecule_type: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


_ORIGIN_RE = re.compile(r"^ORIGIN\s*$", re.M)
_LOCUS_RE = re.compile(r"^LOCUS\s+(\S+)\s+(\d+)\s+bp\s+(\S+)?", re.M)
_VERSION_RE = re.compile(r"^VERSION\s+(\S+)", re.M)
_DEFINITION_RE = re.compile(r"^DEFINITION\s+(.*?)(?=^\S|\Z)", re.M | re.S)
_ORGANISM_RE = re.compile(r"^\s+ORGANISM\s+(.+)$", re.M)
_PUBMED_RE = re.compile(r"^\s+PUBMED\s+(\d+)", re.M)


def parse_genbank(text: str) -> list[GenBankRecord]:
    """把 `rettype=gb` 的平文本切成一条条记录。"""
    import re as _re

    chunks = _re.split(r"^//\s*$", text, flags=_re.M)
    out: list[GenBankRecord] = []
    for chunk in chunks:
        if "LOCUS" not in chunk:
            continue
        rec = GenBankRecord()
        m = _LOCUS_RE.search(chunk)
        if m:
            rec.accession = m.group(1)
            rec.molecule_type = (m.group(3) or "").strip()
        mv = _VERSION_RE.search(chunk)
        if mv:
            rec.accession = mv.group(1)
        md = _DEFINITION_RE.search(chunk)
        if md:
            rec.definition = " ".join(md.group(1).split())
        mo = _ORGANISM_RE.search(chunk)
        if mo:
            rec.organism = mo.group(1).strip()
        rec.pubmeds = sorted(set(_PUBMED_RE.findall(chunk)))

        origin = _ORIGIN_RE.search(chunk)
        if origin:
            body = chunk[origin.end():]
            body = body.split("//")[0]
            letters = []
            for line in body.splitlines():
                # ORIGIN 段：行首是编号，后面是 10 个一组的碱基
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.lower().startswith("//"):
                    break
                parts = stripped.split()
                if parts and parts[0].isdigit():
                    letters.extend(parts[1:])
                elif set(stripped.lower()) <= set("acgtunryswkmbdhv "):
                    letters.append(stripped)
            rec.sequence = "".join(letters).upper()
        out.append(rec)
    return out
