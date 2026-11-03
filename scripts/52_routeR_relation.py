"""RNA-RNA 关系判别：双序列输入 + 问题感知投影。

为什么另起一个脚本
------------------
`50_routeC_e2e.py` 是围绕「单序列 -> FSQ 码 -> 查表」建的，`build_batch`
只接受一个 RNA 占位段。关系判别需要两条序列，而且 FSQ 桥已实测不可行
（4 选 1 上 25% = 随机水平），要换成 ChatNT 式的问题感知投影。

所以这里新写一条链路，不破坏 50_ 的候选式训练。

架构
----
    miRNA (22nt) ──ERNIE(冻结)──> [L1,768] ─┐
                                            ├─ 问题感知 resampler ─> [8,2048] 软提示
    mRNA (50nt) ──ERNIE(冻结)──> [L2,768] ─┘         ↑
                                                问题嵌入
    软提示 + "Q: " + 问题文本 + 答案 ──> LLM(SOS-LoRA 可训, 底座冻结)

为什么两条序列分开编码而不是拼接
--------------------------------
miRNA 只有 22 nt，拼进一条序列会被 50 nt 的 mRNA 淹没，模型也分不清哪段是哪个。
分开编码后各出一个软提示段，边界清晰。

用法
----
    # 小数据快速验证
    python scripts/52_routeR_relation.py --data data/processed/mirbench_clash_zh.jsonl \
        --steps 200 --batch 8

    # A100 主力
    python scripts/52_routeR_relation.py --data data/processed/mirbench.jsonl \
        --steps 8000 --batch 32
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: F401  装 shim

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- 投影层

class QuestionAwareResampler(nn.Module):
    """问题感知的 Perceiver resampler（ChatNT 的 English-aware projection）。

    普通 resampler 对同一条序列永远给出同一组 embedding，与问题无关 —— 这是
    信息瓶颈。这里把问题嵌入投影进 query，让抽取的表示随问题变化。

    论文实测（ChatNT, Nat Mach Intell 2025）：问题感知 0.77 vs 普通 0.71 MCC。
    """

    def __init__(self, enc_dim: int, llm_dim: int, n_query: int = 4, n_head: int = 8,
                 feat_dim: int = 0):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_query, enc_dim) * 0.02)
        self.q_proj = nn.Linear(llm_dim, enc_dim)          # 问题嵌入 -> query 空间
        self.attn = nn.MultiheadAttention(enc_dim, n_head, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(1))           # Flamingo 的 gated xattn
        self.out = nn.Linear(enc_dim, llm_dim)
        self.n_query = n_query
        # 逐位可及性通道。零初始化：step 0 与原模型逐位相同，特征平滑接入。
        # 加在 hidden（key/value 侧）而不是 query：可及性是序列位置的属性，
        # 不是"我要找什么"的属性；加到 query 会把两者混在一起。
        self.feat_proj = None
        if feat_dim > 0:
            self.feat_proj = nn.Linear(feat_dim, enc_dim)
            nn.init.zeros_(self.feat_proj.weight)
            nn.init.zeros_(self.feat_proj.bias)

    def forward(self, hidden, mask, q_vec, feat=None):
        # hidden [B,L,enc_dim], mask [B,L] bool(True=有效), q_vec [B,llm_dim]
        if self.feat_proj is not None and feat is not None:
            hidden = hidden + self.feat_proj(feat)          # [B,L,enc_dim]
        q = self.query.unsqueeze(0).expand(hidden.shape[0], -1, -1)
        q = q + self.q_proj(q_vec).unsqueeze(1)            # [B,n_query,enc_dim]
        a, _ = self.attn(q, hidden, hidden, key_padding_mask=~mask)
        return self.out(torch.sigmoid(self.gate) * a)      # [B,n_query,llm_dim]


# ---------------------------------------------------------------- 模型

class RelationModel(nn.Module):
    def __init__(self, base_m, encoder, llm_dim=2048, enc_dim=768,
                 n_query=4, n_head=8, dtype=torch.bfloat16, acc_dim=0,
                 n_region=4):
        super().__init__()
        self.llm = base_m
        self.encoder = encoder
        self.enc_dim = enc_dim
        self.llm_dim = llm_dim
        self.dtype = dtype
        self.acc_dim = acc_dim
        # 两条序列各一个 resampler（参数不共享：miRNA 和 mRNA 的特征尺度不同）
        self.rs_mi = QuestionAwareResampler(enc_dim, llm_dim, n_query, n_head)
        # 可及性只在 mRNA 窗口上算，所以只有 rs_mr 接这个通道；
        # 每个位置只有 1 维（unpaired 概率），所以 feat_dim 恒为 1。
        # rs_mi 用 feat_dim=0，state dict 与旧版完全一致。
        self.rs_mr = QuestionAwareResampler(enc_dim, llm_dim, n_query, n_head,
                                            feat_dim=1 if acc_dim > 0 else 0)
        self.n_query = n_query
        # 区域四分类辅助头。只在训练时用，predict() 不碰它，
        # 导出 GGUF 时也不会出现在模型里。
        # dtype 必须跟随底座：sel 来自 bf16 的 LLM hidden，float32 的 Linear
        # 会直接抛 "mat1 and mat2 must have the same dtype"。
        self.region_head = nn.Linear(llm_dim, n_region).to(dtype)
        self.encoder.eval()

    def train(self, mode: bool = True):
        """冻结的 ERNIE 必须永远 eval。

        ERNIE-RNA 的 config 里 `hidden_dropout=0.1, attention_dropout=0.1`。
        `nn.Module.train()` 会递归把 encoder 也切到 train —— 于是每条训练样本的
        序列表示都被随机 dropout 10%，而评测时不会。这既让「冻结编码器」的假设
        不成立，也让 train/eval 的输入分布不一致。LLM 底座保留 train 模式，
        因为 SOS 的 LoRA dropout(0.05) 要靠它生效。
        """
        super().train(mode)
        self.encoder.eval()
        return self

    @staticmethod
    def _q_embed(emb_layer, batch):
        """问题 token 的 embedding 均值，只在真实 token 上取（排除右 pad）。"""
        e = emb_layer(batch["q_ids"]).float()                  # [B,Lq,llm_dim]
        m = batch["q_mask"].unsqueeze(-1).to(e.dtype)
        return (e * m).sum(1) / m.sum(1).clamp(min=1.0)         # [B,llm_dim]

    @staticmethod
    def lm_parts(llm):
        body = llm.model
        if hasattr(body, "language_model"):     # VLM
            return body.language_model, llm.lm_head
        return body, llm.lm_head

    def _acc_feat(self, batch, h_mr):
        """把逐位可及性铺到编码器输出上：[BOS][50 nt][EOS] 的中间 50 位。

        每个位置只有 1 维（该位的 unpaired 概率），所以 feat 形状是
        [B, L, 1]，不是 [B, L, 50]。50 是序列长度，不是特征宽度。

        `+2` 是 ERNIE 的 BOS/EOS。硬断言是防 tokenizer 变化时静默错位一位 ——
        错位后形状仍然兼容，训练照样收敛，只是特征对错了位置。
        """
        acc = batch.get("acc")
        if acc is None or self.acc_dim <= 0:
            return None
        L, n = h_mr.shape[1], acc.shape[1]
        if L != n + 2:
            raise RuntimeError(
                f"可及性长度不匹配：编码器输出 {L} 位，缓存 {n} 位（应差 2，BOS/EOS）")
        feat = h_mr.new_zeros(h_mr.shape[0], L, 1)
        feat[:, 1:1 + n, 0] = acc
        return feat

    def forward(self, batch):
        """batch: dict(mi_seqs, mr_seqs, q_ids, q_mask, prefix_ids,
                       suffix_ids, pred_pos, target_ids)

        pred_pos 是 [prefix][soft][suffix] 拼接后的**绝对下标**，指向生成提示的
        最后一个 token；该位置的 hidden 预测下一位，即答案首 token。
        """
        # 1) 两条序列各自编码（ERNIE 冻结）
        with torch.no_grad():
            o_mi = self.encoder(batch["mi_seqs"])
            o_mr = self.encoder(batch["mr_seqs"])
        h_mi, m_mi = o_mi.hidden.float(), o_mi.mask.bool()
        h_mr, m_mr = o_mr.hidden.float(), o_mr.mask.bool()

        body, head = self.lm_parts(self.llm)
        emb_layer = body.embed_tokens

        # 2) 问题嵌入 = 问题文本过 LLM embedding 后取平均（不需要额外训练）
        q_emb = self._q_embed(emb_layer, batch)

        # 3) 问题感知投影 -> 两段软提示
        s_mi = self.rs_mi(h_mi, m_mi, q_emb)                   # [B,nq,llm_dim]
        s_mr = self.rs_mr(h_mr, m_mr, q_emb, feat=self._acc_feat(batch, h_mr))
        soft = torch.cat([s_mi, s_mr], dim=1)                  # [B,2nq,llm_dim]

        # 4) 文本部分
        p = emb_layer(batch["prefix_ids"]).float()             # sys+user+"Q: "
        s = emb_layer(batch["suffix_ids"]).float()             # 问题 + assistant 头 + no-think 段
        emb = torch.cat([p, soft.to(p.dtype), s], dim=1)       # [B, T, llm_dim]
        attn = torch.ones(emb.shape[:2], dtype=torch.long, device=emb.device)

        # 5) LLM 前向；pred_pos 是生成提示的最后一位，它预测下一位 = 答案首 token
        hs = body(inputs_embeds=emb.to(self.dtype), attention_mask=attn,
                  use_cache=False).last_hidden_state
        B = hs.shape[0]
        pos = batch["pred_pos"].clamp(min=0)
        if int(pos.max()) >= hs.shape[1]:
            raise RuntimeError(
                f"pred_pos {int(pos.max())} 越过序列长度 {hs.shape[1]}："
                "make_batch 的下标口径和拼接顺序不一致")
        sel = hs[torch.arange(B, device=hs.device), pos]       # [B, llm_dim]
        # 两个头各自对齐自己的 dtype：真实模型里 lm_head 与 region_head 都是
        # bf16，但这里显式转换一次，免得换底座（或跑 fp32 冒烟测试）时炸在
        # "mat1 and mat2 must have the same dtype"。
        logits = head(sel.to(head.weight.dtype)).float()       # [B, V]

        # 6) 两个头：靶向二分类为主，区域四分类为辅助
        is_reg = batch.get("is_region")
        if is_reg is None or not bool(is_reg.any()):
            return {"target": F.cross_entropy(logits, batch["target_ids"])}
        keep = ~is_reg
        out = {}
        if bool(keep.any()):
            out["target"] = F.cross_entropy(
                logits[keep], batch["target_ids"][keep])
        reg_ids = batch["region_ids"][is_reg]
        if bool((reg_ids >= 0).all()):
            out["region"] = F.cross_entropy(
                self.region_head(sel[is_reg].to(self.region_head.weight.dtype)),
                reg_ids)
        if "target" not in out:
            out["target"] = logits.sum() * 0.0      # 整批都是 region 时的占位
        return out

    @torch.no_grad()
    def predict(self, batch):
        """返回第一个答案 token 上 Yes/No 的 logit 差（用于 AUROC）。"""
        body, head = self.lm_parts(self.llm)
        emb_layer = body.embed_tokens
        o_mi = self.encoder(batch["mi_seqs"])
        o_mr = self.encoder(batch["mr_seqs"])
        h_mi, m_mi = o_mi.hidden.float(), o_mi.mask.bool()
        h_mr, m_mr = o_mr.hidden.float(), o_mr.mask.bool()
        q_emb = self._q_embed(emb_layer, batch)
        soft = torch.cat([self.rs_mi(h_mi, m_mi, q_emb),
                          self.rs_mr(h_mr, m_mr, q_emb,
                                     feat=self._acc_feat(batch, h_mr))], dim=1)
        p = emb_layer(batch["prefix_ids"]).float()
        s = emb_layer(batch["suffix_ids"]).float()
        emb = torch.cat([p, soft.to(p.dtype), s], dim=1)
        attn = torch.ones(emb.shape[:2], dtype=torch.long, device=emb.device)
        hs = body(inputs_embeds=emb.to(self.dtype), attention_mask=attn,
                  use_cache=False).last_hidden_state
        # 与训练用同一个 pred_pos（生成提示最后一位），预测下一位 = 答案首 token
        B = hs.shape[0]
        pos = batch["pred_pos"].clamp(min=0)
        sel = hs[torch.arange(B, device=hs.device), pos]
        return head(sel.to(head.weight.dtype)).float()        # [B, V]


# ---------------------------------------------------------------- 数据

class Tok:
    """Qwen 的 chat 头。**不手搓模板。**

    手搓那版有两个错（2026-11-02 核对 tokenizer 实测）：
      * `<|im_end|>` 后面漏了 `\\n`：正确是 `<|im_end|>\\n<|im_start|>`，
        手搓出来是 `<|im_end|><|im_start|>`；
      * no-think 段用的是 Qwen3 的 `" thinking\\n\\n<｜end▁of▁thinking｜>\\n\\n"`
        （14 个 token，含 158819），而 Qwen3.6 的模板实际是
        `"<think>\\n\\n</think>\\n\\n"`（[248068, 271, 248069, 271]）。
        这等于每条训练样本的 assistant 前缀都和预训练分布不一致。

    现在直接用 `apply_chat_template(..., enable_thinking=False)` 渲染，
    用一个数据里不可能出现的哨兵定位软提示的插入点，再切成 left / right。
    tokenizer 换了、模板改了，哨兵找不到就立刻报错，不会静默跑偏。
    """

    SENT = "ZZRNASOFTZZ"

    def __init__(self, tk):
        self.tk = tk
        self.base_vocab = tk.vocab_size
        self.pad_id = tk.pad_token_id if tk.pad_token_id is not None else tk.eos_token_id
        self.eos = tk.eos_token
        self._cache: dict[str, tuple[list[int], list[int]]] = {}
        # 启动就验一次：模板里必须有哨兵，否则后面每个 batch 都会切错
        self.split("ACGU")

    def enc(self, s):
        return self.tk(s, add_special_tokens=False).input_ids

    def split(self, question: str) -> tuple[list[int], list[int]]:
        """返回 (软提示之前, 软提示之后)。right 从问题正文开始。"""
        hit = self._cache.get(question)
        if hit is not None:
            return hit
        text = self.tk.apply_chat_template(
            [{"role": "system", "content": ""},
             {"role": "user", "content": "Q: " + self.SENT + question}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        i = text.find(self.SENT)
        if i < 0:
            raise RuntimeError(
                "chat template 里找不到哨兵：tokenizer 或模板变了，拒绝按旧假设继续")
        out = (self.enc(text[:i]), self.enc(text[i + len(self.SENT):]))
        self._cache[question] = out
        return out


def load_region_rows(path: Path, split: str | None = None, limit: int = 0,
                     seed: int = 7):
    """区域四分类行。与 load_rows 完全分开，保证 run_eval 永远看不到它们。"""
    rng = random.Random(seed)
    rows: list = []
    seen = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") != "mirna_region":
                continue
            if split and r.get("split") != split:
                continue
            seen += 1
            if limit <= 0:
                rows.append(r)
            elif len(rows) < limit:
                rows.append(r)
            else:
                j = rng.randrange(seen)
                if j < limit:
                    rows[j] = r
    return rows


class AccCache:
    """逐位可及性查表（memmap + 窗口字符串索引）。"""

    def __init__(self, f16_path, idx_path):
        import numpy as np
        import torch as _t

        idx = _t.load(idx_path, map_location="cpu", weights_only=False)
        windows = idx["windows"].decode("utf-8").split("\n")
        self.dim = int(idx["dim"])
        self.n = int(idx["n"])
        self._map = {w: i for i, w in enumerate(windows)}
        self._mm = np.memmap(f16_path, dtype=np.float16, mode="r",
                             shape=(self.n, self.dim))
        if len(self._map) != self.n:
            raise RuntimeError(f"可及性索引损坏：{len(self._map)} 个窗口 != {self.n}")

    def get(self, window: str):
        i = self._map.get((window or "").upper())
        return None if i is None else self._mm[i]


def load_rows(path: Path, split: str | None = None, limit: int = 0, seed: int = 7):
    """流式读取 + reservoir sampling。

    **不能全读进内存。** PLAN §27 记过：把 150 万条 dict（约 1.2 GB）留着，
    会和 35B 装载峰值叠加，在 safetensors 的 .item() 处硬崩 0xC0000005
    （没有 Python 栈）。4.1M 条更不可能全留。

    limit=0 表示不限制；>0 时用蓄水池采样均匀抽 limit 条，不偏向文件前部
    （数据按数据集顺序写，取前 N 条会全是 CLASH）。
    """
    rng = random.Random(seed)
    rows: list = []
    seen = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") != "mirna_target":
                continue
            if split and r.get("split") != split:
                continue
            seen += 1
            if limit <= 0:
                rows.append(r)
            elif len(rows) < limit:
                rows.append(r)
            else:
                j = rng.randrange(seen)
                if j < limit:
                    rows[j] = r
    return rows


def shuffled_seq(s: str, rng: random.Random) -> str:
    """逐字符打乱：长度和碱基组成不变，具体序列被破坏。"""
    chars = list(s)
    rng.shuffle(chars)
    return "".join(chars)


REGION_CLASSES = ["3'UTR", "5'UTR", "CDS", "Intron"]
REGION_IDS = {name: i for i, name in enumerate(REGION_CLASSES)}


def make_batch(rows, tok, *, with_answer=True, n_soft=8, acc=None, acc_dim=0):
    """把一批 (miRNA, mRNA 窗口, 问题, 答案) 打包。

    with_answer=True  训练用：suffix 带答案，loss 算在答案首 token 上
    with_answer=False 评测用：suffix 到生成提示为止，末位即答案首 token 的预测位置

    **pred_pos 是拼接后整条序列里的绝对下标**：
        [prefix][soft × n_soft][suffix]
    模型 forward 里 `hs[arange(B), pred_pos]` 直接按这个下标取 hidden，
    所以必须把 prefix 和软提示的长度算进去。只算 suffix 内部下标会让模型
    去读问题中间某个 token 的 hidden —— 能训，但读错了位置。

    **prefix 是逐行的。** 靶向问题和区域问题的 chat 前缀目前恰好相同（都是
    11 个 token），但右段不同（36 vs 41）。这里仍然逐行算 `pred_pos`，并且
    断言整批 prefix 一致——将来模板分叉时会立刻报错，而不是静默把某一行
    的问题塞到错误的位置。
    """
    mi = [r["mirna"] for r in rows]
    mr = [r["mrna_window"] for r in rows]
    qs = [r["question"] for r in rows]
    ans = [r["answer"] for r in rows]

    q_ids = [tok.enc(q) for q in qs]
    q_len = max(len(x) for x in q_ids)
    # 问题 token（右 pad）
    q_t = torch.full((len(rows), q_len), tok.pad_id, dtype=torch.long, device=DEV)
    for i, x in enumerate(q_ids):
        q_t[i, :len(x)] = torch.tensor(x, device=DEV)

    # prefix / suffix 由 tokenizer 自己的 chat template 渲染后切开：
    #   prefix = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nQ: "
    #   suffix = "<问题><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    # 软提示插在两者之间。模板由 Tok.split 负责，这里不再手拼 chat 头。
    lefts, rights = [], []
    for q in qs:
        l, r = tok.split(q)
        lefts.append(l)
        rights.append(r)
    left = lefts[0]
    if any(x != left for x in lefts):
        raise RuntimeError(
            "同一批里的 chat 模板前缀不一致，pred_pos 无法共用基准；"
            "要么拆批，要么让 make_batch 支持逐行 prefix")
    # 问题向量取均值时只算真实 token：右 pad 会把 pad_id 的 embedding 混进来
    out = {"mi_seqs": mi, "mr_seqs": mr, "q_ids": q_t,
           "q_mask": q_t.ne(tok.pad_id)}
    out["prefix_ids"] = torch.tensor([left] * len(rows), device=DEV)

    # pred_pos = 模板给 assistant 生成的最后一位（no-think 块之后的换行），
    # 它在训练和推理里都是同一个位置：该位置的 hidden 预测下一位 = 答案首 token。
    # 不额外加 " A: " 之类的脚手架 —— 模型预训练时答案就是直接跟在生成提示后面的。
    base = len(left) + n_soft
    suffix = list(rights)
    pred_pos = [base + len(r) - 1 for r in rights]
    if with_answer:
        for i, a in enumerate(ans):
            suffix[i] = suffix[i] + tok.enc(a + tok.eos)

    s_len = max(len(x) for x in suffix)
    s_t = torch.full((len(rows), s_len), tok.pad_id, dtype=torch.long, device=DEV)
    for i, s in enumerate(suffix):
        s_t[i, :len(s)] = torch.tensor(s, device=DEV)
    out["suffix_ids"] = s_t
    out["pred_pos"] = torch.tensor(pred_pos, device=DEV)

    # ---- 任务类型：靶向二分类 vs 区域四分类 ----
    is_region = torch.tensor(
        [r.get("kind") == "mirna_region" for r in rows], device=DEV)
    out["is_region"] = is_region
    if is_region.any():
        out["region_ids"] = torch.tensor(
            [REGION_IDS.get(r["answer"], -1) if r.get("kind") == "mirna_region"
             else 0 for r in rows], device=DEV)

    if with_answer:
        # 目标是「是」或「否」的单 token id（启动时已校验两者都是单 token）
        yes_id = tok.enc("是")[0] if tok.enc("是") else tok.enc("Yes")[0]
        no_id = tok.enc("否")[0] if tok.enc("否") else tok.enc("No")[0]
        target = torch.tensor(
            [yes_id if a == "是" or a == "Yes" else no_id for a in ans],
            device=DEV)
        out["target_ids"] = target
        out["yes_id"] = yes_id
        out["no_id"] = no_id
        # 不变量：pred_pos 的下一位必须就是答案首 token。
        # 这是防"下标口径悄悄错位"的最后一道闸——错位时 loss 照样在降，
        # 但模型读的是问题中间某个 token 的 hidden，没人看得出来。
        # 两类行都追加了 answer+eos，所以对它们分别比对各自的答案 token：
        # target 行比是/否，region 行比区域类别。
        rel = out["pred_pos"] - base + 1
        got = s_t[torch.arange(len(rows), device=DEV), rel]
        want = target.clone()
        if "region_ids" in out:
            reg = is_region & (out["region_ids"] >= 0)
            if bool(reg.any()):
                region_first = torch.tensor(
                    [tok.enc(r["answer"])[0] if r.get("kind") == "mirna_region"
                     else 0 for r in rows], device=DEV)
                want = torch.where(reg, region_first, want)
        if not torch.equal(got, want):
            bad = (got != want).nonzero()[0].item()
            raise RuntimeError(
                f"pred_pos 错位：第 {bad} 行 pred_pos 的下一位是 {int(got[bad])}，"
                f"目标是 {int(want[bad])}（prefix {len(left)} + soft {n_soft}）")

    # ---- 可及性特征（只对 mRNA 窗口有意义）----
    if acc is not None and acc_dim > 0:
        A = torch.zeros(len(rows), acc_dim, dtype=torch.float32, device=DEV)
        for i, r in enumerate(rows):
            vec = acc.get(r["mrna_window"])
            if vec is None:
                raise RuntimeError(
                    f"可及性缓存里没有这个窗口（第 {i} 行）：{r['mrna_window'][:20]}…")
            # vec 是 memmap 切片，只读；显式 copy 避免 "non-writable tensor" 警告
            # 和潜在的未定义行为。
            A[i] = torch.tensor(vec, dtype=torch.float32, device=DEV)
        out["acc"] = A
    return out


# ---------------------------------------------------------------- 评测

def auroc(scores, labels):
    """AUROC：正样本分数 > 负样本分数的概率（Mann-Whitney U）。"""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def mcc_at_best_threshold(scores, labels):
    """MCC 和取得它的阈值。

    为什么加 MCC：ChatNT 论文（Nat Mach Intell 2025）报的就是 MCC
    （问题感知 0.77 vs 普通 0.71），我们之前只有 AUROC，没法跟文献直接比。
    另外 AUROC 只看排序，MCC 同时看四个象限，对"模型输出整体偏移"更敏感。

    **阈值必须在验证集上选**，不能用测试集选——否则 MCC 会虚高。
    这里返回 (best_mcc, best_threshold)，调用方在 eval 子集上算完之后
    应当把它当成"该阈值下的 MCC"，而不是无偏的测试指标。
    """
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    n_pos, n_neg = len(pos), len(neg)
    if not pos or not neg:
        return float("nan"), float("nan")
    # 候选阈值取所有分数的中点，避免落在样本分数上造成偏向
    uniq = sorted(set(scores))
    if len(uniq) == 1:
        cands = [uniq[0] - 1e-6, uniq[0] + 1e-6]
    else:
        cands = [(a + b) / 2 for a, b in zip(uniq, uniq[1:])]
        cands = [uniq[0] - 1e-6] + cands + [uniq[-1] + 1e-6]
    best_mcc, best_t = -2.0, float("nan")
    for t in cands:
        tp = sum(1 for s in pos if s > t)
        fp = sum(1 for s in neg if s > t)
        fn = n_pos - tp
        tn = n_neg - fp
        denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        m = 0.0 if denom == 0 else (tp * tn - fp * fn) / denom
        if m > best_mcc:
            best_mcc, best_t = m, t
    return best_mcc, best_t


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/mirbench_clash_zh.jsonl")
    ap.add_argument("--encoder", default="models/ernierna")
    ap.add_argument("--base", default="qwen36-35b")
    ap.add_argument("--qwen36-mode", default="int4")
    ap.add_argument("--lora-preset", default="C")
    ap.add_argument("--adapter", default="sos")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--sos-k", type=int, default=4)
    ap.add_argument("--sos-ortho-coef", type=float, default=1.0,
                    help="SOS 跨专家正交正则的系数（按层平均，与 50_routeC 一致；0=关）")
    ap.add_argument("--n-query", type=int, default=4, help="每条序列的软提示 token 数")
    ap.add_argument("--acc-cache", default="",
                    help="逐位可及性 memmap（_mirbench_acc.f16）")
    ap.add_argument("--acc-idx", default="",
                    help="可及性索引（_mirbench_acc.pt）")
    ap.add_argument("--hard-neg", default="",
                    help="困难负样本 jsonl（mirbench_hardneg.jsonl）")
    ap.add_argument("--region-frac", type=float, default=0.0,
                    help="批次里区域四分类行所占比例；0 表示不启用辅助任务")
    ap.add_argument("--region-coef", type=float, default=0.1,
                    help="区域辅助损失的权重")
    ap.add_argument("--hard-frac", type=float, default=0.25,
                    help="负样本里困难负样本所占比例")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4,
                    help="SOS-LoRA 的 lr；与 50_routeC_e2e.py 验证过的默认值一致")
    ap.add_argument("--lr-proj", type=float, default=1e-3,
                    help="resampler（随机初始化、从零学）的 lr，单独一组")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-n", type=int, default=400)
    ap.add_argument("--grad-ckpt", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-train", type=int, default=0,
                    help="最多加载多少 train 条（0=全部）。必须限制：4.1M 条全读进内存"
                         "会和 35B 装载峰值叠加导致 0xC0000005")
    ap.add_argument("--max-test", type=int, default=20000,
                    help="最多加载多少 test 条")
    ap.add_argument("--tag", default="relation")
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--save-minutes", type=float, default=10.0,
                    help="距离上次保存超过这么多分钟也存一次（防 Colab 掉线丢步）")
    ap.add_argument("--keep-ckpt", type=int, default=3)
    ap.add_argument("--resume", default="", help="从 step_*.pt 恢复权重/优化器/步数")
    ap.add_argument("--init", default="", help="只加载权重，步数从 0 开始")
    ap.add_argument("--perturb", type=int, default=0,
                    help="训练结束后做输入扰动检验（打乱 miRNA / mRNA 看 AUROC 是否掉到 0.5）")
    args = ap.parse_args()
    if args.max_train <= 0:
        print("[warn] --max-train 0：会读入全部 train 行（约 250 万条、2-3 GB 内存）",
              flush=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    RUNS = Path(os.environ.get("RNAQ_RUNS_DIR", str(ROOT / "runs")))
    outdir = RUNS / f"e2e_{args.tag}"
    ckdir = outdir / "ckpt"
    ckdir.mkdir(parents=True, exist_ok=True)

    # ⚠️ 数据**在模型装载之后**再读（PLAN §27：先建模型再读数据，从没崩过；
    #    反过来会在 safetensors 加载峰值上叠加 dict 内存 -> 0xC0000005）。
    # 先记路径，模型建好后再 load_rows。

    # ---- 编码器（冻结）----
    from rnaqwen.models.encoder import RnaEncoder
    encoder = RnaEncoder(str(ROOT / args.encoder), dtype=torch.float32).to(DEV).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    print(f"[enc] {args.encoder} hidden {encoder.hidden_size}", flush=True)

    # MiRBench 给的是 DNA 字母表（T），ERNIE 的词表只有 U。multimolecule 的
    # RnaTokenizer 默认 replace_T_with_U=True，但这是**版本相关的默认值**，
    # 不能假设。实测一次：T 和 U 必须编成同一个 id。
    _t = encoder.tokenizer("ACGT", add_special_tokens=False).input_ids
    _u = encoder.tokenizer("ACGU", add_special_tokens=False).input_ids
    if _t != _u:
        raise RuntimeError(
            "ERNIE tokenizer 没有把 T 映射成 U：MiRBench 序列是 DNA 字母表，"
            f"直接喂会让约四分之一碱基变成 <unk>（ACGT={_t} vs ACGU={_u}）")
    print("[enc] T->U 映射确认", flush=True)

    # ---- 35B 底座 + SOS-LoRA ----
    from rnaqwen.qwen36 import (load_qwen36_text, lora_targets,
                                set_experts_impl, base_spec_from_ckpt_args)
    set_experts_impl("grouped")
    base_m, _text, _lmh, qstats = load_qwen36_text(mode=args.qwen36_mode)
    print(f"[base] {qstats}", flush=True)
    for p in base_m.parameters():
        p.requires_grad_(False)

    pen_fn = None
    if args.adapter == "sos":
        from rnaqwen.models.sos_lora import (replace_linear_with_sos,
                                             sos_lora_penalty)
        pen_fn = sos_lora_penalty
        rep = replace_linear_with_sos(base_m, tuple(lora_targets(args.lora_preset)),
                                      args.lora_r, args.sos_k, args.lora_r * 2,
                                      dropout=0.05, gamma_max=2.5)
        if not rep:
            raise RuntimeError("SOS 一层都没替换上")
        print(f"[sos] 替换 {len(rep)} 层", flush=True)
        if args.sos_ortho_coef:
            # SOS 的语义包含跨专家正交：不罚的话 K 个子空间会塌成一个，
            # 等于退化成 K=1 的普通 LoRA。系数按层平均，与 50_routeC 同义。
            print(f"[loss] CE + {args.sos_ortho_coef} * SOS 正交正则（按层平均）",
                  flush=True)

    from transformers import AutoTokenizer
    base_dir = Path(os.environ.get("RNAQ_QWEN36_DIR",
                                   str(ROOT / "models/Qwen3.6-35B-A3B")))
    tk0 = AutoTokenizer.from_pretrained(str(base_dir))
    tok = Tok(tk0)
    # 答案必须是单 token：CE 只算 pred_pos 的下一位。
    for _a in ("是", "否"):
        _ids = tok.enc(_a)
        if len(_ids) != 1:
            raise RuntimeError(f"答案 {_a!r} 不是单 token：{_ids}，CE 目标会错位")
    print(f"[tok] 是={tok.enc('是')[0]}  否={tok.enc('否')[0]}  pad={tok.pad_id}",
          flush=True)
    # 找 LLM 的 hidden dim
    body = base_m.model
    lm_body = body.language_model if hasattr(body, "language_model") else body
    llm_dim = lm_body.config.hidden_size
    print(f"[cfg] llm_dim {llm_dim}  n_query {args.n_query}", flush=True)

    acc = None
    if args.acc_cache and args.acc_idx:
        acc = AccCache(args.acc_cache, args.acc_idx)
        print(f"[acc] {acc.n:,} windows, dim {acc.dim}", flush=True)
    acc_dim = acc.dim if acc is not None else 0
    # 写回 args，checkpoint 才会记住特征维度。不写的话续训守卫拿不到旧值，
    # GGUF 评测器也会误判"没有可及性通道"。
    args.acc_dim = acc_dim

    model = RelationModel(base_m, encoder, llm_dim=llm_dim,
                          enc_dim=encoder.hidden_size, n_query=args.n_query,
                          acc_dim=acc_dim)
    # **不要**对整体调 model.to(DEV)：base_m 是 bitsandbytes int4 模型，
    # bnb 不支持量化后再 `.to()`，而 nn.Module.to() 会递归到每一个 4bit 参数。
    # encoder 和底座已经在 cuda 上，只有两个 resampler 是新建在 CPU 的。
    model.rs_mi.to(DEV)
    model.rs_mr.to(DEV)
    model.region_head.to(DEV)

    # ---- 模型建好后再读数据（避免内存峰值叠加）----
    tr = load_rows(Path(args.data), "train", args.max_train, args.seed)
    te = load_rows(Path(args.data), "test", args.max_test, args.seed + 1)
    random.Random(args.seed).shuffle(tr)
    # 打乱 test：文件是按数据集顺序写的，不打乱的话 te[:eval_n] 会全是 CLASH，
    # 每次评测只看一个来源。打乱后 eval 子集才覆盖 CLASH/eCLIP/leftout。
    random.Random(args.seed + 1).shuffle(te)
    print(f"[data] train {len(tr):,}  test {len(te):,}", flush=True)
    assert tr and te, "需要 train 和 test 两个 split"

    # ---- 困难负样本 ----
    hard: list = []
    if args.hard_neg:
        hp = Path(args.hard_neg)
        if hp.exists():
            hard = load_rows(hp, None, args.max_train, args.seed + 3)
            print(f"[hard] {len(hard):,} decoys", flush=True)
        else:
            print(f"[hard] 警告：{hp} 不存在，跳过困难负样本", flush=True)

    # ---- 区域辅助任务 ----
    region_pool: list = []
    if args.region_frac > 0:
        region_pool = load_region_rows(Path(args.data), "train",
                                       args.max_train, args.seed + 5)
        print(f"[aux] region pool {len(region_pool):,}  "
              f"region-frac {args.region_frac}  coef {args.region_coef}",
              flush=True)
        if not region_pool:
            raise RuntimeError("--region-frac > 0 但没有找到 mirna_region 行")
    if args.grad_ckpt:
        main_body = model.llm.model
        (main_body.language_model if hasattr(main_body, "language_model")
         else main_body).gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        try:
            model.llm.enable_input_require_grads()
        except Exception:
            pass
        print("[ckpt] gradient checkpointing 开", flush=True)

    # resampler 是随机初始化、从零开始学；SOS-LoRA 是接在预训练权重上的低秩增量。
    # 两者用同一个 lr 没有道理：50_routeC 在 SOS 上验证过的是 2e-4，
    # 而随机投影层用 2e-4 会学得很慢。分两组，各用各的。
    sos_params, proj_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # region_head 也是随机初始化，必须跟投影层同一组；
        # 落进 SOS 组会以 2e-4 学，慢到看起来像"辅助任务太难"。
        if name.startswith(("rs_mi.", "rs_mr.", "region_head.")):
            proj_params.append(p)
        else:
            sos_params.append(p)
    params = sos_params + proj_params
    n_tr = sum(p.numel() for p in params)
    n_proj = sum(p.numel() for p in proj_params)
    n_reg = sum(p.numel() for p in model.region_head.parameters())
    print(f"[train] 可训 {n_tr/1e6:.2f} M（SOS {n_tr/1e6 - n_proj/1e6:.2f} M"
          f" + resampler {n_proj/1e6 - n_reg/1e6:.2f} M"
          f" + region_head {n_reg/1e6:.3f} M）  lr {args.lr} / {args.lr_proj}",
          flush=True)
    opt = torch.optim.AdamW([
        {"params": sos_params, "lr": args.lr},
        {"params": proj_params, "lr": args.lr_proj},
    ], weight_decay=0.01)

    yes_id = tok.enc("是")[0] if tok.enc("是") else tok.enc("Yes")[0]
    no_id = tok.enc("否")[0] if tok.enc("否") else tok.enc("No")[0]

    def run_eval(n, perm_mi=None, perm_mr=None, shuf_mi=False, shuf_mr=False):
        """AUROC。

        perm_*：行置换——用别的行的同字段替换，检验「配对关系」是否被用到。
        shuf_*：**逐行打乱碱基顺序**——保留长度与碱基组成、破坏具体序列，
                检验「具体序列」是否被用到。比行置换更严格：行置换如果碰巧
                抽到同一条 miRNA，扰动就退化了。
        """
        model.eval()
        rows = te[:n]
        rng_s = random.Random(args.seed + 11)
        scores, labels = [], []
        B = 8
        with torch.no_grad():
            for i in range(0, len(rows), B):
                idx = list(range(i, min(i + B, len(rows))))
                chunk = [rows[j] for j in idx]
                if perm_mi is not None:
                    chunk = [dict(r, mirna=rows[perm_mi[j]]["mirna"])
                             for j, r in zip(idx, chunk)]
                if perm_mr is not None:
                    chunk = [dict(r, mrna_window=rows[perm_mr[j]]["mrna_window"])
                             for j, r in zip(idx, chunk)]
                if shuf_mi:
                    chunk = [dict(r, mirna=shuffled_seq(r["mirna"], rng_s))
                             for r in chunk]
                if shuf_mr:
                    chunk = [dict(r, mrna_window=shuffled_seq(r["mrna_window"], rng_s))
                             for r in chunk]
                b = make_batch(chunk, tok, with_answer=False,
                               n_soft=2 * args.n_query,
                               acc=acc, acc_dim=acc_dim)
                logits = model.predict(b)
                # 「是」vs「否」的 logit 差作为正样本分数
                scores += (logits[:, yes_id] - logits[:, no_id]).float().cpu().tolist()
                labels += [r["label"] for r in chunk]
        model.train()
        auc = auroc(scores, labels)
        mcc, thr = mcc_at_best_threshold(scores, labels)
        return auc, mcc, thr, len(rows)

    # ---- checkpoint：保存可训参数 + 优化器，能真正续训 ----
    def trainable_state():
        return {n: p.detach().to("cpu")
                for n, p in model.named_parameters() if p.requires_grad}

    def save_ckpt(path, step, auc, with_optim=True, mcc=None):
        # auroc 保持原字段名（和 rel 时代兼容），MCC 单独存一份。
        # 之前把 MCC 塞进 auroc 字段，读的人会以为那是 AUROC。
        blob = {"step": step, "args": vars(args), "auroc": auc,
                "weights": trainable_state()}
        if mcc is not None:
            blob["mcc"] = mcc
        if with_optim:
            blob["optim"] = opt.state_dict()
        # 先写 .tmp 再原子改名：notebook 的同步线程每 60s 扫 step_*.pt 上传，
        # 直接写目标文件会让它可能传走一个半截的 200 MB zip。
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(blob, tmp)
        os.replace(tmp, path)

    def prune_steps():
        fs = sorted(ckdir.glob("step_*.pt"))
        for f in fs[:-args.keep_ckpt]:
            f.unlink()

    start_step, best = 0, -1.0
    if args.resume or args.init:
        src = args.resume or args.init
        ck = torch.load(src, map_location="cpu", weights_only=False)
        if args.resume:
            # 不同任务的 checkpoint 结构不同（旧候选式的 step_*.pt 连权重都没有），
            # 接上去只会得到一个悄悄错位的模型。宁可报错让人重跑。
            ck_tag = (ck.get("args") or {}).get("tag")
            if ck_tag != args.tag:
                raise SystemExit(
                    f"{src} 的 tag 是 {ck_tag!r}，本次是 {args.tag!r}；"
                    "不同任务的 checkpoint 不能接")
            if not ck.get("weights"):
                raise SystemExit(f"{src} 没有 weights 字段，是旧格式的占位 checkpoint")
            # 结构相关超参必须显式比对。只比张量数量时，n_query 4->16 数量不变，
            # 会以 copy_ 的形状错误这种难懂的方式失败。
            old_args = ck.get("args") or {}
            for key, want in (("n_query", args.n_query),
                              ("acc_dim", acc_dim)):
                got = old_args.get(key)
                if got is not None and got != want:
                    raise SystemExit(
                        f"{src} 的 {key}={got}，本次是 {want}；"
                        "结构变了，不能续训（换 --tag 或从头训）")
        own = dict(model.named_parameters())
        loaded = 0
        shape_bad = []
        for k, v in ck.get("weights", {}).items():
            if k in own:
                if tuple(v.shape) != tuple(own[k].shape):
                    shape_bad.append((k, tuple(v.shape), tuple(own[k].shape)))
                    continue
                own[k].data.copy_(v.to(own[k].device, own[k].dtype))
                loaded += 1
        if shape_bad:
            k, a, b = shape_bad[0]
            raise SystemExit(
                f"{src} 的 {k} 形状 {a} 与模型 {b} 不一致（共 {len(shape_bad)} 处）")
        expect = sum(1 for _, p in model.named_parameters() if p.requires_grad)
        if loaded != expect:
            # 张量数对不上 = 结构不同（n_query / 替换层数 / preset 变了）。
            # 这时有一部分参数还是随机初始化的，接着训只会得到一个错位模型。
            raise SystemExit(
                f"{src} 只匹配到 {loaded}/{expect} 个可训张量，模型结构不一致")
        print(f"[load] {src}  {loaded} 个张量", flush=True)
        if args.resume:
            if "optim" in ck:
                opt.load_state_dict(ck["optim"])
                for st in opt.state.values():        # load_state_dict 留在 CPU
                    for k, v in st.items():
                        if isinstance(v, torch.Tensor):
                            st[k] = v.to(DEV)
            start_step = int(ck.get("step", 0))
            best = float(ck.get("auroc", -1.0))
            print(f"[resume] 从 step {start_step} 继续，best AUROC {best:.4f}", flush=True)

    # ---- 批次组成 ----
    # 类别先验必须保持 1:1，否则 yes/no 的 logit 偏置会漂移，AUROC 的变化
    # 就分不清是"学得更好了"还是"阈值动了"。
    pos_pool = [r for r in tr if r.get("label") == 1]
    easy_pool = [r for r in tr if r.get("label") == 0]
    hard_pool = list(hard)
    if hard_pool:
        hard_set = {(r["mirna"], r["mrna_window"]) for r in hard_pool}
        easy_pool = [r for r in easy_pool
                     if (r["mirna"], r["mrna_window"]) not in hard_set]
    print(f"[mix] pos {len(pos_pool):,}  easy {len(easy_pool):,}  "
          f"hard {len(hard_pool):,}  region {len(region_pool):,}", flush=True)

    n_region = int(round(args.batch * args.region_frac))
    n_target = args.batch - n_region
    n_pos = n_target // 2
    n_neg = n_target - n_pos
    n_hard = int(round(n_neg * args.hard_frac)) if hard_pool else 0
    n_easy = n_neg - n_hard
    if args.hard_neg and not hard_pool:
        n_easy, n_hard = n_neg, 0
    print(f"[mix] 每批 {args.batch} = pos {n_pos} + easy {n_easy} + hard {n_hard}"
          f" + region {n_region}", flush=True)
    if n_hard and not hard_pool:
        raise RuntimeError("请求了困难负样本但没有加载到任何 decoy")

    def next_batch(step: int):
        off = step * args.batch
        chunk = [pos_pool[(off + j) % len(pos_pool)] for j in range(n_pos)]
        chunk += [easy_pool[(off + j) % len(easy_pool)] for j in range(n_easy)]
        if n_hard:
            chunk += [hard_pool[(off + j) % len(hard_pool)] for j in range(n_hard)]
        if n_region:
            chunk += [region_pool[(off + j) % len(region_pool)]
                      for j in range(n_region)]
        random.Random(args.seed + step).shuffle(chunk)
        return chunk

    print(f"\n=== 训练 {start_step + 1} .. {args.steps} 步 ===", flush=True)
    model.train()
    t0 = time.time()
    last_log = time.time()
    last_save = time.time()
    last_auc, last_mcc = None, None
    for step in range(start_step + 1, args.steps + 1):
        chunk = next_batch(step)
        b = make_batch(chunk, tok, n_soft=2 * args.n_query,
                       acc=acc, acc_dim=acc_dim)
        out = model.forward(b)
        loss = out["target"]
        if "region" in out:
            # region_head 在 bf16 下算出来的 CE 仍是 fp32（cross_entropy 内部
            # 会升精度），但显式转一次，避免和 target 相加时出现精度混杂。
            loss = loss + args.region_coef * out["region"].float()
        l_sos = (pen_fn(model.llm) * args.sos_ortho_coef
                 if pen_fn is not None and args.sos_ortho_coef else None)
        total = loss if l_sos is None else loss + l_sos
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % max(1, args.log_every) == 0:
            now = time.time()
            sos_txt = "" if l_sos is None else f"  sos {float(l_sos):.4f}"
            reg_txt = (f"  region {float(out['region']):.4f}"
                       if "region" in out else "")
            print(f"  step {step:>5}  ce {float(out['target']):.4f}{reg_txt}{sos_txt}  "
                  f"{args.log_every / max(now - last_log, 1e-6):.2f} 步/s  "
                  f"段内 {now - t0:.0f}s", flush=True)
            last_log = now
        if step % args.eval_every == 0 or step == args.steps:
            auc, mcc, thr, n = run_eval(args.eval_n)
            last_auc, last_mcc = auc, mcc
            print(f"  [eval@{step}] AUROC {auc:.4f}  MCC {mcc:.4f} (thr {thr:+.3f})"
                  f"  (n={n}, 随机 AUROC 0.5 / MCC 0)", flush=True)
            # 选 best 用 MCC：它同时看四个象限，比只看排序的 AUROC 更接近
            # 「这个模型能不能真的判对」；AUROC 一并记录便于和历史对比。
            if mcc > best:
                best = mcc
                save_ckpt(ckdir / "best.pt", step, auc, with_optim=False, mcc=mcc)
                print(f"  [best] 保存，MCC {mcc:.4f}（AUROC {auc:.4f}）", flush=True)
        due_steps = args.save_every > 0 and step % args.save_every == 0
        due_time = (args.save_minutes > 0
                    and time.time() - last_save >= 60.0 * args.save_minutes)
        if due_steps or due_time:
            save_ckpt(ckdir / f"step_{step:06d}.pt", step,
                      last_auc if last_auc is not None else -1.0,
                      mcc=last_mcc)
            prune_steps()
            last_save = time.time()

    print(f"\n[best] AUROC {best:.4f}  ckpt {ckdir/'best.pt'}", flush=True)

    # ---- 输入扰动检验：模型到底有没有在用两条序列？----
    # 只看 AUROC 会被「问题文本 / 数据集来源」之类的捷径骗过去。
    # 打乱 miRNA 或 mRNA 后 AUROC 应该掉回 ~0.5；如果不掉，说明模型没读那条序列。
    if args.perturb:
        n = min(args.eval_n, len(te))
        rng = random.Random(args.seed + 5)
        perm_mi = list(range(n))
        rng.shuffle(perm_mi)
        perm_mr = list(range(n))
        rng.shuffle(perm_mr)
        base, base_m, _, _ = run_eval(n)
        p_mi, p_mi_m, _, _ = run_eval(n, perm_mi=perm_mi)
        p_mr, p_mr_m, _, _ = run_eval(n, perm_mr=perm_mr)
        s_mi, s_mi_m, _, _ = run_eval(n, shuf_mi=True)
        s_mr, s_mr_m, _, _ = run_eval(n, shuf_mr=True)
        print("\n[perturb] 输入扰动检验（AUROC 应回 ~0.5，MCC 应回 ~0）", flush=True)
        for tag, a, m in (("正常", base, base_m),
                          ("行置换 miRNA", p_mi, p_mi_m),
                          ("行置换 mRNA", p_mr, p_mr_m),
                          ("打乱 miRNA", s_mi, s_mi_m),
                          ("打乱 mRNA", s_mr, s_mr_m)):
            print(f"  {tag:<14} AUROC {a:.4f}  MCC {m:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
