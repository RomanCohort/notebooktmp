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
    软提示 + "Q: " + 问题文本 + "A: " + 答案 ──> LLM(SOS-LoRA 可训, 底座冻结)

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

    def __init__(self, enc_dim: int, llm_dim: int, n_query: int = 4, n_head: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_query, enc_dim) * 0.02)
        self.q_proj = nn.Linear(llm_dim, enc_dim)          # 问题嵌入 -> query 空间
        self.attn = nn.MultiheadAttention(enc_dim, n_head, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(1))           # Flamingo 的 gated xattn
        self.out = nn.Linear(enc_dim, llm_dim)
        self.n_query = n_query

    def forward(self, hidden, mask, q_vec):
        # hidden [B,L,enc_dim], mask [B,L] bool(True=有效), q_vec [B,llm_dim]
        q = self.query.unsqueeze(0).expand(hidden.shape[0], -1, -1)
        q = q + self.q_proj(q_vec).unsqueeze(1)            # [B,n_query,enc_dim]
        a, _ = self.attn(q, hidden, hidden, key_padding_mask=~mask)
        return self.out(torch.sigmoid(self.gate) * a)      # [B,n_query,llm_dim]


# ---------------------------------------------------------------- 模型

class RelationModel(nn.Module):
    def __init__(self, base_m, encoder, llm_dim=2048, enc_dim=768,
                 n_query=4, n_head=8, dtype=torch.bfloat16):
        super().__init__()
        self.llm = base_m
        self.encoder = encoder
        self.enc_dim = enc_dim
        self.llm_dim = llm_dim
        self.dtype = dtype
        # 两条序列各一个 resampler（参数不共享：miRNA 和 mRNA 的特征尺度不同）
        self.rs_mi = QuestionAwareResampler(enc_dim, llm_dim, n_query, n_head)
        self.rs_mr = QuestionAwareResampler(enc_dim, llm_dim, n_query, n_head)
        self.n_query = n_query

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

    def forward(self, batch):
        """batch: dict(mi_ids, mi_mask, mr_ids, mr_mask, q_ids, prefix_ids,
                       suffix_ids, label_pos, labels)"""
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
        s_mr = self.rs_mr(h_mr, m_mr, q_emb)
        soft = torch.cat([s_mi, s_mr], dim=1)                  # [B,2nq,llm_dim]

        # 4) 文本部分
        p = emb_layer(batch["prefix_ids"]).float()             # sys+user+"Q: "
        s = emb_layer(batch["suffix_ids"]).float()             # 问题+assistant+no_think+"A: "
        emb = torch.cat([p, soft.to(p.dtype), s], dim=1)       # [B, T, llm_dim]
        attn = torch.ones(emb.shape[:2], dtype=torch.long, device=emb.device)

        # 5) LLM 前向；pred_pos 是 " A: " 的最后一位，它预测下一位 = 答案首 token
        hs = body(inputs_embeds=emb.to(self.dtype), attention_mask=attn,
                  use_cache=False).last_hidden_state
        B = hs.shape[0]
        pos = batch["pred_pos"].clamp(min=0)
        sel = hs[torch.arange(B, device=hs.device), pos]       # [B, llm_dim]
        logits = head(sel).float()                             # [B, V]
        return F.cross_entropy(logits, batch["target_ids"])

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
                          self.rs_mr(h_mr, m_mr, q_emb)], dim=1)
        p = emb_layer(batch["prefix_ids"]).float()
        s = emb_layer(batch["suffix_ids"]).float()
        emb = torch.cat([p, soft.to(p.dtype), s], dim=1)
        attn = torch.ones(emb.shape[:2], dtype=torch.long, device=emb.device)
        hs = body(inputs_embeds=emb.to(self.dtype), attention_mask=attn,
                  use_cache=False).last_hidden_state
        # 与训练用同一个 pred_pos（" A: " 的最后一位），预测下一位 = 答案首 token
        B = hs.shape[0]
        pos = batch["pred_pos"].clamp(min=0)
        sel = hs[torch.arange(B, device=hs.device), pos]
        return head(sel).float()                              # [B, V]


# ---------------------------------------------------------------- 数据

class Tok:
    def __init__(self, tk):
        self.tk = tk
        self.base_vocab = tk.vocab_size
        self.pad_id = tk.pad_token_id if tk.pad_token_id is not None else tk.eos_token_id
        self.eos = tk.eos_token
        self.im_start = tk("<|im_start|>", add_special_tokens=False).input_ids
        self.im_end = tk("<|im_end|>", add_special_tokens=False).input_ids
        self.sys_ids = self.im_start + tk("system").input_ids + tk("\n").input_ids
        self.user_ids = (self.im_end + self.im_start + tk("user").input_ids
                         + tk("\n").input_ids)
        self.assistant_ids = (self.im_end + self.im_start + tk("assistant").input_ids
                              + tk("\n").input_ids)
        self.no_think = tk(" thinking\n\n<｜end▁of▁thinking｜>\n\n",
                           add_special_tokens=False).input_ids

    def enc(self, s):
        return self.tk(s, add_special_tokens=False).input_ids


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


def make_batch(rows, tok, *, with_answer=True):
    """把一批 (miRNA, mRNA 窗口, 问题, 答案) 打包。

    with_answer=True  训练用：suffix 带答案，loss 算在答案首 token 上
    with_answer=False 评测用：suffix 到 " A: " 为止，末位即答案首 token 的预测位置
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

    # prefix = sys+user+"Q: "；suffix = 问题 + assistant + no_think + " A: "
    head = (tok.sys_ids + tok.user_ids + tok.enc("Q: "))
    # 问题向量取均值时只算真实 token：右 pad 会把 pad_id 的 embedding 混进来
    out = {"mi_seqs": mi, "mr_seqs": mr, "q_ids": q_t,
           "q_mask": q_t.ne(tok.pad_id)}
    pref = [head] * len(rows)
    p_len = len(head)
    out["prefix_ids"] = torch.tensor([pref[0]] * len(rows), device=DEV)

    # suffix = 问题 + assistant + no_think + " A: " [+ 答案 + eos]
    # **pred_pos 统一记录「答案前一位」**（即 " A: " 的最后一个 token）。
    # 训练和推理都用同一个位置：模型在该位置的 hidden state 预测下一位，
    # 也就是答案首 token。之前训练取 suffix_lens-2（答案本身）是错位的：
    # 那个位置的 hidden 预测的是 eos，不是答案。
    suffix, pred_pos = [], []
    for q in qs:
        pre = tok.enc(q) + tok.assistant_ids + tok.no_think + tok.enc(" A: ")
        pred_pos.append(len(pre) - 1)
        suffix.append(pre)
    if with_answer:
        for i, a in enumerate(ans):
            suffix[i] = suffix[i] + tok.enc(a + tok.eos)

    s_len = max(len(x) for x in suffix)
    s_t = torch.full((len(rows), s_len), tok.pad_id, dtype=torch.long, device=DEV)
    l_t = torch.full((len(rows), s_len), -100, dtype=torch.long, device=DEV)
    for i, s in enumerate(suffix):
        s_t[i, :len(s)] = torch.tensor(s, device=DEV)
        if with_answer:
            a_ids = tok.enc(" A: " + ans[i] + tok.eos)
            n_head = len(s) - len(a_ids)
            l_t[i, n_head:len(s)] = torch.tensor(s[n_head:], device=DEV)
    out["suffix_ids"] = s_t
    out["labels"] = l_t
    out["suffix_lens"] = torch.tensor([len(x) for x in suffix], device=DEV)
    out["pred_pos"] = torch.tensor(pred_pos, device=DEV)
    if with_answer:
        # 目标是「是」或「否」的单 token id（已确认两者都是单 token）
        yes_id = tok.enc("是")[0] if tok.enc("是") else tok.enc("Yes")[0]
        no_id = tok.enc("否")[0] if tok.enc("否") else tok.enc("No")[0]
        out["target_ids"] = torch.tensor(
            [yes_id if a == "是" or a == "Yes" else no_id for a in ans],
            device=DEV)
        out["yes_id"] = yes_id
        out["no_id"] = no_id
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
    ap.add_argument("--n-query", type=int, default=4, help="每条序列的软提示 token 数")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
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

    # ---- 35B 底座 + SOS-LoRA ----
    from rnaqwen.qwen36 import (load_qwen36_text, lora_targets,
                                set_experts_impl, base_spec_from_ckpt_args)
    set_experts_impl("grouped")
    base_m, _text, _lmh, qstats = load_qwen36_text(mode=args.qwen36_mode)
    print(f"[base] {qstats}", flush=True)
    for p in base_m.parameters():
        p.requires_grad_(False)

    if args.adapter == "sos":
        from rnaqwen.models.sos_lora import replace_linear_with_sos
        rep = replace_linear_with_sos(base_m, tuple(lora_targets(args.lora_preset)),
                                      args.lora_r, args.sos_k, args.lora_r * 2,
                                      dropout=0.05, gamma_max=2.5)
        if not rep:
            raise RuntimeError("SOS 一层都没替换上")
        print(f"[sos] 替换 {len(rep)} 层", flush=True)

    from transformers import AutoTokenizer
    base_dir = Path(os.environ.get("RNAQ_QWEN36_DIR",
                                   str(ROOT / "models/Qwen3.6-35B-A3B")))
    tk0 = AutoTokenizer.from_pretrained(str(base_dir))
    tok = Tok(tk0)
    # 找 LLM 的 hidden dim
    body = base_m.model
    lm_body = body.language_model if hasattr(body, "language_model") else body
    llm_dim = lm_body.config.hidden_size
    print(f"[cfg] llm_dim {llm_dim}  n_query {args.n_query}", flush=True)

    model = RelationModel(base_m, encoder, llm_dim=llm_dim,
                          enc_dim=encoder.hidden_size, n_query=args.n_query).to(DEV)

    # ---- 模型建好后再读数据（避免内存峰值叠加）----
    tr = load_rows(Path(args.data), "train", args.max_train, args.seed)
    te = load_rows(Path(args.data), "test", args.max_test, args.seed + 1)
    random.Random(args.seed).shuffle(tr)
    # 打乱 test：文件是按数据集顺序写的，不打乱的话 te[:eval_n] 会全是 CLASH，
    # 每次评测只看一个来源。打乱后 eval 子集才覆盖 CLASH/eCLIP/leftout。
    random.Random(args.seed + 1).shuffle(te)
    print(f"[data] train {len(tr):,}  test {len(te):,}", flush=True)
    assert tr and te, "需要 train 和 test 两个 split"
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

    params = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in params)
    print(f"[train] 可训 {n_tr/1e6:.2f} M（SOS + 2×resampler）", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    yes_id = tok.enc("是")[0] if tok.enc("是") else tok.enc("Yes")[0]
    no_id = tok.enc("否")[0] if tok.enc("否") else tok.enc("No")[0]

    def run_eval(n, perm_mi=None, perm_mr=None):
        """AUROC。perm_mi/perm_mr 是行置换：用这一批行的索引去取别的行的序列，
        即「换个不配对的 miRNA / mRNA」——输入扰动检验用。"""
        model.eval()
        rows = te[:n]
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
                b = make_batch(chunk, tok, with_answer=False)
                logits = model.predict(b)
                # 「是」vs「否」的 logit 差作为正样本分数
                scores += (logits[:, yes_id] - logits[:, no_id]).float().cpu().tolist()
                labels += [r["label"] for r in chunk]
        model.train()
        return auroc(scores, labels), len(rows)

    # ---- checkpoint：保存可训参数 + 优化器，能真正续训 ----
    def trainable_state():
        return {n: p.detach().to("cpu")
                for n, p in model.named_parameters() if p.requires_grad}

    def save_ckpt(path, step, auc, with_optim=True):
        blob = {"step": step, "args": vars(args), "auroc": auc,
                "weights": trainable_state()}
        if with_optim:
            blob["optim"] = opt.state_dict()
        torch.save(blob, path)

    def prune_steps():
        fs = sorted(ckdir.glob("step_*.pt"))
        for f in fs[:-args.keep_ckpt]:
            f.unlink()

    start_step, best = 0, -1.0
    if args.resume or args.init:
        src = args.resume or args.init
        ck = torch.load(src, map_location="cpu", weights_only=False)
        own = dict(model.named_parameters())
        loaded = 0
        for k, v in ck.get("weights", {}).items():
            if k in own:
                own[k].data.copy_(v.to(own[k].device, own[k].dtype))
                loaded += 1
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

    print(f"\n=== 训练 {start_step + 1} .. {args.steps} 步 ===", flush=True)
    model.train()
    t0 = time.time()
    last_log = time.time()
    last_save = time.time()
    for step in range(start_step + 1, args.steps + 1):
        chunk = [tr[(step * args.batch + j) % len(tr)] for j in range(args.batch)]
        b = make_batch(chunk, tok)
        loss = model.forward(b)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % max(1, args.log_every) == 0:
            now = time.time()
            print(f"  step {step:>5}  loss {float(loss):.4f}  "
                  f"{args.log_every / max(now - last_log, 1e-6):.2f} 步/s  "
                  f"段内 {now - t0:.0f}s", flush=True)
            last_log = now
        if step % args.eval_every == 0 or step == args.steps:
            auc, n = run_eval(args.eval_n)
            print(f"  [eval@{step}] AUROC {auc:.4f}  (n={n}, 随机 0.5)", flush=True)
            if auc > best:
                best = auc
                save_ckpt(ckdir / "best.pt", step, auc, with_optim=False)
                print(f"  [best] 保存，AUROC {auc:.4f}", flush=True)
        due_steps = args.save_every > 0 and step % args.save_every == 0
        due_time = time.time() - last_save >= 60.0 * args.save_minutes
        if due_steps or due_time:
            save_ckpt(ckdir / f"step_{step:06d}.pt", step, best)
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
        base, _ = run_eval(n)
        mi, _ = run_eval(n, perm_mi=perm_mi)
        mr, _ = run_eval(n, perm_mr=perm_mr)
        print("\n[perturb] 输入扰动检验", flush=True)
        print(f"  正常        AUROC {base:.4f}", flush=True)
        print(f"  打乱 miRNA  AUROC {mi:.4f}   (应 ~0.5)", flush=True)
        print(f"  打乱 mRNA   AUROC {mr:.4f}   (应 ~0.5)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
