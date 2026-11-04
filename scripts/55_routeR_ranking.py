"""Listwise ranking fine-tune for the RNA relation model.

For each group: one miRNA, K windows for that same miRNA, exactly one true
binding site.  The model scores every candidate; the loss is a softmax over
the K scores.  Chance is 1/K, and the negatives share the miRNA, so the model
cannot win by learning "this miRNA's seed".

Reuses the model/tokenizer/accessibility code from 52_routeR_relation.py.

Usage
-----
    python scripts/55_routeR_ranking.py \
        --groups-train data/processed/mirbench_rank_train.jsonl \
        --groups-test  data/processed/mirbench_rank_test.jsonl \
        --init runs/e2e_rel2/ckpt/step_002200.pt --tag rank1 --steps 1500
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import rnaqwen  # noqa: F401  shim (must precede transformers)

import numpy as np
import torch
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _load_52():
    spec = importlib.util.spec_from_file_location(
        "r52", ROOT / "scripts/52_routeR_relation.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


r52 = _load_52()
TARGET_Q = ("这条 miRNA 与这段 mRNA 窗口之间是否存在经实验验证的靶向关系？"
            "只回答「是」或「否」。")


def load_groups(path: Path) -> list[dict]:
    groups = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                groups.append(json.loads(line))
    return groups


def fast_auc(scores, labels) -> float:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups-train", required=True)
    ap.add_argument("--groups-test", required=True)
    ap.add_argument("--encoder", default="models/ernierna")
    ap.add_argument("--base", default="qwen36-35b")
    ap.add_argument("--qwen36-mode", default="int4")
    ap.add_argument("--lora-preset", default="C")
    ap.add_argument("--adapter", default="sos")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--sos-k", type=int, default=4)
    ap.add_argument("--sos-ortho-coef", type=float, default=1.0)
    ap.add_argument("--n-query", type=int, default=16)
    ap.add_argument("--acc-cache", default="")
    ap.add_argument("--acc-idx", default="")
    ap.add_argument("--batch-groups", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-proj", type=float, default=5e-4)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-groups", type=int, default=1000)
    ap.add_argument("--grad-ckpt", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="rank1")
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--save-minutes", type=float, default=10.0)
    ap.add_argument("--keep-ckpt", type=int, default=3)
    ap.add_argument("--init", default="", help="只加载权重（从 rel2 checkpoint 微调）")
    ap.add_argument("--resume", default="", help="恢复权重/优化器/步数")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    RUNS = Path(os.environ.get("RNAQ_RUNS_DIR", str(ROOT / "runs")))
    outdir = RUNS / f"e2e_{args.tag}"
    ckdir = outdir / "ckpt"
    ckdir.mkdir(parents=True, exist_ok=True)

    # ---- encoder ----
    from rnaqwen.models.encoder import RnaEncoder
    encoder = RnaEncoder(str(ROOT / args.encoder), dtype=torch.float32).to(DEV).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    _t = encoder.tokenizer("ACGT", add_special_tokens=False).input_ids
    _u = encoder.tokenizer("ACGU", add_special_tokens=False).input_ids
    if _t != _u:
        raise RuntimeError("ERNIE tokenizer 没有把 T 映射成 U")
    print(f"[enc] hidden {encoder.hidden_size}  T->U 确认", flush=True)

    # ---- base + SOS ----
    from rnaqwen.qwen36 import load_qwen36_text, lora_targets, set_experts_impl
    set_experts_impl("grouped")
    base_m, _text, _lmh, qstats = load_qwen36_text(mode=args.qwen36_mode)
    print(f"[base] {qstats}", flush=True)
    for p in base_m.parameters():
        p.requires_grad_(False)
    pen_fn = None
    if args.adapter == "sos":
        from rnaqwen.models.sos_lora import replace_linear_with_sos, sos_lora_penalty
        pen_fn = sos_lora_penalty
        rep = replace_linear_with_sos(base_m, tuple(lora_targets(args.lora_preset)),
                                      args.lora_r, args.sos_k, args.lora_r * 2,
                                      dropout=0.05, gamma_max=2.5)
        if not rep:
            raise RuntimeError("SOS 一层都没替换上")
        print(f"[sos] 替换 {len(rep)} 层", flush=True)

    from transformers import AutoTokenizer
    base_dir = Path(os.environ.get("RNAQ_QWEN36_DIR",
                                   str(ROOT / "models/Qwen3.6-35B-A3B")))
    tok = r52.Tok(AutoTokenizer.from_pretrained(str(base_dir)))
    yes_id = tok.enc("是")[0]
    no_id = tok.enc("否")[0]
    body = base_m.model
    lm_body = body.language_model if hasattr(body, "language_model") else body
    llm_dim = lm_body.config.hidden_size
    print(f"[cfg] llm_dim {llm_dim} n_query {args.n_query} "
          f"是={yes_id} 否={no_id}", flush=True)

    acc = None
    if args.acc_cache and args.acc_idx:
        acc = r52.AccCache(args.acc_cache, args.acc_idx)
        print(f"[acc] {acc.n:,} windows dim {acc.dim}", flush=True)
    acc_dim = acc.dim if acc is not None else 0
    args.acc_dim = acc_dim

    model = r52.RelationModel(base_m, encoder, llm_dim=llm_dim,
                              enc_dim=encoder.hidden_size, n_query=args.n_query,
                              acc_dim=acc_dim)
    model.rs_mi.to(DEV); model.rs_mr.to(DEV); model.region_head.to(DEV)
    # 排序任务不用区域头；冻结它，checkpoint 里就不会带一堆没训过的参数。
    model.region_head.requires_grad_(False)

    # ---- 初始化 / 续训 ----
    start_step, best = 0, -1.0
    if args.resume or args.init:
        src = args.resume or args.init
        ck = torch.load(src, map_location="cpu", weights_only=False)
        if args.resume:
            if (ck.get("args") or {}).get("tag") != args.tag:
                raise SystemExit(f"{src} tag 不符: {(ck.get('args') or {}).get('tag')}")
        own = dict(model.named_parameters())
        loaded = 0
        for k, v in (ck.get("weights") or {}).items():
            if k in own and own[k].requires_grad:
                if tuple(v.shape) != tuple(own[k].shape):
                    raise SystemExit(f"{k} 形状不一致: {tuple(v.shape)} vs {tuple(own[k].shape)}")
                own[k].data.copy_(v.to(own[k].device, own[k].dtype))
                loaded += 1
        expect = sum(1 for _, p in model.named_parameters() if p.requires_grad)
        print(f"[init] {Path(src).name} loaded {loaded}/{expect}", flush=True)
        if args.resume:
            start_step = int(ck.get("step", 0))
            best = float(ck.get("top1", -1.0))
            print(f"[resume] step {start_step} best top1 {best:.4f}", flush=True)

    # ---- data ----
    tr = load_groups(Path(args.groups_train))
    te = load_groups(Path(args.groups_test))
    k_values = {len(g["windows"]) for g in tr + te}
    if len(k_values) != 1:
        raise SystemExit(f"group 的候选数不一致: {sorted(k_values)[:5]}")
    K = k_values.pop()
    print(f"[data] groups train {len(tr):,} test {len(te):,} K={K}", flush=True)
    perm = list(range(len(tr)))
    random.Random(args.seed).shuffle(perm)

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

    sos_params, proj_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(("rs_mi.", "rs_mr.")):
            proj_params.append(p)
        else:
            sos_params.append(p)
    params = sos_params + proj_params
    n_tr = sum(p.numel() for p in params)
    print(f"[train] 可训 {n_tr/1e6:.2f} M  lr {args.lr} / {args.lr_proj}", flush=True)
    opt = torch.optim.AdamW([
        {"params": sos_params, "lr": args.lr},
        {"params": proj_params, "lr": args.lr_proj},
    ], weight_decay=0.01)

    def batch_rows(group_ids):
        rows = []
        for gi in group_ids:
            g = tr[gi] if isinstance(gi, int) and 0 <= gi < len(tr) else gi
            for j, w in enumerate(g["windows"]):
                rows.append({"mirna": g["mirna"], "mrna_window": w,
                             "question": TARGET_Q,
                             "answer": "是" if j == 0 else "否",
                             "label": 1 if j == 0 else 0,
                             "kind": "mirna_target"})
        return rows

    def scores_for(rows):
        b = r52.make_batch(rows, tok, with_answer=False,
                           n_soft=2 * args.n_query, acc=acc, acc_dim=acc_dim)
        logits, _ = model.forward_logits(b)
        return (logits[:, yes_id] - logits[:, no_id]).float()

    @torch.no_grad()
    def run_eval(n_groups):
        model.eval()
        subset = te[:n_groups]
        top1 = mrr = 0.0
        all_scores, all_labels = [], []
        B = args.batch_groups
        for i in range(0, len(subset), B):
            chunk = subset[i:i + B]
            sc = scores_for(batch_rows(chunk)).view(len(chunk), K).cpu()
            labels = torch.zeros(len(chunk), dtype=torch.long)
            top1 += int((sc.argmax(1) == labels).sum())
            ranks = (sc > sc[:, :1]).sum(1) + 1     # 1 = 命中第一名
            mrr += float((1.0 / ranks.float()).sum())
            all_scores += sc.reshape(-1).tolist()
            all_labels += ([1] + [0] * (K - 1)) * len(chunk)
        model.train()
        n = len(subset)
        return top1 / n, mrr / n, fast_auc(all_scores, all_labels), n

    def trainable_state():
        return {n: p.detach().to("cpu")
                for n, p in model.named_parameters() if p.requires_grad}

    def save_ckpt(path, step, top1, with_optim=True):
        blob = {"step": step, "args": vars(args), "top1": top1,
                "weights": trainable_state()}
        if with_optim:
            blob["optim"] = opt.state_dict()
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(blob, tmp)
        os.replace(tmp, path)

    def prune_steps():
        fs = sorted(ckdir.glob("step_*.pt"))
        for f in fs[:-args.keep_ckpt]:
            f.unlink()

    print(f"\n=== 排序训练 {start_step + 1} .. {args.steps} 步 "
          f"(G={args.batch_groups}, K={K}) ===", flush=True)
    model.train()
    t0 = time.time(); last_log = time.time(); last_save = time.time()
    for step in range(start_step + 1, args.steps + 1):
        off = (step * args.batch_groups) % len(perm)
        idx = [perm[(off + j) % len(perm)] for j in range(args.batch_groups)]
        sc = scores_for(batch_rows(idx)).view(len(idx), K)
        target = torch.zeros(len(idx), dtype=torch.long, device=DEV)
        loss = F.cross_entropy(sc, target)
        l_sos = (pen_fn(model.llm) * args.sos_ortho_coef
                 if pen_fn is not None and args.sos_ortho_coef else None)
        total = loss if l_sos is None else loss + l_sos
        opt.zero_grad(); total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % max(1, args.log_every) == 0:
            now = time.time()
            acc1 = float((sc.argmax(1) == target).float().mean())
            sos_txt = "" if l_sos is None else f"  sos {float(l_sos):.4f}"
            print(f"  step {step:>5}  listwise {float(loss):.4f}  "
                  f"top1 {acc1:.3f}{sos_txt}  "
                  f"{args.log_every / max(now - last_log, 1e-6):.2f} 步/s  "
                  f"段内 {now - t0:.0f}s", flush=True)
            last_log = now
        if step % args.eval_every == 0 or step == args.steps:
            top1, mrr, auc, n = run_eval(args.eval_groups)
            print(f"  [eval@{step}] top1 {top1:.4f}  MRR {mrr:.4f}  "
                  f"AUROC {auc:.4f}  (n={n} groups, chance {1/K:.2f})", flush=True)
            if top1 > best:
                best = top1
                save_ckpt(ckdir / "best.pt", step, top1, with_optim=False)
                print(f"  [best] 保存，top1 {top1:.4f}", flush=True)
        due_steps = args.save_every > 0 and step % args.save_every == 0
        due_time = (args.save_minutes > 0
                    and time.time() - last_save >= 60.0 * args.save_minutes)
        if due_steps or due_time:
            save_ckpt(ckdir / f"step_{step:06d}.pt", step, best)
            prune_steps()
            last_save = time.time()
    print(f"\n[best] top1 {best:.4f}  ckpt {ckdir/'best.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
