"""Stage 1 模态对齐：冻结 RNA-FM + 冻结 Qwen3-8B，**只训投影**。

原版设计的关键一条（PLAN §1，也是全计划最重要的防坑设计）：
    **Stage 1 的输入里没有任何文本 prompt。**
    模型只能看到「投影后的 RNA 表征」，然后被要求产出这段 RNA 的描述。
    这从结构上堵死了"绕过编码器、直接抄 prompt 里的文本"这条捷径。

对齐目标（照这个读法实现）：
    inputs_embeds = [ proj(rna_hidden) ; text_embeddings(annotation) ]
    labels        = [ -100 × L_rna        ; annotation_token_ids       ]
    loss          = 只在 annotation token 上算的 causal LM loss

    也就是：模型必须先"消化"RNA 表征，才能把后面的描述说对。

记忆：8B 冻结模型仍要把梯度回传到输入，所以必须开 gradient checkpointing，
      batch=1（原版就是 batch=1）。

E 轨开关（**默认关**，关着就等价于复现轨）
----------------------------------------
`Stage1Config.compress_latent` 是 E 轨（改进轨）RNA token 压缩的唯一入口：

    compress_latent = None   -> 原样：每个 RNA token 都投影（**R 轨 / 复现轨**）
    compress_latent = 64     -> RNA-FM 输出先过 `RnaTokenCompressor` 压到 64 个 latent：
                                RNA-FM → compressor → Linear(640→4096)

为什么：729 nt → 731 个 RNA 软 token，占 Stage 1 序列的 ~2/3；压到 64 个后
序列短 2.5 倍，训练/推理都快 ~2.5 倍。压缩器属于"投影那一侧"，和投影一起训，
编码器与 LLM 依然全冻结。

⚠️ 默认 None 时 `self.compression is None`，`build_batch` **一行都不走**压缩分支，
`inputs/attn/labels` 与加这个开关之前逐位一致（证据：`scripts/_test_compression.py`
第 5 项，拿改动前抓的 golden 快照 `reports/_golden_none_path.pt` 做 `torch.equal` 比对）。
**E 轨与 R 轨的分数不能混在一起报。**
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import torch
import torch.nn as nn

from rnaqwen.models.compression import RnaTokenCompressor
from rnaqwen.models.encoder import RnaFmEncoder
from rnaqwen.models.projection import LinearProjection


@dataclass
class Stage1Config:
    llm_path: str = "models/Qwen3-8B"
    # 默认走 paths.default_encoder()（E 轨 ERNIE-RNA）。R 轨复现请显式给 models/rnafm。
    encoder_path: str = field(default_factory=lambda: __import__("rnaqwen.paths", fromlist=["x"]).default_encoder())
    max_nt: int = 1024
    max_text_tokens: int = 512

    epochs: int = 10
    batch_size: int = 1
    grad_accum: int = 8

    # 照抄原版（PLAN §1）
    lr: float = 1e-4
    lr_final: float = 8e-5
    warmup_steps: int = 2048
    warmup_lr: float = 1e-6
    weight_decay: float = 0.05
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    scheduler: str = "cosine"
    max_grad_norm: float = 1.0

    dtype: str = "bfloat16"
    grad_checkpointing: bool = True
    seed: int = 0
    log_every: int = 10
    save_every: int = 500
    out_dir: str = "checkpoints/stage1"

    # ---- 定步数预算 ----
    # 0 = 不限制（跑满 epochs）。非 0 就训到这么多步停下。
    # 为什么需要它：做「压缩 vs 不压缩」这类**配对对比**时，两个 arm 必须停在同一步数，
    # 而且中途停下来的时间点也要一致，否则比的是「谁跑得久」而不是「谁更好」。
    max_steps: int = 0

    # ---- 收敛监控（没有这些，你只有一条 train loss，分不清「在学 / 在背 / 塌了」）----
    eval_every: int = 250            # 每多少步评估一次（val loss + 置零探针 + 塌缩指纹）
    eval_samples: int = 48           # 每次评估用多少条（控制时间；固定子集保证可比）
    # 塌缩判据用**相对衰减**，不用绝对值。
    # 实测教训：E 轨 run1 的 Δloss 从峰值 0.273（step 1600）慢慢衰减到 0.047（step 13800），
    # 绝对值判据（<0.02）**一次都没触发**，但编码器其实已经被抛弃了（只剩峰值的 17%）。
    collapse_rel_drop: float = 0.75   # Δloss 跌到「滑动峰值」的 (1-0.75)=25% 以下 → 判塌缩
    collapse_min_peak: float = 0.05   # 峰值本身太小就不算「曾经用过编码器」，不触发
    val_patience: int = 12            # val_loss 连续多少次评估没改善 → 停（平台期 = 在过拟合）
    collapse_patience: int = 2
    early_stop_on_collapse: bool = True

    # ---- E 轨（改进轨）开关：默认关，关着 = 复现轨逐位行为 ----
    # None   : 每个 RNA token 都投影（原版，R 轨）
    # 32/64/128: 压到这么多 latent token
    compress_latent: int | None = None
    compress_method: str = "resampler"          # resampler | strided | meanpool
    compress_kwargs: dict | None = None         # 传给 RnaTokenCompressor 的额外参数

    # ---- E 轨：去各向异性 ----
    # 空的 path（None）表示关闭，行为与之前逐位一致。
    # 给一个 34_compute_aniso_mean.py 产出的 .pt，就在编码器输出后减掉那个公共方向。
    # 为什么：ERNIE-RNA 原始输出的模长 91% 落在跨序列公共方向上，
    # 实测 RNA 特异部分只占 42.9%（减完之后）。
    de_aniso: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def cosine_with_warmup(step: int, cfg: Stage1Config, total_steps: int) -> float:
    """linear warmup + cosine 衰减，起点 warmup_lr，峰值 lr，终点 lr_final。"""
    if step < cfg.warmup_steps:
        ratio = (step + 1) / max(cfg.warmup_steps, 1)
        return cfg.warmup_lr + (cfg.lr - cfg.warmup_lr) * ratio
    progress = min(max((step - cfg.warmup_steps) / max(total_steps - cfg.warmup_steps, 1), 0.0), 1.0)
    import math

    return cfg.lr_final + 0.5 * (cfg.lr - cfg.lr_final) * (1 + math.cos(math.pi * progress))


class Stage1Aligner(nn.Module):
    """冻结的编码器 + 冻结的 LLM + 可训练的投影（E 轨时再加一个可训练的压缩器）。"""

    def __init__(self, cfg: Stage1Config):
        super().__init__()
        self.cfg = cfg
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[cfg.dtype]

        from transformers import AutoModelForCausalLM, AutoTokenizer

        from rnaqwen.paths import describe_encoder

        self.encoder = RnaFmEncoder(cfg.encoder_path, dtype=dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = AutoModelForCausalLM.from_pretrained(cfg.llm_path, dtype=dtype).eval()
        for p in self.llm.parameters():
            p.requires_grad_(False)
        if cfg.grad_checkpointing:
            self.llm.gradient_checkpointing_enable()
            self.llm.config.use_cache = False

        self.llm_hidden = int(self.llm.config.hidden_size)
        self.projection = LinearProjection(self.encoder.hidden_size, self.llm_hidden).to(dtype)

        # 把「实际在用哪个 encoder、属于哪条轨」明明白白打出来。
        # 换 backbone 时最怕的就是"以为在用 A、实际在用 B"，所以这里必须出声。
        print(describe_encoder(cfg.encoder_path), flush=True)
        print(f"[stage1] {self.encoder.describe()}  ->  投影 "
              f"{self.encoder.hidden_size} -> {self.llm_hidden}", flush=True)

        # ---- E 轨：去各向异性（默认关）----
        # 注册成 buffer 才会跟着 `.to(device)` 走（踩过：普通属性在 nn.Module.to() 里
        # 不会被搬运）；persistent=False 保证它不进 state_dict —— 它是**冻结编码器的
        # 静态属性**，不是可训参数，混进 checkpoint 只会让人误以为要加载它。
        # 实测：ERNIE-RNA 原始输出模长的 91.1% 落在跨序列公共方向上。
        mean = torch.zeros(self.encoder.hidden_size, dtype=dtype)
        if cfg.de_aniso:
            blob = torch.load(cfg.de_aniso, map_location="cpu")
            m = blob["mean"].to(dtype)
            if m.numel() != self.encoder.hidden_size:
                raise ValueError(
                    f"de_aniso 均值维度 {m.numel()} != encoder hidden "
                    f"{self.encoder.hidden_size}")
            mean = m
            print(f"[stage1] E 轨去各向异性：减掉均值方向（模长 {float(m.norm()):.3f}）",
                  flush=True)
        self.register_buffer("aniso_mean", mean, persistent=False)
        self._de_aniso_on = bool(cfg.de_aniso)

        # ---- E 轨（默认关）：compress_latent=None 时不建任何模块 ----
        self.compression: RnaTokenCompressor | None = None
        if cfg.compress_latent:
            self.compression = RnaTokenCompressor(
                in_dim=self.encoder.hidden_size,
                n_latent=int(cfg.compress_latent),
                method=cfg.compress_method,
                **(cfg.compress_kwargs or {}),
            ).to(dtype)

    # ---- 可训参数（R 轨只有投影；E 轨是压缩器 + 投影）----
    def trainable_modules(self) -> list[nn.Module]:
        mods = [self.projection]
        if self.compression is not None:
            mods.insert(0, self.compression)
        return mods

    def trainable_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for mod in self.trainable_modules():
            params.extend(mod.parameters())
        return params

    def describe_trainable(self) -> str:
        parts = [mod.describe() if hasattr(mod, "describe") else repr(mod)
                 for mod in self.trainable_modules()]
        total = sum(p.numel() for p in self.trainable_parameters())
        proj = sum(p.numel() for p in self.projection.parameters())
        tail = f"可训合计 {total:,}"
        if self.compression is not None:
            tail += f"（复现轨 {proj:,} + 压缩器 {total - proj:,}）"
        parts.append(tail)
        return " | ".join(parts)

    # ---- 组装一条样本 ----
    def build_batch(self, sequences: list[str], targets: list[str], device: torch.device):
        enc = self.encoder(sequences)                     # (B, L, 640)
        rna = enc.hidden.to(device)
        rna_mask = enc.mask.to(device)

        # E 轨：去各向异性。放在**编码器之后、压缩器之前**——
        # 公共分量是编码器输出自带的（实测 91.1%），在这一步减最对症。
        #
        # ⚠️ 必须显式 to(device)：`aniso_mean` 是 **Stage1Aligner 自己**的 buffer，
        #    而训练循环搬的是 `model.encoder/llm/projection` 三个**子模块**，
        #    父模块从来没被 `.to()` 过，那个 buffer 就一直留在 CPU。
        #    现象：RuntimeError: Expected all tensors to be on the same device。
        #    这是「靠搬运机制自动生效」这类假设的第二次翻车，改成就地对齐。
        if self._de_aniso_on:
            rna = rna - self.aniso_mean.to(device=rna.device, dtype=rna.dtype)

        # ---- E 轨（默认关）：RNA-FM -> compressor -> projection ----
        # compress_latent=None 时 self.compression is None，这一段**不执行**，
        # 下面的 proj / labels / attn 与加开关之前逐位一致。
        if self.compression is not None:
            rna, rna_mask = self.compression(rna, rna_mask)   # (B, n_latent, 640), 全 1 mask

        tok = self.tokenizer(
            targets,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_text_tokens,
            add_special_tokens=True,
        )
        text_ids = tok["input_ids"].to(device)
        text_mask = tok["attention_mask"].to(device)

        # ---- 原版的输入模板（arXiv 2411.08900v1 §2.1）----
        #     Q: <RNA><RNAHere></RNA><QuestionPrompts>
        #     A: <Description>
        # Stage 1 里 **Q 留空**（原文："the question Q is left empty"）：
        # 模型只能先从投影后的 RNA 表征里读出信息，才可能把描述说对 ——
        # 这就是它从结构上堵死"抄 prompt 文本"那条捷径的办法。
        #
        # `<RNAHere>` 的位置由**投影后的软提示**占位；其余标记按普通文本 token 走
        # （原文没公开这几个标记的 tokenizer 细节，只能这样等价实现）。
        prefix_ids = self._encode_const(self.PREFIX_TEXT, device)
        suffix_ids = self._encode_const(self.SUFFIX_TEXT, device)

        proj = self.projection(rna)                        # (B, L, 640→4096)
        prefix_emb = self.llm.get_input_embeddings()(prefix_ids)
        suffix_emb = self.llm.get_input_embeddings()(suffix_ids)
        text_emb = self.llm.get_input_embeddings()(text_ids)

        inputs = torch.cat([
            prefix_emb.expand(rna.shape[0], -1, -1),
            proj,
            suffix_emb.expand(rna.shape[0], -1, -1),
            text_emb,
        ], dim=1)

        n_prefix = prefix_emb.shape[1]
        rna_labels = torch.full((text_ids.shape[0], proj.shape[1]), -100,
                                dtype=text_ids.dtype, device=device)
        prefix_labels = torch.full((text_ids.shape[0], n_prefix), -100,
                                   dtype=text_ids.dtype, device=device)
        suffix_labels = torch.full((text_ids.shape[0], suffix_emb.shape[1]), -100,
                                   dtype=text_ids.dtype, device=device)
        ans_labels = torch.where(text_mask.bool(), text_ids, torch.full_like(text_ids, -100))
        # 只有 <Description> 后面的注释文本算 loss
        labels = torch.cat([prefix_labels, rna_labels, suffix_labels, ans_labels], dim=1)

        prefix_mask = torch.ones((rna_mask.shape[0], n_prefix), dtype=rna_mask.dtype, device=device)
        suffix_mask = torch.ones((rna_mask.shape[0], suffix_emb.shape[1]), dtype=rna_mask.dtype, device=device)
        attn = torch.cat([prefix_mask, rna_mask, suffix_mask, text_mask], dim=1)
        return inputs, attn, labels

    # 模板常量**单一来源**在 rnaqwen.models.templates —— 训练、推理、各档消融必须
    # 拼出逐字相同的字符串，否则推理会落到分布外（见 templates.py 顶部注释）。
    from rnaqwen.models.templates import RNA_PREFIX as PREFIX_TEXT  # noqa: E402
    from rnaqwen.models.templates import STAGE1_SUFFIX as SUFFIX_TEXT  # noqa: E402

    def _encode_const(self, text: str, device: torch.device) -> torch.Tensor:
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        return ids.to(device)

    def forward_loss(self, sequences: list[str], targets: list[str], device: torch.device) -> torch.Tensor:
        inputs, attn, labels = self.build_batch(sequences, targets, device)
        out = self.llm(inputs_embeds=inputs, attention_mask=attn, labels=labels)
        return out.loss


@torch.no_grad()
def evaluate_alignment(
    model: "Stage1Aligner",
    samples: list[dict],
    device: torch.device,
    zero_encoder: bool = False,
) -> dict:
    """在给定样本上算 loss，并给出**投影塌缩**的指纹。

    返回的不只是 loss，还有 `proj_std` —— 投影输出在 (batch, seq) 维上、逐维度求标准差
    再取均值。**投影塌缩时它会趋近 0**，而且往往比 loss 更早暴露问题：
    塌缩的投影输出一个近乎常量向量，LLM 照写不误，**loss 会一直好看**。
    """
    from rnaqwen.models.encoder import EncoderOutput

    was_training = model.projection.training
    model.projection.eval()
    if model.compression is not None:
        model.compression.eval()

    orig_forward = model.encoder.forward
    if zero_encoder:
        def zero_forward(seqs):  # noqa: ANN001
            out = orig_forward(seqs)
            return EncoderOutput(hidden=torch.zeros_like(out.hidden), mask=out.mask, lengths=out.lengths)
        model.encoder.forward = zero_forward  # type: ignore[assignment]

    total, n = 0.0, 0
    stds: list[float] = []
    try:
        for s in samples:
            tgt = s.get("annotation") or ""
            if not s.get("sequence") or not tgt.strip():
                continue
            loss = model.forward_loss([s["sequence"]], [tgt], device)
            total += float(loss)
            n += 1
            if not zero_encoder:
                enc = model.encoder([s["sequence"]])
                hidden, mask = enc.hidden.to(device), enc.mask.to(device)
                if model.compression is not None:
                    hidden, mask = model.compression(hidden, mask)
                proj = model.projection(hidden)
                # 只统计有效位置；std 跨 (batch, seq) 逐维度算
                valid = proj[mask.bool()]
                if valid.shape[0] > 1:
                    stds.append(float(valid.float().std(dim=0).mean()))
    finally:
        model.encoder.forward = orig_forward  # type: ignore[assignment]
        model.projection.train(was_training)
        if model.compression is not None:
            model.compression.train(was_training)

    return {"loss": total / max(n, 1), "n": n,
            "proj_std": (sum(stds) / len(stds)) if stds else None}


def train_stage1(records: list[dict], cfg: Stage1Config, device: str | None = None,
                 resume: str | None = None, val_records: list[dict] | None = None) -> dict:
    """records：每条要有 `sequence` 和 `annotation`（或 `literature`）。

    `val_records`：给了就每 `cfg.eval_every` 步算一次验证 loss，并跑**置零编码器探针**。
    **强烈建议给** —— 没有它，你只有一条 train loss，分不清"在学"和"在背"和"塌了"。
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = Stage1Aligner(cfg)
    model.encoder.to(device)
    model.llm.to(device)
    for mod in model.trainable_modules():          # R 轨 = 只有 projection（行为不变）
        mod.to(device)
        mod.train()
    trainable = model.trainable_parameters()

    print(f"[stage1] {model.describe_trainable()}  device={device}")
    print(f"[stage1] 训练样本 {len(records)} 条；epochs={cfg.epochs} bs={cfg.batch_size} "
          f"accum={cfg.grad_accum}")

    # ---- 验证/探针集：固定子集，保证跨步可比（不随机抽）----
    probe_val = (val_records or [])[: cfg.eval_samples]
    if probe_val:
        print(f"[stage1] 验证集 {len(val_records)} 条，每次评估取前 {len(probe_val)} 条；"
              f"每 {cfg.eval_every} 步评估一次（含置零探针）")
    else:
        print("[stage1] ⚠️ 没有验证集 —— 只有 train loss，分不清「在学 / 在背 / 投影塌缩」")

    opt = torch.optim.AdamW(
        trainable,
        lr=cfg.lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.weight_decay,
    )
    total_steps = max(len(records) // max(cfg.batch_size * cfg.grad_accum, 1) * cfg.epochs, 1)

    # ---- 续跑：这个环境会莫名其妙杀进程（实测 3 次），长跑必须能接上 ----
    # 注意：`--resume-from` 给的是**步数**，我们从「第 start_step 个样本」接着喂。
    # batch=1 / accum=1 时样本序号 == 步数，所以直接切片；其它配置下是近似。
    start_step = 0
    if resume:
        rp = Path(resume)
        if not rp.exists():
            print(f"[stage1] ⚠️ --resume-from 指向的文件不存在：{rp}，从头开始")
        else:
            blob = torch.load(rp, map_location="cpu")
            model.projection.load_state_dict(blob["projection"])
            if model.compression is not None and "compression" in blob:
                model.compression.load_state_dict(blob["compression"])
            if blob.get("optimizer"):
                try:
                    opt.load_state_dict(blob["optimizer"])
                except Exception as exc:  # noqa: BLE001
                    print(f"[stage1] ⚠️ 优化器状态载入失败（{type(exc).__name__}），只续权重：{exc}")
            start_step = int(blob.get("step", 0))
            print(f"[stage1] 续跑：从 {rp.name} 接上（step={start_step}）")

    history: list[dict] = []
    step = start_step
    done_micro = 0                     # 已经消费过的 micro-batch 数
    t0 = time.time()
    collapse_streak = 0
    val_bad = 0
    best_val = float("inf")
    max_delta = 0.0
    for epoch in range(cfg.epochs):
        if cfg.max_steps and step >= cfg.max_steps:
            print(f"[stage1] 到达 --max-steps={cfg.max_steps}，停", flush=True)
            break
        for i in range(0, len(records), cfg.batch_size):
            batch = records[i : i + cfg.batch_size]
            # 续跑：跳过已经训过的 micro-batch
            if done_micro < start_step * cfg.grad_accum:
                done_micro += 1
                continue
            done_micro += 1
            seqs = [b["sequence"] for b in batch]
            tgts = [b.get("annotation") or b.get("literature") or "" for b in batch]
            if not all(t.strip() for t in tgts):
                continue

            loss = model.forward_loss(seqs, tgts, torch.device(device)) / cfg.grad_accum
            loss.backward()

            if (i // cfg.batch_size + 1) % cfg.grad_accum == 0:
                lr = cosine_with_warmup(step, cfg, total_steps)
                for g in opt.param_groups:
                    g["lr"] = lr
                torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1

                # 定步数预算：做「压缩 vs 不压缩」这种配对对比时，
                # 两个 arm 必须停在**同一步数**上，否则比的是「谁跑得久」不是「谁更好」。
                if cfg.max_steps and step >= cfg.max_steps:
                    print(f"[stage1] 到达 --max-steps={cfg.max_steps}，停", flush=True)
                    break

                if step % cfg.log_every == 0:
                    hist = {"step": step, "epoch": epoch, "loss": float(loss.detach()) * cfg.grad_accum,
                            "lr": lr, "elapsed_s": round(time.time() - t0, 1)}
                    history.append(hist)
                    print(f"[stage1] {json.dumps(hist)}", flush=True)

                # ---- 周期评估：val loss + 置零探针 + 塌缩指纹 ----
                if probe_val and cfg.eval_every and step % cfg.eval_every == 0:
                    ev_real = evaluate_alignment(model, probe_val, torch.device(device), zero_encoder=False)
                    ev_zero = evaluate_alignment(model, probe_val, torch.device(device), zero_encoder=True)
                    d_loss = ev_zero["loss"] - ev_real["loss"]
                    rec = {"step": step, "epoch": epoch,
                           "train_loss": round(float(loss.detach()) * cfg.grad_accum, 4),
                           "val_loss": round(ev_real["loss"], 4),
                           "val_loss_zeroed": round(ev_zero["loss"], 4),
                           "delta_loss_zeroed": round(d_loss, 4),
                           "proj_std": None if ev_real["proj_std"] is None else round(ev_real["proj_std"], 5),
                           "elapsed_s": round(time.time() - t0, 1)}
                    history.append({"eval": rec})
                    print(f"[stage1][eval] {json.dumps(rec, ensure_ascii=False)}", flush=True)

                    # ---- 判据 1：置零编码器的影响**相对衰减**（不是绝对值）----
                    max_delta = max(max_delta, d_loss)
                    if max_delta >= cfg.collapse_min_peak and d_loss < (1 - cfg.collapse_rel_drop) * max_delta:
                        collapse_streak += 1
                        print(f"[stage1][collapse?] Δloss(置零)={d_loss:.4f} 已跌到峰值 "
                              f"{max_delta:.4f} 的 {100*d_loss/max_delta:.0f}%"
                              f"  连续 {collapse_streak}/{cfg.collapse_patience} 次", flush=True)
                    else:
                        collapse_streak = 0

                    # ---- 判据 2：val_loss 平台期（说明在过拟合，再训只是记数据）----
                    if ev_real["loss"] < best_val - 1e-4:
                        best_val = ev_real["loss"]
                        val_bad = 0
                    else:
                        val_bad += 1
                        print(f"[stage1][plateau] val_loss={ev_real['loss']:.4f} 未改善，"
                              f"连续 {val_bad}/{cfg.val_patience} 次（最优 {best_val:.4f}）", flush=True)

                    reason = None
                    if collapse_streak >= cfg.collapse_patience:
                        reason = (f"投影塌缩：置零编码器对 loss 的影响跌到峰值 {max_delta:.4f} 的 "
                                  f"{100*d_loss/max(max_delta,1e-9):.0f}%（当前 {d_loss:.4f}）"
                                  f" —— 模型已经不用 RNA 表征了")
                    elif val_bad >= cfg.val_patience:
                        reason = (f"val 平台：连续 {val_bad} 次评估无改善（最优 {best_val:.4f}，"
                                  f"当前 {ev_real['loss']:.4f}），train 仍在降 —— 在过拟合")

                    if reason and cfg.early_stop_on_collapse:
                        print(f"[stage1][STOP] {reason}", flush=True)
                        print("[stage1][STOP] 继续训下去只是浪费算力；checkpoint 已存。", flush=True)
                        p = out_dir / f"projection_earlystop_step{step}.pt"
                        save_stage1(model, p, cfg, step)
                        (out_dir / "early_stop_reason.json").write_text(
                            json.dumps({"step": step, "reason": reason, "best_val": best_val,
                                        "max_delta": max_delta, "delta_now": d_loss,
                                        "history": history}, indent=2, ensure_ascii=False),
                            encoding="utf-8")
                        return {"steps": step, "history": history, "stopped_early": True,
                                "reason": reason, "checkpoint": str(p)}

                if cfg.save_every and step % cfg.save_every == 0:
                    save_stage1(model, out_dir / f"projection_step{step}.pt", cfg, step, optimizer=opt)

    final = out_dir / "projection_final.pt"
    save_stage1(model, final, cfg, step, optimizer=opt)
    print(f"[stage1] 完成，{step} 步，用时 {(time.time()-t0)/60:.1f} min -> {final}")
    return {"steps": step, "history": history, "collapsed": False, "checkpoint": str(final)}


def save_stage1(model: Stage1Aligner, path: Path, cfg: Stage1Config, step: int,
                optimizer=None) -> None:
    blob = {
        "projection": model.projection.state_dict(), "config": cfg.as_dict(), "step": step,
        "in_dim": model.encoder.hidden_size, "out_dim": model.llm_hidden,
    }
    if optimizer is not None:
        # 存优化器状态才能真续跑；只存权重的话 AdamW 的一阶/二阶矩会从头开始，
        # 续上之后 loss 会先跳一下。
        blob["optimizer"] = optimizer.state_dict()
    if model.compression is not None:
        # E 轨 checkpoint。⚠️ 读取方必须同时 load `compression`，否则压缩器是随机初始化的，
        # 结果会静默错误。已知会读 E 轨 checkpoint 的三处，都已处理：
        #   * scripts/22_zero_encoder_probe.py  —— 自动从 config 读到 compress_latent 并 load
        #   * src/rnaqwen/models/finetune.py    —— build_stage2_model 要求 --compress-latent 配对
        #   * scripts/30_infer.py               —— --compress-latent 显式指定
        blob["compression"] = model.compression.state_dict()
        blob["compression_spec"] = {
            "method": model.compression.method,
            "n_latent": model.compression.n_latent,
            "in_dim": model.compression.in_dim,
            "params": model.compression.num_parameters,
        }
        print(f"[stage1] ⚠️ E 轨 checkpoint：含压缩器 "
              f"({model.compression.describe()})，读取方必须一并 load")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, path)
