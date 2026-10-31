"""Stage 2 指令微调（LoRA）。

原版：10 epochs，warmup 200，8B 全参 —— 2×A100 40GB（arXiv 2411.08900v1 §A）。
本机：单卡 8060S（99.7 GB 统一内存），全参 8B 训练不现实，
所以按 PLAN §3 P2 走 **LoRA**，其余超参照抄。**这是对原版的偏离，已记入报告。**

样本形态（Stage 1 对齐之后）：

    [ 投影后的 RNA 表征 ] [ </RNA>\\n ] [ ###Human: <question>\\n###Assistant: ] [ <answer> ]
     不计 loss（软提示）    不计 loss          不计 loss（真问题在 prompt 里）        计 loss

和 Stage 1 的关键差别（为什么**不**复用 `alignment.build_batch`）
--------------------------------------------------------------
`alignment.build_batch` 组装的是 **Stage 1 模板**：

    "Q: <RNA>" + [软提示] + "</RNA>\\nA: <Description>" + <目标文本>
    且把 <目标文本> 的**全部 token** 都算 loss。

Stage 2 直接用它会出两个错：

 1. **模板错**。原版 Stage 1 的 Q 是**空**的（"the question Q is left empty"），
    Stage 2 才有真问题（"###Human: {question}"）。复用 build_batch 会得到
    `Q: <RNA></RNA>A: <Description>###Human: ...` —— 对模型而言这是
    "先来一段没有问题的 Stage 1 样本，紧接着又来一段问答"，等于用两个拼错的回合
    去教一个回合的行为，而且白白烧掉 prompt 的上下文预算。
 2. **loss 错**。build_batch 的 labels 是 `text_ids` 全体，问题 token 也会计 loss。
    Stage 2 只该在**答案**上算 loss —— 让模型学"复读问题"是纯噪声目标，
    还会稀释答案的梯度。

所以 Stage 2 在**本模块内**单独组装（`encode_samples`），`alignment.py` 一行不动
（另一个任务正在给它加 token 压缩开关）。
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import torch

from rnaqwen.models.alignment import Stage1Aligner, Stage1Config

# ---------------------------------------------------------------- LoRA 预设
# Qwen3-8B 结构（已核对本地 models/Qwen3-8B/config.json）：
#     36 层 / hidden 4096 / intermediate 12288 / kv 8×128 = 1024
#     q_proj, o_proj: (4096, 4096) | k_proj, v_proj: (1024, 4096)  ← GQA
#     gate_proj, up_proj: (12288, 4096) | down_proj: (4096, 12288)
# 注意 k/v 的 **out_features 是 1024**（GQA），不是 4096。
Q_PROJ, K_PROJ, V_PROJ, O_PROJ, GATE, UP, DOWN = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)
ATTN_TARGETS = (Q_PROJ, K_PROJ, V_PROJ, O_PROJ)
MLP_TARGETS = (GATE, UP, DOWN)
ALL_TARGETS = ATTN_TARGETS + MLP_TARGETS

# (in_features, out_features) —— 用于**离线**复算参数量（不实例化 8B）
QWEN3_8B_SHAPES: dict[str, tuple[int, int]] = {
    Q_PROJ: (4096, 4096),
    K_PROJ: (4096, 1024),
    V_PROJ: (4096, 1024),
    O_PROJ: (4096, 4096),
    GATE: (4096, 12288),
    UP: (4096, 12288),
    DOWN: (12288, 4096),
}
QWEN3_8B_LAYERS = 36
QWEN3_8B_VOCAB = 151936
QWEN3_8B_HIDDEN = 4096

# 四条消融预设。**只**规定 targets / r / alpha 三件事，别的都不碰。
LORA_PRESETS: dict[str, dict] = {
    "attn":        {"targets": ATTN_TARGETS, "r": 16, "alpha": 32},
    "attn_mlp":    {"targets": ALL_TARGETS,  "r": 16, "alpha": 32},
    "attn_mlp_r8": {"targets": ALL_TARGETS,  "r": 8,  "alpha": 16},
    "attn_mlp_r64": {"targets": ALL_TARGETS, "r": 64, "alpha": 128},
}
DEFAULT_PRESET = "attn_mlp"

# 预设**只**覆盖这三个字段
PRESET_FIELDS = ("lora_targets", "lora_r", "lora_alpha")

# Stage 2 模板（原版 §2.1：Stage 1 的 prompt 改成 Llama-3 风格）
TEMPLATE_LLAMA3 = "llama3"
TEMPLATE_QWEN = "qwen"          # Qwen3 自己的 ChatML
TEMPLATE_STAGE1 = "stage1"      # 复用 Stage 1 模板（消融对照用）


class UnknownPresetError(ValueError):
    """预设名不在 LORA_PRESETS 里 —— 必须报错，不许静默退回默认。"""


def resolve_preset(name: str) -> dict:
    """预设名 -> {targets, r, alpha}。未知名直接抛错。"""
    if name not in LORA_PRESETS:
        raise UnknownPresetError(
            f"未知 LoRA preset {name!r}；可用：{sorted(LORA_PRESETS)}"
        )
    p = LORA_PRESETS[name]
    return {"targets": tuple(p["targets"]), "r": int(p["r"]), "alpha": int(p["alpha"])}


def lora_param_count(targets, r: int, layers: int = QWEN3_8B_LAYERS,
                     shapes: dict | None = None) -> int:
    """LoRA 参数量 = Σ_layers Σ_target r × (in_features + out_features)。

    PEFT 的 LoRA 对 `nn.Linear` 加 A(r×in) 与 B(out×r)，bias="none" 不引入额外参数。
    **离线算式**，不需要实例化 8B。
    """
    shapes = shapes or QWEN3_8B_SHAPES
    per_layer = sum(r * (shapes[t][0] + shapes[t][1]) for t in targets)
    return per_layer * layers


def frozen_backbone_params() -> int:
    """冻结底座参数量（36 层线性 + embedding + lm_head，tie=False）。"""
    lin = sum(a * b for (a, b) in QWEN3_8B_SHAPES.values()) * QWEN3_8B_LAYERS
    emb = QWEN3_8B_VOCAB * QWEN3_8B_HIDDEN * 2
    return lin + emb


def preset_budget(preset: str) -> dict:
    """一条预设的理论参数预算（离线），用于报告与自检。"""
    p = resolve_preset(preset)
    n = lora_param_count(p["targets"], p["r"])
    total = frozen_backbone_params()
    return {"preset": preset, "targets": list(p["targets"]), "r": p["r"],
            "alpha": p["alpha"], "lora_params": n, "frozen_params": total,
            "ratio_pct": 100.0 * n / total}


@dataclass
class Stage2Config:
    llm_path: str = "models/Qwen3-8B"
    # 默认走 paths.default_encoder()（E 轨 ERNIE-RNA）。R 轨复现请显式给 models/rnafm。
    encoder_path: str = field(default_factory=lambda: __import__("rnaqwen.paths", fromlist=["x"]).default_encoder())
    projection_path: str = ""          # Stage 1 产出的 projection_*.pt
    out_dir: str = "checkpoints/stage2"

    epochs: int = 10
    batch_size: int = 1
    grad_accum: int = 8
    lr: float = 2e-4
    warmup_steps: int = 200            # 照抄原版
    weight_decay: float = 0.05         # 照抄原版（原文 §A：Stage 2 "maintaining the same ... weight decay of 0.05"）
    max_grad_norm: float = 1.0
    scheduler: str = "cosine"

    # ---- LoRA ----
    # 这三个字段的**显式赋值**由 __init__ 盯 kwargs 记录（见下），
    # 预设只填没被显式指定的那些，别的参数一概不碰。
    preset: str = DEFAULT_PRESET       # attn | attn_mlp | attn_mlp_r8 | attn_mlp_r64
    lora_r: int = 16
    lora_alpha: int = 32
    lora_targets: tuple[str, ...] = ALL_TARGETS
    lora_dropout: float = 0.05
    use_rslora: bool = False           # rsLoRA：缩放 α/√r 而非 α/r（r≥16 更稳）

    # ---- SOS-LoRA（E 轨变体，替代标准 LoRA）----
    # 论文 arXiv 2607.16252v1 §4：r_tot=8, K=4, γ_max=2.5, dropout 0.05。
    # 我们 r_tot 用 16 —— 和 attn 预设同预算，才能做配对对比（不能照抄 8）。
    adapter: str = "lora"          # "lora" | "sos"
    sos_k: int = 4
    sos_gamma_max: float = 2.5
    sos_ortho_coef: float = 1.0    # 跨专家正交正则的权重


    # ---- 模板 / 序列长度 ----
    template: str = TEMPLATE_LLAMA3    # 原版 §2.1 用 Llama-3 风格
    max_nt: int = 1024
    max_prompt_tokens: int = 256       # ###Human: ... ###Assistant:
    max_answer_tokens: int = 512       # 答案（只有这部分算 loss）
    append_eos: bool = True            # 答案尾部补 EOS（否则模型学不会停）

    # ---- E 轨：Stage 1 若用了 RNA token 压缩，Stage 2 必须一致 ----
    # 默认 None = 和 R 轨一样原样投影。留 None 时 encode_samples 一行都不走压缩分支。
    # 传了值就必须同时给 --projection，压缩器权重从那个 checkpoint 的 "compression" 里取。
    compress_latent: int | None = None
    compress_method: str = "resampler"

    dtype: str = "bfloat16"
    grad_checkpointing: bool = True
    seed: int = 0
    log_every: int = 10
    save_every: int = 500

    # 非 dataclass 字段：记录哪些 LoRA 字段被**显式**传过（预设不许覆盖它们）
    explicit_fields: tuple[str, ...] = field(default=(), repr=False)

    def __init__(self, **kwargs) -> None:
        """按生成的 __init__ 逐字段赋值，借此记录**显式**传进来的 LoRA 字段。

        为什么不用 `lora_r: int | None = None` 那种写法：默认值本身就是 None，
        于是那些字段的类型注解骗人（下游拿到 Optional），而且显式传 16 跟
        "没传" 分不开。这里直接盯 kwargs，最诚实。
        """
        names = [f.name for f in dataclasses.fields(self)]
        unknown = set(kwargs) - set(names)
        if unknown:
            raise TypeError(f"Stage2Config 不认识的字段：{sorted(unknown)}")
        object.__setattr__(self, "explicit_fields",
                           tuple(sorted(set(kwargs) & set(PRESET_FIELDS))))
        for f in dataclasses.fields(self):
            if f.name == "explicit_fields":
                continue
            # 用 object.__setattr__ 落值，避免污染 explicit 集合
            object.__setattr__(self, f.name, kwargs.get(f.name, f.default))
        self.__post_init__()

    def __post_init__(self) -> None:
        # 1) 未知 preset 在这里直接抛错，绝不静默退回默认
        p = resolve_preset(self.preset)
        for f in PRESET_FIELDS:
            if f in self.explicit_fields:
                continue                      # 显式的说了算
            mapped = {"lora_targets": "targets", "lora_r": "r", "lora_alpha": "alpha"}[f]
            object.__setattr__(self, f, p[mapped])

        if self.template not in (TEMPLATE_LLAMA3, TEMPLATE_QWEN, TEMPLATE_STAGE1):
            raise ValueError(
                f"未知 template {self.template!r}；"
                f"可用：{TEMPLATE_LLAMA3} / {TEMPLATE_QWEN} / {TEMPLATE_STAGE1}"
            )
        if self.scheduler not in ("cosine", "none"):
            raise ValueError(f"未知 scheduler {self.scheduler!r}；可用：cosine / none")

    # ---- 组装模板 ----
    def format_prompt(self, question: str) -> str:
        """prompt 部分（**不计 loss**）。软提示由 `encode_samples` 插在前面。"""
        q = (question or "").strip()
        if self.template == TEMPLATE_LLAMA3:
            return f"###Human: {q}\n###Assistant:"
        if self.template == TEMPLATE_QWEN:
            return (f"<|im_start|>user\n{q}<|im_end|>\n"
                    f"<|im_start|>assistant\n")
        return ""   # stage1：prompt 留空，问题走 <target> 文本（对照消融）

    def format_target(self, question: str, answer: str) -> str:
        """stage1 模板下，问题与答案一起进 <target> 文本（build_batch 路径）。"""
        return f"###Human: {question}\n###Assistant: {answer}"

    def lora_param_estimate(self) -> int:
        return lora_param_count(self.lora_targets, self.lora_r)

    def describe(self) -> str:
        eff = ("显式覆盖：" + ",".join(sorted(self.explicit_fields))) if self.explicit_fields else "全部来自预设"
        return (f"preset={self.preset} targets={len(self.lora_targets)}个 r={self.lora_r} "
                f"alpha={self.lora_alpha} rslora={self.use_rslora} 理论LoRA参数≈"
                f"{self.lora_param_estimate()/1e6:.1f}M ({eff})")

    def as_dict(self) -> dict:
        d = asdict(self)
        d["lora_targets"] = list(self.lora_targets)
        d["explicit_fields"] = list(self.explicit_fields)
        d["lora_param_estimate"] = self.lora_param_estimate()
        return d


# ------------------------------------------------------------- 学习率调度
def cosine_with_warmup(step: int, cfg: Stage2Config, total_steps: int) -> float:
    """linear warmup + cosine 衰减到 0（和 Stage 1 同款）。"""
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(cfg.warmup_steps, 1)
    if cfg.scheduler == "none":
        return cfg.lr
    progress = min(max((step - cfg.warmup_steps) / max(total_steps - cfg.warmup_steps, 1), 0.0), 1.0)
    return 0.5 * cfg.lr * (1 + math.cos(math.pi * progress))


# --------------------------------------------------------------- 构建模型
def build_lora_config(cfg: Stage2Config, task_type: str = "CAUSAL_LM"):
    """Stage2Config -> peft.LoraConfig（use_rslora 在这里透传）。"""
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_targets),
        bias="none",
        task_type=task_type,
        use_rslora=cfg.use_rslora,
    )


def build_stage2_model(cfg: Stage2Config, device: str = "cuda", report: dict | None = None):
    """Stage1Aligner（冻结底座 + **可训**投影）+ LoRA 适配器。

    投影层在 Stage 2 里保持可训：原版 Stage 2 是整个架构继续训，投影不冻。
    """
    from peft import get_peft_model

    s1 = Stage1Config(
        llm_path=cfg.llm_path, encoder_path=cfg.encoder_path,
        max_nt=cfg.max_nt, max_text_tokens=cfg.max_answer_tokens,
        dtype=cfg.dtype, grad_checkpointing=cfg.grad_checkpointing, seed=cfg.seed,
        compress_latent=cfg.compress_latent, compress_method=cfg.compress_method,
    )
    model = Stage1Aligner(s1)

    if cfg.projection_path and Path(cfg.projection_path).exists():
        blob = torch.load(cfg.projection_path, map_location="cpu")
        model.projection.load_state_dict(blob["projection"])
        print(f"[stage2] 载入 Stage1 投影 <- {cfg.projection_path} (step {blob.get('step')})")
        # E 轨：压缩器和投影是一起训出来的，必须成对装回，否则 token 数对不上。
        if cfg.compress_latent:
            if "compression" not in blob:
                raise SystemExit(
                    f"[fail] 指定了 compress_latent={cfg.compress_latent}，但 "
                    f"{cfg.projection_path} 里没有 compression 权重 —— "
                    f"这个 Stage-1 checkpoint 是 R 轨（未压缩）的，别混用。")
            model.compression.load_state_dict(blob["compression"])
            spec = blob.get("compression_spec") or {}
            print(f"[stage2] 载入压缩器 <- {cfg.projection_path} "
                  f"({spec.get('method')} n_latent={spec.get('n_latent')})")
        elif "compression" in blob:
            raise SystemExit(
                f"[fail] {cfg.projection_path} 是 E 轨（带压缩器）的 Stage-1 产物，"
                f"但没给 --compress-latent。请加上 --compress-latent "
                f"{(blob.get('compression_spec') or {}).get('n_latent')}。")
    elif cfg.projection_path:
        print(f"[warn] 投影文件不存在：{cfg.projection_path}，投影从随机初始化开始")
    else:
        print("[warn] 未指定 --projection，投影从随机初始化开始")

    if cfg.adapter == "sos":
        # ---- SOS-LoRA：自己实现的，不走 peft ----
        from rnaqwen.models.sos_lora import (
            freeze_except_sos, replace_linear_with_sos, sos_lora_penalty, sos_trainable_params,
        )

        # r_tot 直接沿用 cfg.lora_r，alpha 沿用 cfg.lora_alpha ——
        # 这样和标准 LoRA 是**同预算**对比，唯一变量是「单一子空间 vs K 个正交专家」。
        replaced = replace_linear_with_sos(
            model.llm, tuple(cfg.lora_targets),
            r_tot=cfg.lora_r, k=cfg.sos_k, alpha=cfg.lora_alpha,
            dropout=cfg.lora_dropout, gamma_max=cfg.sos_gamma_max,
        )
        if not replaced:
            raise SystemExit(
                f"[fail] SOS 一层都没替换到（targets={cfg.lora_targets}）—— "
                f"静默没接上等于没加适配器，拒绝继续")
        print(f"[stage2] SOS-LoRA 接在 {len(replaced)} 个层上：{replaced[:4]}"
              f"{' ...' if len(replaced) > 4 else ''}")
        n_frozen = freeze_except_sos(model.llm)
        print(f"[stage2] 冻掉底座 {n_frozen/1e6:.3f} M 可训参数；"
              f"SOS 可训 {sos_trainable_params(model.llm):,}")
        print(f"[stage2] SOS 超参：r_tot={cfg.lora_r} K={cfg.sos_k} "
              f"r'={cfg.lora_r // cfg.sos_k} gamma_max={cfg.sos_gamma_max} "
              f"ortho_coef={cfg.sos_ortho_coef}")
    else:
        lora = build_lora_config(cfg, task_type="CAUSAL_LM")
        print(f"[stage2] LoraConfig: r={lora.r} alpha={lora.lora_alpha} "
              f"use_rslora={lora.use_rslora} target_modules={sorted(lora.target_modules)}")
        model.llm = get_peft_model(model.llm, lora)
        n_leaked = _freeze_except_lora(model.llm)
        if n_leaked:
            print(f"[stage2] 关掉 PEFT 漏放的底座可训参数 {n_leaked/1e6:.3f} M")
        model.llm.print_trainable_parameters()

    if report is not None:
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        lora_p = sum(p.numel() for n, p in named if "lora_" in n)
        proj_p = sum(p.numel() for n, p in named
                     if n.startswith("projection") or n.startswith("compression"))
        other_p = sum(p.numel() for n, p in named) - lora_p - proj_p
        report.update({
            "lora_params_peft": lora_p,
            "projection_params": proj_p,
            "other_trainable_params": other_p,
            "trainable_params_total": lora_p + proj_p + other_p,
            "lora_param_estimate": cfg.lora_param_estimate(),
            "n_lora_modules": sum(1 for n, _ in named if n.endswith("lora_A.default.weight")),
            "frozen_params": sum(p.numel() for p in model.parameters() if not p.requires_grad),
            "compress_latent": cfg.compress_latent,
            "compress_method": cfg.compress_method if cfg.compress_latent else None,
        })

    model.encoder.to(device)
    model.llm.to(device)
    model.projection.to(device)
    if model.compression is not None:
        model.compression.to(device)
        model.compression.train()
    model.projection.train()
    model.llm.train()
    return model


# ------------------------------------------------- Stage 2 样本组装（核心）
def _encode(tokenizer, text: str, max_tokens: int, device) -> torch.Tensor:
    """无特殊 token、截断到 max_tokens，返回 (1, n) 的 ids。"""
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False,
                    truncation=True, max_length=max_tokens)["input_ids"]
    return ids.to(device)


def _freeze_except_lora(llm) -> int:
    """PEFT 包装后**只**留 LoRA 参数可训。

    为什么需要：PEFT 的 `_mark_only_adapters_as_trainable` 只处理 base model 里的
    `nn.Linear`（当 bias="none"）。LayerNorm / q_norm / k_norm / embed_tokens
    的 requires_grad 归它管不到，于是这些冻结底座的参数会漏进优化器。
    原版 Stage 2 是全参训，但我们走 LoRA，必须把这扇门关死。
    """
    n_extra = 0
    for name, p in llm.named_parameters():
        keep = "lora_" in name
        if p.requires_grad and not keep:
            n_extra += p.numel()
        p.requires_grad_(keep)
    return n_extra


def encode_samples(model: Stage1Aligner, cfg: Stage2Config, sequences: list[str],
                   prompts: list[str], answers: list[str], device: torch.device,
                   mask_answer_only: bool = True):
    """组装 (inputs_embeds, attention_mask, labels)。

        inputs = [ prefix | proj(rna) | suffix | prompt | answer ]
        labels = [ -100 ×(其余全部)                                  | answer ]

    返回 (inputs, attn, labels, meta)。meta 里有各段 token 数，便于自检。
    """
    tok = model.tokenizer
    enc = model.encoder(sequences)                       # (B, L, 640)
    rna = enc.hidden.to(device)
    rna_mask = enc.mask.to(device)

    # ---- E 轨（默认关）：RNA-FM -> compressor -> projection ----
    # compress_latent=None 时 model.compression is None，这一段不执行，
    # 序列与 R 轨逐位一致。
    if model.compression is not None:
        rna, rna_mask = model.compression(rna, rna_mask)   # (B, n_latent, 640)

    prefix_ids = model._encode_const(Stage1Aligner.PREFIX_TEXT, device)
    suffix_ids = model._encode_const(Stage1Aligner.SUFFIX_TEXT, device)

    prompt_ids = [_encode(tok, p, cfg.max_prompt_tokens, device) for p in prompts]
    answer_ids = [_encode(tok, a, cfg.max_answer_tokens, device) for a in answers]
    if cfg.append_eos and getattr(tok, "eos_token_id", None) is not None:
        eos = torch.tensor([[tok.eos_token_id]], dtype=answer_ids[0].dtype, device=device)
        answer_ids = [torch.cat([a, eos], dim=1) for a in answer_ids]

    proj = model.projection(rna)                          # (B, L, 640→4096)
    embed = model.llm.get_input_embeddings()
    prefix_emb = embed(prefix_ids)
    suffix_emb = embed(suffix_ids)

    B = rna.shape[0]
    dev_dtype = proj.dtype
    blocks, label_blocks, mask_blocks = [], [], []
    lens = []

    n_prefix = prefix_emb.shape[1]
    n_suffix = suffix_emb.shape[1]
    zero = torch.zeros((0,), dtype=torch.long, device=device)
    for b in range(B):
        prompt = zero if cfg.template == TEMPLATE_STAGE1 else prompt_ids[b][0]
        answer = answer_ids[b][0]
        p_emb = embed(prompt.unsqueeze(0))[0]
        a_emb = embed(answer.unsqueeze(0))[0]

        blocks.append(torch.cat([prefix_emb[0], proj[b], suffix_emb[0], p_emb, a_emb], dim=0))
        # prompt 段的 label：默认 -100（prompt 不计 loss）；mask_answer_only=False 时
        # **只是把标签露出来**，序列内容完全不变（否则 lengths 对不上）。
        n_prompt_lab = prompt.clone() if not mask_answer_only else torch.full(
            (prompt.shape[0],), -100, dtype=torch.long, device=device)
        n_prefix_l = torch.full((n_prefix,), -100, dtype=torch.long, device=device)
        n_rna_l = torch.full((proj.shape[1],), -100, dtype=torch.long, device=device)
        n_suffix_l = torch.full((n_suffix,), -100, dtype=torch.long, device=device)
        n_ans_l = answer.clone()
        label_blocks.append(torch.cat([n_prefix_l, n_rna_l, n_suffix_l, n_prompt_lab, n_ans_l], dim=0))
        mask_blocks.append(torch.cat([
            torch.ones((n_prefix,), dtype=rna_mask.dtype, device=device),
            rna_mask[b],
            torch.ones((n_suffix,), dtype=rna_mask.dtype, device=device),
            torch.ones((prompt.shape[0],), dtype=rna_mask.dtype, device=device),
            torch.ones((answer.shape[0],), dtype=rna_mask.dtype, device=device),
        ], dim=0))
        lens.append({"n_prefix": n_prefix, "n_rna": int(rna_mask[b].sum()),
                     "n_suffix": n_suffix, "n_prompt": int(prompt.shape[0]),
                     "n_answer": int(answer.shape[0])})

    max_len = max(t.shape[0] for t in blocks)
    pad = max_len
    inputs = torch.zeros((B, pad, proj.shape[-1]), dtype=dev_dtype, device=device)
    labels = torch.full((B, pad), -100, dtype=torch.long, device=device)
    attn = torch.zeros((B, pad), dtype=rna_mask.dtype, device=device)
    for b, (blk, lab, msk) in enumerate(zip(blocks, label_blocks, mask_blocks)):
        n = blk.shape[0]
        inputs[b, :n] = blk
        labels[b, :n] = lab
        attn[b, :n] = msk

    meta = {"lens": lens, "seq_len": int(pad),
            "n_supervised": int((labels != -100).sum().item()),
            "n_prompt_supervised": 0 if mask_answer_only else int(
                sum(l["n_prompt"] for l in lens)),
            "n_answer_total": int(sum(l["n_answer"] for l in lens))}
    return inputs, attn, labels, meta


def forward_loss(model: Stage1Aligner, cfg: Stage2Config, sequences: list[str],
                 questions: list[str], answers: list[str], device: torch.device):
    """Stage 2 的 forward。返回 (loss, meta)。

    走 `encode_samples` 单独组装 —— **不**复用 `alignment.build_batch`（见模块 docstring）。
    """
    inputs, attn, labels, meta = encode_samples(model, cfg, sequences, questions, answers, device)
    out = model.llm(inputs_embeds=inputs, attention_mask=attn, labels=labels)
    return out.loss, meta


# ------------------------------------------------------------------ 训练
def train_stage2(records: list[dict], cfg: Stage2Config, device: str = "cuda",
                 resume: str | None = None) -> dict:
    """records：每条含 `sequence` 与 `qa`（list[{question, answer, ...}]）。

    每条 QA 展开成一个训练样本（`sequence` 与所属 RNA 配对）。
    """
    torch.manual_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for rec in records:
        seq = rec.get("sequence") or ""
        if not seq:
            continue
        for pair in rec.get("qa", []):
            q = (pair.get("question") or "").strip()
            a = (pair.get("answer") or "").strip()
            if not a:
                continue
            if cfg.template == TEMPLATE_STAGE1:
                # 对照消融：沿用 Stage 1 读法 —— Q 留空，问题与答案全塞进 <target> 文本
                # （这就是"直接把 QA 文本塞进 build_batch 的 target"那条路，用来对照）
                samples.append({"sequence": seq, "prompt": "",
                                "answer": cfg.format_target(q, a)})
            else:
                samples.append({"sequence": seq, "prompt": cfg.format_prompt(q), "answer": a})
    if not samples:
        raise SystemExit("[fail] 没有可用的 QA 样本（检查数据里是否有 qa[].question/answer）")

    report: dict = {"preset": cfg.preset, "template": cfg.template}
    model = build_stage2_model(cfg, device, report=report)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(len(samples) // max(cfg.batch_size, 1), 1)
    total_steps = max(steps_per_epoch // max(cfg.grad_accum, 1) * cfg.epochs, 1)
    print(f"[stage2] {len(samples)} 条指令样本；epochs={cfg.epochs} bs={cfg.batch_size} "
          f"accum={cfg.grad_accum} 共约 {total_steps} 步")

    history: list[dict] = []
    step, micro, t0 = 0, 0, time.time()
    accum_loss, accum_n = 0.0, 0
    for epoch in range(cfg.epochs):
        for i in range(0, len(samples), cfg.batch_size):
            batch = samples[i: i + cfg.batch_size]
            loss, meta = forward_loss(model, cfg, [b["sequence"] for b in batch],
                                      [b["prompt"] for b in batch],
                                      [b["answer"] for b in batch],
                                      torch.device(device))
            accum_loss += float(loss.detach())
            accum_n += 1
            # SOS 的正交正则只有当次 forward 才算得出来，所以加在这里；
            # 它同样被下面的 /grad_accum 除，累积 grad_accum 次后总量正确。
            if cfg.adapter == "sos":
                # 就地 import：build_stage2_model 里那个 import 在函数作用域内，这里看不见
                # （真踩过：NameError: name 'sos_lora_penalty' is not defined）
                from rnaqwen.models.sos_lora import sos_lora_penalty

                loss = loss + cfg.sos_ortho_coef * sos_lora_penalty(model.llm)
            (loss / cfg.grad_accum).backward()
            micro += 1
            if micro % cfg.grad_accum != 0:
                continue

            lr = cosine_with_warmup(step, cfg, total_steps)
            for g in opt.param_groups:
                g["lr"] = lr
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % cfg.log_every == 0:
                h = {"step": step, "epoch": epoch,
                     "loss": accum_loss / max(accum_n, 1), "lr": lr,
                     "n_supervised_tok": meta["n_supervised"], "seq_len": meta["seq_len"],
                     "elapsed_s": round(time.time() - t0, 1)}
                history.append(h)
                print(f"[stage2] {json.dumps(h)}", flush=True)
            accum_loss, accum_n = 0.0, 0
            if cfg.save_every and step % cfg.save_every == 0:
                save_stage2(model, out_dir / f"step{step}", cfg, step)

    final = out_dir / "final"
    save_stage2(model, final, cfg, step)
    result = {"steps": step, "samples": len(samples), "epochs": cfg.epochs,
              "history": history, "checkpoint": str(final),
              "lora_param_estimate": cfg.lora_param_estimate()}
    result.update(report)
    print(f"[stage2] 完成 {step} 步，用时 {(time.time()-t0)/60:.1f} min -> {final}")
    return result


def save_stage2(model, path: Path, cfg: Stage2Config, step: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if cfg.adapter == "sos":
        # SOS 是**可合并**的（论文的核心卖点）：把 ΔW = Σ s_k A_k B_k 烘进底座权重，
        # 存出来就是普通 Qwen —— 30_infer 不需要认识 SOS，推理端零改动、零额外延迟。
        # 代价：原地改结构，合并后这个 model 不能再训。所以放在存盘路径里。
        from rnaqwen.models.sos_lora import merge_sos_layers

        n_merged = merge_sos_layers(model.llm)
        print(f"[stage2] 已把 SOS 合并进底座：{n_merged} 层")
        model.llm.save_pretrained(path / "merged_llm")
    else:
        model.llm.save_pretrained(path / "lora")
    blob = {"projection": model.projection.state_dict(), "config": cfg.as_dict(), "step": step}
    # E 轨：压缩器是投影的上游，必须一起存，否则 30_infer 装不回同一条前向路径。
    if model.compression is not None:
        blob["compression"] = model.compression.state_dict()
        blob["compression_spec"] = {
            "method": model.compression.method,
            "n_latent": model.compression.n_latent,
            "in_dim": model.compression.in_dim,
            "params": model.compression.num_parameters,
        }
    torch.save(blob, path / "projection.pt")
