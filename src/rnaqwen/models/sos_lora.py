"""SOS-LoRA：Static Orthogonal-Subspace Low-Rank Adaptation with Fixed Multi-Scale Scaling

论文：arXiv 2607.16252v1（ACL 2026 Main），Yupeng Chang / Yuan Wu / Yi Chang
官方实现：https://github.com/llm172/sos-lora

**超参全部来自论文正文 §4 "Implementation Details"，不是猜的：**
    r_tot = 8 per adapted matrix, K = 4 experts (r' = 2), LoRA dropout 0.05
    γ_k 在 [1, γ_max=2.5] 上**线性等距**，s_k = (α/r_tot)·γ_k
    跨专家正交初始化（B 零初始化 -> 初始更新为零）
    正则：惩罚 A 的**跨专家**相关（同专家内部不罚）

前向：Y = X·W₀ + Σ_k s_k·X·A_k·B_k
合并：ΔW = Σ_k s_k·A_k·B_k      —— 可并进 W₀，推理零开销

**为什么不直接搬官方代码**：它的 requirements 钉 `peft==0.14.0`，但
`utils/init_crossover.py` 里 `from peft import CrossoverConfig` —— stock peft 没这个符号，
说明那个文件依赖某个 fork 或是死代码。而且它要 CUDA/flash-attn/vllm，我们是 ROCm。
核心逻辑只有几十行，自己实现更可控。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def fixed_scales(r_tot: int, k: int, alpha: float, gamma_max: float = 2.5) -> list[float]:
    """论文 Eq.(3)：s_k = (α/r_tot)·γ_k，γ_k 在 [1, gamma_max] 上线性等距。

    K=1 时退化成标准 LoRA 的 α/r，所以 gamma_max 只在 K>1 时才有意义。
    """
    if k <= 1:
        return [alpha / r_tot]
    step = (gamma_max - 1.0) / (k - 1)
    return [(alpha / r_tot) * (1.0 + step * i) for i in range(k)]


class SOSLoRALayer(nn.Module):
    """把一个 nn.Linear 换成 K 个静态专家的和。

    `base.weight` 会被冻结；只训 {A_k, B_k}。
    """

    def __init__(self, base: nn.Linear, r_tot: int = 8, k: int = 4,
                 alpha: float = 16.0, dropout: float = 0.05,
                 gamma_max: float = 2.5, ortho_init: bool = True):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"SOSLoRALayer 需要 nn.Linear，收到 {type(base)!r}")
        if k <= 0 or r_tot <= 0:
            raise ValueError(f"k 和 r_tot 必须为正，收到 k={k} r_tot={r_tot}")
        if r_tot % k != 0:
            raise ValueError(
                f"r_tot={r_tot} 必须能被 K={k} 整除（论文 rank_mode=total）；"
                f"否则每个专家的 rank 不是整数")

        self.base = base
        self.r_tot, self.k, self.alpha = r_tot, k, alpha
        self.r = r_tot // k                      # 每个专家的 rank
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.gamma_max = gamma_max

        for p in self.base.parameters():
            p.requires_grad_(False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scales = fixed_scales(r_tot, k, alpha, gamma_max)

        # ⚠️ dtype 和 device 都必须跟随底座，两个都踩过。
        #    dtype: A/B 默认 float32 而 Qwen 跑 bfloat16，前向直接
        #      `expected mat1 and mat2 to have the same dtype, got BFloat16 != float`。
        #    device: A/B 默认建在 CPU，而底座在 cuda —— 35B 上第 1 步就炸
        #      `Expected all tensors to be on the same device, but got mat2 is on cpu,
        #       different from other tensors on cuda:0`（sos_lora.py:110 的 self.B[k]）。
        #    这两个坑在**全 CPU 的单元测试里都照不出来** ——
        #    所以 scripts/_scratch/_test_sos_lora.py 里专门加了「底座放 cuda」那条断言。
        p_dtype = base.weight.dtype
        p_device = base.weight.device
        self.A = nn.ParameterList([
            nn.Parameter(torch.empty(self.r, self.in_features,
                                     dtype=p_dtype, device=p_device))
            for _ in range(k)])
        self.B = nn.ParameterList([
            nn.Parameter(torch.zeros(self.out_features, self.r,
                                     dtype=p_dtype, device=p_device))
            for _ in range(k)])

        self.reset_parameters(ortho_init)

    def reset_parameters(self, ortho_init: bool) -> None:
        """B 一律零初始化（论文："zero initial update"）—— 初始 ΔW=0，训练从原模型出发。

        A 用跨专家正交：把 K 个专家的 A 竖着拼成 (K·r, in)，做 thin QR 后切回去。
        这样不同专家的输入侧方向互相正交，起点就不在同一个子空间里。
        """
        if ortho_init:
            # thin QR: G (in, K*r) -> Q (in, K*r)，Q 的列正交。
            # G 也要建在底座那个设备上：建在 CPU 上 QR 的结果再 copy_ 回 cuda 虽然能跑
            # （copy_ 允许跨设备），但每层都要同步一次，190 层就是 760 次没必要的往返。
            dev = self.A[0].device
            G = torch.empty(self.in_features, self.k * self.r,
                            device=dev, dtype=torch.float32)
            nn.init.kaiming_uniform_(G, a=math.sqrt(5))
            q, _ = torch.linalg.qr(G, mode="reduced")        # (in, K*r)
            blocks = q.t().contiguous().view(self.k, self.r, self.in_features)
            for i in range(self.k):
                with torch.no_grad():
                    self.A[i].copy_(blocks[i])
        else:
            for a in self.A:
                nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        for b in self.B:
            nn.init.zeros_(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        xd = self.dropout(x)
        for k in range(self.k):
            out = out + self.scales[k] * ((xd @ self.A[k].t()) @ self.B[k].t())
        return out

    # ---- 正则：只罚**跨专家**的 A 方向相关 ----
    def ortho_penalty(self) -> torch.Tensor:
        """论文的轻量正则：惩罚不同专家 A 之间的相关。

        **不要先归一化。** 老版本写的是
            dirs = a / ||a||;  penalty = sum(cos²) / (k(k-1))
        两个除法把梯度压死了：A 的范数约 6.0，归一化让梯度除以 ||a||²≈36，
        再除 k(k-1)=12，合计 432 倍衰减。实测（step 5550）单层梯度范数
        2.0e-4，乘 coef=1.0 后对 A 的推动只有 4e-8/步 —— 相对 Adam 步长
        2e-4 是 0.02%，等于没接上。惩罚项的值也小到 2e-5（占 ce 的 0.007%）。

        现在直接罚**未归一化**的余弦平方：penalty = Σ_{k≠j} (a_k·a_j)² / (k(k-1))。
        这样梯度尺度与 A 自身量级一致（不再是 1/||a||²），coef 才是有意义的
        物理量。余弦仍由内积隐含 —— a_k·a_j 大就意味着方向接近。
        """
        if self.k <= 1:
            return torch.zeros((), device=self.A[0].device)
        flat = torch.stack([a.reshape(-1) for a in self.A], dim=0)   # [k, r*in]
        gram = flat @ flat.t()                                        # [k, k]
        offdiag = gram.masked_select(
            ~torch.eye(self.k, dtype=torch.bool, device=gram.device)
        )
        return offdiag.pow(2).sum() / (self.k * (self.k - 1))

    # ---- 合并：ΔW = Σ s_k A_k B_k ----
    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        dw = torch.zeros_like(self.base.weight)
        for k in range(self.k):
            dw = dw + self.scales[k] * (self.B[k] @ self.A[k])
        return self.base.weight + dw

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> str:
        return (f"SOSLoRALayer(r_tot={self.r_tot}, K={self.k}, r'={self.r}, "
                f"alpha={self.alpha}, gamma_max={self.gamma_max}, scales="
                f"[{', '.join('%.3f' % s for s in self.scales)}], "
                f"trainable={self.n_trainable():,})")


def sos_lora_penalty(model: nn.Module) -> torch.Tensor:
    """整个模型的跨专家正交正则，**按层平均**后加进 loss。

    老版本是 `total += p` 对 190 个 SOS 层求和，再乘 coef —— 于是同一个
    coef 在不同 preset（190 / 310 层）下含义完全不同，调参没有可比性。
    改成按层平均：coef 就是「单层惩罚项的权重」，与层数无关。
    """
    total = None
    n = 0
    for m in model.modules():
        if isinstance(m, SOSLoRALayer) and m.k > 1:
            p = m.ortho_penalty()
            total = p if total is None else total + p
            n += 1
    if total is None:
        dev = next(model.parameters()).device
        return torch.zeros((), device=dev)
    return total / n


def count_sos_layers(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, SOSLoRALayer))


def replace_linear_with_sos(module: nn.Module, suffixes: tuple[str, ...],
                            r_tot: int, k: int, alpha: float,
                            dropout: float = 0.05, gamma_max: float = 2.5,
                            prefix: str = "") -> list[str]:
    """把名字以 `suffixes` 之一结尾的 nn.Linear 换成 SOSLoRALayer。

    返回被替换的模块名，便于核对（**必须核对** —— 静默没替换等于没加 LoRA）。
    """
    replaced: list[str] = []
    for name, child in list(module.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and any(name.endswith(s) for s in suffixes):
            setattr(module, name, SOSLoRALayer(child, r_tot=r_tot, k=k, alpha=alpha,
                                               dropout=dropout, gamma_max=gamma_max))
            replaced.append(full)
        else:
            replaced.extend(replace_linear_with_sos(
                child, suffixes, r_tot, k, alpha, dropout, gamma_max, full))
    return replaced


def merge_sos_layers(model: nn.Module) -> int:
    """把所有 SOSLoRALayer 的 ΔW = Σ s_k A_k B_k 烘进底座的 weight，返回合并了几层。

    这是论文的核心卖点：合并后推理端**不需要认识 SOS** —— 结构就是原来的 nn.Linear，
    没有额外参数、没有额外延迟（arXiv 2607.16252v1 §1）。
    副作用是原地改结构，所以合并后不能再训练；存 checkpoint 前调用正合适。
    """
    n = 0
    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, SOSLoRALayer):
                with torch.no_grad():
                    child.base.weight.copy_(child.merged_weight())
                setattr(module, name, child.base)
                n += 1
    return n


def freeze_except_sos(model: nn.Module) -> int:
    """把除了 SOS 的 A/B 之外的参数全部冻住，返回被冻掉的参数量。

    为什么需要：标准 LoRA 那条路有 peft 兜底（_freeze_except_lora），
    SOS 这条是我们自己实现的，必须自己保证「底座真的冻住了」——
    否则会静默变成全量微调，而 loss 曲线看起来完全正常。
    返回 >0 说明确实有该冻的东西被冻了；返回 0 也要留意是不是基座本来就没开梯度。
    """
    keep = set()
    for m in model.modules():
        if isinstance(m, SOSLoRALayer):
            for a in m.A:
                keep.add(id(a))
            for b in m.B:
                keep.add(id(b))
    n = 0
    for p in model.parameters():
        if id(p) not in keep and p.requires_grad:
            p.requires_grad_(False)
            n += p.numel()
    return n


def sos_trainable_params(model: nn.Module) -> int:
    """SOS 那部分的可训参数总量（A + B），用于和同预算 LoRA 对比。"""
    return sum(p.numel() for m in model.modules()
               if isinstance(m, SOSLoRALayer)
               for p in (list(m.A) + list(m.B)))
