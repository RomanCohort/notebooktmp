"""RNA token 压缩模块 —— **E 轨（改进轨）**，不属于复现轨。

为什么需要它（算力瓶颈）
------------------------
现在每个碱基一个 token：729 nt 的 RNA 经 RNA-FM 得到 **731** 个 token（含 BOS/EOS），
投影后直接铺进 `inputs_embeds`。Stage 1 的一条样本里：

    731 个 RNA 软 token  +  368 个答案 token  ≈ 1099
            ↑ 66%

而 RNA-FM 的输出在信息上高度冗余（相邻残基的表征几乎一样）。多模态领域早就把这件事
做掉了：Qwen-VL 用 resampler、MiniGPT-4 用 Q-Former、LLaVA 用 MLP 投影。
把 RNA 压到 32–64 个 latent token，序列长度直接短 2.5 倍 → 训练/推理都快 ~2.5 倍
（线性层 O(n)，SDPA 下注意力也接近线性主导）。

**边界声明**：原版 RNA-GPT 是"每个 token 都投影"。本模块是 E 轨新增，
必须挂在 `Stage1Config.compress_latent` 开关后面；`compress_latent=None` 时
`alignment.py` 一行都不走这里，复现轨（R 轨）行为逐位不变。

三个 method
-----------
`resampler`  Perceiver / Q-Former 式：n_latent 个可学习 query + 多层 cross-attention。
             容量最大，唯一能"学会看哪些残基"的方法。
`strided`    depthwise 1D 卷积（平滑）+ 分块池化（stride ≈ L_valid / n_latent）。便宜、可解释。
`meanpool`   按块平均池化。0 参数，下界基线。

⚠️ padding 的坑（本文件里大半篇幅在防它）
----------------------------------------
RNA-FM 的输出里 **padding 位置是有值的（不是 0）**。如果压缩器没有正确屏蔽 padding：
  * 它会把 padding 当真实信号学进去；
  * **训练 loss 照常下降**（答案那边没问题），你几乎发现不了；
  * 而且训练/推理的 padding 长度一变，表征就漂移。
所以本模块做两件事：
  1. 所有聚合（attention / 卷积 / 池化）都用 mask 加权，padding 的贡献**恒为 0**；
  2. 额外做一个 `trim_padding` 快路径：mask 是右填充（RNA-FM tokenizer 的行为）时，
     先砍掉"所有行都是 padding"的尾列。数学上等价（那些列对输出贡献恒为 0），
     但让结果与 padding 长度**逐位无关**（不是"近似无关"，是 bitwise 相同）。
     `scripts/_test_compression.py` 的第 2 项就是验这个。

用法
----
    from rnaqwen.models.compression import RnaTokenCompressor
    comp = RnaTokenCompressor(in_dim=640, n_latent=64, method="resampler")
    lat, lat_mask = comp(hidden, mask)      # (B, 64, 640), (B, 64) 全 1
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["RnaTokenCompressor", "METHODS", "masked_bin_pool", "trim_padding"]

METHODS = ("resampler", "strided", "meanpool")

# resampler 的默认内部宽度。
# 为什么不是 640：本工程只训"投影 + 压缩器"，数据 3,800 条，单卡 iGPU。
# d_model=640/2 层会把可训参数从 2.6M 顶到 ~6.6M（2.5 倍投影）；
# d_model=256/2 层 ≈ 1.4M，量级与投影相当，是能训得动的区间。
# （完整参数量对照见 reports/token_compression_test.json）
DEFAULT_D_MODEL = 256


# --------------------------------------------------------------------------- #
# mask 工具
# --------------------------------------------------------------------------- #
def _valid_len(mask: torch.Tensor) -> torch.Tensor:
    """(B, L) -> (B,) 每条序列的有效长度（mask 转成 0/1 再求和）。"""
    return (mask > 0).sum(-1)


def _is_right_padded(mask: torch.Tensor) -> bool:
    """mask 是否形如 `[1]*Lb + [0]*(L-Lb)`（右填充）。

    RNA-FM（HF tokenizer `padding=True`）与 Qwen tokenizer 都是右填充，
    所以正常路径恒为 True；非右填充时只是退化成"不做 trim"，mask 仍然生效。
    """
    if mask.shape[1] == 0:
        return False
    idx = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
    return bool(torch.equal(mask > 0, idx < _valid_len(mask).unsqueeze(-1)))


def trim_padding(hidden: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """砍掉"所有行都是 padding"的尾列。

    数学上不改结果（这些列在下面每个聚合里权重都是 0），但把"与 padding 长度无关"
    从"数值上近似成立"变成"bitwise 成立" —— 因为送进 attention / conv 的张量
    在两个 padding 长度下**就是同一个张量**。
    顺带省算力：batch 里最长序列 731、padding 到 1024 时，attention 的 K/V 短 30%。
    """
    if hidden.shape[1] == 0 or not _is_right_padded(mask):
        return hidden, mask
    lmax = int(_valid_len(mask).max().item())
    lmax = max(lmax, 1)                       # 防御：空 mask 行不该出现，但别让它炸
    if lmax < hidden.shape[1]:
        return hidden[:, :lmax], mask[:, :lmax]
    return hidden, mask


def _safe_valid(mask: torch.Tensor) -> torch.Tensor:
    """转成 bool，并保证每行至少有一个有效位（否则 attention 的 softmax 全是 NaN）。

    正常输入（RNA-FM 每条序列至少 BOS/EOS 两个有效 token）永远走不到兜底分支。
    """
    valid = mask > 0
    empty = ~valid.any(dim=-1)
    if bool(empty.any()):
        valid = valid.clone()
        valid[empty, 0] = True
    return valid


def masked_bin_pool(hidden: torch.Tensor, mask: torch.Tensor, n_latent: int) -> torch.Tensor:
    """按**有效位置**等分成 n_latent 块，块内做 mask 加权平均 -> (B, n_latent, D)。

    分块边界由"有效长度"决定，不由 padding 后的长度决定 —— 这是 padding 无关的关键。
    padding 位置乘 0 后计入累加器，加 0.0 在浮点上是精确的，所以 padding 长度改变
    不会动结果的最低有效位。
    """
    b, l, d = hidden.shape
    w = (mask > 0).to(hidden.dtype).unsqueeze(-1)                    # (B, L, 1)
    lengths = _valid_len(mask).clamp(min=1)                          # (B,)
    idx = torch.arange(l, device=hidden.device).unsqueeze(0)         # (1, L)
    bin_idx = torch.div(idx * n_latent, lengths.unsqueeze(-1), rounding_mode="floor")
    bin_idx = bin_idx.clamp_(max=n_latent - 1)                       # (B, L)

    num = hidden.new_zeros(b, n_latent, d)
    den = hidden.new_zeros(b, n_latent)
    num.scatter_add_(1, bin_idx.unsqueeze(-1).expand(b, l, d), hidden * w)
    den.scatter_add_(1, bin_idx, (mask > 0).to(hidden.dtype))
    out = num / den.clamp(min=1.0).unsqueeze(-1)

    empty = den <= 0                                                 # 有效长度 < n_latent 时的空桶
    if bool(empty.any()):
        row_mean = (hidden * w).sum(1, keepdim=True) / lengths.to(hidden.dtype).view(b, 1, 1)
        out = torch.where(empty.unsqueeze(-1), row_mean.expand_as(out), out)
    return out


# --------------------------------------------------------------------------- #
# 三个 core
# --------------------------------------------------------------------------- #
class _MeanPool(nn.Module):
    """块平均池化：0 参数。下界基线，同时也是"压缩本身就是全部信息损失"的对照组。"""

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor, n_latent: int) -> torch.Tensor:
        return masked_bin_pool(hidden, mask, n_latent)


class _StridedConvPool(nn.Module):
    """depthwise 1D 卷积（局部平滑）+ 分块池化。

    卷积那一步是 **mask 归一化** 的重心平均：

        y = conv(h ⊙ m) / conv(m)

    h ⊙ m 把 padding 清零，conv(m) 是窗口内有效点的个数 —— 相除正好是"窗口内有效
    邻居的加权平均"。padding 位置对分子分母的贡献都是 0，所以 padding 长度与内容
    都不影响结果。

    初始化成 `1/kernel` 的均值滤波（等价于移动平均），所以 step 0 的行为是可解释的
    "局部去噪平均"，不是随机卷积。
    """

    def __init__(self, in_dim: int, kernel: int = 5) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.kernel = int(kernel)
        self.pad = self.kernel // 2
        if self.kernel > 1:
            conv = nn.Conv1d(in_dim, in_dim, self.kernel, padding=self.pad, groups=in_dim, bias=False)
            with torch.no_grad():
                conv.weight.fill_(1.0 / self.kernel)
            self.conv = conv
        else:
            self.conv = None
        self.register_buffer("ones", torch.ones(1, 1, max(self.kernel, 1)), persistent=False)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor, n_latent: int) -> torch.Tensor:
        if self.conv is not None:
            h = hidden.transpose(1, 2)                                   # (B, D, L)
            m = (mask > 0).to(hidden.dtype).unsqueeze(1)                 # (B, 1, L)
            num = F.conv1d(h * m, self.conv.weight, padding=self.pad, groups=self.in_dim)
            den = F.conv1d(m, self.ones.to(m.dtype), padding=self.pad)   # 窗口内有效点个数
            h = (num / den.clamp(min=1.0)).transpose(1, 2)
            hidden = h
        return masked_bin_pool(hidden, mask, n_latent)


class _CrossBlock(nn.Module):
    """一层 cross-attention + FFN（latent 侧有残差，K/V 侧是 RNA 表征本身）。"""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: float, dropout: float,
                 zero_init_q: bool) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} 必须能被 n_heads={n_heads} 整除")
        self.h = int(n_heads)
        self.dh = d_model // self.h
        self.scale = self.dh ** -0.5

        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.to_q = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)
        self.to_out = nn.Linear(d_model, d_model)
        if zero_init_q:
            # to_q 置零 => attention logits 全 0 => 均匀注意力 = 对有效位置做平均池化。
            # 于是 resampler 在 step 0 就"退化成 meanpool"，冻结的 LLM 一上来看到的就是
            # 真实 RNA 信号，而不是随机 query 招出来的噪声；训练中梯度会自然把均匀解打破。
            nn.init.zeros_(self.to_q.weight)
            nn.init.zeros_(self.to_q.bias)

        self.ln_mlp = nn.LayerNorm(d_model)
        hid = max(int(round(d_model * ffn_mult)), 1)
        self.mlp = nn.Sequential(nn.Linear(d_model, hid), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hid, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        b, n, d = q.shape
        l = kv.shape[1]
        qq = self.to_q(self.ln_q(q)).view(b, n, self.h, self.dh).transpose(1, 2)   # (B,h,N,dh)
        kvn = self.ln_kv(kv)
        kk = self.to_k(kvn).view(b, l, self.h, self.dh).transpose(1, 2)            # (B,h,L,dh)
        vv = self.to_v(kvn).view(b, l, self.h, self.dh).transpose(1, 2)

        attn = (qq @ kk.transpose(-2, -1)) * self.scale                            # (B,h,N,L)
        # 屏蔽 padding：用 finfo.min 而不是 -inf —— 逐行减去最大值后 exp(finfo.min - max)
        # 精确下溢到 0，padding 的权重**恰好是 0**，不会污染分母（也不会制造 NaN）。
        attn = attn.masked_fill(~valid[:, None, None, :], torch.finfo(attn.dtype).min)
        attn = self.drop(attn.softmax(dim=-1))
        o = (attn @ vv).transpose(1, 2).reshape(b, n, d)

        q = q + self.drop(self.to_out(o))
        q = q + self.mlp(self.ln_mlp(q))
        return q


class _Resampler(nn.Module):
    """Perceiver / Q-Former 式 resampler：可学习 query + 多层 cross-attention。"""

    def __init__(self, in_dim: int, n_latent: int, d_model: int, n_heads: int, n_layers: int,
                 ffn_mult: float, dropout: float, zero_init_q: bool, query_init_std: float) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model) if d_model != in_dim else nn.Identity()
        self.out_proj = nn.Linear(d_model, in_dim) if d_model != in_dim else nn.Identity()
        self.queries = nn.Parameter(torch.randn(n_latent, d_model) * query_init_std)
        self.query_ln = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList([
            _CrossBlock(d_model, n_heads, ffn_mult, dropout, zero_init_q) for _ in range(n_layers)
        ])
        self.out_ln = nn.LayerNorm(d_model)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor, n_latent: int) -> torch.Tensor:
        b = hidden.shape[0]
        kv = self.in_proj(hidden)                                  # (B, L, d_model)
        valid = _safe_valid(mask)                                  # (B, L) bool
        q = self.query_ln(self.queries).unsqueeze(0).expand(b, -1, -1).contiguous()
        for blk in self.blocks:
            q = blk(q, kv, valid)
        return self.out_proj(self.out_ln(q))                       # (B, n_latent, in_dim)


# --------------------------------------------------------------------------- #
# 对外接口
# --------------------------------------------------------------------------- #
class RnaTokenCompressor(nn.Module):
    """RNA-FM 输出 -> 定长 latent、可插拔的 token 压缩器。

    接口固定为 `forward(hidden, mask) -> (latent, latent_mask)`：
        hidden (B, L, in_dim)   RNA-FM 的 last_hidden_state（padding 位置**不是 0**）
        mask   (B, L)           1 = 有效；形状必须是右填充或能被 mask 正确描述
        -> latent (B, n_latent, in_dim)  与 hidden 同宽度，后面直接接 640→4096 的投影
        -> latent_mask (B, n_latent)     全 1（latent 是定长的，没有 padding 概念）

    放在 `encoder` 与 `projection` 之间：RNA-FM → compressor → Linear(640→4096)。
    可训参数属于"投影那一侧"（编码器与 LLM 依然冻结）。
    """

    def __init__(self, in_dim: int = 640, n_latent: int = 64, method: str = "resampler",
                 *, n_heads: int = 8, n_layers: int = 2, ffn_mult: float = 2.0,
                 d_model: int | None = None, dropout: float = 0.0, conv_kernel: int = 5,
                 trim: bool = True, zero_init_q: bool = True, query_init_std: float = 0.02,
                 dtype: torch.dtype | None = None) -> None:
        super().__init__()
        method = str(method).lower()
        if method not in METHODS:
            raise ValueError(f"未知 method={method!r}，可选 {METHODS}")
        if int(n_latent) < 1:
            raise ValueError("n_latent 必须 >= 1")
        self.in_dim = int(in_dim)
        self.n_latent = int(n_latent)
        self.method = method
        self.trim = bool(trim)

        if method == "meanpool":
            self.core: nn.Module = _MeanPool()
        elif method == "strided":
            self.core = _StridedConvPool(self.in_dim, kernel=int(conv_kernel))
        else:
            dm = int(d_model) if d_model is not None else min(self.in_dim, DEFAULT_D_MODEL)
            self.core = _Resampler(self.in_dim, self.n_latent, dm, int(n_heads), int(n_layers),
                                   float(ffn_mult), float(dropout), bool(zero_init_q),
                                   float(query_init_std))
        if dtype is not None:
            self.to(dtype)

    # ---- 属性 ----
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_breakdown(self) -> dict[str, int]:
        return {name: sum(p.numel() for p in mod.parameters())
                for name, mod in self.core.named_children()}

    def describe(self) -> str:
        return (f"RnaTokenCompressor(method={self.method}, {self.in_dim} -> {self.n_latent} latent, "
                f"params={self.num_parameters:,}, trim={self.trim})")

    def extra_repr(self) -> str:  # pragma: no cover - 只为打印好看
        return f"method={self.method}, n_latent={self.n_latent}, params={self.num_parameters:,}"

    # ---- 前向 ----
    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden.dim() != 3:
            raise ValueError(f"hidden 期望 (B, L, D)，收到 {tuple(hidden.shape)}")
        if hidden.shape[:2] != mask.shape[:2]:
            raise ValueError(f"hidden {tuple(hidden.shape)} 与 mask {tuple(mask.shape)} 的前两维必须一致")
        if hidden.shape[-1] != self.in_dim:
            raise ValueError(f"hidden 最后一维 {hidden.shape[-1]} != in_dim {self.in_dim}")

        h, m = (trim_padding(hidden, mask) if self.trim else (hidden, mask))
        latent = self.core(h, m, self.n_latent)
        out_mask = torch.ones((hidden.shape[0], self.n_latent), dtype=mask.dtype, device=mask.device)
        return latent, out_mask


def _selfcheck() -> None:  # pragma: no cover - 手动跑用
    torch.manual_seed(0)
    for method in METHODS:
        comp = RnaTokenCompressor(640, 64, method)
        h = torch.randn(2, 731, 640)
        m = torch.ones(2, 731)
        m[1, 500:] = 0
        out, om = comp(h, m)
        print(f"{comp.describe()}  ->  {tuple(out.shape)} {tuple(om.shape)} "
              f"mask_sum={float(om.sum())}")


if __name__ == "__main__":  # pragma: no cover
    _selfcheck()
