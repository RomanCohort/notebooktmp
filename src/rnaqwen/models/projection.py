"""投影层：640 → 4096。

**维度为什么不改**
------------------
原版是 `Llama-3-8B（hidden=4096）`，投影是 `RNA-FM(640) → 4096` 的线性层。
本复现换 `Qwen3-8B`，其 `hidden_size` **同样是 4096**（已核对本地 config.json），
所以这条投影的 in/out 维度与原版逐位相同 —— 换底座这件事**没有污染投影层这个变量**。

（这正是否掉 Qwen2.5-7B 的原因：它是 3584，会强迫改投影，等于多引入一个自由度。）
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearProjection(nn.Module):
    """就是原版那个线性投影，没有花活。

    初始化：原版没写；这里用 nn.Linear 默认（kaiming uniform）。
    `scaled_init` 备选放在那儿，是为了万一投影塌缩时可以换成小方差初始化对照 ——
    但**默认路径必须与原版一致**，别自己加戏。
    """

    def __init__(self, in_dim: int = 640, out_dim: int = 4096, bias: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.proj = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)

    def describe(self) -> str:
        return (f"LinearProjection({self.in_dim} -> {self.out_dim}, bias={self.proj.bias is not None}, "
                f"params={sum(p.numel() for p in self.parameters()):,})")
