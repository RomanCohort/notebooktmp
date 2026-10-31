"""RNA-GPT 复现工程（Milivia / 2026-10-08 起）。

模块布局：
    rnaqwen.env      本机（AMD ROCm Windows）环境修补
    rnaqwen.eval     评测：ROUGE-1/2/L + 三种嵌入的 semantic P/R/F1
    rnaqwen.data     RNA-QA 重建管线
    rnaqwen.models   编码器 / 投影层 / 两阶段训练

⚠️ import 本包时**自动**装载 `rnaqwen.env.shim`。
   本机 torch（ROCm Windows nightly）没编入分布式后端，而 danling / transformers 5
   在 import 阶段就引用 `torch.distributed.*`，不打补丁连 `AutoModel` 都 import 不了。
   详见 `rnaqwen/env/shim.py` 顶部注释。
"""

__version__ = "0.0.1"

from rnaqwen.env import shim as _shim  # noqa: E402,F401  —— 必须最早执行

__all__ = ["__version__", "_shim"]
