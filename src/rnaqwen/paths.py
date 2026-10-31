"""集中管理默认路径 + 轨道归属。

为什么要集中：原先 `models/rnafm` 硬编码在 8 个地方（6 个脚本的 argparse 默认值 +
`Stage1Config.encoder_path` + `Stage2Config.encoder_path`）。
换 backbone 时要改 8 处、且很容易漏一处 —— 漏了就是"以为在用 A、实际在用 B"的静默错配。

--- 2026-10-09 决定（用户拍板）---

**改用 ERNIE-RNA 作为 backbone。** 理由是它是结构增强的（带 attention bias /
pairwise 表示），性能显著强于 BERT 基座的 RNA-FM。

⚠️ **这是 E 轨（改进轨）配置，不是 R 轨。**
   PLAN §7：R 轨 = 照抄论文（RNA-FM 640 维），E 轨 = 改进，一次只动一个变量。
   论文用的是 RNA-FM。**用 ERNIE-RNA 跑出来的任何数字都必须标 E 轨，
   绝对不能和论文的分数并列汇报。**
   R 轨要复现时，显式设 `RNAQWEN_ENCODER=models/rnafm` 或命令行 `--encoder models/rnafm`。

覆盖方式（优先级从高到低）：
    1) 命令行 --encoder
    2) 环境变量 RNAQWEN_ENCODER
    3) 这里的默认值
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# R 轨（论文原样）用的 backbone
R_TRACK_ENCODER = "models/rnafm"
# E 轨（当前实际在用）的 backbone
E_TRACK_ENCODER = "models/ernierna"

_ENV_VAR = "RNAQWEN_ENCODER"


def default_encoder() -> str:
    """解析默认 encoder 路径。"""
    return os.environ.get(_ENV_VAR, E_TRACK_ENCODER)


def track_of(encoder_path: str) -> str:
    """这个 encoder 属于哪条轨。用于在报告里标注，防止 R/E 混报。"""
    p = str(encoder_path).replace("\\", "/").rstrip("/")
    if p.endswith(R_TRACK_ENCODER):
        return "R"
    if p.endswith(E_TRACK_ENCODER):
        return "E"
    return "E?"          # 不认识的一律当 E 轨，宁可保守


def describe_encoder(encoder_path: str) -> str:
    """一行说明，脚本启动时打出来 —— 让"实际在用哪个"永远不含糊。"""
    from rnaqwen.models.encoder import read_model_type

    mt = read_model_type(encoder_path)
    tr = track_of(encoder_path)
    note = {
        "R": "R 轨（论文原样，可与论文并列）",
        "E": "E 轨（改进轨，**不可**与论文/ R 轨并列汇报）",
    }.get(tr, "E? 轨（未知 backbone，按改进轨处理）")
    return f"[encoder] {encoder_path}  arch={mt or '?'}  {note}"


def resolve(path: str | Path) -> Path:
    """把相对路径按仓库根解析成绝对路径。"""
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p)
