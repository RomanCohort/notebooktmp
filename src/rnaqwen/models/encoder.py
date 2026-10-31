"""RNA 编码器（冻结）—— 支持多个 backbone。

支持哪些（全部出自 `multimolecule`，同一个移植体系）：

    arch       类名             hidden   vocab   pos     checkpoint
    ---------- ---------------- -------- ------- ------- -----------------------------
    rnafm      RnaFmModel       640      28      1026    multimolecule/rnafm
    ernierna   ErnieRnaModel    768      28      1026    multimolecule/ernierna
    rnaernie   RnaErnieModel    768      28      1026    multimolecule/rnaernie

**默认自动识别**：读 `model_path/config.json` 的 `model_type` 决定用哪个类。
所以调用方不用改，把路径指到 `models/ernierna` 就自动走 ernierna。
也可以 `arch="ernierna"` 显式指定。

⚠️ 可用长度 **1022 nt**（1022 nt → 1024 token 能过；1023 nt → 1025 token 直接
   `IndexError`）。`scripts/01b_probe_rnafm_limit.py` 逐长度逼过边界。
   config 里的 1026 和 `position_ids` 缓冲区的 1026 都对得上，但挡住你的是 1024 这个
   token 数上限。ernierna/rnaernie 的 config 同样是 1026，但**换 backbone 后必须重跑一次
   那个边界探针**，不要假设它一样。

⚠️ 2026-10-09 事故：`models/rnafm/model.safetensors` 曾因 aria2 预分配留下零填充空洞，
   205 个张量里 155 个全零，`word_embeddings` 全零 ⇒ encoder 对序列**零敏感**，而
   `from_pretrained` 不报错。**换任何 backbone 之前先跑 `scripts/01c_check_encoder_sanity.py`。**

用法：
    import rnaqwen                 # import 本包即自动装载 env.shim
    enc = RnaEncoder("models/rnafm")          # 自动识别 rnafm
    enc = RnaEncoder("models/ernierna")       # 自动识别 ernierna
    enc = RnaEncoder("models/ernierna", arch="ernierna")   # 显式
    h = enc(["ACGU...", "GGGU..."])  # -> EncoderOutput(hidden=(B,L,H), mask, lengths)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

MAX_NT = 1022  # 实测上限 = 1024 token（含 BOS/EOS）


@dataclass(frozen=True)
class EncoderArch:
    """一个 backbone 的加载信息。"""
    name: str
    module: str                 # 从哪个模块导入
    model_cls: str              # 拿哪个类
    hidden_size: int            # 期望的 hidden（会和 config 对一次，对不上就报错）
    max_nt: int


ARCHS: dict[str, EncoderArch] = {
    "rnafm": EncoderArch("rnafm", "multimolecule.models.rnafm",
                         "RnaFmModel", 640, MAX_NT),
    # ERNIE-RNA 的 max_nt 是**实测**定下来的，不是抄 rnafm 的：
    #   * `position_embedding_type = 'sinusoidal'` —— 位置编码是算出来的，
    #     **没有** RNA-FM 那种固定大小的 position 表，所以不存在 1022 的硬上限；
    #     `scripts/01b_probe_rnafm_limit.py` 实测 1030 nt（1032 token）也能过。
    #   * 但 config 写 `max_position_embeddings = 1026`，那是**预训练窗口**。
    #     超出去是外推，质量没有保证。所以上限取预训练范围：1026 token − BOS/EOS = 1024 nt。
    #   * 现有数据是按 RNA-FM 的 1022 过滤的，所以这个数目前不会真的被用到。
    "ernierna": EncoderArch("ernierna", "multimolecule.models.ernierna",
                            "ErnieRnaModel", 768, 1024),
    # ⚠️ 未实测：multimolecule 的 rnaernie 还没跑过 01b 边界探针。保守用 1022，
    #    真要换过去之前必须先跑探针。
    "rnaernie": EncoderArch("rnaernie", "multimolecule.models.rnaernie",
                            "RnaErnieModel", 768, MAX_NT),
}


def read_model_type(model_path: str | Path) -> str | None:
    """读 config.json 的 model_type。读不到返回 None。"""
    cfg = Path(model_path) / "config.json"
    if not cfg.exists():
        return None
    try:
        return json.loads(cfg.read_text(encoding="utf-8")).get("model_type")
    except Exception:
        return None


def resolve_arch(model_path: str | Path, arch: str | None = None) -> EncoderArch:
    """决定用哪个 backbone。

    arch 给了就用它（但会核对 config.json 的 model_type，不一致直接报错 —— 
    防止"路径指向 rnafm、却按 ernierna 加载"这种会静默跑歪的错配）。
    arch 没给就按 config.json 的 model_type 自动识别。
    """
    mt = read_model_type(model_path)
    if arch is None:
        if mt is None:
            raise ValueError(
                f"{model_path}/config.json 读不到 model_type，无法自动识别 backbone。"
                f"请显式传 arch={'/'.join(ARCHS)}")
        if mt not in ARCHS:
            raise ValueError(
                f"{model_path} 的 model_type={mt!r} 不在支持的列表里：{list(ARCHS)}")
        return ARCHS[mt]

    if arch not in ARCHS:
        raise ValueError(f"未知 arch={arch!r}，可选：{list(ARCHS)}")
    a = ARCHS[arch]
    if mt is not None and mt != arch:
        raise ValueError(
            f"arch={arch!r} 与 {model_path}/config.json 里的 model_type={mt!r} 不一致。"
            f"路径和架构指错了会静默加载成另一个模型，拒绝继续。")
    return a


@dataclass
class EncoderOutput:
    hidden: torch.Tensor          # (B, L, H)
    mask: torch.Tensor            # (B, L) 1=有效
    lengths: torch.Tensor         # (B,)


class RnaEncoder(nn.Module):
    """包一层，把「冻结 + 批处理 + 定长截断 + 多 backbone」固定下来。

    冻结是硬约束：PLAN §1 原版就是冻结编码器只训投影。
    """

    def __init__(self, model_path: str, arch: str | None = None,
                 device: str | None = None, dtype: torch.dtype | None = None,
                 max_nt: int | None = None):
        super().__init__()
        self.arch = resolve_arch(model_path, arch)
        self.model_path = str(model_path)
        self.max_nt = max_nt or self.arch.max_nt

        import importlib

        mod = importlib.import_module(self.arch.module)  # 延迟导入
        model_cls = getattr(mod, self.arch.model_cls)
        tokenizer_cls = getattr(mod, "RnaTokenizer")

        self.tokenizer = tokenizer_cls.from_pretrained(str(model_path))
        model = model_cls.from_pretrained(str(model_path))
        if dtype is not None:
            model = model.to(dtype)
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.hidden_size = int(self.model.config.hidden_size)
        if self.hidden_size != self.arch.hidden_size:
            raise ValueError(
                f"{self.arch.name} 期望 hidden_size={self.arch.hidden_size}，"
                f"但 {model_path} 加载出来是 {self.hidden_size}。"
                f"checkpoint 和 arch 不匹配，拒绝继续。")
        self._explicit_device = device

    def describe(self) -> str:
        return (f"RnaEncoder(arch={self.arch.name}, hidden={self.hidden_size}, "
                f"max_nt={self.max_nt}, path={self.model_path})")

    # ⚠️ `device` 必须是**属性**，不能靠覆写 `.to()` 去维护一个普通字段。
    #
    # 踩过的坑（真踩了）：原来 `self.device` 是个普通属性、只在 `.to()` 覆写里更新。
    # 但 `nn.Module.to()` 用的是内部 `_apply()` 递归搬参数，**根本不会调用子模块的 `.to()` 覆写**。
    # 所以只要不是"有人显式地 `encoder.to(device)`"（`train_stage1` 恰好这么写了），
    # 这个字段就一直是 None → tokenizer 的输出留在 CPU、权重在 CUDA →
    #   `RuntimeError: Expected all tensors to be on the same device, but got index is on cpu`
    # 任何走"父模块 `.to(device)`"的代码路径都会踩到（比如批量扫描脚本）。
    @property
    def device(self) -> torch.device | None:
        if self._explicit_device is not None:
            return torch.device(self._explicit_device)
        try:
            return next(self.model.parameters()).device
        except StopIteration:  # pragma: no cover
            return None

    @device.setter
    def device(self, value) -> None:
        self._explicit_device = value

    @property
    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    @torch.no_grad()
    def forward(self, sequences: list[str] | tuple[str, ...]) -> EncoderOutput:
        enc = self.tokenizer(
            list(sequences),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_nt,   # 特殊 token 由 tokenizer 自己加
        )
        if self.device is not None:
            enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc)
        return EncoderOutput(
            hidden=out.last_hidden_state,
            mask=enc["attention_mask"],
            lengths=enc["attention_mask"].sum(-1),
        )


# 向后兼容：老代码 import 的是 RnaFmEncoder。它现在只是 RnaEncoder 的别名。
RnaFmEncoder = RnaEncoder


def zero_probe(encoder: RnaEncoder, sequences: list[str]) -> EncoderOutput:
    """PLAN §3 P1 的验收手段：把编码器输出置零，看下游指标掉不掉。

    掉得明显 = 模态真的用上了；
    不掉     = 模型绕过了编码器（投影塌缩 / 抄文本），必须回头查。
    """
    out = encoder(sequences)
    return EncoderOutput(hidden=torch.zeros_like(out.hidden), mask=out.mask, lengths=out.lengths)
