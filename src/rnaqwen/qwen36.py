"""Qwen3.6-35B-A3B 的流式装载 + 专家量化（可调用模块）。

从 scripts/84_qwen36_quant.py 抽出来，因为训练脚本 (50_) 也要用它。
两份实现必然漂移，所以只留这一份。

为什么需要这些东西（全部是实测结论，见 PLAN §14）
-------------------------------------------------
1. **必须走 VLM 类。** checkpoint 的 key 是 `model.language_model.layers.*`，
   只有 `Qwen3_5MoeForConditionalGeneration` 认识这个前缀。
   用 `AutoModelForCausalLM`（= 文本-only 类）会静默装出一堆随机权重：
   `load_state_dict` 把 690 个文本 key 全报成 unexpected，模型留在 meta 上。

2. **专家是 3D `nn.Parameter`，bnb 量化不到。**
       experts.gate_up_proj [256, 1024, 2048]
       experts.down_proj    [256, 2048, 512]
   41 层合计 33.01 B，占 text 的 96.7%。拆成 per-expert `nn.Linear`
   （20,992 个）之后 bnb 才能处理。forward 与官方实现**逐位等价**
   （scripts/85_moe_forward_equiv.py 验过，含反向）。

3. **量化必须先于搬到显卡。** bnb 的 `Int8Params.to()`：
       if not is_quantized and device is not None and self.data.device.type == "cpu":
           return self._quantize(device)
   —— 只有数据还在 CPU 上时才量化。先 `.to(dev)` 再换 Linear，
   量化**静默不发生**，权重原样 bf16 留着。

4. **meta buffer 要重建。** `rotary_emb.inv_freq` 是 `__init__` 里算的、
   不在权重文件里，`strict=False` 不碰它，它会一直停在 meta，
   只在第一次前向或 `.to()` 时炸。

5. **内存要看显存计数器，不要看系统"空闲内存"。**
   每层要从磁盘读约 1.7 GB 的 bf16 张量，那部分进 page cache，
   而 page cache 也算进"空闲内存"。实测：量化一个 Linear 只让系统掉
   0.16 GiB、进程 RSS 没变，但每层读数会掉 1.5~2.5 GiB。
"""
from __future__ import annotations

import gc
import json
import os
import re
import subprocess
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_QWEN36 = Path(r"C:\qwen-api\models\Qwen3.6-35B-A3B")
if not _DEFAULT_QWEN36.exists():
    _DEFAULT_QWEN36 = _REPO_ROOT / "models" / "Qwen3.6-35B-A3B"

# Colab/Linux must not depend on the Windows checkout layout.
MDL = Path(os.environ.get("RNAQ_QWEN36_DIR", str(_DEFAULT_QWEN36)))
# 8B 在仓库内（50_ 里的 QWEN 就是它）。parents: [0]=rnaqwen [1]=src [2]=RNAQwen
QWEN8 = Path(os.environ.get(
    "RNAQ_QWEN8_DIR", str(_REPO_ROOT / "models" / "Qwen3-8B")
))

# 层内张量名 -> 层号。用来算每个分片覆盖的层范围（见 ShardReader.trim）。
_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")

# 打开它会逐张量打印 `[get] 分片 :: 名字`。原生崩溃（0xC0000005）没有 Python
# 栈，只能靠"最后打印的是谁"定位 —— 2026-10-14 就是靠这个把崩溃点锁到具体张量的。
_TRACE_GET = os.environ.get("RNAQWEN_TRACE_GET") == "1"


def base_spec_from_ckpt_args(cargs: dict) -> tuple[Path, dict]:
    """从训练 checkpoint 的 args 推出 (底座目录, E2EModel 的构造参数)。

    **单一来源。** 76_/87_ 以前各自写死 `QWEN = models/Qwen3-8B`，于是拿 35B 的
    checkpoint 去跑会**静默**加载 8B 底座、再 `load_state_dict(strict=False)` ——
    匹配不上的适配器参数**不报错**，日志看着正常、数字全假。
    （同一个坑的另一种形态：50_ 里 tokenizer 写死 8B 那一次，也是不报错。）

    修的时候抽成一个函数，两个脚本都调它，避免第三次分叉 —— 复制粘贴正是这次分叉的起因。
    """
    base = cargs.get("base", "qwen3-8b")
    base_dir = MDL if base == "qwen36-35b" else QWEN8
    kw = dict(
        base=base,
        base_mode=cargs.get("qwen36_mode", "int4"),
        base_bits=cargs.get("base_bits", "bf16"),
        lora_preset=cargs.get("lora_preset", "C"),
        pool=int(cargs.get("pool", 4)),
        adapter=cargs.get("adapter", "lora"),
        sos_k=int(cargs.get("sos_k", 4)),
        sos_gamma_max=float(cargs.get("sos_gamma_max", 2.5)),
        sos_dropout=float(cargs.get("sos_dropout", 0.05)),
    )
    return base_dir, kw

# 实测的 LoRA target 四套（scripts/86_lora_targets_qwen36.py 全验过）
LORA_TARGETS = {
    "A": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "B": ["q_proj", "k_proj", "v_proj", "o_proj",
          "in_proj_qkv", "in_proj_z", "out_proj"],
    "C": ["q_proj", "k_proj", "v_proj", "o_proj",
          "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"],
    "D": ["q_proj", "k_proj", "v_proj", "o_proj",
          "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
          "gate_proj", "up_proj", "down_proj"],
}
# 默认 C：40 层里 30 层是线性注意力，只打 full 层的 q/k/v/o 等于让 3/4 的
# 网络混合器完全不动（PLAN §14.4）。C = 190 模块 / 8.13 M。
LORA_PRESET = "C"


def trim_working_set() -> bool:
    """把本进程的常驻页还给系统。**装载过程中必须周期性调用。**

    为什么必需（2026-10-12 实测定位）
    ---------------------------------
    safetensors 是 **mmap** 的。量化一个专家要 `.half()` 它的权重，这会**读遍每一页**
    —— 那些页 fault 进进程工作集，而 `WorkingSet64` **计入常驻的文件映射页**。

    实测：
        摸 8.0 GiB 的分片 -> RSS +8.1 GiB，系统空闲同步掉 8.1 GiB（1:1）
        一层专家的 bf16 是 1.6 GiB，40 层 = 64 GiB
        -> 直接压死这台 63.6 GB 的机器，**表现是硬崩、没有异常栈**

    这个数很难查，因为它不在 `torch.cuda.memory_allocated()` 里（那只有 0.46 GiB/层），
    也不在普通 `fh.read()` 里（不留映射），只在 mmap + 真读页时出现。

    修剪后实测：摸 11.2 GiB，RSS 保持 0.07 GiB，系统空闲不动。

    两个 API 都是公开的、不需要提权：
        psapi.EmptyWorkingSet(h)
        kernel32.SetProcessWorkingSetSize(h, (SIZE_T)-1, (SIZE_T)-1)
    **argtypes 必须声明**，否则 HANDLE 会被当 int，报 OverflowError。
    """
    if os.name != "nt":
        return False
    try:
        import ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.GetCurrentProcess.argtypes = []
        psapi.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
        psapi.EmptyWorkingSet.restype = ctypes.c_int
        k32.SetProcessWorkingSetSize.argtypes = [ctypes.c_void_p,
                                                 ctypes.c_size_t, ctypes.c_size_t]
        k32.SetProcessWorkingSetSize.restype = ctypes.c_int
        h = k32.GetCurrentProcess()
        a = psapi.EmptyWorkingSet(h)
        b = k32.SetProcessWorkingSetSize(h, ctypes.c_size_t(-1), ctypes.c_size_t(-1))
        return bool(a and b)
    except Exception:
        return False


def free_gb() -> float:
    """系统物理空闲内存（GiB）。APU 上 mem_get_info() 会撒谎，只有这个可信。

    **但这个数不是模型占用**：它同时被 page cache 和进程 RSS 影响。
    排除法实测（2026-10-12）：
      · 顺序读 18 GiB 分片 -> 空闲一点没掉（page cache 不是原因）
      · 每层显存只涨 0.456 GiB，而系统空闲掉 1.1 GiB
      -> 差额在**主机内存**上，要看 rss_gb() 才分得开
    """
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
            capture_output=True, text=True).stdout.strip()
        return float(out) / 1024 / 1024
    except (ValueError, OSError):
        return -1.0


def rss_gb() -> float:
    """本进程的工作集（GiB）。用来把「模型显存」和「主机内存」分开。

    **必须用 os.getpid()。** 在 `powershell -Command` 里 `$PID` 是
    PowerShell 自己的进程号，不是 Python 的 —— 那样量出来的永远是 0.07 GiB，
    看起来"进程没占内存"，会误导排查方向。
    """
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"[math]::Round((Get-Process -Id {os.getpid()}).WorkingSet64/1GB,3)"],
            capture_output=True, text=True).stdout.strip()
        return float(out)
    except (ValueError, OSError):
        return -1.0
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
            capture_output=True, text=True).stdout.strip()
        return float(out) / 1024 / 1024
    except (ValueError, OSError):
        return -1.0


def commit_gb() -> tuple[float, float]:
    """(已用, 额度) GiB —— **这台机器上真正会杀人的额度**。

    物理 63.6 + 固定页面文件 32 = 95.6 GiB。映射分片、GPU 分配、页面文件
    全都算在里面。物理报"空闲 36 GiB"的时候 commit 可能已经见底，
    症状是 `OSError 1455 页面文件太小`，或者更早、更没栈的 0xC0000005
    （2026-10-14 实测，见 ShardReader 的 docstring）。
    `free_gb()` 量不到这一层，所以两个都要看。
    """
    import ctypes

    class _PERF(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("CommitTotal", ctypes.c_size_t),
            ("CommitLimit", ctypes.c_size_t),
            ("CommitPeak", ctypes.c_size_t),
            ("PhysicalTotal", ctypes.c_size_t),
            ("PhysicalAvailable", ctypes.c_size_t),
            ("SystemCache", ctypes.c_size_t),
            ("KernelTotal", ctypes.c_size_t),
            ("KernelPaged", ctypes.c_size_t),
            ("KernelNonpaged", ctypes.c_size_t),
            ("PageSize", ctypes.c_size_t),
            ("HandleCount", ctypes.c_ulong),
            ("ProcessCount", ctypes.c_ulong),
            ("ThreadCount", ctypes.c_ulong),
        ]

    m = _PERF()
    m.cb = ctypes.sizeof(m)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(m), m.cb):
        return -1.0, -1.0
    gb = m.PageSize / 2 ** 30
    return m.CommitTotal * gb, m.CommitLimit * gb


def make_quant_experts_cls(base_cls):
    """生成 drop-in 替换 Qwen3_5MoeExperts 的类（专家改成 per-expert nn.Linear）。"""
    from transformers.activations import ACT2FN

    class Qwen3_5MoeExperts(nn.Module):
        """与官方实现逐句等价，只是专家是 nn.Linear（bnb 才能量化）。"""

        def __init__(self, config):
            super().__init__()
            self.num_experts = config.num_experts
            self.hidden_dim = config.hidden_size
            self.intermediate_dim = config.moe_intermediate_size
            self.gate_up = nn.ModuleList([
                nn.Linear(self.hidden_dim, 2 * self.intermediate_dim, bias=False)
                for _ in range(self.num_experts)])
            self.down = nn.ModuleList([
                nn.Linear(self.intermediate_dim, self.hidden_dim, bias=False)
                for _ in range(self.num_experts)])
            self.act_fn = ACT2FN[config.hidden_act]

        def forward(self, hidden_states, top_k_index, top_k_weights):
            final_hidden_states = torch.zeros_like(hidden_states)
            with torch.no_grad():
                expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                # **一次同步取出 Python 列表。**
                #
                # 原来写的是 `for expert_idx in expert_hit: e = int(expert_idx[0])`
                # —— `expert_hit` 是 GPU 张量，`int()` 每取一个元素就做一次
                # GPU->CPU 同步。而 batch 小的时候 top-8 路由到 256 个专家上
                # **几乎全部命中**，40 层就是每步约一万次同步。
                #
                # 实测（scripts/92_moe_bench.py，固定样本总数只改 batch）：
                #     batch 1 -> 8：每步 10.50s -> 12.57s（只 1.20x）
                #                   每样本 10.500s -> 4.191s
                # 8 倍样本只花 1.2 倍时间 -> **启动/同步延迟主导，不是算力**。
                #
                # .tolist() 只同步一次，取值和顺序与逐个 int() 完全一致
                # （nonzero 按索引升序，flatten/tolist 保序），所以数学上等价。
                hit = torch.greater(
                    expert_mask.sum(dim=(-1, -2)), 0).nonzero().flatten().tolist()

            for e in hit:
                if e == self.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(expert_mask[e])
                current_state = hidden_states[token_idx]
                gate, up = self.gate_up[e](current_state).chunk(2, dim=-1)
                cur = self.act_fn(gate) * up
                cur = self.down[e](cur)
                cur = cur * top_k_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(
                    0, token_idx, cur.to(final_hidden_states.dtype))
            return final_hidden_states

    Qwen3_5MoeExperts.__name__ = base_cls.__name__
    Qwen3_5MoeExperts.__qualname__ = base_cls.__qualname__
    return Qwen3_5MoeExperts


def patch_conv1d(mdl_module=None):
    """把 linear_attention 里的 depthwise causal Conv1d 换成不用 MIOpen 的等价实现。

    为什么必须换
    ------------
    35B 的 30/40 层是 linear_attention，每层都有

        self.conv1d = nn.Conv1d(conv_dim, conv_dim, bias=False,
                                kernel_size=4, groups=conv_dim, padding=3)

    而 **MIOpen 在 gfx1151 上对 Conv1d 直接报 miopenStatusInternalError**
    （PLAN §12.6 已记；本项目的 FSQCode 压缩层当初就是因此改成 unfold+Linear 的）。
    实测在极小 MoE 上排练时崩在这里：

        modeling_qwen3_5_moe.py:493  F.silu(self.conv1d(mixed_qkv)[..., :L])
        torch/nn/modules/conv.py:380 -> F.conv1d -> miopenStatusInternalError

    等价实现
    --------
    深度可分离 + kernel=4 + padding=3 的因果卷积就是 4 个移位乘加：

        out[c, t] = sum_{k=0..3} w[c, 0, k] * x[c, t + k - 3]

    纯逐元素与广播，不经过任何 conv 算子。kernel 只有 4，所以代价很小。
    **保持参数名 `conv1d.weight` 不变**，checkpoint 的 key 才对得上。
    """
    import torch.nn.functional as F

    class DepthwiseCausalConv1d(nn.Module):
        def __init__(self, weight: torch.Tensor, bias=None):
            super().__init__()
            self.weight = nn.Parameter(weight.detach().clone(),
                                       requires_grad=weight.requires_grad)
            self.bias = (nn.Parameter(bias.detach().clone(),
                                      requires_grad=bias.requires_grad)
                         if bias is not None else None)
            self.K = int(weight.shape[-1])

        def forward(self, x):
            # x: (B, C, L)   weight: (C, 1, K)
            B, C, L = x.shape
            xp = F.pad(x, (self.K - 1, 0))
            w = self.weight.view(C, self.K)
            y = xp[:, :, 0:L] * w[:, 0].view(1, C, 1)
            for k in range(1, self.K):
                y = y + xp[:, :, k:k + L] * w[:, k].view(1, C, 1)
            if self.bias is not None:
                y = y + self.bias.view(1, C, 1)
            return y

    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as M
    if mdl_module is not None:
        M = mdl_module
    if getattr(M, "_rnaqwen_conv_patched", False):
        return
    G = M.Qwen3_5MoeGatedDeltaNet
    orig_init = G.__init__

    def __init__(self, config, *a, **kw):
        orig_init(self, config, *a, **kw)
        old = self.conv1d
        self.conv1d = DepthwiseCausalConv1d(old.weight, old.bias)

    G.__init__ = __init__

    # ---- 解码路径是**另一条路**，只修 self.conv1d 不够 ----
    # Qwen3_5MoeGatedDeltaNet.__init__ 里
    #     self.causal_conv1d_update = causal_conv1d_update or torch_causal_conv1d_update
    # 而 `torch_causal_conv1d_update` 内部**直接调 F.conv1d**，不经过 self.conv1d。
    # 于是：训练（纯预填充，走 self.conv1d）一直正常，而**生成第一步**就在
    #     modeling_qwen3_5_moe.py:467  self.causal_conv1d_update(...)
    #     modeling_qwen3_5_moe.py:230  F.conv1d  ->  miopenStatusInternalError
    # 崩掉。2026-10-15 实测：76_ 的 120 条生成全部变成 <ERR RuntimeError>，
    # 准确率 0.0%，看着像仪器坏了，其实是这里。
    #
    # 因为 __init__ 是**按模块名解析**取函数的，所以只要在建模型之前把
    # M.torch_causal_conv1d_update 换掉，实例就会拿到这一版。
    # 与 F.conv1d(padding=0, groups=C) 逐位等价的移位乘加，见
    # scripts/_scratch/_test_conv1d_update.py（CPU 上对拍了 F.conv1d）。
    def conv1d_update_eager(hidden_states, conv_state, weight, bias=None,
                            activation=None):
        _, hidden_size, seq_len = hidden_states.shape
        state_len = conv_state.shape[-1]
        x = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
        conv_state.copy_(x[:, :, -state_len:])
        w = weight.reshape(hidden_size, -1)          # (C, K)
        K = w.shape[-1]
        T = x.shape[-1]
        acc = None
        for k in range(K):
            seg = (x[:, :, k:k + (T - K + 1)].float()
                   * w[:, k].view(1, -1, 1).float())
            acc = seg if acc is None else acc + seg
        out = acc[:, :, -seq_len:]
        if bias is not None:
            out = out + bias.view(1, -1, 1).float()
        return F.silu(out).to(hidden_states.dtype)

    M.torch_causal_conv1d_update = conv1d_update_eager
    M._rnaqwen_conv_patched = True
    return DepthwiseCausalConv1d


def patch_moe_experts(mdl_module=None):
    """把 Qwen3_5MoeExperts 换成 per-expert Linear 版，**同时包掉 _init_weights**。

    为什么必须连 _init_weights 一起包
    ---------------------------------
    模型自己的 `_init_weights` 是**按属性名**访问专家的：

        elif isinstance(module, Qwen3_5MoeExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=...)
            init.normal_(module.down_proj, ...)

    而改写后的类只有 `gate_up` / `down` 两个 ModuleList，没有 `gate_up_proj` ——
    于是 `AttributeError: 'Qwen3_5MoeExperts' object has no attribute 'gate_up_proj'`。

    现在没撞上，只是因为 `load_qwen36_text` 总在 `torch.device("meta")` 下建模型
    （meta 上会跳过权重初始化）。**这个前提一旦不成立就崩** ——
    比如在 CPU/GPU 上直接建、或者哪天 transformers 改了 meta 的行为。
    所以这里显式包掉，让补丁在真实设备上也成立。
    """
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as M
    if mdl_module is not None:
        M = mdl_module
    if getattr(M, "_rnaqwen_experts_patched", False):
        return M.Qwen3_5MoeExperts
    base = M.Qwen3_5MoeExperts
    cls = make_quant_experts_cls(base)
    M.Qwen3_5MoeExperts = cls

    orig_init_w = M.Qwen3_5MoePreTrainedModel._init_weights

    def _init_weights(self, module):
        if isinstance(module, cls):
            std = getattr(self.config, "initializer_range", 0.02)
            for lin in list(module.gate_up) + list(module.down):
                nn.init.normal_(lin.weight, mean=0.0, std=std)
            return
        return orig_init_w(self, module)

    M.Qwen3_5MoePreTrainedModel._init_weights = _init_weights
    M._rnaqwen_experts_patched = True
    return cls


def remap(k: str):
    """checkpoint key -> 模型 key。None 表示丢弃。

    **不做任何前缀改写。** 装的是 VLM 类，它的 state_dict key 与 checkpoint
    逐字相同（含 `model.language_model.` 这一层）。
    那个前缀重映射只在装文本-only 类时才需要 —— 用在这里会把对的改错。
    """
    if k.startswith("mtp.") or k.startswith("model.visual."):
        return None
    return k


def build_index():
    idx = json.loads((MDL / "model.safetensors.index.json").read_text(encoding="utf-8"))
    wmap = idx["weight_map"]
    return wmap, sorted(set(wmap.values()))


class ShardReader:
    """按前缀枚举 + 按名取张量，底层 safetensors 惰性句柄。

    **句柄是占 commit 的，用完必须还。** 实测（`scripts/_scratch/_probe_commit.py`）：
    `safe_open` 一个分片在这台 Windows 上按**私有映射**计提交量，值约等于文件大小
    （26 个分片合计 67 GB），而同为 mmap 的 68 GB numpy 副缓存只算 1 GB。
    这台机器的 commit 额度只有 95.6 GB（物理 63.6 + 固定页面文件 32），
    于是"26 个句柄全开着"的后果是装载走到一半就报
    `OSError 1455 页面文件太小`，或者更早以 0xC0000005 硬崩
    （2026-10-14 实测：loader 停在 layer 11 / layer 35）。

    commit 是**会还回来的**（实测 5 个分片：71.5 -> 57.1 GB），但 safetensors
    没有公开的 `close()`，所以只能丢引用 + `gc.collect()` —— 这就是 `trim()`。

    为了让 `trim()` 随时可用，`get()` **一律返回私有副本**：mapped 视图活过
    `trim()` 就是 use-after-free，而这种 bug 只在几十层之后的前向里炸、
    没有任何栈能指回来。
    """

    def __init__(self, wmap: dict[str, str]):
        from safetensors import safe_open
        self.wmap = wmap
        self._open = safe_open
        self._h: dict[str, object] = {}
        self.names = sorted(wmap)
        # 每个分片覆盖的层范围；含非层内张量（embed / norm / lm_head）的分片
        # 在收尾才读，整个钉住、不参与驱逐。
        self._span: dict[str, tuple[int, int]] = {}
        self._pinned: set[str] = set()
        for name, shard in wmap.items():
            m = _LAYER_RE.match(name)
            if m is None:
                self._pinned.add(shard)
                continue
            i = int(m.group(1))
            lo, hi = self._span.get(shard, (i, i))
            self._span[shard] = (min(lo, i), max(hi, i))
        self.peak_open = 0

    def _handle(self, shard: str):
        if shard not in self._h:
            self._h[shard] = self._open(str(MDL / shard), framework="pt", device="cpu")
            self.peak_open = max(self.peak_open, len(self._h))
        return self._h[shard]

    def keys(self, prefix: str):
        import bisect
        i = bisect.bisect_left(self.names, prefix)
        out = []
        while i < len(self.names) and self.names[i].startswith(prefix):
            out.append(self.names[i])
            i += 1
        return out

    def get(self, name: str) -> torch.Tensor:
        if _TRACE_GET:
            print(f"    [get] {self.wmap[name]} :: {name}", flush=True)
        return self._handle(self.wmap[name]).get_tensor(name).clone()

    def trim(self, done_layer: int) -> int:
        """关掉层号范围已经全部处理完的分片，把 commit 还给系统。

        调用时机必须是"该层的所有张量都已经脱离映射之后"（量化是复制、
        `.to(dev)` 也是复制），也就是 `text.layers[i].to(dev)` 之后。
        """
        drop = [s for s, (_, hi) in self._span.items()
                if hi <= done_layer and s not in self._pinned and s in self._h]
        for s in drop:
            del self._h[s]
        if drop:
            gc.collect()
        return len(drop)

    def close_all(self) -> int:
        n = len(self._h)
        self._h.clear()
        gc.collect()
        return n


def layer_fragment(layer: int, reader: ShardReader) -> dict[str, torch.Tensor]:
    """读出一层的全部张量，专家按需拆分。key 保持 checkpoint 原样（VLM 命名）。"""
    frag: dict[str, torch.Tensor] = {}
    pre = f"model.language_model.layers.{layer}."
    for k in reader.keys(prefix=pre):
        if remap(k) is None:
            continue
        short = k[len(pre):]
        t = reader.get(k)
        if short == "mlp.experts.gate_up_proj":
            for e in range(t.shape[0]):
                frag[f"{pre}mlp.experts.gate_up.{e}.weight"] = t[e]
            del t
        elif short == "mlp.experts.down_proj":
            for e in range(t.shape[0]):
                frag[f"{pre}mlp.experts.down.{e}.weight"] = t[e]
            del t
        else:
            frag[k] = t
    return frag


def to_bnb_linear(lin: nn.Linear, mode: str) -> nn.Module:
    """nn.Linear -> bnb 量化 Linear。

    **让权重在 CPU 上交给 bnb，再 .to(dev) 触发量化**（见模块 docstring 第 3 条）。
    """
    dev = lin.weight.device
    w = lin.weight.data.detach().to("cpu").contiguous()
    if mode == "int8":
        from bitsandbytes.nn import Int8Params, Linear8bitLt
        new = Linear8bitLt(lin.in_features, lin.out_features,
                           bias=lin.bias is not None,
                           has_fp16_weights=False, threshold=6.0)
        new.weight = Int8Params(w, has_fp16_weights=False, requires_grad=False)
    elif mode == "int4":
        from bitsandbytes.nn import Linear4bit, Params4bit
        new = Linear4bit(lin.in_features, lin.out_features,
                         bias=lin.bias is not None,
                         compute_dtype=torch.bfloat16, compress_statistics=True,
                         quant_type="nf4")
        new.weight = Params4bit(w, requires_grad=False, compress_statistics=True,
                                quant_type="nf4")
    else:
        return lin
    if lin.bias is not None:
        new.bias = nn.Parameter(lin.bias.data.detach().to("cpu").clone())
    return new.to(dev)


def swap_linears(mod: nn.Module, mode: str, prefix: str,
                 experts_only: bool = True) -> int:
    """递归把 Linear 换成 bnb 版本。experts_only 时只碰 mlp.experts 下面那些。"""
    n = 0
    for name, child in list(mod.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            is_expert = ".mlp.experts." in full or full.endswith(".mlp.experts")
            if experts_only and not is_expert:
                continue
            if full.endswith("lm_head"):
                continue
            setattr(mod, name, to_bnb_linear(child, mode))
            n += 1
        else:
            n += swap_linears(child, mode, full, experts_only)
    return n


def fix_meta_buffers(model, tcfg, dev) -> list[str]:
    """重建建在 meta 上、且不在 checkpoint 里的 buffer（rotary inv_freq）。"""
    fixed = []
    if not [n for n, b in model.named_buffers() if b.device.type == "meta"]:
        return fixed
    text = model.model.language_model
    text.rotary_emb = type(text.rotary_emb)(tcfg, device=dev)
    fixed.append("model.language_model.rotary_emb")
    return fixed


def load_qwen36_text(mode: str = "int4", smoke: int = -1, min_free: float = 0.0,
                     dev: str | None = None, log=print):
    """流式装载 Qwen3.6-35B-A3B 的文本部分。

    返回 (model, text, lm_head, stats)。
    model 是 VLM 类（`Qwen3_5MoeForConditionalGeneration`），它的 forward
    **接受 `inputs_embeds`**，所以 route C 的注入路径可以直接用。
    """
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as M
    from transformers import AutoConfig

    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    cfg_vlm = AutoConfig.from_pretrained(str(MDL))
    tcfg = cfg_vlm.text_config

    patch_moe_experts(M)
    log(f"[qwen36] 专家类已替换为 per-expert nn.Linear（含 _init_weights 包装）")
    patch_conv1d(M)
    log(f"[qwen36] linear_attention 的 Conv1d 已换成等价实现（MIOpen 不支持 Conv1d）")

    with torch.device("meta"):
        model = M.Qwen3_5MoeForConditionalGeneration(cfg_vlm)
    text = model.model.language_model
    nl = len(text.layers)
    log(f"[qwen36] meta 建好：{sum(p.numel() for p in model.parameters())/1e9:.3f} B  "
        f"{nl} 层  hidden {tcfg.hidden_size}  vocab {tcfg.vocab_size}")

    fx = fix_meta_buffers(model, tcfg, dev)
    if fx:
        log(f"[qwen36] meta buffer 已重建 {fx}")

    wmap, _ = build_index()
    reader = ShardReader(wmap)
    n_layers = nl if smoke < 0 else min(smoke, nl)
    total_lin = 0
    gpu_gb = 0.0
    for i in range(n_layers):
        ts = time.time()
        frag = layer_fragment(i, reader)
        nkeys = len(frag)
        res = model.load_state_dict(frag, strict=False, assign=True)
        del frag
        gc.collect()
        stray = [k for k in res.unexpected_keys if f"layers.{i}." in k]
        if stray:
            raise RuntimeError(
                f"layer {i} 有 unexpected key：{stray[:3]} —— "
                f"key 前缀不对（不要改前缀，见 qwen36.remap 的注释）")
        # 量化先于搬到显卡
        nq = swap_linears(text.layers[i], mode, f"model.language_model.layers.{i}")
        text.layers[i].to(dev)
        total_lin += nq
        gc.collect()
        # **把量化临时量还给系统。**
        # bnb 量化一层要临时造一份 fp16 权重（0.805 B 参数 ~= 1.6 GB），量化完释放，
        # 但 PyTorch 的 caching allocator 默认**不还**给系统，于是每层 reserved 涨一点、
        # 累积到 40 层就把这台 APU（GPU 与 page cache 共用物理内存）挤爆 ——
        # 表现是硬崩、没有异常栈。实测 allocated 只涨 0.46 GiB/层，
        # 而系统空闲掉 1.6~1.9 GiB/层，差额就是这个缓存。
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()
        # **还要修剪工作集**：量化时读遍的 mmap 页会常驻在 WorkingSet 里，
        # 一层 1.6 GiB，40 层就是 64 GiB —— 不修剪会把整台机器压死（硬崩、无栈）。
        trim_working_set()
        if torch.cuda.is_available():
            gpu_gb = torch.cuda.memory_allocated() / 2 ** 30
            res_gb = torch.cuda.memory_reserved() / 2 ** 30
        else:
            res_gb = 0.0
        p = free_gb()
        cu, cl = commit_gb()
        log(f"    layer {i:>2}  {nkeys:>4} 张量  量化 {nq:>5} 个  "
            f"{time.time()-ts:>5.1f}s  显存 {gpu_gb:>5.2f}/{res_gb:>5.2f}  "
            f"RSS {rss_gb():>5.2f}  空闲 {p:>5.1f} GiB  "
            f"commit {cu:>5.1f}/{cl:.1f}  "
            f"句柄 {len(reader._h):>2}")
        # 本层的张量在 `text.layers[i].to(dev)` 之后就已经全部脱离映射
        # （专家是换成 bnb Linear 时复制、其余是 .to(dev) 复制），所以
        # 不再需要的分片现在关掉是安全的 —— 不关就是 67 GB 的 commit。
        reader.trim(i)
        # 空闲量只打印，**不再据此中止**。
        #
        # 原守卫是 `if p >= 0 and p < min_free: raise`。2026-10-14 把它关掉，理由有两条：
        #   1. 这台机器上量空闲内存的几个工具互相矛盾、也不可复现：
        #      同一时刻 CIM FreePhysicalMemory 报 46.8 GB、而装载循环里 free_gb() 报 1.1 GB；
        #      而 free_gb() 本身就是同一个 CIM 字段。既然量不准，拿它当"中止训练"的判据
        #      只会误杀 —— 实测 84_ 在 layer 28 报 1.1 GiB 而 76_/87_ 同层报 34 GiB。
        #   2. 它保护不了真正要防的东西：84_ 在 28 层中止，而 76_/87_ 走到 35 层
        #      以 0xC0000005 崩掉 —— 有守卫的那个没崩、没触发守卫的反而崩了。
        #      说明崩溃点在守卫够不到的地方，留着它只是多一个假阳性来源。
        # 想恢复就把 min_free 设成正数。
        if min_free > 0 and 0 <= p < min_free:
            raise RuntimeError(f"系统空闲 {p:.1f} GiB < {min_free} GiB，中止")

    # ---- 收尾：先丢掉视觉塔和 mtp，再把**所有**仍在 meta 上的参数补上 ----
    #
    # **不要硬编码收尾的名字。** 之前我写死 embed_tokens + lm_head，
    # 漏了 `model.language_model.norm.weight`（最后那个 RMSNorm，2048 个参数）——
    # 它既不在 layers.* 下，也不是那两个。漏掉的后果是它在 meta 上,
    # 前向时才炸，而且尺寸小到在"还有 X B 在 meta 上"里显示成 0.000 B，
    # 看不出是谁。
    if hasattr(model.model, "visual"):
        del model.model.visual
    if hasattr(model, "mtp"):
        del model.mtp
    gc.collect()

    leftover = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if leftover:
        frag = {n: reader.get(n) for n in leftover if n in reader.wmap}
        log(f"[qwen36] 收尾：补载 {len(frag)}/{len(leftover)} 个非层内参数")
        if frag:
            model.load_state_dict(frag, strict=False, assign=True)
        del frag
        gc.collect()
        orphan = [n for n in leftover if n not in reader.wmap]
        if orphan:
            log(f"  ** checkpoint 里没有的参数（会停在 meta）: {orphan[:5]}")
    trim_working_set()

    # ---- 把**非层内**模块搬到设备上 ----
    #
    # 层是逐个 `.to(dev)` 搬的，而 embed_tokens / norm / lm_head / rotary_emb
    # 不在 layers 里 —— 漏搬的症状是前向时报
    #     RuntimeError: index is on cuda:0, different from other tensors on cpu
    # （embed_tokens 的权重在 CPU，输入的 index 在 GPU）。
    # 这个不会在装载阶段暴露，只在第一次前向时炸。
    lm = model.model.language_model
    for m in (getattr(lm, "embed_tokens", None), getattr(lm, "norm", None),
              getattr(lm, "rotary_emb", None), getattr(model, "lm_head", None)):
        if m is not None:
            m.to(dev)
    gc.collect()
    # 非层内参数也已经在显卡上了，剩下的句柄全部还掉。
    log(f"[qwen36] 分片句柄 峰值 {reader.peak_open} 个同时打开，"
        f"收尾释放 {reader.close_all()} 个")

    cpu_left = [n for n, p in model.named_parameters()
                if p.device.type == "cpu"]
    if cpu_left:
        log(f"  ** 仍有 {len(cpu_left)} 个参数在 CPU 上: {cpu_left[:5]}")

    meta_left = [(n, p.numel()) for n, p in model.named_parameters()
                 if p.device.type == "meta"]
    if meta_left:
        tot = sum(s for _, s in meta_left)
        log(f"  ** 仍有 {len(meta_left)} 个参数在 meta 上，合计 {tot:,}")
        for n, s in sorted(meta_left, key=lambda x: -x[1])[:5]:
            log(f"       {s:>12,}  {n}")
        raise RuntimeError(f"还有 {tot} 个参数在 meta 上 —— 前向会炸")
    if cpu_left:
        raise RuntimeError(f"还有 {len(cpu_left)} 个参数在 CPU 上 —— 前向会炸")

    stats = dict(mode=mode, layers=n_layers, quantized=total_lin,
                 gpu_gb=gpu_gb, secs=time.time() - t0,
                 hidden=tcfg.hidden_size, vocab=tcfg.vocab_size,
                 shard_peak=reader.peak_open)
    log(f"[qwen36] 装载完成：量化 {total_lin:,} 个 Linear  "
        f"显存 {gpu_gb:.2f} GiB  用时 {time.time()-t0:.0f}s")
    return model, text, model.lm_head, stats


def lora_targets(preset: str = LORA_PRESET) -> list[str]:
    if preset in LORA_TARGETS:
        return list(LORA_TARGETS[preset])
    return [s.strip() for s in preset.split(",") if s.strip()]
