"""grouped vs per-expert 专家路径的数值等价性测试（需 CUDA + bitsandbytes）。

在小规模专家上对比两条路径的 forward 输出与 dL/dx：
grouped 必须在 bf16 容差内和 per-expert 一致，否则不许启用。
另外断言 grouped 真的跑了（没有静默退回 per-expert）。

用法（A100 / CUDA）：
    python scripts/_test_grouped_experts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from rnaqwen.qwen36 import (make_quant_experts_cls, set_experts_impl,  # noqa: E402
                            to_bnb_linear)


class _Cfg:
    num_experts = 8
    hidden_size = 64
    moe_intermediate_size = 32
    hidden_act = "silu"


class _Base:
    """make_quant_experts_cls 只用到 base_cls 的名字。"""


def build(dev: str):
    cls = make_quant_experts_cls(_Base)
    # 真实装载路径里专家权重来自 safetensors（bf16），这里先转 bf16 再量化，
    # 让 quant_state.dtype 与真实路径一致，比较才有意义。
    m = cls(_Cfg()).to(dev).to(torch.bfloat16)
    for e in range(m.num_experts):
        m.gate_up[e] = to_bnb_linear(m.gate_up[e], "int4").to(dev)
        m.down[e] = to_bnb_linear(m.down[e], "int4").to(dev)
    return m


def run(m, x, idx, w, impl):
    set_experts_impl(impl)
    x = x.detach().clone().requires_grad_(True)
    y = m(x, idx, w)
    y.float().pow(2).sum().backward()
    return y.detach().float(), x.grad.detach().float()


def main() -> int:
    if not torch.cuda.is_available():
        print("需要 CUDA（bitsandbytes int4 反量化）")
        return 2
    dev = "cuda"
    torch.manual_seed(0)
    m = build(dev)
    T, H, K = 16, _Cfg.hidden_size, 2
    x = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
    idx = torch.randint(0, _Cfg.num_experts, (T, K), device=dev)
    w = torch.softmax(torch.randn(T, K, device=dev), dim=-1)

    y1, g1 = run(m, x, idx, w, "per_expert")
    m._grouped_broken = False
    y2, g2 = run(m, x, idx, w, "grouped")
    if m._grouped_broken:
        print("FAIL: grouped 路径抛异常并静默退回 per-expert（见上面的警告）")
        return 1

    dy = (y1 - y2).abs().max().item()
    dg = (g1 - g2).abs().max().item()
    scale_g = max(g1.abs().max().item(), 1e-6)
    rel_g = dg / scale_g
    print(f"forward max|diff| {dy:.3e}   (scale {y1.abs().max().item():.3e})")
    print(f"grad    max|diff| {dg:.3e}   rel {rel_g:.3e}   (scale {scale_g:.3e})")
    ok = dy <= 2e-2 and rel_g <= 2e-2
    print("EQUIV", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
