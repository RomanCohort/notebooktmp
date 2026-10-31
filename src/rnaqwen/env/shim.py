"""Runtime shims for this machine's AMD/ROCm Windows torch build.

Why this file exists
--------------------
本机 torch 是 TheRock ROCm Windows nightly（2.12.0a0+rocm7.13.0a20260313），
编译时**没有**编入分布式后端：

    >>> import torch.distributed as dist
    >>> dist.is_available()
    False
    >>> import torch.distributed.checkpoint as dcp
    ModuleNotFoundError: No module named 'torch._C._distributed_c10d'

但 `danling`（MultiMolecule 的运行时依赖）在 import 阶段就**无条件**地引用这些符号：

    danling/__init__.py -> .runners -> torch_runner.py:34  import torch.distributed.distributed_c10d
                                   -> torch_runner.py:71  from torch.distributed.tensor import DTensor
                                   -> torch_runner.py:1328 dist.ReduceOp.SUM   (default arg)

于是 `import multimolecule` 在本机直接不可行，RNA-FM 这条官方路径被环境卡住。

本 shim 的做法
--------------
在 import multimolecule **之前**，把缺失的分布式符号补成占位符：

  * `torch.distributed.distributed_c10d` / `torch.distributed.tensor(.placement_types)`
    —— 换成"要什么给什么"的 permissive 模块；
  * `torch.distributed` 本身 —— 换掉它的 `__class__`，让缺失属性（`ReduceOp` 等）
    按需生成占位类。

我们**永远不会真正执行**分布式通信（单卡 forward / 单卡 LoRA），
所以占位符不会被调用，只负责让 import 通过。

用法
----
    import rnaqwen.env.shim  # noqa: F401  —— 必须早于任何 multimolecule/danling import
"""

from __future__ import annotations

import sys
import types
from typing import Any

__all__ = ["install", "is_installed", "report"]

_SENTINEL = "_rnaqwen_shim_installed"


class _DummyMeta(type):
    """Metaclass for fabricated stand-ins.

    Supports the three ways danling touches these symbols:
      * 取值        `dist.ReduceOp`          -> 生成类
      * 取类属性    `dist.ReduceOp.SUM`      -> 再生成类
      * 调用/实例化 `get_world_size()`       -> 返回实例
    """

    def __getattr__(cls, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        dummy = _DummyMeta(name, (_DummyBase,), {"__module__": cls.__module__})
        setattr(cls, name, dummy)
        return dummy


class _DummyBase(metaclass=_DummyMeta):
    """Fabricated stand-in for anything torch.distributed would have provided."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args: Any, **kwargs: Any) -> "_DummyBase":
        return _DummyBase(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _DummyBase()


def _make_dummy(name: str, module: str = "torch.distributed") -> type:
    return _DummyMeta(name, (_DummyBase,), {"__module__": module})


class _PermissiveModule(types.ModuleType):
    """A module that fabricates any attribute on demand.

    `__path__ = []` 把它标记成 package —— 否则
    `import torch.distributed.tensor.device_mesh` 会在父模块这一步就报
    "'torch.distributed.tensor' is not a package"，连 _PermissiveFinder 都轮不到。
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.__path__ = []  # type: ignore[attr-defined]

    def __getattr__(self, name: str) -> Any:  # only called when normal lookup fails
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        dummy = _make_dummy(name, self.__name__)
        setattr(self, name, dummy)
        return dummy


class _PermissiveLoader:
    """Loader for fabricated submodules."""

    def create_module(self, spec):  # noqa: ANN001
        return _PermissiveModule(spec.name)

    def exec_module(self, module) -> None:  # noqa: ANN001
        return None


class _PermissiveFinder:
    """Meta-path finder placed at the **end** of sys.meta_path.

    只有在真正的 import 机制找不到模块时才会被调用 —— 也就是说：
    torch.distributed 下真实存在的模块照常加载，只有这个残废 nightly 里
    **确实缺失**的那些（distributed_c10d / tensor.* / checkpoint.*）才被伪造。
    """

    PREFIXES = ("torch.distributed.",)

    def find_spec(self, fullname: str, path=None, target=None):  # noqa: ANN001
        import importlib.util

        if not any(fullname.startswith(p) for p in self.PREFIXES):
            return None
        return importlib.util.spec_from_loader(fullname, _PermissiveLoader())

    # 兼容旧式 find_module 协议（importlib 已不用，留着无害）
    def find_module(self, fullname: str, path=None):  # noqa: ANN001
        return None


def _noop(*args: Any, **kwargs: Any) -> None:
    return None


class _StateDictOptions:  # pragma: no cover - never instantiated in our runs
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


def install(verbose: bool = False) -> bool:
    """Install every stub multimolecule/danling needs on this torch build."""
    if getattr(sys.modules.get("__rnaqwen_shim__"), _SENTINEL, False):
        return False

    marker = types.ModuleType("__rnaqwen_shim__")
    setattr(marker, _SENTINEL, True)
    sys.modules["__rnaqwen_shim__"] = marker

    import torch  # noqa: F401
    import torch.distributed  # noqa: F401

    installed: list[str] = []

    # ---- 1. torch.distributed.checkpoint (+ .state_dict) -------------------
    try:
        import torch.distributed.checkpoint  # noqa: F401
    except Exception:
        pkg = types.ModuleType("torch.distributed.checkpoint")
        pkg.__doc__ = "rnaqwen shim: ROCm-Windows torch has no distributed backend."
        state_dict = types.ModuleType("torch.distributed.checkpoint.state_dict")
        state_dict.StateDictOptions = _StateDictOptions  # type: ignore[attr-defined]
        state_dict.get_state_dict = _noop  # type: ignore[attr-defined]
        state_dict.set_model_state_dict = _noop  # type: ignore[attr-defined]
        state_dict.set_optimizer_state_dict = _noop  # type: ignore[attr-defined]
        pkg.state_dict = state_dict  # type: ignore[attr-defined]
        sys.modules["torch.distributed.checkpoint"] = pkg
        sys.modules["torch.distributed.checkpoint.state_dict"] = state_dict
        torch.distributed.checkpoint = pkg  # type: ignore[attr-defined]
        installed.append("torch.distributed.checkpoint")

    # ---- 2. deep submodules danling/transformers import by name -------------
    for name in (
        "torch.distributed.distributed_c10d",
        "torch.distributed.tensor",
        "torch.distributed.tensor.placement_types",
        "torch.distributed.tensor._dtensor_spec",
        "torch.distributed.tensor.device_mesh",
    ):
        if name in sys.modules:
            continue
        try:
            __import__(name)
            continue
        except Exception:
            pass
        stub = _PermissiveModule(name)
        sys.modules[name] = stub
        parent_name, _, leaf = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None:
            try:
                setattr(parent, leaf, stub)
            except Exception:
                pass
        installed.append(name)

    # ---- 2b. 兜底：任何仍然缺失的 torch.distributed.* 子模块按需伪造 -------
    if not any(isinstance(f, _PermissiveFinder) for f in sys.meta_path):
        sys.meta_path.append(_PermissiveFinder())
        installed.append("meta_path:_PermissiveFinder")

    # ---- 3. make torch.distributed itself permissive (ReduceOp, ...) -------
    dist = torch.distributed
    if dist.__class__ is not _PermissiveModule:
        try:
            dist.__class__ = _PermissiveModule
            installed.append("torch.distributed (permissive)")
        except Exception:
            pass

    marker.installed = installed  # type: ignore[attr-defined]
    if verbose:
        print("[shim] patched:", ", ".join(installed) or "(nothing)")
    return True


def report() -> list[str]:
    marker = sys.modules.get("__rnaqwen_shim__")
    return list(getattr(marker, "installed", []) or [])


def is_installed() -> bool:
    return bool(getattr(sys.modules.get("__rnaqwen_shim__"), _SENTINEL, False))


install()
