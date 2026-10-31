"""凭据解析：**不把密钥写进代码、也不打印出来**。

优先级：
    1. 环境变量（`DEEPSEEK_API_KEY` 等）
    2. DSH 的凭据文件 `C:\dsh\home\.credentials.yaml`（本机已有，DSH 自己也在用）

设计原则：这个模块只返回字符串；调用方**永远不要**把返回值 print 出来，
日志里只允许出现 `sk-***` 这种掩码形式。
"""

from __future__ import annotations

import os
import re
from pathlib import Path

CRED_FILES = [
    Path(r"C:\dsh\home\.credentials.yaml"),
    Path.home() / ".dsh" / ".credentials.yaml",
    Path.home() / ".config" / "rnaqwen" / "credentials.yaml",
]

_KNOWN = ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "QWEN_LOCAL_API_KEY", "DASHSCOPE_API_KEY")


def _read_cred_file(name: str) -> str | None:
    for path in CRED_FILES:
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        m = re.search(rf"^\s*{re.escape(name)}\s*:\s*(\S+)\s*$", text, flags=re.M)
        if m:
            return m.group(1).strip().strip("\"'")
    return None


def get_secret(name: str) -> str | None:
    val = os.environ.get(name)
    if val:
        return val.strip()
    return _read_cred_file(name)


def mask(secret: str | None) -> str:
    if not secret:
        return "(missing)"
    return f"{secret[:3]}***{secret[-2:]}" if len(secret) > 8 else "***"


def resolve_api_key(explicit: str | None = None, provider: str = "deepseek") -> str:
    if explicit:
        return explicit
    env_name = {
        "deepseek": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY",
        "dashscope": "DASHSCOPE_API_KEY",
    }.get(provider, f"{provider.upper()}_API_KEY")
    key = get_secret(env_name)
    if not key:
        raise RuntimeError(
            f"找不到 {env_name}。放到环境变量里，或写进 "
            f"{CRED_FILES[0]}（键名 {env_name}）。"
        )
    return key
