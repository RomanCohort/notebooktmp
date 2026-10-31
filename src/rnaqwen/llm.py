"""生成器：把 GPT-4o / 4o-mini 换成**本地 Qwen**（PLAN §3 P0 的成本对策）。

原版用 GPT-4o 造 12 万条 QA —— 那是真金白银。本复现改用本地 Qwen：
成本≈0、可复现、而且这一步本身就构成开源贡献的一部分。

两个后端：
    LocalHFGenerator      —— transformers 直接跑本地权重（主力，批量生成）
    OpenAICompatGenerator —— 任何 OpenAI 协议的 HTTP 服务（llama.cpp / vLLM / 网关）
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


class Generator(Protocol):
    name: str

    def chat(
        self,
        conversations: Sequence[Sequence[dict]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> list[str]: ...


# ---------------------------------------------------------------- 本地 HF
@dataclass
class LocalHFGenerator:
    """本地权重生成。默认关 thinking（Qwen3 系列默认会吐 思考…）。"""

    model_path: str
    name: str = ""
    device: str | None = None
    dtype: str = "bfloat16"
    batch_size: int = 8
    enable_thinking: bool = False
    seed: int | None = 0

    def __post_init__(self) -> None:
        self.name = self.name or self.model_path
        self._model = None

    def _lazy(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._torch = torch
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[self.dtype]
            t0 = time.time()
            self._tok = AutoTokenizer.from_pretrained(self.model_path, padding_side="left")
            self._model = AutoModelForCausalLM.from_pretrained(self.model_path, dtype=dtype).to(self.device).eval()
            if self.seed is not None:
                torch.manual_seed(self.seed)
            print(f"[gen] 载入 {self.model_path} 用时 {time.time()-t0:.1f}s  device={self.device} dtype={dtype}",
                  flush=True)
        return self._model

    def _apply_template(self, messages: list[dict]) -> str:
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        try:
            return self._tok.apply_chat_template(messages, enable_thinking=self.enable_thinking, **kwargs)
        except TypeError:
            return self._tok.apply_chat_template(messages, **kwargs)

    def chat(
        self,
        conversations: Sequence[Sequence[dict]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> list[str]:
        model = self._lazy()
        torch = self._torch
        out: list[str] = []
        convs = list(conversations)
        for i in range(0, len(convs), self.batch_size):
            batch = convs[i : i + self.batch_size]
            prompts = [self._apply_template(m) for m in batch]
            enc = self._tok(prompts, return_tensors="pt", padding=True)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-4),
                    top_p=top_p,
                    pad_token_id=self._tok.pad_token_id or self._tok.eos_token_id,
                )
            new = gen[:, enc["input_ids"].shape[1] :]
            out.extend(self._tok.batch_decode(new, skip_special_tokens=True))
        return out


# ---------------------------------------------------------------- OpenAI 协议
@dataclass
class OpenAICompatGenerator:
    """走任何 OpenAI 兼容的 /chat/completions（llama.cpp server、vLLM、网关都行）。"""

    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "local"
    api_key: str = "sk-local"
    name: str = ""
    timeout: int = 300
    retries: int = 3
    concurrency: int = 8          # 网络 API 可以并发；本地 llama.cpp 就设 1
    # DeepSeek 系的思考模式开关。**默认关掉** —— 见下方注释。
    thinking: str | None = "disabled"     # "enabled" | "disabled" | None（不传这个字段）
    reasoning_effort: str | None = None   # "low" | "high" | "max"

    def __post_init__(self) -> None:
        self.name = self.name or f"openai:{self.model}@{self.base_url}"

    def _body(self, messages: Sequence[dict], max_new_tokens: int, temperature: float, top_p: float) -> bytes:
        payload: dict = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        # DeepSeek（V4 系）**默认开着思考模式，effort=high**，实测每次调用会把
        # 上千个 token 全烧在 reasoning 上：32 次抽样里大部分 max_tokens=1100 被顶满、
        # `reasoning_tokens` 几乎占满（think=1100/out=1100）—— 不但贵，而且
        # **JSON 会被截断**。数据生成这种"照抄文本造问答"的任务不需要思考，
        # 直接关掉。要做推理蒸馏（E 轨）时再显式打开。
        if self.thinking is not None:
            payload["thinking"] = {"type": self.thinking}
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        return json.dumps(payload).encode()

    def _one(self, messages: Sequence[dict], max_new_tokens: int, temperature: float,
             top_p: float) -> str:
        body = self._body(messages, max_new_tokens, temperature, top_p)
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(
                    f"{self.base_url.rstrip('/')}/chat/completions",
                    data=body,
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode())
                msg = payload["choices"][0]["message"]
                text = msg.get("content") or ""
                if not text.strip():
                    # 思考模式下 content 可能为空、全在 reasoning_content 里
                    text = msg.get("reasoning_content") or ""
                return text
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"生成失败: {last}") from last

    def chat(
        self,
        conversations: Sequence[Sequence[dict]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> list[str]:
        convs = list(conversations)
        if self.concurrency <= 1 or len(convs) <= 1:
            return [self._one(m, max_new_tokens, temperature, top_p) for m in convs]

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            return list(ex.map(
                lambda m: self._one(m, max_new_tokens, temperature, top_p), convs
            ))


# ---------------------------------------------------------------- 工厂
DEEPSEEK_BASE = "https://api.deepseek.com/v1"


def build_generator(spec: str, **kwargs) -> Generator:
    """按 `spec` 造生成器。统一入口，别在脚本里到处写 if。

        "models/Qwen3-8B"                本地权重（transformers）
        "deepseek:deepseek-flash"        DeepSeek（**默认关思考**，数据生成用这个）
        "deepseek-think:deepseek-flash"  同上但**开思考**（推理蒸馏 E 轨用）
        "http://host:port/v1:model"      任意 OpenAI 兼容服务
        "openai:<model>"                 OpenAI 官方

    密钥从 `rnaqwen.secrets` 解析（环境变量 → DSH 凭据文件），**不落代码**。
    """
    from rnaqwen import secrets as _secrets

    if spec.startswith("deepseek-think:") or spec.startswith("deepseek:"):
        thinking = "enabled" if spec.startswith("deepseek-think:") else "disabled"
        model = spec.split(":", 1)[1] or "deepseek-flash"
        key = _secrets.resolve_api_key(kwargs.pop("api_key", None), provider="deepseek")
        kwargs.setdefault("thinking", thinking)
        return OpenAICompatGenerator(
            base_url=kwargs.pop("base_url", DEEPSEEK_BASE), model=model, api_key=key,
            name=f"deepseek{'（思考）' if thinking == 'enabled' else ''}:{model}", **kwargs,
        )
    if spec.startswith("openai:"):
        model = spec.split(":", 1)[1]
        key = _secrets.resolve_api_key(kwargs.pop("api_key", None), provider="openai")
        kwargs.setdefault("thinking", None)
        return OpenAICompatGenerator(base_url=kwargs.pop("base_url", "https://api.openai.com/v1"),
                                     model=model, api_key=key, name=f"openai:{model}", **kwargs)
    if "://" in spec:
        base, _, model = spec.partition(":")
        kwargs.setdefault("thinking", None)   # 本地服务不认这个字段
        return OpenAICompatGenerator(base_url=base, model=model or "local",
                                     api_key=kwargs.pop("api_key", "sk-local"),
                                     name=f"openai:{model}@{base}", **kwargs)
    return LocalHFGenerator(model_path=spec, **kwargs)


# ---------------------------------------------------------------- 工具
_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.S | re.I)


def strip_thinking(text: str) -> str:
    """Qwen3 的思考段（若模板没关干净）从这里剥掉。"""
    return _THINK_RE.sub("", text).strip()


def extract_json(text: str) -> Any:
    """从模型输出里抠出第一段合法 JSON（容忍 ```json 围栏和前后废话）。"""
    text = strip_thinking(text)
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"没有可解析的 JSON：{text[:200]!r}")
