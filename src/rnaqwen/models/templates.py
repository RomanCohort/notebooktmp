"""训练/推理**共用**的 prompt 模板 —— 两边必须拼出逐字相同的字符串。

为什么单独放一个模块
--------------------
Stage 1、Stage 2 训练、以及 `30_infer.py` 推理三处的拼接顺序必须完全一致，
否则推理时输入直接落到**分布外**（不是精度损失，是直接失效）。

踩过的坑（agent D 审计出来的）：`30_infer.py` 原先用 `tokenizer.apply_chat_template`
（Qwen 的 ChatML），而 Stage 2 训练用的是原版论文的 Llama-3 风格
`###Human: …\n###Assistant:`。两个模板不一样 —— 训练时模型从没见过 ChatML 那种写法。

模板出处（arXiv 2411.08900v1）：
  * Stage 1 §2.1：`Q: <RNA><RNAHere></RNA><QuestionPrompts>` / `A: <Description>`
    —— 其中 **Q 留空**，`<RNAHere>` 位置由投影后的软提示占位
  * Stage 2 §2.1：改成 Llama-3 风格 `###Human: …` / `###Assistant: …`
"""

from __future__ import annotations

# ---- RNA 槽位的定界标记（Stage 1 与 Stage 2 共用） ----
RNA_PREFIX = "Q: <RNA>"
RNA_SUFFIX = "</RNA>\n"

# ---- Stage 1：答案段标记 ----
STAGE1_SUFFIX = "</RNA>\nA: <Description>"

# ---- Stage 2：指令模板 ----
TEMPLATE_LLAMA3 = "llama3"
TEMPLATE_QWEN = "qwen"
TEMPLATE_STAGE1 = "stage1"


def format_prompt(question: str, template: str = TEMPLATE_LLAMA3) -> str:
    """prompt 部分（**不计 loss**）。软提示插在它前面、`Q: <RNA>` 之后。"""
    q = (question or "").strip()
    if template == TEMPLATE_LLAMA3:
        return f"###Human: {q}\n###Assistant:"
    if template == TEMPLATE_QWEN:
        return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    return ""   # stage1：Q 留空


def format_target(question: str, answer: str, template: str = TEMPLATE_LLAMA3) -> str:
    """只在 `template="stage1"` 的对照消融里用（问题也进 target）。"""
    return f"###Human: {question}\n###Assistant: {answer}"


def sequence_as_text_input(question: str, sequence: str, template: str = TEMPLATE_LLAMA3) -> str:
    """**RNA Sequence 基线**：不走编码器，把序列当普通文本塞进同一个模板。

    这是原版消融阶梯的第一档（"LLM with RNA sequence as text input"）。
    保持与 RNA-GPT 档**同一个模板结构**，只把 `<RNAHere>` 软提示换成字面序列 ——
    这样三档之间的差异只剩"序列是怎么进去的"这一个变量。
    """
    return f"{RNA_PREFIX}{sequence}{RNA_SUFFIX}{format_prompt(question, template)}"
