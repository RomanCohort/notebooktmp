"""数据生成的提示词 —— 集中放一处，方便以后对齐/改写。

两条硬约束贯穿所有提示词：
  1. **不许出现碱基串。** 一出现就等于把答案写进了题目，训练会直接学成"抄序列"，
     PLAN §5 明确把这条列为"指标虚高"的头号来源。
  2. **只许用给定文本里的事实。** 生成器不许自由发挥 —— 否则评测时
     reference 里没有的东西会被当成模型能力，指标就不可解释了。
"""

from __future__ import annotations

SYSTEM_RNA_EXPERT = (
    "You are an expert RNA biologist writing precise, factual annotations. "
    "You never invent facts and you never write nucleotide sequence strings."
)

NO_SEQUENCE_RULE = (
    "Hard rule: never output any nucleotide sequence or run of bases "
    "(no strings like ACGUACGU, no UUUU, no single-base repeats). "
    "Describe molecules by name and function only."
)

# ---- D&C：分块摘要 ----
CHUNK_SUMMARY = """Summarize the excerpt below in 1-2 sentences.

Keep only facts about RNA molecules: their type, function, structure, interaction partners,
regulatory mechanism, or disease relevance.
Do not mention the study, the authors, the journal, the methods, or the model organism's
general biology unless it is needed to explain the RNA.
{rules}

Excerpt:
{chunk}

Summary:"""

# ---- D&C：合并成注释 ----
COMBINE_ANNOTATION = """You are writing the reference annotation for one RNA molecule.

Molecule context:
- Definition line: {definition}
- Organism: {organism}

Below are summaries of consecutive parts of the scientific literature about it:
{summaries}

Write a single coherent annotation paragraph of 3-5 sentences that describes this RNA:
what it is, what it does, how it is regulated, and why it matters.
Use only the information above. Do not describe the studies themselves.
{rules}

Annotation:"""

# ---- AIS：把摘要拆成句子 ----
SENTENCE_SPLIT = """Split the following scientific abstract into its individual sentences.
Copy each sentence verbatim (do not rewrite, do not summarize, do not merge).
Return a JSON array of strings.

Abstract:
{abstract}

JSON:"""

# ---- 指令数据：QA 生成 ----
#
# 原文 §2.3 说得很明确：
#   "we prompt GPT-4o-mini to generate **both open-ended and close-ended QA pairs**
#    ... to decompose the abstract into **atom-level QA pairs**."
# 所以不能只造一种。我们之前只造了 open-ended，等于漏了一半。
QA_FROM_TEXT = """You are building an instruction-tuning dataset about RNA molecules.

Source text:
\"\"\"
{text}
\"\"\"

Write exactly {n} **open-ended** question-answer pairs that a biologist would find useful.
Rules:
1. Every answer must be fully supported by the source text above.
2. Questions must be self-contained: a reader who has not seen the source text must
   understand what is being asked.
3. Answers are 1-2 complete sentences, not fragments.
4. Vary the questions: cover identity/type, function, mechanism, interaction partners,
   regulation, and disease or application relevance where the text supports it.
5. {rules}

Return a JSON array of objects with keys "question" and "answer". No other text.

JSON:"""

# close-ended：答案短、可判定（yes/no 或多选）。原文要的是 "atom-level" 的问答对 ——
# 这类问题的价值在于**答案几乎无法含糊**，评测时不容易被"话术"糊过去。
QA_CLOSED = """You are building an instruction-tuning dataset about RNA molecules.

Source text:
\"\"\"
{text}
\"\"\"

Write exactly {n} **close-ended** question-answer pairs about the RNA described above.
Allowed forms:
  (a) yes/no questions — the answer must start with "Yes" or "No" and then add one
      short justifying clause;
  (b) multiple-choice questions with 4 options (A-D) — the answer must be the option
      letter followed by the option text.
Rules:
1. Every answer must be fully supported by the source text.
2. Keep answers under 25 words. They must be checkable at a glance.
3. Do not ask about the study itself (authors, methods, sample sizes).
4. {rules}

Return a JSON array of objects with keys "question", "answer" and "kind"
(kind is "yes_no" or "multiple_choice"). No other text.

JSON:"""

# ---- 反向质控：判断答案是否只靠文本 ----
FAITHFULNESS_CHECK = """Decide whether the ANSWER is fully supported by the SOURCE.

SOURCE:
{text}

QUESTION: {question}
ANSWER: {answer}

Reply with exactly one word: SUPPORTED or UNSUPPORTED."""


def fill(template: str, **kwargs) -> str:
    rules = kwargs.pop("rules", NO_SEQUENCE_RULE)
    return template.format(rules=rules, **kwargs)
