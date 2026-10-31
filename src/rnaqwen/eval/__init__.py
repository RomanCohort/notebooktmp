"""评测口径（PLAN §0 锁定）。

两套指标，照抄原论文：

1. **ROUGE-1 / ROUGE-2 / ROUGE-L**（F-measure）
2. **Semantic Precision / Recall / F1** —— 句子级 max-cosine，三种嵌入
   记作 `S_BERT` / `S_Pub` / `S_GPT`。

口径一旦锁定就不再改（PLAN §5 风险登记："打平"不能变成长期漂移目标）。
"""

from rnaqwen.eval.rouge import rouge_scores, rouge_1, rouge_2, rouge_l
from rnaqwen.eval.semantic import semantic_scores, split_sentences

__all__ = [
    "rouge_scores",
    "rouge_1",
    "rouge_2",
    "rouge_l",
    "semantic_scores",
    "split_sentences",
]
