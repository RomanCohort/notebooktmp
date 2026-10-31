"""RNA-QA 数据管线（PLAN §3 P0）。

模块：
    sources      数据源客户端（NCBI Entrez）
    records      RNA-QA 记录结构与 JSONL 读写
    annotate     D&C 摘要（本地 Qwen 生成，替代 GPT-4o）
    qa           指令数据：abstract/annotation → QA 对
    guard        验收：QA 里不许出现序列原文（出现就白训）
"""
