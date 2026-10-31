"""模型侧：编码器 / 投影 / 两阶段训练。

    encoder     RNA-FM（冻结）—— 640 维
    projection  线性 640 → 4096（**与原版同维度**，因为 Qwen3-8B 的 hidden 也是 4096）
    alignment   Stage 1 模态对齐（只训投影）
    finetune    Stage 2 指令微调（LoRA）
"""
