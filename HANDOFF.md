# 交接 1 — RNAQwen v3 / MiRBench 关系判别

> 写给下一个会话。代码状态：`main @ 4207426e57687ada74e1b33003a6c76f0a45bec5`。
> 本地仓库 `C:\qwen-api\notebooktmp`，另一个非 git 目录 `C:\qwen-api\RNAQwen` 存着
> `PLAN_v2.md`（设计记录，很长，按需查）和 `.gh_token`。

## 0. 一句话

给 Qwen3.6-35B-A3B 加 RNA 序列模态（SOS-LoRA），当前任务：给一条 miRNA 和一段
mRNA 窗口，判断两者是否存在经实验验证的靶向关系（是/否），指标 AUROC。
**本地 AMD APU 装载 35B 会硬崩 0xC0000005，必须去 Colab A100 跑。**

## 1. 已经做完的

- `RNAQwen_v3_Full.ipynb` 已从旧的「4 选 1 候选式」切到关系判别：
  数据从 Zenodo 拉 MiRBench，训练脚本换成 `scripts/52_routeR_relation.py`。
- 架构：ERNIE-RNA（冻结）分别编码 miRNA / mRNA → 两个**问题感知** Perceiver
  resampler（各出 4 个软提示 token）→ 拼进 Qwen 的 `inputs_embeds` → 冻结的
  35B int4 + SOS-LoRA 出「是/否」的 logit。
- checkpoint：`step_*.pt` 保存可训参数 + AdamW 状态 + step，`--resume` 能真正续训；
  每 200 步或 10 分钟存一次，留 3 份，同步到 release **`train-v3-rel`**。
- 旧候选式 checkpoint 在 **`train-v3-checkpoints`**，tag 不同，且 notebook 和脚本
  都会校验 checkpoint 里的 `tag == "rel"` 且有 `weights` 字段，接错会直接丢弃/报错。
- 数据集：`data/processed/mirbench.jsonl`（4,129,088 行，1.9 GB，已 gitignore）。
  其中 `mirna_target` 2,849,872 行（train 正 1,247,038 / 负 1,256,637，
  test 正 177,898 / 负 168,299），`mirna_region` 1,279,216 行（当前训练不用）。
- **还没在 A100 上完整跑过一次。** 下面所有时间/显存都是设计值，不是实测。

钉死的 notebook 链接：
https://colab.research.google.com/github/RomanCohort/notebooktmp/blob/4207426e57687ada74e1b33003a6c76f0a45bec5/RNAQwen_v3_Full.ipynb

## 2. 新会话第一件事：Colab 怎么接

`colab-mcp` 已经注册到本项目（`~/.claude.json` 的 `C:\qwen-api\notebooktmp` 下），
但 **MCP 工具是会话启动时装载的**，所以先确认：

1. 工具表里有 `open_colab_browser_connection`（没有就说明没加载上）。
2. 调用它。它会在**用户这台机器**的浏览器打开
   `https://colab.research.google.com/notebooks/empty.ipynb#mcpProxyToken=...&mcpProxyPort=...`。
3. 用户在那个页面登录、选 **A100** 运行时（菜单「连接」→ 更改运行时类型）。
4. 页面连回本地 WebSocket 之后，notebook 编辑/执行类工具才会**动态出现**
   （服务器靠 `notifications/tools/list_changed` 注入）。

注意：
- 这不是云端会话，是**桥接用户浏览器里的 Colab**。标签页关掉、机器休眠，桥就断了。
- 如果 Claude Code 不支持 `list_changed`（未实测），工具永远不会出现 → 退回方案 B：
  让用户自己打开上面的链接，手动 Run All，我们只能从 release 里的 `best.pt`
  和用户贴的日志判断结果。

## 3. 运行顺序

按 cell 顺序跑（编号就是 notebook 里的位置）：

| cell | 干什么 | 备注 |
|---|---|---|
| 2 | 常量 + MiRBench 配置 | `CHECKPOINT_RELEASE_TAG='train-v3-rel'` |
| 3 | 环境检查 | |
| 4 | pip 安装 + kernel wheel | `causal-conv1d` 只装预编译轮子，**不允许源码编译** |
| 5 | 拉代码 archive + FSQ 两个文件 | 关系任务其实不用 FSQ，46 MB，无所谓 |
| 6 | 建 `DATA_DIR` | 不再下 663 MB 的旧数据集 |
| 7 | ERNIE-RNA（HF） | |
| 8 | 6 个 MiRBench `tsv.gz` + `build_mirbench.py` | 需要 ≥90 GiB 空闲；转换 2–3 分钟 |
| 9 | Qwen3.6-35B-A3B（HF，回退 ModelScope） | 67 GB |
| 11 | 训练 common 参数 | |
| 12 | checkpoint 同步（需要 Colab Secret `GITHUB_TOKEN`） | 没有 token 也能训，只是不能跨 runtime |
| 13 | **训练**，包在 `timeout --signal=TERM 27000` 里 | |
| 14 | 从 `train.log` 汇总 AUROC + 扰动结果 | |

Colab 上的路径：代码 `/content/RNAQwen`，资产 `/content/rnaqwen_assets`
（数据在 `data/processed/`，运行产物在 `runs/e2e_rel/`）。

## 4. 训练参数与「应该看到什么」

```
--max-train 200000 --max-test 20000     # 4.1M 条不能全读进内存
--batch 64 --steps 8000 --lr 2e-4 --lr-proj 1e-3
--eval-every 500 --eval-n 2000
--save-every 200 --save-minutes 10 --keep-ckpt 3
--grad-ckpt 1 --perturb 1 --tag rel
```

`--lr` 是 SOS-LoRA 的（2e-4，50_routeC 验证过的值），`--lr-proj` 是 resampler 的
（1e-3，随机初始化从零学）。两组分开，别合成一个 lr。

- 每行只有约 80 token（miRNA ≤23 + mRNA 50 + 问题），batch 64 ≈ 5k token。
- 20 万条是跨 CLASH/eCLIP/leftout 的均匀蓄水池样本，8000 步 ≈ 2.6 epoch。
- 损失 = 答案首 token 的 CE + `1.0 × sos_lora_penalty`（按层平均）。启动时会
  打印一行 `[loss] CE + 1.0 * SOS 正交正则`；日志里的 `ce` 和 `sos` 是分开的。
- 启动时还应该看到（缺任何一条都说明校验没跑）：
  `[enc] T->U 映射确认`、`[tok] 是=95761 否=96994 pad=...`、
  `[train] 可训 19.15 M ... lr 0.0002 / 0.001`。
- 日志里该看到：`[data] train 200000 test 20000`、
  每 50 步一行 `step ... ce ... sos ... 步/s`，每 500 步一行 `[eval@N] AUROC ...`。
- loss 应该从 ~0.69 往下走；AUROC 随机是 0.5，**先看它是否稳定高于 0.5 并在爬**。
- 训练结束（`--perturb 1`）会打印五行：正常 / 行置换 miRNA / 行置换 mRNA /
  打乱 miRNA 碱基 / 打乱 mRNA 碱基。后四行都应掉回 ~0.5。
- 真实格式由 tokenizer 的 chat template 渲染（`<think>\n\n</think>\n\n`），
  答案直接跟在生成提示后面，**没有** `" A: "` 脚手架；pred_pos 是拼接后的绝对下标。

## 5. 已知的坑

- **本地不要跑 35B**：AMD APU 上装载在 layer 10 附近硬崩 0xC0000005，没有 Python 栈。
- 脚本里「先建模型、再读数据」的顺序是有意的（内存峰值叠加），别改成先读数据。
- grouped MoE 路径一旦抛错会显式报错，不会静默切回 per-expert 路径
  （checkpoint 重算要求前后路径一致）。
- `.gh_token` 在 `C:\qwen-api\RNAQwen\.gh_token`，**不要把它打印出来**；
  git push 用 askpass 脚本从文件读。
- Colab 上如果 `causal-conv1d` 轮子装不上，不要退回源码编译——notebook 会直接失败，
  这是故意的（源码编译在 Colab 上基本必挂，而且会拖一小时）。
- 旧 checkpoint `step_005550.pt` 属于候选式任务，**不要**接到关系任务上。

## 6. 出问题先看哪里

| 现象 | 先查 |
|---|---|
| OOM | 把 `--batch 64` 降到 32；`--eval-n` 降到 1000 |
| AUROC 一直是 0.5 | loss 有没有降 → 正负样本是否平衡（mirbench.jsonl 里 train 正负约 1:1）→ `pred_pos` 是否指向生成提示的最后一位（启动日志里有 `[tok]`/`[train]` 校验行） |
| 打乱后 AUROC 不降 | 模型在走捷径（比如只看问题文本/数据集来源），**如实记录，不要粉饰** |
| `--resume` 报 tag 不符 | 正常，说明接错 checkpoint 了 |
| 磁盘不够 | cell 8 的 90 GiB 检查会直接报错 |

## 7. 2026-11-02 彻查修掉的问题

这些不是"优化"，是会让第一次 A100 跑出来一个看着正常、其实错位的结果。**别回退。**

1. **`pred_pos` 坐标系错了（最严重）**。`make_batch` 里算的是 suffix 内部下标
   `len(pre)-1`，但模型 `hs[arange, pred_pos]` 索引的是
   `[prefix][soft][suffix]` 拼起来的整条序列。于是模型一直在读问题中间某个
   token 的 hidden 去预测答案。现在按 `len(left) + 2*n_query + len(right) - 1`
   算绝对下标，并且每个 batch 断言 `pred_pos` 的下一位就是目标 token。
2. **chat 模板是手搓的，且是错的**。Qwen3.6 的 no-think 段是
   `<think>\n\n</think>\n\n`（普通字符串），不是 Qwen3 的
   `" thinking\n\n<｜end▁of▁thinking｜>\n\n"`（14 个 token、含 158819）；而且
   `<|im_end|>` 后面漏了换行。现在用 tokenizer 自己的
   `apply_chat_template(enable_thinking=False)` 渲染再切。
3. **冻结的 ERNIE 被 `model.train()` 切成了训练模式**。它的 config 是
   `hidden_dropout=0.1, attention_dropout=0.1`，于是每条训练样本的序列表示都
   被随机丢 10%，而评测不会。`RelationModel.train()` 现在强制 `encoder.eval()`。
4. **`model.to(DEV)` 会递归到 bitsandbytes int4 底座**。bnb 不支持量化后再 `.to()`。
   改成只把两个 resampler 移到 GPU。
5. **SOS 正交正则没接上**。已加，系数 1.0，按层平均。
6. **checkpoint 不是原子写**，而 notebook 的同步线程每 60 s 扫 `step_*.pt` 上传，
   可能传走半截的 200 MB 文件。改成写 `.tmp` 再 `os.replace`，上传前再验 zip。
7. **resume 可能挑中 `best.pt`**（不带优化器状态、step 可能更早）。现在优先
   `step_*.pt`；并且校验 tag、weights 字段和**张量数**，对不上直接报错。
8. **学习率统一给了 1e-3**。SOS 在 50_routeC 上验证的是 2e-4，resampler 是随机
   初始化。拆成两组。
9. **DNA 字母表（T）**：MiRBench 是 T，ERNIE 词表只有 U。multimolecule 的
   `RnaTokenizer` 默认 `replace_T_with_U=True`，但这是版本相关默认值 ——
   启动时实测 `ACGT` 与 `ACGU` 编码相同，不同就报错。
10. **数据审计**（本地实测）：train 2,503,675 / test 346,197 行；同一
    (miRNA, mRNA窗口) 对跨 split 重叠 57 对（占 test 0.02%），标签冲突 15 对。
    test 里 99.4% 来自 Manakov，CLASH 只有 1,018 行、Klimentova 954 行——
    **AUROC 主要由 Manakov 决定**，别把它讲成跨数据集泛化。

## 8. 训练之后

- AUROC 正常 → 往上加 rung：lncRNA-miRNA（starBase/DIANA-LncBase/RAID）、
  general RNA-RNA（NPInter v4/RNAInter），再接 OpenHarness + RAG 的文献证据层。
- AUROC 不行 → 先做输入扰动/消融定位原因，再考虑改投影层或数据配比；
  FSQ 与问题感知投影的对照实验已经做过（FSQ 在 4 选 1 上 25% = 随机），结论在
  `C:\qwen-api\RNAQwen\PLAN_v2.md` §31.5。
- 设计细节、任务阶梯、中英文提示词的取舍都在同一份 PLAN 的 §31。
