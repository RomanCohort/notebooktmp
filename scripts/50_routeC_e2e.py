"""路线 C · 端到端版：让 FSQ 收到 LLM 的梯度

为什么必须重写（而不是给 47_ 打补丁）：

  1. **梯度断路。** 47_ 里 RNA 向量来自 `rna_emb(code)` —— 一个不可微的索引查表。
     `rna_emb` 在学，但**码的切分方式永远收不到 LLM 的梯度**。
     也就是说，码怎么分不可能朝"对问答有用"的方向调整。这是设计缺陷，不是超参问题。
     （对照：Yeti 2605.09981 的 tokenizer 是端到端训的，换了"最好的码本利用率 + token 多样性"。）

  2. **仪器有缺陷。** 47_ 的 `vs 随机表` 同时干了三件事：换掉整张表、破坏码↔内容对应、
     改变向量尺度。判不出到底是哪一条。这里加**码 ID 置换**对照：用随机双射 π 把码 c 换成 π(c)，
     表本身、边缘分布、尺度全部不变，**只切断"这个码来自什么内容"**。
     置换后性能不变 => 模型根本没读码的身份。这才是干净的仪器。

  3. **塌缩无处可生。** 加 `head(q_ste)` 通路后，码→向量是**算出来的**（12→4096 的线性层），
     不是 4096×4096 存出来的。没有可塌的表。（table 通路保留，用来对比。）

损失 = CE  + λ_ent·(-码使用熵)  + λ_commit·||t-q||²  + λ_ortho·码行间余弦²

用法:
    python scripts/50_routeC_e2e.py --smoke
    python scripts/50_routeC_e2e.py --steps 600 --eval-every 150
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = Path(os.environ.get("RNAQ_RUNS_DIR", str(ROOT / "runs")))
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: F401,E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NT = 1024
EARLY_METRIC = "answer_masked_ce_v1"
# LATENT/LEVELS 的**单一真相来源是 checkpoints/fsq_full/tokenizer.pt**，
# 在 main 里读出来覆盖。硬编码会跟磁盘产物漂移 —— 表现是
# `assert w.shape == model.rna_emb.weight.shape` 失败，日志只有一行 [tau] 看得出端倪。
# 下面这两个只是 50_ 被 import 时的兜底值，实际训练不用它们。
LATENT, LEVELS = 12, 2
VOCAB = LEVELS ** LATENT
QWEN = Path(os.environ.get("RNAQ_QWEN8_DIR", str(ROOT / "models/Qwen3-8B")))
_QWEN36_DEFAULT = Path(r"C:\qwen-api\models\Qwen3.6-35B-A3B")
if not _QWEN36_DEFAULT.exists():
    _QWEN36_DEFAULT = ROOT / "models/Qwen3.6-35B-A3B"
QWEN36 = Path(os.environ.get("RNAQ_QWEN36_DIR", str(_QWEN36_DEFAULT)))
DATA = ROOT / "data/processed/rnaqa_n6_merged.jsonl"
HID = ROOT / "data/processed/_fsq_full_pos.pt"
HID_TENSOR = None  # 由 main 赋值；hid_slice 要用（保留兼容单个缓存的旧路径）
# 多缓存：uid -> (cache_key, offset, length)，以及 cache_key -> 张量
HIDS = {}
IDX = {}
CODES = None
CODE_IDX = {}


def register_cache(key, x, lens, offsets, uids):
    """把一份隐藏态缓存登记进全局索引。x 可以是 torch 张量或 np.memmap 的视图。"""
    HIDS[key] = x
    for i, u in enumerate(uids):
        IDX[u] = (key, int(offsets[i]), int(lens[i]))


def slice_hid(uid):
    ck, off, ln = IDX[uid]
    return HIDS[ck][off:off + ln].to(DEV, dtype=torch.float32)
POOL = 4          # 由 --pool 覆盖；per_token_logp 也要用，所以放模块级
TAU = 4.0         # 软分配温度；0 时由 --tau 自动取 latent/2
FSQ_CKPT = ROOT / "checkpoints/fsq_full/tokenizer.pt"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- FSQ

class FSQCode(nn.Module):
    """768 -> latent 维 -> FSQ 量化。同时给出离散码 id 和带 STE 的连续值。

    q_ste 是端到端的那条路：数值上等于量化结果，梯度按恒等映射回传。
    """

    def __init__(self, dim, latent, levels, d_model, pool=4):
        super().__init__()
        self.levels, self.latent, self.pool = levels, latent, pool
        # 学会的 4x 压缩：Conv1d(k=4, s=4)，初始化为平均池化。
        # 初始成平均池化而非随机 —— 第 0 步就有信息，同时保留可学性。
        # 学习式 pool 倍压缩。用 unfold+Linear 而不是 Conv1d：
        # 这个 ROCm 构建的 MIOpen 在 gfx1151 上对 Conv1d 直接报 miopenStatusInternalError。
        # Linear(pool*dim -> dim) 等价但更一般（窗口内可跨通道混合），且只走 matmul。
        self.compress = nn.Linear(pool * dim, dim, bias=False)
        with torch.no_grad():
            # 输入 x.reshape(n, pool*dim) 的一行布局是
            #   [位置0 的 dim 维][位置1 的 dim 维]...[位置(pool-1) 的 dim 维]
            # 所以通道 i 在窗口第 j 个位置上的值在第 j*dim + i 列，
            # 不是 i*pool+j。写成 i*pool:(i+1)*pool 会把通道搅乱（老版本就是这个错）。
            w = torch.zeros(dim, pool * dim)
            for j in range(pool):
                idx = torch.arange(dim)
                w[idx, j * dim + idx] = 1.0 / pool
            self.compress.weight.copy_(w)
        self.enc = nn.Linear(dim, latent)
        # enc 必须小初始化。ERNIE 隐藏态范数约 8.7，默认 std=0.02 会让 tanh 立刻饱和到 ±1，
        # 于是 (t+1)/2 恒等于同一个值 -> 所有位置同一个码 -> 熵归零（实测 ent -0.000）。
        # 注意：这个尺度与 levels 耦合。levels=4 时 (t+1)/2*3 只落在 1..2，
        # 四个 level 只用掉两个（实测 LATENT=3,LEVELS=4 只用到 8/64 个码）。
        nn.init.normal_(self.enc.weight, std=0.01)
        nn.init.zeros_(self.enc.bias)
        # FSQ 的网格点：码 c 的第 d 位数字 -> (k/(levels-1))*2-1，形状 (VOCAB, latent)。
        # 用来做软分配（下面 rna_vectors 里），它只提供梯度通路，不提供任何额外容量。
        grid = torch.zeros(levels ** latent, latent)
        for c in range(levels ** latent):
            x = c
            for d in range(latent - 1, -1, -1):        # 与 forward 的编码顺序一致（高位在前）
                grid[c, d] = (x % levels) / (levels - 1) * 2 - 1
                x //= levels
        self.register_buffer("grid", grid, persistent=False)

    def forward(self, x):
        L = x.shape[0]
        if L < self.pool:
            # 不足一个窗口时补到 pool 行（重复末行）。老版本写 x.repeat(pool,1)，
            # 只对 L==1 成立，L=2,3 时 reshape 会报错。
            x = torch.cat([x, x[-1:].repeat(self.pool - L, 1)], 0)
            L = self.pool
        n = L // self.pool
        x = x[:n * self.pool]
        x = self.compress(x.reshape(n, self.pool * x.shape[1]))
        t = torch.tanh(self.enc(x))                      # [-1,1]
        s = (t + 1) / 2 * (self.levels - 1)
        k = torch.round(s)
        q = k / (self.levels - 1) * 2 - 1
        q_ste = t + (q - t).detach()                     # straight-through
        code = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        for d in range(self.latent):
            code = code * self.levels + k[:, d].long()
        return code, q_ste, t, q


# ---------------------------------------------------------------- data

def load_records(limit=None, sort=False):
    """sort=False 时严格复现 48_ load_all() 的**文件顺序** ——
    隐藏态缓存的 offsets 就是按那个顺序建的，错一位整个靶子就错位。"""
    recs = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            s = r.get("sequence") or ""
            if not s or len(s) > MAX_NT:
                continue
            recs.append(r)
    if sort:
        recs.sort(key=lambda r: hashlib.md5(r["uid"].encode()).hexdigest())
    return recs if limit is None else recs[:limit]


class Tok:
    def __init__(self, tk):
        self._tk = tk
        self.base_vocab = tk.vocab_size
        self.pad_id = tk.pad_token_id if tk.pad_token_id is not None else tk.eos_token_id
        self.eos = tk.eos_token
        self.im_start = tk("<|im_start|>", add_special_tokens=False).input_ids
        self.im_end = tk("<|im_end|>", add_special_tokens=False).input_ids
        self.sys_ids = self.im_start + tk("system").input_ids + tk("\n").input_ids
        self.user_ids = self.im_end + self.im_start + tk("user").input_ids + tk("\n").input_ids
        self.assistant_ids = (self.im_end + self.im_start
                              + tk("assistant").input_ids + tk("\n").input_ids)
        # Qwen3 关思考：assistant 头后插一个空的 think 块。
        # 不加的话模型会先"想"再答，评测时对不上（实测输出 "THINK THE FIRST OF..."）。
        self.no_think = tk(" thinking\n\n<｜end▁of▁thinking｜>\n\n",
                           add_special_tokens=False).input_ids

    def enc(self, s):
        return self._tk(s, add_special_tokens=False).input_ids


# ---------------------------------------------------------------- model

class E2EModel(nn.Module):
    def __init__(self, lora_r, rna_emb_mode, latent=LATENT, levels=LEVELS, pool=4,
                 base="qwen3-8b", base_mode="int4", lora_preset="C", base_bits="bf16",
                 adapter="lora", sos_k=4, sos_gamma_max=2.5, sos_dropout=0.05):
        super().__init__()

        def _attach(base_m, targets, tag):
            """把适配器挂上去。lora = peft；sos = K 个正交子空间。

            SOS 的意义在**分模态**：route C 的输入是两个模态（码信道 + 文本提示），
            答案必须从码信道取，而语言能力来自底座。把它们挤进同一个 rank-8
            子空间会互相拉扯（文献：冲突梯度导致模态塌缩 2206.04496 /
            模态懒惰 2506.11550）。K 个正交子空间天然可以按功能分工，
            后面接 freetext（自由文本）时两个模态的需求分歧更大，收益应更明显。
            """
            if adapter == "sos":
                from rnaqwen.models.sos_lora import (replace_linear_with_sos,
                                                     count_sos_layers)
                rep = replace_linear_with_sos(
                    base_m, tuple(targets), lora_r, sos_k, lora_r * 2,
                    dropout=sos_dropout, gamma_max=sos_gamma_max)
                if not rep:
                    raise RuntimeError(
                        f"SOS 一层都没替换上（targets={targets}）—— "
                        f"静默没加适配器比报错更糟。检查 target 名是否匹配 nn.Linear")
                print(f"[sos] {tag}  替换 {len(rep)} 个 nn.Linear  "
                      f"K={sos_k} r_tot={lora_r} γ_max={sos_gamma_max} "
                      f"dropout={sos_dropout}  层数核对 {count_sos_layers(base_m)}")
                n_tr = sum(p.numel() for p in base_m.parameters() if p.requires_grad)
                if n_tr == 0:
                    raise RuntimeError("SOS 挂上后没有任何可训参数")
                print(f"[sos] 可训 {n_tr/1e6:.2f} M（与同预算的 peft LoRA 应相等）")
                return base_m
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(r=lora_r, lora_alpha=lora_r * 2, lora_dropout=0.0,
                             target_modules=targets, task_type="CAUSAL_LM")
            return get_peft_model(base_m, cfg)

        if base == "qwen36-35b":
            # Qwen3.6-35B-A3B：专家是 3D nn.Parameter，bnb 量化不到，
            # 必须流式装载 + 拆成 per-expert Linear。全部细节见 rnaqwen.qwen36。
            from rnaqwen.qwen36 import load_qwen36_text, lora_targets
            targets = lora_targets(lora_preset)
            print(f"[base] Qwen3.6-35B-A3B  mode={base_mode}  adapter={adapter}  "
                  f"targets={lora_preset}（{len(targets)} 项）")
            base_m, _text, _lmh, qstats = load_qwen36_text(mode=base_mode)
            print(f"[base] {qstats}")
            for p in base_m.parameters():
                p.requires_grad_(False)
            self.llm = _attach(base_m, targets, "Qwen3.6-35B")
            self.base_emb = base_m.model.language_model.embed_tokens
        else:
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
            # bf16 的 8B 权重 15.3 GB。和 35B 并跑时提交量会贴到上限
            # （实测：加载后余量从 20.3 GB 掉到 5.2 GB，然后被看门狗杀掉）。
            # 8-bit 只要 ~8 GB，余量翻倍。88_ 验过 peft 挂得上量化过的 Linear。
            qc = None
            if base_bits == "int8":
                qc = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
            elif base_bits == "int4":
                qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                        bnb_4bit_use_double_quant=True,
                                        bnb_4bit_compute_dtype=torch.bfloat16)
            kw = dict(device_map={"": DEV})
            if qc is not None:
                kw["quantization_config"] = qc
            else:
                kw["dtype"] = torch.bfloat16
            base_m = AutoModelForCausalLM.from_pretrained(str(QWEN), **kw)
            print(f"[base] Qwen3-8B  base_bits={base_bits}  adapter={adapter}"
                  + ("  （量化）" if qc is not None else "  （bf16）"))
            if adapter == "sos" and qc is not None:
                raise RuntimeError(
                    "SOSLoRALayer 只接受 nn.Linear，而 bnb 量化后全是 Linear8bitLt/"
                    "Linear4bit。SOS + 量化 8B 不兼容 —— 用 --base-bits bf16，"
                    "或者 35B 那条路（它默认只量化专家，注意力层保持 nn.Linear）")
            for p in base_m.parameters():
                p.requires_grad_(False)
            self.llm = _attach(base_m,
                               ["q_proj", "k_proj", "v_proj", "o_proj"], "Qwen3-8B")
            self.base_emb = base_m.get_input_embeddings()

        self.base_vocab = self.base_emb.weight.shape[0]
        d = self.base_emb.weight.shape[1]
        dev = self.base_emb.weight.device
        n_tr = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
        print(f"[lora] 可训参数 {n_tr/1e6:.2f} M   hidden {d}   "
              f"base_vocab {self.base_vocab}")
        self.mode = rna_emb_mode
        self.fsq = FSQCode(768, latent, levels, d, pool=pool).to(device=dev, dtype=torch.float32)
        # 码表是唯一的信息通道（FSQ 的梯度经软分配回传，见 rna_vectors）。
        # 不再有 head 那条连续旁路，所以这个配置是可证伪的。
        self.rna_emb = nn.Embedding(VOCAB, d).to(device=dev, dtype=torch.float32)
        nn.init.normal_(self.rna_emb.weight, std=0.02)
        self._usage_parts = None      # 每步由 reset_usage() 清空，见 rna_vectors

    def reset_usage(self):
        """每步训练前调用，否则 fsq_usage 会跨步累积，熵正则算的是历史平均。"""
        self._usage_parts = None
        self.fsq_usage = None

    def rna_vectors(self, x, perm=None, tau=4.0):
        """x: (N,768) -> (N,d) 给 LLM 的向量。

        老版本用 `v + head(q_ste)` 给 FSQ 传梯度 —— 那是一条**连续旁路**，
        梯度路径平滑，优化器会优先走它，硬查表退化成装饰。
        后果：置换/随机/常数码三个消融全落回 50%，**这个配置不可证伪**。

        修法：梯度走**软分配**，前向值仍是硬查表。
            v_hard = rna_emb[code]                      前向值
            v_soft = softmax(-||q_ste - grid_c||²/tau) @ rna_emb   对 q_ste 可微
            v = v_hard + v_soft - v_soft.detach()       前向 == v_hard，反向经 v_soft
        这样 FSQ 拿到梯度，但**没有多出任何容量**：码表是唯一的信息通道。

        perm 非 None 时把查表用的码 id 换成 perm[code]：表、边缘分布、尺度全不变，
        只切断「这个码来自什么 RNA 内容」——最干净的对照。
        """
        code, q_ste, t, q = self.fsq(x)
        if self.rna_emb is None:
            return q_ste, code, t, q
        idx = perm[code] if perm is not None else code
        v_hard = self.rna_emb(idx)
        g = self.fsq.grid                              # (VOCAB, latent)
        d2 = (q_ste.unsqueeze(1) - g.unsqueeze(0)).pow(2).sum(-1)   # (N, VOCAB)
        soft = torch.softmax(-d2 / tau, dim=-1)
        v_soft = soft @ self.rna_emb.weight
        out = v_hard + v_soft - v_soft.detach()
        # 给熵正则留一个**可微**的使用率。
        # 不能用 bincount(codes)：codes 是 round().long()，梯度为零，
        # 算出来的熵是个常数，乘任何系数都不会产生梯度 —— 熵正则等于没接。
        # 另外必须**累加**：rna_vectors 是逐条样本调用的，直接赋值的话
        # 只剩最后一条序列的使用率（而 ent 的打印用的是 append，两者不一致）。
        u = soft.mean(0)
        if self._usage_parts is None:
            self._usage_parts = []
        self._usage_parts.append(u)
        self.fsq_usage = torch.stack(self._usage_parts).mean(0)
        return out, code, t, q

    def rna_vectors_from_codes(self, code, perm=None):
        """Precomputed FSQ code IDs -> trainable code embeddings.

        This path deliberately freezes the FSQ encoder: the trained tokenizer is
        treated as a discrete RNA tokenizer. Only the LLM adapter and embedding
        table are trained.
        """
        idx = perm[code] if perm is not None else code
        return self.rna_emb(idx), code, None, None

    def embed(self, ids, rna_vecs):
        is_rna = ids >= self.base_vocab
        safe = ids.clamp(max=self.base_vocab - 1)
        e = self.base_emb(safe)
        if is_rna.any():
            e = e.clone()
            e[is_rna] = rna_vecs.to(e.dtype)
        return e


def vectors_for_uid(model, uid, perm=None, tau=TAU):
    """Return (vectors, codes, q_ste, target) from codes or hidden states."""
    if CODES is not None:
        c = CODE_IDX[uid]
        code = CODES[c[1]:c[1] + c[2]].to(DEV, dtype=torch.long)
        return model.rna_vectors_from_codes(code, perm=perm)
    return model.rna_vectors(slice_hid(uid), perm=perm, tau=tau)


# ---------------------------------------------------------------- batch

def build_batch(tk, examples, max_len=2048):
    seqs, labs = [], []
    for r, qa, clen, _off in examples:
        a = f" A: {qa['answer']}{tk.eos}"
        ids = (tk.sys_ids + tk.user_ids + tk.enc("Q: <RNA>")
               + [tk.base_vocab] * clen                # 占位，真实向量后面填
               + tk.enc(f"</RNA>{qa['question']}")
               + tk.assistant_ids + tk.no_think + tk.enc(a))
        # labels 从 assistant 头开始（含 <|im_end|><|im_start|>assistant\n）。
        # 这是**有意**的：生成时需要模型自己会输出 <|im_end|> 才停得下来。
        # 若只想训答案正文，改成 len(ids) - len(tk.enc(a))。
        n_head = len(ids) - len(tk.assistant_ids) - len(tk.enc(a))
        lab = [-100] * n_head + ids[n_head:]
        if len(ids) > max_len:
            raise RuntimeError(f"序列 {len(ids)} > {max_len}（码长 {clen}），答案会被截断")
        seqs.append(ids); labs.append(lab)
    mx = max(len(s) for s in seqs)
    ii = torch.full((len(seqs), mx), tk.pad_id, dtype=torch.long, device=DEV)
    ll = torch.full((len(seqs), mx), -100, dtype=torch.long, device=DEV)
    am = torch.zeros((len(seqs), mx), dtype=torch.long, device=DEV)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        ii[i, :len(s)] = torch.tensor(s, device=DEV)
        ll[i, :len(l)] = torch.tensor(l, device=DEV)
        # 按真实长度建 mask，**不能**用 (ii != pad_id)：
        # Qwen 的 pad_token_id 会回退到 eos，那样会把真正的 EOS 也屏蔽掉。
        am[i, :len(s)] = 1
    return ii, ll, am


# ---------------------------------------------------------------- eval

def answer_token_mask(tk, qa, lab):
    """返回与 `lab` 同形状的 bool 掩码，只标出**答案本身**的 token 位置。

    为什么必须单独标出来
    --------------------
    `run_ablation` 的 `frac_neg` 原来把所有非 -100 的标签位置汇到一起。而
    `build_batch` 的标签 span 是「no_think 的后 4 个 token + " A: 答案<eos>"」，
    实测平均只有 **9.89 个 token**，其中：

        no_think 尾巴（固定样板）  4     40.4%
        " A: " + eos（固定样板）  ~3.08  31.1%
        **答案**                 2.81   28.4%

    样板与码无关、天然 50/50，会把 frac_neg 拖向 50% —— 于是
    `50_` 里那条「零表/置换 < 45%」的判据在 seqtask 上永远通不过（假阴性）。
    实测：vszero 52.7% / vsperm 46.4%，而稀释下限就有 35.8%。

    **用 offset mapping 精确定位，不靠长度相减。** BPE 在边界上会合并：
    `len(enc(" A: ")) + len(enc(ans))` 未必等于 `len(enc(" A: " + ans))`。
    实测 ' A: CUGC<|im_end|>' 是 5 个 token，而 " A: " 自身占 3 个 —— 直接相减会错。
    """
    a = f" A: {qa['answer']}{tk.eos}"
    e = tk._tk(a, add_special_tokens=False, return_offsets_mapping=True)
    lo = len(" A: ")                       # 答案在字符串 a 里的字符起点
    hi = lo + len(qa["answer"])
    hit = [i for i, (s, en) in enumerate(e.offset_mapping) if en > lo and s < hi]
    m = torch.zeros_like(lab, dtype=torch.bool)
    if hit:
        # a 是序列的尾巴（build_batch 里 a 拼在最后），所以 a 的第 i 个 token
        # 在序列里的位置就是 (序列长 - len(enc(a))) + i
        base = lab.shape[1] - len(e.input_ids)
        for i in hit:
            m[0, base + i] = True
    return m


# ⚠️ 这个装饰器不能丢。丢了之后 model.rna_emb.weight.copy_(real_table) 会在
#    需要梯度的 leaf 上原地操作，报 "a leaf Variable that requires grad is being
#    used in an in-place operation" —— 而且**只在跑到消融那一步才炸**（训练正常），
#    2026-10-13 就是这么死在 step 4500 的消融上的。
#    起因是我用 `def per_token_logp(` 当锚点插新函数，把上面这行装饰器留给了新函数。
@torch.no_grad()
def per_token_logp(model, tk, examples, hid, perm=None, mode="real", max_n=25):
    """返回 (全部有标签位置的 logp, 用的码 id 集合, **仅答案位置**的 logp)。

    第三个返回值是后加的：判据要用答案那一小段，不能用整段标签 —— 见
    `answer_token_mask` 的注释。返回值从 2 个变成 3 个，调用点只有 run_ablation。

    perm != None 时：把码 c 换成 perm[c]（随机双射）。表、边缘分布、尺度全不变，
    只切断「码 <-> 内容」的对应。这是本脚本最关键的仪器。
    """
    examples = spread(examples, max_n)
    real_table = model.rna_emb.weight.detach().clone() if model.rna_emb is not None else None
    if mode == "zero" and model.rna_emb is not None:
        model.rna_emb.weight.zero_()
    elif mode == "random" and model.rna_emb is not None:
        g = torch.Generator(device="cpu").manual_seed(123)
        model.rna_emb.weight.copy_(
            torch.randn_like(real_table.cpu(), generator=g).to(real_table.device) * 0.02)
    elif mode == "const" and model.rna_emb is not None:
        model.rna_emb.weight.copy_(real_table[0].expand_as(real_table).clone())

    out, out_ans, used = [], [], set()
    for r, qa, clen, off in examples:
        v, code, _, _ = vectors_for_uid(model, r["uid"], perm=perm, tau=TAU)
        used |= set(code.tolist())
        ids, lab, am = build_batch(tk, [(r, qa, clen, off)])
        emask = answer_token_mask(tk, qa, lab)          # [1, T]
        emb = model.embed(ids, v)
        logits = model.llm(inputs_embeds=emb, attention_mask=am).logits.float()[:, :-1]
        tgt = lab[:, 1:]
        m = tgt >= 0
        if m.sum():
            lp = F.log_softmax(logits, -1).gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            out.extend(lp[m].tolist())
            # emask 在 lab 的坐标上，lp 在 tgt 的坐标上（tgt[j] = lab[j+1]），所以错开一位
            ma = emask[:, 1:] & m
            out_ans.extend(lp[ma].tolist())
    if model.rna_emb is not None:
        model.rna_emb.weight.copy_(real_table)
    return torch.tensor(out), used, torch.tensor(out_ans)


def run_ablation(model, tk, ev_ex, hid, max_n=25):
    """5 臂消融。每臂同时给**全部标签位置**和**只看答案位置**两套统计。

    为什么要两套：全部标签位置被样板稀释（答案只占 28.4%），frac_neg 会被
    样板噪声拖到接近 50%，45% 那条判据就失效。答案那一套才是判据该用的。
    两套都留着，是为了能直接看出"稀释了多少"——单看一套会再次踩坑。
    """
    real, _, real_a = per_token_logp(model, tk, ev_ex, hid, max_n=max_n)
    g = torch.Generator(device="cpu").manual_seed(2024)
    perm = torch.randperm(VOCAB, generator=g).to(DEV)
    res = {}
    for name, kw in (("zero", dict(mode="zero")),
                     ("random", dict(mode="random")),
                     ("const", dict(mode="const")),
                     ("perm", dict(perm=perm))):
        a, _, a_a = per_token_logp(model, tk, ev_ex, hid, max_n=max_n, **kw)
        n = min(len(real), len(a))
        d = real[:n] - a[:n]
        na = min(len(real_a), len(a_a))
        da = real_a[:na] - a_a[:na]
        res[name] = dict(
            n=n, mean=float(d.mean()), frac_neg=float((d < 0).float().mean()),
            n_ans=int(na),
            mean_ans=float(da.mean()) if na else float("nan"),
            frac_neg_ans=float((da < 0).float().mean()) if na else float("nan"),
        )
    return res


# ---------------------------------------------------------------- 早停

def spread(ex_list, n):
    """从整条列表里等距取 n 条，而不是取前 n 条。

    **为什么不能取前 n 条**：数据文件是按来源顺序写的（seq8 在前，rfam/rfa 在后）。
    取前 n 条 = 只在 seq8 上监视，而训练数据 85% 是 rfa。
    等距抽样才能覆盖到全部来源。
    """
    if n <= 0 or len(ex_list) <= n:
        return ex_list
    step = len(ex_list) / n
    return [ex_list[min(int(i * step), len(ex_list) - 1)] for i in range(n)]


def sample_batches(bucket_indices, eff_len, batch_size, token_budget, seed):
    """Yield token-packed batches; each row appears once per epoch.

    The old sampler drew every row with replacement.  With 46,500 steps that
    repeated some rows many times while leaving a large part of train unseen.
    Bucket order and within-bucket order are shuffled, but the cursor advances
    without replacement until the epoch is exhausted.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    while True:
        for bi in torch.randperm(len(bucket_indices), generator=gen).tolist():
            bucket = bucket_indices[bi]
            order = torch.randperm(len(bucket), generator=gen).tolist()
            cur, tot, mx = [], 0, 0
            for j in order:
                i = bucket[j]
                eff = eff_len[i]
                nmx = max(mx, eff)
                if cur and (tot + eff > token_budget
                            or nmx * (len(cur) + 1) > token_budget):
                    yield cur
                    cur, tot, mx = [], 0, 0
                    nmx = eff
                cur.append(i)
                tot += eff
                mx = nmx
                if len(cur) >= batch_size:
                    yield cur
                    cur, tot, mx = [], 0, 0
            if cur:
                yield cur


@torch.no_grad()
def val_ce(model, tk, ex_list, arm="A", max_n=25):
    """val 上答案 token 的**逐样本平均** CE，以及它的标准误。

    返回 (mean, se)。

    **为什么必须要 se**：早停判据是"CE 改善了 min_delta 才算数"。但如果估计量本身
    比 min_delta 还抖，判据就是纯噪声 —— patience 会在 min_steps 一过就被耗尽，
    在随机点上早停。
    实测量级：10 个模板每样本 2~6 个答案 token，eval-n=12 时总共约 48 个 token；
    逐 token CE 标准差约 1.5 -> 均值标准误约 1.5/sqrt(48) ≈ 0.22。
    而 min_delta=0.005 在 CE≈1.0 上要求分辨 0.005 —— **信噪比差 40 倍**。
    所以判据里必须有 se 这一项（见下面的调用处）。

    arm A = 真实 RNA 码向量；arm C = 码向量置零。
    两个都要看：单看 CE_A 会被"模型学会了问题的先验"骗过去，
    C 臂把先验单独量出来，健康信号是 gap = CE_C - CE_A 在扩大。
    """
    per_ex = []
    for r, qa, clen, off in spread(ex_list, max_n):
        v, _, _, _ = vectors_for_uid(model, r.get("uid"), tau=TAU)
        if arm == "C":
            v = torch.zeros_like(v)
        ids, lab, am = build_batch(tk, [(r, qa, clen, off)])
        emb = model.embed(ids, v)
        lg = model.llm(inputs_embeds=emb, attention_mask=am).logits.float()[:, :-1]
        tgt = lab[:, 1:]
        m = tgt >= 0
        answer_mask = answer_token_mask(tk, qa, lab)[:, 1:] & m
        if answer_mask.any():
            lp = F.log_softmax(lg, -1).gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            per_ex.append(float(-lp[answer_mask].mean()))
    if not per_ex:
        return float("nan"), float("nan")
    mu = sum(per_ex) / len(per_ex)
    if len(per_ex) < 2:
        return mu, float("inf")
    var = sum((x - mu) ** 2 for x in per_ex) / (len(per_ex) - 1)
    return mu, (var / len(per_ex)) ** 0.5




# ---------------------------------------------------------------- 诊断套件

@torch.no_grad()
def _logp_of(model, tk, ex_list):
    """teacher-forced 答案 token 的 logp 与码频次。

    **必须传 uid。** `off` 是该记录在自己那份缓存内部的偏移：
    rfam/rfa 的 off 指向副缓存（最大 4700 万），而 hid_slice 不传 uid 时会退回
    主缓存（2,018,834 位置）—— 读的是完全无关的位置，而且不会报错。
    之前这里漏了 uid，因为 diagnose 只取 ev_ex[:40]、而数据文件恰好把 seq8
    写在最前面（主缓存），所以一直没暴露。
    """
    out, code_freq = [], {}
    for r, qa, clen, off in ex_list:
        v, code, _, _ = vectors_for_uid(model, r.get("uid"), tau=TAU)
        if len(v) != clen:
            raise RuntimeError(
                f"RNA code length mismatch: got {len(v)}, placeholder expects {clen} "
                f"(uid={r.get('uid')})"
            )
        for c in code.tolist():
            code_freq[c] = code_freq.get(c, 0) + 1
        ids, lab, am = build_batch(tk, [(r, qa, clen, off)])
        emb = model.embed(ids, v)
        lg = model.llm(inputs_embeds=emb, attention_mask=am).logits.float()[:, :-1]
        tgt = lab[:, 1:]
        m = tgt >= 0
        answer_mask = answer_token_mask(tk, qa, lab)[:, 1:] & m
        if answer_mask.any():
            lp = F.log_softmax(lg, -1).gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            out.extend(lp[answer_mask].tolist())
    return torch.tensor(out), code_freq


def hid_slice(off, clen, uid=None):
    """uid 给了就走多缓存索引；否则退回单缓存（旧路径）。"""
    if uid is not None and uid in IDX:
        return slice_hid(uid)
    return HID_TENSOR[off:off + clen * POOL].to(DEV, dtype=torch.float32)


@torch.no_grad()
def diagnose(model, tk, ev_ex, max_n=40):
    """一次跑完所有诊断。返回 dict，全部是可比的标量。"""
    ex = spread(ev_ex, max_n)
    n = len(ex)
    if n < 4:
        return {}
    rep = {}

    # ---- 1. 基准 ----
    real, code_freq = _logp_of(model, tk, ex)
    L = len(real)
    rep["n_answer_tokens"] = L

    # ---- 2. 换 RNA：把第 i 条的 RNA 换成第 j 条的 ----
    g = torch.Generator(device="cpu").manual_seed(7)
    perm = torch.randperm(n, generator=g).tolist()
    paired = [(ex[perm[i]][0], ex[i][1], ex[perm[i]][2], ex[perm[i]][3])
              for i in range(n)]
    sw_rna, _ = _logp_of(model, tk, paired)
    d_rna = real[:min(L, len(sw_rna))] - sw_rna[:min(L, len(sw_rna))]

    # ---- 3. 换问题：把第 i 条的问题文字换成第 j 条的 ----
    sw_q_list = []
    for i in range(n):
        rr, qa, cl, of = ex[i]
        qa2 = dict(qa); qa2["question"] = ex[perm[i]][1]["question"]
        sw_q_list.append((rr, qa2, cl, of))
    sw_q, _ = _logp_of(model, tk, sw_q_list)
    d_q = real[:min(L, len(sw_q))] - sw_q[:min(L, len(sw_q))]

    # ---- 4. 零表 ----
    tbl = model.rna_emb.weight.detach().clone()
    model.rna_emb.weight.zero_()
    z, _ = _logp_of(model, tk, ex)
    model.rna_emb.weight.copy_(tbl)
    d_z = real[:min(L, len(z))] - z[:min(L, len(z))]

    rep["d_swap_rna"] = float(d_rna.abs().mean())
    rep["d_swap_q"] = float(d_q.abs().mean())
    rep["d_zero_table"] = float(d_z.abs().mean())
    # 关键比值：RNA 的影响力相对问题有多小
    rep["rna_vs_question"] = rep["d_swap_rna"] / max(rep["d_swap_q"], 1e-9)

    # ---- 5. 码本健康度（塌缩检测，在评测集上统计）----
    ids = torch.tensor(sorted(code_freq), dtype=torch.long)
    vals = torch.tensor([code_freq[int(i)] for i in ids], dtype=torch.float32)
    cnt = torch.zeros(VOCAB, dtype=torch.float32)
    cnt.index_add_(0, ids, vals)
    p = cnt / cnt.sum().clamp(min=1)
    rep["codes_used"] = int((cnt > 0).sum())
    rep["code_entropy"] = float(-(p * (p + 1e-9).log()).sum())
    rep["code_entropy_max"] = float(torch.log(torch.tensor(float(VOCAB))))
    rep["top_code_share"] = float(p.max())

    # ---- 6. 码表健康度（行间余弦 + 有效秩）----
    u = ids.to(model.rna_emb.weight.device)
    w = model.rna_emb.weight[u].float()
    rep["table_row_norm"] = float(w.norm(dim=1).mean())
    if len(u) > 1:
        wn = F.normalize(w, dim=1)
        c = wn @ wn.t()
        off = c - torch.eye(len(u), device=c.device)
        rep["table_mean_cos"] = float(off.sum() / (len(u) * (len(u) - 1)))
        sv = torch.linalg.svdvals(wn - wn.mean(0, keepdim=True))
        rep["table_eff_rank"] = float(sv.sum() ** 2 / (sv ** 2).sum().clamp(min=1e-9))

    # ---- 7. FSQ 使用率（软，可微的那个量）----
    if getattr(model, "fsq_usage", None) is not None:
        uu = model.fsq_usage / model.fsq_usage.sum().clamp(min=1e-9)
        rep["soft_usage_entropy"] = float(-(uu * (uu + 1e-9).log()).sum())
    return rep


def print_diag(step, rep):
    if not rep:
        return
    print(f"  [diag@{step}] "
          f"Δ换RNA {rep['d_swap_rna']:.4f}  Δ换问题 {rep['d_swap_q']:.4f}  "
          f"Δ零表 {rep['d_zero_table']:.4f}  "
          f"**RNA/问题 = {rep['rna_vs_question']:.3f}**", flush=True)
    print(f"            码本: 用到 {rep['codes_used']}/{VOCAB}  "
          f"熵 {rep['code_entropy']:.2f}/{rep['code_entropy_max']:.2f}  "
          f"最大码占比 {100*rep['top_code_share']:.1f}%   "
          f"码表: 行范数 {rep['table_row_norm']:.3f}  "
          f"行间余弦 {rep.get('table_mean_cos', float('nan')):+.4f}  "
          f"有效秩 {rep.get('table_eff_rank', float('nan')):.1f}", flush=True)
    # 判读：RNA/问题 < 0.2 基本等于模型不看 RNA
    r = rep["rna_vs_question"]
    if r < 0.1:
        print(f"            !! 换掉 RNA 几乎没有影响（{r:.3f}）—— 模型在忽略 RNA", flush=True)
    elif r < 0.35:
        print(f"            ~ RNA 有影响但远弱于问题（{r:.3f}）", flush=True)
    else:
        print(f"            OK RNA 的影响力与问题同量级（{r:.3f}）", flush=True)




# ---------------------------------------------------------------- checkpoint

def adapter_state_dict(model) -> dict:
    """取出**所有可训练参数**，不靠名字过滤。

    原来写死的是 `if "lora_" in k` —— 那只对 peft 的命名有效。
    SOS-LoRA 的参数是 `nn.ParameterList`，key 长成 `...q_proj.A.0` / `...q_proj.B.0`，
    **一个 `lora_` 都没有**。于是 checkpoint 里一个适配器参数都存不下，
    而训练照跑、loss 照降、日志一切正常 —— 静默失效。

    判据改成「这个参数 requires_grad 吗」，与适配器种类无关。
    """
    llm = getattr(model, "llm", None)
    if llm is None:
        return {}
    tw = {n for n, p in llm.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in llm.state_dict().items() if k in tw}


def load_adapter_state(model, state: dict, label: str) -> None:
    """Load exactly the trainable adapter keys, refusing partial matches."""
    want = {n for n, p in model.llm.named_parameters() if p.requires_grad}
    got = set(state or {})
    model_keys = set(model.llm.state_dict())
    missing = want - got
    unexpected = got - model_keys
    if not got or missing or unexpected:
        raise RuntimeError(
            f"{label} 适配器不匹配：checkpoint={len(got)} 张量，"
            f"缺失 {len(missing)}（{sorted(missing)[:3]}），"
            f"多余 {len(unexpected)}（{sorted(unexpected)[:3]}）。"
            "拒绝静默从随机初始化续训。"
        )
    res = model.llm.load_state_dict(state, strict=False)
    if res.unexpected_keys:
        raise RuntimeError(
            f"{label} 适配器仍有 unexpected keys: {res.unexpected_keys[:3]}"
        )


def checkpoint_arg_mismatches(old: dict, current) -> list[str]:
    keys = ("base", "qwen36_mode", "adapter", "lora_r", "lora_preset",
            "sos_k", "sos_gamma_max", "pool", "rna_emb",
            "answer_only_loss", "seed")
    bad = []
    for key in keys:
        if key not in old:
            continue
        a, b = old[key], getattr(current, key, None)
        if isinstance(a, float) or isinstance(b, float):
            same = a is not None and b is not None and abs(float(a) - float(b)) < 1e-12
        else:
            same = str(a) == str(b)
        if not same:
            bad.append(f"{key}: checkpoint={a!r} current={b!r}")
    if bool(old.get("codes")) != bool(getattr(current, "codes", "")):
        bad.append(f"codes: checkpoint={bool(old.get('codes'))!r} "
                   f"current={bool(getattr(current, 'codes', ''))!r}")
    return bad


def save_ckpt(path: Path, step: int, model, opt, log, args, t_elapsed: float):
    """完整状态。只存适配器参数（底座冻结，存了也没用而且几十 GB）。

    **原子落盘**：先写 `*.pt.tmp`，验过 zip 中央目录，再 os.replace 顶上正式名。

    为什么非这样不可：torch.save 直写目标文件时，写到一半被 OOM 杀掉会留下一个
    截断的 .pt，而它**看起来是合法的** —— 文件在、后缀对、体积也不离谱。
    实测踩过：step_000600.pt 254.9 MB（正常 322 MB），torch.load 报
    "failed reading zip archive: failed finding central directory"。
    更糟的是 resume 用 cands[-1] 按名字排序，正好会挑中这个坏文件，
    于是"续训"直接从崩溃点炸掉，人还以为是新问题。

    原子写保证正式名**永远**指向一个完整文件：要么是新的，要么是上一个。
    中途被杀只留一个 .tmp，prune_ckpts 下次会清掉。
    """
    lora = adapter_state_dict(model)
    if not lora:
        raise RuntimeError(f"{path.name}: 适配器参数为空，拒绝写出不可续训的 checkpoint")
    payload = dict(
        step=step,
        fsq=model.fsq.state_dict(),
        rna_emb=model.rna_emb.weight.detach().cpu() if model.rna_emb is not None else None,
        lora=lora,
        opt=opt.state_dict(),
        log=log,
        args=vars(args),
        t_elapsed=t_elapsed,
        rng=torch.get_rng_state(),
        rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        torch.save(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    # 能列出中央目录 = 整个 zip 是完整的。比 testzip() 便宜得多（不动数据段），
    # 而且恰好覆盖"写到一半被截断"这个失败模式。
    with zipfile.ZipFile(tmp) as zf:
        n_entry = len(zf.namelist())
    if n_entry == 0:
        raise RuntimeError(f"{tmp.name} 写出来是空 zip，拒绝顶替 {path.name}")
    os.replace(tmp, path)


def load_ckpt(path: Path, model, opt, args, allow_opt_reset: bool = False):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    old_args = ck.get("args") or {}
    bad = checkpoint_arg_mismatches(old_args, args)
    if bad:
        raise RuntimeError(f"{path.name} 与当前训练配置不兼容：\n  " + "\n  ".join(bad))
    old_fp = old_args.get("fingerprint") or {}
    new_fp = getattr(args, "fingerprint", {}) or {}
    fp_bad = [f"{k}: checkpoint={old_fp.get(k)} current={new_fp.get(k)}"
              for k in ("dataset_sha256", "codes_sha256", "fsq_sha256", "hid_sha256")
              if old_fp.get(k) and new_fp.get(k) and old_fp[k] != new_fp[k]]
    if fp_bad:
        raise RuntimeError(f"{path.name} 的数据/码缓存/FSQ 指纹不一致：\n  "
                           + "\n  ".join(fp_bad))
    if getattr(args, "codes", ""):
        cur_fsq = model.fsq.state_dict()
        ck_fsq = ck["fsq"]
        fsq_bad = [k for k in cur_fsq
                   if k not in ck_fsq or cur_fsq[k].shape != ck_fsq[k].shape
                   or not torch.allclose(cur_fsq[k].cpu(), ck_fsq[k].cpu())]
        if fsq_bad:
            raise RuntimeError(
                f"{path.name} 的 FSQ 与 --fsq 不是同一套：{fsq_bad[:3]}。"
                "预计算码路径拒绝覆盖当前冻结 FSQ。"
            )
    else:
        model.fsq.load_state_dict(ck["fsq"])
    if model.rna_emb is not None:
        if ck["rna_emb"] is None:
            raise RuntimeError(f"{path.name} 缺少 rna_emb，拒绝只恢复半套状态")
        if tuple(ck["rna_emb"].shape) != tuple(model.rna_emb.weight.shape):
            raise RuntimeError(
                f"{path.name} rna_emb 形状 {tuple(ck['rna_emb'].shape)} != "
                f"当前 {tuple(model.rna_emb.weight.shape)}")
        model.rna_emb.weight.data.copy_(
            ck["rna_emb"].to(device=model.rna_emb.weight.device,
                             dtype=model.rna_emb.weight.dtype))
    load_adapter_state(model, ck["lora"], path.name)
    torch.set_rng_state(ck["rng"])
    if ck.get("rng_cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ck["rng_cuda"])
    try:
        opt.load_state_dict(ck["opt"])
        opt_ok = True
    except Exception as e:
        if not allow_opt_reset:
            raise RuntimeError(
                f"{path.name} 优化器状态载入失败（{type(e).__name__}: {e}）。"
                "默认拒绝静默重置；确认要重置时显式加 --allow-opt-reset 1。"
            ) from e
        opt_ok = False
        print(f"  [resume] 优化器状态载入失败（{type(e).__name__}），按参数要求从零开始")
    return ck["step"], ck.get("log", []), ck.get("t_elapsed", 0.0), opt_ok


def prune_ckpts(d: Path, keep: int) -> int:
    """只留最近 keep 个（按 step 排序），返回清掉了几个。

    顺带清掉 `*.pt.tmp` —— 那是上一次被中途杀掉留下的半成品。
    不碰 best.pt / final.pt：它们不匹配 `step_*.pt`，也不参与轮转，
    是整个训练里唯一「不可再生」的两个档（best 是早停要载回的，
    final 是最终产物）。
    """
    for t in d.glob("*.pt.tmp"):
        try:
            t.unlink()
        except OSError:
            pass
    fs = sorted(d.glob("step_*.pt"),
                key=lambda x: int(x.stem.split("_")[1]))
    n = 0
    if keep > 0:
        for f in fs[:-keep]:
            try:
                f.unlink()
                n += 1
            except OSError:
                pass
    return n


# ---------------------------------------------------------------- 省内存的 loss

def _lm_parts(model):
    """返回 (文本主体, lm_head)。两种底座的结构不同，这里统一。

    35B 走 VLM 类 Qwen3_5MoeForConditionalGeneration：
        model.llm.model.language_model    +  model.llm.lm_head
        （权重名就是 model.language_model.* 和 lm_head.weight）
    8B 走 AutoModelForCausalLM：
        model.llm.model                   +  model.llm.lm_head
    用 peft 时外面还套一层，再剥掉 base_model。
    """
    llm = model.llm
    # ⚠️ 不能用 hasattr(llm, "base_model") 判断 peft —— transformers 里**任何**
    #    PreTrainedModel 都有 base_model 属性（Qwen3ForCausalLM.base_model 就返回
    #    自己的 .model），那会把所有模型都误剥一层，报
    #    "'Qwen3Model' object has no attribute 'model'"。PeftModel 的标志是 peft_config。
    if hasattr(llm, "peft_config"):
        llm = getattr(llm.base_model, "model", llm.base_model)
    body = llm.model
    if hasattr(body, "language_model"):                # VLM
        return body.language_model, llm.lm_head
    return body, llm.lm_head


def answer_only_loss(model, emb, am, lab):
    """只在**有标签的位置**算 logits。数学上等于 HF 的 out.loss，但省掉全序列 logits。

    为什么必须自己写
    ----------------
    HF 的 ForCausalLM 会先算出 `[B, T, V]` 的 logits，再上采样到 float32 交给
    CrossEntropyLoss，而反向还要把这两份都留着。B=8 / T=1200 / V=248320 时：

        logits  bf16   8 x 1200 x 248320 x 2B = 4.77 GB
        .float() fp32  同形状                  = 9.53 GB
        反向保存                                = 两份都留

    而 `lab` 里非 -100 的只有助手尾部那一段（assistant 头 + no_think + 答案 + eos，
    见 build_batch），**每条约 25 个 token**。8 条 = 200 个位置：

        200 x 248320 x 4B = 0.20 GB

    差两个数量级。这不是优化，是原来算了 48 倍用不到的东西。

    对齐方式与 HF 一致：**位置 t 的 logits 预测标签 t+1**，所以正文取 [:, :-1]、
    标签取 [:, 1:]。改错这里不会报错，只会训出一个 loss 看着正常但语义错位的模型 ——
    所以训练脚本里有 --verify-loss，前几步会把本函数和 out.loss 逐位比。
    """
    body, head = _lm_parts(model)
    hs = body(inputs_embeds=emb, attention_mask=am, use_cache=False).last_hidden_state[:, :-1]
    tgt = lab[:, 1:]
    m = tgt >= 0
    if not bool(m.any()):
        # 没有任何标签时也要给一个**可反传**的零。返回 Python 的 0.0 会让
        # 后面的 loss.backward() 直接炸（"element 0 of tensors does not require grad"）。
        return hs.sum() * 0.0
    sel = m.nonzero(as_tuple=False)                    # [N, 2]
    h = hs[sel[:, 0], sel[:, 1]]                       # [N, H]
    lg = head(h).float()                               # [N, V]
    return F.cross_entropy(lg, tgt[sel[:, 0], sel[:, 1]])


# ---------------------------------------------------------------- main

def main() -> int:
    # global 必须在函数内首次使用之前声明，所以放最前面
    global LATENT, LEVELS, VOCAB
    global POOL, TAU

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rna", type=int, default=4000)
    ap.add_argument("--eval-rna", type=int, default=481)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--eval-every", type=int, default=150)
    ap.add_argument("--log-every", type=int, default=200,
                    help="每 N 步打一行日志。固定值，不跟总步数挂钩 —— steps//10 会让长训练几小时没输出")
    ap.add_argument("--profile-steps", type=int, default=0,
                    help="前 N 步输出 batch/forward/backward/optimizer 分段时间，0=关闭")
    ap.add_argument("--eval-n", type=int, default=25)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--rna-emb", choices=["table"], default="table",
                    help="只有码表一条通路；head 连续旁路已删除（它使实验不可证伪）")
    ap.add_argument("--tau", type=float, default=0.0,
                    help="软分配温度（只影响梯度）。0 = 自动取 latent/2。"
                         "q_ste 与 grid 都在 [-1,1]^latent，d2 范围约 0~4*latent；"
                         "tau 太小时 softmax 退化成 one-hot 把梯度掐死"
                         "（实测 tau=0.1 时 enc 梯度 1.8e-15，等于零）")
    ap.add_argument("--ent-coef", type=float, default=0.01, help="码使用熵正则（复用压力）")
    ap.add_argument("--commit-coef", type=float, default=0.01, help="commitment: ||t-q||^2")
    ap.add_argument("--ortho-coef", type=float, default=0.01, help="码行间余弦^2（防表塌缩）")
    ap.add_argument("--tag", default="")
    ap.add_argument("--pool", type=int, default=4,
                    help="学习式压缩倍率: Conv1d(stride=pool)，初始化为平均池化")
    ap.add_argument("--token-budget", type=int, default=6000,
                    help="每步 B*T 上限：B=样例数，T=压缩码长+问题/答案文本 token。"
                         "只数码长的话，长答案的数据会把 B*T 顶到几倍大 -> OOM")
    ap.add_argument("--grad-ckpt", type=int, default=0,
                    help="1 = 打开 gradient checkpointing。40GB 的 A100 上 int4 底座占 "
                         "18.2 GiB，剩下的激活空间不够 B*T=2048；打开后用约 30%% 速度"
                         "换 2-4 倍的 token-budget。冻结底座 + LoRA/SOS 时必须同时 "
                         "enable_input_require_grads，否则梯度断在 inputs_embeds 上。")
    ap.add_argument("--bucket", type=int, default=256,
                    help="长度分桶的桶大小；0=不分桶")
    ap.add_argument("--save-dir", default="", help="最终产物目录")
    ap.add_argument("--save-every", type=int, default=500,
                    help="每 N 步存一个 checkpoint（含优化器状态）；0 = 不存")
    ap.add_argument("--keep-ckpt", type=int, default=3, help="只保留最近 N 个")
    ap.add_argument("--save-minutes", type=float, default=0.0,
                    help="墙上时钟兜底：距上次存盘超过这么多分钟就强制存一次"
                         "（0 = 关）。存盘本来只看步数，而 35B 实测 18.5 s/步，"
                         "--save-every 300 等于 92 分钟才存一次 —— 崩一次最多丢 1.5 小时。"
                         "按时间兜底后，丢失窗口与步速无关，有硬上限")
    ap.add_argument("--resume", default="", help="从 checkpoint 续训（目录或 .pt 文件）")
    ap.add_argument("--allow-opt-reset", type=int, default=0,
                    help="1 = checkpoint 优化器状态不兼容时允许从零恢复（默认拒绝）")
    ap.add_argument("--warm-lora", default="",
                    help="只加载旧 checkpoint 的 LoRA 权重（不动 FSQ / rna_emb / 优化器）。"
                         "用于换码表后的热启动：让「读码」的行为继承下来，只重学码表映射。")
    ap.add_argument("--answer-only-loss", type=int, default=1,
                    help="1=只在有标签的位置算 logits（默认）。"
                         "HF 内置 loss 会算 [B,T,248320] 的全序列 logits 再上采样 fp32，"
                         "B=8/T=1200 时是 4.77+9.53 GB 且反向保留；而每批真正有标签的"
                         "只有约 200 个位置 = 0.2 GB。0=退回 out.loss（对照用）")
    ap.add_argument("--verify-loss", type=int, default=0,
                    help="前 N 步同时算手工 loss 和 out.loss 并逐位比对，差 >1e-3 直接抛。"
                         "换了 loss 实现就必须开，否则错位是静默的")
    ap.add_argument("--hid2", default="", help="第二份隐藏态 memmap（如 _rfam_hidden.f16）")
    ap.add_argument("--hid2-idx", default="", help="第二份的索引 .pt")
    ap.add_argument("--hid", default="",
                    help="主隐藏态缓存。留空且给了 --hid2 时跳过旧主缓存，"
                         "适合只加载按新数据生成的一份 cache。")
    ap.add_argument("--codes", default="",
                    help="预计算 FSQ 码 ID 的 uint16 memmap。给定时冻结 FSQ，"
                         "只训练 LLM adapter 和码 embedding。")
    ap.add_argument("--codes-idx", default="",
                    help="--codes 对应的 uid/offset/length 索引。")
    ap.add_argument("--diag", type=int, default=1, help="每个 eval 点跑诊断套件")
    ap.add_argument("--diag-n", type=int, default=40, help="诊断用多少条样本")
    ap.add_argument("--seed", type=int, default=20261101,
                    help="不放回 bucket sampler 的确定性种子")
    # ---- 换底座 ----
    ap.add_argument("--base", choices=["qwen3-8b", "qwen36-35b"], default="qwen3-8b")
    ap.add_argument("--qwen36-mode", choices=["int4", "int8"], default="int4",
                    help="Qwen3.6 专家的量化位数。int8 显存 34.7 GiB，int4 18.3 GiB")
    ap.add_argument("--lora-preset", default="C",
                    help="Qwen3.6 的 LoRA target 预设 A/B/C/D（见 rnaqwen.qwen36），"
                         "或逗号分隔的模块名。C = 190 模块 / 8.13 M")
    ap.add_argument("--base-bits", choices=["bf16", "int8", "int4"], default="bf16",
                    help="8B 底座的精度。bf16 权重 15.3 GB；和 35B 并跑时会贴到提交上限，"
                         "8-bit 只要 ~8 GB")
    ap.add_argument("--adapter", choices=["lora", "sos"], default="lora",
                    help="适配器种类。lora = peft 标准 LoRA；"
                         "sos = SOS-LoRA（arXiv 2607.16252v1，K 个静态正交子空间 + "
                         "固定多尺度，同参数量、可合并）。"
                         "**SOS 只能挂在 nn.Linear 上，所以和 --base-bits int8/int4 的 8B 不兼容**")
    ap.add_argument("--sos-k", type=int, default=4, help="SOS 专家数 K，r_tot 必须能被它整除")
    ap.add_argument("--sos-gamma-max", type=float, default=2.5, help="多尺度上界 γ_max")
    ap.add_argument("--sos-dropout", type=float, default=0.05, help="SOS 的 LoRA dropout")
    ap.add_argument("--sos-ortho-coef", type=float, default=1.0,
                    help="跨专家正交正则的权重，加进 loss")
    # ---- 早停 ----
    ap.add_argument("--early-stop", type=int, default=0,
                    help="1 = 按监视指标早停，代替固定步数")
    ap.add_argument("--min-delta", type=float, default=0.005,
                    help="相对改善小于这个比例视为无改善")
    ap.add_argument("--patience", type=int, default=6, help="连续多少个 eval 点无改善就停")
    ap.add_argument("--min-steps", type=int, default=4000,
                    help="早停生效前至少训这么多步（防早期噪声）")
    ap.add_argument("--ablation-every", type=int, default=-1,
                    help="每隔多少步跑一次 5 臂消融。-1 = 跟 eval-every 同步（旧行为）；"
                         "0 = 训练中不跑，只在最后跑。35B 上 5 臂 x eval-n 次前向很贵")
    ap.add_argument("--trim-every", type=int, default=100,
                    help="每多少步修剪一次工作集。**用 mmap 副缓存时必须 > 0**："
                         "slice_hid 会把 memmap 的页 fault 进工作集，"
                         "而 WorkingSet 计入常驻的文件映射页 —— 不修剪会一路涨到把机器压死，"
                         "表现是硬崩、没有异常栈（和 84_ 装载时同一个机制）")
    ap.add_argument("--data", default="",
                    help="新格式数据（一条 QA 一行，带 uid/sequence/question/answer/split）。"
                         "给了就用它，否则用旧的 rnaqa_n6_merged 的 qa 数组格式")
    ap.add_argument("--init-code-emb", default="",
                    help="按内容初始化的码表 checkpoints/fsq_full/code_emb_contentinit.pt")
    ap.add_argument("--fsq", default="",
                    help="换码表检查点（默认 checkpoints/fsq_full/tokenizer.pt）。"
                         "换码表时必须同时用 --init-code-emb 给配套的内容初始化码表，"
                         "否则 rna_emb 与码 ID 语义错位。")
    args = ap.parse_args()
    if bool(args.codes) != bool(args.codes_idx):
        raise SystemExit("--codes 与 --codes-idx 必须同时提供")
    if args.fsq:
        global FSQ_CKPT
        FSQ_CKPT = Path(args.fsq)
        print(f"[cfg] FSQ 检查点改为 {FSQ_CKPT}")
    if (args.fsq or args.codes) and not FSQ_CKPT.is_file():
        raise SystemExit(f"FSQ 检查点不存在：{FSQ_CKPT}")
    fsq_meta = None
    if FSQ_CKPT.is_file():
        fsq_meta = torch.load(FSQ_CKPT, map_location="cpu", weights_only=False)
        LATENT = int(fsq_meta["latent"])
        LEVELS = int(fsq_meta["levels"])
        VOCAB = LEVELS ** LATENT
        print(f"[cfg] 从 {FSQ_CKPT.name} 读：LATENT={LATENT} LEVELS={LEVELS} VOCAB={VOCAB}")
    args.fingerprint = {"fsq_sha256": sha256_file(FSQ_CKPT)} if FSQ_CKPT.is_file() else {}
    if args.data:
        data_path = Path(args.data)
        if not data_path.is_file():
            raise SystemExit(f"数据文件不存在：{data_path}")
        args.fingerprint["dataset_sha256"] = sha256_file(data_path)
    if args.smoke:
        args.rna, args.eval_rna, args.steps = 200, 40, 40
        args.eval_every = 20; args.eval_n = 8

    tag = args.tag or ("smoke" if args.smoke else f"{args.rna}_s{args.steps}_{args.rna_emb}")
    outdir = RUNS_DIR / f"e2e_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    frozen = "disabled (frozen FSQ)" if args.codes else args.ent_coef
    commit = "disabled (frozen FSQ)" if args.codes else args.commit_coef
    print(f"[out] {outdir}\n[rna-emb] {args.rna_emb}  ent {frozen}  "
          f"commit {commit}  ortho {args.ortho_coef}")

    global HID_TENSOR, CODES, CODE_IDX
    hid, lens, offsets = None, [], []
    allr = []
    off_of, len_of = {}, {}
    use_primary = False
    if args.codes:
        import numpy as np
        print(f"[codes] 1/3 torch.load 索引 {args.codes_idx}", flush=True)
        dc = torch.load(args.codes_idx, map_location="cpu", weights_only=False)
        n_pos = int(dc["n_pos"])
        code_bytes = Path(args.codes).stat().st_size
        checks = {
            "n_pos": n_pos == int(dc["lens"].sum()),
            "bytes": code_bytes == n_pos * 2,
            "uids": len(dc["uids"]) == len(dc["offsets"]) == len(dc["lens"]),
            "vocab": int(dc["vocab"]) == VOCAB,
            "latent": int(dc["latent"]) == LATENT,
            "levels": int(dc["levels"]) == LEVELS,
            "pool": int(dc["pool"]) == args.pool,
            "dataset": dc.get("dataset_sha256") == args.fingerprint.get("dataset_sha256"),
            "fsq": dc.get("fsq_sha256") == args.fingerprint.get("fsq_sha256"),
            "offset_start": int(dc["offsets"][0]) == 0,
            "offset_end": int(dc["offsets"][-1] + dc["lens"][-1]) == n_pos,
            "contiguous": bool(torch.equal(
                dc["offsets"][1:], dc["offsets"][:-1] + dc["lens"][:-1])),
        }
        if not all(checks.values()):
            raise RuntimeError(f"码缓存校验失败：{checks}")
        args.fingerprint["codes_sha256"] = sha256_file(Path(args.codes))
        args.fingerprint["codes_index_sha256"] = sha256_file(Path(args.codes_idx))
        mmc = np.memmap(args.codes, dtype=np.uint16, mode="r",
                        shape=(n_pos,))
        CODES = torch.from_numpy(mmc).clone()
        if int(CODES.max()) >= VOCAB:
            raise RuntimeError(f"码缓存最大 ID {int(CODES.max())} >= VOCAB {VOCAB}")
        CODE_IDX = {
            uid: ("codes", int(dc["offsets"][i]), int(dc["lens"][i]))
            for i, uid in enumerate(dc["uids"])
        }
        if len(CODE_IDX) != len(dc["uids"]):
            raise RuntimeError("码缓存 uid 有重复，无法建立唯一索引")
        print(f"[codes] 2/3 memmap ok  {tuple(mmc.shape)}  vocab={dc['vocab']}", flush=True)
        print(f"[codes] 3/3 uid={len(CODE_IDX):,}  来自 {Path(args.codes).name}", flush=True)
        print("[codes] FSQ encoder 冻结；训练 LLM adapter + code embedding", flush=True)
    else:
        primary_hid = Path(args.hid) if args.hid else HID
        use_primary = primary_hid.exists() or not args.hid2
        if use_primary:
            d = torch.load(primary_hid, map_location="cpu", weights_only=False)
            hid, lens, offsets = d["x"], d["lens"], d["offsets"]
            HID_TENSOR = hid
            print(f"[hid] 主缓存 {hid.shape} 序列 {len(lens)}  来自 {primary_hid.name}")
        else:
            print(f"[hid] 跳过主缓存；只使用 --hid2 {args.hid2}")

    # 第二份缓存（rfam）：uid 是裸 md5，且是 np.memmap 不是 torch 张量
    if args.hid2 and not args.codes:
        import numpy as np
        print(f"[hid2] 1/4 torch.load 索引 {args.hid2_idx}", flush=True)
        d2 = torch.load(args.hid2_idx, map_location="cpu", weights_only=False)
        print(f"[hid2] 2/4 索引 ok  n_pos={int(d2['n_pos']):,} dim={int(d2['dim'])} "
              f"uids={len(d2['uids']):,}", flush=True)
        mm = np.memmap(args.hid2, dtype=np.float16, mode="r",
                       shape=(int(d2["n_pos"]), int(d2["dim"])))
        print(f"[hid2] 3/4 memmap ok  {tuple(mm.shape)}", flush=True)
        register_cache("aux", torch.from_numpy(mm), d2["lens"], d2["offsets"],
                       d2["uids"])
        print(f"[hid2] 4/4 register_cache ok  IDX={len(IDX):,}", flush=True)
        print(f"[hid] 副缓存 {tuple(mm.shape)} 序列 {len(d2['uids'])}  "
              f"来自 {Path(args.hid2).name}", flush=True)

    # 关键：按文件顺序建 uid -> 缓存下标/长度，与 48_ 完全一致
    if use_primary:
        allr = load_records(None, sort=False)
        assert len(allr) == len(lens), f"序列数不符 {len(allr)} vs {len(lens)}"
        off_of = {r["uid"]: int(offsets[i]) for i, r in enumerate(allr)}
        len_of = {r["uid"]: int(lens[i]) for i, r in enumerate(allr)}
        register_cache("main", hid, lens, offsets, [r["uid"] for r in allr])
    # 抽样再按 md5 排序，保证可复现且不偏向文件前部
    if args.data:
        # 新格式下 split 由数据文件的 split 字段决定，不再按 RNA 抽样
        picked = allr
    else:
        picked = sorted(allr, key=lambda r: hashlib.md5(r["uid"].encode()).hexdigest())
        picked = picked[:args.rna + args.eval_rna]
    train_recs, eval_recs = picked[:-args.eval_rna], picked[-args.eval_rna:]

    P = args.pool
    global POOL, TAU

    def _mk_new():
        """新格式：每行一条 QA，split 字段决定归属。
        隐藏态缓存按 uid 走多缓存索引（主 _fsq_full_pos.pt + 副 rfam memmap）。

        **val 和 test 必须分开**：val 用来早停（每 eval 点都要读），
        test 只用来出最终数字。混在一起的话早停就是在 test 上选的。
        """
        tr, va, te = [], [], []
        miss = 0
        buckets = {"train": tr, "val": va, "test": te}
        with open(args.data, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                q = json.loads(line)
                u = q.get("uid")
                source_idx = CODE_IDX if args.codes else IDX
                if u not in source_idx:
                    miss += 1
                    continue
                sp = q.get("split", "train")
                if sp not in buckets:
                    raise RuntimeError(f"未知 split={sp!r}（uid={u}）")
                clen = (source_idx[u][2] if args.codes
                        else source_idx[u][2] // P)
                item = ({"uid": u, "sequence": q.get("sequence", "")},
                        {"question": q["question"], "answer": q["answer"]},
                        clen, source_idx[u][1])
                buckets[sp].append(item)
        if miss:
            kind = "码缓存" if args.codes else "隐藏态缓存"
            print(f"[warn] {miss:,} 条 QA 的 uid 不在{kind}里")
        return tr, va, te

    POOL = args.pool
    # tau 必须和 latent 同尺度：q_ste 与 grid 都在 [-1,1]^latent，d2 上限约 4*latent。
    # tau=0.1 时 logits 跨度 320，softmax 退化成 one-hot，enc 梯度 1.8e-15（= 零）。
    TAU = args.tau if args.tau > 0 else LATENT / 2
    print(f"[tau] 软分配温度 {TAU}（latent {LATENT}，d2 上限约 {4 * LATENT}）")
    def mk(rs):
        # clen 是压缩后的 token 数（决定占位符个数）；切片时取 clen*P 个原始隐藏态
        return [(r, qa, len_of[r["uid"]] // P, off_of[r["uid"]]) for r in rs
                for qa in (r.get("qa") or []) if qa.get("question") and qa.get("answer")]
    if args.data:
        tr_ex, va_ex, te_ex = _mk_new()
        # 监视用 val；没有 val 就退回 test（并明确告警，避免在 test 上早停）
        ev_ex = va_ex if va_ex else te_ex
        if not va_ex:
            print("[data] ** 没有 val，早停会在 test 上选模型 **")
        print(f"[data] 新格式 {args.data}")
        print(f"[data] train {len(tr_ex):,}  val {len(va_ex):,}  test {len(te_ex):,}")
    else:
        tr_ex, ev_ex = mk(train_recs), mk(eval_recs)
        te_ex = []
    print(f"[data] 训练 {len(tr_ex):,} QA / 监视 {len(ev_ex):,} QA / "
          f"最终 {len(te_ex or ev_ex):,} QA")
    assert tr_ex and ev_ex

    from transformers import AutoTokenizer
    # **tokenizer 必须跟着底座走。**
    # 写死 QWEN 的话，跑 35B 时会拿 8B 的 tokenizer：
    #   im_start 151644 vs 248045、pad 151643 vs 248044、no_think 13 vs 14 token
    # 而 tk.base_vocab 又会被强制设成模型的 248320，占位符照样"能跑"——
    # 于是模型在一堆错的特殊 token 上训练，**不报任何错**。
    base_dir = QWEN36 if args.base == "qwen36-35b" else QWEN
    tk0 = AutoTokenizer.from_pretrained(str(base_dir))
    tk = Tok(tk0)
    print(f"[tok] {base_dir.name}  vocab {tk0.vocab_size}  "
          f"im_start {tk.im_start[-4:]}  pad {tk.pad_id}  "
          f"no_think {len(tk.no_think)} token")
    fast_path = {}
    for module in ("fla", "causal_conv1d"):
        try:
            mod = importlib.import_module(module)
            if module == "fla":
                required = ("chunk_gated_delta_rule", "recurrent_gated_delta_rule")
                missing = [x for x in required if not hasattr(mod, x)]
                if missing:
                    raise ImportError(f"fla missing {missing}")
            fast_path[module] = getattr(mod, "__version__", "installed")
        except Exception as exc:
            fast_path[module] = None
            print(f"[perf] {module} 未安装：{type(exc).__name__}")
    print(f"[perf] linear-attention kernels: {fast_path}")
    if not all(fast_path.values()):
        print("[perf] ** 正在走纯 PyTorch fallback；30 层 linear attention 会非常慢 **")
    print("\n=== 建模型 ===")
    model = E2EModel(args.lora_r, args.rna_emb, latent=LATENT, levels=LEVELS,
                     pool=args.pool,
                     base=args.base, base_mode=args.qwen36_mode,
                     lora_preset=args.lora_preset, base_bits=args.base_bits,
                     adapter=args.adapter, sos_k=args.sos_k,
                     sos_gamma_max=args.sos_gamma_max,
                     sos_dropout=args.sos_dropout)
    # Tok.base_vocab 来自 tokenizer.vocab_size，模型的来自 embedding 行数。二者若不相等，
    # build_batch 写进去的占位 id 会小于模型的 base_vocab -> is_rna 恒 False ->
    # RNA 段原样当成普通 token，四次消融结果完全相同。必须先对齐。
    print(f"[vocab] tokenizer {tk.base_vocab}   embedding {model.base_vocab}")
    if tk.base_vocab != model.base_vocab:
        print(f"[vocab] **不一致**，以 embedding 的 {model.base_vocab} 为准")
        tk.base_vocab = model.base_vocab

    if args.grad_ckpt:
        # 冻结底座 + LoRA/SOS 时，transformer 的 checkpoint 需要输入 requires_grad，
        # 否则重算阶段拿不到梯度，backward 会静默变成空操作。
        enabled = False
        for target in (model.llm, getattr(model.llm, "model", None)):
            if target is None:
                continue
            try:
                target.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
                enabled = True
                break
            except TypeError:
                try:
                    target.gradient_checkpointing_enable()
                    enabled = True
                    break
                except Exception as exc:                     # noqa: BLE001
                    print(f"[ckpt] {type(target).__name__}: {exc}")
            except Exception as exc:                         # noqa: BLE001
                print(f"[ckpt] {type(target).__name__}: {exc}")
        if enabled:
            try:
                model.llm.enable_input_require_grads()
            except Exception as exc:                         # noqa: BLE001
                print(f"[ckpt] enable_input_require_grads 失败: {exc}")
            print("[ckpt] gradient checkpointing 已打开（use_reentrant=False）")
        else:
            raise RuntimeError(
                "--grad-ckpt 1 但底座没有 gradient_checkpointing_enable()，"
                "拒绝在没开成的情况下继续（否则会白跑一轮 OOM）")
    # ---- 每步的真实步长 ----
    # 显存跟着 B*T 走（B=每步样例数，T=补齐后的序列长度），而 --token-budget 原先只
    # 累加压缩码长 clen，**完全没算问题和答案的文本 token**：seqtask 的答案是 6~8 个
    # 字符，码长和真实长度近似，所以 D 臂 7500 步一直没事；freetext 的答案是整段话
    # （中位数 278、最长 551 字符），一条码长记 4 的样例会照样塞满 8 条，把真实长度
    # 顶到四五百 —— B*T 比 D 臂最坏情况大好几倍。2026-10-16 连续四次 CUDA OOM 都是
    # 这么来的（报错时设备还"有" 30~43 GiB 空闲，因为这个核显真正的天花板是它能占
    # 多少系统内存，不是它自己报的那个数）。所以预算改成按真实步长算。
    _HEAD = (len(tk.sys_ids) + len(tk.user_ids) + len(tk.enc("Q: <RNA>"))
             + len(tk.assistant_ids) + len(tk.no_think))
    _qcache, _acache = {}, {}

    def _eff_len(item):
        _r, qa, clen, _off = item
        q = qa["question"]
        hq = _qcache.get(q)
        if hq is None:
            hq = len(tk.enc(f"</RNA>{q}"))
            _qcache[q] = hq
        a = qa["answer"]
        ha = _acache.get(a)
        if ha is None:
            ha = len(tk.enc(f" A: {a}{tk.eos}"))
            _acache[a] = ha
        return clen + _HEAD + hq + ha

    elen = [_eff_len(it) for it in tr_ex]
    _s = sorted(elen)
    _c = [x[2] for x in tr_ex]
    print(f"[len] 真实步长（码长+文本 token）p50 {_s[len(_s)//2]}  "
          f"p90 {_s[int(len(_s) * 0.9)]}  p99 {_s[int(len(_s) * 0.99)]}  "
          f"max {_s[-1]}   旧口径码长 {min(_c)}~{max(_c)}")
    if args.init_code_emb and model.rna_emb is not None:
        init_path = Path(args.init_code_emb)
        if not init_path.is_file():
            raise RuntimeError(f"--init-code-emb 不存在：{init_path}")
        ci = torch.load(init_path, map_location='cpu', weights_only=False)
        w = ci['code_emb'].to(device=model.rna_emb.weight.device,
                             dtype=model.rna_emb.weight.dtype)
        if w.shape != model.rna_emb.weight.shape:
            raise RuntimeError(
                f"--init-code-emb 形状 {tuple(w.shape)} != 模型码表 "
                f"{tuple(model.rna_emb.weight.shape)}。"
                f"说明 LATENT/LEVELS({LATENT}/{LEVELS}) 与那个文件不是同一版。"
                f"先用同一个 tokenizer 检查点重跑 51_。")
        model.rna_emb.weight.data.copy_(w)
        args.fingerprint["init_code_emb_sha256"] = sha256_file(init_path)
        print('[init] 码表按内容初始化（同 6-mer 余弦 +0.313 vs 随机 +0.017）')
    if FSQ_CKPT.is_file():
        ck = fsq_meta
        if ck is None:
            ck = torch.load(FSQ_CKPT, map_location="cpu", weights_only=False)
        # 只加载形状匹配的权重。码表尺寸变了（4096->256）时 enc 是 (8,768)、检查点是 (12,768)，
        # **strict=False 也挡不住 shape mismatch**，会直接 RuntimeError。
        cur = model.fsq.state_dict()
        sd = {k: v for k, v in ck["state"].items()
              if k in cur and tuple(v.shape) == tuple(cur[k].shape)}
        skipped = [k for k in ck["state"] if k.startswith("enc.") and k not in sd]
        if sd:
            model.fsq.load_state_dict(sd, strict=False)
        elif skipped:
            # 静默降级比报错危险：形状全不匹配时老代码只打「载入 0 个张量」，
            # 那行看起来完全正常，于是 FSQ.enc 实际是随机初始化却没被发现。
            raise RuntimeError(
                f"FSQ 检查点 {FSQ_CKPT.name} 与当前配置形状全不匹配（跳过 {skipped}）。"
                f"当前 LATENT={LATENT} LEVELS={LEVELS}，检查点 latent="
                f"{ck['state']['enc.weight'].shape[0]}。请用配套 FSQ 重新生成。")
        print(f"[init] FSQ.enc 载入 {len(sd)} 个张量"
              + (f"，跳过形状不符的 {skipped}" if skipped else ""))
    if args.codes:
        for p in model.fsq.parameters():
            p.requires_grad_(False)
    n_lora = sum(p.numel() for p in model.llm.parameters() if p.requires_grad)
    n_fsq = sum(p.numel() for p in model.fsq.parameters())
    n_tab = model.rna_emb.weight.numel() if model.rna_emb is not None else 0
    print(f"  LoRA {n_lora:,}  FSQ+head {n_fsq:,}  码表 {n_tab:,}")

    params = ([p for p in model.llm.parameters() if p.requires_grad]
              + ([] if args.codes else list(model.fsq.parameters()))
              + ([model.rna_emb.weight] if model.rna_emb is not None else []))
    opt = torch.optim.AdamW(params, lr=args.lr)
    w_init = (model.rna_emb.weight.detach().clone()
              if model.rna_emb is not None else None)

    # 长度分桶：同批内长度接近，padding 浪费大幅下降
    if args.bucket and args.bucket > 0:
        order = sorted(range(len(tr_ex)), key=lambda i: tr_ex[i][2])
        buckets = [order[i:i + args.bucket] for i in range(0, len(order), args.bucket)]
        print(f"[bucket] {len(buckets)} 个桶（桶宽 {args.bucket}），"
              f"长度 {tr_ex[order[0]][2]} ~ {tr_ex[order[-1]][2]}")
    else:
        buckets = [list(range(len(tr_ex)))]

    ckdir = outdir / "ckpt"
    ckdir.mkdir(parents=True, exist_ok=True)
    start_step, log, t_prev, opt_ok = 1, [], 0.0, True
    if args.resume:
        rp = Path(args.resume)
        if rp.is_dir():
            cands = sorted(rp.glob("step_*.pt"), key=lambda x: int(x.stem.split("_")[1]))
            if not cands:
                raise SystemExit(f"{rp} 下没有 step_*.pt")
            rp = cands[-1]
        start_step, log, t_prev, opt_ok = load_ckpt(
            rp, model, opt, args, allow_opt_reset=bool(args.allow_opt_reset))
        start_step += 1
        print(f"[resume] 从 {rp.name} 续训，step {start_step}，"
              f"优化器 {'已恢复' if opt_ok else '未恢复'}，历史 {len(log)} 条记录")

    if args.warm_lora:
        wp = Path(args.warm_lora)
        if wp.is_dir():
            cands = sorted(wp.glob("step_*.pt"), key=lambda x: int(x.stem.split("_")[1]))
            if not cands:
                raise SystemExit(f"{wp} 下没有 step_*.pt")
            wp = cands[-1]
        wk = torch.load(wp, map_location="cpu", weights_only=False)
        load_adapter_state(model, wk["lora"], f"warm {wp.name}")
        print(f"[warm] 只载 LoRA：{wp.name}  {len(wk['lora'])} 个张量，全部匹配")
        print(f"[warm] FSQ 与 rna_emb 保持前面载入的新码表，优化器从零开始")

    batch_stream = sample_batches(
        buckets, elen, args.batch, args.token_budget, args.seed)
    for _ in range(start_step - 1):
        next(batch_stream)
    print(f"[sample] 不放回 bucket sampler  seed={args.seed}  "
          f"batch<={args.batch}  token-budget={args.token_budget}")

    # ---- 早停状态 ----
    best_ce = float("inf")
    best_step = -1
    best_gap = float("nan")
    patience_left = args.patience
    stop_reason = ""
    stopped_early = False
    ce_hist: list[dict] = []
    ran_steps = start_step <= args.steps
    if args.early_stop and ran_steps:
        # 续训时从 log 里恢复最优值和耐心计数，否则 patience 会被重置
        for rec in log:
            e = rec.get("early")
            if e and e.get("metric") == EARLY_METRIC:
                if e.get("ce_a", float("inf")) < best_ce:
                    best_ce, best_step, best_gap = e["ce_a"], rec["step"], e["gap"]
                if "patience_left" in e:
                    patience_left = e["patience_left"]
        if best_step > 0 and not (ckdir / "best.pt").is_file():
            print("[早停] 历史记录有 best 但 best.pt 不存在；本轮重新建立 best")
            best_ce, best_step, best_gap = float("inf"), -1, float("nan")
            patience_left = args.patience
        if best_step > 0:
            print(f"[早停] 从历史恢复：最优 CE_A {best_ce:.4f} @ step {best_step}  "
                  f"耐心剩 {patience_left}")
        print(f"[早停] 开  监视 val CE(A 臂)  相对改善阈值 {args.min_delta:.3%}  "
              f"patience {args.patience}  最少 {args.min_steps} 步  "
              f"eval 间隔 {args.eval_every}")
        if args.eval_every > 1000:
            print(f"[早停] ** 警告：eval-every={args.eval_every} 太稀，"
                  f"patience {args.patience} 等于 {args.eval_every*args.patience:,} 步 **")

    print(f"\n=== 训练 {args.steps} 步（从 {start_step}）===")
    model.train()
    t_last_save = time.time()
    _ver = {"d": [], "jit": []}      # 手工 loss 与 HF 的逐点差，累积后按中位数判（见下）
    last_step = start_step - 1
    for step in range(start_step, args.steps + 1):
        profile = bool(args.profile_steps and step - start_step < args.profile_steps)
        if profile and torch.cuda.is_available():
            torch.cuda.synchronize()
        t_step = time.perf_counter()
        last_step = step
        ex = [tr_ex[i] for i in next(batch_stream)]
        ids, lab, am = build_batch(tk, ex)
        model.reset_usage()          # 不清空的话，fsq_usage 会跨步累积
        vecs, codes_all, t_all, q_all = [], [], [], []
        for r, qa, clen, off in ex:
            v, c, t, q = vectors_for_uid(model, r["uid"], tau=TAU)
            vecs.append(v); codes_all.append(c); t_all.append(t); q_all.append(q)
        v = torch.cat(vecs)
        if step == 1 or step % max(1, args.log_every) == 0:
            with torch.no_grad():
                nv = float(v.norm(dim=1).mean())
                nc = int(codes_all[0].unique().numel())
                print(f"  [mag] 码表项范数 {nv:.3f}   本样本用到码 {nc}/"
                      f"{LEVELS ** LATENT}", flush=True)
        if step == 1:
            n_rna = int((ids >= model.base_vocab).sum())
            assert n_rna == v.shape[0], f"RNA 占位 {n_rna} != 向量 {v.shape[0]}：替换没生效"
            print(f"  [check] RNA 位置 {n_rna} 个，向量 {tuple(v.shape)} —— 替换已生效",
                  flush=True)
        if profile and torch.cuda.is_available():
            torch.cuda.synchronize()
        t_forward = time.perf_counter()
        emb = model.embed(ids, v)
        if args.answer_only_loss:
            loss = answer_only_loss(model, emb, am, lab)
            # ⚠️ 必须用「本次运行的相对步数」，不能用 `step <= verify_loss`。
            #    续训时 start_step 是 389，那样条件永远不成立 —— 验证会静默地
            #    一次都不跑，而这正是它存在的意义（防错位）。踩过一次。
            if args.verify_loss and (step - start_step) < args.verify_loss:
                # **必须验。** 手工 loss 和 HF 的内置 loss 算的是同一个数，
                # 但错位（比如忘了 [:, :-1]）不会报错，只会训出一个 loss 曲线
                # 看起来完全正常、语义却错位的模型。宁可在这里炸。
                #
                # ⚠️ 判据是**累积 N 步后的中位数**，不是单步。
                #    2026-10-13 踩过：step 3569 上 |手工-HF| 冲到 1.15e-02，而
                #    HF 自己两次只差 8.83e-04，单步判据当场把训练掐了。
                #    但**前六步里五步都对得上** —— 而对齐错位是**确定性**的，
                #    会在每一步都偏，不可能六分之五都对。所以那是这条平台上前向
                #    的浮点路径差异（走 body() 和走 model.llm() 的注意力掩码构造
                #    方式和算子选择不同），不是接错了。
                #    取中位数就天然区分这两种：错位 -> 每一步都偏、中位数很大；
                #    算子抖动 -> 偶发离群、中位数很小。
                with torch.no_grad():
                    kw = dict(inputs_embeds=emb, attention_mask=am, use_cache=False)
                    ref = float(model.llm(labels=lab, **kw).loss)
                    ref2 = float(model.llm(labels=lab, **kw).loss)
                got = float(loss.detach())
                d = abs(got - ref)
                jit = abs(ref - ref2)
                _ver["d"].append(d)
                _ver["jit"].append(jit)
                print(f"  [verify-loss] step {step}  手工 {got:.6f}  HF {ref:.6f}  "
                      f"HF重复 {ref2:.6f}  |手工-HF| {d:.2e}  |HF重复差| {jit:.2e}",
                      flush=True)
                if len(_ver["d"]) >= args.verify_loss:
                    md = sorted(_ver["d"])[len(_ver["d"]) // 2]
                    mj = sorted(_ver["jit"])[len(_ver["jit"]) // 2]
                    need = max(6 * mj, 2e-3)
                    tag = "等价 ✅" if md <= need else "不一致 ❌"
                    print(f"  [verify-loss] {len(_ver['d'])} 步汇总  "
                          f"|手工-HF| 中位数 {md:.2e}   |HF重复差| 中位数 {mj:.2e}   "
                          f"阈值 {need:.2e}  -> {tag}", flush=True)
                    if md > need:
                        raise RuntimeError(
                            f"answer_only_loss 与 out.loss 系统性不符：{len(_ver['d'])} 步的"
                            f"|手工-HF| 中位数 {md:.2e}，而 HF 自身重复差中位数只有 {mj:.2e}。"
                            f"对齐方式错了，不许继续训。")
                    _ver["done"] = True
        else:
            # 对照路径也要 use_cache=False：35B 有 30 层线性注意力，
            # use_cache 一变就走分块/递推两条不同数值路径，两条 loss 实现必须同参才有可比性。
            out = model.llm(inputs_embeds=emb, attention_mask=am, labels=lab, use_cache=False)
            loss = out.loss

        if profile and torch.cuda.is_available():
            torch.cuda.synchronize()
        t_lm_loss = time.perf_counter()
        codes = torch.cat(codes_all)
        hist = torch.bincount(codes, minlength=VOCAB).float()
        p = hist / hist.sum().clamp(min=1)
        ent = float(-(p * (p + 1e-9).log()).sum())     # 仅用于打印（整数统计，无梯度）
        # 熵正则必须走**可微**的使用率。老版本用 bincount(codes) 算，那是个常数，
        # 乘任何 ent_coef 都不产生梯度 —— 熵正则从来没生效过（这才是码一直塌的原因）。
        u = getattr(model, "fsq_usage", None)
        if u is not None and args.ent_coef:
            u = u / u.sum().clamp(min=1e-9)
            l_ent = (u * (u + 1e-9).log()).sum() * args.ent_coef    # 最小化 -> 最大化熵
        else:
            l_ent = torch.zeros((), device=DEV)
        if not args.codes:
            l_com = args.commit_coef * F.mse_loss(torch.cat(t_all), torch.cat(q_all))
        else:
            l_com = torch.zeros((), device=DEV)
        l_orth = torch.zeros((), device=DEV)
        if args.ortho_coef and model.rna_emb is not None:
            u = torch.unique(codes)
            if u.numel() > 1:
                w = F.normalize(model.rna_emb.weight[u].float(), dim=1)
                g = w @ w.t()
                l_orth = args.ortho_coef * (g - torch.eye(len(u), device=DEV)).pow(2).mean()
        # SOS 的跨专家正交正则：罚 A 的**跨专家**相关（同专家内部不罚），
        # 防止 K 个子空间塌到同一个方向上 —— 那等于退化成 K=1 的普通 LoRA。
        l_sos = torch.zeros((), device=DEV)
        if args.adapter == "sos" and args.sos_ortho_coef:
            from rnaqwen.models.sos_lora import sos_lora_penalty
            l_sos = args.sos_ortho_coef * sos_lora_penalty(model.llm)
        total = loss + l_ent + l_com + l_orth + l_sos
        if profile and torch.cuda.is_available():
            torch.cuda.synchronize()
        t_aux_done = time.perf_counter()
        t_backward = time.perf_counter()
        total.backward()
        if profile and torch.cuda.is_available():
            torch.cuda.synchronize()
        t_after_backward = time.perf_counter()
        gn = (float(model.fsq.enc.weight.grad.norm())
              if model.fsq.enc.weight.grad is not None else -1)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad()
        if profile and torch.cuda.is_available():
            torch.cuda.synchronize()
        t_after_opt = time.perf_counter()
        if profile:
            print(
                f"  [profile] step {step}  batch+data {t_forward-t_step:.2f}s  "
                f"model+lm_loss {t_lm_loss-t_forward:.2f}s  "
                f"aux_loss {t_aux_done-t_lm_loss:.2f}s  "
                f"backward {t_after_backward-t_backward:.2f}s  "
                f"opt {t_after_opt-t_after_backward:.2f}s  "
                f"total {t_after_opt-t_step:.2f}s  "
                f"samples {len(ex)}  eff_tokens {sum(_eff_len(x) for x in ex)}  "
                f"verify={bool(args.verify_loss and step-start_step < args.verify_loss)}"
                + (f"  vram {torch.cuda.memory_allocated()/2**30:.2f}/"
                   f"{torch.cuda.memory_reserved()/2**30:.2f} GiB"
                   if torch.cuda.is_available() else ""),
                flush=True,
            )

        # 修剪工作集：slice_hid 读 mmap 会把页 fault 进 WorkingSet，
        # 而 WorkingSet 计入常驻的文件映射页。不修剪的话空闲内存会一路下滑，
        # 到零时**硬崩、没有异常栈**（84_ 装载时踩过同一个坑）。
        # 放在 log 块**外面**：trim_every 必须是独立的周期，
        # 放进 log 块的话要求它整除 log_every，配错了就静默不修剪。
        if args.trim_every and step % args.trim_every == 0:
            try:
                from rnaqwen.qwen36 import trim_working_set
                trim_working_set()
            except Exception:
                pass

        if step % max(1, args.log_every) == 0 or step == 1:
            drift = (float((model.rna_emb.weight.detach() - w_init).norm() / w_init.norm())
                     if w_init is not None else 0.0)
            print(f"  step {step:>5}  ce {float(loss):.4f}  ent {ent:.3f}  "
                  f"|grad_fsq {gn:.3e}|  码表漂移 {100*drift:.1f}%  {time.time()-t0:.0f}s",
                  flush=True)
            if step % max(1, args.log_every * 5) == 0:
                try:
                    from rnaqwen.qwen36 import free_gb, rss_gb
                    print(f"           [mem] RSS {rss_gb():.2f} GiB  "
                          f"系统空闲 {free_gb():.1f} GiB", flush=True)
                except Exception:
                    pass
            log.append(dict(step=step, ce=float(loss), ent=ent, grad_fsq=gn, drift=drift))

        # 存盘时机：步数到点，**或者**距上次存盘超过 --save-minutes 分钟。
        # 后者是「动态」的那一半 —— 纯按步数时，步速多慢都不会提前存：
        # 实测 18.5 s/步 × --save-every 300 = 92 分钟一个档，崩一次最多丢 1.5 小时。
        # 按墙上时钟兜底，丢失窗口就与步速无关，有硬上限。
        due_step = bool(args.save_every) and step % args.save_every == 0
        due_time = (args.save_minutes > 0
                    and (time.time() - t_last_save) >= args.save_minutes * 60.0)
        if due_step or due_time:
            cp = ckdir / f"step_{step:06d}.pt"
            save_ckpt(cp, step, model, opt, log, args, time.time() - t0)
            t_last_save = time.time()
            gone = prune_ckpts(ckdir, args.keep_ckpt)
            why = []
            if due_step:
                why.append("步数到点")
            if due_time and not due_step:
                why.append("时间兜底")
            print(f"  [ckpt] {cp.name}  ({cp.stat().st_size/1024/1024:.1f} MB, "
                  f"保留最近 {args.keep_ckpt} 个"
                  + (f", 清掉 {gone} 个旧档" if gone else "")
                  + f", {why[0]})", flush=True)

        if args.eval_every and step % args.eval_every == 0:
            model.eval()
            # 5 臂消融在 35B 上一次 = 5 x eval-n 次前向，很贵；用 --ablation-every 0 关掉它，
            # 训练中只用便宜的 A/C 两臂 CE 监视（见下面的早停块）。
            if args.ablation_every < 0:
                do_ab = True                       # -1：跟 eval-every 同步（旧行为）
            elif args.ablation_every == 0:
                do_ab = False                      # 0：训练中不跑
            else:
                do_ab = (step % args.ablation_every == 0)
            ab = (run_ablation(model, tk, ev_ex, hid, max_n=args.eval_n)
                  if do_ab else None)
            if ab is not None:
                print(f"  [eval@{step}] 全部标签 n={ab['zero']['n']}  " + "  ".join(
                    f"vs{k} {100*v['frac_neg']:.1f}%(m{v['mean']:+.3f})"
                    for k, v in ab.items()), flush=True)
                # **只看答案 token 的那一套才是判据。** 全部标签那一套被样板稀释
                # （答案只占 28.4%），frac_neg 天然被拖到 ~44% 以上，45% 那条线永远通不过。
                print(f"  [eval@{step}] 只看答案 n={ab['zero']['n_ans']}  " + "  ".join(
                    f"vs{k} {100*v['frac_neg_ans']:.1f}%(m{v['mean_ans']:+.3f})"
                    for k, v in ab.items()), flush=True)
                # 仪器灵敏度：若 |Δ| 中位数 ~1e-3，说明消融几乎没改变任何东西，
                # 这时 frac_neg 就是噪声，不能拿它下任何结论。
                mags = [abs(v["mean_ans"]) for v in ab.values()]
                print(f"           |Δ| 均值（答案）{max(mags):.4f}  "
                      f"{'<- 太小，frac_neg 无意义' if max(mags) < 0.02 else ''}",
                      flush=True)
                log.append(dict(step=step, ablation=ab))
            if args.diag:
                dg = diagnose(model, tk, ev_ex, max_n=args.diag_n)
                if dg:
                    print_diag(step, dg)
                    log.append(dict(step=step, diag=dg))

            # ---- 早停：只算 CE，不生成 ----
            if args.early_stop:
                ce_a, se_a = val_ce(model, tk, ev_ex, arm="A", max_n=args.eval_n)
                ce_c, _ = val_ce(model, tk, ev_ex, arm="C", max_n=args.eval_n)
                gap = ce_c - ce_a
                # 判据的**有效阈值**：既要超过相对改善 min_delta，
                # 也要超过一个标准误。少了 se 这一项，patience 会被噪声吃光。
                if best_ce == float("inf"):
                    need = 0.0
                else:
                    need = max(args.min_delta * abs(best_ce), se_a)
                improved = best_ce == float("inf") or ce_a < best_ce - need
                if improved:
                    best_ce, best_step, best_gap = ce_a, step, gap
                    patience_left = args.patience
                elif step >= args.min_steps:
                    patience_left -= 1

                ce_hist.append(dict(step=step, ce_a=ce_a, se_a=se_a, ce_c=ce_c,
                                    gap=gap, best=best_ce))
                # 先更新 state 再落 log；best.pt 里的 log 才会恢复正确 patience。
                log.append(dict(step=step, early=dict(
                    metric=EARLY_METRIC,
                    ce_a=ce_a, se_a=se_a, ce_c=ce_c, gap=gap, best=best_ce,
                    patience_left=patience_left)))
                if improved:
                    bp = ckdir / "best.pt"
                    save_ckpt(bp, step, model, opt, log, args, time.time() - t0)
                print(f"  [早停@{step}] CE_A {ce_a:.4f}±{se_a:.4f}  CE_C {ce_c:.4f}  "
                      f"gap {gap:+.4f}   最优 {best_ce:.4f}@{best_step}  "
                      f"需改善 {need:.4f}  耐心 {patience_left}/{args.patience}",
                      flush=True)
                if improved:
                    print(f"           ↑ 新的最优，已存 {(ckdir / 'best.pt').name}",
                          flush=True)
                if step >= args.min_steps and patience_left <= 0:
                    stopped_early = True
                    stop_reason = (f"CE_A 连续 {args.patience} 个 eval 点"
                                   f"（{args.patience*args.eval_every:,} 步）"
                                   f"改善 < max({args.min_delta:.1%}×CE, 1 se)"
                                   f"={need:.4f}")
                    print(f"  ** 早停触发：{stop_reason}  "
                          f"最优 {best_ce:.4f}@{best_step}（gap {best_gap:+.4f}）**",
                          flush=True)
                    break
            model.train()

    if args.early_stop:
        bp = ckdir / "best.pt"
        if best_step <= 0 or not bp.is_file():
            raise RuntimeError(
                "早停已开启，但 best.pt/最优步不存在；拒绝把最后一 step 冒充 best")
        if best_step < last_step:
            print(f"\n[早停] 载回 best.pt（step {best_step}，CE_A {best_ce:.4f}）")
            loaded_step, best_log, _, _ = load_ckpt(
                bp, model, opt, args, allow_opt_reset=False)
            if loaded_step != best_step:
                raise RuntimeError(f"best.pt step={loaded_step} 与记录 best_step={best_step} 不一致")
            log = best_log
            ce_hist = [r["early"] | {"step": r["step"]}
                       for r in log if r.get("early")]

    # final.pt 的 step 必须与内存中的权重一致。
    end_step = best_step if (args.early_stop and ran_steps) else last_step
    fp = ckdir / "final.pt"
    save_ckpt(fp, end_step, model, opt, log, args, time.time() - t0)
    print(f"[ckpt] final.pt  (step {end_step}, "
          f"{fp.stat().st_size/1024/1024:.1f} MB)", flush=True)

    if args.early_stop:
        print(f"\n[早停汇总] 最优 CE_A {best_ce:.4f} @ step {best_step}"
              f"（gap {best_gap:+.4f}）")
        print(f"           停止原因：{stop_reason or '未触发（跑满步数）'}")
        print(f"           {'有' if (ckdir/'best.pt').exists() else '没有'} best.pt")
        for h in ce_hist:
            print(f"   step {h['step']:>6}  CE_A {h['ce_a']:.4f}  "
                  f"CE_C {h['ce_c']:.4f}  gap {h['gap']:+.4f}")

    print("\n=== 最终消融 ===")
    model.eval()
    ab = run_ablation(model, tk, te_ex or ev_ex, hid, max_n=40)
    for k, v in ab.items():
        print(f"  vs {k:7}  全部标签 n {v['n']:>5}  Δlogp 均值 {v['mean']:+.4f}  "
              f"负值 {100*v['frac_neg']:.1f}%")
        print(f"             **只看答案** n {v['n_ans']:>5}  Δlogp 均值 {v['mean_ans']:+.4f}  "
              f"负值 {100*v['frac_neg_ans']:.1f}%")
    print()
    print("  怎么读（关键在对照，不在第一个数）:")
    print("    vs 零表  低 => RNA 位置上「有东西」在起作用")
    print("    vs 随机表 低 => 换掉整张表会变差")
    print("    vs 常数码 低 => 码的区分度在起作用")
    print("    ** vs 置换 低 => 码与内容的对应在起作用（最干净的那个）**")
    print()
    print("  ⚠️ 判据用「只看答案」那一套。全部标签那一套被样板稀释：")
    print("     标签 span 平均 9.89 个 token，其中答案只占 2.81（28.4%），")
    print("     其余是 no_think 尾巴 + ' A: ' + eos，与码无关、天然 50/50。")
    print("     稀释下限就有 ~36%，所以「全部标签」永远通不过 45% 的线。")
    print()
    pk = ab["perm"]["frac_neg_ans"]
    zk = ab["zero"]["frac_neg_ans"]
    if zk < 0.45 and pk < 0.45:
        print(f"  ✅ **码真的被读了。** 答案 token 上：零表 {100*zk:.1f}%、"
              f"置换 {100*pk:.1f}% 都低于 45%。可上全量。")
    elif zk < 0.45:
        print(f"  ❌ **模型只用了「RNA 位置有个向量」，没用码的身份。**")
        print(f"     答案 token 上零表 {100*zk:.1f}% 说明有东西在用；"
              f"但置换 {100*pk:.1f}% 说明把码打乱不影响结果。")
        print("     => 码的切分方式对 LLM 无意义，要改的是 tokenizer 的目标，不是加步数。")
    else:
        hint = "--codes 路径看码表漂移/梯度" if args.codes else "看 |grad_fsq|"
        print(f"  ❌ 连「有东西」都不起作用（答案 token 上零表 {100*zk:.1f}%）。"
              f"先查梯度是否回传（{hint}）。")
    print()
    print("  注意：Δlogp 的**均值**是另一个独立证据，它不被稀释。若均值上")
    print("       零表 < 置换 ≈ 常数码，说明「错误但看似合理的码」比「没有码」更有害")
    print("       —— 那是码身份在起作用的强证据（存在感假设会给出相反的排序）。")

    (outdir / "summary.json").write_text(json.dumps(
        dict(args=vars(args), end_step=end_step, log=log, final=ab),
        ensure_ascii=False, indent=2), encoding="utf-8")

    # 交付产物：adapter / FSQ / 码表 分开存，推理时都要用
    sd = Path(args.save_dir) if args.save_dir else (outdir / "artifacts")
    sd.mkdir(parents=True, exist_ok=True)
    if args.adapter == "sos":
        # SOS 没有 peft 的 save_pretrained。存 A/B 的 state_dict，
        # 并给出**合并后的权重**选项（论文的核心卖点：合并后推理端不需要认识 SOS，
        # ΔW = Σ_k s_k A_k B_k 并进 base.weight，零额外延迟）。
        ad = adapter_state_dict(model)
        torch.save(ad, sd / "sos_adapter.pt")
        print(f"[ship] SOS 适配器 {len(ad)} 个张量 -> sos_adapter.pt")
        print(f"       合并用 rnaqwen.models.sos_lora.merge_sos_layers(model)，"
              f"合并后结构就是普通 nn.Linear")
    else:
        model.llm.save_pretrained(str(sd / "lora_adapter"))
    torch.save(dict(state=model.fsq.state_dict(), latent=LATENT, levels=LEVELS,
                    d_model=model.base_emb.weight.shape[1]),
               sd / "fsq.pt")
    if model.rna_emb is not None:
        torch.save(model.rna_emb.weight.detach().cpu(), sd / "code_emb.pt")
    meta = dict(
        rna_emb_mode=args.rna_emb,
        base_vocab=model.base_vocab,
        vocab=VOCAB,
        latent=LATENT,
        levels=LEVELS,
        base=args.base,
        qwen36_mode=args.qwen36_mode if args.base == "qwen36-35b" else None,
        base_bits=args.base_bits if args.base == "qwen3-8b" else None,
        adapter=args.adapter,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_preset=args.lora_preset if args.base == "qwen36-35b" else None,
        sos_k=args.sos_k if args.adapter == "sos" else None,
        sos_gamma_max=args.sos_gamma_max if args.adapter == "sos" else None,
        sos_dropout=args.sos_dropout if args.adapter == "sos" else None,
        pool=args.pool,
        seed=args.seed,
        step=end_step,
        fingerprint=getattr(args, "fingerprint", {}),
        note=("SOS merge scales need lora_r, sos_k, sos_gamma_max; "
              "the base model is not embedded in this artifact directory."),
    )
    (sd / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ship] {sd}  adapter({'sos' if args.adapter=='sos' else 'lora'}) "
          f"+ fsq.pt + code_emb.pt + meta.json")
    print(f"\n[save] {outdir/'summary.json'}   用时 {(time.time()-t0)/60:.1f} min")
    return 0


def _hard_exit(code: int) -> None:
    """绕过解释器收尾硬退。

    实测（2026-10-12，35B/SOS）：python 崩在第 1 步之后**不会退出** —— 它挂在 ROCm
    的上下文销毁上，进程一直留着，继续占着 18 GiB 显存和日志文件句柄。后果有两层：
      1. 下一轮启动的 Tee-Object 直接打不开日志（"being used by another process"），
         报错却出现在"装载前一刻"，看着像训练崩了；
      2. "空闲显存"凭空少 18 GiB —— 看着像 OOM，其实是上一个进程没死。
    而且 Stop-Process -Force 都杀不掉（内核态线程被驱动攥着），只能 taskkill /F /T。
    连踩两次之后确认：光靠启动前扫描是治标，得从这里治本。

    os._exit 跳过解释器收尾，也就跳过了那段会挂的析构。句柄和显存由操作系统回收 ——
    这正是我们要的。代价是 atexit 钩子不跑、python 层缓冲不自动刷，所以先 flush。
    （torch.save 是同步写完的，不受影响；checkpoint 全是周期落盘，
      没有"退出时再存一次"的逻辑，所以跳过 atexit 不丢东西。）
    """
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(code)


if __name__ == "__main__":
    try:
        _code = main()
    except SystemExit as e:                     # main 内部可能有 sys.exit
        _code = e.code if isinstance(e.code, int) else 1
    except BaseException:
        import traceback
        traceback.print_exc()
        _code = 1
    _hard_exit(_code)
