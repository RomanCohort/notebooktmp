"""把 notebook 从「候选式单序列任务」切到「RNA-RNA 关系判别」。

改动：
  cell 1   markdown 说明
  cell 2   CHECKPOINT_RELEASE_TAG -> train-v3-rel，追加 MiRBench 配置
  cell 6   不再下载 663 MB 的旧数据集，只建 DATA_DIR
  cell 8   Zenodo 下载 MiRBench 原始数据 + build_mirbench.py 转 jsonl
  cell 10  markdown 显存/参数说明
  cell 11 common 改成调 52_routeR_relation.py
  cell 12/13 运行目录 e2e_v3_cand -> e2e_rel
  cell 13 full 参数改成关系判别的
  cell 14 从训练日志汇总 AUROC + 扰动检验结果

先备份：cp RNAQwen_v3_Full.ipynb RNAQwen_v3_Full.candidate.ipynb
"""
import json
import re
from pathlib import Path

NB = Path("RNAQwen_v3_Full.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))


def set_src(idx: int, text: str) -> None:
    lines = text.strip("\n").split("\n")
    nb["cells"][idx]["source"] = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
    nb["cells"][idx]["outputs"] = []
    nb["cells"][idx]["execution_count"] = None


def src(idx: int) -> str:
    return "".join(nb["cells"][idx]["source"])


# ---- cell 1: markdown 说明 ----
set_src(1, """
# RNAQwen v3 · RNA-RNA 关系判别

No Google Drive required. The notebook downloads:

- code and the two FSQ checkpoints from this GitHub repository;
- **MiRBench** (Zenodo 20540907) raw `tsv.gz` files, converted to
  `mirbench.jsonl` by `scripts/build_mirbench.py`;
- ERNIE-RNA and Qwen3.6-35B-A3B from their public model repositories.

Training is `scripts/52_routeR_relation.py`: two sequences (miRNA, mRNA window)
are encoded by a frozen ERNIE-RNA, projected through question-aware Perceiver
resamplers into soft prompts, and the frozen 35B reads them with SOS-LoRA
adapters. The task is binary: is there an experimentally validated targeting
relationship? Answer 是 / 否.

Checkpoints sync to the GitHub Release tagged in the next cell when a Colab
secret named `GITHUB_TOKEN` is present.
""")

# ---- cell 2: 常量 ----
c2 = src(2).replace(
    "CHECKPOINT_RELEASE_TAG = 'train-v3-checkpoints'",
    "CHECKPOINT_RELEASE_TAG = 'train-v3-rel'",
)
if "train-v3-rel" not in c2:
    raise SystemExit("cell2 没找到 CHECKPOINT_RELEASE_TAG")
c2 = c2.rstrip("\n") + """

# ---- MiRBench：RNA-RNA 关系判别的主数据源 ----
# Zenodo 20540907，CC-BY-4.0。原始是 tsv.gz，由 scripts/build_mirbench.py 转 jsonl。
ZENODO_RECORD_ID = '20540907'
MIRBENCH_URL = f'https://zenodo.org/records/{ZENODO_RECORD_ID}/files'
MIRBENCH_FILES = [
    'AGO2_CLASH_Hejret2023_train.tsv.gz',
    'AGO2_CLASH_Hejret2023_test.tsv.gz',
    'AGO2_eCLIP_Manakov2022_train.tsv.gz',
    'AGO2_eCLIP_Manakov2022_test.tsv.gz',
    'AGO2_eCLIP_Manakov2022_leftout.tsv.gz',
    'AGO2_eCLIP_Klimentova2022_test.tsv.gz',
]
print('MiRBench source:', MIRBENCH_URL)"""
set_src(2, c2)

# ---- cell 6: 不再下载旧数据集 ----
set_src(6, """
DATA_DIR = ASSET_DIR / 'data' / 'processed'
DATA_DIR.mkdir(parents=True, exist_ok=True)
# 关系判别不用旧的 rnaqa_v3_cand.jsonl：数据来自 MiRBench（下一格下载并转换）。
print('Data dir:', DATA_DIR)
""")

# ---- cell 8: MiRBench 下载 + 转换 ----
set_src(8, '''
# ---- MiRBench -> mirbench.jsonl ----
# 原始 6 个 tsv.gz，转换后约 4.1M 条、1.9 GB（含 target 与 region 两种任务）。
RAW_DIR = DATA_DIR / 'mirbench_raw'
RAW_DIR.mkdir(parents=True, exist_ok=True)
MIRBENCH_JSONL = DATA_DIR / 'mirbench.jsonl'


def fetch_zenodo(name):
    dst = RAW_DIR / name
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    part = RAW_DIR / (name + '.part')
    url = f'{MIRBENCH_URL}/{name}?download=1'
    print('  fetch', name, flush=True)
    with requests.get(url, stream=True, timeout=(30, 300)) as r:
        r.raise_for_status()
        with part.open('wb') as fh:
            for ch in r.iter_content(8 * 1024 * 1024):
                if ch:
                    fh.write(ch)
    os.replace(part, dst)
    return dst


if not MIRBENCH_JSONL.exists():
    free_gb = shutil.disk_usage('/content').free / 2 ** 30
    if free_gb < 90.0:
        raise RuntimeError(f'需要至少 90 GiB 空闲（模型 67 + 数据 2），当前 {free_gb:.1f}')
    for name in MIRBENCH_FILES:
        fetch_zenodo(name)
    subprocess.run([
        sys.executable, '-X', 'utf8', '-u',
        str(ROOT / 'scripts/build_mirbench.py'),
        '--out', str(MIRBENCH_JSONL),
        '--cache', str(RAW_DIR),
        '--lang', 'zh',
    ], cwd=ROOT, check=True)
else:
    print('MiRBench jsonl already present:', MIRBENCH_JSONL)

size_gb = MIRBENCH_JSONL.stat().st_size / 2 ** 30
print(f'MiRBench ready: {MIRBENCH_JSONL}  {size_gb:.2f} GiB')
''')

# ---- cell 10: markdown 说明 ----
set_src(10, """
## A100 40GB memory budget

| item | GiB |
|---|---|
| Qwen3.6-35B-A3B int4 weights | 18.2 |
| ERNIE-RNA encoder (frozen) | ~0.6 |
| SOS-LoRA + 2 resamplers (trainable) | 0.04 |
| activations, batch 64, ~80 tokens/row | small |

Relation rows are short: miRNA ≤ 23 nt, mRNA window 50 nt. The LLM sees
`prefix (12) + 8 soft prompt tokens + question (~60) + " A: "`, so a batch of 64
is only ~5k tokens — well inside 40 GB with `--grad-ckpt 1`.

Qwen3.6 has 30 linear-attention layers. The setup cell installs
`flash-linear-attention[cuda]` and `causal-conv1d`; without them Transformers
falls back to the slow pure-PyTorch chunk/recurrent implementation. The exact
Python 3.13 / torch 2.11 / CUDA 13 wheel is published at `kernel-wheels-v1` and
checked by SHA256 before installation.

200,000 train rows are a uniform reservoir sample of the 2.5M MiRBench train
rows (CLASH + eCLIP), so every source dataset is represented. 8,000 steps at
batch 64 is ~2.6 epochs. `--max-train` is a memory guard, not a data limit:
raise it to 500k if the run finishes early.

This notebook starts real training directly. The first step includes cold-start
kernel initialization; compare the profile from steps 2-3. GPU utilization is
sampled every 0.75 seconds and reported as a three-second window average and
peak, with a min/median/mean/max summary at segment exit.
""")

# ---- cell 11: 训练 common ----
set_src(11, '''
common = [
    sys.executable, '-X', 'utf8', '-u',
    str(ROOT / 'scripts/52_routeR_relation.py'),
    '--data', str(MIRBENCH_JSONL),
    '--encoder', str(ERNIE_DIR),
    # 4.1M 条不能全读进内存（和 35B 装载峰值叠加会硬崩）。
    # 20 万条是跨 CLASH/eCLIP/leftout 的均匀蓄水池样本。
    '--max-train', '200000',
    '--max-test', '20000',
    '--eval-every', '500',
    '--eval-n', '2000',
    '--log-every', '50',
    '--grad-ckpt', '1',
]
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
print('Relation-discrimination training configured.')
''')

# ---- cell 12: 运行目录 ----
c12 = src(12).replace('e2e_v3_cand', 'e2e_rel')
if 'e2e_rel' not in c12:
    raise SystemExit('cell12 没找到 e2e_v3_cand')
set_src(12, c12)

# ---- cell 13: full 参数 ----
c13 = src(13).replace('e2e_v3_cand', 'e2e_rel')
new_full = """full = common + [
    # 关系判别：20 万 train，batch 64 -> 1 epoch 约 3,125 步，8,000 步约 2.6 epoch。
    # 每行只有 ~80 token（miRNA 22 + mRNA 50 + 问题），比候选式短得多。
    '--steps', '8000',
    '--batch', '64',
    '--lr', '1e-3',
    # 每 200 步或 10 分钟存一次（每步 ~1s，10 分钟兜底掉线），留 3 份。
    '--save-every', '200', '--save-minutes', '10', '--keep-ckpt', '3',
    '--perturb', '1',
    '--tag', 'rel'
]"""
c13, n_sub = re.subn(r"full = common \+ \[.*?\n\]", new_full, c13,
                     count=1, flags=re.DOTALL)
if n_sub != 1:
    raise SystemExit('cell13 full 块没找到，notebook 结构变了')
set_src(13, c13)

# ---- cell 14: 从日志汇总 ----
set_src(14, '''
# ---- 关系判别评测 ----
# 训练脚本每 --eval-every 步打印 [eval@N] AUROC ...；
# 训练结束时 --perturb 1 会打印输入扰动检验。
import re

log = RUNS_DIR / 'e2e_rel' / 'train.log'
print('log:', log, 'exists:', log.exists())

if log.exists():
    txt = log.read_text(encoding='utf-8', errors='replace')
    evals = [(int(m.group(1)), float(m.group(2)))
             for m in re.finditer(r'\\[eval@(\\d+)\\] AUROC ([0-9.]+)', txt)]
    if evals:
        print(f'{"step":>6}  AUROC')
        for s, a in evals:
            print(f'{s:>6}  {a:.4f}')
        bs, ba = max(evals, key=lambda x: x[1])
        print(f'\\nbest: step {bs}  AUROC {ba:.4f}')
    i = txt.find('[perturb]')
    if i >= 0:
        print('\\n' + txt[i:].strip())

ck = RUNS_DIR / 'e2e_rel' / 'ckpt'
print('\\ncheckpoints:', sorted(p.name for p in ck.glob('*.pt')) if ck.exists() else [])

print('\\nAUROC：正样本(是)分数 > 负样本(否)分数 的概率，随机 = 0.5。')
print('扰动检验：打乱 miRNA / mRNA 后 AUROC 应回到 ~0.5。')
print('若打乱后仍然很高，说明模型没在读序列（见 PLAN 31.7）。')
''')

for c in nb["cells"]:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None

NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
t = NB.read_text(encoding="utf-8")
print("cells:", len(nb["cells"]))
print("smoke:", t.lower().count("smoke"), "| ablation:", t.lower().count("ablation"))
print("has 52_routeR:", "52_routeR_relation" in t)
print("has mirbench:", "mirbench" in t.lower())
print("has e2e_rel:", t.count("e2e_rel"))
print("has old cand run dir:", "e2e_v3_cand" in t)
print("has old dataset url:", "dataset-v3-cand" in t)
