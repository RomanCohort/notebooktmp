# RNAQwen v3 Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RomanCohort/notebooktmp/blob/main/RNAQwen_v3_Full.ipynb)

Open `RNAQwen_v3_Full.ipynb` in Google Colab and select an A100 40GB runtime.
No Google Drive is required.

## Scope

RNA literature databases are growing faster than curators can maintain them.
General-purpose LLMs have no native representation for the RNA sequence modality,
so asking them to reason over sequences directly produces hallucinations — and in
literature-database maintenance (deciding whether two RNAs have a known functional
or regulatory relationship in the literature) a hallucinated answer is actively
harmful to the research it is supposed to support.

Following RNA-GPT (arXiv:2411.08900 [q-bio.GN]), this project trains an adapter
that gives Qwen3.6-35B-A3B a native representation of RNA sequences, using
SOS-LoRA (10.18653/v1/2026.acl-long.184) rather than a plain LoRA. A retrieval
tool over the literature database (OpenHarness + RAG) is built on top of it.

The work is split by whether a fact can be computed, retrieved, or neither:

- **Computable** → a tool, not the model. Structure is ViennaRNA (MFE, pairing
  fraction, stem lengths); family lookup is Rfam/HMMER; sequence detail is string
  arithmetic. Making the model learn a property an O(1) algorithm computes exactly
  is both unreliable and unnecessary.
- **Retrievable** → RAG. Literature facts, functional descriptions, known
  interactions.
- **Neither** → the model. Reading a raw sequence and judging whether two
  sequences are related, when no tool answers it directly.

That last part is what this repository trains. The first two are harness and
adapter layers around it.

### Target task: RNA-RNA relationship discrimination

The end goal is literature-database maintenance: given two RNAs, decide whether
they have a documented functional or regulatory relationship, with a literature
snippet as evidence. The task ladder is ordered by how trustworthy the labels are:

1. **miRNA-mRNA targeting** — MiRBench (CLASH / chimeric eCLIP, bias-corrected)
   and TargetScan (~560k human/mouse pairs, 280k positive / 280k negative).
   Cleanest labels, largest data, and both sequences fit the encoder window.
2. **lncRNA-miRNA sponging** — starBase, DIANA-LncBase, RAID. lncRNAs often
   exceed the 1024-nt window, so truncation or fragmentation policy matters.
3. **General RNA-RNA associations** — NPInter v4, RNAInter. Positive-sample
   quality varies; keep only those with explicit experimental descriptions.

Sequence understanding is the prerequisite. The repository trains the first rung
(miRNA-mRNA targeting) on MiRBench; the RNAcentral/Rfam corpus and the earlier
4-way candidate task remain as the sequence-understanding prelude.

### What RNAcentral gives us

The existing corpus is 145,201 unique RNAs across 1,947 Rfam families (rRNA 52%,
tRNA 12%, miRNA 3%, mRNA 2%, lncRNA 0%). It is a **sequence corpus, not a
relationship-label corpus**:

| resource | reuse |
|---|---|
| sequences + family annotations | yes — trains the sequence-understanding layer |
| miRNA sequences (4,645 rows, 718 families) | yes — maps onto TargetScan miRNAs |
| mRNA 3'UTR sequences | no — must come from Ensembl/GENCODE |
| lncRNA sequences | no — must come from GENCODE/NONCODE |
| miRNA-mRNA pair labels | no — must come from TargetScan/MiRBench |
| literature evidence snippets | no — must come from PubMed |

## Current task: miRNA-target relationship discrimination

`scripts/52_routeR_relation.py` trains the task the project actually needs — a
binary judgement over two raw sequences:

```
<|im_start|>system
<|im_end|><|im_start|>user
Q: <4 soft tokens: miRNA><4 soft tokens: mRNA>这条 miRNA 与这段 mRNA 窗口之间
是否存在经实验验证的靶向关系？只回答「是」或「否」。
<|im_end|><|im_start|>assistant
Answer: 是
```

- **Encoder** — frozen ERNIE-RNA encodes each sequence separately (miRNA ≤ 23 nt,
  mRNA window 50 nt). Separate encoding keeps the short miRNA from being drowned
  by the mRNA and gives the LLM an unambiguous boundary.
- **Projection** — one question-aware Perceiver resampler per sequence
  (ChatNT, Nat Mach Intell 2025): the question embedding is added to the learned
  queries, so the extracted soft prompt depends on what is being asked. Each
  resampler emits 4 tokens, 8 total. Plain 1-token linear projection (RNA-GPT) and
  the discrete FSQ code channel both lost on a controlled bridge comparison —
  FSQ scored 25.8% / 25.2% on a 4-way probe, i.e. chance.
- **LLM** — frozen Qwen3.6-35B-A3B int4 with SOS-LoRA adapters (r=8, k=4,
  preset C). Training loss is CE on the first answer token at the position just
  after `" A: "`; the same position is used at eval, so train and test never
  disagree about where the answer starts.
- **Metric** — AUROC on `logit(是) − logit(否)`. Random is 0.5.

Data is MiRBench (Zenodo 20540907, CC-BY-4.0): `AGO2_CLASH_Hejret2023`,
`AGO2_eCLIP_Manakov2022` (train/test/leftout), `AGO2_eCLIP_Klimentova2022`
(test). Negatives are bias-corrected by MiRBench, not random dinucleotides.
`scripts/build_mirbench.py` converts the six `tsv.gz` files into
`mirbench.jsonl` (4.13M rows: 2.85M `mirna_target`, 1.28M `mirna_region`).

20,000 train rows are a uniform reservoir sample of the 2.5M target-task train
rows, so CLASH and eCLIP are both represented. Reading all 4.1M rows into Python
dicts was what hard-crashed local runs at model-load time; the script builds the
model first and then reservoir-samples, and the notebook caps `--max-test`.

The end of training optionally runs an **input-perturbation check**
(`--perturb 1`): AUROC is recomputed with miRNAs permuted across rows, then with
mRNAs permuted. Both should fall to ~0.5. A high AUROC that survives permutation
means the model is exploiting something other than the sequences.

## GitHub Release assets

Training reads no GitHub dataset asset. The three assets still published are
legacy from the FSQ route and are only needed by `scripts/50_routeC_e2e.py`:

```text
dataset-v3-cand/rnaqa_v3_cand.jsonl            88 MB
codes-v3-cand/_rnaqa_v3_cand_codes.u16         18 MB
codes-v3-cand/_rnaqa_v3_cand_codes_index.pt    5.5 MB
```

Checkpoints for the relation run sync to a separate release,
`train-v3-rel`, so they can never be confused with the older candidate-task
checkpoints in `train-v3-checkpoints`.

## Training

The notebook:

1. clones this repository;
2. downloads ERNIE-RNA and Qwen3.6-35B-A3B;
3. downloads the six MiRBench files from Zenodo and builds `mirbench.jsonl`;
4. runs resumable full training directly.

The full-run cell samples GPU utilization every 0.75 seconds and prints a
three-second window average plus peak, writes the raw samples to
`runs/e2e_rel/gpu_metrics.csv`, and reports min/median/mean/max window
utilization when the segment exits.

## A100 40GB settings

The int4 base is 18.2 GiB; relation rows are short (miRNA 22 + mRNA 50 + prompt
≈ 80 tokens), so a batch of 64 is only ~5k tokens.

| setting | value | why |
|---|---|---|
| `--batch` | `64` | ~5k tokens per step; well inside 40 GB |
| `--grad-ckpt` | `1` | cheap here, and keeps the MoE transient buffers safe |
| `--steps` | `8000` | ~2.6 epochs over 200k sampled rows (1 epoch ≈ 3,125 steps) |
| `--eval-every` / `--eval-n` | `500` / `2000` | AUROC every 500 steps on a mixed-source test subset |
| `--save-every` / `--save-minutes` | `200` / `10` | ~1 s/step, so 10 minutes is the real bound; keep 3 |
| `--perturb` | `1` | permutation check on both inputs at the end |

## MoE expert execution

The 35B base has 256 experts per MoE layer and two int4 linears per expert.
Running them as a per-expert loop launches hundreds of tiny `Linear4bit` kernels
per layer — each dequantizes a small weight and then does a small GEMM — so the
GPU spends most of its time waiting between launches instead of computing.

Training now uses `--experts grouped` by default. It sorts routes by expert,
batch-dequantizes the hit experts' int4 weights into one contiguous buffer
(nested double-quant absmax reconstructed for all experts at once), and runs two
`bmm` ops per layer instead of the per-expert loop. The base weights stay frozen;
autograd propagates only to `hidden_states`. On an A100 40GB, one real-shape
layer (256 experts, hidden 2048, intermediate 512, 8192 tokens, top-8) measured
**14.3x** faster forward+backward than the per-expert path (0.405 s → 0.027 s),
with `max|Δforward| ≈ 3–8e-3` in bf16 and peak layer VRAM rising from ~1.2 to
~3.6 GiB (transient, under `--grad-ckpt 1`). The batched dequant is bit-identical
to the per-expert `dequantize_4bit` path; profiling showed the remaining per-layer
cost is now the routed-token gather/scatter plus the `bmm` GEMMs, not dequant
launches.

`torch._grouped_mm` is not used: on torch 2.11.0+cu130 its backward raises
`Invalid strides/sizes`, and bnb's `matmul_4bit` requires a 2D `quant_state`, so
NV4 experts cannot be batched through either directly. If `grouped` raises for
any reason the trainer logs it and falls back to the per-expert path for the rest
of the run — it never silently returns wrong numbers. `scripts/_test_grouped_experts.py`
compares both paths' forward output and dL/dx on GPU.

Gradient checkpointing costs throughput and buys 2-4x the activation budget.
With a frozen base and LoRA/SOS adapters it must be paired with
`enable_input_require_grads()`, otherwise gradients stop at `inputs_embeds`.
`scripts/50_routeC_e2e.py` does this automatically under `--grad-ckpt 1`, and
raises if checkpointing could not be enabled.

Qwen3.6 has 30 linear-attention layers. The notebook requires both
`flash-linear-attention[cuda]` and `causal-conv1d` before importing the model;
otherwise Transformers falls back to the pure-PyTorch chunk/recurrent
implementation and each step becomes far slower. FLA is installed from its
wheel. `causal-conv1d` is installed only from the exact wheel matching Python,
torch, CUDA, and C++ ABI; this repository publishes the Colab/A100 build in the
`kernel-wheels-v1` release. The notebook never falls back to a source build.
The trainer prints `[profile]` timings for its first three steps: batch/data,
model+LM loss, auxiliary loss, backward, and optimizer. The first step includes
cold-start kernel initialization and is not representative; compare steps 2-3.

The full-run cell wraps the trainer in `timeout --signal=TERM`, so a long run
can span more than one Colab segment. `--resume` restores the trainable weights,
the AdamW state, and the step counter. The cell only accepts a checkpoint from
the `train-v3-rel` release whose stored `tag` is `rel` and which actually
contains weights — an old candidate-task checkpoint is discarded, never
half-loaded.

The full-run cell prints per-step time and live GPU window utilization.

Checkpoint sync is optional. To persist training across Colab runtimes, create a
fine-grained GitHub token with Contents read/write permission and add it to
Colab Secrets as `GITHUB_TOKEN`. The notebook stores the latest checkpoint in a
release tagged `train-v3-rel`.

Checkpoints are saved every 200 steps or every 10 minutes, whichever comes
first — at ~1 s/step, the 10-minute bound is the one that actually fires on a
slow segment. The local runtime keeps the newest three snapshots. The background
GitHub sync uploads the newest complete snapshot and deletes the older remote
step files. On completion it also uploads `best.pt`.

Data, code, checkpoints, and artifacts live under `/content/rnaqwen_assets`, so
refreshing the GitHub code checkout does not delete them. They are still
runtime-local without a mounted persistent store or GitHub checkpoint sync.

GGUF quantization is a separate post-training step. It requires the original
35B base model plus a machine with enough disk for an intermediate merged model
and the final GGUF; it is not run inside the training cell.

## Released assets

- Dataset: https://github.com/RomanCohort/notebooktmp/releases/tag/dataset-v3
- FSQ checkpoints: https://github.com/RomanCohort/notebooktmp/releases/tag/fsq-stream-v1
- Python 3.13 / torch 2.11 / CUDA 13 kernel wheel: https://github.com/RomanCohort/notebooktmp/releases/tag/kernel-wheels-v1
- Relation-task checkpoints: https://github.com/RomanCohort/notebooktmp/releases/tag/train-v3-rel
- Legacy candidate-task checkpoints: https://github.com/RomanCohort/notebooktmp/releases/tag/train-v3-checkpoints
