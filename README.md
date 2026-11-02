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

Sequence understanding is the prerequisite. The current 4-way candidate task
(below) is a first, cheap proxy for it, trained on RNAcentral/Rfam data.

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

## Current proxy task: candidate discrimination (4-way)

This is a first, cheap proxy for *sequence understanding* — it is not the target
relationship task. It tests whether the adapter can tell RNAs apart at all.

## Task: candidate discrimination (4-way)

Earlier runs asked the model to *generate* one of 1,947 free-text family names,
with train and test families disjoint. Exact match was therefore structurally 0
no matter how long it trained, and validation CE plateaued at step 1000.

The current task gives the sequence plus four candidate families and asks for a
letter:

```
Q: <RNA>...sequence...</RNA>Which of these Rfam families does this RNA belong to?
A) RF01788
B) RF00651
C) RF00005
D) RF00053
Answer with the letter only.
A: C
```

Family names appear in the prompt, so the model never has to memorise them, and
the split is random — 100% of test families appear in train. Choices are balanced
(A/B/C/D ≈ 25% each), so the random baseline is 25%.

## GitHub Release asset

Create a release with tag `dataset-v3-cand` and upload:

```text
rnaqa_v3_cand.jsonl           88 MB   sha256 d8704345d58b4db9...
```

Create a second release with tag `codes-v3-cand` and upload the code cache:

```text
_rnaqa_v3_cand_codes.u16      18 MB   sha256 134851c11291ea82...
_rnaqa_v3_cand_codes_index.pt  5.5 MB sha256 a39f15baae38f983...
```

The notebook downloads them from:

```text
https://github.com/RomanCohort/notebooktmp/releases/download/dataset-v3-cand/rnaqa_v3_cand.jsonl
https://github.com/RomanCohort/notebooktmp/releases/download/codes-v3-cand/_rnaqa_v3_cand_codes.u16
https://github.com/RomanCohort/notebooktmp/releases/download/codes-v3-cand/_rnaqa_v3_cand_codes_index.pt
```

The code cache encodes 120,113 unique sequences (9,392,568 pooled positions) with
the frozen FSQ tokenizer. Encoding takes about 30 minutes on a local ROCm APU —
but only 10 minutes with `--sort-by-length`, which packs same-length sequences into
one batch and removes most padding (9 seq/s → 163 seq/s, 17x).

## Training

The notebook:

1. clones this repository;
2. downloads ERNIE-RNA and Qwen3.6-35B-A3B;
3. downloads the published 25 MiB uint16 code cache (or regenerates it from the
   final dataset and FSQ tokenizer as a fallback);
4. runs resumable full training directly.

The full-run cell samples GPU utilization every 0.75 seconds and prints a
three-second window average plus peak, writes the raw samples to
`runs/e2e_v3_cand/gpu_metrics.csv`, and reports min/median/mean/max window
utilization when the segment exits.

## A100 40GB settings

`--token-budget 512` was an APU workaround: on the APU the binding constraint
was unified system memory. On a 40GB A100 the binding constraint is activation
memory, and the int4 base only takes 18.2 GiB.

| setting | value | why |
|---|---|---|
| `--token-budget` | `8192` | doubles the work covered by the fixed MoE dispatch cost |
| `--batch` | `64` | hard cap on samples per step |
| `--grad-ckpt` | `1` | keeps B*T=8192 within the 40GB A100 |
| `--steps` | `8000` | ~4.8 epochs over 87,307 candidate rows (1 epoch ≈ 1,679 steps) |
| `--min-steps` | `2000` | early-stop floor, ~1.2 epochs |
| `--eval-every` / `--eval-n` | `250` / `64` | more eval points over a shorter run |
| `--diag` | `0` | disable training-time diagnostics for throughput |

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

If `--token-budget 8192` OOMs, use `--token-budget 6144 --batch 48`, then
`--token-budget 4096 --batch 32`.

The full-run cell wraps the trainer in `timeout --signal=TERM`, so an 8000-step
run is expected to span more than one Colab segment. Training resumes from the
newest checkpoint each time; the step counter is restored, so `--min-steps`
accumulates across segments. The dataset has 1,060,818 QA rows total
(831,390 train / 66,796 val / 162,632 test), so 23,250 steps must not be
described as one epoch. Sampling is without replacement inside each epoch.

The full-run cell prints per-step time, a three-step profile, and live GPU
window utilization. If reserved VRAM approaches 39 GiB, fall back to 6144/48.

Checkpoint sync is optional. To persist training across Colab runtimes, create a
fine-grained GitHub token with Contents read/write permission and add it to
Colab Secrets as `GITHUB_TOKEN`. The notebook stores the latest checkpoint in a
release tagged `train-v3-checkpoints`.

Checkpoints are saved every 50 steps or every 10 minutes, whichever comes
first. The local runtime keeps the newest three snapshots. The background GitHub
sync validates and uploads the newest complete snapshot. On completion it also
uploads `final.pt`, `best.pt`, and a zipped `artifacts/` directory.

Data, code, checkpoints, and artifacts live under `/content/rnaqwen_assets`, so
refreshing the GitHub code checkout does not delete them. They are still
runtime-local without a mounted persistent store or GitHub checkpoint sync. The
FSQ encoder is frozen in this low-disk mode; the LLM adapter and code embedding
table remain trainable.

GGUF quantization is a separate post-training step. It requires the original
35B base model plus a machine with enough disk for an intermediate merged model
and the final GGUF; it is not run inside the training cell.

## Released assets

- Dataset: https://github.com/RomanCohort/notebooktmp/releases/tag/dataset-v3
- FSQ checkpoints: https://github.com/RomanCohort/notebooktmp/releases/tag/fsq-stream-v1
- Python 3.13 / torch 2.11 / CUDA 13 kernel wheel: https://github.com/RomanCohort/notebooktmp/releases/tag/kernel-wheels-v1
- Checkpoints: https://github.com/RomanCohort/notebooktmp/releases/tag/train-v3-checkpoints
