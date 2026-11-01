# RNAQwen v3 Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RomanCohort/notebooktmp/blob/main/RNAQwen_v3_Full.ipynb)

Open `RNAQwen_v3_Full.ipynb` in Google Colab and select an A100 40GB runtime.
No Google Drive is required.

## GitHub Release asset

Create a release with tag `dataset-v3` and upload:

```text
rnaqa_v3_l234.jsonl
```

The notebook downloads it from:

```text
https://github.com/RomanCohort/notebooktmp/releases/download/dataset-v3/rnaqa_v3_l234.jsonl
```

## Training

The notebook:

1. clones this repository;
2. downloads ERNIE-RNA and Qwen3.6-35B-A3B;
3. downloads the published 25 MiB uint16 code cache (or regenerates it from the
   final dataset and FSQ tokenizer as a fallback);
4. runs resumable full training directly.

The full-run cell prints `[gpu]` utilization and VRAM every three seconds,
writes `runs/e2e_v3_full/gpu_metrics.csv`, and reports min/median/mean/max GPU
utilization when the segment exits.

## A100 40GB settings

`--token-budget 512` was an APU workaround: on the APU the binding constraint
was unified system memory. On a 40GB A100 the binding constraint is activation
memory, and the int4 base only takes 18.2 GiB.

| setting | value | why |
|---|---|---|
| `--token-budget` | `4096` | amortizes the fixed cost of MoE expert dispatch |
| `--batch` | `32` | hard cap on samples per step |
| `--grad-ckpt` | `1` | required at B*T=4096 on a 40GB A100 |
| `--steps` | `23250` | roughly the previous total token budget |
| `--min-steps` | `10000` | token-equivalent early-stop floor |
| `--eval-every` / `--eval-n` | `1000` / `16` | lower evaluation overhead |
| `--diag` | `0` | disable training-time diagnostics for throughput |

Gradient checkpointing costs roughly 30% throughput and buys 2-4x the token
budget. With a frozen base and LoRA/SOS adapters it must be paired with
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
The trainer prints `[profile]` timings for its first
three steps: batch/data, model+LM loss, auxiliary loss, backward, and optimizer.

If `--token-budget 4096` OOMs, use `--token-budget 3072 --batch 24` first.

The full-run cell wraps the trainer in `timeout --signal=TERM`, so a 23250-step
run is expected to span more than one Colab segment. Training resumes from the
newest checkpoint each time; the step counter is restored, so `--min-steps`
accumulates across segments. The dataset has 1,060,818 QA rows total
(831,390 train / 66,796 val / 162,632 test), so 23,250 steps must not be
described as one epoch. Sampling is without replacement inside each epoch.

The full-run cell prints per-step time, a three-step profile, and live GPU
utilization. If reserved VRAM approaches 39 GiB, fall back to 3072/24.

Checkpoint sync is optional. To persist training across Colab runtimes, create a
fine-grained GitHub token with Contents read/write permission and add it to
Colab Secrets as `GITHUB_TOKEN`. The notebook stores the latest checkpoint in a
release tagged `train-v3-checkpoints`.

Checkpoints are saved every 100 steps or every 10 minutes, whichever comes
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
