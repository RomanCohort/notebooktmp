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
4. runs a 10-step smoke test;
5. runs resumable full training.

## A100 40GB settings

`--token-budget 512` was an APU workaround: on the APU the binding constraint
was unified system memory. On a 40GB A100 the binding constraint is activation
memory, and the int4 base only takes 18.2 GiB.

| setting | value | why |
|---|---|---|
| `--token-budget` | `2048` | caps padded `B*T`; the actual sample count varies |
| `--batch` | `16` | hard cap on samples per step |
| `--grad-ckpt` | `1` | activations ~12 GiB instead of ~40 GiB at B*T=2048 |
| `--steps` | `46500` | fixed compute budget, not one epoch |
| `--min-steps` | `20000` | keeps early stopping from firing at 4k steps |
| `--ablation-every` | `500` | vsperm/vszero/vsrandom/vsconst at every eval point |

Gradient checkpointing costs roughly 30% throughput and buys 2-4x the token
budget. With a frozen base and LoRA/SOS adapters it must be paired with
`enable_input_require_grads()`, otherwise gradients stop at `inputs_embeds`.
`scripts/50_routeC_e2e.py` does this automatically under `--grad-ckpt 1`, and
raises if checkpointing could not be enabled.

Qwen3.6 has 30 linear-attention layers. Install `flash-linear-attention[cuda]`
and `causal-conv1d` before importing the model; otherwise Transformers falls
back to the pure-PyTorch chunk/recurrent implementation and each step becomes
far slower. The notebook attempts this installation and prints the detected
kernel status. The trainer also prints `[profile]` timings for its first three
steps: batch/data, model+LM loss, auxiliary loss, backward, and optimizer.

If `--token-budget 2048` still OOMs, drop to `1536` first; only fall back to
`1024` if that fails too.

The full-run cell wraps the trainer in `timeout --signal=TERM`, so a 46500-step
run is expected to span more than one Colab segment. Training resumes from the
newest checkpoint each time; the step counter is restored, so `--min-steps`
accumulates across segments. The dataset has 1,060,818 QA rows total
(831,390 train / 66,796 val / 162,632 test), so 46,500 steps must not be
described as one epoch. Sampling is without replacement inside each epoch.

Measure the first few step timestamps before committing to the long run. One
observed A100 40GB segment ran at about 29 s/step after model load: 46,500 steps
would take about 15 days and roughly 50 Colab segments. 100 compute units are
not enough for that schedule; use a shorter pilot, a persistent GPU, or a larger
token budget only after checking VRAM headroom.

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
- Checkpoints: https://github.com/RomanCohort/notebooktmp/releases/tag/train-v3-checkpoints
