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
3. mean-pools 4 hidden positions and streams them through the trained FSQ into a
   compact uint16 code cache (about 25 MiB);
4. runs a 10-step smoke test;
5. runs resumable full training.

## A100 40GB settings

`--token-budget 512` was an APU workaround: on the APU the binding constraint
was unified system memory. On a 40GB A100 the binding constraint is activation
memory, and the int4 base only takes 18.2 GiB.

| setting | value | why |
|---|---|---|
| `--token-budget` | `2048` | ~18 samples/step instead of ~4 |
| `--batch` | `16` | hard cap on samples per step |
| `--grad-ckpt` | `1` | activations ~12 GiB instead of ~40 GiB at B*T=2048 |
| `--steps` | `46500` | one epoch of `rnaqa_v3_l234.jsonl` (95.2M / 2048) |
| `--min-steps` | `20000` | keeps early stopping from firing at 4k steps |
| `--ablation-every` | `500` | vsperm/vszero/vsrandom/vsconst at every eval point |

Gradient checkpointing costs roughly 30% throughput and buys 2-4x the token
budget. With a frozen base and LoRA/SOS adapters it must be paired with
`enable_input_require_grads()`, otherwise gradients stop at `inputs_embeds`.
`scripts/50_routeC_e2e.py` does this automatically under `--grad-ckpt 1`, and
raises if checkpointing could not be enabled.

If `--token-budget 2048` still OOMs, drop to `1536` first; only fall back to
`1024` if that fails too.

The full-run cell wraps the trainer in `timeout --signal=TERM`, so a 46500-step
epoch is expected to span more than one Colab segment. Training resumes from the
newest checkpoint each time; the step counter is restored, so `--min-steps`
accumulates across segments.

Checkpoint sync is optional. To persist training across Colab runtimes, create a
fine-grained GitHub token with Contents read/write permission and add it to
Colab Secrets as `GITHUB_TOKEN`. The notebook stores the latest checkpoint in a
release tagged `train-v3-checkpoints`.

Checkpoints are saved every 100 steps or every 10 minutes, whichever comes
first. The local runtime keeps the newest three snapshots. The background GitHub
sync uploads the newest complete snapshot before deleting the previous release
asset.

The code cache is runtime-local. A fresh Colab runtime regenerates it from the
dataset and trained FSQ tokenizer before resuming. The FSQ encoder is frozen in
this low-disk mode; the LLM adapter and code embedding table remain trainable.

## Released assets

- Dataset: https://github.com/RomanCohort/notebooktmp/releases/tag/dataset-v3
- FSQ checkpoints: https://github.com/RomanCohort/notebooktmp/releases/tag/fsq-stream-v1
- Checkpoints: https://github.com/RomanCohort/notebooktmp/releases/tag/train-v3-checkpoints
