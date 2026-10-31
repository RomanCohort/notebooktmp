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
3. streams ERNIE hidden states into a compact uint16 code cache (about 100 MiB);
4. runs a 10-step smoke test;
5. runs resumable full training.

Checkpoint sync is optional. To persist training across Colab runtimes, create a
fine-grained GitHub token with Contents read/write permission and add it to
Colab Secrets as `GITHUB_TOKEN`. The notebook stores the latest checkpoint in a
release tagged `train-v3-checkpoints`.

The code cache is runtime-local. A fresh Colab runtime regenerates it from the
dataset and trained FSQ tokenizer before resuming. The FSQ encoder is frozen in
this low-disk mode; the LLM adapter and code embedding table remain trainable.

## Released assets

- Dataset: https://github.com/RomanCohort/notebooktmp/releases/tag/dataset-v3
- Checkpoints: https://github.com/RomanCohort/notebooktmp/releases/tag/train-v3-checkpoints
