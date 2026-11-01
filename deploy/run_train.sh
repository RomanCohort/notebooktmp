#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${RNAQ_ENV_FILE:-$SCRIPT_DIR/train.env}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: $ENV_FILE" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

: "${RNAQ_DIR:?}" "${PYTHON_BIN:?}" "${QWEN_REPO:?}" "${ERNIE_REPO:?}"
: "${DATASET_URL:?}" "${DATASET_SHA256:?}" "${TAG:?}"
: "${BACKUP_DIR:=$RNAQ_DIR/backups}" "${HF_HOME:=$RNAQ_DIR/.hf}"

export HF_HOME
export RNAQ_THREADS="${RNAQ_THREADS:-8}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

DATA_DIR="$RNAQ_DIR/data/processed"
CKPT_DIR="$RNAQ_DIR/runs/e2e_$TAG/ckpt"
MODEL_DIR="$RNAQ_DIR/models/Qwen3.6-35B-A3B"
ERNIE_DIR="$RNAQ_DIR/models/ernierna"
FSQ="$RNAQ_DIR/checkpoints/fsq_stream/tokenizer.pt"
CODE_EMB="$RNAQ_DIR/checkpoints/fsq_stream/code_emb_contentinit.pt"
DATASET="$DATA_DIR/rnaqa_v3_l234.jsonl"
CODES="$DATA_DIR/_rnaqa_v3_codes.u16"
CODE_IDX="$DATA_DIR/_rnaqa_v3_codes_index.pt"
LOG_DIR="$RNAQ_DIR/logs"
LOG="$LOG_DIR/deploy-$TAG.log"
LOCK="$RNAQ_DIR/.train-$TAG.lock"

mkdir -p "$DATA_DIR" "$CKPT_DIR" "$MODEL_DIR" "$ERNIE_DIR" "$LOG_DIR" "$BACKUP_DIR"
cd "$RNAQ_DIR"

# One trainer per tag; a second process exits instead of competing for the GPU.
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "Training is already running for tag=$TAG" >&2
  exit 3
fi

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
log "start tag=$TAG host=$(hostname)"

if [[ ! -x "$PYTHON_BIN" ]]; then
  log "Python not executable: $PYTHON_BIN"
  exit 2
fi

GPU_LINE=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits |
  head -n 1)
GPU_NAME=$(printf '%s' "${GPU_LINE%%,*}" | xargs)
GPU_MIB=$(printf '%s' "${GPU_LINE##*,}" | tr -dc '0-9')
log "gpu=$GPU_NAME memory=${GPU_MIB}MiB"
if (( GPU_MIB < 30000 )); then
  log "GPU has less than 30 GiB. The 35B int4 model cannot train reliably."
  exit 2
fi

FREE_GB=$(df -Pk "$RNAQ_DIR" | awk 'NR==2 {printf "%.1f", $4/1024/1024}')
log "disk_free=${FREE_GB}GiB"
if awk -v f="$FREE_GB" 'BEGIN {exit !(f < 80)}'; then
  log "Need at least 80 GiB free for base model, data, and working files."
  exit 2
fi

RAM_GB=$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
log "ram_total=${RAM_GB}GiB"
if (( RAM_GB < 60 )); then
  log "WARNING: less than 60 GiB RAM; model loading may be unstable."
fi

if ! "$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
  log "torch CUDA is unavailable. Install the wheel matching the NVIDIA driver first."
  exit 2
fi

if [[ -f "$DATASET" ]] && [[ "$(stat -c%s "$DATASET")" != "663240811" ]]; then
  log "removing incomplete dataset"
  rm -f "$DATASET"
fi
if [[ ! -f "$DATASET" ]]; then
  log "downloading dataset"
  curl -fL --retry 8 --retry-all-errors -C - -o "$DATASET.part" "$DATASET_URL"
  mv "$DATASET.part" "$DATASET"
fi
ACTUAL_SHA=$(sha256sum "$DATASET" | awk '{print $1}')
if [[ "$ACTUAL_SHA" != "$DATASET_SHA256" ]]; then
  log "dataset sha256 mismatch: $ACTUAL_SHA"
  rm -f "$DATASET"
  exit 2
fi

if [[ ! -f "$ERNIE_DIR/model.safetensors" ]] || [[ ! -f "$ERNIE_DIR/config.json" ]]; then
  log "downloading ERNIE-RNA"
  "$PYTHON_BIN" - "$ERNIE_REPO" "$ERNIE_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])
PY
fi

if [[ ! -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
  log "downloading base model"
  "$PYTHON_BIN" -u "$RNAQ_DIR/scripts/00b_download_modelscope.py" \
    --repo "$QWEN_REPO" --dest "$MODEL_DIR" --workers "${DOWNLOAD_WORKERS:-4}"
fi

"$PYTHON_BIN" - "$MODEL_DIR" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
index = json.loads((root / "model.safetensors.index.json").read_text())
missing = sorted({n for n in index["weight_map"].values() if not (root / n).is_file()})
if missing:
    raise SystemExit(f"base model incomplete: {missing[:5]}")
PY

if [[ ! -f "$FSQ" ]] || [[ ! -f "$CODE_EMB" ]]; then
  log "missing FSQ assets; run git pull or check checkpoints/fsq_stream"
  exit 2
fi

export RNAQ_QWEN36_DIR="$MODEL_DIR"
export RNAQ_QWEN8_DIR="$RNAQ_DIR/models/Qwen3-8B"

if [[ ! -f "$CODE_IDX" ]]; then
  log "encoding v3 dataset into uint16 codes"
  "$PYTHON_BIN" -X utf8 -u "$RNAQ_DIR/scripts/81_encode_v3_codes.py" \
    --data "$DATASET" --encoder "$ERNIE_DIR" --fsq "$FSQ" \
    --out "$CODES" --idx "$CODE_IDX" --pool 4 --batch "${CODE_BATCH:-16}" \
    2>&1 | tee -a "$LOG"
fi

resume_args=()
LATEST=$(find "$CKPT_DIR" -maxdepth 1 -type f -name 'step_*.pt' -printf '%f\n' 2>/dev/null |
  sort | tail -n 1 || true)
if [[ -n "$LATEST" ]]; then
  resume_args=(--resume "$CKPT_DIR/$LATEST")
  log "resume=$CKPT_DIR/$LATEST"
else
  log "starting from scratch"
fi

# Keep the latest completed checkpoint on a second local path. The trainer writes
# step_*.pt atomically, so this loop never copies a partially written checkpoint.
(
  while sleep 60; do
    SRC=$(find "$CKPT_DIR" -maxdepth 1 -type f -name 'step_*.pt' -printf '%T@ %p\n' 2>/dev/null |
      sort -n | tail -n 1 | cut -d' ' -f2- || true)
    [[ -n "$SRC" ]] || continue
    TMP="$BACKUP_DIR/${TAG}-latest.pt.tmp"
    cp -f "$SRC" "$TMP"
    mv -f "$TMP" "$BACKUP_DIR/${TAG}-latest.pt"
  done
) &
BACKUP_PID=$!
trap 'kill "$BACKUP_PID" 2>/dev/null || true' EXIT

log "launching SOS-LoRA training"
set +e
"$PYTHON_BIN" -X utf8 -u "$RNAQ_DIR/scripts/50_routeC_e2e.py" \
  --data "$DATASET" \
  --codes "$CODES" --codes-idx "$CODE_IDX" \
  --fsq "$FSQ" --init-code-emb "$CODE_EMB" \
  --base qwen36-35b --qwen36-mode int4 \
  --adapter sos --lora-r "$LORA_R" --lora-preset "$LORA_PRESET" \
  --token-budget "$TOKEN_BUDGET" \
  --steps "$STEPS" --min-steps "$MIN_STEPS" \
  --early-stop 1 --eval-every "$EVAL_EVERY" --patience "$PATIENCE" \
  --save-every "$SAVE_EVERY" --save-minutes "$SAVE_MINUTES" \
  --keep-ckpt "$KEEP_CKPT" --grad-ckpt 1 --experts grouped \
  --tag "$TAG" "${resume_args[@]}" \
  2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
set -e
log "trainer exit=$RC"
exit "$RC"
