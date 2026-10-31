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

SESSION="${TMUX_SESSION:-rnaqwen}"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" \
  "RNAQ_ENV_FILE='$ENV_FILE' '$SCRIPT_DIR/run_train.sh'"
echo "started tmux session: $SESSION"
echo "attach: tmux attach -t $SESSION"
echo "logs:   tail -f '$RNAQ_DIR/logs/deploy-$TAG.log'"
