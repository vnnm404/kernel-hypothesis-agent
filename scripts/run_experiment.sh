#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${KHA_MODEL:?Set KHA_MODEL to an explicit model ID available in your configured provider}"
CONFIG=${CONFIG:-experiments/blackwell.json}
BACKEND=${BACKEND:-claude}
GPU=${GPU:-0}
OUT_DIR=${OUT_DIR:-runs/$(date -u +%Y%m%dT%H%M%SZ)}
REPETITIONS=${REPETITIONS:-3}
kha doctor --config "$CONFIG" --backend "$BACKEND" --gpu "$GPU"
exec kha campaign --config "$CONFIG" --backend "$BACKEND" --model "$KHA_MODEL" \
  --gpu "$GPU" --repetitions "$REPETITIONS" --out "$OUT_DIR" "$@"
