#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
GENERATIONS="${GENERATIONS:-3}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SAMPLING_STEPS="${SAMPLING_STEPS:-}"
CHECKPOINT="${CHECKPOINT:-checkpoints/checkpoint-best.pt}"
RENDER_FLAG="${RENDER_FLAG:---save-mp4}"
GT_RENDER_FLAG="${GT_RENDER_FLAG:---save-gt-mp4}"
OVERWRITE_FLAG="${OVERWRITE_FLAG:---no-overwrite}"
DRY_RUN_FLAG="${DRY_RUN_FLAG:-}"

COMMON_ARGS=(
  --checkpoint "$CHECKPOINT"
  --include all
  --generations "$GENERATIONS"
  --max-samples "$MAX_SAMPLES"
  "$OVERWRITE_FLAG"
)

if [[ -n "$SAMPLING_STEPS" ]]; then
  COMMON_ARGS+=(--num-sampling-steps "$SAMPLING_STEPS")
fi

if [[ -n "$RENDER_FLAG" ]]; then
  COMMON_ARGS+=("$RENDER_FLAG")
fi
if [[ -n "$GT_RENDER_FLAG" ]]; then
  COMMON_ARGS+=("$GT_RENDER_FLAG")
fi
if [[ -n "$DRY_RUN_FLAG" ]]; then
  COMMON_ARGS+=("$DRY_RUN_FLAG")
fi

# Run rigid and soft separately so each group gets the correct material conditioning.
"$PYTHON" run_official_demo_inference.py \
  --demo-root indistri_examples/rigid \
  "${COMMON_ARGS[@]}" \
  --rigid all

"$PYTHON" run_official_demo_inference.py \
  --demo-root indistri_examples/soft \
  "${COMMON_ARGS[@]}" \
  --elastic all
