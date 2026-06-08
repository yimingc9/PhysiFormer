#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
GENERATIONS="${GENERATIONS:-3}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SAMPLING_STEPS="${SAMPLING_STEPS:-}"
CHECKPOINT="${CHECKPOINT:-checkpoints/checkpoint-best.pt}"
RENDER_FLAG="${RENDER_FLAG:---save-mp4}"
OVERWRITE_FLAG="${OVERWRITE_FLAG:---no-overwrite}"
DRY_RUN_FLAG="${DRY_RUN_FLAG:-}"
OOD_ELASTIC="${OOD_ELASTIC:-horse fish bunny}"
OOD_RIGID="${OOD_RIGID:-cow teapot}"

COMMON_ARGS=(
  --demo-root ood_examples
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
if [[ -n "$DRY_RUN_FLAG" ]]; then
  COMMON_ARGS+=("$DRY_RUN_FLAG")
fi

MATERIAL_ARGS=()
for pattern in $OOD_ELASTIC; do
  MATERIAL_ARGS+=(--elastic "$pattern")
done
for pattern in $OOD_RIGID; do
  MATERIAL_ARGS+=(--rigid "$pattern")
done

# OOD folders: ood_examples/2obj_cow_horse and ood_examples/3obj_teapot_fish_bunny.
# Available object material names: cow, horse, teapot, fish, bunny.
"$PYTHON" run_official_demo_inference.py \
  "${COMMON_ARGS[@]}" \
  "${MATERIAL_ARGS[@]}"
