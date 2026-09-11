#!/usr/bin/env bash
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=physiformer_toy
#SBATCH --output=physiformer-toy-%j.out

set -euo pipefail

usage() {
  cat <<'EOF'
Train PhysiFormer-L (default) or PhysiFormer-B on the mixed rigid/elastic toy 90/10 split.
Usage: bash scripts/train_toy_physiformer.sh [--gpus N] [--dry-run] [-- TRAINER_ARGS...]

Defaults: PhysiFormer-L, 2 GPUs, batch 8/GPU, accumulation 4, effective batch 64, 6000 epochs.
AltObj protocol: virtual length 10000, epoch LR, EMA validation every 10 epochs.
Activate your PyTorch environment first; PYTHON_BIN defaults to python3 on PATH.
Dataset overrides: DATA_ROOT, SPLIT_FILE, TRAIN_SPLIT, VAL_SPLIT, NUM_FRAMES,
NUM_VERTICES, MAX_NUM_OBJECTS, COND_OBJECT_MATERIAL (0|1), OBJECT_MATERIAL_DIM.
Other overrides: REPO_ROOT, OUT_DIR, PYTHON_BIN, NPROC_PER_NODE, MODEL,
BATCH_SIZE, GRAD_ACCUM, EVAL_BATCH_SIZE, EPOCHS, TRAIN_VIRTUAL_LENGTH, NUM_WORKERS,
LR, MIN_LR, WARMUP_EPOCHS, AMP, SEED, RESUME (auto|none|checkpoint path),
LOG_FREQ, EVAL_EPOCH_FREQ, SAVE_EPOCH_FREQ, SAVE_LAST_FREQ.
Additional trainer flags after -- take precedence.
CUDA_VISIBLE_DEVICES is respected; the launcher never replaces the GPU allocation.

Examples:
  bash scripts/train_toy_physiformer.sh --dry-run
  bash scripts/train_toy_physiformer.sh --gpus 1
  MODEL=PhysiFormer-B bash scripts/train_toy_physiformer.sh
  DATA_ROOT=/path/to/npzs SPLIT_FILE=/path/to/split.json NUM_VERTICES=0 MAX_NUM_OBJECTS=5 COND_OBJECT_MATERIAL=0 bash scripts/train_toy_physiformer.sh
  RESUME=none OUT_DIR=runs/toy_finetune bash scripts/train_toy_physiformer.sh -- --init_ckpt /path/to/checkpoint.pt
  sbatch --partition=YOUR_PARTITION scripts/train_toy_physiformer.sh
  sbatch --gres=gpu:h100:1 scripts/train_toy_physiformer.sh --gpus 1
Slurm resources must be changed with sbatch options, separately from --gpus.
EOF
}

die() { echo "[error] $*" >&2; exit 2; }
dry_run=0
extra_args=()
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
while (($#)); do
  case "$1" in
    --gpus) [[ $# -ge 2 ]] || die '--gpus requires a count'; NPROC_PER_NODE="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; extra_args=("$@"); break ;;
    *) die "Unknown option: $1 (see --help)" ;;
  esac
done

# Prefer the script's checkout for direct execution, including inside an
# interactive Slurm allocation whose SLURM_SUBMIT_DIR may point elsewhere.
# sbatch spools the script, so fall back to the working/submission directory.
script_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${REPO_ROOT:-}" ]]; then
  for candidate in "${script_root}" "${PWD}" "${SLURM_SUBMIT_DIR:-}"; do
    if [[ -n "${candidate}" && -f "${candidate}/src/physiformer/scripts/train_npz_elastic.py" ]]; then
      REPO_ROOT="${candidate}"
      break
    fi
  done
fi
[[ -n "${REPO_ROOT:-}" ]] || die 'Cannot locate the PhysiFormer repository; set REPO_ROOT explicitly.'
[[ -f "${REPO_ROOT}/src/physiformer/scripts/train_npz_elastic.py" ]] || die "REPO_ROOT=${REPO_ROOT} does not contain src/physiformer/scripts/train_npz_elastic.py"
cd -- "${REPO_ROOT}"
REPO_ROOT="$(pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v "${PYTHON_BIN}" >/dev/null || die "Python not found: ${PYTHON_BIN}"
# Dataset-specific settings: change these when adapting this training example.
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data_toy}"
SPLIT_FILE="${SPLIT_FILE:-${DATA_ROOT}/split.json}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-val}"
NUM_FRAMES="${NUM_FRAMES:-49}"
NUM_VERTICES="${NUM_VERTICES:-356}"
MAX_NUM_OBJECTS="${MAX_NUM_OBJECTS:-10}"
COND_OBJECT_MATERIAL="${COND_OBJECT_MATERIAL:-1}"
OBJECT_MATERIAL_DIM="${OBJECT_MATERIAL_DIM:-1}"
[[ "${COND_OBJECT_MATERIAL}" =~ ^[01]$ ]] || die 'COND_OBJECT_MATERIAL must be 0 or 1'
if [[ "${COND_OBJECT_MATERIAL}" == 1 ]]; then
  [[ "${OBJECT_MATERIAL_DIM}" =~ ^[1-9][0-9]*$ ]] || die 'OBJECT_MATERIAL_DIM must be positive'
else
  OBJECT_MATERIAL_DIM=0
fi

# Training recipe from the full-dataset AltObj run.
MODEL="${MODEL:-PhysiFormer}"
case "${MODEL}" in
  PhysiFormer) model_tag=physiformer_l ;;
  PhysiFormer-B) model_tag=physiformer_b ;;
  *) die 'MODEL must be PhysiFormer (L) or PhysiFormer-B' ;;
esac
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/runs/toy_${model_tag}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
for name in NPROC_PER_NODE BATCH_SIZE; do
  [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer"
done
micro_batch=$((NPROC_PER_NODE * BATCH_SIZE))
if [[ -z "${GRAD_ACCUM:-}" ]]; then
  ((64 % micro_batch == 0)) || die 'Set GRAD_ACCUM explicitly when GPU count * batch size does not divide 64.'
  GRAD_ACCUM=$((64 / micro_batch))
fi
[[ "${GRAD_ACCUM}" =~ ^[1-9][0-9]*$ ]] || die 'GRAD_ACCUM must be a positive integer'
RESUME="${RESUME:-auto}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

# Validate the selected dataset without imposing the toy group layout or count.
data_args=(--data-root "${DATA_ROOT}" --split-file "${SPLIT_FILE}"
  --train-split "${TRAIN_SPLIT}" --val-split "${VAL_SPLIT}" --material-dim "${OBJECT_MATERIAL_DIM}")
"${PYTHON_BIN}" scripts/prepare_training_inputs.py "${data_args[@]}"
train_args=(
  --precomp_root "${DATA_ROOT}"
  --precomp_roots_json "${OUT_DIR}/precomp_roots.json"
  --split_file "${SPLIT_FILE}" --split_name "${TRAIN_SPLIT}" --val_split_name "${VAL_SPLIT}"
  --output_dir "${OUT_DIR}" --stats_json "${OUT_DIR}/train_position_stats.json"
  --recompute_norm
  --training_protocol altobj --ema_decay 0.9999
  --model "${MODEL}" --num_frames "${NUM_FRAMES}" --num_vertices "${NUM_VERTICES}"
  --num_classes 1 --max_num_objects "${MAX_NUM_OBJECTS}"
  --object_material_dim "${OBJECT_MATERIAL_DIM}"
  --material_alias_values_json "${OUT_DIR}/material_alias_values.json"
  --use_rope --num_register_tokens 16 --max_frames 128 --max_vertices 8192
  --grad_checkpoint --noise_scale 0.1 --amp "${AMP:-bf16}"
  --batch_size "${BATCH_SIZE}" --eval_batch_size "${EVAL_BATCH_SIZE:-${BATCH_SIZE}}"
  --grad_accum "${GRAD_ACCUM}" --train_virtual_length "${TRAIN_VIRTUAL_LENGTH:-10000}"
  --num_workers "${NUM_WORKERS:-4}"
  --lr "${LR:-4e-5}" --min_lr "${MIN_LR:-5e-6}"
  --warmup_epochs "${WARMUP_EPOCHS:-5}" --lr_schedule cosine
  --epochs "${EPOCHS:-6000}" --seed "${SEED:-0}"
  --log_freq "${LOG_FREQ:-5}" --eval_epoch_freq "${EVAL_EPOCH_FREQ:-10}"
  --save_epoch_freq "${SAVE_EPOCH_FREQ:-10}" --save_last_freq "${SAVE_LAST_FREQ:-1000}"
)
if [[ "${COND_OBJECT_MATERIAL}" == 1 ]]; then
  train_args+=(--cond_object_material)
else
  train_args+=(--no_cond_object_material)
fi
if [[ "${RESUME}" == auto ]]; then
  if [[ -f "${OUT_DIR}/checkpoint-last.pt" ]]; then
    train_args+=(--resume "${OUT_DIR}/checkpoint-last.pt")
  fi
elif [[ "${RESUME}" != none ]]; then
  [[ -f "${RESUME}" ]] || die "Checkpoint not found: ${RESUME}"
  train_args+=(--resume "${RESUME}")
fi
train_args+=("${extra_args[@]}")
command_args=("${PYTHON_BIN}" -m torch.distributed.run --standalone --nnodes=1
  --nproc-per-node "${NPROC_PER_NODE}" -m physiformer.scripts.train_npz_elastic "${train_args[@]}")
echo "[train] model=${MODEL} GPUs=${NPROC_PER_NODE} batch/GPU=${BATCH_SIZE} accumulation=${GRAD_ACCUM} nominal_effective_batch=$((micro_batch * GRAD_ACCUM))"
printf '[command] '; printf '%q ' "${command_args[@]}"; printf '\n'
if ((dry_run)); then exit 0; fi

# Validate the final trainer arguments and GPU allocation before creating outputs.
"${PYTHON_BIN}" - "${NPROC_PER_NODE}" "${train_args[@]}" <<'PY'
import math
import sys
from pathlib import Path
import torch
from physiformer.scripts.train_npz_elastic import build_argparser, load_split_entries
args = build_argparser().parse_args(sys.argv[2:])
if args.init_ckpt and args.resume:
    raise SystemExit("Choose checkpoint initialization or resume; set RESUME=none when using --init_ckpt.")
for checkpoint in (args.init_ckpt, args.resume):
    if checkpoint and not Path(checkpoint).is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
for name in ("epochs", "batch_size", "eval_batch_size", "grad_accum", "log_freq", "save_last_freq"):
    if getattr(args, name) <= 0:
        raise SystemExit(f"{name} must be positive")
n = int(sys.argv[1])
train_size = args.train_virtual_length if args.train_virtual_length > 0 else len(load_split_entries(args.split_file, args.split_name))
per_rank = train_size // n if args.training_protocol == "altobj" else math.ceil(train_size / n)
if args.training_protocol == "altobj" and per_rank // args.batch_size < args.grad_accum:
    raise SystemExit("No complete accumulation group; reduce GRAD_ACCUM/BATCH_SIZE or increase TRAIN_VIRTUAL_LENGTH.")
if math.ceil(train_size / n) < args.batch_size:
    raise SystemExit("No complete training minibatch per GPU; reduce BATCH_SIZE or GPU count.")
if torch.cuda.device_count() < n:
    raise SystemExit(f"Requested {n} GPUs, but only {torch.cuda.device_count()} CUDA devices are visible.")
if args.amp == "bf16":
    for i in range(n):
        with torch.cuda.device(i):
            if not torch.cuda.is_bf16_supported():
                raise SystemExit(f"GPU {i} does not support BF16; set AMP=fp16 or AMP=none.")
print(f"[env] torch={torch.__version__} CUDA={torch.version.cuda}")
print("[env] GPUs:", [torch.cuda.get_device_name(i) for i in range(n)])
PY

"${PYTHON_BIN}" scripts/prepare_training_inputs.py "${data_args[@]}" --output-dir "${OUT_DIR}"
"${PYTHON_BIN}" - "${OUT_DIR}" "${command_args[@]}" <<'PY'
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
out = Path(sys.argv[1])
revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
record = {"time_utc": datetime.now(timezone.utc).isoformat(),
          "git_commit": revision.stdout.strip(), "command": sys.argv[2:]}
with (out / "launch_history.jsonl").open("a") as f:
    f.write(json.dumps(record) + "\n")
PY
exec "${command_args[@]}"
