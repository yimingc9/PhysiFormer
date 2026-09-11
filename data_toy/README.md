# Toy training data

100 trajectories: elastic scenes with 1–5 objects and rigid scenes with 6–10
objects, with 10 samples per object count. Each sample contains `sample.npz`
and `trajectory.mp4`.

[split.json](split.json) assigns **`sample_0` from each group to validation** and
`sample_1`–`sample_9` to training: **90 train / 10 validation**. Each category also
has its own 45/5 split. `eval` aliases `val`; `test` is empty.

Create or check the split files from the repository root:

```bash
python scripts/prepare_toy_split.py
python scripts/prepare_toy_split.py --check
```

## Training

Activate the [repository environment](../README.md), then run from the repository root:

```bash
bash scripts/train_toy_physiformer.sh             # Default: two H100 GPUs
bash scripts/train_toy_physiformer.sh --gpus 1    # Override GPU count
bash scripts/train_toy_physiformer.sh --dry-run   # Check paths and print command
```

Default training recipe:

| Setting | Default |
| --- | --- |
| Model | PhysiFormer-L (`MODEL=PhysiFormer`); B: `MODEL=PhysiFormer-B` |
| Batch × GPUs × accumulation | 8 × 2 × 4 = 64 |
| Epochs / virtual samples | 6,000 / 10,000 per epoch (156 optimizer updates) |
| Optimizer / precision | AdamW, weight decay 0, gradient clipping 1; BF16, gradient checkpointing |
| Learning rate | Epoch-based cosine, `4e-5` → `5e-6`; 5 warmup epochs, starting at LR zero |
| Validation | EMA weights (decay `0.9999`), every 10 epochs and at the final epoch |

Virtual epochs repeat the training samples. Incomplete minibatches and accumulation
groups are discarded; distributed validation may repeat entries to divide evenly
across GPUs. Normalization uses training data only. Accumulation automatically
adjusts to maintain effective batch 64; override with `GRAD_ACCUM`.

A **dry run** checks split paths and prints the launch command without training,
checking GPU memory, or writing outputs. For a short test that actually trains:

Training command: 

```bash
EPOCHS=1 WARMUP_EPOCHS=0 TRAIN_VIRTUAL_LENGTH=0 GRAD_ACCUM=1 \
BATCH_SIZE=2 EVAL_BATCH_SIZE=2 NUM_WORKERS=0 RESUME=none OUT_DIR=runs/toy_smoke \
  bash scripts/train_toy_physiformer.sh --gpus 1
```

Use `--help` for overrides, including `PYTHON_BIN`, `BATCH_SIZE`, and `EPOCHS`.
Batch size 8 has not been profiled on H100s; reduce training and validation batch
sizes if needed.

## Use your own data

```bash
DATA_ROOT=/path/to/npzs SPLIT_FILE=/path/to/split.json \
NUM_FRAMES=49 NUM_VERTICES=0 MAX_NUM_OBJECTS=5 COND_OBJECT_MATERIAL=0 \
OUT_DIR=runs/my_dataset bash scripts/train_toy_physiformer.sh
```

The split contains nonempty, disjoint NPZ paths relative to `DATA_ROOT`:

```json
{"train": ["a.npz", "b.npz"], "val": ["c.npz"]}
```

Each NPZ needs `vertices` `(F,V,3)`, `mask` `(F,V)`, `object_ids` `(V,)`, and
`first_frame_velocity` `(V,3)`; scalar `num_objects` is recommended. Use zero-based
object IDs, matching frame counts, and consistent padded vertex capacity.
`NUM_VERTICES=0` infers capacity from the first training NPZ.

Toy defaults: 49 frames, 356 vertices, up to 10 objects, and material conditioning
(rigid `[0.0]`, elastic `[1.0]`). For mixed categories, copy the `category_roots`,
`material_alias_values`, and `alias::path.npz` structure in [split.json](split.json).
Set `COND_OBJECT_MATERIAL=0` to disable material features. For fixed normalization,
append `-- --norm_mean X Y Z --norm_std X Y Z` to the command.

## Checkpoints and Slurm

```bash
# Load compatible weights with a fresh optimizer and schedule.
RESUME=none OUT_DIR=runs/toy_finetune \
  bash scripts/train_toy_physiformer.sh -- --init_ckpt /path/to/checkpoint.pt

# Resume model, optimizer, EMA, epoch, and step.
RESUME=/path/to/checkpoint-last.pt bash scripts/train_toy_physiformer.sh
```