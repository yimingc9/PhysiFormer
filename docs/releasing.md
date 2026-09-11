# Releasing model weights

Keep full `.pt` checkpoints for training resumption. Publish one evaluated weight
set as SafeTensors, together with its configuration, model card, and weight license.
Use EMA if that is what was evaluated; exporting preserves tensor values and dtype.

From the repository root, in the installed environment:

```bash
python scripts/export_safetensors.py \
  --checkpoint /path/to/checkpoint-best.pt \
  --output-dir /path/to/release \
  --weights ema
```

The output directory must be new. The exporter writes `model.safetensors` and
`config.json`, reloads them, and verifies every tensor byte-for-byte. It exports
only inference arguments, excluding paths, tracking metadata, and optimizer state.
The config records the weight source and format version; a digest in the weights
detects a missing or mismatched configuration. Use `--weights model` for raw weights.

Verify complete parameter loading and a short deterministic CPU rollout with the
full model (two frames, four vertices, two sampling steps):

```bash
python scripts/verify_safetensors.py \
  --checkpoint /path/to/checkpoint-best.pt \
  --release /path/to/release/model.safetensors \
  --report reports/release_verification.json
```

This check uses the included two-object elastic example and supports models with
object-material conditioning and no scene tokens. It complements a full evaluation;
it does not establish dataset-wide metrics or GPU numerical equivalence.

## Load and train

```bash
# Inference: the same CHECKPOINT override also works with .pt.
CHECKPOINT=/path/to/release/model.safetensors bash scripts/run_indistri_example.sh

# Fine-tune: fresh optimizer/schedule; match MODEL and conditioning to the release.
RESUME=none OUT_DIR=runs/finetune \
  bash scripts/train_toy_physiformer.sh -- --init_ckpt /path/to/release/model.safetensors

# Resume your own training with its full .pt checkpoint.
RESUME=runs/finetune/checkpoint-last.pt bash scripts/train_toy_physiformer.sh
```

SafeTensors holds only the exported weight set; EMA-selection flags cannot switch
it to a discarded set. For training initialization, the trainer retains its selected
training configuration and normalization. Supply `--norm_mean` and `--norm_std`
after `--` when you want to reuse the release's normalization.

The inference/evaluation loaders read the sibling config automatically. Their
normal inference overrides remain available. The demo downloader defaults to
`model.safetensors` and downloads `config.json` with it from `yslan/physiformer`.
Use `PHYSIFORMER_CKPT_REPO_ID` and `PHYSIFORMER_CKPT_FILENAME` for another release,
including full `.pt` checkpoints.

## Hugging Face package

Upload exactly these files to a model repository:

```text
model.safetensors
config.json
README.md
LICENSE
```

The model card should identify the evaluated weight set, input format, supported
conditioning, usage, limitations, weight license, and the GitHub code repository.
Choose the weight license before publishing. Keep training checkpoints, reports,
private backups, local data mappings, and caches outside the uploaded file list.
The GitHub repository contains the implementation, dependencies, and toy examples.
