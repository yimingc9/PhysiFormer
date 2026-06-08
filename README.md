<p align="center">
  <img src="assets/PhysFormer%20Logo.png" alt="PhysFormer logo" width="25%">
</p>

<h1 align="center">PhysFormer: Learning to Simulate Mechanics in World Space</h1>

<p align="center">
  Yiming Chen, Yushi Lan, Andrea Vedaldi<br>
  Visual Geometry Group, University of Oxford
</p>

<p align="center">
  <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b" alt="arXiv">
  <img src="https://img.shields.io/badge/Project-Page-orange" alt="Project Page">
  <img src="https://img.shields.io/badge/HuggingFace-Model-green?logo=huggingface" alt="Models">
</p>

<p align="center">
  <video src="assets/physformer_teaser.mp4" width="75%" autoplay loop muted playsinline controls>
    Your browser does not support the video tag.
  </video>
</p>

PhysFormer is a unified diffusion transformer that generates 4D multi-object mesh dynamics directly
in world coordinates for both rigid and elastic materials. Rather than predicting future frames in
pixel space or rolling out next-step system states autoregressively, PhysFormer models motion as
full-trajectory coordinate diffusion: given initial per-vertex positions, velocities, and material
conditions, it denoises entire future vertex trajectories in one process, with mesh topology imposed
at inference. This design enables physically plausible object-object and object-environment
interactions without hard-coded constraints, simulator priors, or learned shape latents. Its
DiT-style backbone uses factorized temporal, spatial, and object-level attention to capture coherent
structure across time, vertices, and objects. Trained on over 100k collision-rich, single-material
simulated trajectories, PhysFormer generalizes to unseen real-world geometries, larger object counts,
and mixed-material scenes.

## Installation

```bash
# Create conda environment
conda create -n physformer python=3.10 -y
conda activate physformer

# Install PyTorch (CUDA 12.4)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# Install remaining requirements
pip install -r requirements.txt
```

The copied model uses PyTorch scaled-dot-product attention and does not import the external
`flash-attn` package.

## Checkpoint

The demo uses:

```text
checkpoints/checkpoint-best.pt
```

Command-line inference expects this file to exist locally. Download it before running scripts:

```bash
huggingface-cli download yslan/physformer checkpoint-best.pt --local-dir checkpoints
```

If the model repo is private or gated, authenticate first or set `HF_TOKEN`. You can also pass a
different checkpoint path:

```bash
CHECKPOINT=/path/to/checkpoint.pt bash scripts/run_ood_example.sh
```

The Gradio app in `app.py` downloads the checkpoint automatically if missing. The downloaded file is
stored at `checkpoints/checkpoint-best.pt`.

Override the Hugging Face source with:

```bash
export PHYSFORMER_CKPT_REPO_ID=yslan/physformer
export PHYSFORMER_CKPT_FILENAME=checkpoint-best.pt
```

For private or gated model repos, set `HF_TOKEN`.

Packaged demo data is local to this repository:

```text
ood_examples/         OOD inference inputs
indistri_examples/    in-distribution visualization inputs
eval_assets/          compact inputs used directly by eval_publish_losses.py
```

Run commands below from this directory:

```bash
cd official_demo/code
```

## Inference

The example scripts write predicted rollout samples into each input sample directory as
`sample_00/`, `sample_01/`, etc. Existing outputs are kept unless
`OVERWRITE_FLAG=--overwrite` is set.

### Hugging Face ZeroGPU Demo

This repository includes a minimal Gradio Space app in `app.py`. It uses the same checkpoint
location/download logic described above, runs one small inference rollout, and returns the rendered
`inference.mp4`. Evaluation precomputation is not used by this demo.

Create a Gradio Space and select **ZeroGPU** in the Space hardware settings. Use this README
configuration block if this repository is pushed directly as a Space:

```yaml
sdk: gradio
app_file: app.py
python_version: 3.10.13
```

If the checkpoint repo is private or gated, add `HF_TOKEN` as a Space secret. The app also accepts:

- `PHYSFORMER_CKPT_REPO_ID`: checkpoint repo, default `yslan/physformer`.
- `PHYSFORMER_CKPT_FILENAME`: checkpoint file in that repo, default `checkpoint-best.pt`.
- `PHYSFORMER_AMP`: inference precision, default `bf16`.

Run OOD inference:

```bash
bash scripts/run_ood_example.sh
```

This uses the OOD folders:

```text
ood_examples/2obj_cow_horse
ood_examples/3obj_teapot_fish_bunny
```

Default OOD materials are:

```text
elastic: horse fish bunny
rigid:   cow teapot
```

Run in-distribution example inference:

```bash
bash scripts/run_indistri_example.sh
```

This runs `indistri_examples/rigid` as rigid and `indistri_examples/soft` as elastic.
It writes inference-only renders as `inference.mp4` and ground-truth-only renders as `GT.mp4`.
OOD example runs write only inference renders.

Common user controls:

```bash
# Fast smoke test: one rollout sample, one input sample, fewer denoising steps.
GENERATIONS=1 MAX_SAMPLES=1 SAMPLING_STEPS=5 bash scripts/run_ood_example.sh

# Dry-run script wiring without loading the model or touching CUDA.
DRY_RUN_FLAG=--dry-run GENERATIONS=1 bash scripts/run_ood_example.sh

# Change OOD object material assignment.
OOD_ELASTIC="horse bunny" OOD_RIGID="cow teapot fish" bash scripts/run_ood_example.sh

# Skip MP4 rendering.
RENDER_FLAG="" bash scripts/run_indistri_example.sh

# Replace existing sample_* outputs.
OVERWRITE_FLAG=--overwrite bash scripts/run_indistri_example.sh

# Use a specific Python executable or checkpoint path.
PYTHON=/path/to/python CHECKPOINT=/path/to/checkpoint.pt bash scripts/run_ood_example.sh
```

`GENERATIONS` is the number of independent rollout samples generated per input sample. It maps to
the launcher argument `--generations`.

Direct launcher form, if you do not want to use the scripts:

```bash
python run_official_demo_inference.py \
  --include ood \
  --generations 3 \
  --elastic horse --elastic fish --elastic bunny \
  --rigid cow --rigid teapot \
  --save-mp4
```

Useful direct launcher flags:

- `--generations`: number of rollout samples per input sample.
- `--max-samples`: limit how many input samples are run; `0` means all.
- `--num-sampling-steps`: override denoising steps; useful for quick tests.
- `--elastic PATTERN`, `--rigid PATTERN`: set OOD object materials by object-name substring.
- `--save-mp4`: render prediction MP4s as `inference.mp4`; omit it to save only `vertices.npz`.
- `--save-gt-mp4`: render ground-truth MP4s as `GT.mp4`.
- `--save-scene-metadata`: optionally write debug/provenance `scene_multiobj.json` files.
- `--overwrite` / `--no-overwrite`: replace or preserve existing outputs.
- `--dry-run`: print selected samples and command without running the model.
- `--attention-debug`: report the first PyTorch attention backend used.

## Evaluation

The repository packages compact evaluation inputs in `eval_assets/`. The evaluator checks for
`eval_split.json`, `eval_precomp/`, and `eval_data/` before loading the model. If they are already
present, it prints that preparation is skipped. Run the evaluator to generate rollout samples and
compute losses:

```bash
python eval_publish_losses.py \
  --ckpt checkpoints/checkpoint-best.pt \
  --out_json reports/publication_losses.json \
  --out_tsv reports/publication_losses.tsv \
  -k 4
```

The evaluator reports:

- `mse`: masked MSE on raw vertex positions.
- `rigidity`: per-object Kabsch residual over frames `1..--rigidity_last_frame`.
- `conservation_of_momentum`: normalized system momentum drift over frames
  `1..--momentum_last_frame`.

Common evaluation controls:

```bash
# Fast evaluation smoke test.
python eval_publish_losses.py \
  --ckpt checkpoints/checkpoint-best.pt \
  --out_json reports/smoke_losses.json \
  --out_tsv reports/smoke_losses.tsv \
  -k 1 \
  --limit 1 \
  --num_sampling_steps 5
```

Useful eval flags:

- `-k`, `--num_generations`: number of rollout samples evaluated per input sample.
- `--limit`: evaluate only the first N split entries.
- `--num_sampling_steps`, `--cfg_scale`: sampling controls.
- `--split_file`, `--split_name`: choose the evaluation split.
- `--precomp_root`, `--data_root`: choose prepared evaluation assets.
- `--sample_root`: raw ground-truth sample root used only if prepared eval assets are missing.
- `--prepare_sample_names`: comma-separated raw sample folders to prepare from `--sample_root`.
- `--prepare_overwrite`: regenerate prepared eval assets even if they already exist.
- `--device`, `--amp`: runtime device and precision.
- `--out_json`, `--out_tsv`: report paths.

`prepare_publish_eval_inputs.py` is the standalone version of the same preparation step. It is only
needed to regenerate `eval_assets/` from raw ground-truth sample folders. Those raw samples are not
packaged by default. If you have them separately, either pass `--sample_root` to
`eval_publish_losses.py` or run:

```bash
python prepare_publish_eval_inputs.py --sample_root /path/to/raw_eval_samples
```

## Runtime Dependencies

Use the project inference environment with PyTorch installed. The copied code expects:

- Python 3.10+
- PyTorch 2.5.1 installed from the CUDA 12.4 wheel index for practical runtime
- NumPy
- tqdm
- matplotlib
- imageio and imageio-ffmpeg only if saving GIF/MP4 renders

`requirements.txt` includes PyTorch so Hugging Face Spaces can build directly from this repository.
For local installs, you may still install PyTorch explicitly first so pip chooses the intended CUDA
wheel. The copied model uses PyTorch scaled-dot-product attention and requires a CUDA fast-attention
backend; it does not directly import the external `flash-attn` package.

The packaged `src/official_demo_inference/configs/vertex_counts_multiobj_all.json` replaces the
checkpoint's original training-machine absolute vertex-count path. A legacy copy is also kept under
`src/mesh_primitives/` so direct calls into the copied PhysFormer script still have a local fallback.
