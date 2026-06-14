<p align="center">
  <img src="assets/PhysFormer%20Logo.png" alt="PhysFormer logo" width="20%">
</p>

<h1 align="center">PhysFormer: Learning to Simulate Mechanics<br> in World Space</h1>

<p align="center">
  Yiming Chen, Yushi Lan, Andrea Vedaldi<br>
  Visual Geometry Group, University of Oxford
</p>

<p align="center">
  <a href="https://arxiv.org/abs/YOUR_ARXIV_ID">
    <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg">
  </a>
  <a href="https://yimingc9.github.io/physformer/">
    <img src="https://img.shields.io/badge/Project-Page-4285F4.svg">
  </a>
  <a href="https://huggingface.co/spaces/yslan/PhysFormer">
    <img src="https://img.shields.io/badge/🤗-Demo-yellow.svg">
  </a>
</p>

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/4e5308a2-45ee-4a43-8337-2b51916cb3a0"
    alt="physformer teaser"
    width="640">
</p>

PhysFormer is a unified diffusion transformer that generates 4D multi-object mesh dynamics directly
in world coordinates for rigid and elastic materials. Rather than predicting future frames in
pixel space or rolling out next-step system states autoregressively, PhysFormer models motion as
full-trajectory coordinate diffusion: given initial per-vertex positions, velocities, and material
conditions, it denoises entire future vertex trajectories, with mesh topology imposed
at inference. This design enables physically plausible interactions without hard-coded constraints, 
simulator priors, or learned shape latents. Generative modelling captures uncertainty in variables not provided as input (e.g. mass, friction), 
generating diverse yet physically plausible futures from the same initial conditions unlike deterministic autoregressive methods.
Its DiT-style backbone uses factorized temporal, spatial, and object-level attention to capture coherent
structure across time, vertices, and objects. Trained on over 100k collision-rich, single-material
simulated trajectories, PhysFormer generalizes to unseen real-world geometries, larger object counts,
and mixed-material scenes.

## ⚙️ Setup

```bash
# Create conda environment
conda create -n physformer python=3.10 -y
conda activate physformer

# Install PyTorch (CUDA 12.4)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# Install remaining requirements, including the prebuilt flash-attn wheel.
pip install -r requirements.txt
```
If flash-attn is unavailable, the model falls back to PyTorch CUDA SDPA kernels,
The copied model prefers external FlashAttention via a prebuilt `flash-attn==2.8.3` wheel:

## 🤗 Model Access

Download checkpoint from HuggingFace before running scripts:

```bash
huggingface-cli download yslan/physformer checkpoint-best.pt --local-dir checkpoints
```

## 🌟 Minimal Inference

The example scripts write predicted rollout samples into each input sample directory as
`sample_00/`, `sample_01/`, etc. Existing outputs are kept unless
`OVERWRITE_FLAG=--overwrite` is set.

Run in-distribution test set inference:

```bash
bash scripts/run_indistri_example.sh
```

This runs `indistri_examples/rigid` as all rigid materials and `indistri_examples/elastic` as all elastic materials.
It writes inference-only renders as `inference.mp4`. Ground truth for each example rendered with blender are present. 

Run OOD inference for generalization to complex geometries:

```bash
bash scripts/run_ood_example.sh
```

Common usage controls:

```bash
# Fast smoke test: one rollout sample, one input sample, fewer denoising steps.
GENERATIONS=1 MAX_SAMPLES=1 SAMPLING_STEPS=25 bash scripts/run_ood_example.sh

# Change OOD object material conditioning.
# By default, 2obj: horse elastic, cow rigid; 3obj: fish and bunny elastic, teapot rigid
OOD_ELASTIC="horse bunny" OOD_RIGID="cow teapot fish" bash scripts/run_ood_example.sh
```

Useful direct launcher flags:

- `--generations`: number of rollout samples per input sample.
- `--max-samples`: limit how many input samples are run; `0` means all.
- `--num-sampling-steps`: override denoising steps; useful for quick tests.
- `--elastic OBJECT(S)`, `--rigid OBJECT(S)`: set OOD object materials by object-name substring.
- `--overwrite` / `--no-overwrite`: replace or preserve existing outputs.
- `--dry-run`: print selected samples and command without running the model.

## ✏️ Evaluation
```bash
python eval_publish_losses.py \
  --ckpt checkpoints/checkpoint-best.pt \
  --out_json reports/publication_losses.json \
  --out_tsv reports/publication_losses.tsv \
  --num_generations 3
```

Useful eval flags:

- `--num_generations`: number of rollout samples evaluated per input sample.
- `--limit`: evaluate only the first N split entries.
- `--num_sampling_steps`: sampling step number, default 50.

Reported Evaluation Metrics:
- `mse`: MSE on vertex position trajectory.
- `rigidity`: per-object Kabsch rigid alignment residual across all frames against frame 0. 
- `conservation_of_momentum`: system momentum drift compared against GT motion. 
