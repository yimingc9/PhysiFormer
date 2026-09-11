


<h1 align="center">🫧PhysiFormer: Learning to Simulate Mechanics<br> in World Space</h1>

<p align="center">
  Yiming Chen, Yushi Lan, Andrea Vedaldi<br>
  Visual Geometry Group, University of Oxford
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2606.27364">
    <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg">
  </a>
  <a href="https://yimingc9.github.io/physiformer/">
    <img src="https://img.shields.io/badge/Project-Page-4285F4.svg">
  </a>
  <a href="https://huggingface.co/spaces/yslan/PhysiFormer">
    <img src="https://img.shields.io/badge/🤗-Demo-yellow.svg">
  </a>
</p>


<p align="center">
  <img
    src="https://github.com/user-attachments/assets/9da40717-32df-4a9e-87cc-aea28fdfbbfa"
    alt="physiformer teaser"
    width="640">
</p>
We present PhysiFormer(pronouced 🫧fizzy🫧former), a diffusion transformer for physically-plausible 3D object motion. Unlike video world models that operate in view-dependent pixel space, PhysiFormer represents objects as 3D meshes expressed in world coordinates. Given the initial vertex positions and velocities, as well as object material type, rigid or elastic, the model samples future vertex trajectories. While related neural physics approaches build on ad-hoc latent spaces or explicitly enforce rigidity and causality, PhysiFormer shows that excellent results can be obtained without any such inductive biases, by casting vertex trajectory prediction as a single denoising diffusion process directly in world coordinates. The probabilistic formulation captures uncertainty in the learned dynamics, enabling diverse plausible futures from initial conditions, making this framework potentially useful for applications with unobserved uncertainty. The model features attention factorized over time, space, and objects for efficiency, enabling permutation-invariant multi-object reasoning without needing explicit object encoding. Trained on over 100k simulated trajectories, PhysiFormer generates rigid and elastic mechanics, and generalises to mixed-material settings, unseen real-world geometries, and larger object counts. It substantially outperforms autoregressive baselines in trajectory accuracy, rigidity preservation, and momentum-based physical consistency. Our results position coordinate-space diffusion as a promising step toward view-invariant, geometry-aware world modelling for robotics, graphics, and physical design.

## 📦 Coming Soon
The dataset will be released soon. Please stay tuned! 

## ⚙️ Setup

```bash
# Create conda environment
conda create -n physiformer python=3.10 -y
conda activate physiformer

# Install PyTorch (CUDA 12.4)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# Install remaining requirements 
pip install -r requirements.txt
```
Model defauts to PyTorch CUDA SDPA kernels. If unavailable, model uses math/chunked attention fallback.

## 🤗 Model Checkpoint

Download pretrained model weights: 

```bash
hf download yslan/physiformer model.safetensors config.json --local-dir checkpoints
```

## 🧸 Toy training example

The toy dataset contains 90 training and 10 validation trajectories. 
Training configuration and dataset schema is detailed [here](data_toy/README.md). 

```bash
# Train the toy example with the full training recipe.
bash scripts/train_toy_physiformer.sh

# Adapt the same launcher to another dataset with a train/val split.
DATA_ROOT=/path/to/npzs SPLIT_FILE=/path/to/split.json \
NUM_VERTICES=0 MAX_NUM_OBJECTS=5 COND_OBJECT_MATERIAL=0 OUT_DIR=runs/my_dataset \
  bash scripts/train_toy_physiformer.sh
```

To finetune from published model weights
```bash
# Initialize weights for fine-tuning with a fresh optimizer and schedule.
RESUME=none OUT_DIR=runs/toy_finetune \
  bash scripts/train_toy_physiformer.sh -- --init_ckpt checkpoints/model.safetensors
```

## 🌟 Minimal Inference

The example scripts write predicted rollout samples into each input sample directory as
`sample_00/`, `sample_01/`, etc. Existing outputs are kept unless
`OVERWRITE_FLAG=--overwrite` is set.

Run in-distribution inference with the released SafeTensors weights:

```bash
# in-distribution inference
CHECKPOINT=checkpoints/model.safetensors bash scripts/run_indistri_example.sh

# OOD distribution inference
CHECKPOINT=checkpoints/model.safetensors bash scripts/run_ood_example.sh
```

Or use a `.pt` checkpoint from your own training:

```bash
CHECKPOINT=/path/to/checkpoint-best.pt bash scripts/run_indistri_example.sh

CHECKPOINT=/path/to/checkpoint-best.pt bash scripts/run_ood_example.sh
```

Common usage controls:

```bash
# Fast smoke test: one rollout sample, one input sample, fewer denoising steps.
CHECKPOINT=checkpoints/model.safetensors GENERATIONS=1 MAX_SAMPLES=1 SAMPLING_STEPS=25 \
  bash scripts/run_ood_example.sh

# Change OOD object material conditioning.
# By default, 2obj: horse elastic, cow rigid; 3obj: fish and bunny elastic, teapot rigid
CHECKPOINT=checkpoints/model.safetensors OOD_ELASTIC="horse bunny" OOD_RIGID="cow teapot fish" \
  bash scripts/run_ood_example.sh
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
  --ckpt checkpoints/model.safetensors \
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
- `momentum_drift_ratio`: ratio of predicted mean system-momentum drift to GT mean system-momentum drift; values closer to 1 are better.
