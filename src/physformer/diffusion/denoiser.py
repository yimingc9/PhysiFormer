from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiffusionConfig:
    # JiT-style continuous-time sampling for t.
    P_mean: float = -0.8
    P_std: float = 0.8
    t_eps: float = 5e-2
    noise_scale: float = 1.0

    # Classifier-free guidance.
    label_drop_prob: float = 0.1
    cfg_scale: float = 1.0
    cfg_interval_min: float = 0.0
    cfg_interval_max: float = 1.0

    # First-frame velocity-only guidance.
    vel_cfg_scale: float = 1.0
    vel_cfg_interval_min: float = 0.0
    vel_cfg_interval_max: float = 1.0

    # Sampling.
    sampling_method: str = "heun"
    num_sampling_steps: int = 50
