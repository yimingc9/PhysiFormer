from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiffusionConfig:
    # JiT-style continuous-time sampling for t.
    P_mean: float = -0.8
    P_std: float = 0.8
    t_eps: float = 5e-2
    noise_scale: float = 1.0

    # Sampling.
    sampling_method: str = "heun"
    num_sampling_steps: int = 50
