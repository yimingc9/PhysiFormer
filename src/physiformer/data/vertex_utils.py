from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def fix_num_vertices(
    vertices: np.ndarray,
    *,
    num_vertices: int,
    vertex_sampling: str,
    pad_value: float,
    sample_idx: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Pad or sample a vertex array to a fixed token count.

    Returns:
        vertices_fixed: Array with shape ``(num_vertices, 3)``.
        mask: Float mask with shape ``(num_vertices,)``; real vertices are 1 and padding is 0.
        sample_idx: Reusable sampled indices when the input was downsampled.
    """
    target_vertices = int(num_vertices)
    current_vertices = int(vertices.shape[0])
    if current_vertices == target_vertices:
        return vertices, np.ones((target_vertices,), dtype=np.float32), sample_idx

    if current_vertices > target_vertices:
        if sample_idx is not None and int(sample_idx.max()) >= current_vertices:
            sample_idx = None
        if sample_idx is None:
            if vertex_sampling == "random":
                sample_idx = np.random.choice(current_vertices, target_vertices, replace=False)
            elif vertex_sampling == "first":
                sample_idx = np.arange(target_vertices)
            else:
                raise ValueError(f"Unknown vertex_sampling: {vertex_sampling}")
        return vertices[sample_idx], np.ones((target_vertices,), dtype=np.float32), sample_idx

    pad = np.full((target_vertices - current_vertices, 3), pad_value, dtype=vertices.dtype)
    vertices_fixed = np.concatenate([vertices, pad], axis=0)
    mask = np.zeros((target_vertices,), dtype=np.float32)
    mask[:current_vertices] = 1.0
    return vertices_fixed, mask, sample_idx
