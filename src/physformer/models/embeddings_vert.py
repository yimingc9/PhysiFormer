from __future__ import annotations

import math

import torch
import torch.nn as nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    # HuggingFace/LLaMA-style RoPE: treat the last dim as (d/2 real | d/2 imag).
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class VertexRotaryEmbedding(nn.Module):
    """
    Coordinate-conditioned rotary embedding for vertex tokens.

    Derives per-token rotary phases from the 3D vertex position (x,y,z).
    """

    def __init__(self, dim: int, *, double_max_freq: bool = False) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got dim={dim}")
        half = dim // 2
        if half <= 0:
            raise ValueError(f"dim must be >= 2, got dim={dim}")

        if half == 1:
            freqs = torch.ones((1,), dtype=torch.float32)
        else:
            max_freq = math.log(half - 1, 2) if not bool(double_max_freq) else math.log(dim - 1, 2)
            freqs = 2.0 ** torch.linspace(0.0, float(max_freq), steps=half, dtype=torch.float32)
        self.register_buffer("freqs", freqs, persistent=False)  # (half,)

    def phases(self, vertex_pos: torch.Tensor, *, head_dim: int) -> torch.Tensor:
        """
        vertex_pos: (B, S, 3) float
        Returns: (B, S, head_dim//2) phases.
        """
        if vertex_pos.ndim != 3 or vertex_pos.shape[-1] != 3:
            raise ValueError(f"vertex_pos must be (B,S,3), got {tuple(vertex_pos.shape)}")
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got head_dim={head_dim}")

        bsz, seq_len, _ = vertex_pos.shape
        half = head_dim // 2
        vertex_pos = vertex_pos.to(dtype=torch.float32)
        freqs = self.freqs.to(device=vertex_pos.device, dtype=torch.float32)  # (F,)

        # (B,S,3,F) -> (B,S,3*F)
        phi = (vertex_pos.unsqueeze(-1) * freqs).reshape(bsz, seq_len, -1)

        # Truncate or pad to head_dim//2.
        if phi.shape[-1] >= half:
            return phi[:, :, :half]
        pad = torch.zeros((bsz, seq_len, half - phi.shape[-1]), device=phi.device, dtype=phi.dtype)
        return torch.cat([phi, pad], dim=-1)


class VertexCoordRoPE(nn.Module):
    """
    Applies vertex-coordinate RoPE to Q/K given per-token vertex positions.

    Usage:
      rope = VertexCoordRoPE(vert_embed, head_dim=...)
      rope.set_vertex_pos(vertex_pos_flat, dtype=q.dtype)
      q = rope(q); k = rope(k)
    """

    def __init__(self, vert_embed: VertexRotaryEmbedding, *, head_dim: int) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got head_dim={head_dim}")
        self.vert_embed = vert_embed
        self.head_dim = int(head_dim)
        self._cos = None
        self._sin = None

    def set_vertex_pos(self, vertex_pos: torch.Tensor, *, dtype: torch.dtype) -> None:
        # vertex_pos: (B, S, 3)
        phases_half = self.vert_embed.phases(vertex_pos, head_dim=self.head_dim)  # (B,S,Hd/2)
        phases = torch.cat([phases_half, phases_half], dim=-1)  # (B,S,Hd)
        cos = phases.cos().to(dtype=dtype)
        sin = phases.sin().to(dtype=dtype)
        self._cos = cos.unsqueeze(1)  # (B,1,S,Hd)
        self._sin = sin.unsqueeze(1)  # (B,1,S,Hd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected (B,H,S,D), got {tuple(x.shape)}")
        if x.shape[-1] != self.head_dim:
            raise ValueError(f"Expected head_dim={self.head_dim}, got {x.shape[-1]}")
        if self._cos is None or self._sin is None:
            raise RuntimeError("VertexCoordRoPE context is not set. Call set_vertex_pos(...) before forward().")
        cos = self._cos
        sin = self._sin
        if cos.shape[0] != x.shape[0] or cos.shape[2] != x.shape[2]:
            raise ValueError(f"RoPE cos shape {tuple(cos.shape)} must match batch/seq of x {tuple(x.shape)}")
        return (x * cos) + (_rotate_half(x) * sin)

