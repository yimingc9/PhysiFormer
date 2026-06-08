from __future__ import annotations

import math

import torch
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.

    Matches the JiT-style sinusoidal embedding + MLP.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    """
    Embeds class labels. Reserve index `num_classes` for the dropped/unconditional label.
    """

    def __init__(self, num_classes: int, hidden_size: int) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.embedding_table = nn.Embedding(self.num_classes + 1, hidden_size)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding_table(labels)


class RotaryEmbedding1D(nn.Module):
    """
    Temporal RoPE applied to Q/K using frame indices only.

    - x is expected to be (B, H, S, D) with D even.
    """

    def __init__(self, dim: int, *, max_frames: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dim must be even, got dim={dim}")
        self.dim = int(dim)
        self.max_frames = int(max_frames)

        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)  # (D/2,)

        self._cos = None
        self._sin = None
        self._cache_key = None

    def set_token_layout(
        self,
        *,
        num_frames: int,
        num_tokens_per_frame: int,
        num_register_tokens: int = 0,
        device,
        dtype: torch.dtype,
    ) -> None:
        num_frames = int(num_frames)
        num_tokens_per_frame = int(num_tokens_per_frame)
        num_register_tokens = int(num_register_tokens)
        if num_frames > self.max_frames:
            raise ValueError(f"num_frames={num_frames} exceeds max_frames={self.max_frames}")
        if num_frames <= 0 or num_tokens_per_frame <= 0 or num_register_tokens < 0:
            raise ValueError(
                f"Invalid layout: num_frames={num_frames}, num_tokens_per_frame={num_tokens_per_frame}, "
                f"num_register_tokens={num_register_tokens}"
            )

        cache_key = (num_frames, num_tokens_per_frame, num_register_tokens, str(device), str(dtype))
        if self._cache_key == cache_key and self._cos is not None and self._sin is not None:
            return

        # Compute RoPE for frame indices only: every token in a frame shares that frame index.
        half = self.dim // 2
        frame_pos = torch.arange(num_frames, device=device, dtype=torch.float32)  # (F,)
        inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)  # (D/2,)
        freqs = torch.einsum("i,j->ij", frame_pos, inv_freq)  # (F, D/2)
        freqs = freqs[:, :half]
        emb = torch.cat([freqs, freqs], dim=-1)  # (F, D)
        cos_f = emb.cos()
        sin_f = emb.sin()

        seq_len = num_frames * num_tokens_per_frame
        token_frame_for_sequence = (torch.arange(seq_len, device=device) // num_tokens_per_frame).long()
        if num_register_tokens > 0:
            token_frame = torch.cat(
                [
                    torch.zeros((num_register_tokens,), device=device, dtype=torch.long),
                    token_frame_for_sequence,
                ],
                dim=0,
            )
        else:
            token_frame = token_frame_for_sequence
        cos = cos_f.index_select(0, token_frame).to(dtype=dtype)  # (S_total, D)
        sin = sin_f.index_select(0, token_frame).to(dtype=dtype)  # (S_total, D)

        self._cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,S,D)
        self._sin = sin.unsqueeze(0).unsqueeze(0)  # (1,1,S,D)
        self._cache_key = cache_key

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected (B,H,S,D), got {tuple(x.shape)}")
        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {x.shape[-1]}")
        if self._cos is None or self._sin is None:
            raise RuntimeError("RotaryEmbedding1D layout is not set. Call set_token_layout(...) before forward().")

        seq_len = x.shape[2]
        cos = self._cos.to(device=x.device, dtype=x.dtype)
        sin = self._sin.to(device=x.device, dtype=x.dtype)
        if cos.shape[2] != seq_len:
            raise ValueError(f"Cached RoPE seq_len={cos.shape[2]} does not match x seq_len={seq_len}")

        # Broadcast (1,1,S,D) -> (B,H,S,D).
        cos = cos.expand(x.shape[0], -1, -1, -1)
        sin = sin.expand(x.shape[0], -1, -1, -1)

        return (x * cos) + (_rotate_half(x) * sin)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    # HuggingFace/LLaMA-style RoPE: treat the last dim as (d/2 real | d/2 imag).
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
