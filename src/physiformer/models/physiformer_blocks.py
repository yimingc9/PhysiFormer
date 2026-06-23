from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn

from .blocks import Attention, RMSNorm, SwiGLUFFN, modulate

AttnAxis = Literal["spatial", "temporal", "object"]


def _reduce_replicated_register_tokens(reg_rep: torch.Tensor, group_keep: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Reduces replicated register tokens produced by factorized attention.

    reg_rep: (B, G, N_reg, D) where G is the number of factorized groups.
    group_keep: Optional (B, G) bool/float indicating which groups contain any valid tokens.
    Returns: (B, N_reg, D)
    """
    if reg_rep.ndim != 4:
        raise ValueError(f"reg_rep must be (B,G,N_reg,D), got {tuple(reg_rep.shape)}")

    if group_keep is None:
        return reg_rep.mean(dim=1)

    if group_keep.shape != reg_rep.shape[:2]:
        raise ValueError(f"group_keep must be (B,G)={tuple(reg_rep.shape[:2])}, got {tuple(group_keep.shape)}")

    w = group_keep.to(device=reg_rep.device, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
    denom = w.sum(dim=1).clamp_min(1.0)
    out = (reg_rep.to(dtype=torch.float32) * w).sum(dim=1) / denom
    return out.to(dtype=reg_rep.dtype)


class PhysiFormerBlock(nn.Module):
    """
    PhysiFormer transformer block with AdaLN and three attention axes:

      - Spatial: full attention across all vertices within each frame.
      - Temporal: full attention across frames for each vertex.
      - Object: attention within each object's vertices inside each frame, then scatter back.

    Register tokens follow the divided-attention convention for all factorized axes:
      - Replicate across groups before attention.
      - Reduce back via (weighted) mean after attention.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        attn_axis: AttnAxis,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if attn_axis not in ("spatial", "temporal", "object"):
            raise ValueError(f"attn_axis must be 'spatial', 'temporal', or 'object', got {attn_axis}")
        self.attn_axis: AttnAxis = attn_axis

        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            qkv_bias=True,
            qk_norm=True,
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def _spatial_attn(
        self,
        x: torch.Tensor,
        *,
        num_frames: int,
        num_vertices: int,
        num_register_tokens: int,
        rope=None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        group_keep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, _, dim = x.shape
        tokens = x[:, num_register_tokens:, :].reshape(bsz, num_frames, num_vertices, dim)
        tok_sp = tokens.reshape(bsz * num_frames, num_vertices, dim)

        if num_register_tokens > 0:
            reg = x[:, :num_register_tokens, :]
            reg_rep = reg[:, None, :, :].expand(bsz, num_frames, num_register_tokens, dim).reshape(
                bsz * num_frames, num_register_tokens, dim
            )
            x_sp = torch.cat([reg_rep, tok_sp], dim=1)
        else:
            x_sp = tok_sp

        y_sp = self.attn(x_sp, rope=rope, src_key_padding_mask=src_key_padding_mask)

        if num_register_tokens > 0:
            reg_rep_out = y_sp[:, :num_register_tokens, :].reshape(bsz, num_frames, num_register_tokens, dim)
            reg_out = _reduce_replicated_register_tokens(reg_rep_out, group_keep)
            tok_out = y_sp[:, num_register_tokens:, :].reshape(bsz, num_frames, num_vertices, dim)
            tok_out = tok_out.reshape(bsz, num_frames * num_vertices, dim)
            return torch.cat([reg_out, tok_out], dim=1)

        return y_sp.reshape(bsz, num_frames * num_vertices, dim)

    def _temporal_attn(
        self,
        x: torch.Tensor,
        *,
        num_frames: int,
        num_vertices: int,
        num_register_tokens: int,
        rope=None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        group_keep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, _, dim = x.shape
        tokens = x[:, num_register_tokens:, :].reshape(bsz, num_frames, num_vertices, dim)
        tok_t = tokens.permute(0, 2, 1, 3).reshape(bsz * num_vertices, num_frames, dim)

        if num_register_tokens > 0:
            reg = x[:, :num_register_tokens, :]
            reg_rep = reg[:, None, :, :].expand(bsz, num_vertices, num_register_tokens, dim).reshape(
                bsz * num_vertices, num_register_tokens, dim
            )
            x_t = torch.cat([reg_rep, tok_t], dim=1)
        else:
            x_t = tok_t

        y_t = self.attn(x_t, rope=rope, src_key_padding_mask=src_key_padding_mask)

        if num_register_tokens > 0:
            reg_rep_out = y_t[:, :num_register_tokens, :].reshape(bsz, num_vertices, num_register_tokens, dim)
            reg_out = _reduce_replicated_register_tokens(reg_rep_out, group_keep)
            tok_out = y_t[:, num_register_tokens:, :].reshape(bsz, num_vertices, num_frames, dim)
            tok_out = tok_out.permute(0, 2, 1, 3).reshape(bsz, num_frames * num_vertices, dim)
            return torch.cat([reg_out, tok_out], dim=1)

        tok_out = y_t.reshape(bsz, num_vertices, num_frames, dim).permute(0, 2, 1, 3).reshape(
            bsz, num_frames * num_vertices, dim
        )
        return tok_out

    def _object_attn(
        self,
        x: torch.Tensor,
        *,
        num_frames: int,
        num_vertices: int,
        num_register_tokens: int,
        num_objects: int,
        object_nmax: int,
        object_vertex_index: torch.Tensor,
        object_token_keep: torch.Tensor,
        rope=None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        group_keep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if num_objects <= 0:
            raise ValueError(f"num_objects must be > 0, got {num_objects}")
        if object_nmax <= 0:
            raise ValueError(f"object_nmax must be > 0, got {object_nmax}")

        bsz, _, dim = x.shape
        bf = bsz * num_frames

        if object_vertex_index.shape != (bf, num_objects * object_nmax):
            raise ValueError(
                "object_vertex_index must be "
                f"(B*F,O*Nmax)={(bf, num_objects * object_nmax)}, got {tuple(object_vertex_index.shape)}"
            )
        if object_token_keep.shape != (bf, num_objects, object_nmax):
            raise ValueError(
                "object_token_keep must be "
                f"(B*F,O,Nmax)={(bf, num_objects, object_nmax)}, got {tuple(object_token_keep.shape)}"
            )

        tokens = x[:, num_register_tokens:, :].reshape(bsz, num_frames, num_vertices, dim)
        tok_flat = tokens.reshape(bf, num_vertices, dim)

        gather_idx = object_vertex_index.unsqueeze(-1).expand(-1, -1, dim)
        tok_obj = tok_flat.gather(1, gather_idx).reshape(bf, num_objects, object_nmax, dim)
        tok_obj = tok_obj.reshape(bf * num_objects, object_nmax, dim)

        if num_register_tokens > 0:
            reg = x[:, :num_register_tokens, :]
            reg_rep = reg[:, None, None, :, :].expand(bsz, num_frames, num_objects, num_register_tokens, dim).reshape(
                bf * num_objects, num_register_tokens, dim
            )
            x_obj = torch.cat([reg_rep, tok_obj], dim=1)
        else:
            x_obj = tok_obj

        y_obj = self.attn(x_obj, rope=rope, src_key_padding_mask=src_key_padding_mask)
        if y_obj.dtype != x.dtype:
            y_obj = y_obj.to(dtype=x.dtype)

        if num_register_tokens > 0:
            reg_rep_out = y_obj[:, :num_register_tokens, :].reshape(bsz, num_frames * num_objects, num_register_tokens, dim)
            reg_out = _reduce_replicated_register_tokens(reg_rep_out, group_keep)
            tok_out = y_obj[:, num_register_tokens:, :].reshape(bf, num_objects, object_nmax, dim)
        else:
            reg_out = None
            tok_out = y_obj.reshape(bf, num_objects, object_nmax, dim)

        tok_out = tok_out * object_token_keep.unsqueeze(-1).to(dtype=tok_out.dtype)
        tok_src = tok_out.reshape(bf, num_objects * object_nmax, dim).to(dtype=x.dtype)
        tok_full = x.new_zeros((bf, num_vertices, dim), dtype=x.dtype)
        tok_full.scatter_add_(
            1,
            object_vertex_index.unsqueeze(-1).expand(-1, -1, dim),
            tok_src,
        )
        tok_full = tok_full.reshape(bsz, num_frames * num_vertices, dim)

        if reg_out is not None:
            return torch.cat([reg_out, tok_full], dim=1)
        return tok_full

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        *,
        num_frames: int,
        num_vertices: int,
        rope_spatial=None,
        rope_temporal=None,
        rope_object=None,
        src_key_padding_mask_spatial: Optional[torch.Tensor] = None,
        src_key_padding_mask_temporal: Optional[torch.Tensor] = None,
        src_key_padding_mask_object: Optional[torch.Tensor] = None,
        spatial_group_keep: Optional[torch.Tensor] = None,
        temporal_group_keep: Optional[torch.Tensor] = None,
        object_group_keep: Optional[torch.Tensor] = None,
        object_vertex_index: Optional[torch.Tensor] = None,
        object_token_keep: Optional[torch.Tensor] = None,
        num_objects: int = 0,
        object_nmax: int = 0,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must be (B,S,D), got {tuple(x.shape)}")
        if c.ndim != 2:
            raise ValueError(f"c must be (B,D), got {tuple(c.shape)}")
        if x.shape[0] != c.shape[0]:
            raise ValueError(f"Batch mismatch: x has B={x.shape[0]}, c has B={c.shape[0]}")

        bsz, seq_len, _ = x.shape
        num_frames = int(num_frames)
        num_vertices = int(num_vertices)
        if num_frames <= 0 or num_vertices <= 0:
            raise ValueError(f"num_frames and num_vertices must be >0, got {num_frames}, {num_vertices}")

        num_token = num_frames * num_vertices
        if seq_len < num_token:
            raise ValueError(f"seq_len={seq_len} must be >= num_frames*num_vertices={num_token}")
        num_register_tokens = seq_len - num_token

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)

        x_attn_in = modulate(self.norm1(x), shift_msa, scale_msa)
        if self.attn_axis == "spatial":
            attn_out = self._spatial_attn(
                x_attn_in,
                num_frames=num_frames,
                num_vertices=num_vertices,
                num_register_tokens=num_register_tokens,
                rope=rope_spatial,
                src_key_padding_mask=src_key_padding_mask_spatial,
                group_keep=spatial_group_keep,
            )
        elif self.attn_axis == "temporal":
            attn_out = self._temporal_attn(
                x_attn_in,
                num_frames=num_frames,
                num_vertices=num_vertices,
                num_register_tokens=num_register_tokens,
                rope=rope_temporal,
                src_key_padding_mask=src_key_padding_mask_temporal,
                group_keep=temporal_group_keep,
            )
        else:
            if object_vertex_index is None or object_token_keep is None:
                raise ValueError("object attention requires object_vertex_index and object_token_keep")
            attn_out = self._object_attn(
                x_attn_in,
                num_frames=num_frames,
                num_vertices=num_vertices,
                num_register_tokens=num_register_tokens,
                num_objects=int(num_objects),
                object_nmax=int(object_nmax),
                object_vertex_index=object_vertex_index,
                object_token_keep=object_token_keep,
                rope=rope_object,
                src_key_padding_mask=src_key_padding_mask_object,
                group_keep=object_group_keep,
            )

        x = x + gate_msa.unsqueeze(1) * attn_out
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
