from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .blocks import FinalLayer
from .blocks_spacetemp_altobj import DiTBlockSpaceTempAltObj
from .embeddings import LabelEmbedder, RotaryEmbedding1D, TimestepEmbedder
from .embeddings_vert import VertexCoordRoPE, VertexRotaryEmbedding


@dataclass(frozen=True)
class MeshVideoDiTSpaceTempVertMultiObjAltObjConfig:
    num_vertices: int
    num_frames: int
    in_features: int = 3
    hidden_size: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    num_classes: int = 1
    use_rope: bool = True
    num_register_tokens: int = 16
    max_frames: int = 128
    max_vertices: int = 8192
    grad_checkpoint: bool = False
    spatial_first: bool = True

    # multi-object
    max_num_objects: int = 5
    num_scene_tokens: int = 0
    scene_cond_dim: int = 0
    scene_cond_embed_out_tokens: int = 0
    object_material_dim: int = 0
    use_object_id_embed: bool = False
    block_attn_pattern: tuple[str, ...] = ("spatial", "temporal", "object", "temporal")


class MeshVideoDiTSpaceTempVertMultiObjAltObj(nn.Module):
    """
    Multi-object DiT with a repeated spatial -> temporal -> object -> temporal attention pattern.

    The object-local blocks gather vertices by object id, pad each object group to the largest object
    in the batch, run self-attention with proper masking, and scatter the outputs back to the standard
    per-frame vertex layout. Additive object-id embeddings are disabled by default.
    """

    def __init__(self, cfg: MeshVideoDiTSpaceTempVertMultiObjAltObjConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_frames = int(cfg.num_frames)
        self.num_vertices = int(cfg.num_vertices)
        self.in_features = int(cfg.in_features)
        self.hidden_size = int(cfg.hidden_size)
        self.num_heads = int(cfg.num_heads)
        self.num_classes = int(cfg.num_classes)
        self.use_rope = bool(cfg.use_rope)
        self.num_register_tokens = int(cfg.num_register_tokens)
        self.num_scene_tokens = int(cfg.num_scene_tokens)
        self.scene_cond_dim = int(cfg.scene_cond_dim)
        self.scene_cond_embed_out_tokens = int(cfg.scene_cond_embed_out_tokens)
        self.use_object_id_embed = bool(cfg.use_object_id_embed)
        if self.num_register_tokens < 0:
            raise ValueError(f"num_register_tokens must be >= 0, got {self.num_register_tokens}")
        if self.num_scene_tokens < 0:
            raise ValueError(f"num_scene_tokens must be >= 0, got {self.num_scene_tokens}")
        if self.scene_cond_embed_out_tokens < 0:
            raise ValueError(
                f"scene_cond_embed_out_tokens must be >= 0, got {self.scene_cond_embed_out_tokens}"
            )
        if self.num_scene_tokens > 0 and self.scene_cond_embed_out_tokens <= 0:
            self.scene_cond_embed_out_tokens = self.num_scene_tokens
        self.total_prefix_tokens = self.num_register_tokens + self.num_scene_tokens

        self.max_num_objects = int(cfg.max_num_objects)
        if self.max_num_objects <= 0:
            raise ValueError(f"max_num_objects must be >0, got {self.max_num_objects}")
        self.pad_object_id = int(self.max_num_objects)

        self.x_embedder = nn.Linear(self.in_features, self.hidden_size)
        self.cond_x_embedder = nn.Linear(self.in_features, self.hidden_size)
        self.v_embedder = nn.Linear(self.in_features, self.hidden_size)
        self.object_material_dim = int(cfg.object_material_dim)
        self.object_material_embed = (
            nn.Sequential(
                nn.Linear(self.object_material_dim, self.hidden_size),
                nn.SiLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
            if self.object_material_dim > 0
            else None
        )

        # Keep the table for checkpoint compatibility even though the AltObj backbone does not add it by default.
        self.object_id_embed = nn.Embedding(self.max_num_objects + 1, self.hidden_size, padding_idx=self.pad_object_id)
        if not self.use_object_id_embed:
            self.object_id_embed.weight.requires_grad_(False)

        self.vert_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        if self.num_register_tokens > 0:
            self.reg_tokens = nn.Parameter(torch.randn(1, self.num_register_tokens, self.hidden_size))
        else:
            self.reg_tokens = None
        if self.num_scene_tokens > 0:
            self.scene_token_base = nn.Parameter(torch.randn(1, self.num_scene_tokens, self.hidden_size))
            self.scene_cond_embed = nn.Sequential(
                nn.Linear(self.scene_cond_dim, self.hidden_size),
                nn.SiLU(),
                nn.Linear(self.hidden_size, self.scene_cond_embed_out_tokens * self.hidden_size),
            )
        else:
            self.scene_token_base = None
            self.scene_cond_embed = None

        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.y_embedder = LabelEmbedder(self.num_classes, self.hidden_size)

        head_dim = self.hidden_size // self.num_heads
        if self.use_rope:
            self.rope_time = RotaryEmbedding1D(head_dim, max_frames=int(cfg.max_frames))
            per_coord = max(1, (head_dim // 2) // 3)
            vert_rope_dim = int(2 * per_coord)
            self.rope_vert_full = VertexCoordRoPE(VertexRotaryEmbedding(dim=vert_rope_dim), head_dim=head_dim)
            self.rope_vert_object = VertexCoordRoPE(VertexRotaryEmbedding(dim=vert_rope_dim), head_dim=head_dim)
        else:
            self.rope_time = None
            self.rope_vert_full = None
            self.rope_vert_object = None

        pattern = tuple(str(x) for x in cfg.block_attn_pattern)
        if not pattern:
            raise ValueError("block_attn_pattern must contain at least one attention axis")
        for axis in pattern:
            if axis not in ("spatial", "temporal", "object"):
                raise ValueError(f"Unsupported axis in block_attn_pattern: {axis!r}")
        self.block_attn_pattern = pattern

        self.blocks = nn.ModuleList(
            [
                DiTBlockSpaceTempAltObj(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    attn_axis=self.block_attn_pattern[i % len(self.block_attn_pattern)],
                    mlp_ratio=float(cfg.mlp_ratio),
                    attn_drop=float(cfg.attn_drop),
                    proj_drop=float(cfg.proj_drop),
                )
                for i in range(int(cfg.depth))
            ]
        )
        self.has_object_blocks = any(block.attn_axis == "object" for block in self.blocks)
        self.final_layer = FinalLayer(self.hidden_size, out_dim=self.in_features)

        self._init_weights()

    def _init_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        self.cond_x_embedder.load_state_dict(self.x_embedder.state_dict())

        nn.init.normal_(self.object_id_embed.weight, std=0.02)
        with torch.no_grad():
            self.object_id_embed.weight[self.pad_object_id].zero_()

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        nn.init.normal_(self.vert_token, std=0.02)
        if self.reg_tokens is not None:
            nn.init.normal_(self.reg_tokens, std=0.02)
        if self.scene_token_base is not None:
            nn.init.normal_(self.scene_token_base, std=0.02)

    def _canonical_object_ids(
        self,
        object_ids: Optional[torch.Tensor],
        *,
        bsz: int,
        num_frames: int,
        num_vertices: int,
    ) -> Optional[torch.Tensor]:
        if object_ids is None:
            return None

        if object_ids.ndim == 2:
            if object_ids.shape != (bsz, num_vertices):
                raise ValueError(f"object_ids must be (B,V)={(bsz, num_vertices)}, got {tuple(object_ids.shape)}")
            obj_ids = object_ids
        elif object_ids.ndim == 3:
            if object_ids.shape != (bsz, num_frames, num_vertices):
                raise ValueError(
                    f"object_ids must be (B,F,V)={(bsz, num_frames, num_vertices)}, got {tuple(object_ids.shape)}"
                )
            if not bool((object_ids == object_ids[:, :1, :]).all().item()):
                raise ValueError("AltObj object attention expects frame-invariant object_ids across the clip.")
            obj_ids = object_ids[:, 0, :]
        else:
            raise ValueError(f"object_ids must be (B,V) or (B,F,V), got {tuple(object_ids.shape)}")

        obj_ids = obj_ids.to(dtype=torch.long)
        obj_min = int(obj_ids.min().item())
        obj_max = int(obj_ids.max().item())
        if obj_min < 0 or obj_max > self.pad_object_id:
            raise ValueError(
                f"object_ids must be in [0, {self.pad_object_id}] (pad_id={self.pad_object_id}), got min={obj_min} max={obj_max}"
            )
        return obj_ids

    def _build_object_layout(
        self,
        *,
        object_ids: torch.Tensor,
        mask_bool: Optional[torch.Tensor],
        vert_pos: torch.Tensor,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor | int]:
        bsz, num_frames, num_vertices, _ = vert_pos.shape
        bf = bsz * num_frames
        device = vert_pos.device

        per_sample_object_indices: list[list[torch.Tensor]] = []
        object_nmax = 1
        for b in range(bsz):
            sample_groups: list[torch.Tensor] = []
            for obj_id in range(self.max_num_objects):
                idx = torch.nonzero(object_ids[b] == obj_id, as_tuple=False).flatten()
                sample_groups.append(idx)
                object_nmax = max(object_nmax, int(idx.numel()))
            per_sample_object_indices.append(sample_groups)

        object_index = torch.zeros((bsz, self.max_num_objects, object_nmax), device=device, dtype=torch.long)
        object_slot_keep = torch.zeros((bsz, self.max_num_objects, object_nmax), device=device, dtype=torch.bool)
        for b, sample_groups in enumerate(per_sample_object_indices):
            for obj_id, idx in enumerate(sample_groups):
                n = int(idx.numel())
                if n <= 0:
                    continue
                object_index[b, obj_id, :n] = idx
                object_slot_keep[b, obj_id, :n] = True

        object_index_flat = object_index[:, None, :, :].expand(bsz, num_frames, self.max_num_objects, object_nmax)
        object_index_flat = object_index_flat.reshape(bf, self.max_num_objects * object_nmax)

        if mask_bool is None:
            object_token_keep = object_slot_keep[:, None, :, :].expand(bsz, num_frames, self.max_num_objects, object_nmax)
            object_token_keep = object_token_keep.reshape(bf, self.max_num_objects, object_nmax)
        else:
            mask_flat = mask_bool.reshape(bf, num_vertices)
            gathered_mask = mask_flat.gather(1, object_index_flat).reshape(bf, self.max_num_objects, object_nmax)
            object_slot_keep_bt = object_slot_keep[:, None, :, :].expand(bsz, num_frames, self.max_num_objects, object_nmax)
            object_token_keep = gathered_mask & object_slot_keep_bt.reshape(bf, self.max_num_objects, object_nmax)

        object_group_keep = object_token_keep.any(dim=-1).reshape(bsz, num_frames * self.max_num_objects)

        object_attn_keep = object_token_keep.reshape(bf * self.max_num_objects, object_nmax)
        if self.total_prefix_tokens == 0:
            empty_groups = ~object_attn_keep.any(dim=1)
            if bool(empty_groups.any().item()):
                object_attn_keep = object_attn_keep.clone()
                object_attn_keep[empty_groups, 0] = True
        if self.total_prefix_tokens > 0:
            prefix_keep = torch.ones(
                (bf * self.max_num_objects, self.total_prefix_tokens), device=device, dtype=torch.bool
            )
            src_key_padding_mask_object = torch.cat([prefix_keep, object_attn_keep], dim=1)
        else:
            src_key_padding_mask_object = object_attn_keep

        rope_object_pos = None
        if self.rope_vert_object is not None:
            vert_pos_flat = vert_pos.reshape(bf, num_vertices, self.in_features)
            pos_obj = vert_pos_flat.gather(
                1,
                object_index_flat.unsqueeze(-1).expand(-1, -1, self.in_features),
            ).reshape(bf, self.max_num_objects, object_nmax, self.in_features)
            keep_f = object_token_keep.to(dtype=vert_pos.dtype)
            pos_obj = pos_obj * keep_f.unsqueeze(-1)
            denom = keep_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
            center = (pos_obj.to(dtype=torch.float32).sum(dim=2) / denom.to(dtype=torch.float32)).to(dtype=vert_pos.dtype)
            if self.total_prefix_tokens > 0:
                prefix_pos = center[:, :, None, :].expand(
                    bf, self.max_num_objects, self.total_prefix_tokens, self.in_features
                )
                rope_object_pos = torch.cat([prefix_pos, pos_obj], dim=2)
            else:
                rope_object_pos = pos_obj
            rope_object_pos = rope_object_pos.reshape(
                bf * self.max_num_objects,
                self.total_prefix_tokens + object_nmax,
                self.in_features,
            )

        return {
            "object_vertex_index": object_index_flat,
            "object_token_keep": object_token_keep,
            "object_group_keep": object_group_keep,
            "src_key_padding_mask_object": src_key_padding_mask_object,
            "object_nmax": int(object_nmax),
            "rope_object_pos": rope_object_pos,
        }

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        *,
        cond: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        object_ids: Optional[torch.Tensor] = None,
        scene_cond: Optional[torch.Tensor] = None,
        object_materials: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.ndim != 4 or x.shape[-1] != self.in_features:
            raise ValueError(f"Expected (B,F,V,{self.in_features}), got {tuple(x.shape)}")

        bsz, num_frames, num_vertices, _ = x.shape
        if num_frames > self.cfg.max_frames:
            raise ValueError(f"num_frames={num_frames} exceeds max_frames={self.cfg.max_frames}")
        if num_vertices > self.cfg.max_vertices:
            raise ValueError(f"num_vertices={num_vertices} exceeds max_vertices={self.cfg.max_vertices}")

        object_ids_per_vertex = self._canonical_object_ids(
            object_ids,
            bsz=bsz,
            num_frames=num_frames,
            num_vertices=num_vertices,
        )
        if self.has_object_blocks and object_ids_per_vertex is None:
            raise ValueError("AltObj attention pattern includes object blocks, so object_ids are required.")

        src_key_padding_mask_spatial = None
        src_key_padding_mask_temporal = None
        src_key_padding_mask_object = None
        spatial_group_keep = None
        temporal_group_keep = None
        object_group_keep = None
        object_vertex_index = None
        object_token_keep = None
        object_nmax = 0
        rope_object = None
        mask_bool = None
        if mask is not None:
            if mask.shape != x.shape[:3]:
                raise ValueError(f"mask must match x shape (B,F,V), got {tuple(mask.shape)} vs {tuple(x.shape[:3])}")

            mask_bool = mask.to(device=x.device, dtype=torch.bool)
            spatial_group_keep = mask_bool.any(dim=2)
            temporal_group_keep = mask_bool.any(dim=1)

            keep_spatial = mask_bool.reshape(bsz * num_frames, num_vertices)
            keep_temporal = mask_bool.permute(0, 2, 1).reshape(bsz * num_vertices, num_frames)
            if self.total_prefix_tokens > 0:
                prefix_keep_sp = torch.ones((bsz * num_frames, self.total_prefix_tokens), device=x.device, dtype=torch.bool)
                prefix_keep_t = torch.ones((bsz * num_vertices, self.total_prefix_tokens), device=x.device, dtype=torch.bool)
                src_key_padding_mask_spatial = torch.cat([prefix_keep_sp, keep_spatial], dim=1)
                src_key_padding_mask_temporal = torch.cat([prefix_keep_t, keep_temporal], dim=1)
            else:
                src_key_padding_mask_spatial = keep_spatial
                src_key_padding_mask_temporal = keep_temporal

        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb

        if scene_cond is not None:
            if scene_cond.ndim != 2 or scene_cond.shape[0] != bsz:
                raise ValueError(f"scene_cond must be (B,C) with B={bsz}, got {tuple(scene_cond.shape)}")
            if self.scene_cond_dim <= 0:
                raise ValueError("scene_cond was provided, but this model was built without scene conditioning enabled")
            if scene_cond.shape[1] != self.scene_cond_dim:
                raise ValueError(
                    f"scene_cond last dim must be {self.scene_cond_dim} for this checkpoint, got {tuple(scene_cond.shape)}"
                )
        elif self.num_scene_tokens > 0:
            raise ValueError("scene_cond must be provided when num_scene_tokens > 0")

        vert_pos = x
        vert_pos_flat = vert_pos.reshape(bsz, num_frames * num_vertices, self.in_features)

        x = self.x_embedder(x)
        if cond is not None:
            if cond.ndim == 3:
                if cond.shape[0] != bsz or cond.shape[1] != num_vertices:
                    raise ValueError(f"cond must be (B,V,C) with B={bsz}, V={num_vertices}, got {tuple(cond.shape)}")
                if cond.shape[2] == self.in_features:
                    cond_pos = cond
                    cond_vel = None
                elif cond.shape[2] == 2 * self.in_features:
                    cond_pos = cond[:, :, : self.in_features]
                    cond_vel = cond[:, :, self.in_features :]
                else:
                    raise ValueError(
                        f"cond last dim must be {self.in_features} (pos) or {2 * self.in_features} (pos+vel), "
                        f"got {tuple(cond.shape)}"
                    )
                cond_pos_f = cond_pos[:, None, :, :].expand(bsz, num_frames, num_vertices, self.in_features)
                x = x + self.cond_x_embedder(cond_pos_f)
                if cond_vel is not None:
                    cond_vel_f = cond_vel[:, None, :, :].expand(bsz, num_frames, num_vertices, self.in_features)
                    x = x + self.v_embedder(cond_vel_f)
            elif cond.ndim == 4:
                if cond.shape[0] != bsz or cond.shape[1] != num_frames or cond.shape[2] != num_vertices:
                    raise ValueError(
                        f"cond must be (B,F,V,C) with (B,F,V)={(bsz, num_frames, num_vertices)}, got {tuple(cond.shape)}"
                    )
                if cond.shape[3] == self.in_features:
                    x = x + self.cond_x_embedder(cond)
                elif cond.shape[3] == 2 * self.in_features:
                    cond_pos = cond[:, :, :, : self.in_features]
                    cond_vel = cond[:, :, :, self.in_features :]
                    x = x + self.cond_x_embedder(cond_pos) + self.v_embedder(cond_vel)
                else:
                    raise ValueError(
                        f"cond last dim must be {self.in_features} (pos) or {2 * self.in_features} (pos+vel), "
                        f"got {tuple(cond.shape)}"
                    )
            else:
                raise ValueError(f"cond must be 3D or 4D, got {tuple(cond.shape)}")

        if object_ids_per_vertex is not None and self.use_object_id_embed:
            obj_emb = self.object_id_embed(object_ids_per_vertex.to(device=x.device, dtype=torch.long))
            obj_emb = obj_emb[:, None, :, :].expand(bsz, num_frames, num_vertices, self.hidden_size)
            if mask_bool is not None:
                obj_emb = obj_emb * mask_bool.unsqueeze(-1).to(dtype=obj_emb.dtype)
            x = x + obj_emb

        if object_materials is not None:
            if self.object_material_embed is None or self.object_material_dim <= 0:
                raise ValueError(
                    "object_materials were provided, but this model was built without object material conditioning enabled"
                )
            if object_ids_per_vertex is None:
                raise ValueError("object_materials require object_ids so the per-object rows can be broadcast to vertices")

            if object_materials.ndim == 3:
                if object_materials.shape[0] != bsz:
                    raise ValueError(f"object_materials batch dim must be {bsz}, got {tuple(object_materials.shape)}")
                if object_materials.shape[2] != self.object_material_dim:
                    raise ValueError(
                        f"object_materials last dim must be {self.object_material_dim}, got {tuple(object_materials.shape)}"
                    )
                if object_materials.shape[1] == self.max_num_objects + 1:
                    gather_idx = object_ids_per_vertex.to(device=x.device, dtype=torch.long).unsqueeze(-1).expand(
                        -1,
                        -1,
                        self.object_material_dim,
                    )
                    mat_v = torch.gather(object_materials.to(device=x.device, dtype=x.dtype), 1, gather_idx)
                    mat_v = mat_v[:, None, :, :].expand(bsz, num_frames, num_vertices, self.object_material_dim)
                elif object_materials.shape[1] == num_vertices:
                    mat_v = object_materials.to(device=x.device, dtype=x.dtype)[:, None, :, :].expand(
                        bsz, num_frames, num_vertices, self.object_material_dim
                    )
                else:
                    raise ValueError(
                        "object_materials must be either a per-object table shaped "
                        f"(B,{self.max_num_objects + 1},M) or per-vertex tensor shaped (B,V,M), got {tuple(object_materials.shape)}"
                    )
            elif object_materials.ndim == 4:
                if object_materials.shape[:3] != (bsz, num_frames, num_vertices):
                    raise ValueError(
                        f"object_materials must be (B,F,V,M) with (B,F,V)={(bsz, num_frames, num_vertices)}, got {tuple(object_materials.shape)}"
                    )
                if object_materials.shape[3] != self.object_material_dim:
                    raise ValueError(
                        f"object_materials last dim must be {self.object_material_dim}, got {tuple(object_materials.shape)}"
                    )
                mat_v = object_materials.to(device=x.device, dtype=x.dtype)
            else:
                raise ValueError(
                    f"object_materials must be (B,O,M), (B,V,M), or (B,F,V,M), got {tuple(object_materials.shape)}"
                )

            mat_emb = self.object_material_embed(mat_v)
            if mask_bool is not None:
                mat_emb = mat_emb * mask_bool.unsqueeze(-1).to(dtype=mat_emb.dtype)
            x = x + mat_emb

        rope_spatial = None
        rope_temporal = None
        if self.rope_time is not None and self.rope_vert_full is not None:
            rope_temporal = self.rope_time
            rope_spatial = self.rope_vert_full

            self.rope_time.set_token_layout(
                num_frames=num_frames,
                num_tokens_per_frame=1,
                num_register_tokens=self.total_prefix_tokens,
                device=x.device,
                dtype=x.dtype,
            )

            vert_pos_sp = vert_pos.reshape(bsz * num_frames, num_vertices, self.in_features)
            if self.total_prefix_tokens > 0:
                if mask_bool is not None:
                    w = mask_bool.reshape(bsz, num_frames * num_vertices).to(dtype=vert_pos_flat.dtype)
                    w = w / (w.sum(dim=1, keepdim=True) + 1e-5)
                    center3 = (w.unsqueeze(-1) * vert_pos_flat).sum(dim=1)
                else:
                    center3 = vert_pos_flat.mean(dim=1)
                center_pos = center3[:, None, :].repeat(1, self.total_prefix_tokens, 1)
                reg_pos_sp = center_pos[:, None, :, :].expand(bsz, num_frames, self.total_prefix_tokens, 3).reshape(
                    bsz * num_frames, self.total_prefix_tokens, 3
                )
                vert_pos_sp = torch.cat([reg_pos_sp, vert_pos_sp], dim=1)

            self.rope_vert_full.set_vertex_pos(vert_pos_sp, dtype=x.dtype)

        if self.has_object_blocks:
            layout = self._build_object_layout(
                object_ids=object_ids_per_vertex.to(device=x.device, dtype=torch.long),
                mask_bool=mask_bool,
                vert_pos=vert_pos,
                dtype=x.dtype,
            )
            object_vertex_index = layout["object_vertex_index"]
            object_token_keep = layout["object_token_keep"]
            object_group_keep = layout["object_group_keep"]
            src_key_padding_mask_object = layout["src_key_padding_mask_object"]
            object_nmax = int(layout["object_nmax"])
            if self.rope_vert_object is not None:
                rope_object = self.rope_vert_object
                rope_object_pos = layout["rope_object_pos"]
                if rope_object_pos is not None:
                    self.rope_vert_object.set_vertex_pos(rope_object_pos, dtype=x.dtype)

        x = x.reshape(bsz, num_frames * num_vertices, self.hidden_size)
        x = x + self.vert_token
        prefix_tokens: list[torch.Tensor] = []
        if self.reg_tokens is not None:
            prefix_tokens.append(self.reg_tokens.expand(bsz, -1, -1))
        if self.scene_token_base is not None and self.scene_cond_embed is not None:
            scene_delta = self.scene_cond_embed(scene_cond.to(device=x.device, dtype=x.dtype)).reshape(
                bsz, self.scene_cond_embed_out_tokens, self.hidden_size
            )
            if self.scene_cond_embed_out_tokens == 1 and self.num_scene_tokens > 1:
                scene_delta = scene_delta.expand(bsz, self.num_scene_tokens, self.hidden_size)
            elif self.scene_cond_embed_out_tokens != self.num_scene_tokens:
                raise ValueError(
                    "scene_cond_embed output token count must be 1 or match num_scene_tokens, got "
                    f"{self.scene_cond_embed_out_tokens} vs {self.num_scene_tokens}"
                )
            prefix_tokens.append(self.scene_token_base.expand(bsz, -1, -1) + scene_delta)
        if prefix_tokens:
            x = torch.cat(prefix_tokens + [x], dim=1)

        if self.cfg.grad_checkpoint and self.training:
            for block in self.blocks:

                def _run(x_in: torch.Tensor, c_in: torch.Tensor, _block: DiTBlockSpaceTempAltObj = block) -> torch.Tensor:
                    return _block(
                        x_in,
                        c_in,
                        num_frames=num_frames,
                        num_vertices=num_vertices,
                        rope_spatial=rope_spatial,
                        rope_temporal=rope_temporal,
                        rope_object=rope_object,
                        src_key_padding_mask_spatial=src_key_padding_mask_spatial,
                        src_key_padding_mask_temporal=src_key_padding_mask_temporal,
                        src_key_padding_mask_object=src_key_padding_mask_object,
                        spatial_group_keep=spatial_group_keep,
                        temporal_group_keep=temporal_group_keep,
                        object_group_keep=object_group_keep,
                        object_vertex_index=object_vertex_index,
                        object_token_keep=object_token_keep,
                        num_objects=self.max_num_objects,
                        object_nmax=object_nmax,
                    )

                x = checkpoint(_run, x, c, use_reentrant=False)
        else:
            for block in self.blocks:
                x = block(
                    x,
                    c,
                    num_frames=num_frames,
                    num_vertices=num_vertices,
                    rope_spatial=rope_spatial,
                    rope_temporal=rope_temporal,
                    rope_object=rope_object,
                    src_key_padding_mask_spatial=src_key_padding_mask_spatial,
                    src_key_padding_mask_temporal=src_key_padding_mask_temporal,
                    src_key_padding_mask_object=src_key_padding_mask_object,
                    spatial_group_keep=spatial_group_keep,
                    temporal_group_keep=temporal_group_keep,
                    object_group_keep=object_group_keep,
                    object_vertex_index=object_vertex_index,
                    object_token_keep=object_token_keep,
                    num_objects=self.max_num_objects,
                    object_nmax=object_nmax,
                )

        x = self.final_layer(x, c)
        if self.total_prefix_tokens > 0:
            x = x[:, self.total_prefix_tokens :, :]
        x = x.reshape(bsz, num_frames, num_vertices, self.in_features)
        return x


def MeshVideoDiT_ST_Vert_MultiObj_AltObj_B(**kwargs) -> MeshVideoDiTSpaceTempVertMultiObjAltObj:
    return MeshVideoDiTSpaceTempVertMultiObjAltObj(
        MeshVideoDiTSpaceTempVertMultiObjAltObjConfig(depth=12, hidden_size=768, num_heads=12, **kwargs)
    )


def MeshVideoDiT_ST_Vert_MultiObj_AltObj_L(**kwargs) -> MeshVideoDiTSpaceTempVertMultiObjAltObj:
    return MeshVideoDiTSpaceTempVertMultiObjAltObj(
        MeshVideoDiTSpaceTempVertMultiObjAltObjConfig(depth=24, hidden_size=1024, num_heads=16, **kwargs)
    )


def MeshVideoDiT_ST_Vert_MultiObj_AltObj_H(**kwargs) -> MeshVideoDiTSpaceTempVertMultiObjAltObj:
    return MeshVideoDiTSpaceTempVertMultiObjAltObj(
        MeshVideoDiTSpaceTempVertMultiObjAltObjConfig(depth=32, hidden_size=1280, num_heads=16, **kwargs)
    )


MeshVideoDiT_ST_Vert_MultiObj_AltObj_models = {
    "MeshVideoDiT-ST-Vert-B-MultiObj-AltObj": MeshVideoDiT_ST_Vert_MultiObj_AltObj_B,
    "MeshVideoDiT-ST-Vert-L-MultiObj-AltObj": MeshVideoDiT_ST_Vert_MultiObj_AltObj_L,
    "MeshVideoDiT-ST-Vert-H-MultiObj-AltObj": MeshVideoDiT_ST_Vert_MultiObj_AltObj_H,
}
