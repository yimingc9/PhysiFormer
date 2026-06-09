from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

from .denoiser import DiffusionConfig
from physformer.models.mesh_video_dit_spacetemp_vert_multiobj_altobj import MeshVideoDiT_ST_Vert_MultiObj_AltObj_models


class DenoiserMeshVideoSpaceTempVertMultiObjAltObj(nn.Module):
    """
    Video diffusion denoiser for vertex-tokenized multi-object mesh trajectories.

    - Input x: (B, F, V, 3)
    - Predicts x_pred and trains via v-loss (as in JiT)
    - Accepts per-vertex object ids (B,V) or (B,F,V) that are passed to the backbone.
    """

    def __init__(
        self,
        *,
        model_name: str,
        num_frames: int,
        num_vertices: int,
        num_classes: int,
        model_kwargs: dict,
        diffusion: DiffusionConfig,
    ) -> None:
        super().__init__()
        if model_name not in MeshVideoDiT_ST_Vert_MultiObj_AltObj_models:
            raise ValueError(
                f"Unknown model_name={model_name}. Available: {sorted(MeshVideoDiT_ST_Vert_MultiObj_AltObj_models)}"
            )

        self.net = MeshVideoDiT_ST_Vert_MultiObj_AltObj_models[model_name](
            num_frames=num_frames,
            num_vertices=num_vertices,
            num_classes=num_classes,
            **model_kwargs,
        )
        self.num_classes = int(num_classes)
        self.diff = diffusion

    def sample_t(self, n: int, device=None) -> torch.Tensor:
        z = torch.randn(n, device=device) * self.diff.P_std + self.diff.P_mean
        return torch.sigmoid(z)

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        *,
        cond_first_frame: Optional[torch.Tensor] = None,
        object_ids: Optional[torch.Tensor] = None,
        scene_cond: Optional[torch.Tensor] = None,
        object_materials: Optional[torch.Tensor] = None,
        return_x_pred: bool = False,
    ):
        if x.ndim != 4 or x.shape[-1] != 3:
            raise ValueError(f"x must be (B,F,V,3), got {tuple(x.shape)}")

        if cond_first_frame is not None:
            if cond_first_frame.ndim not in (3, 4):
                raise ValueError(
                    f"cond_first_frame must be (B,V,C) or (B,F,V,C), got {tuple(cond_first_frame.shape)}"
                )
            if cond_first_frame.shape[0] != x.shape[0]:
                raise ValueError(
                    f"cond_first_frame batch dim must match x (B={x.shape[0]}), got {tuple(cond_first_frame.shape)}"
                )
            if cond_first_frame.ndim == 3:
                if cond_first_frame.shape[1] != x.shape[2]:
                    raise ValueError(
                        f"cond_first_frame must be (B,V,C) with (B,V)={(x.shape[0], x.shape[2])}, got {tuple(cond_first_frame.shape)}"
                    )
            else:
                if cond_first_frame.shape[1] != x.shape[1] or cond_first_frame.shape[2] != x.shape[2]:
                    raise ValueError(
                        f"cond_first_frame must be (B,F,V,C) with (B,F,V)={(x.shape[0], x.shape[1], x.shape[2])}, got {tuple(cond_first_frame.shape)}"
                    )
            if cond_first_frame.shape[-1] not in (3, 6):
                raise ValueError(
                    f"cond_first_frame last dim must be 3 (pos) or 6 (pos+vel), got {cond_first_frame.shape[-1]}"
                )

        if object_ids is not None:
            if object_ids.ndim == 2:
                if object_ids.shape != (x.shape[0], x.shape[2]):
                    raise ValueError(f"object_ids must be (B,V)={(x.shape[0], x.shape[2])}, got {tuple(object_ids.shape)}")
            elif object_ids.ndim == 3:
                if object_ids.shape != x.shape[:3]:
                    raise ValueError(f"object_ids must be (B,F,V)={tuple(x.shape[:3])}, got {tuple(object_ids.shape)}")
            else:
                raise ValueError(f"object_ids must be (B,V) or (B,F,V), got {tuple(object_ids.shape)}")

        if scene_cond is not None:
            if scene_cond.ndim != 2 or scene_cond.shape[0] != x.shape[0]:
                raise ValueError(f"scene_cond must be (B,C) with B={x.shape[0]}, got {tuple(scene_cond.shape)}")

        if object_materials is not None:
            if object_materials.ndim not in (3, 4) or object_materials.shape[0] != x.shape[0]:
                raise ValueError(
                    f"object_materials must be (B,O,M), (B,V,M), or (B,F,V,M) with B={x.shape[0]}, got {tuple(object_materials.shape)}"
                )

        mask4 = None
        if mask is not None:
            if mask.ndim != 3:
                raise ValueError(f"mask must be (B,F,V), got {tuple(mask.shape)}")
            if mask.shape != x.shape[:3]:
                raise ValueError(f"mask shape {tuple(mask.shape)} must match x (B,F,V,3)={tuple(x.shape)}")

            # Keep padded tokens inert during training.
            mask4 = mask.unsqueeze(-1).to(dtype=x.dtype)  # (B,F,V,1)
            x = x * mask4
            if cond_first_frame is not None:
                if cond_first_frame.ndim == 3:
                    cond_first_frame = cond_first_frame * mask[:, 0].unsqueeze(-1).to(dtype=cond_first_frame.dtype)
                else:
                    cond_first_frame = cond_first_frame * mask4.to(dtype=cond_first_frame.dtype)

        t = self.sample_t(x.size(0), device=x.device).view(-1, 1, 1, 1)  # (B,1,1,1)
        e = torch.randn_like(x) * self.diff.noise_scale
        if mask4 is not None:
            e = e * mask4

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.diff.t_eps)

        x_pred = self.net(
            z,
            t.flatten(),
            labels,
            cond=cond_first_frame,
            mask=mask,
            object_ids=object_ids,
            scene_cond=scene_cond,
            object_materials=object_materials,
        )
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.diff.t_eps)

        loss = (v - v_pred) ** 2  # (B,F,V,3)
        loss = loss.mean(dim=-1)  # (B,F,V)
        if mask is not None:
            if mask.shape != loss.shape:
                raise ValueError(f"mask shape {tuple(mask.shape)} must match (B,F,V)={tuple(loss.shape)}")
            denom = mask.sum(dim=(1, 2)).clamp_min(1.0)
            loss = (loss * mask).sum(dim=(1, 2)) / denom
        else:
            loss = loss.mean(dim=(1, 2))
        loss = loss.mean()

        if return_x_pred:
            return loss, x_pred
        return loss

    @torch.no_grad()
    def _forward_v(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
        *,
        cond_first_frame: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        object_ids: Optional[torch.Tensor] = None,
        scene_cond: Optional[torch.Tensor] = None,
        object_materials: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_pred = self.net(
            z,
            t.flatten(),
            labels,
            cond=cond_first_frame,
            mask=mask,
            object_ids=object_ids,
            scene_cond=scene_cond,
            object_materials=object_materials,
        )
        return (x_pred - z) / (1.0 - t.view(-1, 1, 1, 1)).clamp_min(self.diff.t_eps)

    @torch.no_grad()
    def _euler_step(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        labels: torch.Tensor,
        *,
        cond_first_frame: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        object_ids: Optional[torch.Tensor] = None,
        scene_cond: Optional[torch.Tensor] = None,
        object_materials: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        v_pred = self._forward_v(
            z,
            t,
            labels,
            cond_first_frame=cond_first_frame,
            mask=mask,
            object_ids=object_ids,
            scene_cond=scene_cond,
            object_materials=object_materials,
        )
        dt = (t_next - t).view(-1, 1, 1, 1)
        return z + dt * v_pred

    @torch.no_grad()
    def _heun_step(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        labels: torch.Tensor,
        *,
        cond_first_frame: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        object_ids: Optional[torch.Tensor] = None,
        scene_cond: Optional[torch.Tensor] = None,
        object_materials: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        v_t = self._forward_v(
            z,
            t,
            labels,
            cond_first_frame=cond_first_frame,
            mask=mask,
            object_ids=object_ids,
            scene_cond=scene_cond,
            object_materials=object_materials,
        )
        dt = (t_next - t).view(-1, 1, 1, 1)
        z_euler = z + dt * v_t
        v_next = self._forward_v(
            z_euler,
            t_next,
            labels,
            cond_first_frame=cond_first_frame,
            mask=mask,
            object_ids=object_ids,
            scene_cond=scene_cond,
            object_materials=object_materials,
        )
        v = 0.5 * (v_t + v_next)
        return z + dt * v

    @torch.no_grad()
    def generate(
        self,
        labels: torch.Tensor,
        *,
        num_frames: int,
        num_vertices: int,
        displacement: Optional[torch.Tensor] = None,
        cond_first_frame: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        object_ids: Optional[torch.Tensor] = None,
        scene_cond: Optional[torch.Tensor] = None,
        object_materials: Optional[torch.Tensor] = None,
        clamp_cond_first_frame: bool = True,
        trace_callback: Optional[Callable[[int, float, torch.Tensor], None]] = None,
        trace_every: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = labels.device
        bsz = labels.size(0)

        mask4 = None
        if mask is not None:
            if mask.shape != (bsz, num_frames, num_vertices):
                raise ValueError(f"mask must be (B,F,V)={bsz, num_frames, num_vertices}, got {tuple(mask.shape)}")
            mask4 = mask.unsqueeze(-1).to(device=device, dtype=torch.float32)  # (B,F,V,1)

        z = self.diff.noise_scale * torch.randn(bsz, num_frames, num_vertices, 3, device=device)
        if displacement is not None:
            if displacement.shape != (3,):
                raise ValueError("displacement must be shape (3,)")
            z = z + displacement.view(1, 1, 1, 3)
        if mask4 is not None:
            z = z * mask4.to(dtype=z.dtype)

        steps = int(self.diff.num_sampling_steps)
        ts = torch.linspace(0.0, 1.0, steps + 1, device=device)

        cond_noise = None
        cond_first_pos = None
        if cond_first_frame is not None and clamp_cond_first_frame:
            if cond_first_frame.ndim != 3:
                raise ValueError(f"cond_first_frame must be (B,V,C), got {tuple(cond_first_frame.shape)}")
            if cond_first_frame.shape[0] != bsz or cond_first_frame.shape[1] != num_vertices:
                raise ValueError(
                    f"cond_first_frame must be (B,V,C) with (B,V)={(bsz, num_vertices)}, got {tuple(cond_first_frame.shape)}"
                )
            if cond_first_frame.shape[2] not in (3, 6):
                raise ValueError(
                    f"cond_first_frame last dim must be 3 (pos) or 6 (pos+vel), got {cond_first_frame.shape[2]}"
                )
            cond_first = cond_first_frame
            cond_first_pos = cond_first_frame[:, :, :3] if cond_first_frame.shape[2] == 6 else cond_first_frame
            if mask is not None:
                w = mask[:, 0].unsqueeze(-1).to(dtype=cond_first.dtype)
                cond_first = cond_first * w
                cond_first_pos = cond_first_pos * w.to(dtype=cond_first_pos.dtype)
            cond_noise = self.diff.noise_scale * torch.randn(bsz, 1, num_vertices, 3, device=device)
            if mask4 is not None:
                cond_noise = cond_noise * mask4[:, 0:1].to(dtype=cond_noise.dtype)
            z[:, 0:1] = cond_noise
        else:
            cond_first = cond_first_frame

        z0 = z.clone()

        if self.diff.sampling_method == "euler":
            stepper = self._euler_step
        elif self.diff.sampling_method == "heun":
            stepper = self._heun_step
        else:
            raise ValueError(f"Unknown sampling_method={self.diff.sampling_method}")

        def maybe_trace(step_idx: int, t_val: torch.Tensor) -> None:
            if trace_callback is None:
                return
            every = int(trace_every)
            if every <= 0:
                return
            if step_idx % every != 0:
                return
            trace_callback(int(step_idx), float(t_val[0].item()), z)

        maybe_trace(0, ts[0].expand(bsz))
        for i in range(steps - 1):
            t = ts[i].expand(bsz)
            t_next = ts[i + 1].expand(bsz)
            z = stepper(
                z,
                t,
                t_next,
                labels,
                cond_first_frame=cond_first,
                mask=mask,
                object_ids=object_ids,
                scene_cond=scene_cond,
                object_materials=object_materials,
            )
            if mask4 is not None:
                z = z * mask4.to(dtype=z.dtype)

            if cond_first is not None and clamp_cond_first_frame and cond_noise is not None:
                tn = t_next.view(-1, 1, 1, 1)
                if cond_first_pos is None:
                    raise RuntimeError("cond_first_pos must be set when clamp_cond_first_frame is enabled")
                z[:, 0:1] = tn * cond_first_pos[:, None, :, :] + (1.0 - tn) * cond_noise
                if mask4 is not None:
                    z[:, 0:1] = z[:, 0:1] * mask4[:, 0:1].to(dtype=z.dtype)

            maybe_trace(i + 1, t_next)

        z = self._euler_step(
            z,
            ts[-2].expand(bsz),
            ts[-1].expand(bsz),
            labels,
            cond_first_frame=cond_first,
            mask=mask,
            object_ids=object_ids,
            scene_cond=scene_cond,
            object_materials=object_materials,
        )
        if mask4 is not None:
            z = z * mask4.to(dtype=z.dtype)
        if cond_first is not None and clamp_cond_first_frame and cond_noise is not None:
            if cond_first_pos is None:
                raise RuntimeError("cond_first_pos must be set when clamp_cond_first_frame is enabled")
            z[:, 0:1] = cond_first_pos[:, None, :, :]
            if mask4 is not None:
                z[:, 0:1] = z[:, 0:1] * mask4[:, 0:1].to(dtype=z.dtype)
        maybe_trace(steps, ts[-1].expand(bsz))
        return z, z0


# Script-friendly alias.
DenoiserMeshVideoMultiObjAltObj = DenoiserMeshVideoSpaceTempVertMultiObjAltObj
