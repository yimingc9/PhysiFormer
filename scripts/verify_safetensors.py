#!/usr/bin/env python3
"""Compare .pt and SafeTensors inference on a short CPU trajectory (full model)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from physiformer.checkpoints import load_checkpoint, select_weights
from physiformer.diffusion.denoiser import DiffusionConfig
from physiformer.diffusion.physiformer_denoiser import PhysiFormerDenoiser
from physiformer.models.physiformer import canonical_model_name
from physiformer.scripts.infer import (
    _add_cond_x_embedder_keys_from_x_embedder, _infer_conditioning_dims_from_state_dict,
    _rename_legacy_x_embed_cond_keys, _resolve_norm_stats,
)


def verify_inference(checkpoint: Path, release: Path, sample: Path) -> dict:
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    original = load_checkpoint(checkpoint)
    exported = load_checkpoint(release)
    source = exported["weight_source"]
    states = [select_weights(c, use_ema=source == "ema")[0] for c in (original, exported)]
    with np.load(sample, allow_pickle=False) as data:
        # Two real objects, two vertices each, with the original initial conditions.
        ids = data["object_ids"]
        objects = np.unique(ids[data["mask"][0].astype(bool)])[:2]
        indices = np.concatenate([np.flatnonzero(ids == obj)[:2] for obj in objects])
        positions = data["vertices"][0, indices].astype(np.float32)
        velocities = data["first_frame_velocity"][indices].astype(np.float32)
        selected_ids = ids[indices].astype(np.int64)
    outputs = []
    model = None
    first_kwargs = None
    for payload, state in zip((original, exported), states):
        saved = payload["args"]
        scene_tokens, scene_dim, scene_out, mat_dim = _infer_conditioning_dims_from_state_dict(state)
        if scene_dim or scene_tokens:
            raise ValueError("This verification sample supports object-material conditioning without scene tokens")
        kwargs = {
            "use_rope": bool(saved.get("use_rope", True)),
            "num_register_tokens": int(saved.get("num_register_tokens", 16)),
            "max_frames": int(saved.get("max_frames", 128)),
            "max_vertices": int(saved.get("max_vertices", 8192)),
            "attn_drop": float(saved.get("attn_drop", 0.0)),
            "proj_drop": float(saved.get("proj_drop", 0.0)),
            "max_num_objects": int(saved["max_num_objects"]), "use_object_id_embed": False,
            "num_scene_tokens": scene_tokens, "scene_cond_dim": scene_dim,
            "scene_cond_embed_out_tokens": scene_out, "object_material_dim": mat_dim,
        }
        constructor = dict(model_name=canonical_model_name(saved["model"]),
                           num_frames=int(saved["num_frames"]), num_vertices=int(saved["num_vertices"]),
                           num_classes=int(saved.get("num_classes", 1)), model_kwargs=kwargs)
        diffusion = DiffusionConfig(**{key: saved.get(key, getattr(DiffusionConfig, key))
                                      for key in DiffusionConfig.__dataclass_fields__})
        # The full architecture/weights are tested; reduce rollout length and steps for CPU.
        diffusion = DiffusionConfig(**{**vars(diffusion), "num_sampling_steps": 2})
        if model is None:
            model = PhysiFormerDenoiser(**constructor, diffusion=diffusion).eval()
            first_kwargs = constructor
        else:
            assert constructor == first_kwargs, "Model configuration changed"
            assert diffusion == model.diff, "Diffusion configuration changed"
        adapted = dict(state)
        _rename_legacy_x_embed_cond_keys(adapted, model)
        _add_cond_x_embedder_keys_from_x_embedder(adapted, model)
        # Copy into the same buffers for both runs: mmap alignment can change CPU kernel choices.
        model.load_state_dict(adapted, strict=True)
        assert not any(v.is_meta for v in model.state_dict().values())
        mean, std = _resolve_norm_stats(saved, argparse.Namespace(norm_mean=None, norm_std=None))
        scale, shift = float(saved.get("coord_scale", 1)), float(saved.get("coord_shift", 0))
        normalized = (positions * scale + shift - mean) / std
        velocity = velocities * scale / std
        cond = torch.from_numpy(np.concatenate([normalized, velocity], axis=-1)).unsqueeze(0)
        ids_tensor = torch.from_numpy(selected_ids).unsqueeze(0)
        frames, vertices = 2, len(indices)
        materials = torch.ones(1, kwargs["max_num_objects"] + 1, mat_dim)
        materials *= float(saved.get("material_softness_soft", 1))
        materials[:, -1] = 0
        torch.manual_seed(1234)
        with torch.inference_mode():
            result, _ = model.generate(torch.zeros(1, dtype=torch.long), num_frames=frames,
                                       num_vertices=vertices, cond_first_frame=cond,
                                       mask=torch.ones(1, frames, vertices), object_ids=ids_tensor,
                                       object_materials=materials if mat_dim else None)
        result = (result.numpy() * std + mean - shift) / scale
        assert np.isfinite(result).all(), "Non-finite trajectory"
        outputs.append(result.copy())
        print(f"Verified complete parameter loading and CPU generation: {payload.get('format', 'pt')}", flush=True)
    np.testing.assert_array_equal(outputs[0], outputs[1])
    return {"weight_source": source, "device": "cpu", "precision": "float32",
            "frames": 2, "vertices": len(indices), "sampling_steps": 2, "seed": 1234,
            "max_absolute_output_difference": float(np.max(np.abs(outputs[0] - outputs[1]))),
            "parameter_loading": "strict, complete", "scope": "short trajectory with full model"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--sample", type=Path, default=Path(__file__).resolve().parents[1] / "data_toy/elastic/2_obj/sample_0/sample.npz")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = verify_inference(args.checkpoint, args.release, args.sample)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
