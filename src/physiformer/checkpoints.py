"""Load training checkpoints and portable, weights-only PhysiFormer releases."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any
import zipfile

import torch

from physiformer.models.physiformer import canonical_model_name


# Only inference settings may leave a training checkpoint in a release config.
INFERENCE_DEFAULTS = {
    "num_classes": 1, "use_rope": True, "num_register_tokens": 16,
    "max_frames": 128, "max_vertices": 8192, "attn_drop": 0.0, "proj_drop": 0.0,
    "vertex_sampling": "first", "pad_value": 0.0,
    "coord_scale": 1.0, "coord_shift": 0.0, "delta_to_first_frame": False,
    "cond_first_frame": True, "cond_first_frame_velocity": True,
    "cond_first_frame_normal": False, "cond_scene": False,
    "cond_object_material": False, "normalize_to_scene_box": False,
    "material_mode": "auto", "material_softness_rigid": 0.0,
    "material_softness_soft": 1.0, "material_friction_rigid": 0.01,
    "material_friction_soft": 0.15,
    "P_mean": -0.8, "P_std": 0.8, "t_eps": 0.05, "noise_scale": 0.1,
    "sampling_method": "heun", "num_sampling_steps": 50,
}
REQUIRED_ARGS = ("model", "num_frames", "num_vertices", "max_num_objects", "norm_mean", "norm_std")


def config_digest(config: dict) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_config(config: dict) -> dict:
    if not isinstance(config, dict) or config.get("format") != "physiformer" or config.get("format_version") != 1:
        raise ValueError("Expected a PhysiFormer config.json with format_version=1")
    if config.get("weight_source") not in ("ema", "model"):
        raise ValueError("config.json weight_source must be 'ema' or 'model'")
    args = config.get("model_args")
    if not isinstance(args, dict) or any(k not in args for k in REQUIRED_ARGS):
        raise ValueError(f"config.json model_args must include {REQUIRED_ARGS}")
    if canonical_model_name(args["model"]) not in ("PhysiFormer", "PhysiFormer-B"):
        raise ValueError("Unsupported model in config.json")
    for name in ("num_frames", "num_vertices", "max_num_objects"):
        if type(args[name]) is not int or args[name] <= 0:
            raise ValueError(f"config.json {name} must be a positive integer")
    for name in ("norm_mean", "norm_std"):
        values = args[name]
        if not isinstance(values, (list, tuple)) or len(values) != 3 or not all(
            isinstance(v, (int, float)) and math.isfinite(v) for v in values
        ):
            raise ValueError(f"config.json {name} must contain three finite numbers")
    if min(args["norm_std"]) <= 0:
        raise ValueError("config.json norm_std values must be positive")
    return args


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """SafeTensors requires its matching sibling config.json; .pt retains its args."""
    path = Path(path)
    if path.suffix.lower() == ".safetensors":
        from safetensors import safe_open
        from safetensors.torch import load_file

        config_path = path.with_name("config.json")
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing {config_path}; download config.json alongside the weights")
        config = json.loads(config_path.read_text())
        args = validate_config(config)
        with safe_open(str(path), framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
        if metadata.get("config_sha256") != config_digest(config):
            raise ValueError("SafeTensors/config.json mismatch; use the configuration shipped with these weights")
        return {"model": load_file(str(path), device="cpu"), "args": args,
                "weight_source": config["weight_source"], "format": "safetensors"}
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=zipfile.is_zipfile(path))
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a dictionary checkpoint or model state_dict")
    return checkpoint


def select_weights(checkpoint: dict, *, use_ema: bool = True) -> tuple[dict, str]:
    """A weights-only release always uses its single, explicitly identified set."""
    if checkpoint.get("format") == "safetensors":
        return checkpoint["model"], checkpoint["weight_source"]
    ema = checkpoint.get("ema")
    if use_ema and isinstance(ema, dict) and isinstance(ema.get("shadow"), dict):
        return ema["shadow"], "ema"
    return checkpoint.get("model", checkpoint), "model"


def load_training_checkpoint(path: str | Path) -> dict:
    if Path(path).suffix.lower() == ".safetensors":
        raise ValueError("--resume requires a full .pt training checkpoint; use --init_ckpt for SafeTensors weights")
    checkpoint = load_checkpoint(path)
    if not all(k in checkpoint for k in ("model", "optimizer", "epoch", "step")):
        raise ValueError("--resume requires model, optimizer, epoch, and step; use --init_ckpt for weights only")
    return checkpoint


def adapt_legacy_conditioner(state: dict, target: dict) -> dict:
    """Use the same legacy conditioner mapping for initialization and inference."""
    result = dict(state)
    for key in list(result):
        if "x_embed_cond." in key:
            new_key = key.replace("x_embed_cond.", "cond_x_embedder.")
            if new_key in target:
                result.setdefault(new_key, result.pop(key))
    for key in target:
        if "cond_x_embedder." in key and key not in result:
            source = key.replace("cond_x_embedder.", "x_embedder.")
            if source in result:
                result[key] = result[source]
    return result


def release_config(checkpoint: dict, state: dict, weight_source: str) -> dict:
    args = checkpoint.get("args", {})
    missing = [k for k in REQUIRED_ARGS if k not in args]
    if missing:
        raise ValueError(f"Checkpoint lacks inference settings: {missing}; cannot export a self-contained release")
    model_args = {k: args.get(k, default) for k, default in INFERENCE_DEFAULTS.items()}
    model_args.update({k: args[k] for k in REQUIRED_ARGS})
    model_args["model"] = canonical_model_name(model_args["model"])

    def tensor(suffix):
        return next((v for k, v in state.items() if k.endswith(suffix)), None)

    mat = tensor("object_material_embed.0.weight")
    scene_in = tensor("scene_cond_embed.0.weight")
    scene_out = tensor("scene_cond_embed.2.weight")
    scene_base = tensor("scene_token_base")
    scene_dim = int(scene_in.shape[1]) if scene_in is not None else 0
    scene_tokens = int(scene_out.shape[0] // scene_in.shape[0]) if scene_in is not None and scene_out is not None else 0
    model_args.update(
        object_material_dim=int(mat.shape[1]) if mat is not None else 0,
        scene_cond_dim=scene_dim, scene_cond_embed_out_tokens=scene_tokens,
        num_scene_tokens=int(scene_base.shape[1]) if scene_base is not None else scene_tokens,
    )
    model_args["cond_object_material"] = bool(model_args["cond_object_material"] or mat is not None)
    model_args["cond_scene"] = bool(model_args["cond_scene"] or scene_dim or scene_tokens)
    config = {"format": "physiformer", "format_version": 1,
              "weight_source": weight_source, "model_args": model_args}
    # Normalize tuples and reject non-JSON/non-finite metadata before writing.
    config = json.loads(json.dumps(config, allow_nan=False))
    validate_config(config)
    return config
