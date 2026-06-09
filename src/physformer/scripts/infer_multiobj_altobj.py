from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch

_MPLCONFIGDIR = Path(__file__).resolve().parents[3] / ".inference_work" / "matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, *args, **kwargs):  # type: ignore[no-redef]
        return iterable if iterable is not None else ()

    tqdm.write = print  # type: ignore[attr-defined]

# Allow running this file directly without installing the package.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from physformer.data.multiobj_utils_multiobj import (
    default_vertex_count_json_path,
    load_mesh_vertex_counts,
    resolve_velocity_path_from_first_frame_obj,
    scene_info_from_metadata_dict,
)
from physformer.data.obj_io import load_obj_vertices_faces
from physformer.data.vertex_utils import fix_num_vertices
from physformer.diffusion.denoiser import DiffusionConfig
from physformer.diffusion.denoiser_spacetemp_vert_multiobj_altobj import DenoiserMeshVideoMultiObjAltObj


SCENE_COND_DIM = 10
OBJECT_MATERIAL_DIM = 12
ENV_AND_MAT_SCENE_COND_DIM = 9
ENV_AND_MAT_OBJECT_MATERIAL_DIM = 2
NAMED_COLORS = {
    "bunny": (0.82, 0.76, 0.02, 1.0),
    "cow": (0.00, 0.62, 0.66, 1.0),
    "fish": (0.62, 0.80, 0.20, 1.0),
    "horse": (0.88, 0.30, 0.24, 1.0),
    "teapot": (0.68, 0.34, 0.82, 1.0),
}
MESH_EDGE_COLOR = (0.05, 0.06, 0.07, 0.62)
LIGHT_DIRECTION = np.asarray([0.45, -0.65, 0.75], dtype=np.float32)


def _add_cond_x_embedder_keys_from_x_embedder(state_dict: dict[str, Any], module: torch.nn.Module) -> int:
    """Backfill first-frame-position conditioner weights for older checkpoints."""
    target_state = module.state_dict()
    added = 0
    for key, like in target_state.items():
        if "cond_x_embedder." not in str(key) or key in state_dict:
            continue
        source_key = str(key).replace("cond_x_embedder.", "x_embedder.")
        source = state_dict.get(source_key, None)
        if source is None:
            continue
        if not torch.is_tensor(source):
            raise ValueError(f"Expected tensor for checkpoint key {source_key!r}, got {type(source).__name__}")
        if tuple(source.shape) != tuple(like.shape):
            raise ValueError(
                f"Cannot initialize {key!r} from {source_key!r}: shape {tuple(source.shape)} "
                f"does not match expected {tuple(like.shape)}"
            )
        state_dict[key] = source.detach().clone()
        added += 1
    return added


def _rename_legacy_x_embed_cond_keys(state_dict: dict[str, Any], module: torch.nn.Module) -> int:
    """Map legacy x_embed_cond checkpoint keys onto the current cond_x_embedder names."""
    target_state = module.state_dict()
    renamed = 0
    for key in list(state_dict.keys()):
        key_s = str(key)
        if "x_embed_cond." not in key_s:
            continue
        target_key = key_s.replace("x_embed_cond.", "cond_x_embedder.")
        value = state_dict.pop(key)
        target_like = target_state.get(target_key)
        if target_like is None:
            continue
        if not torch.is_tensor(value):
            raise ValueError(f"Expected tensor for checkpoint key {key_s!r}, got {type(value).__name__}")
        if tuple(value.shape) != tuple(target_like.shape):
            raise ValueError(
                f"Cannot rename {key_s!r} to {target_key!r}: shape {tuple(value.shape)} "
                f"does not match expected {tuple(target_like.shape)}"
            )
        if target_key not in state_dict:
            state_dict[target_key] = value
            renamed += 1
    return renamed


def _expand_object_id_embed_in_state_dict(
    state_dict: dict[str, Any],
    *,
    target_max_num_objects: int,
    init_std: float,
) -> bool:
    changed = False
    suffix = "object_id_embed.weight"
    keys = [k for k in state_dict.keys() if str(k).endswith(suffix)]
    for key in keys:
        weight = state_dict.get(key, None)
        if not torch.is_tensor(weight) or weight.ndim != 2:
            continue
        old_num, dim = int(weight.shape[0]), int(weight.shape[1])
        old_max = old_num - 1
        new_num = int(target_max_num_objects) + 1
        if new_num <= old_num:
            continue

        mean_row = weight[:old_max].mean(dim=0, keepdim=True) if old_max > 0 else weight.new_zeros((1, dim))
        new_weight = weight.new_empty((new_num, dim))
        copy_n = min(int(old_max), int(target_max_num_objects))
        if copy_n > 0:
            new_weight[:copy_n] = weight[:copy_n]
        if copy_n < int(target_max_num_objects):
            n_extra = int(target_max_num_objects) - copy_n
            noise = torch.randn((n_extra, dim), dtype=weight.dtype, device=weight.device) * float(init_std)
            new_weight[copy_n : int(target_max_num_objects)] = mean_row + noise
        new_weight[int(target_max_num_objects)].zero_()
        state_dict[key] = new_weight
        changed = True
    return bool(changed)


def _maybe_expand_ckpt(ckpt: Any, *, target_max_num_objects: int, init_std: float) -> bool:
    if not isinstance(ckpt, dict):
        return False

    changed = False
    model_sd = ckpt.get("model", None)
    if isinstance(model_sd, dict):
        changed |= _expand_object_id_embed_in_state_dict(
            model_sd,
            target_max_num_objects=int(target_max_num_objects),
            init_std=float(init_std),
        )

    ema = ckpt.get("ema", None)
    if isinstance(ema, dict):
        shadow_sd = ema.get("shadow", None)
        if isinstance(shadow_sd, dict):
            changed |= _expand_object_id_embed_in_state_dict(
                shadow_sd,
                target_max_num_objects=int(target_max_num_objects),
                init_std=float(init_std),
            )

    if changed and isinstance(ckpt.get("args", None), dict):
        ckpt["args"]["max_num_objects"] = int(target_max_num_objects)
    return bool(changed)


def _load_obj_vertices_only(path: str) -> np.ndarray:
    vertices: list[list[float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise ValueError(f"OBJ has no vertices: {path}")
    return np.asarray(vertices, dtype=np.float32)


def _parse_labels(s: str, num_samples: int) -> torch.Tensor:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    ints = [int(p) for p in parts] if parts else [0]
    if len(ints) == 1:
        ints = ints * num_samples
    if len(ints) != num_samples:
        raise ValueError("--labels must be a single int or a comma-separated list matching --num_samples")
    return torch.tensor(ints, dtype=torch.long)


def load_metadata(meta_path: str) -> dict:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if not isinstance(meta, dict):
        raise ValueError(f"metadata.json must contain a dict, got {type(meta)}: {meta_path}")
    return meta


def _load_conditioned_metadata_jsonl(path: str, *, expected_count: int) -> list[dict] | None:
    path = str(path).strip()
    if not path:
        return None
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Conditioned metadata record must be a JSON object: {path}:{line_no}")
            metadata = payload.get("metadata", payload)
            if not isinstance(metadata, dict):
                raise ValueError(f"Conditioned metadata payload missing object metadata: {path}:{line_no}")
            records.append(metadata)
    if len(records) != int(expected_count):
        raise ValueError(
            f"--conditioned_metadata_jsonl contains {len(records)} records but --num_samples={int(expected_count)}: {path}"
        )
    return records


def _first_numeric_value(container: Any, *keys: str) -> Optional[float]:
    if not isinstance(container, dict):
        return None
    for key in keys:
        value = container.get(key, None)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _first_non_none(*values: Optional[float]) -> Optional[float]:
    for value in values:
        if value is not None:
            return value
    return None


def _safe_float(x: object, default: float = 0.0) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    return float(default)


def _safe_bool01(x: object, default: bool = False) -> float:
    return 1.0 if bool(x) else (1.0 if bool(default) else 0.0)


def _resolve_optional_bool(value: Optional[bool], default: bool) -> bool:
    return bool(default) if value is None else bool(value)


def _log10_clamped(x: object, *, floor: float = 1e-8, default: float = 0.0) -> float:
    if not isinstance(x, (int, float)):
        return float(default)
    return float(np.log10(max(float(x), float(floor))))


def scene_cond_from_metadata_dict(meta: dict) -> np.ndarray:
    bounds_min = np.asarray(meta.get("bounds_min", [-1.0, -1.0, -1.0]), dtype=np.float32).reshape(3)
    bounds_max = np.asarray(meta.get("bounds_max", [1.0, 1.0, 1.0]), dtype=np.float32).reshape(3)
    center = 0.5 * (bounds_min + bounds_max)
    size = np.maximum(bounds_max - bounds_min, 1e-6)
    gravity_z = _safe_float(meta.get("gravity_z", -9.81), -9.81)
    wall_clearance = _safe_float(meta.get("wall_clearance", 0.0), 0.0)
    ceiling = _safe_bool01(meta.get("ceiling", False), False)
    boundary_margin = _safe_float(meta.get("boundary_margin", 0.0), 0.0)
    out = np.asarray(
        [
            float(center[0]),
            float(center[1]),
            float(center[2]),
            float(size[0]),
            float(size[1]),
            float(size[2]),
            float(gravity_z),
            float(wall_clearance),
            float(ceiling),
            float(boundary_margin),
        ],
        dtype=np.float32,
    )
    if out.shape != (SCENE_COND_DIM,):
        raise RuntimeError(f"Internal error: scene_cond shape mismatch {out.shape} != {(SCENE_COND_DIM,)}")
    return out


def _merged_dict(*sources: object) -> dict:
    out: dict = {}
    for src in sources:
        if isinstance(src, dict):
            out.update(src)
    return out


def _infer_env_and_mat_material_mode(meta: dict, meta_path: str, train_args: dict) -> str:
    mode = str(train_args.get("material_mode", "auto"))
    if mode in ("rigid", "soft"):
        return mode

    material_friction_rigid = float(train_args.get("material_friction_rigid", 0.01))
    material_friction_soft = float(train_args.get("material_friction_soft", 0.15))
    material_softness_rigid = float(train_args.get("material_softness_rigid", 0.0))
    material_softness_soft = float(train_args.get("material_softness_soft", 1.0))

    pbd = meta.get("pbd", None)
    if isinstance(pbd, dict):
        friction = _first_numeric_value(pbd, "static_friction", "kinetic_friction", "friction")
        if friction is not None:
            dist_rigid = abs(float(friction) - material_friction_rigid)
            dist_soft = abs(float(friction) - material_friction_soft)
            return "soft" if dist_soft <= dist_rigid else "rigid"
        return "soft"

    material = meta.get("material", None)
    if isinstance(material, dict):
        softness = _first_numeric_value(material, "effective_softness", "softness")
        if softness is not None:
            midpoint = 0.5 * (material_softness_rigid + material_softness_soft)
            return "soft" if float(softness) >= midpoint else "rigid"
        friction = _first_numeric_value(material, "friction", "static_friction", "kinetic_friction")
        if friction is not None:
            dist_rigid = abs(float(friction) - material_friction_rigid)
            dist_soft = abs(float(friction) - material_friction_soft)
            return "soft" if dist_soft <= dist_rigid else "rigid"

    meta_path_l = meta_path.lower()
    if "soft" in meta_path_l:
        return "soft"
    if "rigid" in meta_path_l or "hard" in meta_path_l:
        return "rigid"
    return "rigid"


def _default_env_and_mat_material_tuple(mode: str, train_args: dict) -> np.ndarray:
    if str(mode) == "soft":
        return np.asarray(
            [
                float(train_args.get("material_softness_soft", 1.0)),
                float(train_args.get("material_friction_soft", 0.15)),
            ],
            dtype=np.float32,
        )
    return np.asarray(
        [
            float(train_args.get("material_softness_rigid", 0.0)),
            float(train_args.get("material_friction_rigid", 0.01)),
        ],
        dtype=np.float32,
    )


def _env_and_mat_object_material_row(obj: Any, default_row: np.ndarray) -> np.ndarray:
    row = np.asarray(default_row, dtype=np.float32).copy()
    if not isinstance(obj, dict):
        return row

    material_dict = obj.get("material", None)
    pbd_dict = obj.get("pbd", None)

    softness = _first_non_none(
        _first_numeric_value(material_dict, "effective_softness", "softness"),
        _first_numeric_value(obj, "effective_softness", "softness"),
    )
    friction = _first_non_none(
        _first_numeric_value(material_dict, "friction", "static_friction", "kinetic_friction"),
        _first_numeric_value(obj, "friction", "static_friction", "kinetic_friction"),
    )

    if softness is None and isinstance(pbd_dict, dict):
        softness = 1.0
    if friction is None and isinstance(pbd_dict, dict):
        friction = _first_numeric_value(pbd_dict, "static_friction", "kinetic_friction", "friction")

    if softness is not None:
        row[0] = float(softness)
    if friction is not None:
        row[1] = float(friction)
    return row


@dataclass(frozen=True)
class EnvAndMatConditioning:
    scene_cond: np.ndarray
    object_materials: np.ndarray


def _env_and_mat_conditioning_from_metadata(
    meta: dict,
    *,
    meta_path: str,
    max_num_objects: int,
    train_args: dict,
) -> EnvAndMatConditioning:
    bounds_min = np.asarray(meta.get("bounds_min", [-1.0, -1.0, -1.0]), dtype=np.float32)
    bounds_max = np.asarray(meta.get("bounds_max", [1.0, 1.0, 1.0]), dtype=np.float32)
    if bounds_min.shape != (3,) or bounds_max.shape != (3,):
        raise ValueError(f"bounds_min/bounds_max must be length 3 in {meta_path}")

    center = (0.5 * (bounds_min + bounds_max)).astype(np.float32)
    size = (bounds_max - bounds_min).astype(np.float32)

    gravity_z = float(meta.get("gravity_z", -9.81))
    wall_clearance = float(meta.get("wall_clearance", 0.0))
    ceiling_raw = meta.get("ceiling", None)
    ceiling = float(bounds_max[2] if ceiling_raw is None else ceiling_raw)

    scene_cond = np.asarray(
        [
            float(center[0]),
            float(center[1]),
            float(center[2]),
            float(size[0]),
            float(size[1]),
            float(size[2]),
            gravity_z,
            wall_clearance,
            ceiling,
        ],
        dtype=np.float32,
    )

    mode = _infer_env_and_mat_material_mode(meta, meta_path, train_args)
    default_row = _default_env_and_mat_material_tuple(mode, train_args)
    object_materials = np.zeros((int(max_num_objects) + 1, ENV_AND_MAT_OBJECT_MATERIAL_DIM), dtype=np.float32)
    objects = meta.get("objects", [])
    if not isinstance(objects, list):
        objects = []
    num_objects = min(len(objects), int(max_num_objects))
    for obj_idx in range(num_objects):
        object_materials[obj_idx] = _env_and_mat_object_material_row(objects[obj_idx], default_row)

    return EnvAndMatConditioning(
        scene_cond=scene_cond,
        object_materials=object_materials,
    )


def _material_vector_from_meta(meta: dict, obj: dict) -> np.ndarray:
    pbd_scene = meta.get("pbd") if isinstance(meta.get("pbd"), dict) else {}
    fem_scene = meta.get("fem") if isinstance(meta.get("fem"), dict) else {}
    sap_scene = meta.get("sap") if isinstance(meta.get("sap"), dict) else {}
    walls_scene = meta.get("walls") if isinstance(meta.get("walls"), dict) else {}

    pbd_obj = obj.get("pbd") if isinstance(obj.get("pbd"), dict) else {}
    fem_obj = obj.get("fem") if isinstance(obj.get("fem"), dict) else {}
    material_obj = obj.get("material") if isinstance(obj.get("material"), dict) else {}

    if pbd_scene or pbd_obj:
        pbd = _merged_dict(pbd_scene, pbd_obj, material_obj)
        rho = _safe_float(pbd.get("rho", obj.get("rho", 0.0)), 0.0)
        static_friction = _safe_float(pbd.get("static_friction", material_obj.get("static_friction", 0.0)), 0.0)
        kinetic_friction = _safe_float(
            pbd.get("kinetic_friction", material_obj.get("kinetic_friction", static_friction)),
            static_friction,
        )
        restitution = _safe_float(
            pbd.get("boundary_restitution", material_obj.get("restitution", walls_scene.get("restitution", 0.0))),
            0.0,
        )
        feat = np.asarray(
            [
                0.0,
                1.0,
                0.0,
                _log10_clamped(pbd.get("stretch_compliance"), default=0.0) * -1.0,
                _log10_clamped(pbd.get("bending_compliance"), default=0.0) * -1.0,
                _log10_clamped(pbd.get("volume_compliance"), default=0.0) * -1.0,
                _log10_clamped(rho, default=0.0),
                static_friction,
                kinetic_friction,
                restitution,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
    elif fem_scene or fem_obj:
        fem = _merged_dict(fem_scene, fem_obj, material_obj)
        rho = _safe_float(fem.get("rho", obj.get("rho", 0.0)), 0.0)
        friction = _safe_float(
            fem.get("obj_friction_mu", material_obj.get("obj_friction_mu", material_obj.get("friction", 0.0))),
            0.0,
        )
        restitution = _safe_float(
            material_obj.get("restitution", walls_scene.get("restitution", meta.get("restitution", 0.0))),
            0.0,
        )
        feat = np.asarray(
            [
                0.0,
                0.0,
                1.0,
                _log10_clamped(fem.get("E"), default=0.0),
                _log10_clamped(fem.get("hydroelastic_modulus"), default=0.0),
                0.0,
                _log10_clamped(rho, default=0.0),
                friction,
                friction,
                restitution,
                _safe_float(fem.get("nu", 0.0), 0.0),
                _log10_clamped(sap_scene.get("hydroelastic_stiffness"), default=0.0),
            ],
            dtype=np.float32,
        )
    else:
        rigid = _merged_dict(meta, material_obj)
        rho = _safe_float(rigid.get("rho", obj.get("rho", 0.0)), 0.0)
        friction = _safe_float(
            rigid.get(
                "obj_friction",
                rigid.get("static_friction", material_obj.get("friction", material_obj.get("static_friction", 0.0))),
            ),
            0.0,
        )
        kinetic_friction = _safe_float(rigid.get("kinetic_friction", friction), friction)
        restitution = _safe_float(
            rigid.get("restitution", walls_scene.get("restitution", material_obj.get("restitution", 0.0))),
            0.0,
        )
        feat = np.asarray(
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                _log10_clamped(rho, default=0.0),
                friction,
                kinetic_friction,
                restitution,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

    if feat.shape != (OBJECT_MATERIAL_DIM,):
        raise RuntimeError(f"Internal error: object_material feature shape mismatch {feat.shape}")
    return feat


def object_materials_from_metadata_dict(meta: dict, *, max_num_objects: int) -> np.ndarray:
    objects = meta.get("objects", None)
    if not isinstance(objects, list) or not objects:
        raise ValueError("metadata missing non-empty 'objects' list for object material parsing")
    if len(objects) > int(max_num_objects):
        raise ValueError(
            f"Scene has num_objects={len(objects)} but max_num_objects={int(max_num_objects)} while building materials"
        )

    out = np.zeros((int(max_num_objects) + 1, OBJECT_MATERIAL_DIM), dtype=np.float32)
    for obj_id, obj in enumerate(objects):
        if not isinstance(obj, dict):
            raise ValueError(f"metadata object entry must be dict, got {type(obj)}")
        out[int(obj_id)] = _material_vector_from_meta(meta, obj)
    return out


def _find_state_tensor_by_suffix(state_dict: dict, suffix: str) -> Optional[torch.Tensor]:
    for key, value in state_dict.items():
        if str(key).endswith(str(suffix)) and isinstance(value, torch.Tensor):
            return value
    return None


def _infer_conditioning_dims_from_state_dict(state_dict: dict) -> tuple[int, int, int, int]:
    num_scene_tokens = 0
    scene_cond_dim = 0
    scene_cond_embed_out_tokens = 0
    object_material_dim = 0

    scene_in = _find_state_tensor_by_suffix(state_dict, "scene_cond_embed.0.weight")
    scene_out = _find_state_tensor_by_suffix(state_dict, "scene_cond_embed.2.weight")
    hidden_size = int(scene_in.shape[0]) if isinstance(scene_in, torch.Tensor) and scene_in.ndim == 2 else 0
    if isinstance(scene_in, torch.Tensor) and scene_in.ndim == 2:
        scene_cond_dim = int(scene_in.shape[1])

    if hidden_size > 0 and isinstance(scene_out, torch.Tensor) and scene_out.ndim == 2 and int(scene_out.shape[0]) % int(hidden_size) == 0:
        scene_cond_embed_out_tokens = int(scene_out.shape[0]) // int(hidden_size)

    scene_token_base = _find_state_tensor_by_suffix(state_dict, "scene_token_base")
    if isinstance(scene_token_base, torch.Tensor) and scene_token_base.ndim == 3:
        num_scene_tokens = int(scene_token_base.shape[1])
    elif scene_cond_embed_out_tokens > 0:
        num_scene_tokens = int(scene_cond_embed_out_tokens)

    obj_mat = _find_state_tensor_by_suffix(state_dict, "object_material_embed.0.weight")
    if isinstance(obj_mat, torch.Tensor) and obj_mat.ndim == 2:
        object_material_dim = int(obj_mat.shape[1])

    return int(num_scene_tokens), int(scene_cond_dim), int(scene_cond_embed_out_tokens), int(object_material_dim)


def _match_last_dim(feat: np.ndarray, expected_dim: int) -> np.ndarray:
    feat = np.asarray(feat, dtype=np.float32)
    if expected_dim <= 0:
        return feat
    cur = int(feat.shape[-1])
    if cur == int(expected_dim):
        return feat
    if cur > int(expected_dim):
        return feat[..., : int(expected_dim)].astype(np.float32, copy=False)
    pad_shape = feat.shape[:-1] + (int(expected_dim) - cur,)
    pad = np.zeros(pad_shape, dtype=np.float32)
    return np.concatenate([feat, pad], axis=-1).astype(np.float32, copy=False)


def _parse_tuple3(value: object, *, name: str) -> Optional[Tuple[float, float, float]]:
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s.startswith("(") and s.endswith(")"):
            s = s[1:-1]
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) != 3:
            raise ValueError(f"{name} must have exactly 3 comma-separated values, got: {value}")
        vals = tuple(float(p) for p in parts)
    elif isinstance(value, (list, tuple, np.ndarray)):
        if len(value) != 3:
            raise ValueError(f"{name} must have exactly 3 values, got: {value}")
        vals = tuple(float(p) for p in value)
    else:
        raise ValueError(f"{name} must be a 3-tuple/list or comma-separated string, got type={type(value)}")
    return vals


def _resolve_norm_stats(train_args: dict, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    if (args.norm_mean is None) != (args.norm_std is None):
        raise ValueError("--norm_mean and --norm_std must be set together (both 3-tuples) or both omitted")
    cli_mean = tuple(float(v) for v in args.norm_mean) if args.norm_mean is not None else None
    cli_std = tuple(float(v) for v in args.norm_std) if args.norm_std is not None else None

    ckpt_mean = _parse_tuple3(train_args.get("norm_mean", None), name="checkpoint norm_mean")
    ckpt_std = _parse_tuple3(train_args.get("norm_std", None), name="checkpoint norm_std")
    if (ckpt_mean is None) != (ckpt_std is None):
        raise ValueError("Checkpoint has only one of norm_mean/norm_std; both are required together")

    mean = cli_mean if cli_mean is not None else (ckpt_mean if ckpt_mean is not None else (0.0, 0.0, 0.0))
    std = cli_std if cli_std is not None else (ckpt_std if ckpt_std is not None else (1.0, 1.0, 1.0))
    if any(float(v) <= 0.0 for v in std):
        raise ValueError(f"norm_std must be > 0 for every coordinate, got {std}")
    return np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)


def _normalize_positions(
    vertices: np.ndarray,
    *,
    coord_scale: float,
    coord_shift: float,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
) -> np.ndarray:
    x = (np.asarray(vertices, dtype=np.float32) - float(coord_shift)) / float(coord_scale)
    return (x - norm_mean) / norm_std


def _scene_box_center_half_extent(scene_cond: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scene_cond = np.asarray(scene_cond, dtype=np.float32).reshape(-1)
    if int(scene_cond.shape[0]) < 6:
        raise ValueError(f"scene_cond must have at least 6 values, got shape={scene_cond.shape}")
    center = scene_cond[:3].astype(np.float32, copy=False)
    size = np.maximum(scene_cond[3:6].astype(np.float32, copy=False), 1e-6)
    half_extent = 0.5 * size
    return center, half_extent


def _apply_scene_box_normalization(vertices: np.ndarray, *, scene_cond: np.ndarray) -> np.ndarray:
    center, half_extent = _scene_box_center_half_extent(scene_cond)
    return (np.asarray(vertices, dtype=np.float32) - center.reshape(1, 3)) / half_extent.reshape(1, 3)


def _apply_scene_box_velocity_normalization(vertices: np.ndarray, *, scene_cond: np.ndarray) -> np.ndarray:
    _, half_extent = _scene_box_center_half_extent(scene_cond)
    return np.asarray(vertices, dtype=np.float32) / half_extent.reshape(1, 3)


def _normalize_velocities(vertices: np.ndarray, *, coord_scale: float, norm_std: np.ndarray) -> np.ndarray:
    x = np.asarray(vertices, dtype=np.float32) / float(coord_scale)
    return x / norm_std


def _denormalize_positions(
    vertices: np.ndarray,
    *,
    coord_scale: float,
    coord_shift: float,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
) -> np.ndarray:
    x = np.asarray(vertices, dtype=np.float32) * norm_std + norm_mean
    return x * float(coord_scale) + float(coord_shift)


def _undo_scene_box_normalization(vertices: np.ndarray, *, scene_cond: np.ndarray) -> np.ndarray:
    center, half_extent = _scene_box_center_half_extent(scene_cond)
    return np.asarray(vertices, dtype=np.float32) * half_extent.reshape(1, 1, 3) + center.reshape(1, 1, 3)


def _parse_int_list(s: str) -> list[int]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    return [int(p) for p in parts] if parts else []


def _parse_path_list(spec: str) -> list[str]:
    spec = str(spec).strip()
    if not spec:
        return []
    if os.path.isfile(spec) and spec.lower().endswith(".txt"):
        out: list[str] = []
        with open(spec, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                out.append(line)
        return out
    return [p.strip() for p in spec.split(",") if p.strip()]


def _normalize_fps_points_np(
    fps_points: np.ndarray,
    *,
    coord_scale: float,
    coord_shift: float,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
) -> np.ndarray:
    out = fps_points.astype(np.float32, copy=False)
    out = (out - float(coord_shift)) / float(coord_scale)
    mean = norm_mean.reshape((1,) * (out.ndim - 1) + (3,))
    std = norm_std.reshape((1,) * (out.ndim - 1) + (3,))
    out = (out - mean) / std
    return out.astype(np.float32, copy=False)


def _candidate_fps_paths(
    *,
    fps_precomputed_root: str,
    cond_sample_dir: str,
    cond_data_root: str,
    rel_sample_dir: Optional[str],
) -> list[str]:
    root = os.path.abspath(os.path.expanduser(str(fps_precomputed_root)))
    candidates: list[str] = []
    if rel_sample_dir:
        candidates.append(os.path.join(root, f"{str(rel_sample_dir).strip('/')}.npz"))
    if cond_data_root:
        try:
            rel = os.path.relpath(os.path.abspath(cond_sample_dir), os.path.abspath(cond_data_root))
            if not rel.startswith(".."):
                candidates.append(os.path.join(root, f"{rel.replace(os.sep, '/')}.npz"))
        except ValueError:
            pass
    candidates.append(os.path.join(root, f"{Path(cond_sample_dir).name}.npz"))

    out: list[str] = []
    seen: set[str] = set()
    for path in candidates:
        path = os.path.abspath(os.path.expanduser(str(path)))
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out


def _load_ca_fps_inputs(
    *,
    fps_precomputed_root: str,
    fps_k: int,
    dynamic_anchor: bool,
    max_num_objects: int,
    pad_object_id: int,
    cond_sample_dir: str,
    cond_data_root: str,
    rel_sample_dir: Optional[str],
    infer_num_frames: int,
    coord_scale: float,
    coord_shift: float,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if int(fps_k) <= 0:
        raise ValueError(f"fps_k must be > 0 for FPS-conditioned inference, got {fps_k}")
    if not fps_precomputed_root:
        raise ValueError("FPS-conditioned checkpoint requires --fps_precomputed_root or fps_precomputed_root in checkpoint args.")

    candidates = _candidate_fps_paths(
        fps_precomputed_root=fps_precomputed_root,
        cond_sample_dir=cond_sample_dir,
        cond_data_root=cond_data_root,
        rel_sample_dir=rel_sample_dir,
    )
    fps_path = next((path for path in candidates if os.path.isfile(path)), "")
    if not fps_path:
        raise FileNotFoundError("Missing FPS precomputed file. Tried:\n  " + "\n  ".join(candidates))

    key_points = f"fps_points_k{int(fps_k)}"
    key_object_ids = f"fps_object_ids_k{int(fps_k)}"
    key_dynamic_points = f"fps_dynamic_points_k{int(fps_k)}"
    with np.load(fps_path, allow_pickle=False) as data:
        if key_points not in data:
            raise KeyError(f"Missing key '{key_points}' in FPS precomputed file: {fps_path}")
        if key_object_ids not in data:
            raise KeyError(f"Missing key '{key_object_ids}' in FPS precomputed file: {fps_path}")
        fps_points_static = data[key_points].astype(np.float32, copy=False)
        fps_object_ids_src = data[key_object_ids].astype(np.int64, copy=False)
        if bool(dynamic_anchor):
            if key_dynamic_points not in data:
                raise KeyError(f"Missing key '{key_dynamic_points}' in FPS precomputed file: {fps_path}")
            fps_points_src = data[key_dynamic_points].astype(np.float32, copy=False)
        else:
            fps_points_src = fps_points_static

    if fps_points_static.ndim != 2 or fps_points_static.shape[1] != 3:
        raise ValueError(f"Invalid {key_points} shape in {fps_path}: got {fps_points_static.shape}")
    if fps_object_ids_src.shape != (fps_points_static.shape[0],):
        raise ValueError(f"Invalid {key_object_ids} shape in {fps_path}: got {fps_object_ids_src.shape}")
    if bool(dynamic_anchor):
        if fps_points_src.ndim != 3 or fps_points_src.shape[1:] != fps_points_static.shape:
            raise ValueError(
                f"Invalid {key_dynamic_points} shape in {fps_path}: expected (F,{fps_points_static.shape[0]},3), got {fps_points_src.shape}"
            )
        if int(fps_points_src.shape[0]) < int(infer_num_frames):
            raise ValueError(
                f"{key_dynamic_points} has only {fps_points_src.shape[0]} frames but inference needs {infer_num_frames}: {fps_path}"
            )
        fps_points_src = fps_points_src[: int(infer_num_frames)]

    max_fps_tokens = int(max_num_objects) * int(fps_k)
    n = int(fps_points_static.shape[0])
    if n > max_fps_tokens:
        raise ValueError(f"FPS token count {n} exceeds capacity {max_fps_tokens}: {fps_path}")

    fps_points_norm = _normalize_fps_points_np(
        fps_points_src,
        coord_scale=coord_scale,
        coord_shift=coord_shift,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )
    if bool(dynamic_anchor):
        fps_points_out = np.zeros((int(infer_num_frames), max_fps_tokens, 3), dtype=np.float32)
        fps_points_out[:, :n, :] = fps_points_norm
    else:
        fps_points_out = np.zeros((max_fps_tokens, 3), dtype=np.float32)
        fps_points_out[:n, :] = fps_points_norm

    fps_mask_out = np.zeros((max_fps_tokens,), dtype=np.float32)
    fps_mask_out[:n] = 1.0
    fps_object_ids_out = np.full((max_fps_tokens,), int(pad_object_id), dtype=np.int64)
    fps_object_ids_out[:n] = fps_object_ids_src
    return fps_points_out, fps_mask_out, fps_object_ids_out, fps_path


def _list_cond_sample_dirs(data_root: str, *, metadata_filename: str) -> list[str]:
    """
    Finds sample directories under data_root that look like:
      <sample_dir>/{metadata_filename,meshes/*.obj}
    """
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(data_root):
        if metadata_filename not in filenames:
            continue
        # Heuristic: require a "meshes" directory with at least one .obj.
        meshes_dir = os.path.join(dirpath, "meshes")
        if not os.path.isdir(meshes_dir):
            continue
        try:
            has_obj = any(fn.lower().endswith(".obj") for fn in os.listdir(meshes_dir))
        except Exception:
            has_obj = False
        if has_obj:
            out.append(dirpath)
    out.sort()
    return out


def _pick_first_frame_obj(sample_dir: str) -> str:
    meshes_dir = os.path.join(sample_dir, "meshes")
    preferred = os.path.join(meshes_dir, "combined_frame_000.obj")
    if os.path.isfile(preferred):
        return preferred
    objs = sorted([fn for fn in os.listdir(meshes_dir) if fn.lower().endswith(".obj")])
    if not objs:
        raise FileNotFoundError(f"No .obj files found under: {meshes_dir}")
    return os.path.join(meshes_dir, objs[0])


def _rel_sample_dir_from_data_root(sample_dir: str, data_root: str) -> str:
    sample_dir_abs = os.path.abspath(os.path.expanduser(sample_dir))
    data_root_abs = os.path.abspath(os.path.expanduser(data_root))
    rel_sample_dir = os.path.relpath(sample_dir_abs, data_root_abs).replace("\\", "/").strip("/")
    if rel_sample_dir in ("", "."):
        rel_sample_dir = os.path.basename(sample_dir_abs.rstrip(os.sep))
    if not rel_sample_dir or rel_sample_dir.startswith(".."):
        raise ValueError(
            f"Conditioning sample dir must be inside --cond_data_root when mirroring output layout. "
            f"cond_sample_dir={sample_dir_abs} cond_data_root={data_root_abs}"
        )
    return rel_sample_dir


def _resolve_sample_out_dir(
    *,
    out_dir: str,
    sample_index: int,
    cond_sample_dir: str,
    cond_data_root: str,
    out_layout: str,
) -> tuple[str, Optional[str]]:
    if out_layout == "indexed":
        return os.path.join(out_dir, f"sample_{sample_index:03d}"), None
    if out_layout == "cond_relpath":
        if not cond_data_root:
            raise ValueError("--out_layout=cond_relpath requires --cond_data_root so relative sample paths are well-defined.")
        rel_sample_dir = _rel_sample_dir_from_data_root(cond_sample_dir, cond_data_root)
        return os.path.join(out_dir, rel_sample_dir), rel_sample_dir
    raise ValueError(f"Unknown --out_layout: {out_layout}")


def _fixed_limits_from_metadata_dict(meta: dict) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    bounds_min = np.asarray(meta.get("bounds_min", [-1.0, -1.0, -1.0]), dtype=np.float32)
    bounds_max = np.asarray(meta.get("bounds_max", [1.0, 1.0, 1.0]), dtype=np.float32)
    if bounds_min.shape != (3,) or bounds_max.shape != (3,):
        return ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
    return tuple((float(bounds_min[i]), float(bounds_max[i])) for i in range(3))  # type: ignore[return-value]


def _fixed_limits_from_cli(raw: str) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
    text = str(raw or "").strip()
    if not text:
        return None
    parts = [p for p in text.replace(";", ",").split(",") if p.strip()]
    if len(parts) != 6:
        raise ValueError(
            "--viz_fixed_limits must contain 6 comma-separated numbers: xmin,xmax,ymin,ymax,zmin,zmax; "
            f"got {raw!r}"
        )
    vals = [float(p.strip()) for p in parts]
    limits = ((vals[0], vals[1]), (vals[2], vals[3]), (vals[4], vals[5]))
    for lo, hi in limits:
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError(f"Invalid --viz_fixed_limits range: {raw!r}")
    return limits


def _gt_frame_paths(sample_dir: str) -> list[str]:
    meshes_dir = os.path.join(sample_dir, "meshes")
    if not os.path.isdir(meshes_dir):
        raise FileNotFoundError(f"Missing meshes directory: {meshes_dir}")
    out = sorted(
        [
            os.path.join(meshes_dir, name)
            for name in os.listdir(meshes_dir)
            if name.startswith("combined_frame_") and name.endswith(".obj")
        ]
    )
    if not out:
        raise FileNotFoundError(f"No combined_frame_*.obj files found under: {meshes_dir}")
    return out


def _load_gt_vertices(frame_paths: list[str], num_frames: int) -> np.ndarray:
    if len(frame_paths) < int(num_frames):
        raise ValueError(f"Need {num_frames} GT frames, found only {len(frame_paths)}")
    frames: list[np.ndarray] = []
    for path in frame_paths[: int(num_frames)]:
        verts, _ = load_obj_vertices_faces(path)
        frames.append(verts.astype(np.float32, copy=False))
    return np.stack(frames, axis=0).astype(np.float32, copy=False)


def _shaded_facecolors(v: np.ndarray, f: np.ndarray, base_color: Tuple[float, float, float, float]) -> np.ndarray:
    tris = v[f]
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    light = LIGHT_DIRECTION / np.linalg.norm(LIGHT_DIRECTION)
    intensity = 0.42 + 0.58 * np.clip(normals @ light, 0.0, 1.0)
    base = np.asarray(base_color, dtype=np.float32)
    facecolors = np.empty((f.shape[0], 4), dtype=np.float32)
    facecolors[:, :3] = np.clip(base[:3][None, :] * intensity[:, None] + 0.10 * (1.0 - intensity[:, None]), 0.0, 1.0)
    facecolors[:, 3] = base[3]
    return facecolors


def _color_for_object(
    index: int,
    object_name: str | None,
    colors: list[Tuple[float, float, float, float]],
) -> Tuple[float, float, float, float]:
    name = str(object_name or "").lower()
    for pattern, color in NAMED_COLORS.items():
        if pattern in name:
            return color
    return colors[int(index) % len(colors)]


def _render_multiobj_frame(
    vertices_by_obj: list[np.ndarray],
    faces_by_obj: list[np.ndarray],
    *,
    colors: list[Tuple[float, float, float, float]],
    fixed_limits: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    elev: float,
    azim: float,
    dpi: int = 150,
    title: str = "",
    object_names: list[str] | None = None,
) -> np.ndarray:
    fig = plt.figure(figsize=(5.4, 5.4), dpi=dpi, facecolor="#f7f8fb")
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.set_facecolor("#f7f8fb")

    for i, (v, f) in enumerate(zip(vertices_by_obj, faces_by_obj)):
        v = np.asarray(v, dtype=np.float32)
        f = np.asarray(f, dtype=np.int64)
        if v.size == 0 or f.size == 0:
            continue
        tris = v[f]
        object_name = object_names[i] if object_names is not None and i < len(object_names) else None
        facecolors = _shaded_facecolors(v, f, _color_for_object(i, object_name, colors))
        poly = Poly3DCollection(
            tris,
            facecolors=facecolors,
            edgecolors=MESH_EDGE_COLOR,
            linewidths=0.28,
            alpha=0.96,
            antialiased=True,
        )
        ax.add_collection3d(poly)

    (x_min, x_max), (y_min, y_max), (z_min, z_max) = fixed_limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=float(elev), azim=float(azim))
    try:
        ax.set_proj_type("persp", focal_length=0.85)
    except TypeError:
        ax.set_proj_type("persp")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.tick_params(axis="both", which="major", labelsize=7, colors="#667085", pad=1)
    ax.grid(True, linestyle="-", linewidth=0.45, alpha=0.22)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.95, 0.96, 0.98, 0.72))
        axis.pane.set_edgecolor((0.78, 0.81, 0.86, 0.45))
    if title:
        ax.set_title(str(title), fontsize=13, fontweight="bold", color="#111827", pad=10)

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.10, top=0.93 if title else 0.99)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    img = img[:, :, :3]
    plt.close(fig)
    return img


def _compose_side_by_side(
    left: np.ndarray,
    right: np.ndarray,
    *,
    sample_title: str,
    subset_label: str,
    dpi: int,
) -> np.ndarray:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=dpi)
    title = str(sample_title).strip()
    subset = str(subset_label).strip()
    if subset:
        title = f"{subset} | {title}" if title else subset
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    for ax, img, panel_title in zip(axes, [left, right], ["Ground Truth", "Inference"]):
        ax.imshow(img)
        ax.set_title(panel_title, fontsize=11)
        ax.axis("off")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95] if title else None)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    out = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
    plt.close(fig)
    return out


def _save_animation(
    frames: list[np.ndarray],
    *,
    out_gif: Optional[str],
    out_mp4: Optional[str],
    fps: int,
) -> None:
    if out_gif is None and out_mp4 is None:
        return
    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception as e:
        raise RuntimeError("Saving GIF/MP4 requires imageio. Install with: pip install imageio imageio-ffmpeg") from e

    if out_gif is not None:
        imageio.mimsave(out_gif, frames, duration=1.0 / max(1, fps), loop=0)

    if out_mp4 is not None:
        try:
            with imageio.get_writer(out_mp4, fps=max(1, fps), codec="libx264", quality=8) as w:
                for fr in frames:
                    w.append_data(fr)
        except Exception as e:
            raise RuntimeError("MP4 saving failed. You likely need ffmpeg support. Try: pip install imageio-ffmpeg") from e


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("PhysFormer multi-object vertex-token inference (spacetime, AltObj)")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--num_generations_per_sample", type=int, default=1)
    p.add_argument("--labels", type=str, default="0")
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--no_ema", action="store_false", dest="use_ema")
    p.set_defaults(use_ema=True)
    p.add_argument("--denorm", action="store_true")
    p.add_argument("--no_denorm", action="store_false", dest="denorm")
    p.set_defaults(denorm=True)
    p.add_argument(
        "--norm_mean",
        type=float,
        nargs=3,
        default=None,
        metavar=("MEAN_X", "MEAN_Y", "MEAN_Z"),
        help="Override checkpoint per-coordinate normalization mean (used for conditioning + denormalization).",
    )
    p.add_argument(
        "--norm_std",
        type=float,
        nargs=3,
        default=None,
        metavar=("STD_X", "STD_Y", "STD_Z"),
        help="Override checkpoint per-coordinate normalization std (used for conditioning + denormalization).",
    )

    # conditioning: sample dirs (preferred), or data_root selection
    p.add_argument("--cond_sample_dir", type=str, default="", help="Single <sample_dir> for conditioning (reused for all samples).")
    p.add_argument(
        "--cond_sample_dirs",
        type=str,
        default="",
        help="Multiple <sample_dir> entries (one per sample). Provide comma-separated list or a .txt file.",
    )
    p.add_argument(
        "--conditioned_metadata_jsonl",
        type=str,
        default="",
        help=(
            "Optional JSONL file with one material-conditioned metadata object per selected sample. "
            "When set, the original sample directory is still used for meshes and velocities."
        ),
    )
    p.add_argument("--cond_data_root", type=str, default="", help="Scan this root for sample dirs (must contain metadata.json + meshes/*.obj).")
    p.add_argument("--cond_indices", type=str, default="", help="Comma-separated indices into the sorted cond_sample_dirs list.")
    p.add_argument("--cond_random", action="store_true", help="Randomly select conditioning samples from --cond_data_root.")
    p.add_argument("--cond_seed", type=int, default=0)

    p.add_argument(
        "--cond_first_frame_velocity",
        action="store_true",
        help="Concatenate first-frame per-vertex velocities to conditioning (pos+vel).",
    )
    p.add_argument("--no_cond_first_frame_velocity", action="store_false", dest="cond_first_frame_velocity")
    p.set_defaults(cond_first_frame_velocity=True)
    p.add_argument("--velocity_dirname", type=str, default="vertex_velocities")

    # multi-object mapping
    p.add_argument("--mesh_vertex_count_json", type=str, default="", help="Override mesh->vertex_count JSON path.")
    p.add_argument("--max_num_objects", type=int, default=0, help="Override max_num_objects from the checkpoint (0 = use ckpt).")
    p.add_argument(
        "--max_vertices",
        type=int,
        default=0,
        help=(
            "Override checkpoint max_vertices for dynamic-vertex inference (0 = use ckpt). "
            "This only changes the runtime model cap; it does not change checkpoint training coverage."
        ),
    )
    p.add_argument("--metadata_filename", type=str, default="metadata.json")
    p.add_argument("--fps_precomputed_root", type=str, default="", help="Override FPS precomputed root for CA-FPS checkpoints.")
    p.add_argument("--fps_k", type=int, default=0, help="Override FPS anchors per object for CA-FPS checkpoints (0 = use ckpt).")
    p.add_argument(
        "--dynamic_anchor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether a CA-FPS checkpoint uses dynamic per-frame anchors.",
    )

    # viz / output
    p.add_argument(
        "--out_layout",
        type=str,
        default="indexed",
        choices=["indexed", "cond_relpath"],
        help=(
            "Output directory layout under --out_dir. "
            "'indexed' uses sample_{i:03d}; 'cond_relpath' mirrors the selected sample's relative path "
            "under --cond_data_root, e.g. <out_dir>/1_obj/sample_000008."
        ),
    )
    p.add_argument("--save_gif", action="store_true")
    p.add_argument("--save_mp4", action="store_true")
    p.add_argument("--save_gt_gif", action="store_true")
    p.add_argument("--save_gt_mp4", action="store_true")
    p.add_argument("--save_compare_gif", action="store_true")
    p.add_argument("--save_compare_mp4", action="store_true")
    p.add_argument(
        "--save_scene_metadata",
        action="store_true",
        help="Write debug/provenance scene_multiobj.json into each output sample directory.",
    )
    p.add_argument(
        "--compare_out_name",
        type=str,
        default="traj_compare_gt_vs_infer",
        help="Base output name for GT-vs-inference side-by-side renders (without extension).",
    )
    p.add_argument("--compare_subset_label", type=str, default="", help="Optional label such as TEST, SEEN, or UNSEEN.")
    p.add_argument("--compare_render_dpi", type=int, default=150)
    p.add_argument("--compare_compose_dpi", type=int, default=140)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--viz_elev", type=float, default=30.0)
    p.add_argument("--viz_azim", type=float, default=-45.0)
    p.add_argument(
        "--viz_fixed_limits",
        type=str,
        default="",
        help="Optional Matplotlib axis limits as xmin,xmax,ymin,ymax,zmin,zmax. Overrides metadata bounds for rendering only.",
    )
    p.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing outputs under --out_dir. Use --no-overwrite to resume/skip samples that already have vertices.npz.",
    )

    # sampling overrides
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16", "fp16"])
    p.add_argument("--sampling_method", type=str, default="", choices=["", "euler", "heun"])
    p.add_argument("--num_sampling_steps", type=int, default=0)
    p.add_argument("--cfg_scale", type=float, default=None)
    p.add_argument("--cfg_interval_min", type=float, default=None)
    p.add_argument("--cfg_interval_max", type=float, default=None)
    p.add_argument(
        "--vel_cfg_scale",
        type=float,
        default=None,
        help="Velocity-only CFG scale on first-frame conditioning (requires pos+vel conditioning, i.e. --cond_first_frame_velocity).",
    )
    p.add_argument(
        "--vel_cfg_interval_min",
        type=float,
        default=None,
        help="Velocity-only CFG lower interval bound in t (default: 0.0).",
    )
    p.add_argument(
        "--vel_cfg_interval_max",
        type=float,
        default=None,
        help="Velocity-only CFG upper interval bound in t (default: 1.0).",
    )
    p.add_argument("--infer_num_frames", type=int, default=0)
    p.add_argument("--infer_num_vertices", type=int, default=0)
    p.add_argument(
        "--auto_infer_num_vertices",
        action="store_true",
        help="Auto-set infer_num_vertices per sample to the number of vertices in the conditioning first-frame OBJ.",
    )
    p.add_argument(
        "--env_and_mat",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable the scene/environment and per-object material inference path added for env+mat-conditioned "
            "checkpoints. Default: false, which preserves the original multi-object inference behavior."
        ),
    )
    p.add_argument(
        "--material_mode",
        type=str,
        default="",
        choices=["", "auto", "rigid", "soft"],
        help="Override checkpoint material_mode for env/material conditioning. Use 'soft' to force soft defaults.",
    )
    p.add_argument("--verbose", action="store_true")
    return p


@torch.no_grad()
def main() -> None:
    args = build_argparser().parse_args()
    if int(args.num_samples) <= 0:
        raise ValueError(f"--num_samples must be >= 1, got {args.num_samples}")
    if int(args.num_generations_per_sample) <= 0:
        raise ValueError(f"--num_generations_per_sample must be >= 1, got {args.num_generations_per_sample}")
    if bool(args.auto_infer_num_vertices) and int(args.infer_num_vertices) > 0:
        raise ValueError("--auto_infer_num_vertices cannot be combined with an explicit --infer_num_vertices.")
    args.out_dir = os.path.abspath(os.path.expanduser(str(args.out_dir)))
    if args.cond_data_root:
        args.cond_data_root = os.path.abspath(os.path.expanduser(str(args.cond_data_root)))
    os.makedirs(args.out_dir, exist_ok=True)

    def vlog(msg: str) -> None:
        if args.verbose:
            print(msg, flush=True)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    train_args = ckpt.get("args", {})
    if str(args.material_mode).strip():
        if not isinstance(train_args, dict):
            train_args = {}
            ckpt["args"] = train_args
        train_args["material_mode"] = str(args.material_mode).strip()
    delta_to_first_frame = bool(train_args.get("delta_to_first_frame", False))
    coord_scale = float(train_args.get("coord_scale", 1.0))
    coord_shift = float(train_args.get("coord_shift", 0.0))
    norm_mean, norm_std = _resolve_norm_stats(train_args, args)

    ckpt_max_num_objects = int(train_args.get("max_num_objects", 3) or 0)
    if ckpt_max_num_objects <= 0:
        ckpt_max_num_objects = 3
    max_num_objects = int(args.max_num_objects) if int(args.max_num_objects) > 0 else int(ckpt_max_num_objects)
    if max_num_objects <= 0:
        raise ValueError(f"Invalid max_num_objects={max_num_objects}")
    if int(args.max_num_objects) > 0 and int(max_num_objects) < int(ckpt_max_num_objects):
        raise ValueError(
            f"--max_num_objects={max_num_objects} is smaller than checkpoint max_num_objects={ckpt_max_num_objects}."
        )
    if int(max_num_objects) > int(ckpt_max_num_objects):
        init_std = float(os.environ.get("JMT4D_OBJ_EMBED_INIT_STD", "0.02"))
        changed = _maybe_expand_ckpt(ckpt, target_max_num_objects=int(max_num_objects), init_std=float(init_std))
        if bool(changed):
            vlog(
                f"[INFO] Expanded object_id_embed rows in memory: {ckpt_max_num_objects} -> {max_num_objects} "
                f"(init_std={init_std:.6g})."
            )
        train_args_any = ckpt.get("args", {})
        train_args = train_args_any if isinstance(train_args_any, dict) else {}
    pad_object_id = max_num_objects

    mesh_vertex_count_json = str(args.mesh_vertex_count_json).strip() or str(train_args.get("mesh_vertex_count_json", "")).strip()
    if not mesh_vertex_count_json:
        mesh_vertex_count_json = default_vertex_count_json_path()
    if not os.path.isfile(mesh_vertex_count_json):
        raise FileNotFoundError(
            f"mesh_vertex_count_json not found: {mesh_vertex_count_json}. "
            "Use the packaged official-demo config or pass --mesh_vertex_count_json."
        )
    vertex_counts = load_mesh_vertex_counts(mesh_vertex_count_json)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    vlog(f"device={device} cuda_available={torch.cuda.is_available()}")
    use_amp = args.amp != "none"
    if device.type == "cpu":
        amp_dtype = torch.bfloat16 if args.amp == "bf16" else None
        use_amp = amp_dtype is not None
    else:
        if args.amp == "bf16":
            is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
            if callable(is_bf16_supported) and not bool(is_bf16_supported()):
                vlog("[WARN] --amp bf16 is not supported on this GPU; switching to --amp fp16.")
                args.amp = "fp16"
        amp_dtype = torch.bfloat16 if args.amp == "bf16" else (torch.float16 if args.amp == "fp16" else None)
    vlog(f"amp={args.amp} enabled={use_amp}")
    vlog(
        "coord_norm: "
        f"coord_scale={coord_scale} "
        f"coord_shift={coord_shift} "
        f"norm_mean={[float(v) for v in norm_mean.tolist()]} "
        f"norm_std={[float(v) for v in norm_std.tolist()]}"
    )

    state_dict_to_load = dict(ckpt["ema"]["shadow"] if args.use_ema and "ema" in ckpt else ckpt["model"])
    if bool(args.env_and_mat):
        ckpt_num_scene_tokens, ckpt_scene_cond_dim, ckpt_scene_cond_embed_out_tokens, ckpt_object_material_dim = _infer_conditioning_dims_from_state_dict(
            state_dict_to_load
        )

        num_scene_tokens = int(ckpt_num_scene_tokens) if int(ckpt_num_scene_tokens) > 0 else int(train_args.get("num_scene_tokens", 0))
        cond_scene = bool(train_args.get("cond_scene", False)) or int(num_scene_tokens) > 0 or int(ckpt_scene_cond_dim) > 0
        cond_object_material = bool(train_args.get("cond_object_material", False)) or int(ckpt_object_material_dim) > 0
        normalize_to_scene_box = bool(train_args.get("normalize_to_scene_box", False))
        scene_cond_dim = int(ckpt_scene_cond_dim) if int(ckpt_scene_cond_dim) > 0 else (int(SCENE_COND_DIM) if bool(cond_scene) else 0)
        scene_cond_embed_out_tokens = (
            int(ckpt_scene_cond_embed_out_tokens)
            if int(ckpt_scene_cond_embed_out_tokens) > 0
            else (int(num_scene_tokens) if int(num_scene_tokens) > 0 else 0)
        )
        object_material_dim = (
            int(ckpt_object_material_dim) if int(ckpt_object_material_dim) > 0 else (int(OBJECT_MATERIAL_DIM) if bool(cond_object_material) else 0)
        )
    else:
        num_scene_tokens = 0
        cond_scene = False
        cond_object_material = False
        normalize_to_scene_box = False
        scene_cond_dim = 0
        scene_cond_embed_out_tokens = 0
        object_material_dim = 0

    model_name = str(train_args.get("model", "MeshVideoDiT-ST-Vert-B-MultiObj"))
    is_ca_fps_model = model_name.endswith("-CA-FPS")
    is_rwonce_invarobj_topoonce_model = "SummaryRWOnceInvarObjTopoOnce" in model_name
    is_rwonce_invarobj_topo_model = (
        "SummaryRWOnceInvarObjTopo" in model_name and not bool(is_rwonce_invarobj_topoonce_model)
    )
    is_rwonce_invarobj_model = "SummaryRWOnceInvarObj" in model_name and not (
        bool(is_rwonce_invarobj_topo_model) or bool(is_rwonce_invarobj_topoonce_model)
    )
    is_summary_rwonce_model = "SummaryRWOnce" in model_name and not (
        bool(is_rwonce_invarobj_model)
        or bool(is_rwonce_invarobj_topo_model)
        or bool(is_rwonce_invarobj_topoonce_model)
    )
    is_summary_rw_model = "SummaryRW" in model_name
    uses_fps_inputs = bool(is_ca_fps_model or is_rwonce_invarobj_topo_model or is_rwonce_invarobj_topoonce_model)
    fps_k = int(args.fps_k) if int(args.fps_k) > 0 else int(train_args.get("fps_k", 0) or train_args.get("shape_fps_k", 0) or 0)
    dynamic_anchor = _resolve_optional_bool(args.dynamic_anchor, bool(train_args.get("dynamic_anchor", False)))
    fps_precomputed_root = str(args.fps_precomputed_root).strip() or str(train_args.get("fps_precomputed_root", "")).strip()

    model_kwargs = {
        "use_rope": bool(train_args.get("use_rope", True)),
        "num_register_tokens": int(train_args.get("num_register_tokens", 16)),
        "max_frames": int(train_args.get("max_frames", 128)),
        "max_vertices": int(args.max_vertices) if int(args.max_vertices) > 0 else int(train_args.get("max_vertices", 8192)),
        "attn_drop": float(train_args.get("attn_drop", 0.0)),
        "proj_drop": float(train_args.get("proj_drop", 0.0)),
        "max_num_objects": int(max_num_objects),
        "use_object_id_embed": bool(
            is_summary_rw_model
            and not (is_rwonce_invarobj_model or is_rwonce_invarobj_topo_model or is_rwonce_invarobj_topoonce_model)
        ),
        "num_scene_tokens": int(num_scene_tokens),
        "scene_cond_dim": int(scene_cond_dim),
        "scene_cond_embed_out_tokens": int(scene_cond_embed_out_tokens),
        "object_material_dim": int(object_material_dim),
    }
    if bool(is_summary_rw_model):
        for key in ("num_scene_tokens", "scene_cond_dim", "scene_cond_embed_out_tokens", "object_material_dim"):
            model_kwargs.pop(key, None)
    if bool(is_ca_fps_model):
        if int(fps_k) <= 0:
            raise ValueError("CA-FPS checkpoint is missing fps_k; pass --fps_k.")
        model_kwargs["max_fps_tokens"] = int(max_num_objects) * int(fps_k)
        model_kwargs["dynamic_anchor"] = bool(dynamic_anchor)
    if bool(is_rwonce_invarobj_topo_model or is_rwonce_invarobj_topoonce_model):
        if int(fps_k) <= 0:
            raise ValueError("RWOnceInvarObjTopo checkpoint is missing shape_fps_k; pass --fps_k.")
        model_kwargs["shape_tokens_per_object"] = int(train_args.get("shape_tokens_per_object", 32))
        model_kwargs["shape_encoder_layers"] = int(train_args.get("shape_encoder_layers", 2))
        model_kwargs["shape_point_fourier_dim"] = int(train_args.get("shape_point_fourier_dim", 48))
        model_kwargs["require_shape_tokens"] = True
    if bool(is_summary_rw_model):
        model_kwargs["vertex_read_rope"] = bool(train_args.get("vertex_read_rope", True))
    sampling_method = str(train_args.get("sampling_method", "heun"))
    if args.sampling_method:
        sampling_method = str(args.sampling_method)
    num_sampling_steps = int(train_args.get("num_sampling_steps", 50))
    if int(args.num_sampling_steps) > 0:
        num_sampling_steps = int(args.num_sampling_steps)
    cfg_scale = float(train_args.get("cfg", 1.0))
    if args.cfg_scale is not None:
        cfg_scale = float(args.cfg_scale)
    cfg_interval_min = float(train_args.get("cfg_interval_min", 0.0))
    if args.cfg_interval_min is not None:
        cfg_interval_min = float(args.cfg_interval_min)
    cfg_interval_max = float(train_args.get("cfg_interval_max", 1.0))
    if args.cfg_interval_max is not None:
        cfg_interval_max = float(args.cfg_interval_max)
    vel_cfg_scale = 1.0
    if args.vel_cfg_scale is not None:
        vel_cfg_scale = float(args.vel_cfg_scale)
    vel_cfg_interval_min = 0.0
    if args.vel_cfg_interval_min is not None:
        vel_cfg_interval_min = float(args.vel_cfg_interval_min)
    vel_cfg_interval_max = 1.0
    if args.vel_cfg_interval_max is not None:
        vel_cfg_interval_max = float(args.vel_cfg_interval_max)
    diff_cfg = DiffusionConfig(
        P_mean=float(train_args.get("P_mean", -0.8)),
        P_std=float(train_args.get("P_std", 0.8)),
        t_eps=float(train_args.get("t_eps", 5e-2)),
        noise_scale=float(train_args.get("noise_scale", 1.0)),
        label_drop_prob=float(train_args.get("label_drop_prob", 0.1)),
        cfg_scale=cfg_scale,
        cfg_interval_min=cfg_interval_min,
        cfg_interval_max=cfg_interval_max,
        vel_cfg_scale=vel_cfg_scale,
        vel_cfg_interval_min=vel_cfg_interval_min,
        vel_cfg_interval_max=vel_cfg_interval_max,
        sampling_method=sampling_method,
        num_sampling_steps=num_sampling_steps,
    )
    vlog(
        "ckpt_cfg: "
        f"model={model_name} "
        f"num_frames={int(train_args.get('num_frames', 32))} "
        f"num_vertices={int(train_args.get('num_vertices', 1024))} "
        f"max_vertices={int(model_kwargs['max_vertices'])} "
        f"env_and_mat={bool(args.env_and_mat)} "
        f"max_num_objects={max_num_objects} "
        f"fps_k={fps_k if uses_fps_inputs else 0} "
        f"dynamic_anchor={bool(dynamic_anchor) if is_ca_fps_model else False} "
        f"num_scene_tokens={num_scene_tokens} "
        f"scene_cond_dim={scene_cond_dim} "
        f"scene_cond_embed_out_tokens={scene_cond_embed_out_tokens} "
        f"cond_scene={cond_scene} "
        f"object_material_dim={object_material_dim} "
        f"cond_object_material={cond_object_material} "
        f"normalize_to_scene_box={normalize_to_scene_box} "
        f"sampling_method={diff_cfg.sampling_method} "
        f"num_sampling_steps={diff_cfg.num_sampling_steps} "
        f"cfg_scale={diff_cfg.cfg_scale}"
    )
    vlog(f"delta_to_first_frame={delta_to_first_frame}")

    if bool(
        is_ca_fps_model
        or is_rwonce_invarobj_topoonce_model
        or is_rwonce_invarobj_topo_model
        or is_rwonce_invarobj_model
        or is_summary_rwonce_model
        or is_summary_rw_model
    ):
        raise NotImplementedError(
            "This publication export contains the plain MultiObj-AltObj inference path used by "
            "checkpoint-best.pt. Re-export the full PhysFormer package for CA-FPS/RW checkpoint families."
        )
    denoiser_cls = DenoiserMeshVideoMultiObjAltObj
    model = denoiser_cls(
        model_name=model_name,
        num_frames=int(train_args.get("num_frames", 32)),
        num_vertices=int(train_args.get("num_vertices", 1024)),
        num_classes=int(train_args.get("num_classes", 1)),
        model_kwargs=model_kwargs,
        diffusion=diff_cfg,
    ).to(device)

    renamed_x_embed_cond_keys = _rename_legacy_x_embed_cond_keys(state_dict_to_load, model)
    if int(renamed_x_embed_cond_keys) > 0:
        vlog(
            "[INFO] Renamed legacy x_embed_cond checkpoint keys to cond_x_embedder "
            f"({int(renamed_x_embed_cond_keys)} tensors)."
        )
    added_cond_x_embedder_keys = _add_cond_x_embedder_keys_from_x_embedder(state_dict_to_load, model)
    if int(added_cond_x_embedder_keys) > 0:
        vlog(
            "[INFO] Initialized missing cond_x_embedder checkpoint keys from x_embedder "
            f"({int(added_cond_x_embedder_keys)} tensors)."
        )

    incompat = model.load_state_dict(state_dict_to_load, strict=False)
    allowed_missing = {"net.scene_token_base"} if bool(args.env_and_mat) else set()
    allowed_unexpected_prefixes = (
        ("net.scene_token_mlp.", "net.scene_token_embed.")
        if bool(args.env_and_mat)
        else ()
    )
    missing_keys = set(incompat.missing_keys)
    unexpected_keys = set(incompat.unexpected_keys)
    bad_missing = sorted(k for k in missing_keys if k not in allowed_missing)
    bad_unexpected = sorted(
        k for k in unexpected_keys if not any(str(k).startswith(pref) for pref in allowed_unexpected_prefixes)
    )
    if bad_missing or bad_unexpected:
        if (not bool(args.env_and_mat)) and any(
            ("scene_" in str(k)) or ("material" in str(k)) for k in (list(bad_missing) + list(bad_unexpected))
        ):
            raise RuntimeError(
                "This checkpoint appears to use environment/material conditioning, but inference was run without "
                "--env_and_mat. Re-run with --env_and_mat to enable the scene/material inference path."
            )
        raise RuntimeError(
            "Checkpoint/model state_dict mismatch. "
            f"missing_keys={bad_missing} unexpected_keys={bad_unexpected}"
        )
    model.eval()

    infer_num_frames = int(args.infer_num_frames) if int(args.infer_num_frames) > 0 else int(model.net.num_frames)
    infer_num_vertices_default = (
        int(args.infer_num_vertices) if int(args.infer_num_vertices) > 0 else int(model.net.num_vertices)
    )
    max_vertices = int(getattr(getattr(model, "net", None), "cfg", None).max_vertices) if hasattr(model.net, "cfg") else None

    labels = _parse_labels(args.labels, args.num_samples).to(device)
    vlog(f"labels={labels.detach().cpu().tolist()}")

    # Resolve conditioning sample dirs.
    cond_sample_dirs: list[str] = []
    if args.cond_sample_dir:
        cond_sample_dirs = [str(args.cond_sample_dir)] * int(args.num_samples)
    elif args.cond_sample_dirs:
        cond_sample_dirs = _parse_path_list(args.cond_sample_dirs)
        if len(cond_sample_dirs) == 1:
            cond_sample_dirs = cond_sample_dirs * int(args.num_samples)
        if len(cond_sample_dirs) != int(args.num_samples):
            raise ValueError("--cond_sample_dirs must contain 1 entry or exactly --num_samples entries.")
    elif args.cond_data_root:
        all_dirs = _list_cond_sample_dirs(str(args.cond_data_root), metadata_filename=str(args.metadata_filename))
        if not all_dirs:
            raise ValueError(f"No conditioning sample dirs found under: {args.cond_data_root}")
        if args.cond_random:
            rng = np.random.RandomState(int(args.cond_seed))
            picks = rng.choice(len(all_dirs), size=int(args.num_samples), replace=False if len(all_dirs) >= int(args.num_samples) else True)
            cond_sample_dirs = [all_dirs[int(i)] for i in picks.tolist()]
        else:
            idxs = _parse_int_list(args.cond_indices)
            if not idxs:
                idxs = list(range(int(args.num_samples)))
            if len(idxs) != int(args.num_samples):
                raise ValueError("--cond_indices must have length --num_samples (or be omitted).")
            for i in idxs:
                if i < 0 or i >= len(all_dirs):
                    raise IndexError(f"cond_index {i} out of range (0..{len(all_dirs)-1})")
            cond_sample_dirs = [all_dirs[int(i)] for i in idxs]
    else:
        raise ValueError("Must provide conditioning via --cond_sample_dir, --cond_sample_dirs, or --cond_data_root.")

    cond_sample_dirs = [os.path.abspath(os.path.expanduser(str(p))) for p in cond_sample_dirs]
    conditioned_metadata_records = _load_conditioned_metadata_jsonl(
        str(args.conditioned_metadata_jsonl),
        expected_count=int(args.num_samples),
    )
    sample_dirs: list[str] = []
    rel_sample_dirs: list[Optional[str]] = []
    seen_sample_dirs: dict[str, int] = {}
    for i, cond_sample_dir in enumerate(cond_sample_dirs):
        sample_dir, rel_sample_dir = _resolve_sample_out_dir(
            out_dir=str(args.out_dir),
            sample_index=i,
            cond_sample_dir=cond_sample_dir,
            cond_data_root=str(args.cond_data_root),
            out_layout=str(args.out_layout),
        )
        sample_dir = os.path.abspath(sample_dir)
        prev_i = seen_sample_dirs.get(sample_dir)
        if prev_i is not None:
            raise ValueError(
                f"--out_layout={args.out_layout} resolved the same output directory for multiple samples: "
                f"sample[{prev_i}] and sample[{i}] -> {sample_dir}. "
                "Use --out_layout indexed or ensure each selected conditioning sample maps to a unique relative path."
            )
        seen_sample_dirs[sample_dir] = i
        sample_dirs.append(sample_dir)
        rel_sample_dirs.append(rel_sample_dir)

    # Colors per object (cycled).
    colors = [
        (0.86, 0.24, 0.20, 1.0),
        (0.20, 0.64, 0.42, 1.0),
        (0.20, 0.44, 0.86, 1.0),
        (0.92, 0.67, 0.22, 1.0),
        (0.62, 0.32, 0.76, 1.0),
    ]

    sample_range = range(int(args.num_samples))
    if not bool(args.verbose):
        sample_range = tqdm(sample_range, desc="samples", unit="sample")

    for i in sample_range:
        sample_dir = sample_dirs[i]
        cond_sample_dir = cond_sample_dirs[i]
        rel_sample_dir = rel_sample_dirs[i]
        if not bool(args.overwrite):
            expected_npzs: list[str] = []
            if int(args.num_generations_per_sample) == 1:
                expected_npzs = [os.path.join(sample_dir, "vertices.npz")]
            else:
                expected_npzs = [
                    os.path.join(sample_dir, f"sample_{repeat_idx:02d}", "vertices.npz")
                    for repeat_idx in range(int(args.num_generations_per_sample))
                ]
            if expected_npzs and all(os.path.isfile(p) for p in expected_npzs):
                tqdm.write(f"[SKIP] exists: {sample_dir} (all vertices.npz)")  # type: ignore[attr-defined]
                continue

        os.makedirs(sample_dir, exist_ok=True)

        meta_path = os.path.join(cond_sample_dir, str(args.metadata_filename))
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"Missing metadata file: {meta_path}")
        if conditioned_metadata_records is None:
            meta = load_metadata(meta_path)
            meta_source = meta_path
        else:
            meta = conditioned_metadata_records[i]
            meta_source = f"{args.conditioned_metadata_jsonl}#{i}"
        first_obj_path = _pick_first_frame_obj(cond_sample_dir)
        cond_verts_full = _load_obj_vertices_only(first_obj_path)  # (V_full,3)

        infer_num_vertices = int(cond_verts_full.shape[0]) if bool(args.auto_infer_num_vertices) else int(infer_num_vertices_default)
        if max_vertices is not None and int(infer_num_vertices) > int(max_vertices):
            raise ValueError(
                f"auto infer_num_vertices={infer_num_vertices} exceeds model max_vertices={max_vertices}. "
                "Pass a larger --max_vertices for dynamic-vertex inference, train with a larger --max_vertices, "
                "or use a conditioning sample with fewer vertices."
            )

        scene = scene_info_from_metadata_dict(
            meta,
            vertex_counts=vertex_counts,
            max_num_objects=max_num_objects,
            source=str(meta_source),
        )
        if int(cond_verts_full.shape[0]) != int(scene.total_vertices):
            raise ValueError(
                f"First-frame vertex count mismatch: obj has V={int(cond_verts_full.shape[0])} "
                f"but metadata sum is V={int(scene.total_vertices)}. obj={first_obj_path} meta={meta_source}"
            )
        if int(scene.total_vertices) > int(infer_num_vertices):
            raise ValueError(
                f"infer_num_vertices={infer_num_vertices} is smaller than this scene total_vertices={scene.total_vertices}. "
                "Increase --infer_num_vertices or train with a larger --num_vertices. "
                f"cond_sample_dir={cond_sample_dir}"
            )

        pad_value = float(train_args.get("pad_value", 0.0))
        vertex_sampling = str(train_args.get("vertex_sampling", "first"))

        cond_verts_fixed, cond_mask, _ = fix_num_vertices(
            cond_verts_full.astype(np.float32),
            num_vertices=int(infer_num_vertices),
            vertex_sampling=vertex_sampling,
            pad_value=pad_value,
            sample_idx=None,
        )
        if int(cond_verts_full.shape[0]) > int(infer_num_vertices):
            raise RuntimeError("Internal error: expected to prevent truncation earlier.")

        scene_cond_np = None
        object_materials_np = None
        if bool(args.env_and_mat):
            env_and_mat_cond = _env_and_mat_conditioning_from_metadata(
                meta,
                meta_path=meta_path,
                max_num_objects=max_num_objects,
                train_args=train_args,
            )
            if bool(cond_scene) or bool(normalize_to_scene_box):
                scene_cond_np = _match_last_dim(
                    env_and_mat_cond.scene_cond.astype(np.float32, copy=False),
                    int(scene_cond_dim),
                )
            if bool(cond_object_material):
                object_materials_np = _match_last_dim(
                    env_and_mat_cond.object_materials.astype(np.float32, copy=False),
                    int(object_material_dim),
                )
        else:
            if bool(cond_scene) or bool(normalize_to_scene_box):
                scene_cond_np = _match_last_dim(
                    scene_cond_from_metadata_dict(meta).astype(np.float32, copy=False),
                    int(scene_cond_dim),
                )
            if bool(cond_object_material):
                object_materials_np = _match_last_dim(
                    object_materials_from_metadata_dict(meta, max_num_objects=max_num_objects).astype(np.float32, copy=False),
                    int(object_material_dim),
                )

        cond_verts_for_model = cond_verts_fixed.astype(np.float32, copy=False)
        if bool(normalize_to_scene_box):
            if scene_cond_np is None:
                raise RuntimeError("normalize_to_scene_box=True requires scene_cond to be available from metadata")
            cond_verts_for_model = _apply_scene_box_normalization(cond_verts_for_model, scene_cond=scene_cond_np)

        cond_pos = _normalize_positions(
            cond_verts_for_model,
            coord_scale=coord_scale,
            coord_shift=coord_shift,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )
        cond_mask_t = torch.from_numpy(cond_mask.astype(np.float32)).to(device=device, dtype=torch.float32)  # (V,)
        sample_mask = cond_mask_t[None, None, :].expand(1, int(infer_num_frames), int(infer_num_vertices))  # (1,F,V)

        if args.cond_first_frame_velocity:
            vel_path = resolve_velocity_path_from_first_frame_obj(first_obj_path, velocity_dirname=str(args.velocity_dirname))
            if not os.path.isfile(vel_path):
                raise FileNotFoundError(f"Missing velocity file: {vel_path}")
            vverts = np.load(vel_path).astype(np.float32)  # (V_full,3)
            if vverts.ndim != 2 or vverts.shape[1] != 3:
                raise ValueError(f"Velocity file must be (V,3), got {vverts.shape}: {vel_path}")
            if int(vverts.shape[0]) != int(cond_verts_full.shape[0]):
                raise ValueError(
                    f"Velocity vertex count mismatch: obj has V={int(cond_verts_full.shape[0])}, velocity has V={int(vverts.shape[0])}: {vel_path}"
                )
            vverts_fixed, vmask, _ = fix_num_vertices(
                vverts,
                num_vertices=int(infer_num_vertices),
                vertex_sampling=vertex_sampling,
                pad_value=pad_value,
                sample_idx=None,
            )
            if not np.array_equal(vmask, cond_mask):
                raise RuntimeError(f"Velocity mask mismatch vs position mask for {first_obj_path} (vel={vel_path})")
            if bool(normalize_to_scene_box):
                if scene_cond_np is None:
                    raise RuntimeError("normalize_to_scene_box=True requires scene_cond to be available from metadata")
                vverts_fixed = _apply_scene_box_velocity_normalization(vverts_fixed, scene_cond=scene_cond_np)
            vverts_fixed = _normalize_velocities(vverts_fixed, coord_scale=coord_scale, norm_std=norm_std)
            cond_parts = [cond_pos, vverts_fixed]
            cond_first_np = np.concatenate(cond_parts, axis=-1).astype(np.float32)  # (V,6)
        else:
            cond_first_np = cond_pos.astype(np.float32)  # (V,3)

        cond_first = torch.from_numpy(cond_first_np).to(device=device, dtype=torch.float32)[None, :, :]  # (1,V,C)
        scene_cond_t = (
            torch.from_numpy(scene_cond_np).to(device=device, dtype=torch.float32)[None, :]
            if scene_cond_np is not None and bool(cond_scene)
            else None
        )
        object_materials_t = (
            torch.from_numpy(object_materials_np).to(device=device, dtype=torch.float32)[None, :, :]
            if object_materials_np is not None
            else None
        )

        # Object ids (B,V). Padded vertices use pad_object_id.
        object_ids_full = np.asarray([obj_id for obj_id, v in enumerate(scene.vertex_counts) for _ in range(int(v))], dtype=np.int64)
        if int(object_ids_full.shape[0]) != int(scene.total_vertices):
            raise RuntimeError("Internal error: object_ids_full length mismatch")
        if int(scene.total_vertices) < int(infer_num_vertices):
            pad = np.full((int(infer_num_vertices) - int(scene.total_vertices),), int(pad_object_id), dtype=np.int64)
            object_ids_full = np.concatenate([object_ids_full, pad], axis=0)
        object_ids = torch.from_numpy(object_ids_full.astype(np.int64)).to(device=device, dtype=torch.long)[None, :]  # (1,V)

        fps_points_t = None
        fps_mask_t = None
        fps_object_ids_t = None
        fps_path = ""
        if bool(uses_fps_inputs):
            fps_points_np, fps_mask_np, fps_object_ids_np, fps_path = _load_ca_fps_inputs(
                fps_precomputed_root=fps_precomputed_root,
                fps_k=int(fps_k),
                dynamic_anchor=bool(dynamic_anchor) if bool(is_ca_fps_model) else False,
                max_num_objects=int(max_num_objects),
                pad_object_id=int(pad_object_id),
                cond_sample_dir=cond_sample_dir,
                cond_data_root=str(args.cond_data_root),
                rel_sample_dir=rel_sample_dir,
                infer_num_frames=int(infer_num_frames),
                coord_scale=coord_scale,
                coord_shift=coord_shift,
                norm_mean=norm_mean,
                norm_std=norm_std,
            )
            fps_points_t = torch.from_numpy(fps_points_np).to(device=device, dtype=torch.float32)[None, ...]
            fps_mask_t = torch.from_numpy(fps_mask_np).to(device=device, dtype=torch.float32)[None, :]
            fps_object_ids_t = torch.from_numpy(fps_object_ids_np).to(device=device, dtype=torch.long)[None, :]

        # Per-object topology (faces) from the first-frame combined OBJ (split by vertex slices).
        faces_by_obj: list[np.ndarray] = []
        v0, f0 = load_obj_vertices_faces(first_obj_path)
        if int(v0.shape[0]) != int(scene.total_vertices):
            raise ValueError(
                f"Combined first-frame OBJ vertex count mismatch: obj has V={int(v0.shape[0])} "
                f"but metadata sum is V={int(scene.total_vertices)}. obj={first_obj_path} meta={meta_path}"
            )
        for (s, e) in scene.vertex_slices:
            s = int(s)
            e = int(e)
            in_range = (f0 >= s) & (f0 < e)
            keep = np.all(in_range, axis=1)
            f_obj = f0[keep] - s
            if f_obj.size == 0:
                raise ValueError(
                    "No faces found for an object slice when splitting combined OBJ faces. "
                    f"slice=({s},{e}) obj={first_obj_path} meta={meta_path}"
                )
            faces_by_obj.append(f_obj.astype(np.int64))

        fixed_limits = _fixed_limits_from_cli(str(args.viz_fixed_limits)) or _fixed_limits_from_metadata_dict(meta)
        object_names = [
            f"{str(name)} {str(path)}"
            for name, path in zip(scene.mesh_names, scene.mesh_paths)
        ]
        gt_vertices = None
        if args.save_gt_gif or args.save_gt_mp4 or args.save_compare_gif or args.save_compare_mp4:
            gt_vertices = _load_gt_vertices(_gt_frame_paths(cond_sample_dir), int(infer_num_frames))
            if gt_vertices.shape[1] != int(scene.total_vertices):
                raise ValueError(
                    f"GT vertices have V={int(gt_vertices.shape[1])} but metadata expects V={int(scene.total_vertices)}. "
                    f"sample_dir={cond_sample_dir}"
                )

        if bool(args.save_scene_metadata):
            out_sample_rel_dir = os.path.relpath(sample_dir, str(args.out_dir)).replace("\\", "/").strip("/")
            with open(os.path.join(sample_dir, "scene_multiobj.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "out_layout": str(args.out_layout),
                        "out_sample_rel_dir": out_sample_rel_dir,
                        "cond_rel_sample_dir": rel_sample_dir,
                        "cond_sample_dir": cond_sample_dir,
                        "first_frame_obj": first_obj_path,
                        "metadata": meta_path,
                        "metadata_source": str(meta_source),
                        "conditioned_metadata_jsonl": str(args.conditioned_metadata_jsonl),
                        "mesh_vertex_count_json": mesh_vertex_count_json,
                        "fps_precomputed_path": fps_path,
                        "objects": [
                            {
                                "obj_id": int(k),
                                "mesh_name": str(scene.mesh_names[k]),
                                "mesh_path": str(scene.mesh_paths[k]),
                                "num_vertices": int(scene.vertex_counts[k]),
                                "vertex_slice": [int(scene.vertex_slices[k][0]), int(scene.vertex_slices[k][1])],
                            }
                            for k in range(len(scene.vertex_counts))
                        ],
                        "infer_num_frames": int(infer_num_frames),
                        "infer_num_vertices": int(infer_num_vertices),
                        "pad_object_id": int(pad_object_id),
                    },
                    f,
                    indent=2,
                )

        y = labels[i : i + 1]
        gen_range = range(int(args.num_generations_per_sample))
        if int(args.num_generations_per_sample) > 1 and not bool(args.verbose):
            gen_range = tqdm(gen_range, desc=f"rollout_samples[input={i:03d}]", unit="sample", leave=False)

        for repeat_idx in gen_range:
            gen_dir = sample_dir if int(args.num_generations_per_sample) == 1 else os.path.join(sample_dir, f"sample_{repeat_idx:02d}")
            os.makedirs(gen_dir, exist_ok=True)
            out_npz = os.path.join(gen_dir, "vertices.npz")
            if os.path.isfile(out_npz) and not bool(args.overwrite):
                tqdm.write(f"[SKIP] exists: {out_npz}")  # type: ignore[attr-defined]
                continue

            t0 = time.perf_counter()
            generate_kwargs = {
                "num_frames": int(infer_num_frames),
                "num_vertices": int(infer_num_vertices),
                "cond_first_frame": cond_first,
                "mask": sample_mask,
                "object_ids": object_ids,
                "scene_cond": scene_cond_t,
                "object_materials": object_materials_t,
                "clamp_cond_first_frame": not bool(delta_to_first_frame),
            }
            if bool(uses_fps_inputs):
                generate_kwargs.update(
                    {
                        "fps_points": fps_points_t,
                        "fps_mask": fps_mask_t,
                        "fps_object_ids": fps_object_ids_t,
                    }
                )
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                x_gen, _ = model.generate(y, **generate_kwargs)
            if device.type == "cuda":
                torch.cuda.synchronize(device=device)
            t1 = time.perf_counter()
            vlog(f"sample[{i}] gen[{repeat_idx}] sec={t1 - t0:.3f} x_gen shape={tuple(x_gen.shape)} dtype={x_gen.dtype}")

            x_np_norm = x_gen[0].detach().cpu().numpy().astype(np.float32)  # (F,V,3) (abs or delta depending on ckpt)
            if bool(delta_to_first_frame):
                # Model predicts deltas relative to the (normalized) first frame positions.
                x_np_norm[0] = 0.0
                vmask = cond_mask.astype(np.float32)  # (V,)
                x_np_norm = x_np_norm * vmask[None, :, None]
                x_np_norm = x_np_norm + cond_pos[None, :, :]
            x_np_denorm = _denormalize_positions(
                x_np_norm,
                coord_scale=coord_scale,
                coord_shift=coord_shift,
                norm_mean=norm_mean,
                norm_std=norm_std,
            )
            if bool(normalize_to_scene_box):
                if scene_cond_np is None:
                    raise RuntimeError("normalize_to_scene_box=True requires scene_cond to be available from metadata")
                x_np_denorm = _undo_scene_box_normalization(x_np_denorm, scene_cond=scene_cond_np)
            x_np = x_np_denorm if args.denorm else x_np_norm

            np.savez_compressed(
                out_npz,
                vertices=x_np,
                cond_sample_dir=str(cond_sample_dir),
                cond_first_frame_obj=str(first_obj_path),
                cond_metadata_path=str(meta_path),
                conditioned_metadata_source=str(meta_source),
            )

            want_pred_render = bool(args.save_gif or args.save_mp4)
            want_gt_render = bool(args.save_gt_gif or args.save_gt_mp4)
            want_compare_render = bool(args.save_compare_gif or args.save_compare_mp4)
            if want_pred_render or want_gt_render or want_compare_render:
                pred_vertices_vis = x_np_denorm[:, : int(scene.total_vertices), :].astype(np.float32, copy=False)
                pred_frames: list[np.ndarray] = []
                gt_frames: list[np.ndarray] = []
                compare_frames: list[np.ndarray] = []
                frame_range = range(int(infer_num_frames))
                if not bool(args.verbose):
                    frame_range = tqdm(
                        frame_range,
                        desc=f"render[sample={i:03d} gen={repeat_idx:02d}]",
                        unit="frame",
                        leave=False,
                    )

                sample_title = str(rel_sample_dir) if rel_sample_dir else Path(cond_sample_dir).name
                for fidx in frame_range:
                    pred_by_obj = [pred_vertices_vis[fidx, s:e, :].astype(np.float32) for (s, e) in scene.vertex_slices]
                    pred_img = _render_multiobj_frame(
                        pred_by_obj,
                        faces_by_obj,
                        colors=colors,
                        fixed_limits=fixed_limits,
                        elev=float(args.viz_elev),
                        azim=float(args.viz_azim),
                        dpi=int(args.compare_render_dpi if want_compare_render else 150),
                        title="inference" if want_pred_render else "",
                        object_names=object_names,
                    )
                    if want_pred_render:
                        pred_frames.append(pred_img)
                    if want_gt_render or want_compare_render:
                        if gt_vertices is None:
                            raise RuntimeError("GT vertices must be loaded when GT rendering is enabled")
                        gt_by_obj = [gt_vertices[fidx, s:e, :].astype(np.float32) for (s, e) in scene.vertex_slices]
                        gt_img = _render_multiobj_frame(
                            gt_by_obj,
                            faces_by_obj,
                            colors=colors,
                            fixed_limits=fixed_limits,
                            elev=float(args.viz_elev),
                            azim=float(args.viz_azim),
                            dpi=int(args.compare_render_dpi),
                            title="GT" if want_gt_render else "",
                            object_names=object_names,
                        )
                        if want_gt_render:
                            gt_frames.append(gt_img)
                    if want_compare_render:
                        compare_frames.append(
                            _compose_side_by_side(
                                gt_img,
                                pred_img,
                                sample_title=sample_title,
                                subset_label=str(args.compare_subset_label),
                                dpi=int(args.compare_compose_dpi),
                            )
                        )

                if want_pred_render:
                    out_gif = os.path.join(gen_dir, "inference.gif") if args.save_gif else None
                    out_mp4 = os.path.join(gen_dir, "inference.mp4") if args.save_mp4 else None
                    _save_animation(pred_frames, out_gif=out_gif, out_mp4=out_mp4, fps=int(args.fps))

                if want_gt_render:
                    out_gt_gif = os.path.join(gen_dir, "GT.gif") if args.save_gt_gif else None
                    out_gt_mp4 = os.path.join(gen_dir, "GT.mp4") if args.save_gt_mp4 else None
                    _save_animation(gt_frames, out_gif=out_gt_gif, out_mp4=out_gt_mp4, fps=int(args.fps))

                if want_compare_render:
                    compare_base = os.path.splitext(str(args.compare_out_name))[0]
                    out_compare_gif = os.path.join(gen_dir, f"{compare_base}.gif") if args.save_compare_gif else None
                    out_compare_mp4 = os.path.join(gen_dir, f"{compare_base}.mp4") if args.save_compare_mp4 else None
                    _save_animation(compare_frames, out_gif=out_compare_gif, out_mp4=out_compare_mp4, fps=int(args.fps))


if __name__ == "__main__":
    main()
