from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import torch
except Exception as e:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = e
else:
    _TORCH_IMPORT_ERROR = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover

    def tqdm(x, *args, **kwargs):  # type: ignore[no-redef]
        return x


_CODE_ROOT = Path(__file__).resolve().parent
_SRC_ROOT = _CODE_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from physformer.data.obj_io import load_obj_vertices_faces
def _require_torch_runtime() -> None:
    if torch is None:
        raise RuntimeError(
            "This evaluation script requires PyTorch. Activate an environment with PyTorch installed first."
        ) from _TORCH_IMPORT_ERROR


DEFAULT_CODE_ROOT = _CODE_ROOT
DEFAULT_EVAL_ASSETS_ROOT = DEFAULT_CODE_ROOT / "eval_assets"
DEFAULT_RUN_DIR = DEFAULT_CODE_ROOT / "checkpoints"
DEFAULT_SPLIT_FILE = DEFAULT_EVAL_ASSETS_ROOT / "eval_split.json"
DEFAULT_PRECOMP_ROOT = DEFAULT_EVAL_ASSETS_ROOT / "eval_precomp"
DEFAULT_DATA_ROOT = DEFAULT_EVAL_ASSETS_ROOT / "eval_data"
DEFAULT_PREP_SAMPLE_NAMES = "2obj_elastic,3obj_rigid,4obj_elastic,5obj_rigid"


def _resolve_default_ckpt(run_dir: Path) -> Path:
    for name in ("checkpoint-best.pt", "checkpoint-last.pt"):
        path = run_dir / name
        if path.is_file():
            return path
    epoch_re = re.compile(r"^checkpoint-epoch(\d+)\.pt$")
    best: tuple[int, Path] | None = None
    for path in run_dir.glob("checkpoint-epoch*.pt"):
        match = epoch_re.match(path.name)
        if not match:
            continue
        epoch = int(match.group(1))
        if best is None or epoch > best[0]:
            best = (epoch, path)
    if best is not None:
        return best[1]
    raise FileNotFoundError(
        f"Could not find checkpoint-best.pt, checkpoint-last.pt, or checkpoint-epoch*.pt under {run_dir}. "
        "Pass --ckpt explicitly."
    )


def _parse_tuple3(value: object, *, name: str) -> Optional[tuple[float, float, float]]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1]
        parts = [part.strip() for part in text.split(",") if part.strip()]
        if len(parts) != 3:
            raise ValueError(f"{name} must have exactly 3 comma-separated values, got: {value}")
        return tuple(float(part) for part in parts)  # type: ignore[return-value]
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) != 3:
            raise ValueError(f"{name} must have exactly 3 values, got: {value}")
        return tuple(float(part) for part in value)  # type: ignore[return-value]
    raise ValueError(f"{name} must be a 3-tuple/list or comma-separated string, got type={type(value)}")


def _resolve_norm_stats(train_args: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    ckpt_mean = _parse_tuple3(train_args.get("norm_mean", None), name="checkpoint norm_mean")
    ckpt_std = _parse_tuple3(train_args.get("norm_std", None), name="checkpoint norm_std")
    if (ckpt_mean is None) != (ckpt_std is None):
        raise ValueError("Checkpoint has only one of norm_mean/norm_std; both are required together")
    mean = ckpt_mean if ckpt_mean is not None else (0.0, 0.0, 0.0)
    std = ckpt_std if ckpt_std is not None else (1.0, 1.0, 1.0)
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


def _normalize_velocities(vertices: np.ndarray, *, coord_scale: float, norm_std: np.ndarray) -> np.ndarray:
    x = np.asarray(vertices, dtype=np.float32) / float(coord_scale)
    return x / norm_std


def _denormalize_positions_t(
    vertices_norm: torch.Tensor,
    *,
    coord_scale: float,
    coord_shift: float,
    norm_mean_t: torch.Tensor,
    norm_std_t: torch.Tensor,
) -> torch.Tensor:
    x = vertices_norm * norm_std_t + norm_mean_t
    return x * float(coord_scale) + float(coord_shift)


def _normalize_split_entry(value: object) -> int | str:
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Split entry is an empty string")
        if ":" not in text:
            try:
                return int(text)
            except ValueError:
                return text
        return text
    raise ValueError(f"Unsupported split entry type: {type(value)} (value={value!r})")


def _unique_preserve_order(entries: list[int | str]) -> list[int | str]:
    seen: set[str] = set()
    out: list[int | str] = []
    for entry in entries:
        key = f"i:{entry}" if isinstance(entry, int) else f"s:{entry}"
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
    return out


def _load_split_entries(split_file: str, split_name: str, split_groups: str) -> list[int | str]:
    with open(split_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if split_name not in data:
        raise ValueError(f"Split {split_name!r} not found in {split_file}. Available keys: {sorted(data.keys())}")
    split = data[split_name]
    if isinstance(split, list):
        return _unique_preserve_order([_normalize_split_entry(x) for x in split])
    if isinstance(split, dict):
        groups = [group.strip() for group in split_groups.split(",") if group.strip()]
        if not groups:
            raise ValueError("--split_groups must contain at least one group key")
        out: list[int | str] = []
        for group in groups:
            if group not in split:
                raise ValueError(f"Group {group!r} not found under split {split_name!r}; keys={sorted(split.keys())}")
            out.extend(_normalize_split_entry(x) for x in split[group])
        return _unique_preserve_order(out)
    raise ValueError(f"Unsupported split format for {split_name!r}: expected list or dict, got {type(split)}")


def _selector_parts(selector: int | str) -> tuple[str, int]:
    if isinstance(selector, int):
        raise ValueError(f"Expected selector '<obj_subdir>:<sample_index>', got int {selector}")
    text = str(selector).strip()
    if ":" not in text:
        raise ValueError(f"Invalid selector {text!r}: expected '<obj_subdir>:<sample_index>'")
    obj_subdir, idx_str = text.split(":", 1)
    obj_subdir = obj_subdir.strip().replace("\\", "/").strip("/")
    idx_str = idx_str.strip()
    if not obj_subdir or not idx_str.lstrip("-").isdigit():
        raise ValueError(f"Invalid selector {text!r}: expected '<obj_subdir>:<sample_index>'")
    return obj_subdir, int(idx_str)


def _selector_to_precomp_path(precomp_root: str | Path, selector: int | str) -> Path:
    obj_subdir, idx = _selector_parts(selector)
    return Path(precomp_root) / obj_subdir / f"sample_{idx:06d}.npz"


def _selector_to_metadata_path(data_root: str | Path, selector: int | str) -> Path:
    obj_subdir, idx = _selector_parts(selector)
    return Path(data_root) / obj_subdir / f"sample_{idx:06d}" / "metadata.json"


def _split_dataset_root(split_file: str | Path) -> Optional[Path]:
    try:
        with open(split_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    root = payload.get("dataset_root", None) if isinstance(payload, dict) else None
    if isinstance(root, str) and root.strip():
        return Path(root)
    return None


def _eval_asset_missing_paths(
    *,
    split_file: str | Path,
    split_name: str,
    split_groups: str,
    precomp_root: str | Path,
    data_root: str | Path,
) -> list[Path]:
    split_path = Path(split_file)
    if not split_path.is_file():
        return [split_path]

    entries = _load_split_entries(str(split_path), str(split_name), str(split_groups))
    split_root = _split_dataset_root(split_path)
    missing: list[Path] = []
    for entry in entries:
        npz_path = _selector_to_precomp_path(precomp_root, entry)
        if not npz_path.is_file():
            missing.append(npz_path)
        meta_path = _selector_to_metadata_path(data_root, entry)
        if not meta_path.is_file():
            fallback_meta = _selector_to_metadata_path(split_root, entry) if split_root is not None else None
            if fallback_meta is None or not fallback_meta.is_file():
                missing.append(meta_path)
    return missing


def _maybe_prepare_eval_assets(args: argparse.Namespace) -> None:
    missing = _eval_asset_missing_paths(
        split_file=args.split_file,
        split_name=args.split_name,
        split_groups=args.split_groups,
        precomp_root=args.precomp_root,
        data_root=args.data_root,
    )
    if not missing and not bool(args.prepare_overwrite):
        print(
            f"EVAL ASSETS: found {args.split_file}, {args.precomp_root}, and {args.data_root}; "
            "skipping preparation.",
            flush=True,
        )
        return

    sample_root = str(args.sample_root).strip()
    if not sample_root:
        if bool(args.prepare_overwrite):
            raise ValueError("--prepare_overwrite requires --sample_root so evaluation assets can be regenerated.")
        preview = ", ".join(str(path) for path in missing[:5])
        more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        raise FileNotFoundError(
            "Evaluation assets are incomplete and no raw --sample_root was provided. "
            f"Missing: {preview}{more}. "
            "Either keep the packaged eval_assets/ directory, run prepare_publish_eval_inputs.py, "
            "or pass --sample_root to let eval_publish_losses.py prepare them."
        )

    print(
        f"EVAL ASSETS: {'regenerating' if bool(args.prepare_overwrite) else 'missing assets detected; preparing'} "
        f"from {sample_root}",
        flush=True,
    )
    from prepare_publish_eval_inputs import prepare_eval_inputs

    prepare_args = argparse.Namespace(
        sample_root=Path(sample_root),
        sample_names=str(args.prepare_sample_names),
        precomp_root=Path(args.precomp_root),
        data_root=Path(args.data_root),
        split_file=Path(args.split_file),
        split_name=str(args.split_name),
        density=float(args.density),
        mesh_template_dir=Path(args.mesh_template_dir) if str(args.mesh_template_dir).strip() else None,
        overwrite=bool(args.prepare_overwrite),
    )
    prepare_eval_inputs(prepare_args)


def _expand_object_id_embed_in_state_dict(
    state_dict: dict[str, Any],
    *,
    target_max_num_objects: int,
    init_std: float,
) -> bool:
    changed = False
    for key in [k for k in state_dict.keys() if str(k).endswith("object_id_embed.weight")]:
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
        copy_n = min(old_max, int(target_max_num_objects))
        if copy_n > 0:
            new_weight[:copy_n] = weight[:copy_n]
        if copy_n < int(target_max_num_objects):
            n_extra = int(target_max_num_objects) - copy_n
            noise = torch.randn((n_extra, dim), dtype=weight.dtype, device=weight.device) * float(init_std)
            new_weight[copy_n : int(target_max_num_objects)] = mean_row + noise
        new_weight[int(target_max_num_objects)].zero_()
        state_dict[key] = new_weight
        changed = True
    return changed


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
    if changed:
        args = ckpt.get("args", None)
        if isinstance(args, dict):
            args["max_num_objects"] = int(target_max_num_objects)
    return changed


def _find_state_tensor_by_suffix(state_dict: dict[str, Any], suffix: str) -> Optional[torch.Tensor]:
    for key, value in state_dict.items():
        if str(key).endswith(str(suffix)) and torch.is_tensor(value):
            return value
    return None


def _infer_conditioning_dims_from_state_dict(state_dict: dict[str, Any]) -> tuple[int, int, int, int]:
    num_scene_tokens = 0
    scene_cond_dim = 0
    scene_cond_embed_out_tokens = 0
    object_material_dim = 0

    scene_in = _find_state_tensor_by_suffix(state_dict, "scene_cond_embed.0.weight")
    scene_out = _find_state_tensor_by_suffix(state_dict, "scene_cond_embed.2.weight")
    hidden_size = int(scene_in.shape[0]) if torch.is_tensor(scene_in) and scene_in.ndim == 2 else 0
    if torch.is_tensor(scene_in) and scene_in.ndim == 2:
        scene_cond_dim = int(scene_in.shape[1])
    if hidden_size > 0 and torch.is_tensor(scene_out) and scene_out.ndim == 2 and int(scene_out.shape[0]) % hidden_size == 0:
        scene_cond_embed_out_tokens = int(scene_out.shape[0]) // hidden_size

    scene_token_base = _find_state_tensor_by_suffix(state_dict, "scene_token_base")
    if torch.is_tensor(scene_token_base) and scene_token_base.ndim == 3:
        num_scene_tokens = int(scene_token_base.shape[1])
    elif scene_cond_embed_out_tokens > 0:
        num_scene_tokens = int(scene_cond_embed_out_tokens)

    obj_mat = _find_state_tensor_by_suffix(state_dict, "object_material_embed.0.weight")
    if torch.is_tensor(obj_mat) and obj_mat.ndim == 2:
        object_material_dim = int(obj_mat.shape[1])

    return int(num_scene_tokens), int(scene_cond_dim), int(scene_cond_embed_out_tokens), int(object_material_dim)


def _add_cond_x_embedder_keys_from_x_embedder(state_dict: dict[str, Any], module: torch.nn.Module) -> int:
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
        state_dict[target_key] = value
        renamed += 1
    return renamed


def masked_mse(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: pred={tuple(pred.shape)} gt={tuple(gt.shape)}")
    if pred.ndim != 4 or pred.shape[-1] != 3:
        raise ValueError(f"pred must be (B,F,V,3), got {tuple(pred.shape)}")
    if mask.shape != pred.shape[:3]:
        raise ValueError(f"mask must match pred/gt (B,F,V), got {tuple(mask.shape)} vs {tuple(pred.shape[:3])}")
    weights = mask.to(dtype=pred.dtype).unsqueeze(-1)
    denom = torch.clamp_min(weights.sum() * 3.0, 1.0)
    return ((pred - gt) ** 2 * weights).sum() / denom


def rigid_error(v0: torch.Tensor, vt: torch.Tensor) -> torch.Tensor:
    if v0.ndim != 2 or vt.ndim != 2 or v0.shape != vt.shape or v0.shape[1] != 3:
        raise ValueError(f"rigid_error expects v0/vt as (N,3) with same shape, got {tuple(v0.shape)} {tuple(vt.shape)}")
    if v0.shape[0] == 0:
        return torch.tensor(float("nan"), device=v0.device, dtype=v0.dtype)
    centroid0 = v0.mean(dim=0, keepdim=True)
    centroidt = vt.mean(dim=0, keepdim=True)
    x = v0 - centroid0
    y = vt - centroidt
    u, _, vt_svd = torch.linalg.svd(x.T @ y)
    rotation = vt_svd.T @ u.T
    if torch.det(rotation) < 0:
        vt_svd = vt_svd.clone()
        vt_svd[-1, :] *= -1
        rotation = vt_svd.T @ u.T
    translation = centroidt - centroid0 @ rotation.T
    aligned = v0 @ rotation.T + translation
    return ((aligned - vt) ** 2).mean()


def _safe_float(value: object, default: float) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _nested_get(data: dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _object_density(meta: dict[str, Any], obj: dict[str, Any], default_density: float) -> float:
    for value in (
        obj.get("rho"),
        obj.get("density"),
        _nested_get(obj, "rigid", "rho"),
        _nested_get(obj, "material", "rho"),
        _nested_get(obj, "material", "density"),
        meta.get("rho"),
        meta.get("density"),
        _nested_get(meta, "pbd", "rho"),
        _nested_get(meta, "pbd", "density"),
        _nested_get(meta, "rigid", "rho"),
        _nested_get(meta, "material", "rho"),
        _nested_get(meta, "material", "density"),
    ):
        if value is not None:
            return _safe_float(value, float(default_density))
    return float(default_density)


def _resolve_mesh_path(mesh_used: str, *, template_dir: str = "") -> Path:
    candidates: list[Path] = []
    raw = Path(str(mesh_used))
    candidates.append(raw)
    candidates.append(DEFAULT_CODE_ROOT / raw)
    if template_dir:
        tdir = Path(template_dir)
        candidates.append(tdir / raw.name)
        candidates.append(tdir / f"{raw.stem}.obj")
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not resolve mesh path for {mesh_used!r}. Tried: {', '.join(map(str, candidates))}")


def _mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    verts = np.asarray(vertices, dtype=np.float64)
    faces_i64 = np.asarray(faces, dtype=np.int64)
    tris = verts[faces_i64]
    signed_volume = float(np.sum(np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2]))) / 6.0)
    return abs(signed_volume)


def _load_mesh_volume_cache_json(path: str | Path) -> dict[str, float]:
    cache_path = Path(path)
    if not str(cache_path).strip() or not cache_path.is_file():
        return {}
    with cache_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    raw = payload.get("volumes", payload) if isinstance(payload, dict) else {}
    if not isinstance(raw, dict):
        raise ValueError(f"Unsupported mesh volume cache JSON format: {cache_path}")
    out: dict[str, float] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        try:
            volume = float(value)
        except Exception:
            continue
        if math.isfinite(volume) and volume > 0.0:
            out[key] = volume
    return out


def _write_mesh_volume_cache_json(path: str | Path, cache: dict[str, float]) -> None:
    cache_path = Path(path)
    if not str(cache_path).strip():
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "volume_units": "template_obj_coordinate_units_cubed",
        "volumes": {str(k): float(v) for k, v in sorted(cache.items())},
    }
    tmp_path = cache_path.with_name(f"{cache_path.name}.tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, cache_path)


def _load_object_masses(
    meta_path: Path,
    *,
    default_density: float,
    template_dir: str,
    volume_cache: dict[str, float],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    with meta_path.open("r", encoding="utf-8") as f:
        meta_any = json.load(f)
    if not isinstance(meta_any, dict):
        raise ValueError(f"metadata.json must contain a dict: {meta_path}")
    meta: dict[str, Any] = meta_any
    objects = meta.get("objects", None)
    if not isinstance(objects, list) or not objects:
        raise ValueError(f"metadata missing non-empty objects list: {meta_path}")

    masses: list[float] = []
    details: list[dict[str, Any]] = []
    for obj_idx, obj_any in enumerate(objects):
        if not isinstance(obj_any, dict):
            raise ValueError(f"metadata objects[{obj_idx}] is not a dict: {meta_path}")
        obj: dict[str, Any] = obj_any
        mesh_used = obj.get("mesh_used") or obj.get("mesh_source")
        if not isinstance(mesh_used, str) or not mesh_used:
            raise ValueError(f"metadata objects[{obj_idx}] missing mesh_used/mesh_source: {meta_path}")
        mesh_path = _resolve_mesh_path(mesh_used, template_dir=template_dir)
        mesh_key = str(mesh_path.resolve())
        if mesh_key not in volume_cache:
            verts, faces = load_obj_vertices_faces(str(mesh_path))
            volume_cache[mesh_key] = _mesh_volume(verts, faces)
        template_volume = float(volume_cache[mesh_key])
        scale = _safe_float(obj.get("scale"), 1.0)
        density = _object_density(meta, obj, float(default_density))
        mass = float(density) * template_volume * float(scale) ** 3
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError(
                f"Invalid mass for object {obj_idx} in {meta_path}: "
                f"density={density}, template_volume={template_volume}, scale={scale}, mass={mass}"
            )
        masses.append(mass)
        details.append(
            {
                "object_index": int(obj_idx),
                "mesh": str(mesh_path),
                "density": float(density),
                "template_volume": float(template_volume),
                "scale": float(scale),
                "mass": float(mass),
            }
        )
    return np.asarray(masses, dtype=np.float64), details


def _object_centers(vertices: np.ndarray, mask: np.ndarray, object_ids: np.ndarray, *, num_objects: int) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float64)
    mask_f = (np.asarray(mask) > 0).astype(np.float64, copy=False)
    obj_ids = np.asarray(object_ids).astype(np.int64, copy=False)
    if verts.ndim != 3 or verts.shape[-1] != 3:
        raise ValueError(f"vertices must be (F,V,3), got {verts.shape}")
    if mask_f.shape != verts.shape[:2]:
        raise ValueError(f"mask shape {mask_f.shape} does not match vertices shape {verts.shape[:2]}")
    if obj_ids.shape != (verts.shape[1],):
        raise ValueError(f"object_ids must be (V,), got {obj_ids.shape}, V={verts.shape[1]}")

    centers = np.zeros((int(verts.shape[0]), int(num_objects), 3), dtype=np.float64)
    for oid in range(int(num_objects)):
        sel = obj_ids == int(oid)
        if not np.any(sel):
            raise ValueError(f"No vertices found for object id {oid}")
        obj_mask = mask_f[:, sel]
        counts = obj_mask.sum(axis=1)
        if np.any(counts <= 0.0):
            bad = int(np.where(counts <= 0.0)[0][0])
            raise ValueError(f"No present vertices for object id {oid} at frame {bad}")
        centers[:, oid, :] = (verts[:, sel, :] * obj_mask[:, :, None]).sum(axis=1) / counts[:, None]
    return centers


def _frame_velocities(centers: np.ndarray, *, dt: float, scheme: str) -> np.ndarray:
    c = np.asarray(centers, dtype=np.float64)
    if c.shape[0] < 2:
        raise ValueError("Need at least 2 frames to finite-difference velocities")
    if float(dt) <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")
    v = np.empty_like(c)
    if scheme == "backward":
        v[0] = (c[1] - c[0]) / float(dt)
        v[1:] = (c[1:] - c[:-1]) / float(dt)
    elif scheme == "forward":
        v[:-1] = (c[1:] - c[:-1]) / float(dt)
        v[-1] = v[-2]
    elif scheme == "central":
        v[0] = (c[1] - c[0]) / float(dt)
        v[-1] = (c[-1] - c[-2]) / float(dt)
        if c.shape[0] > 2:
            v[1:-1] = (c[2:] - c[:-2]) / (2.0 * float(dt))
    else:
        raise ValueError(f"Unsupported velocity scheme: {scheme}")
    return v


def _system_momentum(
    vertices: np.ndarray,
    mask: np.ndarray,
    object_ids: np.ndarray,
    *,
    masses: np.ndarray,
    num_objects: int,
    dt: float,
    velocity_scheme: str,
) -> np.ndarray:
    centers = _object_centers(vertices, mask, object_ids, num_objects=int(num_objects))
    velocities = _frame_velocities(centers, dt=float(dt), scheme=str(velocity_scheme))
    masses_f = np.asarray(masses, dtype=np.float64)
    if masses_f.shape != (int(num_objects),):
        raise ValueError(f"masses must be ({int(num_objects)},), got {masses_f.shape}")
    return (velocities * masses_f[None, :, None]).sum(axis=1)


@dataclass(frozen=True)
class MomentumDriftMetrics:
    momentum_drift_ratio: float
    gt_per_frame_drift_norms: list[float]
    pred_per_frame_drift_norms: list[float]
    gt_initial_momentum: list[float]
    pred_initial_momentum: list[float]
    gt_initial_momentum_norm_sq: float
    pred_initial_momentum_norm_sq: float
    gt_momentum_drift_norm_mean: float
    pred_momentum_drift_norm_mean: float
    gt_drift_denominator: float
    gt_drift_denominator_was_clamped: bool


def momentum_drift_ratio(
    pred_vertices: np.ndarray,
    gt_vertices: np.ndarray,
    mask: np.ndarray,
    object_ids: np.ndarray,
    *,
    masses: np.ndarray,
    num_objects: int,
    dt: float,
    velocity_scheme: str,
    last_frame: int,
    eps: float,
) -> MomentumDriftMetrics:
    gt_p = _system_momentum(
        gt_vertices,
        mask,
        object_ids,
        masses=masses,
        num_objects=int(num_objects),
        dt=float(dt),
        velocity_scheme=str(velocity_scheme),
    )
    pred_p = _system_momentum(
        pred_vertices,
        mask,
        object_ids,
        masses=masses,
        num_objects=int(num_objects),
        dt=float(dt),
        velocity_scheme=str(velocity_scheme),
    )
    f_last = min(int(last_frame), int(gt_p.shape[0]) - 1)
    if f_last < 1:
        raise ValueError(f"Need last_frame >= 1 after clipping, got {f_last}")
    idx = np.arange(1, f_last + 1, dtype=np.int64)
    gt_delta = gt_p[idx] - gt_p[0][None, :]
    pred_delta = pred_p[idx] - pred_p[0][None, :]
    gt_drift_norm = np.linalg.norm(gt_delta, axis=1)
    pred_drift_norm = np.linalg.norm(pred_delta, axis=1)
    gt_drift_norm_mean = float(gt_drift_norm.mean())
    pred_drift_norm_mean = float(pred_drift_norm.mean())
    denom = max(gt_drift_norm_mean, float(eps))
    return MomentumDriftMetrics(
        momentum_drift_ratio=float(pred_drift_norm_mean / denom),
        gt_per_frame_drift_norms=[float(x) for x in gt_drift_norm.tolist()],
        pred_per_frame_drift_norms=[float(x) for x in pred_drift_norm.tolist()],
        gt_initial_momentum=[float(x) for x in gt_p[0].tolist()],
        pred_initial_momentum=[float(x) for x in pred_p[0].tolist()],
        gt_initial_momentum_norm_sq=float(np.dot(gt_p[0], gt_p[0])),
        pred_initial_momentum_norm_sq=float(np.dot(pred_p[0], pred_p[0])),
        gt_momentum_drift_norm_mean=float(gt_drift_norm_mean),
        pred_momentum_drift_norm_mean=float(pred_drift_norm_mean),
        gt_drift_denominator=float(denom),
        gt_drift_denominator_was_clamped=bool(gt_drift_norm_mean < float(eps)),
    )


@dataclass(frozen=True)
class PerGenerationLosses:
    generation_index: int
    mse: float
    rigidity: float
    momentum_drift_ratio: float
    pred_momentum_drift_norm_mean: float


@dataclass(frozen=True)
class PerSampleLosses:
    selector: str
    precomp_path: str
    metadata_path: str
    num_objects: int
    num_present_vertices: int
    masses: list[float]
    losses: list[PerGenerationLosses]
    mse_mean: float
    mse_std: float
    mse_min: float
    mse_max: float
    rigidity_mean: float
    rigidity_std: float
    rigidity_min: float
    rigidity_max: float
    momentum_drift_ratio_mean: float
    momentum_drift_ratio_std: float
    momentum_drift_ratio_min: float
    momentum_drift_ratio_max: float
    gt_initial_momentum_norm_sq: float
    gt_momentum_drift_norm_mean: float
    pred_momentum_drift_norm_mean_mean: float
    gt_drift_denominator: float
    gt_drift_denominator_was_clamped: bool


def _finite_stats(values: list[float] | np.ndarray) -> tuple[float, float, float, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return float(x.mean()), float(x.std(ddof=0)), float(x.min()), float(x.max())


def _nanmean(values: list[float] | np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


def _nanstd(values: list[float] | np.ndarray, *, ddof: int = 0) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0 or x.size <= int(ddof):
        return float("nan")
    return float(x.std(ddof=int(ddof)))


def _global_momentum_drift_summary(rows: list[PerSampleLosses], *, eps: float) -> dict[str, Any]:
    gt_mean = _nanmean([r.gt_momentum_drift_norm_mean for r in rows])
    denom = max(float(gt_mean), float(eps)) if math.isfinite(float(gt_mean)) else float("nan")
    denom_clamped = bool(math.isfinite(float(gt_mean)) and float(gt_mean) < float(eps))

    max_generations = max((len(r.losses) for r in rows), default=0)
    pred_by_generation: list[float] = []
    ratio_by_generation: list[float] = []
    for gen_idx in range(max_generations):
        pred_mean = _nanmean(
            [
                r.losses[gen_idx].pred_momentum_drift_norm_mean
                for r in rows
                if gen_idx < len(r.losses)
            ]
        )
        pred_by_generation.append(float(pred_mean))
        ratio = float(pred_mean / denom) if math.isfinite(pred_mean) and math.isfinite(denom) else float("nan")
        ratio_by_generation.append(ratio)

    best_pred_mean = _nanmean(
        [
            min(
                (
                    loss.pred_momentum_drift_norm_mean
                    for loss in r.losses
                    if math.isfinite(float(loss.pred_momentum_drift_norm_mean))
                ),
                default=float("nan"),
            )
            for r in rows
        ]
    )
    best_ratio = (
        float(best_pred_mean / denom)
        if math.isfinite(float(best_pred_mean)) and math.isfinite(float(denom))
        else float("nan")
    )

    return {
        "gt_momentum_drift_norm_mean": float(gt_mean),
        "pred_momentum_drift_norm_mean_by_generation": pred_by_generation,
        "momentum_drift_ratio_by_generation": ratio_by_generation,
        "momentum_drift_ratio": ratio_by_generation[0] if ratio_by_generation else float("nan"),
        "momentum_drift_ratio_first_generation": ratio_by_generation[0] if ratio_by_generation else float("nan"),
        "momentum_drift_ratio_mean_over_generations": _nanmean(ratio_by_generation),
        "momentum_drift_ratio_std_across_generations": _nanstd(ratio_by_generation, ddof=1),
        "pred_momentum_drift_norm_mean_best_of_generations": float(best_pred_mean),
        "momentum_drift_ratio_best_of_generations": float(best_ratio),
        "global_momentum_drift_denominator": float(denom),
        "global_momentum_drift_denominator_was_clamped": bool(denom_clamped),
    }


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Publication evaluator for PhysFormer vertex-precomputed multi-object checkpoints. "
            "Loads each conditioning sample once and evaluates MSE, Kabsch rigidity loss, "
            "and momentum drift ratio on the same generated rollouts."
        )
    )
    p.add_argument(
        "--ckpt",
        type=str,
        default="",
        help=(
            "Model checkpoint (.pt). If omitted, resolves checkpoint-best.pt, checkpoint-last.pt, "
            "then highest checkpoint-epoch*.pt under --run_dir."
        ),
    )
    p.add_argument("--run_dir", type=str, default=str(DEFAULT_RUN_DIR), help="Used only when --ckpt is omitted.")
    p.add_argument("--split_file", type=str, default=str(DEFAULT_SPLIT_FILE))
    p.add_argument("--split_name", type=str, default="test")
    p.add_argument("--split_groups", type=str, default="rest,move", help="Used if split_name maps to grouped lists.")
    p.add_argument("--precomp_root", type=str, default=str(DEFAULT_PRECOMP_ROOT))
    p.add_argument("--data_root", type=str, default=str(DEFAULT_DATA_ROOT), help="Root containing sample metadata.json.")
    p.add_argument(
        "--sample_root",
        type=str,
        default="",
        help=(
            "Optional raw ground-truth sample root. If eval precomp/data assets are missing, "
            "the evaluator prepares them from this directory before running losses."
        ),
    )
    p.add_argument(
        "--prepare_sample_names",
        type=str,
        default=DEFAULT_PREP_SAMPLE_NAMES,
        help="Comma-separated raw sample folders to prepare when --sample_root is used.",
    )
    p.add_argument(
        "--prepare_overwrite",
        action="store_true",
        help="Regenerate eval precomp/data assets even if complete assets already exist.",
    )
    p.add_argument("--mesh_template_dir", type=str, default="", help="Fallback directory for metadata mesh basenames.")
    p.add_argument(
        "--mesh_volume_cache_json",
        type=str,
        default=str(DEFAULT_CODE_ROOT / "mesh_volume_cache.json"),
        help="Cache for unscaled template mesh volumes. Set to '' to disable disk caching.",
    )
    p.add_argument("--density", type=float, default=1000.0, help="Fallback density when metadata omits rho/density.")
    p.add_argument("-k", "--num_generations", type=int, default=4, help="Independent samples per conditioning input.")
    p.add_argument("--limit", type=int, default=0, help="If >0, evaluate only the first N split samples.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16", "fp16"])
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--no_ema", action="store_false", dest="use_ema")
    p.set_defaults(use_ema=True)

    p.add_argument("--infer_num_frames", type=int, default=0, help="0 = infer from NPZ shape.")
    p.add_argument("--infer_num_vertices", type=int, default=0, help="0 = infer from NPZ shape.")
    p.add_argument(
        "--max_num_objects",
        type=int,
        default=0,
        help="Override checkpoint max_num_objects. Larger values expand object-id embeddings.",
    )
    p.add_argument("--precomp_pad_object_id", type=int, default=5)

    p.add_argument("--sampling_method", type=str, default="", choices=["", "euler", "heun"])
    p.add_argument("--num_sampling_steps", type=int, default=0)

    p.add_argument(
        "--rigidity_last_frame",
        type=int,
        default=48,
        help="Rigidity is averaged over frames 1..N inclusive against frame 0, clipped to available frames.",
    )
    p.add_argument("--dt", type=float, default=0.004166666666666667, help="Frame interval for momentum velocities.")
    p.add_argument("--velocity_scheme", type=str, default="backward", choices=["backward", "forward", "central"])
    p.add_argument(
        "--momentum_last_frame",
        type=int,
        default=48,
        help=(
            "Momentum drift ratio uses frames 1..N inclusive against frame 0, "
            "clipped to available frames."
        ),
    )
    p.add_argument(
        "--momentum_eps",
        type=float,
        default=1e-12,
        help="Small clamp only for per-sample diagnostic ratios when GT drift is zero.",
    )

    p.add_argument("--out_json", type=str, required=True, help="Publication JSON report path.")
    p.add_argument("--out_tsv", type=str, default="", help="Optional compact per-sample TSV path.")
    p.add_argument("--store_object_mass_details", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


def _resolve_amp(device: torch.device, amp: str) -> tuple[bool, Optional[torch.dtype], str]:
    if amp == "none":
        return False, None, amp
    if device.type == "cpu":
        return amp == "bf16", torch.bfloat16 if amp == "bf16" else None, amp
    if amp == "bf16":
        is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(is_bf16_supported) and not bool(is_bf16_supported()):
            amp = "fp16"
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return True, dtype, amp


def _load_model_and_runtime(args: argparse.Namespace) -> tuple[
    PhysFormerDenoiser,
    Path,
    dict[str, Any],
    DiffusionConfig,
    int,
    np.ndarray,
    np.ndarray,
    float,
    float,
    torch.device,
    bool,
    Optional[torch.dtype],
]:
    _require_torch_runtime()
    from physformer.diffusion.denoiser import DiffusionConfig
    from physformer.diffusion.physformer_denoiser import PhysFormerDenoiser
    from physformer.models.physformer import canonical_model_name

    ckpt_path = Path(str(args.ckpt)) if str(args.ckpt).strip() else _resolve_default_ckpt(Path(str(args.run_dir)))
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    train_args_any = ckpt.get("args", {})
    train_args: dict[str, Any] = train_args_any if isinstance(train_args_any, dict) else {}

    ckpt_max_num_objects = int(train_args.get("max_num_objects", 3) or 0)
    if ckpt_max_num_objects <= 0:
        ckpt_max_num_objects = 3
    target_max_num_objects = int(args.max_num_objects) if int(args.max_num_objects) > 0 else ckpt_max_num_objects
    if target_max_num_objects <= 0:
        raise ValueError(f"Invalid max_num_objects={target_max_num_objects}")
    if int(args.max_num_objects) > 0 and target_max_num_objects < ckpt_max_num_objects:
        raise ValueError(
            f"--max_num_objects={target_max_num_objects} is smaller than checkpoint max_num_objects={ckpt_max_num_objects}."
        )

    if target_max_num_objects > ckpt_max_num_objects:
        init_std = float(os.environ.get("PHYSFORMER_OBJ_EMBED_INIT_STD", "0.02"))
        if _maybe_expand_ckpt(ckpt, target_max_num_objects=target_max_num_objects, init_std=init_std):
            print(
                f"[HACK] Expanded object_id_embed rows: {ckpt_max_num_objects} -> {target_max_num_objects} "
                f"(init_std={init_std:.6g}).",
                flush=True,
            )
        train_args = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}
        ckpt["args"] = train_args
        train_args["max_num_objects"] = target_max_num_objects

    device = torch.device(str(args.device) if torch.cuda.is_available() else "cpu")
    use_amp, amp_dtype, resolved_amp = _resolve_amp(device, str(args.amp))
    args.amp = resolved_amp

    coord_scale = float(train_args.get("coord_scale", 1.0))
    coord_shift = float(train_args.get("coord_shift", 0.0))
    norm_mean, norm_std = _resolve_norm_stats(train_args)

    if bool(args.use_ema) and "ema" in ckpt:
        print("[CKPT] Loading EMA weights: ckpt['ema']['shadow']", flush=True)
        state_dict_to_load = dict(ckpt["ema"]["shadow"])
    else:
        if bool(args.use_ema):
            print("[CKPT] Requested EMA but checkpoint has no 'ema'; loading ckpt['model']", flush=True)
        else:
            print("[CKPT] Loading raw weights: ckpt['model']", flush=True)
        state_dict_to_load = dict(ckpt["model"])

    ckpt_num_scene_tokens, ckpt_scene_cond_dim, ckpt_scene_cond_embed_out_tokens, ckpt_object_material_dim = (
        _infer_conditioning_dims_from_state_dict(state_dict_to_load)
    )

    model_kwargs = {
        "use_rope": bool(train_args.get("use_rope", True)),
        "num_register_tokens": int(train_args.get("num_register_tokens", 16)),
        "max_frames": int(train_args.get("max_frames", 128)),
        "max_vertices": int(train_args.get("max_vertices", 8192)),
        "attn_drop": float(train_args.get("attn_drop", 0.0)),
        "proj_drop": float(train_args.get("proj_drop", 0.0)),
        "max_num_objects": int(target_max_num_objects),
        "use_object_id_embed": False,
        "num_scene_tokens": int(ckpt_num_scene_tokens),
        "scene_cond_dim": int(ckpt_scene_cond_dim),
        "scene_cond_embed_out_tokens": int(ckpt_scene_cond_embed_out_tokens),
        "object_material_dim": int(ckpt_object_material_dim),
    }

    sampling_method = str(train_args.get("sampling_method", "heun"))
    if args.sampling_method:
        sampling_method = str(args.sampling_method)
    num_sampling_steps = int(train_args.get("num_sampling_steps", 50))
    if int(args.num_sampling_steps) > 0:
        num_sampling_steps = int(args.num_sampling_steps)

    diff_cfg = DiffusionConfig(
        P_mean=float(train_args.get("P_mean", -0.8)),
        P_std=float(train_args.get("P_std", 0.8)),
        t_eps=float(train_args.get("t_eps", 5e-2)),
        noise_scale=float(train_args.get("noise_scale", 1.0)),
        sampling_method=sampling_method,
        num_sampling_steps=num_sampling_steps,
    )

    model_name = canonical_model_name(str(train_args.get("model", "PhysFormer-B")))
    model = PhysFormerDenoiser(
        model_name=model_name,
        num_frames=int(train_args.get("num_frames", 49)),
        num_vertices=int(train_args.get("num_vertices", 88)),
        num_classes=int(train_args.get("num_classes", 1)),
        model_kwargs=model_kwargs,
        diffusion=diff_cfg,
    ).to(device)

    _rename_legacy_x_embed_cond_keys(state_dict_to_load, model)
    _add_cond_x_embedder_keys_from_x_embedder(state_dict_to_load, model)
    model.load_state_dict(state_dict_to_load, strict=True)
    model.eval()

    model_pad_object_id = int(getattr(getattr(model, "net", None), "pad_object_id", model_kwargs["max_num_objects"]))
    return (
        model,
        ckpt_path,
        train_args,
        diff_cfg,
        model_pad_object_id,
        norm_mean,
        norm_std,
        coord_scale,
        coord_shift,
        device,
        use_amp,
        amp_dtype,
    )


def _rigidity_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    object_ids: torch.Tensor,
    *,
    pad_object_id: int,
    last_frame: int,
) -> float:
    f_last = int(min(max(int(last_frame), 1), int(pred.shape[1]) - 1))
    mask0 = mask[0, 0] > 0.0
    obj_ids_1d = object_ids[0]
    unique_obj_ids = sorted(set(obj_ids_1d.detach().cpu().tolist()) - {int(pad_object_id)})
    per_frame_err: list[torch.Tensor] = []
    for t in range(1, f_last + 1):
        maskt = mask[0, t] > 0.0
        per_obj_err: list[torch.Tensor] = []
        for oid in unique_obj_ids:
            sel = (obj_ids_1d == int(oid)) & mask0 & maskt
            if not torch.any(sel):
                continue
            v_ref = gt[0, 0][sel]
            v_pred = pred[0, t][sel]
            if v_ref.shape[0] >= 3:
                per_obj_err.append(rigid_error(v_ref, v_pred))
        if per_obj_err:
            per_frame_err.append(torch.stack(per_obj_err).mean())
    if not per_frame_err:
        return float("nan")
    return float(torch.stack(per_frame_err).mean().item())


def _write_tsv(path: Path, rows: list[PerSampleLosses]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "selector",
        "num_objects",
        "num_present_vertices",
        "mse_mean",
        "mse_std",
        "rigidity_mean",
        "rigidity_std",
        "momentum_drift_ratio_mean",
        "momentum_drift_ratio_std",
        "gt_momentum_drift_norm_mean",
        "pred_momentum_drift_norm_mean_mean",
        "gt_drift_denominator_was_clamped",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: getattr(row, k) for k in fieldnames})


def main() -> None:
    args = _build_argparser().parse_args()
    _require_torch_runtime()
    with torch.no_grad():
        _run_main(args)


def _run_main(args: argparse.Namespace) -> None:
    if int(args.num_generations) <= 0:
        raise ValueError(f"--num_generations must be >= 1, got {args.num_generations}")

    _maybe_prepare_eval_assets(args)

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    (
        model,
        ckpt_path,
        train_args,
        diff_cfg,
        model_pad_object_id,
        norm_mean,
        norm_std,
        coord_scale,
        coord_shift,
        device,
        use_amp,
        amp_dtype,
    ) = _load_model_and_runtime(args)

    print(f"CHECKPOINT: {ckpt_path}", flush=True)
    print(f"DEVICE: requested={args.device} actual={device} amp={args.amp}", flush=True)

    split_entries = _load_split_entries(str(args.split_file), str(args.split_name), str(args.split_groups))
    if int(args.limit) > 0:
        split_entries = split_entries[: int(args.limit)]
    if not split_entries:
        raise ValueError("Split contains 0 entries after filtering.")

    split_root = _split_dataset_root(str(args.split_file))
    volume_cache_path = str(args.mesh_volume_cache_json).strip()
    volume_cache = _load_mesh_volume_cache_json(volume_cache_path) if volume_cache_path else {}
    initial_volume_cache_size = len(volume_cache)
    mass_details_by_selector: dict[str, list[dict[str, Any]]] = {}

    norm_mean_t = torch.from_numpy(norm_mean.reshape(1, 1, 1, 3)).to(device=device, dtype=torch.float32)
    norm_std_t = torch.from_numpy(norm_std.reshape(1, 1, 1, 3)).to(device=device, dtype=torch.float32)
    use_cond_norm = bool(train_args.get("cond_first_frame_normal", False))
    if use_cond_norm:
        raise ValueError(
            "This official-demo-only evaluator uses the exported PhysFormer model, which supports "
            "first-frame position/velocity conditioning but not the older normal-conditioned checkpoint path. "
            "Use a non-normal-conditioned checkpoint or re-export the full model package."
        )

    results: list[PerSampleLosses] = []
    iterator = split_entries if bool(args.verbose) else tqdm(split_entries, desc=f"publish-eval[{args.split_name}]", unit="sample")

    for entry in iterator:
        selector = str(entry)
        npz_path = _selector_to_precomp_path(str(args.precomp_root), entry)
        if not npz_path.is_file():
            raise FileNotFoundError(f"Missing precomputed NPZ: {npz_path} (selector={selector})")

        meta_path = _selector_to_metadata_path(str(args.data_root), entry)
        if not meta_path.is_file() and split_root is not None:
            fallback_meta = _selector_to_metadata_path(split_root, entry)
            if fallback_meta.is_file():
                meta_path = fallback_meta
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"Missing metadata for selector={selector}: tried {meta_path}"
                + (f" and split dataset_root={split_root}" if split_root is not None else "")
            )

        with np.load(npz_path, allow_pickle=False) as data:
            gt_raw = data["vertices"].astype(np.float32)
            mask_np = np.asarray(data["mask"])
            object_ids_np = np.asarray(data["object_ids"]).astype(np.int64)
            num_objects = int(np.asarray(data["num_objects"]).item())
            object_masses_np = data["object_masses"].astype(np.float64) if "object_masses" in data else None
            if "first_frame_velocity" not in data:
                raise KeyError(f"Missing key 'first_frame_velocity' in {npz_path}")
            first_vel_raw = data["first_frame_velocity"].astype(np.float32)
            first_nrm_raw = None
            if use_cond_norm:
                if "first_frame_normal" not in data:
                    raise KeyError(f"Missing key 'first_frame_normal' in {npz_path}")
                first_nrm_raw = data["first_frame_normal"].astype(np.float32)

        if gt_raw.ndim != 3 or gt_raw.shape[-1] != 3:
            raise ValueError(f"Invalid vertices shape in {npz_path}: expected (F,V,3), got {gt_raw.shape}")
        f_gt, v_gt, _ = gt_raw.shape
        infer_num_frames = int(args.infer_num_frames) if int(args.infer_num_frames) > 0 else int(f_gt)
        infer_num_vertices = int(args.infer_num_vertices) if int(args.infer_num_vertices) > 0 else int(v_gt)
        if infer_num_frames != int(f_gt):
            raise ValueError(f"infer_num_frames={infer_num_frames} must match GT frames={f_gt} for {npz_path}")
        if infer_num_vertices < int(v_gt):
            raise ValueError(f"infer_num_vertices={infer_num_vertices} is smaller than GT vertices={v_gt}")

        if infer_num_vertices > int(v_gt):
            pad_v = infer_num_vertices - int(v_gt)
            gt_raw = np.pad(gt_raw, ((0, 0), (0, pad_v), (0, 0)), mode="constant", constant_values=0.0)
            mask_np = np.pad(mask_np, ((0, 0), (0, pad_v)), mode="constant", constant_values=0)
            first_vel_raw = np.pad(first_vel_raw, ((0, pad_v), (0, 0)), mode="constant", constant_values=0.0)
            if first_nrm_raw is not None:
                first_nrm_raw = np.pad(first_nrm_raw, ((0, pad_v), (0, 0)), mode="constant", constant_values=0.0)
            object_ids_np = np.pad(
                object_ids_np,
                ((0, pad_v),),
                mode="constant",
                constant_values=int(args.precomp_pad_object_id),
            )

        mask_f = (mask_np > 0).astype(np.float32, copy=False)
        num_present_vertices = int(mask_f[0].sum().item())
        object_ids_mapped = object_ids_np.copy()
        object_ids_mapped[object_ids_mapped == int(args.precomp_pad_object_id)] = int(model_pad_object_id)

        if object_masses_np is not None:
            masses = np.asarray(object_masses_np, dtype=np.float64).reshape(-1)
            mass_details = []
        else:
            masses, mass_details = _load_object_masses(
                meta_path,
                default_density=float(args.density),
                template_dir=str(args.mesh_template_dir),
                volume_cache=volume_cache,
            )
        if int(masses.shape[0]) != int(num_objects):
            raise ValueError(
                f"Metadata/NPZ num_objects mismatch for selector={selector}: "
                f"metadata masses={masses.shape[0]} npz num_objects={num_objects}"
            )
        if bool(args.store_object_mass_details):
            mass_details_by_selector[selector] = mass_details

        cond_pos = _normalize_positions(
            gt_raw[0],
            coord_scale=coord_scale,
            coord_shift=coord_shift,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )
        cond_vel = _normalize_velocities(first_vel_raw, coord_scale=coord_scale, norm_std=norm_std)
        cond_parts = [cond_pos, cond_vel]
        if first_nrm_raw is not None:
            cond_parts.append(first_nrm_raw.astype(np.float32, copy=False))
        cond_first_np = np.concatenate(cond_parts, axis=-1).astype(np.float32)

        cond_first = torch.from_numpy(cond_first_np).to(device=device, dtype=torch.float32)[None, :, :]
        mask_t = torch.from_numpy(mask_f).to(device=device, dtype=torch.float32)[None, :, :]
        object_ids_t = torch.from_numpy(object_ids_mapped).to(device=device, dtype=torch.long)[None, :]
        gt_raw_t = torch.from_numpy(gt_raw).to(device=device, dtype=torch.float32)[None, :, :, :] * mask_t.unsqueeze(-1)
        labels = torch.zeros((1,), device=device, dtype=torch.long)

        generation_losses: list[PerGenerationLosses] = []
        gt_initial_momentum_norm_sq = float("nan")
        gt_momentum_drift_norm_mean = float("nan")
        gt_drift_denominator = float("nan")
        gt_drift_denominator_was_clamped = False
        pred_momentum_drift_norm_means: list[float] = []

        for gen_idx in range(int(args.num_generations)):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                x_gen_norm, _ = model.generate(
                    labels,
                    num_frames=int(infer_num_frames),
                    num_vertices=int(infer_num_vertices),
                    cond_first_frame=cond_first,
                    mask=mask_t,
                    object_ids=object_ids_t,
                )

            x_gen_raw = _denormalize_positions_t(
                x_gen_norm.to(dtype=torch.float32),
                coord_scale=coord_scale,
                coord_shift=coord_shift,
                norm_mean_t=norm_mean_t,
                norm_std_t=norm_std_t,
            )
            x_gen_raw = x_gen_raw * mask_t.unsqueeze(-1)

            mse = float(masked_mse(x_gen_raw, gt_raw_t, mask_t).item())
            rigidity = _rigidity_loss(
                x_gen_raw,
                gt_raw_t,
                mask_t,
                object_ids_t,
                pad_object_id=int(model_pad_object_id),
                last_frame=int(args.rigidity_last_frame),
            )
            momentum = momentum_drift_ratio(
                x_gen_raw[0].detach().cpu().numpy().astype(np.float32, copy=False),
                gt_raw,
                mask_f,
                object_ids_np,
                masses=masses,
                num_objects=int(num_objects),
                dt=float(args.dt),
                velocity_scheme=str(args.velocity_scheme),
                last_frame=int(args.momentum_last_frame),
                eps=float(args.momentum_eps),
            )
            gt_initial_momentum_norm_sq = float(momentum.gt_initial_momentum_norm_sq)
            gt_momentum_drift_norm_mean = float(momentum.gt_momentum_drift_norm_mean)
            gt_drift_denominator = float(momentum.gt_drift_denominator)
            gt_drift_denominator_was_clamped = bool(momentum.gt_drift_denominator_was_clamped)
            pred_momentum_drift_norm_means.append(float(momentum.pred_momentum_drift_norm_mean))
            generation_losses.append(
                PerGenerationLosses(
                    generation_index=int(gen_idx),
                    mse=mse,
                    rigidity=rigidity,
                    momentum_drift_ratio=float(momentum.momentum_drift_ratio),
                    pred_momentum_drift_norm_mean=float(momentum.pred_momentum_drift_norm_mean),
                )
            )

        mse_mean, mse_std, mse_min, mse_max = _finite_stats([x.mse for x in generation_losses])
        rigidity_mean, rigidity_std, rigidity_min, rigidity_max = _finite_stats([x.rigidity for x in generation_losses])
        momentum_ratio_mean, momentum_ratio_std, momentum_ratio_min, momentum_ratio_max = _finite_stats(
            [x.momentum_drift_ratio for x in generation_losses]
        )

        result = PerSampleLosses(
            selector=selector,
            precomp_path=str(npz_path),
            metadata_path=str(meta_path),
            num_objects=int(num_objects),
            num_present_vertices=int(num_present_vertices),
            masses=[float(x) for x in masses.tolist()],
            losses=generation_losses,
            mse_mean=mse_mean,
            mse_std=mse_std,
            mse_min=mse_min,
            mse_max=mse_max,
            rigidity_mean=rigidity_mean,
            rigidity_std=rigidity_std,
            rigidity_min=rigidity_min,
            rigidity_max=rigidity_max,
            momentum_drift_ratio_mean=momentum_ratio_mean,
            momentum_drift_ratio_std=momentum_ratio_std,
            momentum_drift_ratio_min=momentum_ratio_min,
            momentum_drift_ratio_max=momentum_ratio_max,
            gt_initial_momentum_norm_sq=float(gt_initial_momentum_norm_sq),
            gt_momentum_drift_norm_mean=float(gt_momentum_drift_norm_mean),
            pred_momentum_drift_norm_mean_mean=_nanmean(pred_momentum_drift_norm_means),
            gt_drift_denominator=float(gt_drift_denominator),
            gt_drift_denominator_was_clamped=bool(gt_drift_denominator_was_clamped),
        )
        results.append(result)

        if bool(args.verbose):
            print(
                f"{selector} objects={num_objects} presentV={num_present_vertices} "
                f"mse={mse_mean:.6g}+/-{mse_std:.6g} "
                f"rigidity={rigidity_mean:.6g}+/-{rigidity_std:.6g} "
                f"momentum_drift_ratio={momentum_ratio_mean:.6g}+/-{momentum_ratio_std:.6g}",
                flush=True,
            )

    if volume_cache_path:
        _write_mesh_volume_cache_json(volume_cache_path, volume_cache)

    momentum_summary = _global_momentum_drift_summary(results, eps=float(args.momentum_eps))
    summary = {
        "metric": "physformer_publication_losses",
        "definitions": {
            "mse": "masked mean squared error over generated and ground-truth raw vertex positions",
            "rigidity": "mean Kabsch residual per object over frames 1..rigidity_last_frame against frame 0",
            "momentum_drift_ratio": (
                "For each generation g: mean_i mean_t ||P_pred_{i,g}(t)-P_pred_{i,g}(0)||_2 "
                "/ mean_i mean_t ||P_gt_i(t)-P_gt_i(0)||_2, with t=1..momentum_last_frame. "
                "This matches the mean-drift ratio used for momentum_drift_ratio_by_generation_from_npz.json."
            ),
            "per_sample_momentum_drift_ratio": (
                "Diagnostic only: for one sample and one generation, pred_momentum_drift_norm_mean "
                "/ max(gt_momentum_drift_norm_mean, momentum_eps). The reported aggregate uses "
                "the global ratio above, not the mean of these per-sample ratios."
            ),
        },
        "source_code": {
            "mse_and_rigidity": "eval_publish_losses.py",
            "momentum": "eval_publish_losses.py",
            "consolidated": "eval_publish_losses.py",
        },
        "ckpt": str(ckpt_path),
        "use_ema": bool(args.use_ema),
        "split_file": str(args.split_file),
        "split_name": str(args.split_name),
        "split_groups": str(args.split_groups),
        "num_samples": int(len(results)),
        "num_generations": int(args.num_generations),
        "precomp_root": str(args.precomp_root),
        "data_root": str(args.data_root),
        "mesh_volume_cache_json": str(volume_cache_path),
        "mesh_volume_cache_entries_before": int(initial_volume_cache_size),
        "mesh_volume_cache_entries_after": int(len(volume_cache)),
        "density_fallback": float(args.density),
        "dt": float(args.dt),
        "velocity_scheme": str(args.velocity_scheme),
        "rigidity_last_frame": int(args.rigidity_last_frame),
        "momentum_last_frame": int(args.momentum_last_frame),
        "momentum_eps": float(args.momentum_eps),
        "coord_scale": float(coord_scale),
        "coord_shift": float(coord_shift),
        "norm_mean": [float(v) for v in norm_mean.tolist()],
        "norm_std": [float(v) for v in norm_std.tolist()],
        "sampling_method": str(diff_cfg.sampling_method),
        "num_sampling_steps": int(diff_cfg.num_sampling_steps),
        "model_pad_object_id": int(model_pad_object_id),
        "precomp_pad_object_id": int(args.precomp_pad_object_id),
        "mse_mean": _nanmean([r.mse_mean for r in results]),
        "mse_std_mean": _nanmean([r.mse_std for r in results]),
        "rigidity_mean": _nanmean([r.rigidity_mean for r in results]),
        "rigidity_std_mean": _nanmean([r.rigidity_std for r in results]),
        "gt_initial_momentum_norm_sq_mean": _nanmean([r.gt_initial_momentum_norm_sq for r in results]),
        "gt_drift_denominator_clamped_count": int(sum(r.gt_drift_denominator_was_clamped for r in results)),
        "per_sample_momentum_drift_ratio_mean": _nanmean([r.momentum_drift_ratio_mean for r in results]),
        "per_sample_momentum_drift_ratio_std_mean": _nanmean([r.momentum_drift_ratio_std for r in results]),
        **momentum_summary,
    }

    report: dict[str, Any] = {
        "summary": summary,
        "per_sample": [asdict(r) for r in results],
    }
    if bool(args.store_object_mass_details):
        report["object_mass_details"] = mass_details_by_selector

    out_json = Path(str(args.out_json))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    if str(args.out_tsv).strip():
        _write_tsv(Path(str(args.out_tsv)), results)

    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote JSON report: {out_json}", flush=True)
    if str(args.out_tsv).strip():
        print(f"Wrote TSV table: {Path(str(args.out_tsv))}", flush=True)


if __name__ == "__main__":
    main()
