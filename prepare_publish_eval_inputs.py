from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

CODE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from physformer.data.obj_io import load_obj_vertices_faces


DEFAULT_EVAL_ASSETS_ROOT = CODE_ROOT / "eval_assets"
DEFAULT_PRECOMP_ROOT = DEFAULT_EVAL_ASSETS_ROOT / "eval_precomp"
DEFAULT_DATA_ROOT = DEFAULT_EVAL_ASSETS_ROOT / "eval_data"
DEFAULT_SPLIT_FILE = DEFAULT_EVAL_ASSETS_ROOT / "eval_split.json"
DEFAULT_SAMPLE_NAMES = "2obj_elastic,3obj_rigid,4obj_elastic,5obj_rigid"


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


def _mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    verts = np.asarray(vertices, dtype=np.float64)
    faces_i64 = np.asarray(faces, dtype=np.int64)
    tris = verts[faces_i64]
    signed_volume = float(np.sum(np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2]))) / 6.0)
    return abs(signed_volume)


def _resolve_mesh_path(mesh_used: str, *, template_dir: Path | None) -> Path:
    raw = Path(mesh_used)
    candidates = [raw, CODE_ROOT / raw]
    if template_dir is not None:
        candidates.extend([template_dir / raw.name, template_dir / f"{raw.stem}.obj"])
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not resolve mesh path {mesh_used!r}; tried {', '.join(map(str, candidates))}")


def _load_object_masses(meta: dict[str, Any], *, default_density: float, template_dir: Path | None) -> np.ndarray:
    objects = meta.get("objects", None)
    if not isinstance(objects, list) or not objects:
        raise ValueError("metadata.json must contain a non-empty objects list")
    masses: list[float] = []
    for obj_idx, obj_any in enumerate(objects):
        if not isinstance(obj_any, dict):
            raise ValueError(f"objects[{obj_idx}] must be a dict")
        mesh_used = obj_any.get("mesh_used") or obj_any.get("mesh_source")
        if not isinstance(mesh_used, str) or not mesh_used:
            raise ValueError(f"objects[{obj_idx}] needs mesh_used or mesh_source")
        mesh_path = _resolve_mesh_path(mesh_used, template_dir=template_dir)
        verts, faces = load_obj_vertices_faces(str(mesh_path))
        volume = _mesh_volume(verts, faces)
        scale = _safe_float(obj_any.get("scale"), 1.0)
        density = _object_density(meta, obj_any, default_density)
        mass = float(density) * float(volume) * float(scale) ** 3
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError(f"Invalid object mass for object {obj_idx}: {mass}")
        masses.append(mass)
    return np.asarray(masses, dtype=np.float64)


def _object_ids_from_metadata(
    meta: dict[str, Any],
    *,
    total_vertices: int,
    template_dir: Path | None,
) -> np.ndarray:
    object_ids = np.full((int(total_vertices),), -1, dtype=np.int64)
    objects = meta.get("objects", None)
    if not isinstance(objects, list) or not objects:
        raise ValueError("metadata.json must contain a non-empty objects list")
    cursor = 0
    for obj_idx, obj_any in enumerate(objects):
        if not isinstance(obj_any, dict):
            raise ValueError(f"objects[{obj_idx}] must be a dict")
        vertex_range = obj_any.get("vertex_range")
        if isinstance(vertex_range, list) and len(vertex_range) == 2:
            start, end = int(vertex_range[0]), int(vertex_range[1])
        else:
            count = int(obj_any.get("vertex_count", 0))
            if count <= 0:
                mesh_used = obj_any.get("mesh_used") or obj_any.get("mesh_source")
                if isinstance(mesh_used, str) and mesh_used:
                    mesh_path = _resolve_mesh_path(mesh_used, template_dir=template_dir)
                    mesh_vertices, _ = load_obj_vertices_faces(str(mesh_path))
                    count = int(mesh_vertices.shape[0])
            if count <= 0:
                raise ValueError(f"objects[{obj_idx}] needs vertex_range or positive vertex_count")
            start, end = cursor, cursor + count
        if start < 0 or end > int(total_vertices) or end <= start:
            raise ValueError(f"Invalid vertex range for object {obj_idx}: {start}, {end}")
        object_ids[start:end] = int(obj_idx)
        cursor = end
    if np.any(object_ids < 0):
        missing = int(np.where(object_ids < 0)[0][0])
        raise ValueError(f"No object id assigned for vertex {missing}")
    return object_ids


def _load_gt_vertices(meshes_dir: Path) -> np.ndarray:
    frame_paths = sorted(meshes_dir.glob("combined_frame_*.obj"))
    if not frame_paths:
        frame_paths = sorted(meshes_dir.glob("*.obj"))
    if not frame_paths:
        raise FileNotFoundError(f"No OBJ frames found in {meshes_dir}")
    frames: list[np.ndarray] = []
    for path in frame_paths:
        vertices, _ = load_obj_vertices_faces(str(path))
        frames.append(vertices.astype(np.float32, copy=False))
    first_shape = frames[0].shape
    for path, vertices in zip(frame_paths, frames):
        if vertices.shape != first_shape:
            raise ValueError(f"Frame vertex shape mismatch in {path}: {vertices.shape} != {first_shape}")
    return np.stack(frames, axis=0).astype(np.float32, copy=False)


def _parse_sample_names(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise ValueError("Expected at least one sample name")
    return names


def _relative_to_code(path: Path) -> str:
    try:
        return path.resolve().relative_to(CODE_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _publish_metadata(sample_name: str, meta: dict[str, Any], object_masses: np.ndarray, object_ids: np.ndarray) -> dict[str, Any]:
    objects: list[dict[str, Any]] = []
    for obj_idx, mass in enumerate(object_masses.tolist()):
        where = np.where(object_ids == int(obj_idx))[0]
        vertex_range = [int(where[0]), int(where[-1]) + 1] if where.size else [0, 0]
        objects.append(
            {
                "object_index": int(obj_idx),
                "mass": float(mass),
                "vertex_range": vertex_range,
            }
        )
    return {
        "sample_name": str(sample_name),
        "num_objects": int(len(objects)),
        "frames": int(meta.get("frames", 0) or 0),
        "dt": float(meta.get("dt", 0.0) or 0.0),
        "steps_per_frame": int(meta.get("steps_per_frame", 0) or 0),
        "original_dataset": {
            "category": meta.get("category"),
            "sample_index": meta.get("sample_index"),
            "mode": meta.get("mode"),
        },
        "objects": objects,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Regenerate compact publication loss-evaluation inputs from raw ground-truth sample folders."
    )
    parser.add_argument(
        "--sample_root",
        type=Path,
        default=None,
        required=True,
        help=(
            "Directory containing raw sample folders such as 2obj_elastic and 3obj_rigid. "
            "Raw samples are not packaged by default; eval_assets/ is already included for normal evaluation."
        ),
    )
    parser.add_argument("--sample_names", type=str, default=DEFAULT_SAMPLE_NAMES)
    parser.add_argument("--precomp_root", type=Path, default=DEFAULT_PRECOMP_ROOT)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split_file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--split_name", type=str, default="test")
    parser.add_argument("--density", type=float, default=1000.0)
    parser.add_argument("--mesh_template_dir", type=Path, default=None)
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Regenerate outputs even when eval precomp/data files already exist.",
    )
    return parser


def prepare_eval_inputs(args: argparse.Namespace) -> list[str]:
    args.sample_root = args.sample_root.expanduser().resolve()
    if not args.sample_root.is_dir():
        raise FileNotFoundError(f"--sample_root is not a directory: {args.sample_root}")
    sample_names = _parse_sample_names(str(args.sample_names))
    selectors: list[str] = []
    prepared_any = False

    for sample_name in sample_names:
        selector = f"{sample_name}:0"
        selectors.append(selector)
        out_npz = args.precomp_root / sample_name / "sample_000000.npz"
        out_sample = args.data_root / sample_name / "sample_000000"
        out_metadata = out_sample / "metadata.json"
        if out_npz.is_file() and out_metadata.is_file() and not bool(args.overwrite):
            print(f"eval assets already present for {selector}; skipping precompute: {out_npz}")
            continue

        src_sample_dir = args.sample_root / sample_name
        metadata_path = src_sample_dir / "metadata.json"
        meshes_dir = src_sample_dir / "meshes"
        velocity_path = src_sample_dir / "vertex_velocities" / "combined_frame_000.npy"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing metadata: {metadata_path}")
        if not meshes_dir.is_dir():
            raise FileNotFoundError(f"Missing meshes dir: {meshes_dir}")
        if not velocity_path.is_file():
            raise FileNotFoundError(f"Missing first-frame velocity: {velocity_path}")

        with metadata_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        vertices = _load_gt_vertices(meshes_dir)
        first_vel = np.load(velocity_path).astype(np.float32, copy=False)
        if first_vel.shape != vertices.shape[1:]:
            raise ValueError(f"Velocity shape mismatch for {sample_name}: {first_vel.shape} != {vertices.shape[1:]}")
        object_ids = _object_ids_from_metadata(
            meta,
            total_vertices=int(vertices.shape[1]),
            template_dir=args.mesh_template_dir,
        )
        num_objects = int(meta.get("num_objects", len(meta.get("objects", []))))
        object_masses = _load_object_masses(
            meta,
            default_density=float(args.density),
            template_dir=args.mesh_template_dir,
        )
        if int(object_masses.shape[0]) != int(num_objects):
            raise ValueError(f"num_objects mismatch for {sample_name}: masses={object_masses.shape[0]} meta={num_objects}")

        out_npz.parent.mkdir(parents=True, exist_ok=True)
        out_sample.mkdir(parents=True, exist_ok=True)
        publish_meta = _publish_metadata(sample_name, meta, object_masses, object_ids)
        with out_metadata.open("w", encoding="utf-8") as f:
            json.dump(publish_meta, f, indent=2)
            f.write("\n")
        np.savez_compressed(
            out_npz,
            vertices=vertices,
            mask=np.ones(vertices.shape[:2], dtype=np.uint8),
            object_ids=object_ids.astype(np.int64, copy=False),
            num_objects=np.asarray(num_objects, dtype=np.int64),
            first_frame_velocity=first_vel,
            object_masses=object_masses,
            source_sample_name=np.asarray(sample_name),
        )
        prepared_any = True
        print(f"prepared {selector}: {out_npz}")

    payload = {
        "description": "Official demo four-sample publication evaluation split.",
        "dataset_root": _relative_to_code(args.data_root),
        str(args.split_name): selectors,
    }
    if not prepared_any and args.split_file.is_file() and not bool(args.overwrite):
        print(f"split already present; skipping write: {args.split_file}")
        print(f"precomp_root: {args.precomp_root}")
        print(f"data_root: {args.data_root}")
        return selectors
    args.split_file.parent.mkdir(parents=True, exist_ok=True)
    with args.split_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"wrote split: {args.split_file}")
    print(f"precomp_root: {args.precomp_root}")
    print(f"data_root: {args.data_root}")
    return selectors


def main() -> None:
    prepare_eval_inputs(build_argparser().parse_args())


if __name__ == "__main__":
    main()
