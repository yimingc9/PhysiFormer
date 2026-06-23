from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass(frozen=True)
class MultiObjSceneInfo:
    mesh_paths: list[str]
    mesh_names: list[str]
    vertex_counts: list[int]
    vertex_slices: list[tuple[int, int]]  # (start,end) into the combined vertex array
    total_vertices: int


def default_vertex_count_json_path() -> str:
    src_root = Path(__file__).resolve().parents[2]
    code_root = src_root.parent
    candidates = [
        src_root / "official_demo_inference" / "configs" / "vertex_counts_multiobj_all.json",
        code_root / "configs" / "vertex_counts_multiobj_all.json",
        src_root / "mesh_primitives" / "vertex_counts_multiobj_all.json",
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return str(candidates[0])


def load_mesh_vertex_counts(json_path: str) -> dict[str, int]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "counts" in data and isinstance(data["counts"], dict):
        counts = data["counts"]
    elif isinstance(data, dict):
        # Back-compat: allow direct mapping dict[str,int].
        counts = data
    else:
        raise ValueError(f"Unsupported JSON format in {json_path}: expected dict, got {type(data)}")

    out: dict[str, int] = {}
    for k, v in counts.items():
        if not isinstance(k, str):
            k = str(k)
        if not isinstance(v, (int, float)) or int(v) <= 0:
            raise ValueError(f"Invalid vertex count for key='{k}' in {json_path}: {v}")
        out[k] = int(v)
    return out


def _mesh_key_candidates(mesh_path: str) -> list[str]:
    p = Path(str(mesh_path))
    stem = p.stem
    name = p.name
    # Some metadata stores absolute paths; some users prefer stem keys.
    # Try several candidates (and their lowercase forms).
    cands = [stem, name, str(p)]
    out: list[str] = []
    for c in cands:
        if c and c not in out:
            out.append(c)
        lc = c.lower()
        if lc and lc not in out:
            out.append(lc)
    return out


def vertex_count_for_mesh(mesh_path: str, vertex_counts: dict[str, int]) -> int:
    for k in _mesh_key_candidates(mesh_path):
        if k in vertex_counts:
            return int(vertex_counts[k])
    raise KeyError(
        f"Mesh '{mesh_path}' not found in vertex count map. Tried keys: {', '.join(_mesh_key_candidates(mesh_path))}"
    )


def _load_metadata(meta_path: str) -> dict:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if not isinstance(meta, dict):
        raise ValueError(f"metadata.json must contain a dict, got {type(meta)}: {meta_path}")
    return meta


def scene_info_from_metadata_dict(
    meta: dict,
    *,
    vertex_counts: dict[str, int],
    max_num_objects: int,
    source: str = "<metadata>",
) -> MultiObjSceneInfo:
    objects = meta.get("objects", None)
    if not isinstance(objects, list) or not objects:
        raise ValueError(f"metadata missing non-empty 'objects' list: {source}")

    mesh_paths: list[str] = []
    mesh_names: list[str] = []
    counts: list[int] = []
    for obj in objects:
        if not isinstance(obj, dict):
            raise ValueError(f"metadata 'objects' entries must be dicts, got {type(obj)}: {source}")
        mesh_used = obj.get("mesh_used") or obj.get("mesh_source")
        if not isinstance(mesh_used, str) or not mesh_used:
            raise ValueError(f"metadata object missing mesh_used/mesh_source string: {source}")
        mesh_paths.append(mesh_used)
        mesh_names.append(Path(mesh_used).stem)
        # Prefer per-object vertex_count stored in metadata when available.
        # This makes inference/precompute robust to new primitive sets without having to regenerate
        # a vertex_counts JSON, while still validating against the JSON when possible.
        meta_vc = obj.get("vertex_count", None)
        meta_vc_int = int(meta_vc) if isinstance(meta_vc, (int, float)) and int(meta_vc) > 0 else None
        try:
            vc = vertex_count_for_mesh(mesh_used, vertex_counts)
        except KeyError:
            if meta_vc_int is None:
                raise
            vc = int(meta_vc_int)
        else:
            if meta_vc_int is not None and int(meta_vc_int) != int(vc):
                raise ValueError(
                    f"metadata vertex_count={int(meta_vc_int)} disagrees with vertex_counts map={int(vc)} for mesh_used={mesh_used!r}. "
                    f"meta={source}"
                )
        counts.append(int(vc))

    num_obj = len(counts)
    if int(num_obj) > int(max_num_objects):
        raise ValueError(
            f"Scene has num_objects={num_obj} but max_num_objects={int(max_num_objects)}. "
            f"Increase --max_num_objects. meta={source}"
        )

    slices: list[tuple[int, int]] = []
    cur = 0
    for v in counts:
        start = cur
        cur += int(v)
        slices.append((start, cur))

    return MultiObjSceneInfo(
        mesh_paths=mesh_paths,
        mesh_names=mesh_names,
        vertex_counts=counts,
        vertex_slices=slices,
        total_vertices=int(cur),
    )


def scene_info_from_metadata(
    meta_path: str,
    *,
    vertex_counts: dict[str, int],
    max_num_objects: int,
) -> MultiObjSceneInfo:
    meta = _load_metadata(meta_path)
    return scene_info_from_metadata_dict(
        meta,
        vertex_counts=vertex_counts,
        max_num_objects=max_num_objects,
        source=meta_path,
    )


def object_ids_from_scene_info(scene: MultiObjSceneInfo) -> list[int]:
    ids: list[int] = []
    for obj_id, v in enumerate(scene.vertex_counts):
        ids.extend([int(obj_id)] * int(v))
    if len(ids) != int(scene.total_vertices):
        raise RuntimeError("object_ids_from_scene_info internal size mismatch")
    return ids


def resolve_metadata_path_from_sequence_dir(sequence_dir: str, *, metadata_filename: str = "metadata.json") -> str:
    """
    Given a frame directory like <sample_dir>/meshes, return <sample_dir>/metadata.json.
    """
    sample_dir = os.path.dirname(str(sequence_dir).rstrip(os.sep))
    return os.path.join(sample_dir, str(metadata_filename))


def resolve_sample_dir_from_first_frame_obj(obj_path: str) -> str:
    """
    Expected layout:
      <sample_dir>/meshes/combined_frame_000.obj
    """
    frame_dir = os.path.dirname(str(obj_path))
    return os.path.dirname(frame_dir)


def resolve_velocity_path_from_first_frame_obj(obj_path: str, *, velocity_dirname: str) -> str:
    sample_dir = resolve_sample_dir_from_first_frame_obj(obj_path)
    stem = os.path.splitext(os.path.basename(obj_path))[0]
    return os.path.join(sample_dir, str(velocity_dirname), f"{stem}.npy")
