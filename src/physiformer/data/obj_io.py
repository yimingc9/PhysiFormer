from __future__ import annotations

import os
from itertools import product
from typing import Dict, List, Tuple

import numpy as np


def load_obj_vertices_faces(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimal OBJ loader (positions + faces).
    - Supports `v` and `f` lines.
    - Faces are triangulated via fan triangulation.
    """
    vertices: List[List[float]] = []
    faces: List[List[int]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                if len(parts) < 3:
                    continue
                face = []
                for p in parts:
                    idx_str = p.split("/")[0]
                    if not idx_str:
                        continue
                    idx = int(idx_str)
                    # OBJ indices are 1-based; negatives are relative to end
                    if idx < 0:
                        idx = len(vertices) + idx
                    else:
                        idx = idx - 1
                    face.append(idx)
                if len(face) >= 3:
                    faces.append(face)

    if not vertices or not faces:
        raise ValueError(f"OBJ has no vertices/faces: {path}")
    verts = np.asarray(vertices, dtype=np.float32)

    tri_faces = []
    for face in faces:
        v0 = face[0]
        for i in range(1, len(face) - 1):
            tri_faces.append([v0, face[i], face[i + 1]])

    if not tri_faces:
        raise ValueError(f"OBJ has no triangulated faces: {path}")
    return verts, np.asarray(tri_faces, dtype=np.int64)


def vertices_faces_to_triangles(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tris = vertices[faces]  # (T,3,3)
    return tris.reshape(tris.shape[0], 9).astype(np.float32)


def load_obj_triangles(path: str) -> np.ndarray:
    v, f = load_obj_vertices_faces(path)
    return vertices_faces_to_triangles(v, f)


def triangles_to_vertices_faces(triangles: np.ndarray, tolerance: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts (T,9) triangles back to a vertex list + face indices by merging near-duplicate vertices.
    """
    triangles = np.asarray(triangles, dtype=np.float32)
    if triangles.ndim != 2 or triangles.shape[1] != 9:
        raise ValueError(f"Expected triangles shape (T,9), got {triangles.shape}")

    vertices_list: List[np.ndarray] = []
    faces: List[List[int]] = []

    # Hash buckets of vertices on a tolerance grid to avoid O(N^2) allclose checks.
    # We still verify with np.allclose, but only against candidates in neighboring buckets.
    if tolerance <= 0:
        raise ValueError(f"tolerance must be > 0, got {tolerance}")
    inv_tol = 1.0 / float(tolerance)
    buckets: Dict[Tuple[int, int, int], List[int]] = {}

    neighbor_offsets = list(product((-1, 0, 1), repeat=3))

    def _bucket_key(v: np.ndarray) -> Tuple[int, int, int]:
        # floor-based bucket; neighbor search handles boundary cases.
        q = np.floor(v * inv_tol).astype(np.int64)
        return int(q[0]), int(q[1]), int(q[2])

    def find_vertex_idx(vertex: np.ndarray) -> int:
        key = _bucket_key(vertex)
        for dx, dy, dz in neighbor_offsets:
            cand_key = (key[0] + dx, key[1] + dy, key[2] + dz)
            for idx in buckets.get(cand_key, ()):
                if np.allclose(vertex, vertices_list[idx], atol=tolerance):
                    return idx
        idx = len(vertices_list)
        vertices_list.append(np.array(vertex, dtype=np.float32, copy=True))
        buckets.setdefault(key, []).append(idx)
        return idx

    for tri in triangles:
        tri3 = tri.reshape(3, 3)
        face_idx = [find_vertex_idx(v) for v in tri3]
        faces.append(face_idx)

    vertices = np.stack(vertices_list, axis=0).astype(np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    return vertices, faces


def save_obj(path: str, triangles: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    v, f = triangles_to_vertices_faces(triangles)
    with open(path, "w", encoding="utf-8") as fp:
        for vert in v:
            fp.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")
        for face in f:
            # OBJ is 1-indexed
            fp.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def save_obj_vertices_faces(path: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    """
    Saves an OBJ from explicit vertices and faces (triangles).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected vertices shape (V,3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected faces shape (F,3), got {faces.shape}")

    with open(path, "w", encoding="utf-8") as fp:
        for vert in vertices:
            fp.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")
        for face in faces:
            fp.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def fix_num_triangles(
    triangles: np.ndarray,
    *,
    num_triangles: int,
    triangle_sampling: str,
    pad_value: float,
    sample_idx: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Enforces a fixed number of triangles.
    Returns: (triangles_fixed, mask, sample_idx_used)
      - mask is (num_triangles,) with 1 for real triangles, 0 for padding.
      - if input has more triangles and sample_idx is None, generates sample_idx (for temporal consistency).
    """
    t = triangles.shape[0]
    if t == num_triangles:
        return triangles, np.ones((num_triangles,), dtype=np.float32), sample_idx

    if t > num_triangles:
        if sample_idx is not None and int(sample_idx.max()) >= t:
            # Topology / face count changed across frames; fall back to re-sampling.
            sample_idx = None
        if sample_idx is None:
            if triangle_sampling == "random":
                sample_idx = np.random.choice(t, num_triangles, replace=False)
            elif triangle_sampling == "first":
                sample_idx = np.arange(num_triangles)
            else:
                raise ValueError(f"Unknown triangle_sampling: {triangle_sampling}")
        triangles = triangles[sample_idx]
        return triangles, np.ones((num_triangles,), dtype=np.float32), sample_idx

    pad = np.full((num_triangles - t, 9), pad_value, dtype=triangles.dtype)
    out = np.concatenate([triangles, pad], axis=0)
    mask = np.zeros((num_triangles,), dtype=np.float32)
    mask[:t] = 1.0
    return out, mask, sample_idx


def fix_num_faces(
    faces: np.ndarray,
    *,
    num_faces: int,
    face_sampling: str,
    sample_idx: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Enforces a fixed number of faces (triangles) for a connectivity template.
    Returns: (faces_fixed, mask, sample_idx_used)
      - mask is (num_faces,) with 1 for real faces, 0 for padding.
      - sample_idx_used is the indices into the original faces when sub-sampling.
    """
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected faces shape (F,3), got {faces.shape}")

    f = faces.shape[0]
    if f == num_faces:
        return faces, np.ones((num_faces,), dtype=np.float32), sample_idx

    if f > num_faces:
        if sample_idx is None:
            if face_sampling == "random":
                sample_idx = np.random.choice(f, num_faces, replace=False)
            elif face_sampling == "first":
                sample_idx = np.arange(num_faces)
            else:
                raise ValueError(f"Unknown face_sampling: {face_sampling}")
        faces = faces[sample_idx]
        return faces, np.ones((num_faces,), dtype=np.float32), sample_idx

    pad = np.zeros((num_faces - f, 3), dtype=np.int64)
    out = np.concatenate([faces, pad], axis=0)
    mask = np.zeros((num_faces,), dtype=np.float32)
    mask[:f] = 1.0
    return out, mask, sample_idx
