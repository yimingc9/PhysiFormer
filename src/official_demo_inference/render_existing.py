from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional, Sequence

_MPLCONFIGDIR = Path(__file__).resolve().parents[2] / ".inference_work" / "matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, *args, **kwargs):  # type: ignore[no-redef]
        return iterable if iterable is not None else ()

from official_demo_inference.paths import code_root as _code_root
from official_demo_inference.paths import default_demo_root as _default_demo_root
from official_demo_inference.paths import default_vertex_count_json as _default_vertex_count_json
from physformer.data.multiobj_utils_multiobj import load_mesh_vertex_counts, scene_info_from_metadata
from physformer.data.obj_io import load_obj_vertices_faces


EXCLUDED_DIR_NAMES = {"code", ".inference_work", "__pycache__"}
SPLIT_DIR_ALIASES = {
    "ood": "ood",
    "ood_examples": "ood",
    "indistribution": "indistribution",
}
COLORS = [
    (0.86, 0.24, 0.20, 1.0),
    (0.20, 0.64, 0.42, 1.0),
    (0.20, 0.44, 0.86, 1.0),
    (0.92, 0.67, 0.22, 1.0),
    (0.62, 0.32, 0.76, 1.0),
]
NAMED_COLORS = {
    "bunny": (0.82, 0.76, 0.02, 1.0),
    "cow": (0.00, 0.72, 0.78, 1.0),
    "fish": (0.62, 0.80, 0.20, 1.0),
    "horse": (0.90, 0.32, 0.26, 1.0),
    "teapot": (0.68, 0.34, 0.82, 1.0),
}
MESH_EDGE_COLOR = (0.05, 0.06, 0.07, 0.62)
LIGHT_DIRECTION = np.asarray([0.45, -0.65, 0.75], dtype=np.float32)


def _scalar_str(data: np.lib.npyio.NpzFile, key: str) -> Optional[str]:
    if key not in data.files:
        return None
    value = data[key]
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        if value.size == 1:
            return str(value.reshape(-1)[0])
    return str(value)


def _sample_dir_from_vertices_path(vertices_path: Path) -> Path:
    if re.fullmatch(r"(?:gen|sample)_\d{2}", vertices_path.parent.name):
        return vertices_path.parent.parent
    return vertices_path.parent


def _natural_path_key(path: Path) -> tuple[object, ...]:
    parts: list[object] = []
    for part in path.parts:
        parts.extend(int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", part) if text)
    return tuple(parts)


def discover_vertices(root: Path, include: str, generations: set[int] | None) -> list[Path]:
    out: list[Path] = []
    for path in sorted(root.glob("**/vertices.npz"), key=_natural_path_key):
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        parts = rel.parts
        if any(part in EXCLUDED_DIR_NAMES for part in parts):
            continue
        split = SPLIT_DIR_ALIASES.get(parts[0], "other") if parts else "other"
        if include != "all" and split != include:
            continue
        if generations is not None:
            parent_name = path.parent.name
            match = re.fullmatch(r"(?:gen|sample)_(\d{2})", parent_name)
            if match is None:
                if 0 not in generations:
                    continue
            else:
                gen_idx = int(match.group(1))
                if gen_idx not in generations:
                    continue
        out.append(path)
    return out


def _fixed_limits_from_metadata(meta: dict) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    bounds_min = np.asarray(meta.get("bounds_min", [-1.0, -1.0, -1.0]), dtype=np.float32)
    bounds_max = np.asarray(meta.get("bounds_max", [1.0, 1.0, 1.0]), dtype=np.float32)
    if bounds_min.shape != (3,) or bounds_max.shape != (3,):
        return ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
    return tuple((float(bounds_min[i]), float(bounds_max[i])) for i in range(3))  # type: ignore[return-value]


def _fixed_limits_from_cli(raw: str) -> Optional[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]:
    text = str(raw or "").strip()
    if not text:
        return None
    parts = [p for p in text.replace(";", ",").split(",") if p.strip()]
    if len(parts) != 6:
        raise ValueError(
            "--viz-fixed-limits must contain 6 comma-separated numbers: xmin,xmax,ymin,ymax,zmin,zmax; "
            f"got {raw!r}"
        )
    vals = [float(p.strip()) for p in parts]
    limits = ((vals[0], vals[1]), (vals[2], vals[3]), (vals[4], vals[5]))
    for lo, hi in limits:
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError(f"Invalid --viz-fixed-limits range: {raw!r}")
    return limits


def _gt_frame_paths(sample_dir: Path) -> list[Path]:
    meshes_dir = sample_dir / "meshes"
    if not meshes_dir.is_dir():
        raise FileNotFoundError(f"Missing meshes directory: {meshes_dir}")
    out = sorted(meshes_dir.glob("combined_frame_*.obj"))
    if not out:
        raise FileNotFoundError(f"No combined_frame_*.obj files found under: {meshes_dir}")
    return out


def _load_gt_vertices(frame_paths: Sequence[Path], num_frames: int) -> np.ndarray:
    if len(frame_paths) < int(num_frames):
        raise ValueError(f"Need {num_frames} GT frames, found only {len(frame_paths)}")
    frames: list[np.ndarray] = []
    for path in frame_paths[: int(num_frames)]:
        verts, _ = load_obj_vertices_faces(str(path))
        frames.append(verts.astype(np.float32, copy=False))
    return np.stack(frames, axis=0).astype(np.float32, copy=False)


def _faces_by_object(first_obj_path: Path, vertex_slices: Sequence[tuple[int, int]], total_vertices: int) -> list[np.ndarray]:
    vertices, faces = load_obj_vertices_faces(str(first_obj_path))
    if int(vertices.shape[0]) != int(total_vertices):
        raise ValueError(
            f"Combined first-frame OBJ vertex count mismatch: obj has V={int(vertices.shape[0])} "
            f"but metadata sum is V={int(total_vertices)}. obj={first_obj_path}"
        )
    out: list[np.ndarray] = []
    for start, end in vertex_slices:
        start = int(start)
        end = int(end)
        in_range = (faces >= start) & (faces < end)
        keep = np.all(in_range, axis=1)
        obj_faces = faces[keep] - start
        if obj_faces.size == 0:
            raise ValueError(f"No faces found for object vertex slice ({start}, {end}) in {first_obj_path}")
        out.append(obj_faces.astype(np.int64))
    return out


def _shaded_facecolors(vertices: np.ndarray, faces: np.ndarray, base_color: tuple[float, float, float, float]) -> np.ndarray:
    tris = vertices[faces]
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    light = LIGHT_DIRECTION / np.linalg.norm(LIGHT_DIRECTION)
    intensity = 0.42 + 0.58 * np.clip(normals @ light, 0.0, 1.0)
    base = np.asarray(base_color, dtype=np.float32)
    colors = np.empty((faces.shape[0], 4), dtype=np.float32)
    colors[:, :3] = np.clip(base[:3][None, :] * intensity[:, None] + 0.10 * (1.0 - intensity[:, None]), 0.0, 1.0)
    colors[:, 3] = base[3]
    return colors


def _color_for_object(index: int, object_name: str | None) -> tuple[float, float, float, float]:
    name = str(object_name or "").lower()
    for pattern, color in NAMED_COLORS.items():
        if pattern in name:
            return color
    return COLORS[int(index) % len(COLORS)]


def _object_names_from_scene_json(sample_dir: Path, expected_count: int) -> list[str] | None:
    scene_path = sample_dir / "scene_multiobj.json"
    if not scene_path.is_file():
        return None
    try:
        with scene_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    objects = payload.get("objects") if isinstance(payload, dict) else None
    if not isinstance(objects, list) or len(objects) != int(expected_count):
        return None
    names: list[str] = []
    for obj in objects:
        if not isinstance(obj, dict):
            return None
        parts = [str(obj.get(key, "")) for key in ("name", "mesh_name", "mesh_path", "mesh_source", "mesh_used")]
        names.append(" ".join(part for part in parts if part.strip()))
    return names


def _object_names_from_metadata(meta: dict, expected_count: int) -> list[str] | None:
    objects = meta.get("objects") if isinstance(meta, dict) else None
    if not isinstance(objects, list) or len(objects) != int(expected_count):
        return None
    names: list[str] = []
    for obj in objects:
        if not isinstance(obj, dict):
            return None
        parts = [str(obj.get(key, "")) for key in ("name", "mesh_name", "mesh_used", "mesh_source", "mesh_path")]
        names.append(" ".join(part for part in parts if part.strip()))
    return names


def _render_multiobj_frame(
    vertices_by_obj: Sequence[np.ndarray],
    faces_by_obj: Sequence[np.ndarray],
    *,
    fixed_limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    elev: float,
    azim: float,
    dpi: int,
    title: str = "",
    object_names: Sequence[str] | None = None,
) -> np.ndarray:
    fig = plt.figure(figsize=(5.4, 5.4), dpi=int(dpi), facecolor="#f7f8fb")
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.set_facecolor("#f7f8fb")

    for i, (vertices, faces) in enumerate(zip(vertices_by_obj, faces_by_obj)):
        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int64)
        if vertices.size == 0 or faces.size == 0:
            continue
        object_name = object_names[i] if object_names is not None and i < len(object_names) else None
        facecolors = _shaded_facecolors(vertices, faces, _color_for_object(i, object_name))
        poly = Poly3DCollection(
            vertices[faces],
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
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
    plt.close(fig)
    return image


def _compose_side_by_side(
    left: np.ndarray,
    right: np.ndarray,
    *,
    sample_title: str,
    subset_label: str,
    dpi: int,
) -> np.ndarray:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=int(dpi))
    title = str(sample_title).strip()
    subset = str(subset_label).strip()
    if subset:
        title = f"{subset} | {title}" if title else subset
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    for ax, image, panel_title in zip(axes, [left, right], ["Ground Truth", "Inference"]):
        ax.imshow(image)
        ax.set_title(panel_title, fontsize=11)
        ax.axis("off")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95] if title else None)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    out = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
    plt.close(fig)
    return out


def _save_animation(frames: Sequence[np.ndarray], *, out_gif: Optional[Path], out_mp4: Optional[Path], fps: int) -> None:
    if out_gif is None and out_mp4 is None:
        return
    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception as exc:
        raise RuntimeError("Saving GIF/MP4 requires imageio. Install with: pip install imageio imageio-ffmpeg") from exc

    if out_gif is not None:
        imageio.mimsave(str(out_gif), list(frames), duration=1.0 / max(1, int(fps)), loop=0)
    if out_mp4 is not None:
        try:
            with imageio.get_writer(str(out_mp4), fps=max(1, int(fps)), codec="libx264", quality=8) as writer:
                for frame in frames:
                    writer.append_data(frame)
        except Exception as exc:
            raise RuntimeError("MP4 saving failed. You likely need ffmpeg support: pip install imageio-ffmpeg") from exc


def _paths_from_npz(vertices_path: Path, data: np.lib.npyio.NpzFile) -> tuple[Path, Path, Path]:
    sample_dir = _sample_dir_from_vertices_path(vertices_path)
    cond_sample_dir = Path(_scalar_str(data, "cond_sample_dir") or sample_dir)
    metadata_path = Path(_scalar_str(data, "cond_metadata_path") or (cond_sample_dir / "metadata.json"))

    if not cond_sample_dir.is_dir():
        cond_sample_dir = sample_dir
    if not metadata_path.is_file():
        metadata_path = cond_sample_dir / "metadata.json"
    if not metadata_path.is_file():
        metadata_path = sample_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Could not find metadata for {vertices_path}")

    first_obj_path = cond_sample_dir / "meshes" / "combined_frame_000.obj"
    if not first_obj_path.is_file():
        first_obj_path = sample_dir / "meshes" / "combined_frame_000.obj"
    if not first_obj_path.is_file():
        raise FileNotFoundError(f"Could not find first-frame combined OBJ for {vertices_path}")
    return cond_sample_dir, metadata_path, first_obj_path


def render_vertices_file(vertices_path: Path, args: argparse.Namespace, vertex_counts: dict[str, int]) -> bool:
    gen_dir = vertices_path.parent
    pred_gif = gen_dir / "inference.gif" if args.save_gif else None
    pred_mp4 = gen_dir / "inference.mp4" if args.save_mp4 else None
    gt_gif = gen_dir / "GT.gif" if args.save_gt_gif else None
    gt_mp4 = gen_dir / "GT.mp4" if args.save_gt_mp4 else None
    compare_base = Path(str(args.compare_out_name)).stem
    compare_gif = gen_dir / f"{compare_base}.gif" if args.save_compare_gif else None
    compare_mp4 = gen_dir / f"{compare_base}.mp4" if args.save_compare_mp4 else None
    requested = [p for p in [pred_gif, pred_mp4, gt_gif, gt_mp4, compare_gif, compare_mp4] if p is not None]
    if requested and all(p.is_file() for p in requested) and not bool(args.overwrite):
        print(f"[SKIP] renders exist: {gen_dir}", flush=True)
        return False

    with np.load(vertices_path, allow_pickle=False) as data:
        if "vertices" not in data.files:
            raise KeyError(f"{vertices_path} does not contain a 'vertices' array")
        vertices = np.asarray(data["vertices"], dtype=np.float32)
        cond_sample_dir, metadata_path, first_obj_path = _paths_from_npz(vertices_path, data)

    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError(f"Expected vertices shape (F,V,3), got {tuple(vertices.shape)} in {vertices_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    scene = scene_info_from_metadata(str(metadata_path), vertex_counts=vertex_counts, max_num_objects=int(args.max_num_objects))
    if int(vertices.shape[1]) < int(scene.total_vertices):
        raise ValueError(
            f"{vertices_path} has V={int(vertices.shape[1])}, but metadata expects V={int(scene.total_vertices)}"
        )

    pred_vertices = vertices[:, : int(scene.total_vertices), :].astype(np.float32, copy=False)
    faces_by_obj = _faces_by_object(first_obj_path, scene.vertex_slices, int(scene.total_vertices))
    sample_dir = _sample_dir_from_vertices_path(vertices_path)
    object_names = _object_names_from_metadata(meta, len(scene.vertex_slices))
    if object_names is None:
        object_names = _object_names_from_scene_json(sample_dir, len(scene.vertex_slices))
    if object_names is None:
        object_names = [
            f"{str(name)} {str(path)}"
            for name, path in zip(scene.mesh_names, scene.mesh_paths)
        ]
    fixed_limits = _fixed_limits_from_cli(str(args.viz_fixed_limits)) or _fixed_limits_from_metadata(meta)

    gt_vertices = None
    if args.save_gt_gif or args.save_gt_mp4 or args.save_compare_gif or args.save_compare_mp4:
        gt_vertices = _load_gt_vertices(_gt_frame_paths(cond_sample_dir), int(pred_vertices.shape[0]))
        if int(gt_vertices.shape[1]) != int(scene.total_vertices):
            raise ValueError(
                f"GT vertices have V={int(gt_vertices.shape[1])}, but metadata expects V={int(scene.total_vertices)}. "
                f"sample_dir={cond_sample_dir}"
            )

    pred_frames: list[np.ndarray] = []
    gt_frames: list[np.ndarray] = []
    compare_frames: list[np.ndarray] = []
    frame_range = range(int(pred_vertices.shape[0]))
    if not bool(args.verbose):
        frame_range = tqdm(frame_range, desc=f"render[{vertices_path.parent.name}]", unit="frame", leave=False)

    sample_title = str(sample_dir.relative_to(args.demo_root))
    for frame_idx in frame_range:
        pred_by_obj = [pred_vertices[frame_idx, start:end, :] for start, end in scene.vertex_slices]
        pred_img = _render_multiobj_frame(
            pred_by_obj,
            faces_by_obj,
            fixed_limits=fixed_limits,
            elev=float(args.viz_elev),
            azim=float(args.viz_azim),
            dpi=int(args.compare_render_dpi if (args.save_compare_gif or args.save_compare_mp4) else args.render_dpi),
            title="inference" if (args.save_gif or args.save_mp4) else "",
            object_names=object_names,
        )
        if args.save_gif or args.save_mp4:
            pred_frames.append(pred_img)
        if args.save_gt_gif or args.save_gt_mp4 or args.save_compare_gif or args.save_compare_mp4:
            assert gt_vertices is not None
            gt_by_obj = [gt_vertices[frame_idx, start:end, :] for start, end in scene.vertex_slices]
            gt_img = _render_multiobj_frame(
                gt_by_obj,
                faces_by_obj,
                fixed_limits=fixed_limits,
                elev=float(args.viz_elev),
                azim=float(args.viz_azim),
                dpi=int(args.compare_render_dpi),
                title="GT" if (args.save_gt_gif or args.save_gt_mp4) else "",
                object_names=object_names,
            )
            if args.save_gt_gif or args.save_gt_mp4:
                gt_frames.append(gt_img)
        if args.save_compare_gif or args.save_compare_mp4:
            compare_frames.append(
                _compose_side_by_side(
                    gt_img,
                    pred_img,
                    sample_title=sample_title,
                    subset_label=str(args.compare_subset_label),
                    dpi=int(args.compare_compose_dpi),
                )
            )

    if args.save_gif or args.save_mp4:
        _save_animation(
            pred_frames,
            out_gif=pred_gif if (pred_gif is not None and (args.overwrite or not pred_gif.exists())) else None,
            out_mp4=pred_mp4 if (pred_mp4 is not None and (args.overwrite or not pred_mp4.exists())) else None,
            fps=int(args.fps),
        )
    if args.save_gt_gif or args.save_gt_mp4:
        _save_animation(
            gt_frames,
            out_gif=gt_gif if (gt_gif is not None and (args.overwrite or not gt_gif.exists())) else None,
            out_mp4=gt_mp4 if (gt_mp4 is not None and (args.overwrite or not gt_mp4.exists())) else None,
            fps=int(args.fps),
        )
    if args.save_compare_gif or args.save_compare_mp4:
        _save_animation(
            compare_frames,
            out_gif=compare_gif if (compare_gif is not None and (args.overwrite or not compare_gif.exists())) else None,
            out_mp4=compare_mp4 if (compare_mp4 is not None and (args.overwrite or not compare_mp4.exists())) else None,
            fps=int(args.fps),
        )
    print(f"[RENDERED] {vertices_path}", flush=True)
    return True


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render existing official-demo vertices.npz files without running inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--demo-root", type=Path, default=_default_demo_root())
    parser.add_argument("--mesh-vertex-count-json", type=Path, default=_default_vertex_count_json())
    parser.add_argument("--include", choices=["all", "ood", "indistribution"], default="all")
    parser.add_argument("--generation", type=int, action="append", default=None, help="Only render a rollout sample index, e.g. 0 for sample_00. Can be repeated.")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit the number of vertices.npz files rendered; 0 means no limit.")
    parser.add_argument("--max-num-objects", type=int, default=10)
    parser.add_argument("--save-gif", action="store_true")
    parser.add_argument("--save-mp4", action="store_true")
    parser.add_argument("--save-gt-gif", action="store_true")
    parser.add_argument("--save-gt-mp4", action="store_true")
    parser.add_argument("--save-compare-gif", action="store_true")
    parser.add_argument("--save-compare-mp4", action="store_true")
    parser.add_argument("--compare-out-name", default="traj_compare_gt_vs_infer")
    parser.add_argument("--compare-subset-label", default="")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--render-dpi", type=int, default=160)
    parser.add_argument("--compare-render-dpi", type=int, default=160)
    parser.add_argument("--compare-compose-dpi", type=int, default=160)
    parser.add_argument("--viz-elev", type=float, default=30.0)
    parser.add_argument("--viz-azim", type=float, default=-45.0)
    parser.add_argument("--viz-fixed-limits", default="")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    args.demo_root = args.demo_root.expanduser().resolve()
    args.mesh_vertex_count_json = args.mesh_vertex_count_json.expanduser().resolve()

    if not args.demo_root.is_dir():
        parser.error(f"--demo-root is not a directory: {args.demo_root}")
    if not args.mesh_vertex_count_json.is_file():
        parser.error(f"--mesh-vertex-count-json is not a file: {args.mesh_vertex_count_json}")
    if not (args.save_gif or args.save_mp4 or args.save_gt_gif or args.save_gt_mp4 or args.save_compare_gif or args.save_compare_mp4):
        parser.error("Choose at least one render output: --save-mp4, --save-gif, --save-gt-mp4, --save-gt-gif, --save-compare-mp4, or --save-compare-gif")

    generations = set(args.generation) if args.generation is not None else None
    vertex_paths = discover_vertices(args.demo_root, args.include, generations)
    if args.max_samples and int(args.max_samples) > 0:
        vertex_paths = vertex_paths[: int(args.max_samples)]
    if not vertex_paths:
        parser.error(f"No vertices.npz files found under {args.demo_root}")

    vertex_counts = load_mesh_vertex_counts(str(args.mesh_vertex_count_json))

    rendered = 0
    iterator = vertex_paths if args.verbose else tqdm(vertex_paths, desc="outputs", unit="npz")
    for vertices_path in iterator:
        rendered += int(render_vertices_file(vertices_path, args, vertex_counts))
    print(f"[done] rendered {rendered} of {len(vertex_paths)} vertices.npz files", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
