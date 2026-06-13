from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from official_demo_inference.paths import code_root as _code_root
from official_demo_inference.paths import default_demo_root as _default_demo_root
from official_demo_inference.paths import default_checkpoint_path
from official_demo_inference.paths import default_vertex_count_json


DEFAULT_CHECKPOINT = default_checkpoint_path()
EXCLUDED_DIR_NAMES = {"code", ".inference_work", "__pycache__"}
SPLIT_DIR_ALIASES = {
    "ood": "ood",
    "ood_examples": "ood",
    "indistribution": "indistribution",
}
LEGACY_CONDITIONED_SAMPLE_ROOT = "conditioned_samples"
ELASTIC_MATERIAL = {
    "kind": "elastic",
    "effective_softness": 1.0,
    "softness": 1.0,
    "friction": 0.15,
    "static_friction": 0.15,
    "kinetic_friction": 0.15,
}
RIGID_MATERIAL = {
    "kind": "rigid",
    "effective_softness": 0.0,
    "softness": 0.0,
    "friction": 0.01,
    "static_friction": 0.01,
    "kinetic_friction": 0.01,
}


def _is_rollout_sample_dir_name(name: str) -> bool:
    return re.fullmatch(r"(?:gen|sample)_\d{2}", str(name)) is not None


def _natural_path_key(path: Path) -> tuple[tuple[int, object], ...]:
    parts: list[tuple[int, object]] = []
    for part in path.parts:
        for text in re.split(r"(\d+)", part):
            if not text:
                continue
            parts.append((0, int(text)) if text.isdigit() else (1, text.lower()))
    return tuple(parts)


@dataclass(frozen=True)
class DemoSample:
    sample_dir: Path
    rel_dir: Path
    split: str


def _flatten(items: Sequence[Sequence[str]] | None) -> list[str]:
    out: list[str] = []
    for item in items or []:
        out.extend(str(x).strip() for x in item if str(x).strip())
    return out


def _is_sample_dir(path: Path) -> bool:
    return (
        (path / "metadata.json").is_file()
        and (path / "meshes" / "combined_frame_000.obj").is_file()
        and (path / "vertex_velocities" / "combined_frame_000.npy").is_file()
    )


def discover_samples(demo_root: Path, include: str) -> list[DemoSample]:
    samples: list[DemoSample] = []
    for meta_path in sorted(demo_root.rglob("metadata.json"), key=_natural_path_key):
        sample_dir = meta_path.parent
        rel_dir = sample_dir.relative_to(demo_root)
        parts = rel_dir.parts
        if not parts:
            continue
        if any(part in EXCLUDED_DIR_NAMES or _is_rollout_sample_dir_name(part) for part in parts):
            continue
        if not _is_sample_dir(sample_dir):
            continue
        split = SPLIT_DIR_ALIASES.get(parts[0], "other")
        if include != "all" and split != include:
            continue
        samples.append(DemoSample(sample_dir=sample_dir, rel_dir=rel_dir, split=split))
    return samples


def _object_aliases(obj: object) -> set[str]:
    aliases: set[str] = set()
    if not isinstance(obj, dict):
        return aliases
    for key in ("name", "mesh_name", "mesh_source", "mesh_used", "mesh_path"):
        value = obj.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        raw = value.strip().lower()
        path = Path(value)
        if key in {"name", "mesh_name"}:
            aliases.add(raw)
        aliases.add(path.name.lower())
        aliases.add(path.stem.lower())
    return {a for a in aliases if a}


def _pattern_matches(pattern: str, obj: object) -> bool:
    pattern_l = pattern.strip().lower()
    if not pattern_l:
        return False
    if pattern_l == "all":
        return True
    return any(pattern_l in alias for alias in _object_aliases(obj))


def _material_for_kind(kind: str) -> dict[str, object]:
    if kind == "elastic":
        return dict(ELASTIC_MATERIAL)
    if kind == "rigid":
        return dict(RIGID_MATERIAL)
    raise ValueError(f"Unknown material kind: {kind}")


def _indistribution_material_kind(rel_dir: Path) -> str:
    rel_l = str(rel_dir).lower()
    if "rigid" in rel_l and not ("elastic" in rel_l or "soft" in rel_l):
        return "rigid"
    if "elastic" in rel_l or "soft" in rel_l:
        return "elastic"
    raise ValueError(
        f"Cannot infer in-distribution material from sample name {rel_dir}. "
        "Include 'rigid', 'elastic', or 'soft' in the sample directory name."
    )


def _condition_metadata(
    sample: DemoSample,
    *,
    elastic_patterns: Sequence[str],
    rigid_patterns: Sequence[str],
    match_counts: dict[tuple[str, str], int],
) -> dict:
    with (sample.sample_dir / "metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata.json must contain an object: {sample.sample_dir / 'metadata.json'}")

    objects = metadata.get("objects", [])
    if not isinstance(objects, list) or not objects:
        raise ValueError(f"metadata.json has no objects list: {sample.sample_dir / 'metadata.json'}")

    conditioned_objects = []
    for obj in objects:
        obj_out = dict(obj) if isinstance(obj, dict) else {"value": obj}
        if sample.split == "indistribution":
            material_kind = _indistribution_material_kind(sample.rel_dir)
        else:
            elastic_hit = [p for p in elastic_patterns if _pattern_matches(p, obj)]
            rigid_hit = [p for p in rigid_patterns if _pattern_matches(p, obj)]
            for pattern in elastic_hit:
                match_counts[("elastic", pattern)] += 1
            for pattern in rigid_hit:
                match_counts[("rigid", pattern)] += 1
            if elastic_hit and rigid_hit:
                aliases = ", ".join(sorted(_object_aliases(obj)))
                raise ValueError(
                    f"Conflicting material overrides for object in {sample.rel_dir}: "
                    f"elastic={elastic_hit}, rigid={rigid_hit}, aliases=[{aliases}]"
                )
            material_kind = "rigid" if rigid_hit else "elastic"
        obj_out["material"] = _material_for_kind(material_kind)
        conditioned_objects.append(obj_out)

    metadata["objects"] = conditioned_objects
    metadata["material_conditioning_export"] = {
        "split": sample.split,
        "source_sample_rel_dir": sample.rel_dir.as_posix(),
        "default_ood_material": "elastic",
        "elastic_patterns": list(elastic_patterns),
        "rigid_patterns": list(rigid_patterns),
    }
    return metadata


def write_conditioned_metadata(
    samples: Sequence[DemoSample],
    *,
    path: Path,
    elastic_patterns: Sequence[str],
    rigid_patterns: Sequence[str],
    allow_unmatched: bool,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    match_counts: dict[tuple[str, str], int] = {}
    for pattern in elastic_patterns:
        match_counts[("elastic", pattern)] = 0
    for pattern in rigid_patterns:
        match_counts[("rigid", pattern)] = 0

    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            metadata = _condition_metadata(
                sample,
                elastic_patterns=elastic_patterns,
                rigid_patterns=rigid_patterns,
                match_counts=match_counts,
            )
            payload = {
                "sample_rel_dir": sample.rel_dir.as_posix(),
                "metadata": metadata,
            }
            json.dump(payload, f)
            f.write("\n")

    unmatched = [f"--{kind} {pattern}" for (kind, pattern), count in match_counts.items() if count == 0]
    if unmatched:
        message = "Some OOD material override patterns did not match any object: " + ", ".join(unmatched)
        if not allow_unmatched:
            raise ValueError(message + ". Use --allow-unmatched-material-patterns to permit this.")
        print(f"[warn] {message}", file=sys.stderr, flush=True)
    return path


def prepare_work_root(work_root: Path) -> None:
    work_root.mkdir(parents=True, exist_ok=True)
    legacy_root = work_root / LEGACY_CONDITIONED_SAMPLE_ROOT
    if legacy_root.exists():
        if legacy_root.is_dir():
            shutil.rmtree(legacy_root)
        else:
            legacy_root.unlink()


def write_cond_sample_list(samples: Sequence[DemoSample], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(str(sample.sample_dir.resolve()))
            f.write("\n")
    return path


def _append_optional(cmd: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def build_engine_command(args: argparse.Namespace, samples: Sequence[DemoSample], forward_args: Sequence[str]) -> list[str]:
    code_root = _code_root()
    engine = code_root / "src" / "physformer" / "scripts" / "infer_multiobj_altobj.py"
    work_root = code_root / ".inference_work"
    cond_list = work_root / "cond_sample_dirs.txt"
    conditioned_metadata = work_root / "conditioned_metadata.jsonl"

    prepare_work_root(work_root)
    write_conditioned_metadata(
        samples,
        path=conditioned_metadata,
        elastic_patterns=args.elastic_patterns,
        rigid_patterns=args.rigid_patterns,
        allow_unmatched=bool(args.allow_unmatched_material_patterns),
    )
    write_cond_sample_list(samples, cond_list)

    cmd = [
        str(args.python),
        str(engine),
        "--ckpt",
        str(args.checkpoint),
        "--out_dir",
        str(args.demo_root),
        "--out_layout",
        "cond_relpath",
        "--cond_data_root",
        str(args.demo_root),
        "--cond_sample_dirs",
        str(cond_list),
        "--conditioned_metadata_jsonl",
        str(conditioned_metadata),
        "--num_samples",
        str(len(samples)),
        "--num_generations_per_sample",
        str(args.generations),
        "--device",
        str(args.device),
        "--amp",
        str(args.amp),
        "--mesh_vertex_count_json",
        str(args.mesh_vertex_count_json),
        "--max_num_objects",
        str(args.max_num_objects),
        "--cond_first_frame_velocity",
        "--auto_infer_num_vertices",
        "--env_and_mat",
        "--use_ema",
    ]
    cmd.append("--overwrite" if args.overwrite else "--no-overwrite")
    _append_optional(cmd, "--num_sampling_steps", args.num_sampling_steps)
    if args.save_gif:
        cmd.append("--save_gif")
    if args.save_mp4:
        cmd.append("--save_mp4")
    if args.save_gt_gif:
        cmd.append("--save_gt_gif")
    if args.save_gt_mp4:
        cmd.append("--save_gt_mp4")
    if args.save_compare_gif:
        cmd.append("--save_compare_gif")
    if args.save_compare_mp4:
        cmd.append("--save_compare_mp4")
    if args.save_scene_metadata:
        cmd.append("--save_scene_metadata")
    if args.verbose:
        cmd.append("--verbose")
    cmd.extend(forward_args)
    return cmd


def _run_with_log(cmd: Sequence[str], env: dict[str, str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[info] writing inference log to {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = proc.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, list(cmd))


def _format_command(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run PhysFormer inference on every official-demo sample.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--demo-root", type=Path, default=_default_demo_root())
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument(
        "--mesh-vertex-count-json",
        type=Path,
        default=default_vertex_count_json(),
        help="Mesh-name to vertex-count config used to resolve object topology.",
    )
    p.add_argument("--generations", type=int, default=3, help="Number of rollout samples to produce per input sample.")
    p.add_argument("--include", choices=["all", "ood", "indistribution"], default="all")
    p.add_argument("--max-samples", type=int, default=0, help="Limit the number of discovered samples; 0 means all.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", choices=["none", "bf16", "fp16"], default="bf16")
    p.add_argument("--python", default=sys.executable, help="Python executable from the inference environment.")
    p.add_argument("--max-num-objects", type=int, default=10)
    p.add_argument("--num-sampling-steps", type=int, default=None)
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--save-gif", action="store_true")
    p.add_argument("--save-mp4", action="store_true")
    p.add_argument("--save-gt-gif", action="store_true")
    p.add_argument("--save-gt-mp4", action="store_true")
    p.add_argument("--save-compare-gif", action="store_true")
    p.add_argument("--save-compare-mp4", action="store_true")
    p.add_argument(
        "--save-scene-metadata",
        action="store_true",
        help="Write debug/provenance scene_multiobj.json files into output sample directories.",
    )
    p.add_argument(
        "--attention-debug",
        action="store_true",
        help="Print the attention backend used by the first attention call: external flash-attn or PyTorch SDPA fallback.",
    )
    p.add_argument("--elastic", dest="elastic", action="append", nargs="+", default=[], metavar="PATTERN")
    p.add_argument("--rigid", dest="rigid", action="append", nargs="+", default=[], metavar="PATTERN")
    p.add_argument(
        "--allow-unmatched-material-patterns",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warn instead of failing when an OOD material pattern does not match any object.",
    )
    p.add_argument(
        "--strict-material-patterns",
        action="store_false",
        dest="allow_unmatched_material_patterns",
        help="Fail if any OOD material pattern does not match at least one object.",
    )
    p.add_argument("--dry-run", action="store_true", help="Prepare conditioned metadata and print the command.")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argparser()
    args, forward_args = parser.parse_known_args(argv)
    args.demo_root = args.demo_root.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.mesh_vertex_count_json = args.mesh_vertex_count_json.expanduser().resolve()
    args.elastic_patterns = _flatten(args.elastic)
    args.rigid_patterns = _flatten(args.rigid)

    if args.generations <= 0:
        parser.error("--generations must be positive")
    if args.max_samples < 0:
        parser.error("--max-samples must be non-negative")
    if not args.demo_root.is_dir():
        parser.error(f"--demo-root is not a directory: {args.demo_root}")
    if not args.checkpoint.is_file():
        if args.checkpoint.resolve() == DEFAULT_CHECKPOINT.resolve():
            parser.error(
                "Checkpoint is missing. Download it first and place it at "
                f"{DEFAULT_CHECKPOINT}. For example:\n"
                "  huggingface-cli download yslan/physformer checkpoint-best.pt --local-dir checkpoints\n"
                "Or pass a checkpoint explicitly with --checkpoint /path/to/checkpoint.pt."
            )
        parser.error(f"--checkpoint is not a file: {args.checkpoint}")
    if not args.mesh_vertex_count_json.is_file():
        parser.error(f"--mesh-vertex-count-json is not a file: {args.mesh_vertex_count_json}")

    samples = discover_samples(args.demo_root, args.include)
    if args.max_samples:
        samples = samples[: int(args.max_samples)]
    if not samples:
        parser.error(f"No samples found under {args.demo_root} for include={args.include}")

    cmd = build_engine_command(args, samples, forward_args)
    print(f"[info] selected {len(samples)} samples", flush=True)
    if args.verbose or args.dry_run:
        for sample in samples:
            print(f"[sample] {sample.rel_dir.as_posix()}", flush=True)
        print("[command] " + _format_command(cmd), flush=True)
    if args.dry_run:
        return 0

    env = os.environ.copy()
    src_path = str(_code_root() / "src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("MPLCONFIGDIR", str(_code_root() / ".inference_work" / "matplotlib"))
    if args.attention_debug:
        env["JMT4D_SDPA_DEBUG"] = "1"
    _run_with_log(cmd, env=env, cwd=_code_root(), log_path=_code_root() / ".inference_work" / "inference.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
