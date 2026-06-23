from __future__ import annotations

import os
from pathlib import Path


VERTEX_COUNT_CONFIG = "vertex_counts_multiobj_all.json"
DEFAULT_CKPT_REPO_ID = "yslan/physiformer"
DEFAULT_CKPT_FILENAME = "checkpoint-best.pt"


def code_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_demo_root() -> Path:
    return code_root()


def default_checkpoint_path() -> Path:
    return code_root() / "checkpoints" / DEFAULT_CKPT_FILENAME


def default_vertex_count_json() -> Path:
    candidates = [
        Path(__file__).resolve().parent / "configs" / VERTEX_COUNT_CONFIG,
        code_root() / "configs" / VERTEX_COUNT_CONFIG,
        code_root() / "src" / "mesh_primitives" / VERTEX_COUNT_CONFIG,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def checkpoint_repo_id() -> str:
    return os.environ.get("PHYSIFORMER_CKPT_REPO_ID", DEFAULT_CKPT_REPO_ID)


def checkpoint_filename() -> str:
    return os.environ.get("PHYSIFORMER_CKPT_FILENAME", DEFAULT_CKPT_FILENAME)


def ensure_default_checkpoint() -> tuple[Path, str]:
    ckpt_path = default_checkpoint_path()
    if ckpt_path.is_file():
        return ckpt_path, f"Checkpoint found: {ckpt_path}"

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Checkpoint is missing and huggingface_hub is not installed. "
            "Install requirements.txt or place checkpoint-best.pt under checkpoints/."
        ) from exc

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    repo_id = checkpoint_repo_id()
    filename = checkpoint_filename()
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(ckpt_path.parent),
        token=os.environ.get("HF_TOKEN") or None,
    )
    downloaded_path = Path(downloaded)
    if downloaded_path.resolve() != ckpt_path.resolve():
        downloaded_path.replace(ckpt_path)
    return ckpt_path, f"Downloaded checkpoint from {repo_id}/{filename}"
