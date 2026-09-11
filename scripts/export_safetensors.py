#!/usr/bin/env python3
"""Export a checkpoint as model.safetensors + config.json, verifying every tensor."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from safetensors.torch import save_file
import torch

from physiformer.checkpoints import config_digest, load_checkpoint, release_config, select_weights


def tensor_digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def export_checkpoint(source: Path, output_dir: Path, *, weight_source: str = "ema") -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}; choose a new release directory")
    checkpoint = load_checkpoint(source)
    state, selected = select_weights(checkpoint, use_ema=weight_source == "ema")
    if selected != weight_source:
        raise ValueError(f"Requested {weight_source} weights, but checkpoint contains {selected} weights")
    if not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Expected a nonempty tensor state_dict")
    config = release_config(checkpoint, state, selected)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".export-", dir=output_dir.parent))
    try:
        tensors = {}
        seen_storage = set()
        for key, value in state.items():
            value = value.detach().cpu().contiguous()
            storage = value.untyped_storage().data_ptr()
            tensors[key] = value.clone() if storage in seen_storage else value
            seen_storage.add(storage)
        (temporary / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        save_file(tensors, str(temporary / "model.safetensors"),
                  metadata={"format": "physiformer", "config_sha256": config_digest(config)})
        restored = load_checkpoint(temporary / "model.safetensors")
        actual, actual_source = select_weights(restored)
        assert actual_source == selected and actual.keys() == state.keys()
        total_bytes = 0
        for key, expected in state.items():
            found = actual[key]
            assert found.dtype == expected.dtype and found.shape == expected.shape, key
            assert tensor_digest(found) == tensor_digest(expected), key
            total_bytes += expected.numel() * expected.element_size()
        assert restored["args"] == config["model_args"]
        temporary.rename(output_dir)
        return {"weight_source": selected, "verified_tensors": len(state), "tensor_bytes": total_bytes,
                "config_sha256": config_digest(config), "tensor_equality": "bitwise identical"}
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    args = parser.parse_args()
    print(json.dumps(export_checkpoint(args.checkpoint, args.output_dir, weight_source=args.weights), indent=2))


if __name__ == "__main__":
    main()
