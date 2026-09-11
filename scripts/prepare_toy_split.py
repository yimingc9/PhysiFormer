#!/usr/bin/env python3
"""Create the toy 90/10 split without moving or modifying sample files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_splits(root: Path) -> dict[Path, dict]:
    combined = {"train": [], "val": []}
    groups = {}
    outputs = {}
    for category in ("elastic", "rigid"):
        subset = {"train": [], "val": []}
        directories = sorted(
            (root / category).glob("*_obj"), key=lambda p: int(p.name.removesuffix("_obj"))
        )
        if not directories:
            raise ValueError(f"No object-count directories in {root / category}")
        for directory in directories:
            samples = sorted(directory.glob("sample_*/sample.npz"),
                             key=lambda p: int(p.parent.name.removeprefix("sample_")))
            # One held-out sample is exactly 10% only for groups of ten.
            if [p.parent.name for p in samples] != [f"sample_{i}" for i in range(10)]:
                raise ValueError(f"Expected sample_0 through sample_9 in {directory}")
            for i, path in enumerate(samples):
                name = "val" if i == 0 else "train"
                relative = path.relative_to(root / category).as_posix()
                subset[name].append(relative)
                combined[name].append(f"{category}::{relative}")
            groups[f"{category}/{directory.name}"] = {"train": 9, "val": 1}
        outputs[root / category / "split.json"] = {
            "index_kind": "sample_selector",
            "selector_format": f"NPZ path relative to data_toy/{category}",
            "selection": "sample_0 per object count is validation; sample_1 through sample_9 are training.",
            "sizes": {"train": len(subset["train"]), "val": len(subset["val"]), "test": 0},
            **subset, "eval": subset["val"], "test": [],
        }
    outputs[root / "split.json"] = {
        "index_kind": "sample_selector",
        "selector_format": "category::NPZ path relative to data_toy/<category>",
        "selection": "sample_0 per category and object count is validation; sample_1 through sample_9 are training.",
        "category_roots": {"elastic": "elastic", "rigid": "rigid"},
        "material_alias_values": {"elastic": [1.0], "rigid": [0.0]},
        "sizes": {"train": len(combined["train"]), "val": len(combined["val"]), "test": 0},
        "groups": groups,
        **combined, "eval": combined["val"], "test": [],
    }
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data_toy")
    parser.add_argument("--check", action="store_true", help="Verify existing manifests without writing.")
    args = parser.parse_args()
    for path, payload in build_splits(args.data_root.resolve()).items():
        if args.check:
            if not path.is_file() or json.loads(path.read_text()) != payload:
                raise SystemExit(f"Split is missing or stale: {path}; run this script without --check.")
        else:
            path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"{'Verified' if args.check else 'Wrote'} {path}: {payload['sizes']}")


if __name__ == "__main__":
    main()
