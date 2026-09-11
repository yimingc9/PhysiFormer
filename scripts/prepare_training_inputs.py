#!/usr/bin/env python3
"""Validate NPZ splits and optionally write portable trainer input mappings."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def prepare(data_root: Path, split_file: Path, train_split: str = 'train',
            val_split: str = 'val', material_dim: int = 0) -> dict:
    data_root = data_root.resolve()
    split_bytes = split_file.read_bytes()
    split = json.loads(split_bytes)
    roots = {str(alias): str((data_root / path).resolve())
             for alias, path in split.get('category_roots', {}).items()}
    materials = split.get('material_alias_values', {})
    seen: set[Path] = set()
    counts = {}
    if train_split == val_split:
        raise ValueError('Training and validation must use different split keys')
    for name in (train_split, val_split):
        entries = split.get(name)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f'Split {name!r} must be a nonempty list in {split_file}')
        for entry in entries:
            if not isinstance(entry, str):
                raise ValueError(f'Expected a path or sample selector, got {entry!r}')
            selector = entry.replace('\\', '/')
            alias, selector = selector.split('::', 1) if '::' in selector else ('default', selector)
            if alias != 'default' and alias not in roots:
                raise ValueError(f'Missing category_roots entry for {alias!r}')
            root = Path(roots.get(alias, data_root))
            if Path(selector).is_absolute():
                raise ValueError('Use selectors relative to DATA_ROOT or the category root')
            if ':' in selector and '/' not in selector:
                group, index = selector.split(':', 1)
                selector = f'{group}/sample_{int(index):06d}.npz'
            elif not selector.endswith('.npz'):
                selector += '.npz'
            path = (root / selector).resolve()
            if not path.is_file():
                raise ValueError(f'Missing NPZ: {path}')
            if path in seen:
                raise ValueError(f'Duplicate NPZ or train/validation overlap: {path}')
            seen.add(path)
            if material_dim > 0:
                value = materials.get(alias, materials.get('default'))
                if not isinstance(value, list) or len(value) != material_dim or not all(
                    isinstance(x, (int, float)) and math.isfinite(x) for x in value
                ):
                    raise ValueError(f'Provide material_alias_values[{alias!r}] with {material_dim} '
                                     'features, or set COND_OBJECT_MATERIAL=0')
        counts[name] = len(entries)
    return {'precomp_roots.json': roots, 'material_alias_values.json': materials,
            'split.json': split,
            'data_manifest.json': {'data_root': str(data_root), 'split_file': str(split_file.resolve()),
                                   'split_sha256': hashlib.sha256(split_bytes).hexdigest(), 'sizes': counts}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--split-file', type=Path, required=True)
    parser.add_argument('--train-split', default='train')
    parser.add_argument('--val-split', default='val')
    parser.add_argument('--material-dim', type=int, default=0)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    try:
        outputs = prepare(args.data_root, args.split_file, args.train_split, args.val_split, args.material_dim)
    except (ValueError, OSError) as error:
        raise SystemExit(f'[error] {error}') from error
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, payload in outputs.items():
            (args.output_dir / name).write_text(json.dumps(payload, indent=2) + '\n')
    print(f"[data] verified {outputs['data_manifest.json']['sizes']} under {args.data_root}")


if __name__ == '__main__':
    main()
