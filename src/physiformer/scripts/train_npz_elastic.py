from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from physiformer.diffusion.denoiser import DiffusionConfig
from physiformer.diffusion.physiformer_denoiser import PhysiFormerDenoiser


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_dist() else 0


def world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main() -> bool:
    return rank() == 0


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split_entries(split_file: str | Path, split_name: str) -> list[str]:
    with open(split_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if split_name not in payload and split_name == "eval" and "val" in payload:
        split_name = "val"
    entries = payload.get(split_name)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Split {split_name!r} missing or empty in {split_file}")
    return [str(x) for x in entries]


def load_precomp_roots(precomp_root: str | Path, precomp_roots_json: str | Path | None = None) -> dict[str, Path]:
    roots = {"": Path(precomp_root), "default": Path(precomp_root)}
    if precomp_roots_json:
        with open(precomp_roots_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"precomp_roots_json must contain an object mapping aliases to paths: {precomp_roots_json}")
        for key, value in payload.items():
            roots[str(key)] = Path(str(value))
    return roots


def precomp_roots_signature(roots: dict[str, Path]) -> dict[str, str]:
    return {str(k): str(Path(v).resolve()) for k, v in sorted(roots.items()) if str(k)}


def selector_to_npz(precomp_root: str | Path, selector: str, precomp_roots: dict[str, Path] | None = None) -> Path:
    s = str(selector).replace("\\", "/").strip("/")
    root = Path(precomp_root)
    if "::" in s:
        alias, s = s.split("::", 1)
        if precomp_roots is None or alias not in precomp_roots:
            known = sorted(precomp_roots or {})
            raise KeyError(f"Unknown precompute root alias {alias!r} in selector {selector!r}. Known aliases: {known}")
        root = Path(precomp_roots[alias])
    if ":" in s and "/" not in s:
        obj, idx_s = s.split(":", 1)
        return root / obj / f"sample_{int(idx_s):06d}.npz"
    if s.endswith(".npz"):
        return root / s
    return root / f"{s}.npz"


def selector_alias(selector: str) -> str:
    s = str(selector).replace("\\", "/").strip("/")
    if "::" in s:
        return s.split("::", 1)[0]
    return "default"


def load_material_alias_values(path: str | Path) -> dict[str, np.ndarray]:
    if not str(path).strip():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"material_alias_values_json must contain an object mapping aliases to values: {path}")
    out: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if isinstance(value, (int, float)):
            arr = np.asarray([float(value)], dtype=np.float32)
        elif isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
            arr = np.asarray([float(x) for x in value], dtype=np.float32)
        else:
            raise ValueError(f"Material value for alias {key!r} must be a number or list of numbers, got {value!r}")
        out[str(key)] = arr
    return out


def match_material_dim(value: np.ndarray, dim: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if int(dim) <= 0:
        return arr[:0]
    if int(arr.shape[0]) == int(dim):
        return arr
    if int(arr.shape[0]) > int(dim):
        return arr[: int(dim)].astype(np.float32, copy=False)
    pad = np.zeros((int(dim) - int(arr.shape[0]),), dtype=np.float32)
    return np.concatenate([arr, pad], axis=0).astype(np.float32, copy=False)


def material_value_for_selector(
    selector: str,
    material_alias_values: dict[str, np.ndarray],
    default_value: np.ndarray,
    dim: int,
) -> np.ndarray:
    alias = selector_alias(selector)
    value = material_alias_values.get(alias)
    if value is None:
        value = material_alias_values.get("default", default_value)
    return match_material_dim(value, int(dim))


def infer_num_vertices(precomp_root: str | Path, entries: list[str], precomp_roots: dict[str, Path] | None = None) -> int:
    max_v = 0
    for selector in entries:
        path = selector_to_npz(precomp_root, selector, precomp_roots)
        with np.load(path, allow_pickle=False) as z:
            max_v = max(max_v, int(z["vertices"].shape[1]))
    if max_v <= 0:
        raise ValueError("Could not infer num_vertices")
    return max_v


def compute_or_load_stats(
    *,
    precomp_root: str | Path,
    precomp_roots: dict[str, Path] | None,
    split_file: str | Path,
    train_entries: list[str],
    stats_json: str | Path,
    recompute: bool,
) -> tuple[np.ndarray, np.ndarray]:
    stats_path = Path(stats_json)
    expected = {
        "split_file": str(Path(split_file).resolve()),
        "precomp_root": str(Path(precomp_root).resolve()),
        "precomp_roots": precomp_roots_signature(precomp_roots or {}),
        "split": "train",
        "num_samples": int(len(train_entries)),
    }
    if stats_path.is_file() and not bool(recompute):
        with stats_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if all(payload.get(k) == v for k, v in expected.items()):
            mean = np.asarray(payload["position_mean"], dtype=np.float32)
            std = np.asarray(payload["position_std"], dtype=np.float32)
            return mean, std

    sum_x = np.zeros(3, dtype=np.float64)
    sum_x2 = np.zeros(3, dtype=np.float64)
    count = 0
    for selector in train_entries:
        path = selector_to_npz(precomp_root, selector, precomp_roots)
        with np.load(path, allow_pickle=False) as z:
            vertices = np.asarray(z["vertices"], dtype=np.float64)
            mask = np.asarray(z["mask"], dtype=bool)
        vals = vertices[mask]
        if vals.size == 0:
            continue
        sum_x += vals.sum(axis=0)
        sum_x2 += (vals * vals).sum(axis=0)
        count += int(vals.shape[0])
    if count <= 0:
        raise ValueError("No valid vertices found while computing train stats")

    mean = sum_x / float(count)
    var = np.maximum(sum_x2 / float(count) - mean * mean, 1e-12)
    std = np.sqrt(var)

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(expected)
    payload.update(
        position_mean=[float(x) for x in mean.tolist()],
        position_var=[float(x) for x in var.tolist()],
        position_std=[float(x) for x in std.tolist()],
        valid_vertex_count=int(count),
    )
    tmp = stats_path.with_suffix(stats_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    os.replace(tmp, stats_path)
    return mean.astype(np.float32), std.astype(np.float32)


class PrecomputedElasticDataset(Dataset):
    def __init__(
        self,
        *,
        precomp_root: str | Path,
        precomp_roots: dict[str, Path] | None,
        entries: list[str],
        norm_mean: np.ndarray,
        norm_std: np.ndarray,
        num_vertices: int,
        max_num_objects: int,
        virtual_length: int = 0,
        cond_first_frame_velocity: bool = True,
        cond_object_material: bool = False,
        object_material_dim: int = 0,
        material_alias_values: dict[str, np.ndarray] | None = None,
        material_default_value: np.ndarray | None = None,
        sort_paths: bool = False,
    ) -> None:
        self.precomp_root = Path(precomp_root)
        self.precomp_roots = precomp_roots or load_precomp_roots(precomp_root)
        self.entries = list(entries)
        if sort_paths:
            self.entries.sort(key=lambda entry: selector_to_npz(self.precomp_root, entry, self.precomp_roots).as_posix())
        self.norm_mean = np.asarray(norm_mean, dtype=np.float32).reshape(1, 1, 3)
        self.norm_std = np.asarray(norm_std, dtype=np.float32).reshape(1, 1, 3)
        self.num_vertices = int(num_vertices)
        self.max_num_objects = int(max_num_objects)
        self.pad_object_id = int(max_num_objects)
        self.virtual_length = int(virtual_length)
        self.cond_first_frame_velocity = bool(cond_first_frame_velocity)
        self.cond_object_material = bool(cond_object_material)
        self.object_material_dim = int(object_material_dim) if bool(cond_object_material) else 0
        self.material_alias_values = material_alias_values or {}
        if material_default_value is None:
            material_default_value = np.zeros((max(1, self.object_material_dim),), dtype=np.float32)
        self.material_default_value = np.asarray(material_default_value, dtype=np.float32).reshape(-1)

    def __len__(self) -> int:
        return self.virtual_length if self.virtual_length > 0 else len(self.entries)

    def _pad(self, arr: np.ndarray, axis: int, value: float | int) -> np.ndarray:
        cur = int(arr.shape[axis])
        if cur == self.num_vertices:
            return arr
        if cur > self.num_vertices:
            raise ValueError(f"Sample has {cur} vertices, exceeds num_vertices={self.num_vertices}")
        pad_width = [(0, 0)] * arr.ndim
        pad_width[axis] = (0, self.num_vertices - cur)
        return np.pad(arr, pad_width, mode="constant", constant_values=value)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        selector = self.entries[int(idx) % len(self.entries)]
        path = selector_to_npz(self.precomp_root, selector, self.precomp_roots)
        with np.load(path, allow_pickle=False) as z:
            vertices = np.asarray(z["vertices"], dtype=np.float32)
            mask = np.asarray(z["mask"], dtype=np.float32)
            object_ids = np.asarray(z["object_ids"], dtype=np.int64)
            first_vel = np.asarray(z["first_frame_velocity"], dtype=np.float32)
            num_objects = int(np.asarray(z["num_objects"]).item()) if "num_objects" in z.files else self.max_num_objects

        vertices = self._pad(vertices, 1, 0.0)
        mask = self._pad(mask, 1, 0.0)
        object_ids = self._pad(object_ids, 0, self.pad_object_id)
        first_vel = self._pad(first_vel, 0, 0.0)

        object_ids = np.where((object_ids >= 0) & (object_ids < self.max_num_objects), object_ids, self.pad_object_id)
        object_ids = np.where(np.any(mask > 0, axis=0), object_ids, self.pad_object_id)
        vertices_norm = ((vertices - self.norm_mean) / self.norm_std) * mask[..., None]
        first_vel_norm = first_vel / self.norm_std.reshape(1, 3)
        cond = vertices_norm[0]
        if self.cond_first_frame_velocity:
            cond = np.concatenate([cond, first_vel_norm], axis=-1)

        out = {
            "x": torch.from_numpy(vertices_norm.astype(np.float32, copy=False)),
            "mask": torch.from_numpy(mask.astype(np.float32, copy=False)),
            "object_ids": torch.from_numpy(object_ids.astype(np.int64, copy=False)),
            "cond": torch.from_numpy(cond.astype(np.float32, copy=False)),
            "labels": torch.zeros((), dtype=torch.long),
        }
        if self.cond_object_material and self.object_material_dim > 0:
            material_row = material_value_for_selector(
                selector,
                self.material_alias_values,
                self.material_default_value,
                self.object_material_dim,
            )
            object_materials = np.zeros((self.max_num_objects + 1, self.object_material_dim), dtype=np.float32)
            valid_objects = max(0, min(int(num_objects), self.max_num_objects))
            if valid_objects > 0:
                object_materials[:valid_objects] = material_row.reshape(1, self.object_material_dim)
            out["object_materials"] = torch.from_numpy(object_materials.astype(np.float32, copy=False))
        return out


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float, *, on_cpu: bool = True) -> None:
        self.decay = float(decay)
        self.on_cpu = on_cpu
        self.shadow = {k: (v.detach().cpu() if on_cpu else v.detach()).clone()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            v = v.detach().cpu() if self.on_cpu else v.detach()
            if k not in self.shadow:
                self.shadow[k] = v.clone()
                continue
            self.shadow[k] = self.shadow[k].to(device=v.device, dtype=v.dtype)
            if v.is_floating_point() or v.is_complex():
                self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

    @contextmanager
    def average_parameters(self, model: torch.nn.Module):
        """Validate EMA weights, then restore training weights even on error."""
        saved = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        try:
            model.load_state_dict(self.shadow, strict=True)
            yield
        finally:
            model.load_state_dict(saved, strict=True)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.decay = float(payload.get("decay", self.decay))
        shadow = payload.get("shadow", {})
        if isinstance(shadow, dict):
            self.shadow = {k: v.detach().cpu().clone() for k, v in shadow.items()}


def lr_at_epoch(epoch: int, epochs: int, warmup_epochs: int, lr: float, min_lr: float, schedule: str) -> float:
    """Original AltObj schedule, including zero LR in epoch zero."""
    if epoch < warmup_epochs:
        return lr * epoch / max(1, warmup_epochs)
    if schedule == "constant":
        return lr
    return min_lr + (lr - min_lr) * 0.5 * (
        1.0 + math.cos(math.pi * (epoch - warmup_epochs) / max(1, epochs - warmup_epochs))
    )


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def lr_at_step(step: int, total_steps: int, warmup_steps: int, lr: float, min_lr: float, schedule: str) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(lr) * float(step + 1) / float(warmup_steps)
    if schedule == "constant":
        return float(lr)
    denom = max(1, int(total_steps) - int(warmup_steps))
    progress = min(1.0, max(0.0, (float(step) - float(warmup_steps)) / float(denom)))
    return float(min_lr) + 0.5 * (float(lr) - float(min_lr)) * (1.0 + math.cos(math.pi * progress))


def autocast_context(amp: str):
    if not torch.cuda.is_available() or amp == "none":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.cuda.amp.autocast(dtype=dtype)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, amp: str) -> float:
    was_training = model.training
    model.eval()
    # Ranks may have unequal numbers of validation batches. Avoid DDP forward
    # collectives; all ranks reduce their totals once after their local loop.
    eval_model = model.module if isinstance(model, DDP) else model
    loss_sum = 0.0
    sample_count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        model_kwargs: dict[str, torch.Tensor] = {}
        if "object_materials" in batch:
            model_kwargs["object_materials"] = batch["object_materials"]
        with autocast_context(amp):
            loss = eval_model(
                batch["x"],
                batch["labels"],
                mask=batch["mask"],
                cond_first_frame=batch["cond"],
                object_ids=batch["object_ids"],
                **model_kwargs,
            )
        batch_size = int(batch["x"].shape[0])
        loss_sum += float(loss.detach().cpu()) * batch_size
        sample_count += batch_size
    value = torch.tensor([loss_sum, sample_count], device=device, dtype=torch.float64)
    if is_dist():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    model.train(was_training)
    return float(value[0].item() / max(1.0, value[1].item()))


def save_checkpoint(
    *,
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    args: argparse.Namespace,
    ema: EMA,
    best_val_loss: float,
) -> None:
    raw_model = model.module if isinstance(model, DDP) else model
    payload = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "step": int(step),
        "args": vars(args),
        "ema": ema.state_dict(),
        "best_val_loss": float(best_val_loss),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def initialize_from_checkpoint(model: torch.nn.Module, ckpt_path: str | Path, *, use_ema: bool) -> dict[str, int]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if bool(use_ema) and isinstance(ckpt.get("ema"), dict) and isinstance(ckpt["ema"].get("shadow"), dict):
        src_state = ckpt["ema"]["shadow"]
        source = "ema"
    else:
        src_state = ckpt.get("model", ckpt)
        source = "model"
    if not isinstance(src_state, dict):
        raise ValueError(f"Could not find a model state dict in init checkpoint: {ckpt_path}")

    own_state = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    skipped_missing = 0
    skipped_shape = 0
    for key, value in src_state.items():
        if not isinstance(value, torch.Tensor):
            skipped_missing += 1
            continue
        if key not in own_state:
            skipped_missing += 1
            continue
        if tuple(own_state[key].shape) != tuple(value.shape):
            skipped_shape += 1
            continue
        compatible[key] = value

    incompat = model.load_state_dict(compatible, strict=False)
    return {
        "source": 1 if source == "ema" else 0,
        "loaded": int(len(compatible)),
        "skipped_missing": int(skipped_missing),
        "skipped_shape": int(skipped_shape),
        "missing_after_load": int(len(incompat.missing_keys)),
        "unexpected_after_load": int(len(incompat.unexpected_keys)),
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Train official PhysiFormer on precomputed elastic JmT4D NPZs")
    p.add_argument("--precomp_root", type=str, required=True)
    p.add_argument(
        "--precomp_roots_json",
        type=str,
        default="",
        help="Optional JSON alias map for mixed selectors like 'alias::1_obj/sample_000000'.",
    )
    p.add_argument("--split_file", type=str, required=True)
    p.add_argument("--split_name", type=str, default="train")
    p.add_argument("--val_split_name", type=str, default="eval")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--stats_json", type=str, default="")
    p.add_argument("--recompute_norm", action="store_true")
    p.add_argument("--norm_mean", type=float, nargs=3, default=None)
    p.add_argument("--norm_std", type=float, nargs=3, default=None)
    p.add_argument("--training_protocol", choices=["standard", "altobj"], default="standard",
                   help="altobj uses epoch LR, EMA validation, original samplers, and discarded accumulation tails.")

    p.add_argument("--model", type=str, default="PhysiFormer-B", choices=["PhysiFormer-B", "PhysiFormer"])
    p.add_argument("--num_frames", type=int, default=49)
    p.add_argument("--num_vertices", type=int, default=0)
    p.add_argument("--num_classes", type=int, default=1)
    p.add_argument("--max_num_objects", type=int, default=5)
    p.add_argument("--use_rope", action="store_true")
    p.add_argument("--no_rope", action="store_false", dest="use_rope")
    p.set_defaults(use_rope=True)
    p.add_argument("--num_register_tokens", type=int, default=16)
    p.add_argument("--max_frames", type=int, default=128)
    p.add_argument("--max_vertices", type=int, default=8192)
    p.add_argument("--cond_object_material", action="store_true")
    p.add_argument("--no_cond_object_material", action="store_false", dest="cond_object_material")
    p.set_defaults(cond_object_material=False)
    p.add_argument("--object_material_dim", type=int, default=0)
    p.add_argument(
        "--material_alias_values_json",
        type=str,
        default="",
        help="Optional JSON alias map for material features. Example: {'rigid':[0,0.01], 'soft':[1,0.15]}.",
    )
    p.add_argument("--attn_drop", type=float, default=0.0)
    p.add_argument("--proj_drop", type=float, default=0.0)
    p.add_argument("--grad_checkpoint", action="store_true")
    p.add_argument("--no_grad_checkpoint", action="store_false", dest="grad_checkpoint")
    p.set_defaults(grad_checkpoint=True)

    p.add_argument("--P_mean", type=float, default=-0.8)
    p.add_argument("--P_std", type=float, default=0.8)
    p.add_argument("--t_eps", type=float, default=5e-2)
    p.add_argument("--noise_scale", type=float, default=0.1)
    p.add_argument("--sampling_method", type=str, default="heun", choices=["euler", "heun"])
    p.add_argument("--num_sampling_steps", type=int, default=50)

    p.add_argument("--epochs", type=int, default=6000)
    p.add_argument("--train_virtual_length", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--min_lr", type=float, default=5e-6)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--lr_schedule", type=str, default="cosine", choices=["constant", "cosine"])
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16", "fp16"])
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_freq", type=int, default=5)
    p.add_argument("--eval_epoch_freq", type=int, default=1)
    p.add_argument("--save_epoch_freq", type=int, default=10)
    p.add_argument("--save_last_freq", type=int, default=1000)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--init_ckpt", type=str, default="", help="Initialize weights from a checkpoint without optimizer state.")
    p.add_argument("--init_use_ema", action="store_true")
    p.add_argument("--no_init_ema", action="store_false", dest="init_use_ema")
    p.set_defaults(init_use_ema=True)

    return p


def main() -> None:
    args = build_argparser().parse_args()
    reproduce = args.training_protocol == "altobj"
    args.val_use_ema = reproduce
    if (args.norm_mean is None) != (args.norm_std is None):
        raise ValueError("Provide both --norm_mean and --norm_std, or neither")
    if args.norm_std is not None and (not np.isfinite(args.norm_mean).all()
                                    or not np.isfinite(args.norm_std).all()
                                    or min(args.norm_std) <= 0):
        raise ValueError("Normalization must be finite with positive standard deviations")
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    seed_all(int(args.seed) + rank())

    output_dir = Path(args.output_dir)
    if is_main():
        output_dir.mkdir(parents=True, exist_ok=True)

    precomp_roots = load_precomp_roots(args.precomp_root, args.precomp_roots_json or None)
    args.object_material_dim = int(args.object_material_dim) if bool(args.cond_object_material) else 0
    material_alias_values = load_material_alias_values(args.material_alias_values_json)
    material_default_value = material_alias_values.get(
        "default",
        np.zeros((max(1, int(args.object_material_dim)),), dtype=np.float32),
    )
    train_entries = load_split_entries(args.split_file, args.split_name)
    val_entries = load_split_entries(args.split_file, args.val_split_name)
    if int(args.num_vertices) <= 0:
        args.num_vertices = infer_num_vertices(args.precomp_root, train_entries[:1] if reproduce else train_entries + val_entries, precomp_roots)

    stats_json = args.stats_json or str(output_dir / "train_position_stats.json")
    if args.norm_mean is not None:
        mean = np.asarray(args.norm_mean, dtype=np.float32)
        std = np.asarray(args.norm_std, dtype=np.float32)
        if is_main():
            Path(stats_json).parent.mkdir(parents=True, exist_ok=True)
            Path(stats_json).write_text(json.dumps({
                "source": "explicit", "position_mean": mean.tolist(), "position_std": std.tolist(),
            }, indent=2) + "\n")
    elif is_main():
        mean, std = compute_or_load_stats(
            precomp_root=args.precomp_root,
            precomp_roots=precomp_roots,
            split_file=args.split_file,
            train_entries=train_entries,
            stats_json=stats_json,
            recompute=bool(args.recompute_norm),
        )
        if is_dist():
            dist.barrier()
    else:
        dist.barrier()
        mean, std = compute_or_load_stats(
            precomp_root=args.precomp_root,
            precomp_roots=precomp_roots,
            split_file=args.split_file,
            train_entries=train_entries,
            stats_json=stats_json,
            recompute=False,
        )
    args.norm_mean = tuple(float(x) for x in mean.tolist())
    args.norm_std = tuple(float(x) for x in std.tolist())
    args.coord_scale = 1.0
    args.coord_shift = 0.0
    args.cond_first_frame = True
    args.cond_first_frame_velocity = True
    args.delta_to_first_frame = False
    args.material_alias_values = {k: [float(x) for x in v.tolist()] for k, v in sorted(material_alias_values.items())}

    train_set = PrecomputedElasticDataset(
        precomp_root=args.precomp_root,
        precomp_roots=precomp_roots,
        entries=train_entries,
        norm_mean=mean,
        norm_std=std,
        num_vertices=int(args.num_vertices),
        max_num_objects=int(args.max_num_objects),
        virtual_length=int(args.train_virtual_length),
        cond_object_material=bool(args.cond_object_material),
        object_material_dim=int(args.object_material_dim),
        material_alias_values=material_alias_values,
        material_default_value=material_default_value,
        sort_paths=reproduce,
    )
    val_set = PrecomputedElasticDataset(
        precomp_root=args.precomp_root,
        precomp_roots=precomp_roots,
        entries=val_entries,
        norm_mean=mean,
        norm_std=std,
        num_vertices=int(args.num_vertices),
        max_num_objects=int(args.max_num_objects),
        virtual_length=0,
        cond_object_material=bool(args.cond_object_material),
        object_material_dim=int(args.object_material_dim),
        material_alias_values=material_alias_values,
        material_default_value=material_default_value,
        sort_paths=reproduce,
    )

    train_sampler = DistributedSampler(train_set, shuffle=True, seed=0 if reproduce else int(args.seed), drop_last=reproduce) if is_dist() else None
    # DistributedSampler pads with repeated entries when len(val_set) is not
    # divisible by world size, biasing validation on small datasets.
    val_sampler = range(rank(), len(val_set), world_size()) if is_dist() else None
    if reproduce and is_dist():
        val_sampler = DistributedSampler(val_set, shuffle=False, drop_last=False)
    train_loader = DataLoader(
        train_set,
        batch_size=int(args.batch_size),
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(args.eval_batch_size),
        sampler=val_sampler,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    diff_cfg = DiffusionConfig(
        P_mean=float(args.P_mean),
        P_std=float(args.P_std),
        t_eps=float(args.t_eps),
        noise_scale=float(args.noise_scale),
        sampling_method=str(args.sampling_method),
        num_sampling_steps=int(args.num_sampling_steps),
    )
    model_kwargs = {
        "use_rope": bool(args.use_rope),
        "num_register_tokens": int(args.num_register_tokens),
        "max_frames": int(args.max_frames),
        "max_vertices": int(args.max_vertices),
        "attn_drop": float(args.attn_drop),
        "proj_drop": float(args.proj_drop),
        "max_num_objects": int(args.max_num_objects),
        "use_object_id_embed": False,
        "num_scene_tokens": 0,
        "scene_cond_dim": 0,
        "scene_cond_embed_out_tokens": 0,
        "object_material_dim": int(args.object_material_dim),
        "grad_checkpoint": bool(args.grad_checkpoint),
    }
    model = PhysiFormerDenoiser(
        model_name=str(args.model),
        num_frames=int(args.num_frames),
        num_vertices=int(args.num_vertices),
        num_classes=int(args.num_classes),
        model_kwargs=model_kwargs,
        diffusion=diff_cfg,
    ).to(device)

    if args.init_ckpt and not args.resume:
        init_report = initialize_from_checkpoint(model, args.init_ckpt, use_ema=bool(args.init_use_ema))
        if is_main():
            init_source = "ema" if int(init_report["source"]) == 1 else "model"
            print(
                "[init] "
                f"checkpoint={args.init_ckpt} source={init_source} loaded={init_report['loaded']} "
                f"skipped_missing={init_report['skipped_missing']} skipped_shape={init_report['skipped_shape']} "
                f"missing_after_load={init_report['missing_after_load']} "
                f"unexpected_after_load={init_report['unexpected_after_load']}",
                flush=True,
            )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    ema = EMA(model, decay=float(args.ema_decay), on_cpu=not reproduce)

    start_epoch = 0
    global_step = 0
    best_val = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        if isinstance(ckpt.get("ema"), dict):
            ema.load_state_dict(ckpt["ema"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("step", 0))
        best_val = float(ckpt.get("best_val_loss", best_val))

    if is_dist():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=reproduce)

    steps_per_epoch = (len(train_loader) // int(args.grad_accum) if reproduce
                       else math.ceil(len(train_loader) / int(args.grad_accum)))
    if steps_per_epoch < 1:
        raise ValueError("No optimizer updates per epoch; reduce grad_accum/batch_size or increase train_virtual_length")
    total_steps = max(1, int(args.epochs) * steps_per_epoch)
    warmup_steps = max(0, int(args.warmup_epochs) * steps_per_epoch)
    effective_batch = int(args.batch_size) * world_size() * int(args.grad_accum)

    if is_main():
        print(
            "[train] "
            f"model={args.model} num_vertices={args.num_vertices} train={len(train_entries)} val={len(val_entries)} "
            f"virtual_length={len(train_set)} batch_per_gpu={args.batch_size} grad_accum={args.grad_accum} "
            f"world_size={world_size()} effective_batch={effective_batch} lr={args.lr} amp={args.amp}",
            flush=True,
        )
        print(f"[train] norm_mean={args.norm_mean} norm_std={args.norm_std} stats={stats_json}", flush=True)
        print(f"[train] protocol={args.training_protocol} optimizer_steps_per_epoch={steps_per_epoch} "
              f"validation_weights={'ema' if reproduce else 'current'}", flush=True)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp == "fp16" and torch.cuda.is_available()))
    for epoch in range(start_epoch, int(args.epochs)):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accum = 0
        epoch_loss = 0.0
        epoch_count = 0
        t0 = time.time()

        for it, batch in enumerate(train_loader):
            batch = move_batch(batch, device)
            model_extra_kwargs: dict[str, torch.Tensor] = {}
            if "object_materials" in batch:
                model_extra_kwargs["object_materials"] = batch["object_materials"]
            if reproduce:
                lr_now = lr_at_epoch(epoch, int(args.epochs), int(args.warmup_epochs),
                                     float(args.lr), float(args.min_lr), str(args.lr_schedule))
            else:
                lr_now = lr_at_step(global_step, total_steps, warmup_steps, float(args.lr), float(args.min_lr), str(args.lr_schedule))
            set_lr(optimizer, lr_now)
            with autocast_context(str(args.amp)):
                loss = model(
                    batch["x"],
                    batch["labels"],
                    mask=batch["mask"],
                    cond_first_frame=batch["cond"],
                    object_ids=batch["object_ids"],
                    **model_extra_kwargs,
                )
            raw_loss = float(loss.detach().cpu())
            # AltObj intentionally discards the final incomplete group. Standard
            # mode applies it, normalized by its actual number of minibatches.
            group_start = (it // int(args.grad_accum)) * int(args.grad_accum)
            group_size = (int(args.grad_accum) if reproduce
                          else min(int(args.grad_accum), len(train_loader) - group_start))
            scaler.scale(loss / group_size).backward()
            accum += 1
            epoch_loss += raw_loss
            epoch_count += 1

            if accum >= int(args.grad_accum) or (not reproduce and it == len(train_loader) - 1):
                if float(args.grad_clip) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                raw_model = model.module if isinstance(model, DDP) else model
                ema.update(raw_model)
                global_step += 1
                accum = 0

                if is_main() and (global_step % int(args.log_freq) == 0):
                    print(
                        f"[train] epoch={epoch + 1} step={global_step} loss={raw_loss:.6g} lr={lr_now:.6g}",
                        flush=True,
                    )

        avg_train = epoch_loss / max(1, epoch_count)
        if is_main():
            print(f"[epoch] epoch={epoch + 1} train_loss={avg_train:.6g} seconds={time.time() - t0:.1f}", flush=True)

        do_eval = int(args.eval_epoch_freq) > 0 and (((epoch + 1) % int(args.eval_epoch_freq) == 0) or epoch == int(args.epochs) - 1)
        val_loss = float("nan")
        if do_eval:
            raw_model = model.module if isinstance(model, DDP) else model
            with ema.average_parameters(raw_model) if reproduce else nullcontext():
                val_loss = evaluate(model, val_loader, device, str(args.amp))
            if is_main():
                print(f"[val] epoch={epoch + 1} val_loss={val_loss:.6g}", flush=True)

        if is_main():
            if math.isfinite(val_loss) and val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    path=output_dir / "checkpoint-best.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=global_step,
                    args=args,
                    ema=ema,
                    best_val_loss=best_val,
                )
            if ((epoch + 1) % int(args.save_last_freq) == 0) or epoch == int(args.epochs) - 1:
                save_checkpoint(
                    path=output_dir / "checkpoint-last.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=global_step,
                    args=args,
                    ema=ema,
                    best_val_loss=best_val,
                )
            if int(args.save_epoch_freq) > 0 and (((epoch + 1) % int(args.save_epoch_freq) == 0) or epoch == int(args.epochs) - 1):
                save_checkpoint(
                    path=output_dir / f"checkpoint-epoch{epoch + 1:04d}.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=global_step,
                    args=args,
                    ema=ema,
                    best_val_loss=best_val,
                )

    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
