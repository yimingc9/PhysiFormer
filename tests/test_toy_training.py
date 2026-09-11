"""CPU checks: PYTHONPATH=src python -m unittest discover -s tests -v."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.prepare_toy_split import build_splits
from physiformer.scripts import train_npz_elastic as trainer
from physiformer.scripts.train_npz_elastic import evaluate


class MeanLoss(torch.nn.Module):
    def forward(self, x, labels, **kwargs):
        return x.mean()


class QuadraticLoss(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x, labels, **kwargs):
        return (self.weight - 1).square()


def sample(value):
    return {key: torch.tensor([float(value)])
            for key in ("x", "labels", "mask", "cond", "object_ids")}


class ToyTrainingTests(unittest.TestCase):
    def test_split_holds_out_zero_in_every_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for category, counts in (("elastic", range(1, 6)), ("rigid", range(6, 11))):
                for count in counts:
                    for i in reversed(range(10)):
                        p = root / category / f"{count}_obj" / f"sample_{i}" / "sample.npz"
                        p.parent.mkdir(parents=True)
                        p.touch()
            result = build_splits(root)
            split = result[root / "split.json"]
            self.assertEqual(split["sizes"], {"train": 90, "val": 10, "test": 0})
            self.assertEqual(len(set(split["train"] + split["val"])), 100)
            self.assertTrue(all("/sample_0/" in s for s in split["val"]))
            self.assertFalse(any("/sample_0/" in s for s in split["train"]))
            self.assertEqual(split["eval"], split["val"])
            for category in ("elastic", "rigid"):
                self.assertEqual(result[root / category / "split.json"]["sizes"],
                                 {"train": 45, "val": 5, "test": 0})
            # A changed group size must not silently change the requested ratio.
            (root / "rigid/10_obj/sample_9/sample.npz").unlink()
            with self.assertRaises(ValueError):
                build_splits(root)

    def test_validation_weights_partial_batch_by_sample_count(self):
        model = MeanLoss()
        for batch_size in (1, 3, 4, 10):
            loader = DataLoader([sample(i) for i in range(10)], batch_size=batch_size)
            self.assertAlmostEqual(evaluate(model, loader, torch.device("cpu"), "none"), 4.5)
            self.assertTrue(model.training)
        model.eval()
        evaluate(model, loader, torch.device("cpu"), "none")
        self.assertFalse(model.training)

    def test_partial_accumulation_performs_full_sgd_update(self):
        # Five minibatches accumulated in groups of three then two should make
        # two full SGD updates: w=0 -> 0.2 -> 0.36 for loss (w-1)^2 and lr=0.1.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(11):
                np.savez(root / f"{i}.npz", vertices=np.full((1, 1, 3), float(i)),
                         mask=np.ones((1, 1)), object_ids=np.zeros(1, dtype=np.int64),
                         first_frame_velocity=np.zeros((1, 3)), num_objects=np.array(1))
            split = root / "split.json"
            split.write_text(json.dumps({"train": [f"{i}.npz" for i in range(10)],
                                         "val": ["10.npz"]}))
            argv = ["train", "--precomp_root", str(root), "--split_file", str(split),
                    "--output_dir", str(root / "run"), "--val_split_name", "val",
                    "--num_vertices", "1", "--num_frames", "1", "--batch_size", "2",
                    "--grad_accum", "3", "--epochs", "1", "--warmup_epochs", "0",
                    "--lr_schedule", "constant", "--lr", "0.1", "--grad_clip", "0",
                    "--num_workers", "0", "--train_virtual_length", "0", "--amp", "none",
                    "--save_epoch_freq", "0"]
            model = QuadraticLoss()
            with patch("sys.argv", argv), patch.object(trainer, "PhysiFormerDenoiser", return_value=model), \
                 patch.object(torch.cuda, "is_available", return_value=False), \
                 patch.object(torch.optim, "AdamW", side_effect=lambda params, **kw: torch.optim.SGD(params, **kw)):
                trainer.main()
            self.assertAlmostEqual(model.weight.item(), 0.36, places=6)
            self.assertTrue((root / "run/checkpoint-last.pt").is_file())


if __name__ == "__main__":
    unittest.main()
