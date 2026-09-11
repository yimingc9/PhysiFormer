"""Release serialization, initialization, and resume compatibility checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts.export_safetensors import export_checkpoint, tensor_digest
from physiformer.checkpoints import load_checkpoint, load_training_checkpoint, select_weights
from physiformer.scripts.train_npz_elastic import initialize_from_checkpoint


class CheckpointReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "training.pt"
        self.args = {
            "model": "MeshVideoDiT-ST-Vert-L-MultiObj-AltObj",
            "num_frames": 49, "num_vertices": 356, "max_num_objects": 10,
            "norm_mean": (-.01, .02, -.7), "norm_std": (.4, .5, .6),
            "noise_scale": .2, "num_sampling_steps": 7,
            "data_root": "/private/training", "wandb_project": "private-project",
        }
        raw = {"weight": torch.tensor([[1., 2.]]), "bias": torch.tensor([3.])}
        ema = {k: v + 1 for k, v in raw.items()}
        self.payload = {"model": raw, "ema": {"decay": .99, "shadow": ema},
                        "args": self.args, "optimizer": {"state": {}, "param_groups": []},
                        "epoch": 4, "step": 100}
        torch.save(self.payload, self.source)

    def export(self, source="ema"):
        folder = self.root / source
        report = export_checkpoint(self.source, folder, weight_source=source)
        return folder / "model.safetensors", report

    def test_ema_and_raw_exports_preserve_outputs_and_metadata(self):
        for source in ("ema", "model"):
            path, report = self.export(source)
            restored = load_checkpoint(path)
            state, selected = select_weights(restored)
            expected, _ = select_weights(self.payload, use_ema=source == "ema")
            self.assertEqual(selected, source)
            self.assertEqual(report["verified_tensors"], 2)
            a, b = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
            a.load_state_dict(expected, strict=True)
            b.load_state_dict(state, strict=True)
            torch.testing.assert_close(a(torch.ones(1, 2)), b(torch.ones(1, 2)), rtol=0, atol=0)
            self.assertEqual(restored["args"]["model"], "PhysiFormer")
            self.assertEqual(restored["args"]["norm_mean"], list(self.args["norm_mean"]))
            self.assertEqual(restored["args"]["norm_std"], list(self.args["norm_std"]))
            self.assertEqual(restored["args"]["noise_scale"], .2)
            text = path.with_name("config.json").read_text()
            self.assertNotIn("private", text)
            self.assertNotIn("wandb", text)
            self.assertNotIn("optimizer", restored)
            # A single exported set cannot switch back to the discarded weights.
            self.assertEqual(select_weights(restored, use_ema=False)[1], source)

    def test_initialization_accepts_both_formats(self):
        path, _ = self.export()
        for checkpoint in (self.source, path):
            model = torch.nn.Linear(2, 1)
            report = initialize_from_checkpoint(model, checkpoint, use_ema=True)
            self.assertEqual(report["loaded"], 2)
            self.assertEqual(report["missing_after_load"], 0)
            self.assertEqual(report["source"], 1)
            for key, tensor in model.state_dict().items():
                self.assertEqual(tensor_digest(tensor), tensor_digest(self.payload["ema"]["shadow"][key]))

    def test_resume_requires_training_state(self):
        self.assertEqual(load_training_checkpoint(self.source)["step"], 100)
        path, _ = self.export()
        with self.assertRaisesRegex(ValueError, "--init_ckpt"):
            load_training_checkpoint(path)
        weights_only = self.root / "weights.pt"
        torch.save(self.payload["model"], weights_only)
        with self.assertRaisesRegex(ValueError, "optimizer"):
            load_training_checkpoint(weights_only)
        self.assertEqual(select_weights(load_checkpoint(weights_only))[1], "model")

    def test_missing_or_mismatched_config_fails(self):
        path, _ = self.export()
        config_path = path.with_name("config.json")
        config = json.loads(config_path.read_text())
        config["model_args"]["norm_mean"][0] += 1
        config_path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "mismatch"):
            load_checkpoint(path)
        config_path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "config.json"):
            load_checkpoint(path)

    def test_export_requires_normalization_and_requested_weights(self):
        del self.payload["args"]["norm_mean"]
        torch.save(self.payload, self.source)
        with self.assertRaisesRegex(ValueError, "norm_mean"):
            self.export()
        self.payload["args"]["norm_mean"] = [0, 0, 0]
        del self.payload["ema"]
        torch.save(self.payload, self.source)
        with self.assertRaisesRegex(ValueError, "Requested ema"):
            self.export()

    def test_legacy_conditioner_initializes_completely(self):
        model = torch.nn.Module()
        model.cond_x_embedder = torch.nn.Linear(2, 1)
        source = {"x_embed_cond.weight": torch.ones(1, 2), "x_embed_cond.bias": torch.ones(1)}
        path = self.root / "legacy.pt"
        torch.save({"model": source}, path)
        report = initialize_from_checkpoint(model, path, use_ema=False)
        self.assertEqual(report["loaded"], 2)
        self.assertEqual(report["missing_after_load"], 0)

    def test_demo_downloads_config_with_safetensors(self):
        from official_demo_inference import paths
        calls = []

        def download(**kwargs):
            calls.append(kwargs["filename"])
            target = Path(kwargs["local_dir"]) / kwargs["filename"]
            target.write_text("test fixture")
            return str(target)

        with patch.dict("os.environ", {"PHYSIFORMER_CKPT_FILENAME": "model.safetensors"}), \
             patch.object(paths, "code_root", return_value=self.root), \
             patch("huggingface_hub.hf_hub_download", side_effect=download):
            checkpoint, _ = paths.ensure_default_checkpoint()
            self.assertEqual(checkpoint.name, "model.safetensors")
            self.assertEqual(calls, ["config.json", "model.safetensors"])
            paths.ensure_default_checkpoint()
            self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
