"""Launcher path regressions; these checks need no PyTorch or GPU."""
import os
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/train_toy_physiformer.sh"


class ToyLauncherTests(unittest.TestCase):
    def launch(self, script, cwd, **overrides):
        env = {key: value for key, value in os.environ.items()
               if key not in ("REPO_ROOT", "SLURM_SUBMIT_DIR", "DATA_ROOT", "MODEL",
                              "BATCH_SIZE", "GRAD_ACCUM", "NPROC_PER_NODE", "SPLIT_FILE",
                              "TRAIN_SPLIT", "VAL_SPLIT", "COND_OBJECT_MATERIAL", "OBJECT_MATERIAL_DIM",
                              "NUM_FRAMES", "NUM_VERTICES", "MAX_NUM_OBJECTS", "EVAL_BATCH_SIZE",
                              "SAVE_EPOCH_FREQ", "SAVE_LAST_FREQ", "TRAIN_VIRTUAL_LENGTH", "EVAL_EPOCH_FREQ")}
        env.update(PYTHON_BIN=sys.executable, RESUME="none", **overrides)
        return subprocess.run(["bash", str(script), "--dry-run"], cwd=cwd,
                              env=env, capture_output=True, text=True)

    def assert_uses_checkout(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"--precomp_root {ROOT}/data_toy", result.stdout)

    def test_direct_launch_ignores_stale_slurm_submission_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assert_uses_checkout(self.launch(SCRIPT, ROOT, SLURM_SUBMIT_DIR=tmp))
            self.assert_uses_checkout(self.launch(SCRIPT, tmp, SLURM_SUBMIT_DIR=tmp))

    def test_spooled_script_uses_working_or_submission_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            spool = Path(tmp) / "job/slurm_script"
            spool.parent.mkdir()
            shutil.copyfile(SCRIPT, spool)
            self.assert_uses_checkout(self.launch(spool, ROOT, SLURM_SUBMIT_DIR=tmp))
            self.assert_uses_checkout(self.launch(spool, tmp, SLURM_SUBMIT_DIR=str(ROOT)))

    def test_explicit_root_is_honored_and_invalid_root_is_explained(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assert_uses_checkout(self.launch(SCRIPT, tmp, REPO_ROOT=str(ROOT)))
            result = self.launch(SCRIPT, ROOT, REPO_ROOT=tmp)
            self.assertEqual(result.returncode, 2)
            self.assertIn(f"REPO_ROOT={tmp} does not contain", result.stderr)

    def test_default_command_uses_full_dataset_training_recipe(self):
        result = self.launch(SCRIPT, ROOT)
        self.assert_uses_checkout(result)
        command = shlex.split(next(line.removeprefix('[command] ') for line in result.stdout.splitlines()
                                  if line.startswith('[command] ')))
        expected = {'--model': 'PhysiFormer', '--batch_size': '8', '--eval_batch_size': '8',
                    '--grad_accum': '4', '--training_protocol': 'altobj', '--ema_decay': '0.9999',
                    '--train_virtual_length': '10000', '--eval_epoch_freq': '10',
                    '--save_epoch_freq': '10', '--save_last_freq': '1000'}
        for flag, value in expected.items():
            self.assertEqual(command[command.index(flag) + 1], value, flag)

    def test_custom_split_does_not_require_toy_layout_or_sample_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('a.npz', 'b.npz', 'c.npz'):
                (root / name).touch()
            split = root / 'custom split.json'
            split.write_text(json.dumps({'learning': ['a.npz', 'b.npz'], 'validation': ['c.npz']}))
            result = self.launch(SCRIPT, ROOT, DATA_ROOT=tmp, SPLIT_FILE=str(split),
                                 TRAIN_SPLIT='learning', VAL_SPLIT='validation',
                                 COND_OBJECT_MATERIAL='0', NUM_VERTICES='0', MAX_NUM_OBJECTS='5')
            self.assertEqual(result.returncode, 0, result.stderr)
            command = shlex.split(next(line.removeprefix('[command] ') for line in result.stdout.splitlines()
                                      if line.startswith('[command] ')))
            self.assertEqual(command[command.index('--split_file') + 1], str(split))
            self.assertIn('--no_cond_object_material', command)
            split.write_text(json.dumps({'learning': ['a.npz'], 'validation': ['a.npz']}))
            result = self.launch(SCRIPT, ROOT, DATA_ROOT=tmp, SPLIT_FILE=str(split),
                                 TRAIN_SPLIT='learning', VAL_SPLIT='validation', COND_OBJECT_MATERIAL='0')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('overlap', result.stderr)


if __name__ == "__main__":
    unittest.main()
