"""CPU checks for the reproduction protocol and optional original-code parity."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from physiformer.scripts import train_npz_elastic as trainer

REFERENCE = Path(os.environ.get('JMT4D_ROOT', Path(__file__).resolve().parents[3] / 'JmT4D'))


class Quadratic(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x, labels, **kwargs):
        return (self.weight - 1).square()


class ReproductionTests(unittest.TestCase):
    def test_epoch_lr_includes_original_zero_lr_warmup(self):
        self.assertEqual(trainer.lr_at_epoch(0, 6000, 5, 4e-5, 5e-6, 'cosine'), 0.0)
        self.assertAlmostEqual(trainer.lr_at_epoch(1, 6000, 5, 4e-5, 5e-6, 'cosine'), 8e-6)
        self.assertEqual(trainer.lr_at_epoch(5, 6000, 5, 4e-5, 5e-6, 'cosine'), 4e-5)

    def test_ema_validation_restores_live_parameters_on_exception(self):
        model = Quadratic()
        ema = trainer.EMA(model, decay=0.5, on_cpu=False)
        with torch.no_grad():
            model.weight.fill_(2)
        ema.update(model)
        with self.assertRaisesRegex(RuntimeError, 'validation failure'):
            with ema.average_parameters(model):
                self.assertEqual(model.weight.item(), 1.0)
                raise RuntimeError('validation failure')
        self.assertEqual(model.weight.item(), 2.0)

    def test_training_discards_tail_and_selects_using_ema_loss(self):
        # Five minibatches: only the first three update. SGD gives live w=0.2;
        # EMA(decay=.5) gives w=.1, so validation must report (.1-1)^2=.81.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(11):
                np.savez(root / f'{i}.npz', vertices=np.full((1, 1, 3), float(i)),
                         mask=np.ones((1, 1)), object_ids=np.zeros(1, dtype=np.int64),
                         first_frame_velocity=np.zeros((1, 3)), num_objects=np.array(1))
            split = root / 'split.json'
            split.write_text(json.dumps({'train': [f'{i}.npz' for i in range(10)], 'val': ['10.npz']}))
            args = ['train', '--precomp_root', str(root), '--split_file', str(split),
                    '--output_dir', str(root / 'run'), '--val_split_name', 'val',
                    '--num_vertices', '1', '--num_frames', '1', '--batch_size', '2',
                    '--grad_accum', '3', '--epochs', '1', '--warmup_epochs', '0',
                    '--lr_schedule', 'constant', '--lr', '.1', '--grad_clip', '0',
                    '--num_workers', '0', '--train_virtual_length', '0', '--amp', 'none',
                    '--save_epoch_freq', '0', '--training_protocol', 'altobj', '--ema_decay', '.5',
                    '--norm_mean', '0', '0', '0', '--norm_std', '1', '1', '1']
            model = Quadratic()
            with patch('sys.argv', args), patch.object(trainer, 'PhysiFormerDenoiser', return_value=model), \
                 patch.object(torch.cuda, 'is_available', return_value=False), \
                 patch.object(torch.optim, 'AdamW', side_effect=lambda params, **kw: torch.optim.SGD(params, **kw)):
                trainer.main()
            checkpoint = torch.load(root / 'run/checkpoint-best.pt', weights_only=False)
            self.assertEqual(checkpoint['step'], 1)
            self.assertAlmostEqual(checkpoint['best_val_loss'], .81, places=6)
            self.assertAlmostEqual(checkpoint['model']['weight'].item(), .2, places=6)
            self.assertAlmostEqual(model.weight.item(), .2, places=6)
            stats = json.loads((root / 'run/train_position_stats.json').read_text())
            self.assertEqual(stats['source'], 'explicit')
            self.assertEqual(stats['position_mean'], [0, 0, 0])

    @unittest.skipUnless((REFERENCE / 'jmt4d').is_dir(), 'Original JmT4D checkout is optional')
    def test_original_npz_order_and_normalization_match(self):
        with patch.object(sys, 'path', [str(REFERENCE), *sys.path]):
            from jmt4d.data.sequence_dataset_vert_multiobj_precomp import (
                ObjVertexSequenceDatasetMultiObjPrecomp as RefDataset,
                ObjVertexSequenceDatasetMultiObjPrecompConfig as RefConfig,
            )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = ['2_obj:1', '1_obj:10', '1_obj:2']
            for selector in entries:
                p = trainer.selector_to_npz(root, selector)
                p.parent.mkdir(exist_ok=True)
                np.savez(p, vertices=np.arange(18, dtype=np.float32).reshape(2, 3, 3),
                         mask=np.array([[1, 1, 0], [1, 1, 0]], dtype=np.uint8),
                         object_ids=np.array([0, 1, 0]), num_objects=np.array(2),
                         first_frame_velocity=np.ones((3, 3), dtype=np.float32))
            mean, std = (.002, -.0009, -.84), (.39, .39, .16)
            ref = RefDataset(RefConfig(precomputed_root=str(root), num_frames=2,
                                      num_vertices=3, norm_mean=mean, norm_std=std,
                                      include_sample_selectors=entries, return_first_frame=True,
                                      return_first_frame_velocity=True, padding_object_id=2, virtual_length=10))
            new = trainer.PrecomputedElasticDataset(precomp_root=root, precomp_roots=None,
                      entries=entries, norm_mean=np.array(mean), norm_std=np.array(std),
                      num_vertices=3, max_num_objects=2, virtual_length=10, sort_paths=True)
            self.assertEqual([str(trainer.selector_to_npz(root, s)) for s in new.entries], ref._paths)
            self.assertEqual(len(new), len(ref))
            for i in range(len(new)):
                a, b = ref[i], new[i]
                for source, target in [('vertices', 'x'), ('mask', 'mask'), ('object_ids', 'object_ids')]:
                    torch.testing.assert_close(a[source], b[target], rtol=0, atol=0)
                torch.testing.assert_close(torch.cat([a['first_frame'], a['first_frame_velocity']], dim=-1),
                                           b['cond'], rtol=0, atol=0)

    @unittest.skipUnless((REFERENCE / 'jmt4d').is_dir(), 'Original JmT4D checkout is optional')
    def test_original_model_loss_gradients_and_schedule_match(self):
        with patch.object(sys, 'path', [str(REFERENCE), *sys.path]):
            from jmt4d.models.mesh_video_dit_spacetemp_vert_multiobj_altobj import (
                MeshVideoDiTSpaceTempVertMultiObjAltObj as RefModel,
                MeshVideoDiTSpaceTempVertMultiObjAltObjConfig as RefConfig,
            )
            from jmt4d.diffusion.denoiser_spacetemp_vert_multiobj_altobj import (
                DenoiserMeshVideoMultiObjAltObj as RefDenoiser,
                MeshVideoDiT_ST_Vert_MultiObj_AltObj_models as ref_registry,
            )
            from jmt4d.diffusion.denoiser import DiffusionConfig as RefDiffusion
            from jmt4d.utils.lr_sched import adjust_learning_rate
            from physiformer.models.physiformer import PhysiFormerBackbone, PhysiFormerConfig
            from physiformer.diffusion.physiformer_denoiser import PHYSIFORMER_MODELS

        kw = dict(num_frames=3, num_vertices=6, depth=4, hidden_size=48, num_heads=4,
                  max_num_objects=2, num_register_tokens=2, object_material_dim=0, use_rope=True)
        torch.manual_seed(19)
        reference = RefModel(RefConfig(**kw))
        torch.manual_seed(19)
        portable = PhysiFormerBackbone(PhysiFormerConfig(**kw))
        self.assertEqual(reference.state_dict().keys(), portable.state_dict().keys())
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(value, portable.state_dict()[name], rtol=0, atol=0)
        # Nonzero attention gates/output weights exercise the whole forward path.
        with torch.no_grad():
            for value in reference.parameters():
                if value.requires_grad:
                    value.normal_(0, .05)
        portable.load_state_dict(reference.state_dict())
        with patch.dict(ref_registry, {'test': lambda **kwargs: reference}), \
             patch.dict(PHYSIFORMER_MODELS, {'test': lambda **kwargs: portable}):
            ref = RefDenoiser(model_name='test', num_frames=3, num_vertices=6, num_classes=1,
                              model_kwargs={}, diffusion=RefDiffusion(noise_scale=.1, label_drop_prob=0))
            new = trainer.PhysiFormerDenoiser(model_name='test', num_frames=3, num_vertices=6, num_classes=1,
                                              model_kwargs={}, diffusion=trainer.DiffusionConfig(noise_scale=.1))
        x = torch.randn(2, 3, 6, 3)
        cond = torch.randn(2, 6, 6)
        mask = torch.ones(2, 3, 6)
        mask[:, :, -1] = 0
        ids = torch.tensor([[0, 0, 0, 1, 1, 2]]).expand(2, -1)
        kwargs = dict(mask=mask, cond_first_frame=cond, object_ids=ids)
        torch.manual_seed(21)
        loss_ref = ref(x, torch.zeros(2, dtype=torch.long), **kwargs)
        torch.manual_seed(21)
        loss_new = new(x, torch.zeros(2, dtype=torch.long), **kwargs)
        torch.testing.assert_close(loss_ref, loss_new, rtol=0, atol=0)
        loss_ref.backward()
        loss_new.backward()
        for (name, a), (_, b) in zip(reference.named_parameters(), portable.named_parameters()):
            self.assertEqual(a.grad is None, b.grad is None, name)
            if a.grad is not None:
                # Autograd/attention backends can differ at float32 roundoff.
                torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-8, msg=name)
        opt = torch.optim.SGD(portable.parameters(), lr=4e-5)
        for epoch in (0, 1, 4, 5, 500, 5999):
            expected = adjust_learning_rate(opt, epoch, lr=4e-5, min_lr=5e-6,
                                            warmup_epochs=5, epochs=6000, schedule='cosine')
            self.assertEqual(trainer.lr_at_epoch(epoch, 6000, 5, 4e-5, 5e-6, 'cosine'), expected)


if __name__ == '__main__':
    unittest.main()
