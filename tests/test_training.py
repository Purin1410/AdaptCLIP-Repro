"""Numerical, data and recovery contracts for the reproduction runtime."""

import copy
import csv
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train
from adaptcliplib import PQAdapter, TextualAdapter, VisualAdapter, FocalLoss, BinaryDiceLoss
from dataset import Dataset
from tools import get_transform
from training_runtime import (EpochBatchSampler, atomic_checkpoint, capture_rng, convert_syncbn,
                              restore_rng, state_digest, validate_resume)

spec = importlib.util.spec_from_file_location('prepare_visa', ROOT / 'scripts/prepare_visa.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class SmallFrozenCLIP(nn.Module):
    """Use native adapters on synthetic features without downloading CLIP weights."""

    def __init__(self):
        super().__init__()
        self.transformer = SimpleNamespace(get_cast_dtype=lambda: torch.float32)
        self.ln_final = nn.LayerNorm(8)
        self.token_embedding = nn.Embedding(50000, 8)
        self.projection = nn.Parameter(torch.randn(8, 768) / 8)
        self.requires_grad_(False)

    def encode_image(self, image, features, DPAM_layer):
        base = image.mean((1, 2, 3)).reshape(-1, 1)
        vector = torch.sin(torch.arange(768).float()[None] * 0.01 + base)
        patches = [torch.sin(vector[:, None] + torch.arange(17).float()[None, :, None] * (i + 1))
                   for i in range(len(features))]
        return vector, patches

    def encode_text(self, prompts, tokens):
        return prompts.mean(1) @ self.projection


def make_graph():
    torch.manual_seed(10)
    model = SmallFrozenCLIP()
    ta = TextualAdapter(model, 56, 12)
    ta.static_text_features = torch.randn(2, 768)
    args = train.build_parser().parse_args(['--visual_learner', '--textual_learner', '--pq_learner', '--pq_context'])
    return train.TrainingGraph(model, ta, VisualAdapter(56, 14),
                               PQAdapter(56, 14, context=True, layers_num=4), args, 20)


class TrainingContracts(unittest.TestCase):
    def test_native_adapters_loss_and_gradients(self):
        graph = make_graph()
        reference = copy.deepcopy(graph)
        torch.manual_seed(20)
        image, prompt = torch.randn(4, 3, 56, 56), torch.randn(4, 1, 3, 56, 56)
        labels, masks = torch.tensor([0, 1, 0, 1]), (torch.rand(4, 56, 56) > 0.5).float()
        actual = graph(image, prompt, labels, masks)
        with torch.no_grad():
            q, patches = reference.model.encode_image(image, [6, 12, 18, 24], 20)
            p, pp = reference.model.encode_image(prompt.flatten(0, 1), [6, 12, 18, 24], 20)
            p, pp = p[:, None], [value[:, None] for value in pp]
        ga, ma = reference.visual_learner(q, patches, reference.textual_learner.static_text_features)
        text, tokens = reference.textual_learner()
        gb, mb = reference.textual_learner.compute_global_local_score(q, patches, reference.model.encode_text(text, tokens))
        gp, mp, _ = reference.pq_learner(q, patches, p, pp)
        focal, dice = FocalLoss(), BinaryDiceLoss()
        expected = (sum(F.cross_entropy(logit, labels) for logit in [ga, gb, *gp]),
                    sum(focal(value, masks) + dice(value[:, 1], masks) + dice(value[:, 0], 1 - masks)
                        for value in [ma, mb, *mp]))
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-6)
        sum(actual).backward()
        sum(expected).backward()
        for name, module in graph.adapters.items():
            for left, right in zip(module.parameters(), reference.adapters[name].parameters()):
                self.assertIsNotNone(left.grad)
                self.assertTrue(torch.isfinite(left.grad).all())
                torch.testing.assert_close(left.grad, right.grad, rtol=2e-5, atol=2e-6)
        self.assertTrue(all(value.grad is None for value in graph.model.parameters()))

    def test_syncbn_shared_heads_and_local_fallback(self):
        graph = make_graph()
        reference = copy.deepcopy(graph)
        converted = convert_syncbn(graph)
        self.assertEqual(sum(isinstance(m, nn.SyncBatchNorm) for m in converted.modules()), 10)
        self.assertEqual(state_digest(reference), state_digest(converted))
        torch.manual_seed(30)
        inputs = (torch.randn(4, 3, 56, 56), torch.randn(4, 1, 3, 56, 56),
                  torch.tensor([0, 1, 0, 1]), torch.zeros(4, 56, 56))
        for left, right in zip(reference(*inputs), converted(*inputs)):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_sampler_coverage_padding_and_epochs(self):
        for epoch in [0, 1]:
            samplers = [EpochBatchSampler(2162, 4, 10, rank, 2) for rank in range(2)]
            for sampler in samplers:
                sampler.set_epoch(epoch)
                self.assertEqual((len(sampler), sampler.padding), (271, 6))
            values = [index for sampler in samplers for batch in sampler for index in batch]
            self.assertEqual(len(values), 2168)
            self.assertEqual(set(values), set(range(2162)))
            self.assertTrue(all(len(batch) == 4 for sampler in samplers for batch in sampler))
        single = EpochBatchSampler(2162, 8, 10)
        self.assertEqual((len(single), single.padding, len(list(single)[-1])), (271, 0, 2))
        tiny = EpochBatchSampler(1, 4, 10, 0, 2)
        self.assertEqual(list(tiny), [[0, 0, 0, 0]])

    def test_native_dataset_and_mask_batch_dimension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rgb = np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)
            Image.fromarray(rgb).save(root / 'image.png')
            Image.fromarray((np.arange(24 * 32).reshape(24, 32) % 2 * 255).astype(np.uint8)).save(root / 'mask.png')
            normal = dict(img_path='image.png', mask_path='', cls_name='candle', specie_name='', anomaly=0)
            anomalous = dict(normal, anomaly=1, mask_path='mask.png')
            (root / 'meta.json').write_text(json.dumps({'train': {'candle': [normal]}, 'test': {'candle': [anomalous]}}))
            transform, target = get_transform(56)
            dataset = Dataset(root, transform, target, 'visa', 1, directory, mode='train', seed=10)
            item = dataset[0]
            expected = transform(Image.open(root / 'image.png'))
            torch.testing.assert_close(item['img'], expected, rtol=0, atol=0)
            torch.testing.assert_close(item['prompt_img'][0], expected, rtol=0, atol=0)
            batch = torch.utils.data.default_collate([item])
            mask = (batch['img_mask'].squeeze(1) > 0.5).float()
            self.assertEqual(tuple(mask.shape), (1, 56, 56))
            expected_mask = target(Image.fromarray((np.asarray(Image.open(root / 'mask.png')) > 0).astype(np.uint8) * 255))
            torch.testing.assert_close(mask[0], (expected_mask.squeeze(0) > 0.5).float(), rtol=0, atol=0)

    def test_metadata_matches_upstream_schema(self):
        rows = []
        for index in range(10821):
            phase = 'train' if index < 8659 else 'test'
            rows.append({'object': prepare.CLASSES[index % 12], 'split': phase,
                         'label': 'normal' if phase == 'train' or index % 2 else 'anomaly',
                         'image': f'images/{index}.png', 'mask': f'masks/{index}.png'})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'split.csv'
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            meta = prepare.make_meta(path)
        for row in rows:
            native = {'img_path': row['image'], 'mask_path': row['mask'] if row['label'] == 'anomaly' else '',
                      'cls_name': row['object'], 'specie_name': '', 'anomaly': int(row['label'] == 'anomaly')}
            self.assertEqual(meta[row['split']][row['object']].pop(0), native)

    def test_atomic_checkpoint_and_fresh_process_rng_optimizer(self):
        torch.manual_seed(10)
        random.seed(10)
        np.random.seed(10)
        model = nn.Linear(3, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))
        def update():
            x = torch.randn(4, 3) + random.random() + float(np.random.rand())
            optimizer.zero_grad()
            model(x).square().mean().backward()
            optimizer.step()
        update()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'last.pth'
            atomic_checkpoint(checkpoint, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                          'rng': capture_rng(torch.device('cpu'))})
            update()
            expected = state_digest(model)
            program = '''
import sys, random
import numpy as np
from pathlib import Path
import torch
from torch import nn
from training_runtime import restore_rng, state_digest
payload=torch.load(sys.argv[1], map_location='cpu', weights_only=True)
model=nn.Linear(3,2); model.load_state_dict(payload['model'])
optimizer=torch.optim.Adam(model.parameters(), lr=.001, betas=(.5,.999))
optimizer.load_state_dict(payload['optimizer'])
restore_rng(payload['rng'], torch.device('cpu'))
x=torch.randn(4,3)+random.random()+float(np.random.rand())
optimizer.zero_grad();model(x).square().mean().backward();optimizer.step()
Path(sys.argv[2]).write_text(state_digest(model))
'''
            result = Path(directory) / 'digest'
            subprocess.run([sys.executable, '-c', program, str(checkpoint), str(result)], cwd=ROOT, check=True)
            self.assertEqual(result.read_text(), expected)
            self.assertFalse(checkpoint.with_suffix('.pth.partial').exists())
        signature = {'devices': 2, 'batch_size': 4}
        validate_resume({'format_version': 1, 'signature': signature, 'next_epoch': 1}, signature, 15)
        with self.assertRaises(ValueError):
            validate_resume({'format_version': 1, 'signature': signature, 'next_epoch': 1}, {'devices': 1}, 15)
        with self.assertRaises(ValueError):
            validate_resume({}, signature, 15)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
