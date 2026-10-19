"""Bounded native single/DDP training and fresh-process recovery on eight VisA queries."""

import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training_runtime import atomic_json


def compare_tree(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            compare_tree(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            compare_tree(left, right)
    else:
        assert actual == expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--pretrained_dir', required=True)
    parser.add_argument('--scratch', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    data, scratch, output = Path(args.data).resolve(), Path(args.scratch).resolve(), Path(args.out).resolve()
    scratch.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    source = json.loads((data / 'meta.json').read_text())
    meta = {'train': {}, 'test': {}}
    for name in list(source['test'])[:4]:
        rows = source['test'][name]
        meta['test'][name] = [copy.deepcopy(next(row for row in rows if row['anomaly'] == label)) for label in [0, 1]]
        meta['train'][name] = copy.deepcopy(source['train'][name])
        for split in ['train', 'test']:
            for row in meta[split][name]:
                row['img_path'] = str(data / row['img_path'])
                if row['mask_path']:
                    row['mask_path'] = str(data / row['mask_path'])
    fixture = scratch / 'fixture'
    fixture.mkdir()
    (fixture / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    common = [sys.executable, '-u', str(ROOT / 'train.py'), '--dataset', 'visa',
              '--train_data_path', str(fixture), '--pretrained_dir', args.pretrained_dir,
              '--image_size', '518', '--features_list', '6', '12', '18', '24',
              '--n_ctx', '12', '--vl_reduction', '4', '--pq_mid_dim', '128',
              '--visual_learner', '--textual_learner', '--pq_learner', '--pq_context',
              '--seed', '10', '--log_every', '1', '--num_workers', '4']
    def run(name, devices, epochs, resume=None):
        folder = scratch / name
        command = [*common, '--save_path', str(folder), '--epoch', str(epochs),
                   '--devices', str(devices), '--batch_size', str(8 // devices)]
        if devices > 1:
            command.append('--sync_bn')
        if resume:
            command.extend(['--resume', str(resume)])
        print(json.dumps({'event': 'smoke_stage', 'name': name, 'devices': devices, 'epochs': epochs}), flush=True)
        with (output / (name + ('_resume' if resume else '') + '.log')).open('w') as stream:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, start_new_session=True)
            def terminate():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
            timer = threading.Timer(900, terminate)
            timer.start()
            try:
                for line in process.stdout:
                    stream.write(line)
                    print(line, end='', flush=True)
                result = process.wait()
            except BaseException:
                terminate()
                process.wait(timeout=30)
                raise
            finally:
                timer.cancel()
        if result:
            raise RuntimeError(f'GPU smoke stage {name} failed with exit {result}')
        receipt = json.loads((folder / f'epoch_{epochs}.json').read_text())
        assert receipt['backbone_unchanged'] and all(receipt['changed_adapters'].values())
        assert receipt['updates_this_epoch'] == 1 and receipt['global_step'] == epochs
        if devices > 1:
            assert receipt['syncbn_count'] == 10
            assert receipt['syncbn_forward_calls_this_process'] >= 22
        atomic_json(output / (name + f'_epoch{epochs}.json'), receipt)
        return torch.load(folder / 'last.pth', map_location='cpu', weights_only=True)
    run('single_gpu', 1, 1)
    uninterrupted = run('ddp_reference', 2, 2)
    run('ddp_resumed', 2, 1)
    resumed = run('ddp_resumed', 2, 2, scratch / 'ddp_resumed/last.pth')
    assert uninterrupted['signature'] == resumed['signature']
    compare_tree(resumed['adapters'], uninterrupted['adapters'])
    compare_tree(resumed['optimizer'], uninterrupted['optimizer'])
    receipt = {'pass': True, 'queries': 8, 'single_gpu_updates': 1,
               'ddp_reference_updates': 2, 'ddp_interrupted_then_resumed_updates': 2,
               'adapter_and_optimizer_close_to_uninterrupted': True,
               'tolerance': {'rtol': 1e-5, 'atol': 2e-6},
               'scope': 'Bounded runtime and fresh-process epoch-boundary resume; not convergence or benchmark evaluation.'}
    atomic_json(output / 'GPU_SMOKE.json', receipt)
    print(json.dumps({'event': 'gpu_smoke_pass', **receipt}), flush=True)


if __name__ == '__main__':
    main()
