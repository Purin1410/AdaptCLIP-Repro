"""AdaptCLIP training; modified 2026-10-19 for standalone DDP and epoch resume.

The upstream adapters, source dataset and loss expressions are retained.
"""

import argparse
from datetime import datetime, timedelta, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from einops import rearrange

import adaptcliplib
from adaptcliplib import BinaryDiceLoss, FocalLoss, PQAdapter, TextualAdapter, VisualAdapter
from dataset import Dataset
from tools import get_transform, setup_seed
from training_runtime import (EpochBatchSampler, atomic_checkpoint, atomic_json,
                              capture_rng, convert_syncbn, cpu_tree, file_sha256,
                              restore_rng, seed_worker, state_digest, validate_resume)

ROOT = Path(__file__).resolve().parent
BASE_HASH = '3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02'


class TrainingGraph(nn.Module):
    """Wrap the native loss computation once so DDP sees every trainable adapter."""

    def __init__(self, model, textual_learner, visual_learner, pq_learner, args, dpam_layer):
        super().__init__()
        self.model = model
        self.textual_learner = textual_learner
        self.visual_learner = visual_learner
        self.pq_learner = pq_learner
        self.args, self.dpam_layer = args, dpam_layer
        self.loss_focal, self.loss_dice = FocalLoss(), BinaryDiceLoss()

    @property
    def adapters(self):
        return {name: getattr(self, name) for name in
                ['textual_learner', 'visual_learner', 'pq_learner']}

    def forward(self, image, prompt_image, label, gt):
        b, s, c, h, w = prompt_image.shape
        prompt_image = prompt_image.reshape(-1, c, h, w)
        with torch.no_grad():
            query_feats, query_patch_feats = self.model.encode_image(
                image, self.args.features_list, DPAM_layer=self.dpam_layer)
            prompt_feats, prompt_patch_feats = self.model.encode_image(
                prompt_image, self.args.features_list, DPAM_layer=self.dpam_layer)
            prompt_feats = prompt_feats.reshape(b, s, -1)
            for idx in range(len(self.args.features_list)):
                prompt_patch_feats[idx] = rearrange(
                    prompt_patch_feats[idx], '(b s) l d -> b s l d', b=b, s=s)

        global_loss, local_loss = 0, 0
        if self.args.visual_learner:
            global_logit, local_score = self.visual_learner(
                query_feats, query_patch_feats, self.textual_learner.static_text_features)
            global_loss += F.cross_entropy(global_logit, label.long())
            local_loss += self.loss_focal(local_score, gt)
            local_loss += self.loss_dice(local_score[:, 1, :, :], gt)
            local_loss += self.loss_dice(local_score[:, 0, :, :], 1 - gt)

        if self.args.textual_learner:
            learned_prompts, tokenized_prompts = self.textual_learner()
            learned_text_features = self.model.encode_text(learned_prompts, tokenized_prompts).float()
            global_logit, local_score = self.textual_learner.compute_global_local_score(
                query_feats, query_patch_feats, learned_text_features)
            global_loss += F.cross_entropy(global_logit, label.long())
            local_loss += self.loss_focal(local_score, gt)
            local_loss += self.loss_dice(local_score[:, 1, :, :], gt)
            local_loss += self.loss_dice(local_score[:, 0, :, :], 1 - gt)

        if self.args.pq_learner:
            global_logit, local_score_list, align_score_list = self.pq_learner(
                query_feats, query_patch_feats, prompt_feats, prompt_patch_feats)
            for i in range(len(global_logit)):
                global_loss += F.cross_entropy(global_logit[i], label.long())
            for i in range(len(local_score_list)):
                local_loss += self.loss_focal(local_score_list[i], gt)
                local_loss += self.loss_dice(local_score_list[i][:, 1, :, :], gt)
                local_loss += self.loss_dice(local_score_list[i][:, 0, :, :], 1 - gt)
        return global_loss, local_loss


def build_graph(args, device):
    if args.pretrained_model != 'ViT-L/14@336px':
        raise ValueError('This training profile validates ViT-L/14@336px only; other upstream models are unchanged')
    model, _ = adaptcliplib.load(args.pretrained_model, device='cpu', download_root=args.pretrained_dir)
    model.visual.DAPM_replace(DPAM_layer=20)
    model.requires_grad_(False)
    textual_learner = TextualAdapter(model, args.image_size, args.n_ctx)
    visual_learner = VisualAdapter(args.image_size, 14, input_dim=768, reduction=args.vl_reduction)
    pq_learner = PQAdapter(args.image_size, 14, context=args.pq_context, input_dim=768,
                           mid_dim=args.pq_mid_dim, layers_num=len(args.features_list))
    graph = TrainingGraph(model, textual_learner, visual_learner, pq_learner, args, 20).to(device)
    for name, module in graph.adapters.items():
        module.requires_grad_(getattr(args, name))
    graph.train()
    model.eval()
    textual_learner.prepare_static_text_feature(model)
    with torch.no_grad():
        model.encode_image(torch.zeros(1, 3, args.image_size, args.image_size, device=device),
                           args.features_list, DPAM_layer=20)
    return graph


def source_state():
    paths = [ROOT / 'train.py', ROOT / 'training_runtime.py']
    for folder in ['adaptcliplib', 'dataset', 'tools']:
        paths.extend(sorted((ROOT / folder).rglob('*.py')))
    hashes = {str(path.relative_to(ROOT)): file_sha256(path) for path in paths}
    hashes['adaptcliplib/bpe_simple_vocab_16e6.txt.gz'] = file_sha256(
        ROOT / 'adaptcliplib/bpe_simple_vocab_16e6.txt.gz')
    try:
        commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {'commit': commit, 'dirty': dirty, 'files': hashes}


def gather(value, world):
    if world == 1:
        return [value]
    values = [None] * world
    dist.all_gather_object(values, value)
    return values


def mean_ranks(value, world):
    result = value.detach().clone()
    if world > 1:
        dist.all_reduce(result)
        result /= world
    return result


def emit(event, **fields):
    print(json.dumps({'event': event, **fields}, allow_nan=False), flush=True)


def train(rank, args):
    world = args.devices
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    setup_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if world > 1:
        dist.init_process_group('nccl', rank=rank, world_size=world, timeout=timedelta(seconds=300))
    try:
        _train(rank, args, device, world)
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def _train(rank, args, device, world):
    output = Path(args.save_path)
    graph = build_graph(args, device)
    if args.sync_bn:
        graph = convert_syncbn(graph)
    backbone_hash = state_digest(graph.model)
    before = {name: state_digest(module) for name, module in graph.adapters.items()}
    syncbn_count = sum(isinstance(module, nn.SyncBatchNorm) for module in graph.modules())
    bn_calls = [0]
    def record_bn(module, inputs):
        assert module.training and inputs[0].is_cuda and world > 1
        bn_calls[0] += 1
    if args.sync_bn:
        for module in graph.modules():
            if isinstance(module, nn.SyncBatchNorm):
                module.register_forward_pre_hook(record_bn)
    preprocess, target_transform = get_transform(image_size=args.image_size)
    dataset = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform,
                      dataset_name=args.dataset, k_shots=args.k_shots, save_dir=args.save_path,
                      mode='train', seed=args.seed)
    sampler = EpochBatchSampler(len(dataset), args.batch_size, args.seed, rank, world)
    signature_keys = ['dataset', 'pretrained_model', 'n_ctx', 'features_list', 'learning_rate',
                      'batch_size', 'image_size', 'seed', 'k_shots', 'visual_learner',
                      'textual_learner', 'pq_learner', 'vl_reduction', 'pq_mid_dim',
                      'pq_context', 'devices', 'sync_bn', 'num_workers']
    source = source_state()
    signature = {key: getattr(args, key) for key in signature_keys}
    signature.update(meta_sha256=file_sha256(Path(args.train_data_path) / 'meta.json'),
                     base_sha256=BASE_HASH, source_files=source['files'], precision='FP32',
                     tail_policy='pad_global_batch' if world > 1 else 'native_partial_tail')
    optimizer = torch.optim.Adam([p for p in graph.parameters() if p.requires_grad],
                                 lr=args.learning_rate, betas=(0.5, 0.999))
    start_epoch, step = 0, 0
    loaded = None
    if args.resume:
        loaded = torch.load(args.resume, map_location='cpu', weights_only=True)
        validate_resume(loaded, signature, args.epoch)
        for name, module in graph.adapters.items():
            module.load_state_dict(loaded['adapters'][name], strict=True)
        optimizer.load_state_dict(loaded['optimizer'])
        start_epoch, step = loaded['next_epoch'], loaded['global_step']
        assert loaded['backbone_sha256'] == backbone_hash
    wrapped = nn.parallel.DistributedDataParallel(graph, device_ids=[rank], broadcast_buffers=True) if world > 1 else graph
    setup_seed(args.seed + rank)
    if loaded is not None:
        restore_rng(loaded['rng_by_rank'][rank], device)
        before = {name: state_digest(module) for name, module in graph.adapters.items()}
    if rank == 0:
        manifest = {'args': vars(args), 'source': source, 'signature': signature,
                    'public_assets': json.loads((ROOT / 'assets.json').read_text()),
                    'started_at': datetime.now(timezone.utc).isoformat(),
                    'torch': str(torch.__version__), 'torchvision': importlib.metadata.version('torchvision'),
                    'cuda': torch.version.cuda, 'gpus': gather_gpu_names(world),
                    'packages': dict(sorted((d.metadata['Name'], d.version)
                                           for d in importlib.metadata.distributions() if d.metadata['Name'])),
                    'queries': len(dataset), 'query_split': 'source meta[test]',
                    'prompt_split': 'source meta[train]', 'sampler_padding': sampler.padding,
                    'updates_per_epoch': len(sampler), 'global_batch': args.batch_size * world,
                    'syncbn_count': syncbn_count, 'backbone_sha256': backbone_hash,
                    'target_evaluation': False}
        atomic_json(output / (f'resume_from_epoch_{start_epoch}.json' if loaded else 'run_manifest.json'), manifest)
        emit('training_start', resumed=bool(loaded), next_epoch=start_epoch + 1, global_step=step,
             world_size=world, local_batch=args.batch_size, global_batch=args.batch_size * world,
             updates_per_epoch=len(sampler), queries=len(dataset), padding=sampler.padding,
             syncbn_count=syncbn_count, source_commit=source['commit'])
    torch.cuda.reset_peak_memory_stats(device)
    run_start = time.monotonic()
    with (output / f'rank{rank}.jsonl').open('a', buffering=1) as trace:
        for epoch in range(start_epoch, args.epoch):
            epoch_start = time.monotonic()
            sampler.set_epoch(epoch)
            generator = torch.Generator().manual_seed(args.seed + 100000 * epoch + rank)
            loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler,
                num_workers=args.num_workers, worker_init_fn=seed_worker,
                generator=generator, persistent_workers=False)
            totals = torch.zeros(2, device=device)
            for batch_index, items in enumerate(loader):
                image = items['img'].to(device)
                prompt_image = items['prompt_img'].to(device)
                label = items['anomaly'].to(device)
                gt = (items['img_mask'].squeeze(1).to(device) > 0.5).float()
                assert gt.ndim == 3
                optimizer.zero_grad(set_to_none=True)
                global_loss, local_loss = wrapped(image, prompt_image, label, gt)
                loss = global_loss + local_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite loss on rank {rank}, step {step + 1}')
                loss.backward()
                norms = {}
                for name, module in graph.adapters.items():
                    if not getattr(args, name):
                        continue
                    gradients = [p.grad for p in module.parameters() if p.requires_grad]
                    if not gradients or any(value is None for value in gradients):
                        raise RuntimeError(f'Missing gradients for {name}')
                    norm = torch.stack([value.detach().float().square().sum() for value in gradients]).sum().sqrt()
                    if not torch.isfinite(norm) or norm <= 0:
                        raise FloatingPointError(f'Invalid gradient for {name} on rank {rank}')
                    norms[name] = float(norm)
                optimizer.step()
                step += 1
                losses = torch.stack([global_loss.detach(), local_loss.detach()])
                totals += losses
                trace.write(json.dumps({'epoch': epoch + 1, 'batch': batch_index + 1, 'step': step,
                    'rank': rank, 'global_loss': float(global_loss.detach()),
                    'local_loss': float(local_loss.detach()), 'grad_norm': norms,
                    'query_paths': [os.path.relpath(path, args.train_data_path) for path in items['img_path']]},
                    allow_nan=False) + '\n')
                if batch_index == 0 or step % args.log_every == 0 or batch_index + 1 == len(sampler):
                    means = mean_ranks(losses, world)
                    if rank == 0:
                        emit('training', epoch=epoch + 1, epochs=args.epoch, batch=batch_index + 1,
                             batches=len(sampler), step=step, loss=float(means.sum()),
                             global_loss=float(means[0]), local_loss=float(means[1]),
                             local_batch=image.shape[0], global_batch=image.shape[0] * world,
                             seconds=time.monotonic() - epoch_start,
                             peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2 ** 30)
            assert all(parameter.grad is None for parameter in graph.model.parameters())
            assert state_digest(graph.model) == backbone_hash, 'Frozen backbone changed'
            states = gather({name: state_digest(module) for name, module in graph.adapters.items()}, world)
            assert all(value == states[0] for value in states), 'Adapter parameters or BN buffers differ across ranks'
            changed = {name: before[name] != states[0][name] for name in before if getattr(args, name)}
            assert all(changed.values()), 'An enabled adapter did not change'
            rng_states = gather(capture_rng(device), world)
            means = mean_ranks(totals / len(sampler), world)
            receipt = {'epoch': epoch + 1, 'global_step': step, 'updates_this_epoch': len(sampler),
                'seconds': time.monotonic() - epoch_start, 'mean_loss': float(means.sum()),
                'global_loss': float(means[0]), 'local_loss': float(means[1]),
                'local_batch': args.batch_size, 'global_batch': args.batch_size * world,
                'world_size': world, 'adapter_hashes_all_ranks': states,
                'backbone_unchanged': True, 'changed_adapters': changed,
                'syncbn_count': syncbn_count, 'syncbn_forward_calls_this_process': bn_calls[0],
                'peak_reserved_gib': torch.cuda.max_memory_reserved(device) / 2 ** 30}
            if rank == 0:
                adapters = {name: cpu_tree(module.state_dict()) for name, module in graph.adapters.items()}
                if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epoch:
                    atomic_checkpoint(output / f'epoch_{epoch + 1}.pth', adapters)
                checkpoint = {'format_version': 1, 'adapters': adapters,
                    'optimizer': cpu_tree(optimizer.state_dict()), 'next_epoch': epoch + 1,
                    'global_step': step, 'rng_by_rank': rng_states,
                    'signature': signature, 'backbone_sha256': backbone_hash}
                atomic_checkpoint(output / 'last.pth', checkpoint)
                reloaded = torch.load(output / 'last.pth', map_location='cpu', weights_only=True)
                assert reloaded['global_step'] == step and reloaded['next_epoch'] == epoch + 1
                assert all(int(value['step']) == step for value in reloaded['optimizer']['state'].values())
                for name, state in adapters.items():
                    assert all(torch.equal(value, reloaded['adapters'][name][key]) for key, value in state.items())
                receipt['checkpoint_load_verified'] = True
                receipt['checkpoint_sha256'] = file_sha256(output / 'last.pth')
                atomic_json(output / f'epoch_{epoch + 1}.json', receipt)
                emit('epoch_end', **receipt)
            if world > 1:
                dist.barrier()
    if rank == 0:
        atomic_json(output / 'TRAINING_COMPLETE.json', {'epochs': args.epoch, 'global_step': step,
            'seconds_this_process': time.monotonic() - run_start,
            'checkpoint': f'epoch_{args.epoch}.pth', 'target_evaluation': False})
        emit('training_complete', epochs=args.epoch, global_step=step)


def gather_gpu_names(world):
    return [torch.cuda.get_device_name(index) for index in range(world)]


def build_parser():
    parser = argparse.ArgumentParser('AdaptCLIP')
    parser.add_argument('--train_data_path', default='./data/visa')
    parser.add_argument('--save_path', default='./checkpoint')
    parser.add_argument('--dataset', default='mvtec')
    parser.add_argument('--pretrained_model', default='ViT-L/14@336px')
    parser.add_argument('--n_ctx', type=int, default=12)
    parser.add_argument('--features_list', type=int, nargs='+', default=[6, 12, 18, 24])
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU; global batch = batch_size * devices')
    parser.add_argument('--image_size', type=int, default=518)
    parser.add_argument('--print_freq', type=int, default=1, help='Retained for upstream CLI compatibility; epoch receipts are always saved')
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--k_shots', type=int, default=1)
    for name in ['visual_learner', 'textual_learner', 'pq_learner', 'pq_context']:
        parser.add_argument('--' + name, action='store_true')
    parser.add_argument('--vl_reduction', type=int, default=4)
    parser.add_argument('--pq_mid_dim', type=int, default=128)
    parser.add_argument('--devices', type=int, choices=[1, 2], default=1)
    parser.add_argument('--sync_bn', action='store_true')
    parser.add_argument('--pretrained_dir', default=str(Path('~/.cache/clip').expanduser()))
    parser.add_argument('--resume', default=None, help='Path to last.pth; --epoch remains the total epoch target')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_every', type=int, default=10)
    return parser


def main():
    args = build_parser().parse_args()
    if not any([args.visual_learner, args.textual_learner, args.pq_learner]):
        raise ValueError('Enable at least one adapter, as in the upstream training command')
    if args.sync_bn and args.devices == 1:
        raise ValueError('--sync_bn requires --devices 2; omit it for native single-GPU BatchNorm')
    if args.devices > 1 and not args.sync_bn:
        raise ValueError('The validated two-GPU profile requires --sync_bn')
    if min(args.epoch, args.batch_size, args.save_freq, args.log_every) < 1 or args.num_workers < 0:
        raise ValueError('Invalid epoch, batch, checkpoint, logging or worker count')
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.devices:
        raise RuntimeError(f'Training requires {args.devices} visible CUDA GPUs')
    args.train_data_path = str(Path(args.train_data_path).resolve())
    args.save_path = str(Path(args.save_path).resolve())
    args.pretrained_dir = str(Path(args.pretrained_dir).expanduser().resolve())
    base = Path(args.pretrained_dir) / 'ViT-L-14-336px.pt'
    if not base.is_file() or file_sha256(base) != BASE_HASH:
        raise ValueError('Download the official CLIP backbone into --pretrained_dir; hash verification failed')
    output = Path(args.save_path)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError('Output is not empty; choose a new --save_path or explicitly --resume last.pth')
    if args.resume:
        args.resume = str(Path(args.resume).resolve(strict=True))
    output.mkdir(parents=True, exist_ok=True)
    if args.devices > 1:
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        if 'MASTER_PORT' not in os.environ:
            with socket.socket() as handle:
                handle.bind(('127.0.0.1', 0))
                os.environ['MASTER_PORT'] = str(handle.getsockname()[1])
        os.environ.setdefault('OMP_NUM_THREADS', '2')
        os.environ.setdefault('NCCL_DEBUG', 'WARN')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.multiprocessing.spawn(train, args=(args,), nprocs=args.devices, join=True)
    else:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        train(0, args)


if __name__ == '__main__':
    main()
