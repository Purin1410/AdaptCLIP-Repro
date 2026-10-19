"""Runtime support for native AdaptCLIP training, added 2026-10-19."""

import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def state_digest(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def capture_rng(device):
    numpy_state = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
    }


def restore_rng(state, device):
    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                         *numpy_state[2:]))
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'], device)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


class EpochBatchSampler(torch.utils.data.Sampler):
    """Deterministic batches, with full-global-batch padding only for DDP."""

    def __init__(self, length, batch_size, seed, rank=0, world_size=1):
        if length < 1 or batch_size < 1 or not 0 <= rank < world_size:
            raise ValueError('Invalid dataset, batch size or rank')
        self.length, self.batch_size, self.seed = length, batch_size, seed
        self.rank, self.world_size, self.epoch = rank, world_size, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    @property
    def padding(self):
        if self.world_size == 1:
            return 0
        global_batch = self.batch_size * self.world_size
        return math.ceil(self.length / global_batch) * global_batch - self.length

    def __len__(self):
        return math.ceil(self.length / (self.batch_size * self.world_size))

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.length, generator=generator).tolist()
        if self.padding:
            indices += (indices * math.ceil(self.padding / self.length))[:self.padding]
        local = indices[self.rank::self.world_size]
        for start in range(0, len(local), self.batch_size):
            yield local[start:start + self.batch_size]


def convert_syncbn(graph):
    parameters = {key: id(value) for key, value in graph.named_parameters()}
    before = state_digest(graph)
    graph = torch.nn.SyncBatchNorm.convert_sync_batchnorm(graph)
    assert parameters == {key: id(value) for key, value in graph.named_parameters()}
    assert before == state_digest(graph)
    pq = graph.pq_learner
    assert len({id(value) for value in pq.global_adapter}) == 1
    assert len({id(value) for value in pq.local_adapter}) == 1
    return graph


def validate_resume(checkpoint, signature, epochs):
    if checkpoint.get('format_version') != 1:
        raise ValueError('Resume requires last.pth; epoch_N.pth contains inference adapters only')
    if checkpoint['signature'] != signature:
        changed = [key for key in set(checkpoint['signature']) | set(signature)
                   if checkpoint['signature'].get(key) != signature.get(key)]
        raise ValueError(f'Incompatible resume settings: {sorted(changed)}')
    if checkpoint['next_epoch'] >= epochs:
        raise ValueError('--epoch is the total target epoch count and must exceed the saved epoch')
