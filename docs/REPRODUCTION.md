# Training reproduction

This repository is independently maintained by Purin1410. It imports the complete
[gaobb/AdaptCLIP](https://github.com/gaobb/AdaptCLIP) source at commit
`354d9e3332ec5348b3d8e4439111d34f8e94c0a9` into an independent Git history.
The upstream license, copyright notices, adapters, model, losses, dataset loader,
metrics, results and original requirements are retained. Upstream results under
`results/` are the authors' results, not measurements produced by this repository.

## Published training profile

- Source: VisA, official archive and 1-class split, verified using `assets.json`.
- CLIP ViT-L/14@336px, input 518, feature layers 6/12/18/24.
- Fresh TA/VA/PQA adapters, context length 12, VA reduction 4, PQA hidden 128,
  PQA context enabled. Released AdaptCLIP adapters are not used for initialization.
- FP32, Adam, LR 0.001, betas (0.5, 0.999), 15 epochs, training seed 10.
- Two T4s: batch 4 **per GPU**, global batch 8, DDP with SyncBatchNorm.
- Source queries follow upstream `meta['test']` (2,162 labeled images); source
  normal prompts follow `meta['train']` (8,659 images). This is the source-training
  protocol, not evaluation on an unseen target. No target data is used.
- Each distributed epoch deterministically pads to 2,168 query occurrences:
  all 2,162 unique queries plus six repeats, 271 synchronized optimizer updates.
  SyncBN and padding differ from upstream single-GPU training. Equal global batch
  does not imply identical random ordering, floating-point trajectory or metrics.
- Native one-GPU mode uses batch 8 and ordinary BatchNorm with a partial final
  batch of two images. Its deterministic epoch sampler is explicitly seeded.

## What changed

`train.py` retains the native forward/loss computation and adds a shared DDP graph,
standalone process launch, explicit frozen backbone parameters, mask squeezing only
on the channel dimension, deterministic sampling, logs, manifests and recovery.
Freezing CLIP parameters preserves autograd through the text encoder to the learned
text context. Positional embedding resizing is initialized before DDP and checksums.

`training_runtime.py` contains sampling, synchronization conversion, atomic saving
and RNG serialization. `tools/__init__.py` loads optional evaluator/visualization
imports lazily so training does not require the full evaluation dependency stack.
The preparation script accepts paths rather than the upstream machine-specific path,
and generates semantically equivalent metadata. No adapter or loss implementation
is replaced. New scripts and modified code comments are in English.

The upstream `requirements.txt` remains unchanged. The notebook installs the pinned
training profile in `requirements-kaggle.txt`, reusing matching existing packages.
It targets Python 3.12 and Torch 2.11.0/cu128 with torchvision 0.26.0/cu128. Package,
CUDA and GPU inventories are saved with the run. Kaggle image/driver changes can
require a reviewed dependency-profile update; a notebook pin cannot freeze Kaggle's
host OS and driver or guarantee bitwise agreement on other hardware.

## Commands

Download the three public assets listed in `assets.json` into a downloads directory
(the notebook does this automatically), then prepare them:

```bash
python scripts/prepare_visa.py --downloads /path/to/downloads --data /path/to/visa
python train.py --dataset visa --train_data_path /path/to/visa \
  --pretrained_dir /path/to/downloads --save_path /path/to/new/run \
  --devices 2 --sync_bn --batch_size 4 --epoch 15 --seed 10 \
  --features_list 6 12 18 24 --image_size 518 --learning_rate 0.001 \
  --n_ctx 12 --vl_reduction 4 --pq_mid_dim 128 \
  --visual_learner --textual_learner --pq_learner --pq_context
```

For one GPU, replace `--devices 2 --sync_bn --batch_size 4` with
`--devices 1 --batch_size 8`. The upstream adapter flags remain explicit;
`--batch_size` always means the local batch, never the global batch.

## Checkpoints and recovery

`epoch_N.pth` stores exactly the three upstream adapter dictionaries and can be
loaded by the upstream evaluation model. The final checkpoint is `epoch_15.pth`;
there is no validation-based checkpoint selection or early stopping in this profile.
Evaluation dependencies and the copied reverse-direction shell-script assignment
in upstream `scripts/test_adaptclip.sh` need separate review before benchmark use.

`last.pth` is a full, atomically replaced **epoch-boundary** recovery checkpoint:
adapters including BN buffers, Adam state, all-rank RNG, next epoch, global update
and the configuration/source/data signature. It deliberately omits the frozen
backbone, which is rebuilt from the hash-verified official pretrained archive.

```bash
# Same training flags as above, plus:
python train.py ... --epoch 15 --resume /path/to/previous/run/last.pth
```

`--epoch` is the total target, not the number of extra epochs. Resume keeps the
same GPU count, batch, BN policy, worker count, seed, model and source/data identity.
Data/weights/output paths may move. Changing the signature fails explicitly.
An interruption inside an epoch loses that unfinished epoch. New runs reject
nonempty output directories unless recovery is explicitly requested. After an
ended Kaggle session, retain/export the checkpoint and attach it to the next session;
local writes alone do not promise survival of an unexpected provider VM failure.

## Evidence and outputs

`run_manifest.json` records source hashes, arguments, packages, assets, GPU and
sampler settings. Each rank writes a trace; rank 0 prints averaged losses every
10 updates and saves per-epoch receipts with state hashes, VRAM and timing.
`TRAINING_COMPLETE.json` is written only after the total epoch target completes.
A running notebook or passing smoke test is not a completed paper reproduction.

Run the CPU contracts with `python -m unittest discover -s tests -p 'test_*.py' -v`.
The optional notebook smoke cell checks the same training entrypoint on eight
source queries: one single-GPU update, two uninterrupted DDP updates, and a
one-update DDP run resumed in fresh processes. Weights and Adam state must match
within rtol 1e-5 / atol 2e-6. These checks are separate from the fresh 15-epoch run.

The notebook keeps downloaded assets, checkout and smoke scratch under
`/kaggle/temp`; only main-run outputs and small validation receipts are exported.
It pins a training-code commit published before the notebook itself, avoiding a
self-referential Git hash. To develop a new method, keep this baseline commit,
change code in a new branch/commit and update the notebook's revision explicitly.
