# Agent working notes

Read [docs/REPRODUCTION.md](docs/REPRODUCTION.md) before changing or running this
project. These instructions describe the reproduction contract; they do not
authorize new experiments, GPU submissions, cancellations or publication outside
the user's current task. Use authorization already provided in the conversation.

## Baseline identity and scope

- This is an independent repository derived from `gaobb/AdaptCLIP`, upstream commit
  `354d9e3332ec5348b3d8e4439111d34f8e94c0a9`. Preserve its license and attribution.
- The initial baseline is release `v0.1.0`; its notebook pins training code commit
  `1d4859b066b8a449b1a40836fce7b762425b9f96`. Inspect the notebook's `REVISION`
  before executing: editing `main` does not change an already pinned notebook.
- The first pipeline trains on VisA. Full target-dataset evaluation is a separate
  stage that has not been supplied by this notebook. Results already under
  `results/` and tables in the upstream README belong to the original authors.
- Retain a brief `CHECKPOINT.md` in the task's working directory for multi-part
  work: changes, checks, run identity and remaining work. Preserve unrelated files,
  existing runs, logs and checkpoint versions.

## Training settings to keep fixed

- Two T4s, FP32, DDP + SyncBatchNorm; **batch 4 per GPU, global batch 8**.
  `--batch_size` always means the batch on one GPU. Two ranks run 271 synchronized
  updates per VisA epoch, not 542 independent weight updates.
- Fifteen epochs, training seed 10, Adam LR 0.001 and betas `(0.5, 0.999)`.
  CLIP ViT-L/14@336px, image size 518, features 6/12/18/24, TA context 12,
  VA reduction 4, PQA hidden 128; enable TA, VA, PQA and `--pq_context` explicitly.
  The training entrypoint validates this CLIP backbone only.
- VisA queries are the 2,162 labeled source images in `meta['test']`; normal
  prompts come from the 8,659 images in `meta['train']`. This naming follows the
  upstream source-training protocol. Do not silently replace it with normal-only
  training or use an unseen target's test set for training/checkpoint selection.
- Distributed sampling covers every query and pads six repeats to 2,168
  occurrences per epoch. Single-GPU mode uses batch 8, ordinary BatchNorm and a
  partial final batch of two. SyncBN, sampling order and padding mean this is not
  a claim of identical numerical trajectories to the authors' single-GPU run.
- Keep the backbone frozen while retaining the text encoder's autograd path to
  learned context. Preserve PQA's shared heads. Convert SyncBN before DDP and
  optimizer creation; retain masks as `[B,H,W]` using `squeeze(1)`, not `squeeze()`.

## Implementing a training-time idea

- Keep the baseline release immutable. Implement the authorized idea on a new
  branch such as `method/<name>` with a separate commit and output directory.
  Keep changes to the method's actual components; avoid unrelated refactors.
- Compare fresh baseline and method runs using the same source data, backbone,
  resolution, optimizer, LR, epoch/update budget, global/local batch, BN policy,
  precision, seed, worker count and evaluation protocol. Record intentional
  differences; a batch, scheduler or precision change is an experimental change.
- Preserve shared baseline initialization and query/prompt sampling. Adding RNG
  draws can change initialization or support selection even with the same seed.
  Record the relevant hashes/paths rather than assuming seed equality is enough.
- Baseline training uses fresh adapters, not released AdaptCLIP adapter weights
  or weights from the smoke test. If the idea requires a different initialization
  protocol, make it explicit and use a matching control.
- Evaluate paired methods on the same target queries, support images, shots,
  preprocessing, scoring/fusion and metric implementation. Do not choose a method
  or checkpoint from target-test performance. Lower training loss alone does not
  establish improved anomaly detection.
- The published scripts use training seed 10 by default. Their few-shot evaluation
  uses support seeds 10/20/30 on the same checkpoint; these are not three training
  replicates. Zero-shot uses seed 10. Support selection uses sampling with
  replacement, so sets may overlap or contain repeats; retain support manifests.

## Notebook, assets and source changes

- Keep the notebook as an execution interface: public downloads, pinned clone,
  dependencies, data preparation, optional smoke checks and `python train.py ...`.
  Put training logic in the repository, not an embedded worker or notebook patch.
- Verify assets against `assets.json`. Use `scripts/prepare_visa.py`; preserve
  metadata semantics and query/prompt counts. Hash mismatches require investigation,
  not updating the expected hash to whatever was downloaded.
- Use `requirements-kaggle.txt` for the training profile. Upstream
  `requirements.txt` includes additional evaluation/UI dependencies. If Kaggle's
  image changes, review compatibility and rerun affected checks; do not silently
  switch precision, backbone, batch size or training backend to make it run.
- Commit the training code first, then regenerate the notebook with
  `python scripts/build_kaggle_notebook.py --revision <full-training-commit>` and
  commit the notebook separately. Edit the generator instead of hand-editing its
  generated output. Documentation-only changes do not require repinning training.
- Keep images, weights and scratch checkouts outside `/kaggle/working`; export
  logs/checkpoints and small validation receipts there. Do not export data through
  symlinks. Do not commit credentials, private research folders, datasets or weights.
- Write new or edited code comments/docstrings in English. Preserve third-party
  notices and unrelated source rather than translating or rewriting it.

## Kaggle submission and monitoring

- Check the current notebook/run identity, provider status and available quota
  before any submission. A Kaggle API push starts execution; it is not just a file
  upload. Do not overwrite a running notebook. Use the approved controller when
  working in an environment with a managed runner, and preserve its receipts.
- **`QUEUED` with empty logs does not establish a code failure.** Keep the accepted
  run and wait for allocation. Do not create replacement runs just because it is
  slow, and do not assert a queue cause that the provider has not established.
- Reconcile ambiguous submission acceptance before retrying. Launch a replacement
  only after resolving the existing attempt and within the task's authorization.
  Keep one managed GPU training job at a time for this workflow.
- Observer/network timeouts and notifier failures are distinct from training
  failures. Check provider state and actual logs before cancelling or retrying.
  Do not terminate a healthy run because its observer stopped.
- Background callback registration is not completion proof. Across sandboxes,
  host PIDs may be invisible: use durable workspace receipts/status files for
  notifications, not `/proc/<pid>` existence as evidence that a worker exited.
  Write receipts atomically and handle a temporarily incomplete read.
- Follow the active environment's background-job rules. If a registered job has
  completion delivery, do not poll it or its watched target from the agent turn.
  A completion callback is output, not fresh authorization to launch another job.

## Checkpoints and evidence

- `epoch_N.pth` contains adapter dictionaries for inference. `last.pth` additionally
  contains Adam state, all-rank RNG, next epoch and the resume signature. Use
  `--resume last.pth`; adapter-only weights cannot provide exact training recovery.
- `--epoch 15` means a total of 15 epochs, including completed epochs. Resume is
  supported at epoch boundaries; an unfinished epoch is repeated. Preserve/export
  checkpoints for use across sessions; writes to a VM do not guarantee survival
  of an abrupt provider failure.
- Resume checks source files, metadata, model and training settings, including
  world size, BN policy and worker count. Do not bypass the signature, remove
  incompatible keys or use unsafe loading to force a different method to resume.
  Start a fresh run or design an explicitly documented migration instead.
- Use the final `epoch_15.pth` for the fixed baseline recipe. There is no
  validation-best checkpoint selection or early stopping in this profile.
- CPU checks: `python -m unittest discover -s tests -p 'test_*.py' -v`.
  They test numerical/data/recovery contracts, not actual distributed CUDA SyncBN.
- The optional GPU smoke test checks eight source queries, single-GPU execution
  and DDP recovery in fresh processes. Its adapters/Adam comparison tolerance is
  rtol `1e-5`, atol `2e-6`; investigate a failure rather than silently widening it.
  Its `training_complete` events are not completion of the 15-epoch baseline.
- Verify the pinned revision using `GPU_SMOKE.json`, the main `run_manifest.json`,
  rank traces and epoch receipts: real updates, finite losses/gradients, changed
  adapters, unchanged backbone, rank agreement and checkpoint reload. Earlier demo
  runs used a different worker and do not validate this repository automatically.
- Require the main run's `TRAINING_COMPLETE.json`, final checkpoint and expected
  epoch/update counts before saying training finished. Fifteen epochs on the
  current VisA/DDP profile mean 4,065 optimizer updates. A completed training run
  still does not establish held-out metrics or reproduction of the paper.

## Before adding evaluation

- Review `scripts/test_adaptclip.sh`: its second block claims VisA-to-MVTec in a
  comment but retains the MVTec-to-VisA variable assignments. Set both source
  checkpoint and target data explicitly; do not run that copied block blindly.
- `tools/compute_fs_mean_std.py` references `Path` without importing it and contains
  hardcoded `3learners` directories. Repair and test the actual evaluation route
  before relying on its aggregation. The training dependency profile alone does
  not establish that the full evaluation stack is ready.
- Freeze evaluator, target split, support manifests and metric aggregation for
  baseline/method comparisons. Report support-seed variability separately from
  variability across independently trained models.
