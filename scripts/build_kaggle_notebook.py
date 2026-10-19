"""Generate the compact notebook with an explicit immutable code revision."""

import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9a-f]{40}', args.revision):
        raise ValueError('Use a complete Git commit SHA')
    cells = []
    def markdown(source):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': source.strip() + '\n'})
    def code(source):
        cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                      'outputs': [], 'source': source.strip() + '\n'})
    markdown('''
# AdaptCLIP-Repro: train on VisA

Select **GPU T4 ×2**, enable **Internet**, and **Run All**. No secrets or attached
datasets are required. Public assets and training source are pinned and verified.

The baseline trains fresh adapters for **15 epochs**, seed 10, FP32,
**batch 4/GPU = global batch 8**, DDP + SyncBatchNorm. This differs from the authors'
single-GPU BatchNorm recipe; benchmark evaluation is outside this notebook.
The optional smoke test uses separate outputs and does not initialize the baseline.
''')
    code('''
import hashlib, json, os, shutil, subprocess, sys, urllib.request
from pathlib import Path

REPO = 'https://github.com/Purin1410/AdaptCLIP-Repro.git'
REVISION = 'REVISION_VALUE'
DOWNLOADS = Path('/kaggle/temp/adaptclip/downloads')
DATA = Path('/kaggle/temp/adaptclip/data/visa')
CHECKOUT = Path('/kaggle/temp/AdaptCLIP-Repro')
OUTPUT = Path('/kaggle/working/adaptclip_visa_seed10')
RUN_SMOKE_TESTS = True  # Optional: validates training and fresh-process resume first.
DOWNLOADS.mkdir(parents=True, exist_ok=True)
subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv'], check=True)
print('Training source:', REVISION)
'''.replace('REVISION_VALUE', args.revision))
    markdown('## 1. Download official VisA, split CSV and pretrained CLIP')
    code('''
manifest_url = f'https://raw.githubusercontent.com/Purin1410/AdaptCLIP-Repro/{REVISION}/assets.json'
with urllib.request.urlopen(manifest_url, timeout=60) as response:
    assets = json.load(response)['assets']

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

for name, asset in assets.items():
    path = DOWNLOADS / name
    if not path.is_file() or sha256(path) != asset['sha256']:
        print('Downloading', name, flush=True)
        temporary = path.with_suffix(path.suffix + '.partial')
        with urllib.request.urlopen(asset['url'], timeout=120) as source, temporary.open('wb') as target:
            shutil.copyfileobj(source, target, 2 ** 20)
        assert sha256(temporary) == asset['sha256'], f'Hash mismatch: {name}'
        temporary.replace(path)
    print('Verified:', name, flush=True)
''')
    markdown('## 2. Clone the pinned training code and install dependencies')
    code('''
if not CHECKOUT.exists():
    subprocess.run(['git', 'clone', REPO, str(CHECKOUT)], check=True)
subprocess.run(['git', '-C', str(CHECKOUT), 'checkout', '--detach', REVISION], check=True)
assert subprocess.check_output(['git', '-C', str(CHECKOUT), 'rev-parse', 'HEAD'], text=True).strip() == REVISION
assert not subprocess.check_output(['git', '-C', str(CHECKOUT), 'status', '--porcelain'], text=True).strip(), 'Use a clean checkout'
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-input', '-r', str(CHECKOUT / 'requirements-kaggle.txt')], check=True)
os.chdir(CHECKOUT)
''')
    markdown('## 3. Prepare native VisA data and metadata')
    code('''
subprocess.run([sys.executable, 'scripts/prepare_visa.py',
                '--downloads', str(DOWNLOADS), '--data', str(DATA)], check=True)
''')
    markdown('''
## 4. Optional bounded GPU and resume checks

Eight source queries, separate scratch outputs: one single-GPU update and a DDP
run compared with a fresh-process resumed run. This takes a few minutes.
Set `RUN_SMOKE_TESTS = False` to skip it on later runs of an already verified revision.
''')
    code('''
if RUN_SMOKE_TESTS:
    subprocess.run([sys.executable, 'tests/gpu_smoke.py', '--data', str(DATA),
                    '--pretrained_dir', str(DOWNLOADS),
                    '--scratch', '/kaggle/temp/adaptclip/smoke',
                    '--out', '/kaggle/working/runtime_validation'], check=True)
''')
    markdown('''
## 5. Train the baseline

Fresh adapters, 15 epochs, 271 synchronized updates per epoch. The backbone remains
frozen; all three adapters train. Checkpoints/logs are saved under `OUTPUT`.
`epoch_15.pth` is the final inference checkpoint; `last.pth` also stores optimizer/RNG.

For epoch-boundary recovery, keep the same flags and append
`--resume /kaggle/input/YOUR_CHECKPOINT/last.pth`. `--epoch 15` stays the total target.
Choose a new output directory for a fresh run; existing outputs are never silently replaced.
''')
    code('''
!python -u train.py \\
    --dataset visa --train_data_path {DATA} \\
    --pretrained_dir {DOWNLOADS} --save_path {OUTPUT} \\
    --devices 2 --sync_bn --batch_size 4 \\
    --epoch 15 --learning_rate 0.001 --seed 10 \\
    --image_size 518 --features_list 6 12 18 24 \\
    --n_ctx 12 --vl_reduction 4 --pq_mid_dim 128 \\
    --visual_learner --textual_learner --pq_learner --pq_context \\
    --save_freq 1 --print_freq 1
assert _exit_code == 0, 'Training failed; inspect the error and saved logs above.'
''')
    notebook = {'cells': cells, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12'},
        'accelerator': 'GPU', 'adaptclip_training_commit': args.revision},
        'nbformat': 4, 'nbformat_minor': 4}
    path = ROOT / 'notebooks/train_visa_kaggle.ipynb'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + '\n')
    print(path)


if __name__ == '__main__':
    main()
