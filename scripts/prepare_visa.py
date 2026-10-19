"""Verify official assets and prepare native VisA metadata, added 2026-10-19."""

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1',
           'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
META_SHA256 = '2d3c5f96d208faafe863c8dfbe9cf0d33d2aa17b352bfbdd0db7ec445be231d5'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def make_meta(csv_path):
    with Path(csv_path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    meta = {split: {name: [] for name in CLASSES} for split in ['train', 'test']}
    for row in rows:
        abnormal = row['label'] == 'anomaly'
        if row['label'] not in ['normal', 'anomaly']:
            raise ValueError('Unexpected VisA label')
        for key in ['image', 'mask']:
            if row[key] and (PurePosixPath(row[key]).is_absolute() or '..' in PurePosixPath(row[key]).parts):
                raise ValueError('Unsafe VisA metadata path')
        meta[row['split']][row['object']].append({
            'img_path': row['image'], 'mask_path': row['mask'] if abnormal else '',
            'cls_name': row['object'], 'specie_name': '', 'anomaly': int(abnormal)})
    if len(rows) != 10821 or sum(map(len, meta['train'].values())) != 8659 or sum(map(len, meta['test'].values())) != 2162:
        raise ValueError('Unexpected official VisA split counts')
    return meta


def safe_extract(archive, destination):
    with tarfile.open(archive, 'r:') as stream:
        for member in stream:
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Unsafe archive member path')
            target = destination.joinpath(*path.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with stream.extractfile(member) as source, target.open('xb') as output:
                    shutil.copyfileobj(source, output, 2 ** 20)
            else:
                raise ValueError('Only regular files and directories are accepted')


def verify_paths(root, meta):
    for split in meta.values():
        for rows in split.values():
            for row in rows:
                if not (root / row['img_path']).is_file():
                    raise FileNotFoundError(row['img_path'])
                if row['anomaly'] and not (root / row['mask_path']).is_file():
                    raise FileNotFoundError(row['mask_path'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--downloads', required=True)
    parser.add_argument('--data', required=True)
    args = parser.parse_args()
    downloads, root = Path(args.downloads).resolve(), Path(args.data).resolve()
    manifest = json.loads((ROOT / 'assets.json').read_text())
    for name, asset in manifest['assets'].items():
        print(f'Checking {name}', flush=True)
        if sha256(downloads / name) != asset['sha256']:
            raise ValueError(f'Asset SHA256 mismatch: {name}')
    meta = make_meta(downloads / '1cls.csv')
    encoded = (json.dumps(meta, indent=2) + '\n').encode()
    if hashlib.sha256(encoded).hexdigest() != META_SHA256:
        raise ValueError('Metadata differs from the verified official split')
    if root.exists():
        receipt = root / 'PREPARATION.json'
        if not receipt.is_file() or sha256(root / 'meta.json') != META_SHA256:
            raise FileExistsError('Existing data directory is not a verified preparation; preserve it and select a new path')
    else:
        temporary = root.with_name(root.name + '.extracting')
        temporary.mkdir(parents=True, exist_ok=False)
        print('Extracting official VisA archive', flush=True)
        safe_extract(downloads / 'VisA_20220922.tar', temporary)
        content = next(path for path in [temporary, temporary / 'VisA'] if (path / 'candle/Data/Images').is_dir())
        content.rename(root)
        if temporary.exists():
            temporary.rmdir()
        (root / 'meta.json').write_bytes(encoded)
        (root / 'split_csv').mkdir(exist_ok=True)
        shutil.copyfile(downloads / '1cls.csv', root / 'split_csv/1cls.csv')
    verify_paths(root, meta)
    receipt = {'pass': True, 'assets': manifest['assets'], 'meta_sha256': META_SHA256,
               'classes': CLASSES, 'source_query_images': 2162, 'source_prompt_images': 8659,
               'source_query_anomalies': sum(row['anomaly'] for rows in meta['test'].values() for row in rows),
               'query_split': 'test', 'normal_prompt_split': 'train', 'target_data_access': False}
    (root / 'PREPARATION.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'event': 'data_ready', **receipt}), flush=True)


if __name__ == '__main__':
    main()
