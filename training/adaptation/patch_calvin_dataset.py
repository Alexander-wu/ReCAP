#!/usr/bin/env python
"""Teach the vendored iVideoGPT dataloader about CALVIN's explicit split.

The generic SimpleRoboticDatasetv2 branch ignores `train_eps_` / `val_eps_` and
instead holds out every 100th filename. That would mix CALVIN validation files
into training and leak training files into validation. The converter already
writes explicit prefixes, so make the loader respect them.

This patch also registers `calvin_d` and `calvin_d_smoke` as named one-dataset
mixes. It is intentionally tiny and idempotent. It patches only the cluster's
vendored copy, never the upstream source checkout.
"""

import argparse
from pathlib import Path


def patch(path, old, new, label):
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"{label}: already patched")
        return
    if old not in text:
        raise RuntimeError(f"{label}: anchor not found in {path}")
    path.write_text(text.replace(old, new), encoding="utf-8")
    print(f"{label}: patched")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    repo = Path(args.repo)

    loader = repo / "ivideogpt/data/simple_dataloader.py"
    old_loader = """        elif dataset_name == 'tfds_robonet':
            if train:
                parent_dir = yaml.load(open('DATASET.yaml'), Loader=yaml.FullLoader)['robonet_train_dataset']
            else:
                parent_dir = yaml.load(open('DATASET.yaml'), Loader=yaml.FullLoader)['robonet_test_dataset']
            self.filenames = glob.glob(os.path.join(parent_dir, '*.npz'))
            self.filenames.sort()
        else:
            self.filenames = glob.glob(os.path.join(parent_dir, dataset_name, '*.npz'))"""
    new_loader = """        elif dataset_name == 'tfds_robonet':
            if train:
                parent_dir = yaml.load(open('DATASET.yaml'), Loader=yaml.FullLoader)['robonet_train_dataset']
            else:
                parent_dir = yaml.load(open('DATASET.yaml'), Loader=yaml.FullLoader)['robonet_test_dataset']
            self.filenames = glob.glob(os.path.join(parent_dir, '*.npz'))
            self.filenames.sort()
        elif dataset_name.startswith('calvin_d'):
            # CALVIN ships an official train/validation split. The converter
            # preserves it as train_eps_* / val_eps_*; never apply the generic
            # every-100th-file split below or evaluation leaks into training.
            pattern = 'train_eps_*.npz' if train else 'val_eps_*.npz'
            self.filenames = glob.glob(os.path.join(parent_dir, dataset_name, pattern))
            self.filenames.sort()
        else:
            self.filenames = glob.glob(os.path.join(parent_dir, dataset_name, '*.npz'))"""
    patch(loader, old_loader, new_loader, "CALVIN explicit split")

    mixes = repo / "ivideogpt/data/dataset_mixes.py"
    old_mixes = '    "frac": [("fractal20220817_data", 1.0)],\n'
    new_mixes = ('    "frac": [("fractal20220817_data", 1.0)],\n'
                  '    "calvin_d": [("calvin_d", 1.0)],\n'
                  '    "calvin_d_smoke": [("calvin_d_smoke", 1.0)],\n')
    patch(mixes, old_mixes, new_mixes, "CALVIN named mixes")

    print("CALVIN dataset support ready")


if __name__ == "__main__":
    main()
