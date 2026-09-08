#!/usr/bin/env python3
"""Register converted BridgeData V2 datasets in the cluster iVideoGPT checkout."""

import argparse
from pathlib import Path


def patch_loader(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    target = (
        "        elif dataset_name.startswith('calvin_d') or "
        "dataset_name == 'libero_90' or dataset_name.startswith('bridge_v2'):\n"
    )
    if target not in text:
        candidates = [
            "        elif dataset_name.startswith('calvin_d') or dataset_name == 'libero_90':\n",
            "        elif dataset_name.startswith('calvin_d'):\n",
        ]
        for candidate in candidates:
            if candidate in text:
                text = text.replace(candidate, target, 1)
                break
        else:
            original = """        elif dataset_name == 'tfds_robonet':
            if train:
                parent_dir = yaml.load(open('DATASET.yaml'), Loader=yaml.FullLoader)['robonet_train_dataset']
            else:
                parent_dir = yaml.load(open('DATASET.yaml'), Loader=yaml.FullLoader)['robonet_test_dataset']
            self.filenames = glob.glob(os.path.join(parent_dir, '*.npz'))
            self.filenames.sort()
        else:
            self.filenames = glob.glob(os.path.join(parent_dir, dataset_name, '*.npz'))"""
            replacement = """        elif dataset_name == 'tfds_robonet':
            if train:
                parent_dir = yaml.load(open('DATASET.yaml'), Loader=yaml.FullLoader)['robonet_train_dataset']
            else:
                parent_dir = yaml.load(open('DATASET.yaml'), Loader=yaml.FullLoader)['robonet_test_dataset']
            self.filenames = glob.glob(os.path.join(parent_dir, '*.npz'))
            self.filenames.sort()
        elif dataset_name.startswith('calvin_d') or dataset_name == 'libero_90' or dataset_name.startswith('bridge_v2'):
            pattern = 'train_eps_*.npz' if train else 'val_eps_*.npz'
            self.filenames = glob.glob(os.path.join(parent_dir, dataset_name, pattern))
            self.filenames.sort()
        else:
            self.filenames = glob.glob(os.path.join(parent_dir, dataset_name, '*.npz'))"""
            if original not in text:
                raise RuntimeError(f"Bridge split anchor not found in {path}")
            text = text.replace(original, replacement, 1)
        path.write_text(text, encoding="utf-8")
    verified = path.read_text(encoding="utf-8")
    if target not in verified:
        raise RuntimeError(f"Bridge split patch verification failed: {path}")
    print("Bridge explicit train/val split registered", flush=True)


def patch_mixes(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    entries = [
        '    "bridge_v2": [("bridge_v2", 1.0)],\n',
        '    "bridge_v2_smoke": [("bridge_v2_smoke", 1.0)],\n',
    ]
    missing = [entry for entry in entries if entry not in text]
    if missing:
        anchors = [
            '    "libero_90": [("libero_90", 1.0)],\n',
            '    "calvin_d_smoke": [("calvin_d_smoke", 1.0)],\n',
            '    "frac": [("fractal20220817_data", 1.0)],\n',
        ]
        for anchor in anchors:
            if anchor in text:
                text = text.replace(anchor, anchor + "".join(missing), 1)
                break
        else:
            raise RuntimeError(f"Bridge mix anchor not found in {path}")
        path.write_text(text, encoding="utf-8")
    verified = path.read_text(encoding="utf-8")
    for entry in entries:
        if entry not in verified:
            raise RuntimeError(f"Bridge mix patch verification failed: {entry.strip()}")
    print("Bridge named mixes registered", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    repo = Path(args.repo)
    patch_loader(repo / "ivideogpt/data/simple_dataloader.py")
    patch_mixes(repo / "ivideogpt/data/dataset_mixes.py")


if __name__ == "__main__":
    main()
