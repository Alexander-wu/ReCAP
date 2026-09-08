#!/usr/bin/env python3
"""Register LIBERO/Bridge and preserve explicit converted train/val splits.

The patch is idempotent on a clean upstream checkout or one already extended
with CALVIN, LIBERO, or BridgeData V2.
"""

import argparse
from pathlib import Path


def patch_loader(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    target = (
        "        elif dataset_name.startswith('calvin_d') or "
        "dataset_name == 'libero_90' or dataset_name.startswith('bridge_v2'):\n"
    )
    if target in text:
        print("LIBERO/Bridge explicit split: already patched", flush=True)
        return

    libero_only = "        elif dataset_name.startswith('calvin_d') or dataset_name == 'libero_90':\n"
    if libero_only in text:
        path.write_text(text.replace(libero_only, target, 1), encoding="utf-8")
        print("Bridge explicit split: extended LIBERO patch", flush=True)
        return

    calvin = "        elif dataset_name.startswith('calvin_d'):\n"
    if calvin in text:
        path.write_text(text.replace(calvin, target, 1), encoding="utf-8")
        print("LIBERO/Bridge explicit split: extended CALVIN patch", flush=True)
        return

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
            # Converted datasets preserve source train/test splits with explicit
            # prefixes. Never apply the generic every-100th-file split below.
            pattern = 'train_eps_*.npz' if train else 'val_eps_*.npz'
            self.filenames = glob.glob(os.path.join(parent_dir, dataset_name, pattern))
            self.filenames.sort()
        else:
            self.filenames = glob.glob(os.path.join(parent_dir, dataset_name, '*.npz'))"""
    if original not in text:
        raise RuntimeError(f"LIBERO explicit split: anchor not found in {path}")
    path.write_text(text.replace(original, replacement, 1), encoding="utf-8")
    print("LIBERO explicit split: patched clean checkout", flush=True)


def patch_mixes(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    libero = '    "libero_90": [("libero_90", 1.0)],\n'
    bridge = (
        '    "bridge_v2": [("bridge_v2", 1.0)],\n'
        '    "bridge_v2_smoke": [("bridge_v2_smoke", 1.0)],\n'
    )
    if bridge[0] in text and bridge[1] in text:
        print("LIBERO/Bridge named mixes: already patched", flush=True)
        return
    if libero in text:
        insertion = libero + "".join(item for item in bridge if item not in text)
        path.write_text(text.replace(libero, insertion, 1), encoding="utf-8")
        print("Bridge named mixes: extended LIBERO patch", flush=True)
        return

    calvin = '    "calvin_d_smoke": [("calvin_d_smoke", 1.0)],\n'
    if calvin in text:
        path.write_text(text.replace(calvin, calvin + libero, 1), encoding="utf-8")
        print("LIBERO named mix: extended CALVIN patch", flush=True)
        return

    frac = '    "frac": [("fractal20220817_data", 1.0)],\n'
    replacement = (
        frac
        + '    "calvin_d": [("calvin_d", 1.0)],\n'
        + '    "calvin_d_smoke": [("calvin_d_smoke", 1.0)],\n'
        + libero
        + "".join(bridge)
    )
    if frac not in text:
        raise RuntimeError(f"LIBERO named mix: anchor not found in {path}")
    path.write_text(text.replace(frac, replacement, 1), encoding="utf-8")
    print("LIBERO named mix: patched clean checkout", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    repo = Path(args.repo)
    patch_loader(repo / "ivideogpt/data/simple_dataloader.py")
    patch_mixes(repo / "ivideogpt/data/dataset_mixes.py")
    print("LIBERO dataset support ready", flush=True)


if __name__ == "__main__":
    main()
