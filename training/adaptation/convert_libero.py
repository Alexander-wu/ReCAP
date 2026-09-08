"""Convert LIBERO-90 HDF5 demonstrations into the iVideoGPT npz layout.

Target layout, identical to the CALVIN converter so both datasets feed the same
dataloader and the same RT-1 token geometry:

    <out>/<dataset>/<split>_eps_<8-digit>.npz
        image  uint8   [T, 256, 320, 3]
        action float32 [T, 13]

Every field name below was confirmed by reading the files (inspect_libero90.py),
not inferred from the published LIBERO format:

    /data                       attrs: num_demos, total, problem_info (holds
                                language_instruction), bddl_file_name, env_name
    /data/demo_<n>              attrs: num_samples, init_state, model_file
    /data/demo_<n>/actions      float64 [T, 7]
    /data/demo_<n>/obs/
        agentview_rgb           uint8 [T, 128, 128, 3]   <- third-person camera
        eye_in_hand_rgb         uint8 [T, 128, 128, 3]   <- wrist camera
        ee_pos [T,3]  ee_ori [T,3]  ee_states [T,6]
        gripper_states [T,2]  joint_states [T,7]

Decisions that matter:

1. **Episode boundaries are explicit.** Each `demo_<n>` is already one complete
   demonstration, so unlike CALVIN there is no need to slice long rollouts. Demo
   lengths measured across sampled files run 97-322 frames (mean ~120-213), which
   is a sensible NPZ size on its own. Demos are never concatenated.

2. **Camera.** `agentview_rgb` is used, matching CALVIN's `rgb_static` choice: it
   is the third-person view that shows both the arm and the manipulated objects.
   `eye_in_hand_rgb` is deliberately unused for now — mixing two cameras into one
   token stream would change the task definition, not just the data.

3. **Action space.** LIBERO actions are 7-D (measured ranges: dims 0-2 roughly
   +/-1 positional deltas, dims 3-5 small rotational deltas, dim 6 strictly
   {-1, +1} gripper). This is the same structure as CALVIN's `rel_actions`, so the
   identical RT-1 slot mapping applies: deltas to slots 4-6, rotation to 10-12,
   gripper to slot 3. Reusing the mapping keeps BLOCK = 93 and the pretrained
   action embeddings intact.

4. **Range table is fitted on training demos only.** Actions are binned into 256
   bins per dimension; fitting the range on validation data would leak. Note that
   LIBERO's rotational dims span far less than its positional dims, so a shared
   table would waste most of the vocabulary on the rotation channels — the audit
   printed at the end reports per-dimension bin coverage for exactly this reason.

5. **Split.** LIBERO-90 ships no train/validation split, so one is constructed by
   holding out whole tasks (files), not demos inside a task. Holding out demos
   from a task the model also trains on would measure memorisation of that scene
   rather than generalisation.

6. **Resolution.** 128x128 -> 256x320 by aspect-preserving scale plus
   edge-replicated padding, same as CALVIN: stretching would distort the arm, and
   black bars are high-frequency content the tokenizer must spend codes on.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

TARGET_H, TARGET_W = 256, 320
RT1_ACTION_DIM = 13
ACTION_BINS = 256

# LIBERO action index -> RT-1 action slot. Same rationale as the CALVIN
# converter: RT-1 dims 4-6 carry the dominant continuous deltas and dim 3 is
# bimodal +/-1, which matches LIBERO's binary gripper channel exactly.
ACTION_SLOTS = {
    0: 4,    # delta x        -> RT-1 dim 4
    1: 5,    # delta y        -> RT-1 dim 5
    2: 6,    # delta z        -> RT-1 dim 6
    3: 10,   # delta roll     -> RT-1 dim 10
    4: 11,   # delta pitch    -> RT-1 dim 11
    5: 12,   # delta yaw      -> RT-1 dim 12
    6: 3,    # gripper {-1,1} -> RT-1 dim 3
}

IMAGE_KEY = "agentview_rgb"


def letterbox(frame):
    """Scale to fit 256x320 preserving aspect ratio, pad by edge replication."""
    from PIL import Image
    height, width = frame.shape[:2]
    scale = min(TARGET_H / height, TARGET_W / width)
    new_h, new_w = max(1, round(height * scale)), max(1, round(width * scale))
    resized = np.asarray(
        Image.fromarray(frame).resize((new_w, new_h), Image.BILINEAR))
    pad_top = (TARGET_H - new_h) // 2
    pad_left = (TARGET_W - new_w) // 2
    return np.pad(
        resized,
        ((pad_top, TARGET_H - new_h - pad_top),
         (pad_left, TARGET_W - new_w - pad_left),
         (0, 0)),
        mode="edge")


def map_actions(actions):
    """7-DoF LIBERO -> 13-slot RT-1 vector, zeros in the unused slots."""
    mapped = np.zeros((len(actions), RT1_ACTION_DIM), dtype=np.float32)
    for source, target in ACTION_SLOTS.items():
        mapped[:, target] = actions[:, source]
    return mapped


def language_instruction(data_group):
    """Pull the task sentence out of /data.attrs['problem_info'] if present."""
    raw = data_group.attrs.get("problem_info")
    if raw is None:
        return None
    try:
        text = raw.decode() if isinstance(raw, bytes) else str(raw)
        return json.loads(text).get("language_instruction")
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def demo_names(data_group):
    """Numeric ordering, so demo_10 does not sort before demo_2."""
    names = list(data_group.keys())
    return sorted(names,
                  key=lambda n: int(n.split("_")[-1])
                  if n.split("_")[-1].isdigit() else 0)


def convert_file(path, output_dir, prefix, start_index, max_demos,
                 frame_stride, min_length, verbose):
    """Write one NPZ per demonstration. Returns written names and actions."""
    written, observed, records = [], [], []
    with h5py.File(path, "r") as handle:
        data = handle["data"]
        instruction = language_instruction(data)
        names = demo_names(data)
        if max_demos is not None:
            names = names[:max_demos]
        for offset, name in enumerate(names):
            output_index = start_index + offset
            target = output_dir / f"{prefix}_eps_{output_index:08d}.npz"
            group = data[name]
            if target.exists():
                with np.load(target) as payload:
                    observed.append(payload["action"])
                written.append(target.name)
                records.append({"output": target.name, "task_file": path.name,
                                "demo": name, "instruction": instruction})
                continue

            observations = group["obs"]
            if IMAGE_KEY not in observations:
                raise KeyError(
                    f"{path.name}/{name}/obs has no {IMAGE_KEY}; "
                    f"keys={list(observations.keys())}")
            images = observations[IMAGE_KEY][...]
            actions = np.asarray(group["actions"][...], dtype=np.float32)
            if len(images) != len(actions):
                raise ValueError(
                    f"{path.name}/{name}: {len(images)} frames vs "
                    f"{len(actions)} actions - camera and action streams must "
                    "stay aligned")
            if frame_stride > 1:
                images = images[::frame_stride]
                actions = actions[::frame_stride]
            if len(images) < min_length:
                if verbose:
                    print(f"  skip {target.name}: only {len(images)} frames")
                continue

            frames = np.stack([letterbox(frame) for frame in images])
            mapped = map_actions(actions)
            np.savez_compressed(target, image=frames.astype(np.uint8),
                                action=mapped)
            observed.append(mapped)
            written.append(target.name)
            records.append({"output": target.name, "task_file": path.name,
                            "demo": name, "instruction": instruction})
            if verbose and output_index % 50 == 0:
                print(f"  {target.name}  T={len(frames)}  {path.name}/{name}",
                      flush=True)
    return written, observed, records


def build_action_ranges(observed, margin=0.02):
    """Per-dimension [min, max] table for the 256-bin discretiser.

    Fitted on training demos only. Unused slots get a nominal [-1, 1] so a
    degenerate all-zero range never collapses every value into bin 0.
    """
    stacked = np.concatenate(observed, axis=0)
    ranges = np.zeros((RT1_ACTION_DIM, 2), dtype=np.float32)
    used = set(ACTION_SLOTS.values())
    for dim in range(RT1_ACTION_DIM):
        if dim not in used:
            ranges[dim] = (-1.0, 1.0)
            continue
        lo = float(stacked[:, dim].min())
        hi = float(stacked[:, dim].max())
        if hi - lo < 1e-6:
            lo, hi = lo - 1.0, hi + 1.0
        else:
            span = hi - lo
            lo, hi = lo - margin * span, hi + margin * span
        ranges[dim] = (lo, hi)
    return ranges, stacked


def audit_binning(ranges, stacked):
    """How much of the 256-bin vocabulary each mapped dimension actually uses."""
    rows = []
    for source, slot in sorted(ACTION_SLOTS.items(), key=lambda kv: kv[1]):
        lo, hi = ranges[slot]
        values = stacked[:, slot]
        scaled = np.clip((values - lo) / (hi - lo + 1e-8), 0, 1)
        bins = np.floor(scaled * ACTION_BINS).astype(int).clip(0, ACTION_BINS - 1)
        used = int(len(np.unique(bins)))
        rows.append({
            "libero_dim": source, "rt1_slot": slot,
            "range": [round(float(lo), 5), round(float(hi), 5)],
            "bins_used": used,
            "bin_coverage": round(used / ACTION_BINS, 4),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="LIBERO-90 -> iVideoGPT npz converter")
    parser.add_argument("--input",
                        default="./data/libero/datasets/libero_90")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-name", default="libero_90")
    parser.add_argument("--val-task-count", type=int, default=9,
                        help="whole tasks held out for validation; holding out "
                             "demos inside a training task would measure "
                             "memorisation of that scene instead")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-demos-per-task", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--min-length", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.input)
    files = sorted(root.glob("*.hdf5"))
    if not files:
        raise SystemExit(f"no hdf5 files under {root}")
    if args.max_tasks is not None:
        files = files[:args.max_tasks]
    if args.val_task_count >= len(files):
        raise SystemExit(
            f"--val-task-count {args.val_task_count} leaves no training tasks "
            f"out of {len(files)}")

    # Deterministic, evenly spread hold-out: take every k-th task so the
    # validation set covers multiple scene families (KITCHEN, STUDY, LIVING...)
    # instead of whichever alphabetical block happens to sit at the end.
    stride = max(1, len(files) // args.val_task_count)
    val_indices = {min(len(files) - 1, i * stride)
                   for i in range(args.val_task_count)}
    splits = {
        "train": [f for i, f in enumerate(files) if i not in val_indices],
        "val": [f for i, f in enumerate(files) if i in val_indices],
    }

    output_root = Path(args.output) / args.dataset_name
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "input": str(root), "output": str(output_root),
        "image_key": IMAGE_KEY,
        "action_slots": ACTION_SLOTS,
        "target_resolution": [TARGET_H, TARGET_W],
        "frame_stride": args.frame_stride,
        "split_policy": "held-out whole tasks, evenly strided",
        "splits": {},
    }

    training_observed = []
    for split, split_files in splits.items():
        print(f"[{split}] {len(split_files)} task files", flush=True)
        written_total, records = [], []
        observed_split = []
        cursor = 0
        for path in split_files:
            written, observed, rows = convert_file(
                path, output_root, split, cursor, args.max_demos_per_task,
                args.frame_stride, args.min_length, verbose=True)
            cursor += len(written)
            written_total += written
            observed_split += observed
            records += rows
        timesteps = int(sum(len(a) for a in observed_split))
        summary["splits"][split] = {
            "task_files": len(split_files),
            "episodes": len(written_total),
            "timesteps": timesteps,
            "task_names": [p.name for p in split_files],
        }
        (output_root / f"{split}_provenance.json").write_text(
            json.dumps(records, indent=2), encoding="utf-8")
        print(f"[{split}] wrote {len(written_total)} episodes, "
              f"{timesteps} timesteps", flush=True)
        if split == "train":
            training_observed = observed_split

    if not training_observed:
        raise SystemExit("no training episodes converted")

    import torch
    ranges, stacked = build_action_ranges(training_observed)
    torch.save(torch.from_numpy(ranges), output_root / "action_ranges.pth")
    audit = audit_binning(ranges, stacked)
    summary["action_ranges"] = ranges.tolist()
    summary["binning_audit"] = audit
    summary["action_ranges_fitted_on"] = "train split only"
    summary["training_timesteps_for_ranges"] = int(len(stacked))

    print("\nbin coverage per mapped dimension (fitted on train only):")
    for row in audit:
        flag = "  <-- LOW" if row["bin_coverage"] < 0.5 else ""
        print(f"  libero dim {row['libero_dim']} -> rt1 slot {row['rt1_slot']:>2}  "
              f"range {row['range']}  bins {row['bins_used']:>3}/{ACTION_BINS}  "
              f"({row['bin_coverage']:.1%}){flag}")

    (output_root / "manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
