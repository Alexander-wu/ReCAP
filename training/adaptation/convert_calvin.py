"""Convert CALVIN episodes into the npz layout the iVideoGPT dataloader expects.

Target layout, matching `oxe_data_converter.py` output and what
`SimpleRoboticDatasetv2` reads:

    <out>/calvin_<split>/<split>_eps_<8-digit>.npz
        image  uint8   [T, 256, 320, 3]
        action float32 [T, 13]

Two conversions matter and both were decided deliberately:

1. **Action space.** CALVIN gives 7-DoF (`rel_actions`: xyz, euler xyz, gripper).
   The RT-1 checkpoint's token layout is fixed at 13 action tokens per frame
   (BLOCK = 80 visual + 13 action = 93), so the 7 values are placed into 13
   slots and the rest are zero-filled. This keeps the pretrained weights,
   the token geometry, and every ReCAP inference constant untouched. The cost is
   6 constant dimensions the model learns to ignore.

   Slot choice is not arbitrary: RT-1's dims 4-6 are the dominant continuous
   deltas and 3 is bimodal +/-1 (gripper-like), so CALVIN's deltas go to 4-6,
   rotation to 10-12, and gripper to 3. This puts CALVIN's signal where the
   pretrained action embeddings already carry variance.

2. **Resolution.** CALVIN `rgb_static` is 200x200; the frozen compressive
   tokenizer was trained at 256x320. Frames are scaled to fit and padded, so
   aspect ratio is preserved and the arm is not horizontally stretched (which
   would corrupt the Arm-Score geometry the paper relies on). Padding is
   edge-replicated rather than black, because a hard black border creates
   high-frequency content the tokenizer must spend codes on.

CALVIN stores one .npz per timestep, with episode boundaries in
`ep_start_end_ids.npy`. This walks those boundaries so no episode is split.

A third output matters as much as the frames: `action_ranges.pth`. Actions are
discretised into 256 bins against a per-dimension range table, and the RT-1
table cannot be reused. RT-1's slot 11 spans [-5.50, +22.09] while CALVIN's
`rel_actions` are all in [-1, 1], so binning CALVIN with RT-1's table would put
that dimension into 20 of 256 bins - 7.8% of the vocabulary - and throw away
most of the rotation signal. Measured, not assumed: see the audit printed at
the end of a conversion run.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TARGET_H, TARGET_W = 256, 320
RT1_ACTION_DIM = 13
ACTION_BINS = 256

# CALVIN rel_actions index -> RT-1 action slot.
# RT-1 dims 4,5,6 carry the largest continuous ranges (world deltas) and dim 3
# is bimodal +/-1, so the mapping lands CALVIN's signal on slots whose
# pretrained embeddings already model that kind of variation.
ACTION_SLOTS = {
    0: 4,    # tcp dx      -> RT-1 dim 4
    1: 5,    # tcp dy      -> RT-1 dim 5
    2: 6,    # tcp dz      -> RT-1 dim 6
    3: 10,   # euler dx    -> RT-1 dim 10
    4: 11,   # euler dy    -> RT-1 dim 11
    5: 12,   # euler dz    -> RT-1 dim 12
    6: 3,    # gripper     -> RT-1 dim 3 (bimodal in RT-1 too)
}


def letterbox(frame):
    """Scale to fit 256x320 preserving aspect, pad by edge replication."""
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


def map_actions(rel_actions):
    """7-DoF CALVIN -> 13-slot RT-1 vector, zeros elsewhere."""
    count = len(rel_actions)
    mapped = np.zeros((count, RT1_ACTION_DIM), dtype=np.float32)
    for source, target in ACTION_SLOTS.items():
        mapped[:, target] = rel_actions[:, source]
    return mapped


def episode_bounds(root):
    path = root / "ep_start_end_ids.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing - point --input at a CALVIN split directory "
            "containing training/ and validation/")
    return np.load(path)


def output_windows(bounds, clip_length=None, min_clip_length=8):
    """Yield bounded, contiguous windows without crossing source rollouts.

    CALVIN's `ep_start_end_ids.npy` entries in the D split are long continuous
    rollouts (often tens of thousands of frames). Writing one compressed NPZ per
    entry would create multi-gigabyte files; every training sample would then
    decompress an entire file just to read eight frames. Fixed-length windows
    keep random access practical while preserving temporal continuity.
    """
    for source_index, (raw_start, raw_end) in enumerate(bounds):
        start, end = int(raw_start), int(raw_end)
        if clip_length is None:
            yield source_index, start, end
            continue
        cursor = start
        while cursor <= end:
            window_end = min(cursor + clip_length - 1, end)
            if window_end - cursor + 1 >= min_clip_length:
                yield source_index, cursor, window_end
            cursor = window_end + 1


def convert_split(split_dir, output_dir, prefix, action_key, max_episodes,
                  frame_stride, clip_length=None, min_clip_length=8,
                  verbose=True):
    bounds = episode_bounds(split_dir)
    if max_episodes is not None:
        bounds = bounds[:max_episodes]
    windows = list(output_windows(bounds, clip_length, min_clip_length))
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    observed = []
    provenance = []
    for output_index, (source_index, start, end) in enumerate(windows):
        target = output_dir / f"{prefix}_eps_{output_index:08d}.npz"
        if target.exists():
            written.append(target.name)
            with np.load(target) as payload:
                observed.append(payload["action"])
            provenance.append({"output": target.name,
                               "source_rollout": source_index,
                               "source_bounds": [start, end]})
            continue
        frames, actions = [], []
        for step in range(start, end + 1, frame_stride):
            step_file = split_dir / f"episode_{step:07d}.npz"
            if not step_file.exists():
                raise FileNotFoundError(
                    f"{step_file} is missing inside source rollout "
                    f"{source_index} [{start}, {end}]")
            with np.load(step_file) as payload:
                if "rgb_static" not in payload.files:
                    raise KeyError(
                        f"{step_file} has no rgb_static; keys={payload.files}")
                frames.append(letterbox(payload["rgb_static"]))
                if action_key not in payload.files:
                    raise KeyError(
                        f"{step_file} has no {action_key}; keys={payload.files}")
                actions.append(np.asarray(payload[action_key], dtype=np.float32))
        if len(frames) < min_clip_length:
            if verbose:
                print(f"  skip {target.name}: only {len(frames)} usable steps")
            continue
        mapped = map_actions(np.stack(actions))
        np.savez_compressed(
            target,
            image=np.stack(frames).astype(np.uint8),
            action=mapped)
        observed.append(mapped)
        written.append(target.name)
        provenance.append({"output": target.name,
                           "source_rollout": source_index,
                           "source_bounds": [start, end]})
        if verbose and output_index % 100 == 0:
            print(f"  {target.name}  T={len(frames)}  "
                  f"source_rollout={source_index}", flush=True)
    return written, observed, provenance, len(bounds)


def build_action_ranges(observed, margin=0.02):
    """Per-dimension [min, max] table for the 256-bin discretiser.

    The RT-1 table cannot be reused: its slot 11 spans [-5.50, +22.09] against
    CALVIN's [-1, 1], which would compress that dimension into 20 of 256 bins.
    Ranges are taken from the data actually written, with a small margin so a
    value at the extreme does not land outside the last bin.

    Unused slots get a nominal [-1, 1]. A degenerate (all-zero) range would make
    the discretiser divide by ~0; the +1e-8 in the training code guards it, but
    every such value would still collapse to bin 0, so being explicit is safer.
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
    """How much of the 256-bin vocabulary each mapped dimension actually uses.

    This is the check that caught the RT-1 range-table problem. A dimension
    using a small fraction of the bins is losing resolution, and the fix is the
    range table rather than anything downstream.
    """
    rows = []
    for source, slot in sorted(ACTION_SLOTS.items(), key=lambda kv: kv[1]):
        lo, hi = ranges[slot]
        values = stacked[:, slot]
        scaled = np.clip((values - lo) / (hi - lo + 1e-8), 0, 1)
        bins = np.floor(scaled * ACTION_BINS).astype(int).clip(0, ACTION_BINS - 1)
        rows.append({
            "calvin_dim": source, "rt1_slot": slot,
            "range": [round(float(lo), 4), round(float(hi), 4)],
            "bins_used": int(len(np.unique(bins))),
            "bin_coverage": round(len(np.unique(bins)) / ACTION_BINS, 4),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="CALVIN -> iVideoGPT npz converter")
    parser.add_argument("--input", required=True,
                        help="CALVIN split root, e.g. .../task_ABC_D")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-name", default="calvin_abc_d")
    parser.add_argument("--action-key", default="rel_actions",
                        choices=["rel_actions", "actions"],
                        help="rel_actions is normalised to [-1,1] and is what "
                             "the CALVIN env expects back; absolute actions are "
                             "in world coordinates and do not transfer")
    parser.add_argument("--max-episodes-per-split", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1,
                        help=">1 subsamples time, trading temporal resolution "
                             "for disk and training throughput")
    parser.add_argument("--clip-length", type=int, default=None,
                        help="split long CALVIN rollouts into bounded contiguous "
                             "clips; 128 is recommended for the full D split")
    parser.add_argument("--min-clip-length", type=int, default=8,
                        help="drop a final remainder shorter than this many frames")
    args = parser.parse_args()

    root = Path(args.input)
    output_root = Path(args.output) / args.dataset_name
    summary = {"input": str(root), "output": str(output_root),
               "action_key": args.action_key, "frame_stride": args.frame_stride,
               "clip_length": args.clip_length,
               "min_clip_length": args.min_clip_length,
               "action_slots": ACTION_SLOTS,
               "target_resolution": [TARGET_H, TARGET_W], "splits": {}}

    training_observed = []
    validation_timesteps = 0
    for split, prefix in (("training", "train"), ("validation", "val")):
        split_dir = root / split
        if not split_dir.is_dir():
            print(f"missing split: {split_dir}", file=sys.stderr)
            continue
        print(f"[{split}] converting from {split_dir}")
        names, observed, provenance, source_rollouts = convert_split(
            split_dir, output_root, prefix, args.action_key,
            args.max_episodes_per_split, args.frame_stride,
            args.clip_length, args.min_clip_length)
        split_timesteps = int(sum(len(array) for array in observed))
        summary["splits"][split] = {
            "source_rollouts": source_rollouts,
            "output_clips": len(names),
            "timesteps": split_timesteps,
            "provenance": provenance}
        if split == "training":
            training_observed.extend(observed)
        else:
            validation_timesteps += split_timesteps
        print(f"[{split}] wrote {len(names)} episodes")

    if not training_observed:
        raise SystemExit(
            "no training episodes converted; action ranges must be fitted on "
            "the training split only")

    # Fit ranges on training only. Letting validation extremes set quantisation
    # bins is a subtle form of evaluation leakage, and also makes a result depend
    # on how much validation data happened to be converted.
    import torch
    ranges, stacked = build_action_ranges(training_observed)
    ranges_path = output_root / "action_ranges.pth"
    torch.save(torch.from_numpy(ranges), ranges_path)
    audit = audit_binning(ranges, stacked)
    summary["action_ranges"] = ranges.tolist()
    summary["action_ranges_fitted_on"] = "training only"
    summary["binning_audit"] = audit
    summary["training_timesteps_for_ranges"] = int(len(stacked))
    summary["validation_timesteps"] = validation_timesteps
    summary["total_timesteps"] = int(len(stacked) + validation_timesteps)

    print(f"\nWrote {ranges_path}")
    print(f"{'CALVIN dim':<12s}{'RT-1 slot':<11s}{'range':<26s}"
          f"{'bins used':<12s}coverage")
    for row in audit:
        lo, hi = row["range"]
        print(f"  {row['calvin_dim']:<10d}{row['rt1_slot']:<11d}"
              f"[{lo:+.3f},{hi:+.3f}]{'':<8s}"
              f"{row['bins_used']:<12d}{row['bin_coverage']:.1%}")
    low = [r for r in audit if r["bin_coverage"] < 0.5]
    if low:
        print(f"\nWARNING: {len(low)} dimension(s) use under half the bins; "
              "check whether those actions are near-constant in this split.")

    (output_root / "manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
