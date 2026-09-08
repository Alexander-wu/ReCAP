#!/usr/bin/env python3
"""Parallel, resumable BridgeData V2 conversion using deterministic shard offsets."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import re
from pathlib import Path

import numpy as np

import convert_bridge as core


_WORKER = {}


def init_worker(output_dir, prefix, min_length, verify_crc, compress):
    _WORKER.update(
        output_dir=Path(output_dir),
        prefix=prefix,
        min_length=min_length,
        verify_crc=verify_crc,
        compress=compress,
    )


def convert_shard(job):
    shard, offset = job
    records = []
    minimum = np.full(core.RT1_ACTION_DIM, np.inf, dtype=np.float64)
    maximum = np.full(core.RT1_ACTION_DIM, -np.inf, dtype=np.float64)
    timesteps = 0
    for record_index, payload in core.read_tfrecords(
        shard, verify_crc=_WORKER["verify_crc"]
    ):
        output_index = offset + record_index
        name = f"{_WORKER['prefix']}_eps_{output_index:08d}.npz"
        target = _WORKER["output_dir"] / name
        if target.exists():
            stored = core.validate_npz(target, shard.name, record_index)
            actions = stored["actions"]
            frames_count = stored["frames"]
            instruction = stored["instruction"]
        else:
            frames, actions, instruction, state = core.build_episode(payload, drop_first=False)
            if len(frames) < _WORKER["min_length"]:
                records.append(
                    {
                        "source_shard": shard.name,
                        "source_record": record_index,
                        "output": None,
                        "skipped": f"length {len(frames)} < {_WORKER['min_length']}",
                    }
                )
                continue
            arrays = {
                "image": frames.astype(np.uint8),
                "action": actions.astype(np.float32),
                "instruction": np.asarray(instruction),
                "source_shard": np.asarray(shard.name),
                "source_record": np.asarray(record_index, dtype=np.int64),
            }
            if state is not None:
                arrays["state"] = state.astype(np.float32)
            core.atomic_save_npz(target, compress=_WORKER["compress"], **arrays)
            core.validate_npz(target, shard.name, record_index)
            frames_count = int(len(frames))
        minimum = np.minimum(minimum, actions.min(axis=0))
        maximum = np.maximum(maximum, actions.max(axis=0))
        timesteps += frames_count
        records.append(
            {
                "source_shard": shard.name,
                "source_record": record_index,
                "output": name,
                "frames": frames_count,
                "instruction": instruction,
            }
        )
    return {
        "shard": shard.name,
        "episodes": sum(row.get("output") is not None for row in records),
        "timesteps": timesteps,
        "minimum": minimum.tolist(),
        "maximum": maximum.tolist(),
        "records": records,
    }


def shard_offsets(dataset_info, split):
    info = json.loads(Path(dataset_info).read_text(encoding="utf-8"))
    split_info = next(item for item in info["splits"] if item["name"] == split)
    lengths = [int(value) for value in split_info["shardLengths"]]
    offsets = []
    cursor = 0
    for length in lengths:
        offsets.append(cursor)
        cursor += length
    return lengths, offsets


def finish_ranges(results, output_dir, margin=0.02):
    minima = np.asarray([row["minimum"] for row in results], dtype=np.float64)
    maxima = np.asarray([row["maximum"] for row in results], dtype=np.float64)
    lo_all = np.min(minima, axis=0)
    hi_all = np.max(maxima, axis=0)
    ranges = np.zeros((core.RT1_ACTION_DIM, 2), dtype=np.float32)
    for dim in range(core.RT1_ACTION_DIM):
        if dim not in core.USED_SLOTS:
            ranges[dim] = (-1.0, 1.0)
            continue
        lo, hi = float(lo_all[dim]), float(hi_all[dim])
        if hi - lo < 1e-6:
            lo, hi = lo - 1.0, hi + 1.0
        else:
            span = hi - lo
            lo, hi = lo - margin * span, hi + margin * span
        ranges[dim] = (lo, hi)
    import torch

    torch.save(torch.from_numpy(ranges), output_dir / "action_ranges.pth")
    return ranges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-name", default="bridge_v2")
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--min-length", type=int, default=8)
    parser.add_argument("--no-crc", action="store_true")
    parser.add_argument("--uncompressed", action="store_true")
    args = parser.parse_args()

    source = Path(args.input)
    output_dir = Path(args.output) / args.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "train" if args.split == "train" else "val"
    exact_shard = re.compile(
        rf"^bridge-{re.escape(args.split)}\.tfrecord-\d{{5}}-of-\d{{5}}$"
    )
    shards = [
        path
        for path in sorted(source.glob(f"bridge-{args.split}.tfrecord-*"))
        if exact_shard.fullmatch(path.name)
    ]
    lengths, offsets = shard_offsets(source / "dataset_info.json", args.split)
    if len(shards) != len(lengths):
        raise SystemExit(
            f"metadata lists {len(lengths)} {args.split} shards, found {len(shards)}"
        )
    total_shards = len(shards)
    if args.max_shards is not None:
        shards = shards[: args.max_shards]
        offsets = offsets[: args.max_shards]

    jobs = list(zip(shards, offsets))
    results = []
    # Threads avoid fork-related cgroup kills observed on the storage server;
    # JPEG decode and NumPy compression release the GIL for most heavy work.
    init_worker(
        output_dir,
        prefix,
        args.min_length,
        not args.no_crc,
        compress=not args.uncompressed,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for done, result in enumerate(pool.map(convert_shard, jobs), 1):
            results.append(result)
            print(
                "BRIDGE_SHARD_DONE="
                + json.dumps(
                    {
                        "done": done,
                        "total": len(jobs),
                        "shard": result["shard"],
                        "episodes": result["episodes"],
                        "timesteps": result["timesteps"],
                    }
                ),
                flush=True,
            )

    results.sort(key=lambda row: row["shard"])
    manifest = {
        "source": str(source),
        "dataset": args.dataset_name,
        "source_split": args.split,
        "output_prefix": prefix,
        "workers": args.workers,
        "selected_shards": len(shards),
        "total_shards": total_shards,
        "target_resolution": [core.TARGET_H, core.TARGET_W],
        "compression": "none" if args.uncompressed else "zip-deflate",
        "protocol": "existing RLVR-World_bridge 7D + five zero pads + terminate at slot 12",
        "action_slots": core.ACTION_SLOTS,
        "episodes": int(sum(row["episodes"] for row in results)),
        "timesteps": int(sum(row["timesteps"] for row in results)),
        "shards": [
            {
                "name": row["shard"],
                "episodes": row["episodes"],
                "timesteps": row["timesteps"],
            }
            for row in results
        ],
        "records": [record for row in results for record in row["records"]],
    }
    if args.split == "train":
        ranges = finish_ranges(results, output_dir)
        manifest["action_ranges"] = ranges.tolist()
        manifest["action_ranges_fitted_on"] = (
            "full Bridge train split"
            if len(shards) == total_shards
            else f"Bridge train subset: {len(shards)}/{total_shards} shards"
        )

    path = output_dir / f"{prefix}_conversion_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.split == "train":
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(
        "BRIDGE_PARALLEL_RESULT="
        + json.dumps(
            {
                "split": args.split,
                "episodes": manifest["episodes"],
                "timesteps": manifest["timesteps"],
                "output": str(output_dir),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
