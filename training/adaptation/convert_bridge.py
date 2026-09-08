#!/usr/bin/env python3
"""Convert BridgeData V2 TFDS/RLDS shards to the existing iVideoGPT NPZ layout.

The first Bridge adaptation deliberately keeps the RT-1 token geometry used by
CALVIN and LIBERO:

    image  uint8   [T, 256, 320, 3]
    action float32 [T, 13]

Bridge exposes world_vector[3], rotation_delta[3], open_gripper, and a scalar
terminate_episode.  This converter deliberately preserves the mapping used by
the project's existing `RLVR-World_bridge/bridge_data_converter.py` rather than
inventing a new semantic slot layout:

    slots 0:3   world_vector
    slots 3:6   rotation_delta
    slot  6     open_gripper as 0/1
    slots 7:12  zero padding
    slot  12    terminate_episode

This is the previously agreed "pad Bridge actions with zeros to 13D" protocol.
It preserves BLOCK=80+13=93 and makes the released RT-1 tokenizer/world-model
weights directly reusable.  The converter keeps every source step by default,
matching the existing Bridge converter; dropping the artificial first step is
available only as an explicit option.

No TensorFlow dependency is required.  The parser reads TFRecord framing and
TensorFlow Example protobuf fields directly, which makes the converter usable on
the storage server without changing its Python environment.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import struct
from pathlib import Path

import numpy as np
from PIL import Image

TARGET_H, TARGET_W = 256, 320
RT1_ACTION_DIM = 13
ACTION_BINS = 256
FEATURE_TYPE_BYTES = 1
FEATURE_TYPE_FLOAT = 2
FEATURE_TYPE_INT64 = 3
MASK_DELTA = 0xA282EAD8
_CRC32C_POLY = 0x82F63B78

ACTION_SLOTS = {
    "world_vector": [0, 1, 2],
    "rotation_delta": [3, 4, 5],
    "gripper": 6,
    "terminate_episode": 12,
}
USED_SLOTS = {0, 1, 2, 3, 4, 5, 6, 12}


def _build_crc32c_table():
    table = []
    for index in range(256):
        crc = index
        for _ in range(8):
            crc = (crc >> 1) ^ (_CRC32C_POLY if crc & 1 else 0)
        table.append(crc)
    return tuple(table)


_CRC32C_TABLE = _build_crc32c_table()


def crc32c(data):
    crc = 0xFFFFFFFF
    for byte in data:
        crc = _CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def _mask_crc(crc):
    return (((crc >> 15) | (crc << 17)) + MASK_DELTA) & 0xFFFFFFFF


def read_tfrecords(path, verify_crc=True):
    """Yield `(record_index, payload)` and fail loudly on truncated shards."""
    with Path(path).open("rb") as handle:
        index = 0
        while True:
            header = handle.read(12)
            if not header:
                return
            if len(header) != 12:
                raise ValueError(f"{path}: truncated TFRecord header at record {index}")
            (length,) = struct.unpack("<Q", header[:8])
            (length_crc,) = struct.unpack("<I", header[8:12])
            if verify_crc and _mask_crc(crc32c(header[:8])) != length_crc:
                raise ValueError(f"{path}: corrupt length CRC at record {index}")
            payload = handle.read(length)
            if len(payload) != length:
                raise ValueError(f"{path}: truncated payload at record {index}")
            payload_crc_raw = handle.read(4)
            if len(payload_crc_raw) != 4:
                raise ValueError(f"{path}: truncated payload CRC at record {index}")
            if verify_crc:
                (payload_crc,) = struct.unpack("<I", payload_crc_raw)
                if _mask_crc(crc32c(payload)) != payload_crc:
                    raise ValueError(f"{path}: corrupt payload CRC at record {index}")
            yield index, payload
            index += 1


def _read_varint(buffer, offset):
    result = 0
    shift = 0
    while True:
        byte = buffer[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, offset
        shift += 7


def _iter_fields(buffer, start=0, end=None):
    if end is None:
        end = len(buffer)
    offset = start
    while offset < end:
        key, offset = _read_varint(buffer, offset)
        field_number = key >> 3
        wire_type = key & 0x07
        if wire_type == 0:
            value, offset = _read_varint(buffer, offset)
        elif wire_type == 1:
            value = buffer[offset : offset + 8]
            offset += 8
        elif wire_type == 2:
            length, offset = _read_varint(buffer, offset)
            value = buffer[offset : offset + length]
            offset += length
        elif wire_type == 5:
            value = buffer[offset : offset + 4]
            offset += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type}")
        yield field_number, wire_type, value


def _parse_feature(feature_bytes):
    for field_number, _wire_type, value in _iter_fields(feature_bytes):
        if field_number == 1:
            return FEATURE_TYPE_BYTES, [v for fn, _wt, v in _iter_fields(value) if fn == 1]
        if field_number == 2:
            floats = []
            for fn, wt, v in _iter_fields(value):
                if fn == 1 and wt in (2, 5):
                    floats.extend(np.frombuffer(v, dtype="<f4").tolist())
            return FEATURE_TYPE_FLOAT, floats
        if field_number == 3:
            ints = []
            for fn, wt, v in _iter_fields(value):
                if fn != 1:
                    continue
                if wt == 0:
                    ints.append(v)
                elif wt == 2:
                    offset = 0
                    while offset < len(v):
                        item, offset = _read_varint(v, offset)
                        ints.append(item)
            return FEATURE_TYPE_INT64, ints
    return None, []


def _parse_feature_map(features_bytes):
    result = {}
    for field_number, _wire_type, value in _iter_fields(features_bytes):
        if field_number != 1:
            continue
        key = None
        feature = None
        for fn, _wt, v in _iter_fields(value):
            if fn == 1:
                key = v.decode("utf-8")
            elif fn == 2:
                feature = v
        if key is not None and feature is not None:
            result[key] = _parse_feature(feature)
    return result


def _parse_feature_list_map(feature_lists_bytes):
    result = {}
    for field_number, _wire_type, value in _iter_fields(feature_lists_bytes):
        if field_number != 1:
            continue
        key = None
        feature_list = None
        for fn, _wt, v in _iter_fields(value):
            if fn == 1:
                key = v.decode("utf-8")
            elif fn == 2:
                feature_list = v
        if key is not None and feature_list is not None:
            result[key] = [
                _parse_feature(v)
                for fn, _wt, v in _iter_fields(feature_list)
                if fn == 1
            ]
    return result


def parse_example(payload):
    context = {}
    sequence = {}
    for field_number, _wire_type, value in _iter_fields(payload):
        if field_number == 1:
            context.update(_parse_feature_map(value))
        elif field_number == 2:
            sequence.update(_parse_feature_list_map(value))
    return context, sequence


def flatten_features(context, sequence):
    features = dict(context)
    for key, steps in sequence.items():
        if key in features:
            continue
        kind = steps[0][0] if steps else None
        values = []
        for _kind, step_values in steps:
            values.extend(step_values)
        features[key] = (kind, values)
    return features


def decode_image(raw):
    with Image.open(io.BytesIO(raw)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def resize_frame(frame):
    """Resize Bridge RGB directly to the existing 256x320 model input."""
    return np.asarray(
        Image.fromarray(frame).resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )


def _values(features, key, steps, width=1, dtype=np.float32):
    if key not in features:
        raise KeyError(f"missing feature {key}; available={sorted(features)[:30]}")
    _kind, values = features[key]
    expected = steps * width
    if len(values) != expected:
        raise ValueError(f"{key}: expected {expected} values, got {len(values)}")
    return np.asarray(values, dtype=dtype).reshape(steps, width)


def map_actions(features, steps):
    """Use the existing Bridge protocol: 7 source values + 5 zeros + terminate."""
    world = _values(features, "steps/action/world_vector", steps, 3)
    rotation = _values(features, "steps/action/rotation_delta", steps, 3)
    open_gripper = _values(
        features, "steps/action/open_gripper", steps, 1, dtype=np.float32
    )
    terminate = _values(
        features, "steps/action/terminate_episode", steps, 1, dtype=np.float32
    )
    mapped = np.zeros((steps, RT1_ACTION_DIM), dtype=np.float32)
    mapped[:, 0:3] = world
    mapped[:, 3:6] = rotation
    mapped[:, 6] = open_gripper[:, 0]
    mapped[:, 12] = terminate[:, 0]
    return mapped


def build_episode(payload, drop_first=False):
    context, sequence = parse_example(payload)
    features = flatten_features(context, sequence)
    image_key = "steps/observation/image"
    if image_key not in features:
        raise KeyError(f"no {image_key}; available={sorted(features)[:30]}")
    _kind, encoded = features[image_key]
    steps = len(encoded)
    if steps == 0:
        raise ValueError("episode has no frames")
    frames = np.stack([resize_frame(decode_image(raw)) for raw in encoded])
    actions = map_actions(features, steps)
    instruction = ""
    instruction_key = "steps/observation/natural_language_instruction"
    if instruction_key in features:
        _kind, values = features[instruction_key]
        if values:
            instruction = values[0].decode("utf-8", errors="replace").rstrip("\x00")
    state = None
    state_key = "steps/observation/state"
    if state_key in features:
        state = _values(features, state_key, steps, 7)
    if drop_first:
        frames = frames[1:]
        actions = actions[1:]
        if state is not None:
            state = state[1:]
    if len(frames) != len(actions):
        raise ValueError(f"frame/action mismatch: {len(frames)} vs {len(actions)}")
    return frames, actions, instruction, state


def validate_npz(path, expected_shard=None, expected_record=None):
    """Validate a completed output before treating it as resumable."""
    with np.load(path, allow_pickle=False) as stored:
        if "image" not in stored.files or "action" not in stored.files:
            raise ValueError(f"{path}: missing image/action arrays")
        images = stored["image"]
        actions = stored["action"]
        if images.ndim != 4 or tuple(images.shape[1:]) != (TARGET_H, TARGET_W, 3):
            raise ValueError(f"{path}: invalid image shape {images.shape}")
        if actions.ndim != 2 or actions.shape[1] != RT1_ACTION_DIM:
            raise ValueError(f"{path}: invalid action shape {actions.shape}")
        if len(images) != len(actions):
            raise ValueError(f"{path}: image/action length mismatch")
        if not np.all(actions[:, 7:12] == 0):
            raise ValueError(f"{path}: zero-padding slots 7:12 are not zero")
        if expected_shard is not None and "source_shard" in stored.files:
            if str(stored["source_shard"]) != expected_shard:
                raise ValueError(f"{path}: source shard mismatch")
        if expected_record is not None and "source_record" in stored.files:
            if int(stored["source_record"]) != expected_record:
                raise ValueError(f"{path}: source record mismatch")
        return {
            "frames": int(len(images)),
            "actions": np.asarray(actions, dtype=np.float32),
            "instruction": str(stored["instruction"]) if "instruction" in stored.files else "",
        }


def atomic_save_npz(path, compress=True, **arrays):
    """Write an NPZ atomically so interrupted conversions are never reusable."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            writer = np.savez_compressed if compress else np.savez
            writer(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_action_ranges(actions, margin=0.02):
    stacked = np.concatenate(actions, axis=0)
    ranges = np.zeros((RT1_ACTION_DIM, 2), dtype=np.float32)
    for dim in range(RT1_ACTION_DIM):
        if dim not in USED_SLOTS:
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


def binning_audit(ranges, stacked):
    rows = []
    for dim in sorted(USED_SLOTS):
        lo, hi = ranges[dim]
        values = stacked[:, dim]
        scaled = np.clip((values - lo) / (hi - lo + 1e-8), 0, 1)
        bins = np.floor(scaled * ACTION_BINS).astype(int).clip(0, ACTION_BINS - 1)
        rows.append(
            {
                "rt1_slot": dim,
                "range": [float(lo), float(hi)],
                "bins_used": int(len(np.unique(bins))),
                "bin_coverage": round(len(np.unique(bins)) / ACTION_BINS, 4),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Bridge TFDS shards -> iVideoGPT NPZ")
    parser.add_argument("--input", required=True, help="TFDS bridge/0.1.0 directory")
    parser.add_argument("--output", required=True, help="dataset output root")
    parser.add_argument("--dataset-name", default="bridge_v2")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--output-prefix", default=None, help="defaults to train or val")
    parser.add_argument("--shard-glob", default=None)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--min-length", type=int, default=8)
    parser.add_argument("--drop-artificial-first-step", action="store_true")
    parser.add_argument("--no-crc", action="store_true")
    parser.add_argument("--uncompressed", action="store_true",
                        help="use np.savez for much faster full-dataset conversion")
    parser.add_argument("--write-action-ranges", action="store_true")
    args = parser.parse_args()

    source = Path(args.input)
    pattern = args.shard_glob or f"bridge-{args.split}.tfrecord-*"
    shards = sorted(source.glob(pattern))
    exact_shard = re.compile(
        rf"^bridge-{re.escape(args.split)}\.tfrecord-\d{{5}}-of-\d{{5}}$"
    )
    shards = [path for path in shards if exact_shard.fullmatch(path.name)]
    if args.max_shards is not None:
        shards = shards[: args.max_shards]
    if not shards:
        raise SystemExit(f"no shards matching {source / pattern}")

    output_dir = Path(args.output) / args.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix or ("train" if args.split == "train" else "val")
    records = []
    observed = []
    output_index = args.start_index

    for shard_number, shard in enumerate(shards):
        for record_index, payload in read_tfrecords(shard, verify_crc=not args.no_crc):
            if args.max_episodes is not None and len(records) >= args.max_episodes:
                break
            frames, actions, instruction, state = build_episode(
                payload, drop_first=args.drop_artificial_first_step
            )
            if len(frames) < args.min_length:
                continue
            name = f"{prefix}_eps_{output_index:08d}.npz"
            target = output_dir / name
            arrays = {"image": frames.astype(np.uint8), "action": actions.astype(np.float32)}
            if state is not None:
                arrays["state"] = state.astype(np.float32)
            arrays["instruction"] = np.asarray(instruction)
            arrays["source_shard"] = np.asarray(shard.name)
            arrays["source_record"] = np.asarray(record_index, dtype=np.int64)
            atomic_save_npz(target, compress=not args.uncompressed, **arrays)
            validate_npz(target, shard.name, record_index)
            observed.append(actions)
            records.append(
                {
                    "output": name,
                    "source_shard": shard.name,
                    "source_record": record_index,
                    "frames": int(len(frames)),
                    "instruction": instruction,
                }
            )
            output_index += 1
            print("BRIDGE_EPISODE=" + json.dumps(records[-1], ensure_ascii=False), flush=True)
        if args.max_episodes is not None and len(records) >= args.max_episodes:
            break

    manifest = {
        "source": str(source),
        "dataset": args.dataset_name,
        "source_split": args.split,
        "output_prefix": prefix,
        "shards_read": sorted({row["source_shard"] for row in records}),
        "episodes": len(records),
        "timesteps": int(sum(row["frames"] for row in records)),
        "target_resolution": [TARGET_H, TARGET_W],
        "drop_artificial_first_step": args.drop_artificial_first_step,
        "action_slots": ACTION_SLOTS,
        "unused_slots_zero": sorted(set(range(RT1_ACTION_DIM)) - USED_SLOTS),
        "gripper_mapping": "open_gripper bool -> float 0/1",
        "compression": "none" if args.uncompressed else "zip-deflate",
        "protocol": "existing RLVR-World_bridge 7D + zero padding + terminate at slot 12",
        "records": records,
    }

    if args.write_action_ranges:
        if not observed:
            raise SystemExit("cannot write action ranges: no episodes converted")
        try:
            import torch
        except ImportError as error:
            raise SystemExit("torch is required for --write-action-ranges") from error
        ranges, stacked = build_action_ranges(observed)
        torch.save(torch.from_numpy(ranges), output_dir / "action_ranges.pth")
        manifest["action_ranges"] = ranges.tolist()
        manifest["binning_audit"] = binning_audit(ranges, stacked)
        manifest["action_ranges_fitted_on"] = f"converted {args.split} subset"

    manifest_path = output_dir / f"{prefix}_conversion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    # The existing two-stage launcher checks a generic manifest name.
    if prefix == "train":
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print("BRIDGE_CONVERSION_RESULT=" + json.dumps({
        "output": str(output_dir), "episodes": len(records),
        "timesteps": manifest["timesteps"], "manifest": str(manifest_path)
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
