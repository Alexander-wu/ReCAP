"""Full-length baseline rollout: full-history decoding from the first frame.

Given only frame 0 as the visual context plus the ground-truth action sequence,
the frozen world model autoregressively predicts every remaining frame of the
episode. No context eviction is applied, so this reproduces the Full Context
baseline and exposes long-horizon drift and structural collapse.
"""

import argparse
import base64
import importlib
import io
import json
import os
import shutil
import socket
import time
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from ivideogpt.ctx_tokenizer import CompressiveVQModelFSQ
from ivideogpt.processor import ContextMultiStepPredictionProcessor

TOKENS_PER_FRAME = 80
ACTION_DIM = 13
CTX_TOKENS = 1280
BLOCK_TOKENS = TOKENS_PER_FRAME + ACTION_DIM
ARTIFACT_CHUNK_CHARS = 40000


def parse_args():
    parser = argparse.ArgumentParser(description="Full-length full-history baseline rollout")
    parser.add_argument("--input-npz", type=str, required=True)
    parser.add_argument("--image-key", type=str, default="image")
    parser.add_argument("--action-key", type=str, default="action")
    parser.add_argument("--context-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means derive from the context budget")
    parser.add_argument("--tokenizer-model", type=str, default="tokenizer_model")
    parser.add_argument("--world-model", type=str, default="world_model")
    parser.add_argument("--action-ranges", type=str, default="configs/frac_action_ranges.pth")
    parser.add_argument("--output-dir", type=str, default="remote_outputs")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--decode-chunk", type=int, default=8)
    parser.add_argument("--preview-panel-width", type=int, default=160)
    parser.add_argument("--strip-frames", type=int, default=10)
    return parser.parse_args()


def frame_budget(max_model_len):
    """Largest number of rollout frames whose final decoding step still fits."""
    usable = max_model_len - CTX_TOKENS - BLOCK_TOKENS - TOKENS_PER_FRAME
    if usable < 0:
        raise ValueError(f"max_model_len={max_model_len} is too small for a single block")
    return usable // BLOCK_TOKENS + 2


def resolve_model_reference(reference):
    path = Path(reference).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"Explicit local model directory required: {reference}")
    return str(path.resolve())


def load_episode(args, num_frames):
    data = np.load(args.input_npz)
    if args.image_key not in data or args.action_key not in data:
        raise KeyError(f"NPZ must contain {args.image_key!r} and {args.action_key!r}; found {list(data.keys())}")
    images = data[args.image_key]
    actions = data[args.action_key]
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Expected images [T,H,W,3], got {images.shape}")
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected actions [T,{ACTION_DIM}], got {actions.shape}")

    start = args.context_index
    end = start + num_frames
    if start < 0 or end > len(images) or end > len(actions):
        raise ValueError(f"Invalid window: start={start}, frames={num_frames}, episode={len(images)}")

    window_images = images[start:end]
    window_actions = actions[start:end]

    # Frame 0 doubles as the scene context and as the first observed dynamic state.
    image_seq = np.concatenate([window_images[0:1], window_images], axis=0)
    action_seq = np.concatenate([window_actions[0:1], window_actions], axis=0)

    pixels = torch.from_numpy(image_seq).permute(0, 3, 1, 2).float()
    if pixels.max() > 1.5:
        pixels = pixels / 255.0
    pixels = F.interpolate(pixels, size=(256, 320), mode="bilinear", align_corners=False)
    pixels = pixels.clamp(0.0, 1.0).unsqueeze(0)
    actions_tensor = torch.from_numpy(action_seq).float().unsqueeze(0)
    return pixels, actions_tensor, len(images)


def psnr_from_mse(value):
    return float("inf") if value <= 0 else 10.0 * float(np.log10(1.0 / value))


def frames_to_gif_bytes(frames, width=None, duration=160):
    tensor = frames.detach().float().clamp(0.0, 1.0)
    if width is not None and width < tensor.shape[-1]:
        height = max(1, round(tensor.shape[-2] * width / tensor.shape[-1]))
        tensor = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    arrays = [
        (frame.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
        for frame in tensor
    ]
    buffer = io.BytesIO()
    imageio.mimsave(buffer, arrays, format="GIF", duration=duration, loop=0)
    return buffer.getvalue()


def keyframe_strip_png(ground_truth, prediction, static_baseline, count, panel_width=200):
    total = ground_truth.shape[0]
    indices = sorted(set(np.linspace(0, total - 1, min(count, total)).round().astype(int).tolist()))
    gt_row = ground_truth[indices]
    pred_row = prediction[indices]
    static_row = static_baseline[indices]
    stacked = torch.cat([gt_row, pred_row, static_row], dim=0)
    height = max(1, round(stacked.shape[-2] * panel_width / stacked.shape[-1]))
    stacked = F.interpolate(stacked, size=(height, panel_width), mode="bilinear", align_corners=False)
    columns = len(indices)
    gt_strip = torch.cat(list(stacked[:columns]), dim=-1)
    pred_strip = torch.cat(list(stacked[columns : 2 * columns]), dim=-1)
    static_strip = torch.cat(list(stacked[2 * columns :]), dim=-1)
    grid = torch.cat([gt_strip, pred_strip, static_strip], dim=-2).clamp(0.0, 1.0)
    array = (grid.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    buffer = io.BytesIO()
    imageio.imwrite(buffer, array, format="PNG")
    return buffer.getvalue(), indices


def emit_artifact(name, payload):
    encoded = base64.b64encode(payload).decode("ascii")
    total = max(1, (len(encoded) + ARTIFACT_CHUNK_CHARS - 1) // ARTIFACT_CHUNK_CHARS)
    print(f"ARTIFACT_BEGIN name={name} chunks={total} bytes={len(payload)}", flush=True)
    for index in range(total):
        piece = encoded[index * ARTIFACT_CHUNK_CHARS : (index + 1) * ARTIFACT_CHUNK_CHARS]
        print(f"ARTIFACT_CHUNK name={name} idx={index} data={piece}", flush=True)
    print(f"ARTIFACT_END name={name}", flush=True)


def main():
    from vllm import LLM, SamplingParams

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    project_dir = Path(__file__).resolve().parent
    action_ranges_path = Path(args.action_ranges)
    if not action_ranges_path.is_absolute():
        action_ranges_path = project_dir / action_ranges_path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    probe = np.load(args.input_npz)
    episode_length = len(probe[args.image_key])
    budget_frames = frame_budget(args.max_model_len)
    requested = args.max_frames if args.max_frames > 0 else episode_length - args.context_index
    num_frames = min(requested, episode_length - args.context_index, budget_frames)
    if num_frames < 2:
        raise ValueError(f"Need at least 2 frames, resolved {num_frames}")
    truncated = num_frames < episode_length - args.context_index

    started = time.time()
    pixels_cpu, actions_cpu, episode_length = load_episode(args, num_frames)
    device = torch.device("cuda:0")

    tokenizer_started = time.time()
    visual_tokenizer = CompressiveVQModelFSQ.from_pretrained(resolve_model_reference(args.tokenizer_model))
    visual_tokenizer = visual_tokenizer.to(device).eval()
    tokenizer_load_seconds = time.time() - tokenizer_started

    processor = ContextMultiStepPredictionProcessor(
        SimpleNamespace(
            action_ranges_path=str(action_ranges_path),
            tokenizer_micro_batch_size=1,
            visual_token_num=4375,
            action_bins=256,
        ),
        visual_tokenizer,
    )

    pixels = pixels_cpu.to(device)
    actions = actions_cpu.to(device)
    with torch.inference_mode():
        model_input, _, ctx_tokens = processor(pixels, actions, return_interpolated=True, return_ctx_tokens=True)

    all_ids = model_input["input_ids"]
    expected_total = CTX_TOKENS + num_frames * BLOCK_TOKENS
    if all_ids.shape[1] != expected_total:
        raise RuntimeError(f"Unexpected token layout: expected {expected_total}, got {all_ids.shape[1]}")

    world_model_started = time.time()
    model = LLM(
        model=resolve_model_reference(args.world_model),
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        enforce_eager=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
    )
    world_model_load_seconds = time.time() - world_model_started

    prompt_length = CTX_TOKENS + BLOCK_TOKENS
    prompt_tokens = all_ids[0, :prompt_length].detach().cpu().tolist()
    generated_tokens = []
    prompt_lengths = []
    step_seconds = []

    generation_started = time.time()
    for step in range(num_frames - 1):
        step_started = time.time()
        prompt_lengths.append(len(prompt_tokens))
        step_params = SamplingParams(
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=1.0,
            ignore_eos=True,
            max_tokens=TOKENS_PER_FRAME,
            seed=args.seed + step,
        )
        outputs = model.generate(
            [{"prompt_token_ids": prompt_tokens}],
            sampling_params=step_params,
            use_tqdm=False,
        )
        new_tokens = list(outputs[0].outputs[0].token_ids)
        if len(new_tokens) != TOKENS_PER_FRAME:
            raise RuntimeError(f"Step {step} produced {len(new_tokens)} tokens, expected {TOKENS_PER_FRAME}")
        generated_tokens.append(new_tokens)
        prompt_tokens.extend(new_tokens)
        action_start = len(prompt_tokens)
        next_action = all_ids[0, action_start : action_start + ACTION_DIM].detach().cpu().tolist()
        if len(next_action) != ACTION_DIM:
            raise RuntimeError(f"Missing action tokens after step {step}")
        prompt_tokens.extend(next_action)
        step_seconds.append(time.time() - step_started)
        if step % 10 == 0 or step == num_frames - 2:
            print(f"ROLLOUT_PROGRESS step={step + 1}/{num_frames - 1} prompt_tokens={prompt_lengths[-1]}", flush=True)
    generation_seconds = time.time() - generation_started

    generated = torch.tensor(generated_tokens, device=device, dtype=torch.long)
    generated = generated.unsqueeze(0).clamp(0, 4374)

    decode_started = time.time()
    decoded_chunks = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for begin in range(0, generated.shape[1], args.decode_chunk):
            piece = generated[:, begin : begin + args.decode_chunk]
            decoded_chunks.append(visual_tokenizer.detokenize(ctx_tokens, piece)[:, 1:].float())
    decoded = torch.cat(decoded_chunks, dim=1).clamp(0.0, 1.0)
    decode_seconds = time.time() - decode_started

    ground_truth = pixels[0, 1:]
    prediction = torch.cat([ground_truth[0:1], decoded[0]], dim=0)
    if prediction.shape != ground_truth.shape:
        raise RuntimeError(f"Shape mismatch: prediction {prediction.shape}, ground truth {ground_truth.shape}")

    gt_future = ground_truth[1:]
    pred_future = prediction[1:]
    squared = (gt_future - pred_future).pow(2).mean(dim=(1, 2, 3))
    absolute = (gt_future - pred_future).abs().mean(dim=(1, 2, 3))
    per_frame_psnr = [round(psnr_from_mse(value), 3) for value in squared.tolist()]
    per_frame_mae = [round(value, 6) for value in absolute.tolist()]

    static = ground_truth[0:1].expand_as(gt_future)
    static_sequence = ground_truth[0:1].expand_as(ground_truth)
    static_squared = (gt_future - static).pow(2).mean(dim=(1, 2, 3))
    per_frame_static_psnr = [round(psnr_from_mse(value), 3) for value in static_squared.tolist()]

    pred_activity = (prediction[1:] - prediction[:-1]).abs().mean(dim=(1, 2, 3))
    gt_activity = (ground_truth[1:] - ground_truth[:-1]).abs().mean(dim=(1, 2, 3))
    pred_drift = (prediction - prediction[0:1]).abs().mean(dim=(1, 2, 3))
    gt_drift = (ground_truth - ground_truth[0:1]).abs().mean(dim=(1, 2, 3))

    token_array = np.array(generated_tokens, dtype=np.int64)
    if len(token_array) > 1:
        repeat_ratio = (token_array[1:] == token_array[:-1]).mean(axis=1)
        per_frame_token_repeat = [round(float(value), 4) for value in repeat_ratio]
    else:
        per_frame_token_repeat = []
    unique_per_frame = [int(len(np.unique(row))) for row in token_array]

    # Three rows make the qualitative comparison self-contained:
    # ground truth (top), world-model prediction (middle), repeat-last-frame baseline (bottom).
    comparison = torch.cat([ground_truth, prediction, static_sequence], dim=-2)
    strip_png, strip_indices = keyframe_strip_png(
        ground_truth, prediction, static_sequence, args.strip_frames
    )
    preview_gif = frames_to_gif_bytes(comparison, width=args.preview_panel_width)
    full_gif = frames_to_gif_bytes(comparison)
    (output_dir / "comparison_full_gt_pred_static.gif").write_bytes(full_gif)

    result = {
        "status": "ok",
        "mode": "full_history_baseline_rollout",
        "source": str(args.input_npz),
        "host": socket.gethostname(),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "episode_length": int(episode_length),
        "context_index": args.context_index,
        "rollout_frames": int(num_frames),
        "predicted_frames": int(num_frames - 1),
        "truncated_by_context_budget": bool(truncated),
        "frame_budget_at_max_model_len": int(budget_frames),
        "max_model_len": args.max_model_len,
        "sampling": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "seed": args.seed,
        },
        "context": {
            "note": "no eviction; prompt grows by 93 tokens per generated frame",
            "prompt_tokens_first_step": int(prompt_lengths[0]),
            "prompt_tokens_last_step": int(prompt_lengths[-1]),
            "prompt_tokens_per_step": [int(value) for value in prompt_lengths],
        },
        "metrics": {
            "note": "pixel range [0,1]; frame index 0 is the given first frame and excluded",
            "psnr_db_mean": round(psnr_from_mse(float(squared.mean())), 3),
            "mae_mean": round(float(absolute.mean()), 6),
            "per_frame_psnr_db": per_frame_psnr,
            "per_frame_mae": per_frame_mae,
        },
        "static_frame_baseline": {
            "note": "repeat frame 0 for every step",
            "psnr_db_mean": round(psnr_from_mse(float(static_squared.mean())), 3),
            "per_frame_psnr_db": per_frame_static_psnr,
        },
        "visual_layout": {
            "row_1": "ground truth",
            "row_2": "world-model prediction",
            "row_3": "repeat-last-observed-frame baseline",
        },
        "collapse_diagnostics": {
            "note": "activity is mean |f_t - f_{t-1}|; drift is mean |f_t - f_0|",
            "prediction_activity": [round(value, 6) for value in pred_activity.tolist()],
            "ground_truth_activity": [round(value, 6) for value in gt_activity.tolist()],
            "prediction_drift_from_first": [round(value, 6) for value in pred_drift.tolist()],
            "ground_truth_drift_from_first": [round(value, 6) for value in gt_drift.tolist()],
            "token_repeat_ratio_vs_previous_frame": per_frame_token_repeat,
            "unique_tokens_per_frame": unique_per_frame,
            "tokens_per_frame": TOKENS_PER_FRAME,
        },
        "keyframe_strip_indices": strip_indices,
        "tokenizer_load_seconds": round(tokenizer_load_seconds, 3),
        "world_model_load_seconds": round(world_model_load_seconds, 3),
        "generation_seconds": round(generation_seconds, 3),
        "generation_seconds_per_step": [round(value, 4) for value in step_seconds],
        "decode_seconds": round(decode_seconds, 3),
        "total_seconds": round(time.time() - started, 3),
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("ROLLOUT_RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)
    emit_artifact("keyframe_strip_gt_pred_static.png", strip_png)
    emit_artifact("preview_gt_pred_static.gif", preview_gif)


if __name__ == "__main__":
    main()
