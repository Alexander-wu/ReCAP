"""Compare context-management strategies for RLVR-World video extrapolation.

Reproduces, in a single process on one GPU, the three inference schedules used
in the ReCAP main table:

  full_history   Baseline. Prompt keeps every generated block. When the prompt
                 exceeds MAX_PROMPT_LEN it is truncated to
                 ctx_prefix + last (MAX_PROMPT_LEN - CTX_PREFIX_LEN) tokens.
                 That tail length is not a multiple of the 93-token block, so
                 the truncation cuts mid-block; we record when it triggers.

  recap          ReCAP. Prompt = ctx_prefix + frozen anchor block A + the last W
                 complete visual-action blocks. Never leaves token space, so the
                 active context stays constant at ctx + (1 + W) blocks.

  recap_motion   Motion-aware ReCAP. Uses the same bounded prompt, but periodically
                 replaces A with a high-novelty complete block that has already
                 left the recent window. Novelty is visual-token Hamming distance.

Frame alignment follows the reference implementation: frame 0 is the visual
context, frame 1 is the first real dynamic frame, and generation starts at
frame 2. Predicted frame k is therefore compared against ground-truth frame
k + 2.
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

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from ivideogpt.ctx_tokenizer import CompressiveVQModelFSQ
from traj_metrics import trajectory_metrics

IMAGE_SIZE = (256, 320)
TOKENS_PER_FRAME = 80
ACTION_DIM = 13
ACTION_BINS = 256
VISUAL_TOKEN_NUM = 4375
BLOCK = TOKENS_PER_FRAME + ACTION_DIM
CTX_PREFIX_LEN = 1280
MAX_PROMPT_LEN = 7900
DECODE_BATCH = 10
ARTIFACT_CHUNK_CHARS = 40000
STRATEGY_ROWS = [
    "ground_truth",
    "full_history",
    "sliding_window",
    "sliding_window_eq",
    "uniform_sampling",
    "streaming_llm",
    "block_kv",
    "anchor_uniform",
    "stale_inject",
    "recap",
    "recap_motion",
    "repeat_frame",
]
STRATEGY_NAMES = set(STRATEGY_ROWS) - {"ground_truth", "repeat_frame"}


def parse_strategy_spec(spec):
    """Split a strategy token into a base name plus optional inline overrides.

    Loading the tokenizer and the vLLM engine costs far more than a single
    rollout, so a stale-ratio sweep should not pay that cost once per k. A spec
    like `stale_inject@k=4+src=generated` therefore names a distinct run inside
    one process. Overrides are joined with `+` because `,` already separates
    strategies. Plain names such as `recap` keep working unchanged.
    """
    if "@" not in spec:
        return spec, {}, spec
    base, _, raw = spec.partition("@")
    base = base.strip()
    overrides = {}
    for item in raw.split("+"):
        item = item.strip()
        if not item:
            continue
        key, _, value = item.partition("=")
        key = key.strip()
        value = value.strip()
        if key == "k":
            overrides["stale_blocks_k"] = int(value)
        elif key == "src":
            if value not in ("generated", "ground_truth"):
                raise ValueError(f"Unknown stale source '{value}' in '{spec}'")
            overrides["stale_source"] = value
        elif key == "w":
            overrides["window_size_w"] = int(value)
        else:
            raise ValueError(f"Unknown strategy override '{key}' in '{spec}'")
    if not overrides:
        raise ValueError(f"Strategy spec '{spec}' has '@' but no overrides")
    return base, overrides, spec


def parse_args():
    parser = argparse.ArgumentParser(description="ReCAP vs baseline context schedules")
    parser.add_argument("--input-npz", type=str, required=True)
    parser.add_argument("--image-key", type=str, default="image")
    parser.add_argument("--action-key", type=str, default="action")
    parser.add_argument("--context-index", type=int, default=0,
                        help="first observed frame within the episode")
    parser.add_argument("--num-frames", type=int, default=100,
                        help="number of future frames to generate after two observed frames")
    parser.add_argument("--tokenizer-model", type=str, default="tokenizer_model")
    parser.add_argument("--world-model", type=str, default="world_model")
    parser.add_argument("--action-ranges", type=str, default="configs/frac_action_ranges.pth")
    parser.add_argument("--output-dir", type=str, default="remote_outputs")
    parser.add_argument("--strategies", type=str,
                        default="full_history,sliding_window,uniform_sampling,streaming_llm,"
                                "block_kv,recap")
    parser.add_argument("--anchor-frames", type=int, default=1, help="[ReCAP] frozen anchor blocks")
    parser.add_argument("--window-size-w", type=int, default=6, help="[ReCAP] recent block buffer W")
    parser.add_argument("--overlap-k", type=int, default=3, help="[ReCAP] bookkeeping tag K")
    parser.add_argument("--stale-blocks-k", type=int, default=0,
                        help="[Stale intervention] replace this many of the W recent blocks with "
                             "stale content while holding the total block budget fixed")
    parser.add_argument("--stale-source", type=str, default="generated",
                        choices=["generated", "ground_truth"],
                        help="[Stale intervention] 'generated' injects the model's own older "
                             "predictions (confounds temporal distance with accumulated error); "
                             "'ground_truth' injects real frames at matched temporal distance, "
                             "which isolates distance from self-generated error")
    parser.add_argument("--motion-anchor-refresh-interval", type=int, default=8,
                        help="[Motion ReCAP] reconsider the anchor every N generated frames")
    parser.add_argument("--motion-anchor-history", type=int, default=32,
                        help="[Motion ReCAP] eligible old-block lookback")
    parser.add_argument("--motion-anchor-min-novelty", type=float, default=0.12,
                        help="[Motion ReCAP] minimum visual-token Hamming novelty")
    parser.add_argument("--motion-anchor-recency-weight", type=float, default=0.05,
                        help="[Motion ReCAP] tie-break weight favoring recent eligible blocks")
    parser.add_argument("--streaming-sink-tokens", type=int, default=4,
                        help="[StreamingLLM] attention-sink token count; the paper uses 4 tokens, "
                             "which is deliberately not block aligned")
    parser.add_argument("--block-kv-recency-decay", type=float, default=0.08,
                        help="[Block-KV] exponential recency decay in the surrogate score")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--preview-panel-width", type=int, default=176)
    parser.add_argument("--strip-frames", type=int, default=10)
    parser.add_argument("--emit-keyframe", dest="emit_keyframe", action="store_true", default=True,
                        help="stream the keyframe grid back through the job log")
    parser.add_argument("--no-emit-keyframe", dest="emit_keyframe", action="store_false",
                        help="keep the keyframe grid only in the remote output directory")
    parser.add_argument("--emit-preview", dest="emit_preview", action="store_true", default=True,
                        help="stream the comparison GIF back through the job log")
    parser.add_argument("--no-emit-preview", dest="emit_preview", action="store_false",
                        help="skip the GIF in the log; use for batch runs to keep logs small")
    parser.add_argument("--emit-frames", dest="emit_frames", action="store_true", default=False,
                        help="stream the decoded rollouts back as a compressed npz; needed "
                             "for offline VLM judging")
    parser.add_argument("--frames-keep", type=int, default=0,
                        help="when >0, subsample the archived rollouts to this many evenly "
                             "spaced frames per strategy. A VLM judge only ever looks at a "
                             "contact sheet of ~12 frames, so archiving every frame wastes "
                             "roughly an order of magnitude of log bandwidth in batch runs. "
                             "0 keeps the full rollout.")
    return parser.parse_args()


def resolve_model_reference(reference):
    path = Path(reference).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"Explicit local model directory required: {reference}")
    return str(path.resolve())


def discretize_actions(actions, action_ranges, num_bins=ACTION_BINS):
    minimum, maximum = action_ranges[:, 0], action_ranges[:, 1]
    scaled = torch.clip((actions - minimum) / (maximum - minimum + 1e-8), 0, 1)
    return torch.floor(scaled * num_bins).to(torch.int32).clip(0, num_bins - 1)


def action_tokens_at(index, all_actions, action_ranges, total_frames, device):
    source = all_actions[index] if index < total_frames else all_actions[-1]
    tokens = discretize_actions(source.unsqueeze(0).to(device), action_ranges)
    tokens = tokens + VISUAL_TOKEN_NUM * 2
    return tokens[0].detach().cpu().tolist()


def pad_frame_tokens(tokens):
    if len(tokens) < TOKENS_PER_FRAME:
        filler = tokens[-1] if tokens else 0
        return list(tokens) + [filler] * (TOKENS_PER_FRAME - len(tokens))
    return list(tokens[:TOKENS_PER_FRAME])


def decode_frames(tokenizer, ctx_tokens, dyn_tokens_list, device):
    padded = [pad_frame_tokens(tokens) for tokens in dyn_tokens_list]
    generated = torch.tensor(padded, dtype=torch.long).unsqueeze(0).to(device)
    generated = generated.clamp(0, VISUAL_TOKEN_NUM - 1).reshape(1, len(padded), 8 * 10)

    pieces = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for begin in range(0, len(padded), DECODE_BATCH):
            batch = generated[:, begin : begin + DECODE_BATCH, :]
            pieces.append(tokenizer.detokenize(ctx_tokens, batch)[:, 1:])
    return torch.cat(pieces, dim=1).float().clamp(0.0, 1.0)


def sample_one(model, sampling_params, prompt_tokens):
    outputs = model.generate(
        [{"prompt_token_ids": prompt_tokens}],
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    tokens = list(outputs[0].outputs[0].token_ids)
    if len(tokens) != TOKENS_PER_FRAME:
        raise RuntimeError(f"Generated {len(tokens)} visual tokens, expected {TOKENS_PER_FRAME}")
    invalid = [token for token in tokens if token < 0 or token >= VISUAL_TOKEN_NUM]
    if invalid:
        raise RuntimeError(
            f"Generated {len(invalid)} non-visual tokens outside [0,{VISUAL_TOKEN_NUM - 1}]; "
            f"sample={invalid[:8]}"
        )
    return tokens


def build_sampling_params(args):
    from vllm import SamplingParams

    return SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=1.0,
        ignore_eos=True,
        max_tokens=TOKENS_PER_FRAME,
        seed=args.seed,
    )


def initial_prompt(ctx_tokens, dyn_tokens, all_actions, action_ranges, device):
    ctx_prefix = (ctx_tokens.reshape(1, -1) + VISUAL_TOKEN_NUM)[0].detach().cpu().tolist()
    first_frame = dyn_tokens[:, 0, :].reshape(-1).detach().cpu().tolist()
    first_action = discretize_actions(all_actions[1].unsqueeze(0).to(device), action_ranges)
    first_action = (first_action + VISUAL_TOKEN_NUM * 2)[0].detach().cpu().tolist()
    return ctx_prefix, first_frame + first_action


def run_full_history(model, args, context, progress):
    """Baseline: retain everything, token-level truncation once the prompt overflows."""
    ctx_prefix, anchor_block = context["prompt_parts"]
    sampling_params = context["sampling_params"]
    prompt_tokens = list(ctx_prefix) + list(anchor_block)

    generated = []
    prompt_lengths = []
    truncation_steps = []
    max_prompt_len = context["max_prompt_len"]
    keep_recent = max_prompt_len - CTX_PREFIX_LEN
    started = time.time()

    for step in range(context["num_frames"]):
        if len(prompt_tokens) > max_prompt_len:
            prompt_tokens = prompt_tokens[:CTX_PREFIX_LEN] + prompt_tokens[-keep_recent:]
            truncation_steps.append(step)
        prompt_lengths.append(len(prompt_tokens))
        tokens = sample_one(model, sampling_params, prompt_tokens)
        generated.append(tokens)
        prompt_tokens.extend(pad_frame_tokens(tokens))
        prompt_tokens.extend(
            action_tokens_at(step + 2, context["all_actions"], context["action_ranges"],
                             context["total_frames"], context["device"])
        )
        progress("full_history", step, context["num_frames"], prompt_lengths[-1])

    return {
        "tokens": generated,
        "prompt_tokens_per_step": prompt_lengths,
        "seconds": time.time() - started,
        "notes": {
            "truncation_first_step": truncation_steps[0] if truncation_steps else None,
            "truncation_steps": len(truncation_steps),
            "tail_tokens_kept": keep_recent,
            "tail_is_block_aligned": keep_recent % BLOCK == 0,
            "blocks_in_tail": keep_recent / BLOCK,
        },
    }


def run_sliding_window_eq(model, args, context, progress):
    """Sliding Window at exactly ReCAP's token budget (W+1 recent blocks).

    Why this variant is necessary: plain `sliding_window` keeps W blocks while
    ReCAP keeps anchor + W = W+1 blocks, so ReCAP silently enjoys one extra
    block (93 tokens) of context. Any ReCAP advantage over `sliding_window`
    could therefore be a budget effect rather than an anchor effect.

    This control spends the *same* W+1 blocks purely on recent history. Comparing
    ReCAP against it isolates one question only: given a fixed budget, is it
    better to spend one block on the frozen first real frame, or on one more
    recent frame?
    """
    ctx_prefix, anchor_block = context["prompt_parts"]
    sampling_params = context["sampling_params"]
    budget = args.window_size_w + 1

    recent_blocks = [list(anchor_block)]
    generated = []
    prompt_lengths = []
    anchor_dropped_at = None
    started = time.time()

    for step in range(context["num_frames"]):
        prompt_tokens = list(ctx_prefix)
        for block in recent_blocks:
            prompt_tokens.extend(block)
        prompt_lengths.append(len(prompt_tokens))

        tokens = sample_one(model, sampling_params, prompt_tokens)
        generated.append(tokens)

        block = pad_frame_tokens(tokens)
        block.extend(
            action_tokens_at(step + 2, context["all_actions"], context["action_ranges"],
                             context["total_frames"], context["device"])
        )
        recent_blocks.append(block)
        if len(recent_blocks) > budget:
            if anchor_dropped_at is None:
                anchor_dropped_at = step
            recent_blocks = recent_blocks[-budget:]
        progress("sliding_window_eq", step, context["num_frames"], prompt_lengths[-1])

    return {
        "tokens": generated,
        "prompt_tokens_per_step": prompt_lengths,
        "seconds": time.time() - started,
        "notes": {
            "recent_blocks": budget,
            "anchor_preserved": False,
            "anchor_dropped_at_step": anchor_dropped_at,
            "prompt_tokens_bound": CTX_PREFIX_LEN + BLOCK * budget,
            "matches_recap_budget": True,
            "pixel_space_round_trips": 0,
        },
    }


def run_anchor_uniform(model, args, context, progress):
    """Anchor + Uniform History: keep the anchor, replace the recent window.

    This is the clean control for the *recent trajectory* component. The budget is
    identical to ReCAP (anchor + W blocks), but the W non-anchor slots are spread
    uniformly over the entire generated history instead of being the last W blocks.
    No recent slots are reserved, unlike `uniform_sampling`, so the contrast with
    ReCAP isolates exactly one factor: does the bounded budget need to be spent on
    *contiguous recent* history, or does any evenly spaced coverage work?

    The anchor is always replayed first and the sampled blocks stay in temporal
    order, so causal ordering is preserved; only step-to-step contiguity is broken.
    Note that `linspace` includes the right endpoint, so the single most recent
    block is always retained. This is deliberate -- dropping it would also remove
    the immediate predecessor state and confound the comparison with a one-step
    conditioning gap -- but it means the contrast measures the value of a
    *contiguous run* of recent blocks, not of recency per se.
    """
    ctx_prefix, anchor_block = context["prompt_parts"]
    sampling_params = context["sampling_params"]
    slots = args.window_size_w

    history = []
    generated = []
    prompt_lengths = []
    contiguous_steps = 0
    started = time.time()

    for step in range(context["num_frames"]):
        if len(history) <= slots:
            selected = list(range(len(history)))
            contiguous_steps += 1
        else:
            picks = np.linspace(0, len(history) - 1, slots).round().astype(int).tolist()
            selected = sorted(set(picks))

        prompt_tokens = list(ctx_prefix) + list(anchor_block)
        for index in selected:
            prompt_tokens.extend(history[index])
        prompt_lengths.append(len(prompt_tokens))

        tokens = sample_one(model, sampling_params, prompt_tokens)
        generated.append(tokens)

        block = pad_frame_tokens(tokens)
        block.extend(
            action_tokens_at(step + 2, context["all_actions"], context["action_ranges"],
                             context["total_frames"], context["device"])
        )
        history.append(block)
        progress("anchor_uniform", step, context["num_frames"], prompt_lengths[-1])

    return {
        "tokens": generated,
        "prompt_tokens_per_step": prompt_lengths,
        "seconds": time.time() - started,
        "notes": {
            "anchor_frames": 1,
            "uniform_slots": slots,
            "most_recent_block_always_kept": True,
            "contiguous_recent_run": 1,
            "steps_still_contiguous": contiguous_steps,
            "prompt_tokens_bound": CTX_PREFIX_LEN + BLOCK * (1 + slots),
            "matches_recap_budget": True,
            "pixel_space_round_trips": 0,
        },
    }


def ground_truth_block(frame_index, context):
    """Encode a real episode frame into one visual-action block.

    Used only by the ground-truth arm of the stale intervention. The tokenizer is
    conditioned on the same static context frame as the rollout, so the resulting
    block is drawn from the same token distribution as generated blocks and the
    only thing that changes is whether its content is real or self-generated.
    """
    cache = context.setdefault("gt_block_cache", {})
    if frame_index in cache:
        return list(cache[frame_index])
    pixels = context["pixels"]
    clamped = min(max(frame_index, 1), len(pixels) - 1)
    pair = torch.stack([context["ctx_pixel"], pixels[clamped]], dim=0)
    pair = pair.unsqueeze(0).to(context["device"])
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, dyn = context["tokenizer"].tokenize(pair)
    block = dyn[:, 0, :].reshape(-1).detach().cpu().tolist()
    block = pad_frame_tokens(block)
    block.extend(
        action_tokens_at(clamped, context["all_actions"], context["action_ranges"],
                         context["total_frames"], context["device"])
    )
    cache[frame_index] = list(block)
    return list(block)


def run_stale_inject(model, args, context, progress):
    """Stale-history intervention at a fixed context budget.

    ReCAP normally spends its W non-anchor slots on the last W blocks. Here the
    oldest `k` of those slots are instead filled with stale content taken from
    `2W` steps further back, so the budget, block alignment and anchor are all
    unchanged and the only manipulated variable is how much of the window is
    stale. `k=0` reduces exactly to ReCAP.

    Two arms disentangle the two things that co-vary in stale context:
      generated    -- the model's own older predictions (far in time AND already
                      corrupted by accumulated error)
      ground_truth -- real frames at the same temporal distance (far in time but
                      not corrupted)
    If only the generated arm degrades monotonically, the harmful factor is
    self-generated staleness rather than temporal distance itself.
    """
    ctx_prefix, anchor_block = context["prompt_parts"]
    sampling_params = context["sampling_params"]
    window = args.window_size_w
    stale_k = args.stale_blocks_k
    if stale_k < 0 or stale_k > window:
        raise ValueError(f"stale_blocks_k must be within [0,{window}], got {stale_k}")
    stale_lag = 2 * window

    history = []
    generated = []
    prompt_lengths = []
    stale_active_steps = 0
    stale_indices_used = []
    replaced_per_step = []
    started = time.time()

    for step in range(context["num_frames"]):
        recent = history[-window:] if window else []
        oldest_recent = len(history) - len(recent)
        stale_slots = min(stale_k, len(recent))

        # Decide slot by slot. A stale source that predates the rollout does not
        # exist yet, so that slot keeps its original fresh block instead of being
        # back-filled with the anchor. Injecting the anchor there would place the
        # same block in the prompt twice, which is neither ReCAP nor a stale
        # intervention and would contaminate the early part of every k>0 curve.
        window_blocks = []
        replaced_this_step = 0
        for position, block in enumerate(recent):
            if position >= stale_slots:
                window_blocks.append(block)
                continue
            source = oldest_recent + position - stale_lag
            if source < 0:
                window_blocks.append(block)
                continue
            if args.stale_source == "ground_truth":
                window_blocks.append(ground_truth_block(source + 2, context))
                stale_indices_used.append(source + 2)
            else:
                window_blocks.append(list(history[source]))
                stale_indices_used.append(source)
            replaced_this_step += 1
        if replaced_this_step:
            stale_active_steps += 1
            replaced_per_step.append(replaced_this_step)

        prompt_tokens = list(ctx_prefix) + list(anchor_block)
        for block in window_blocks:
            prompt_tokens.extend(block)
        prompt_lengths.append(len(prompt_tokens))

        tokens = sample_one(model, sampling_params, prompt_tokens)
        generated.append(tokens)

        block = pad_frame_tokens(tokens)
        block.extend(
            action_tokens_at(step + 2, context["all_actions"], context["action_ranges"],
                             context["total_frames"], context["device"])
        )
        history.append(block)
        progress("stale_inject", step, context["num_frames"], prompt_lengths[-1])

    return {
        "tokens": generated,
        "prompt_tokens_per_step": prompt_lengths,
        "seconds": time.time() - started,
        "notes": {
            "anchor_frames": 1,
            "window_size_W": window,
            "stale_blocks_k": stale_k,
            "fresh_blocks": window - stale_k,
            "stale_source": args.stale_source,
            "stale_lag_blocks": stale_lag,
            "steps_with_stale_slots": stale_active_steps,
            "max_blocks_replaced_in_a_step": max(replaced_per_step) if replaced_per_step else 0,
            "first_step_with_full_k": (
                replaced_per_step.index(stale_k) if stale_k in replaced_per_step else None
            ),
            "distinct_stale_sources": len(set(stale_indices_used)),
            "anchor_never_duplicated": True,
            "reduces_to_recap": stale_k == 0,
            "prompt_tokens_bound": CTX_PREFIX_LEN + BLOCK * (1 + window),
            "pixel_space_round_trips": 0,
        },
    }


def run_recap(model, args, context, progress):
    """ReCAP: ctx prefix + frozen anchor block + last W complete blocks."""
    ctx_prefix, anchor_block = context["prompt_parts"]
    sampling_params = context["sampling_params"]
    if args.anchor_frames != 1:
        print(f"[recap][warn] anchor_frames={args.anchor_frames} requested but only one real "
              f"dynamic frame is pre-encoded; using 1.", flush=True)

    recent_blocks = []
    generated = []
    prompt_lengths = []
    started = time.time()

    for step in range(context["num_frames"]):
        prompt_tokens = list(ctx_prefix) + list(anchor_block)
        for block in recent_blocks:
            prompt_tokens.extend(block)
        prompt_lengths.append(len(prompt_tokens))

        tokens = sample_one(model, sampling_params, prompt_tokens)
        generated.append(tokens)

        block = pad_frame_tokens(tokens)
        block.extend(
            action_tokens_at(step + 2, context["all_actions"], context["action_ranges"],
                             context["total_frames"], context["device"])
        )
        recent_blocks.append(block)
        if len(recent_blocks) > args.window_size_w:
            recent_blocks = recent_blocks[-args.window_size_w:]
        progress("recap", step, context["num_frames"], prompt_lengths[-1])

    return {
        "tokens": generated,
        "prompt_tokens_per_step": prompt_lengths,
        "seconds": time.time() - started,
        "notes": {
            "anchor_frames": 1,
            "window_size_W": args.window_size_w,
            "overlap_K": args.overlap_k,
            "overlap_K_effective": False,
            "schedule": "frozen first dynamic block + FIFO last-W complete blocks",
            "prompt_tokens_bound": CTX_PREFIX_LEN + BLOCK * (1 + args.window_size_w),
            "pixel_space_round_trips": 0,
        },
    }


def visual_token_novelty(current_tokens, previous_tokens):
    current = np.asarray(pad_frame_tokens(current_tokens), dtype=np.int64)
    previous = np.asarray(pad_frame_tokens(previous_tokens), dtype=np.int64)
    return float(np.mean(current != previous))


def run_recap_motion(model, args, context, progress):
    """Bounded ReCAP with a causal, motion-aware anchor refresh.

    The anchor is selected only from complete blocks that have already left the
    recent W-block window. Every refresh interval, choose a high-novelty block
    from the last H eligible blocks. Novelty is the Hamming distance between
    consecutive predicted visual-token grids; a small recency term prevents a
    permanently frozen early anchor. No pixels are decoded during scheduling.
    """
    ctx_prefix, initial_anchor = context["prompt_parts"]
    sampling_params = context["sampling_params"]
    if args.motion_anchor_refresh_interval < 1:
        raise ValueError("motion_anchor_refresh_interval must be >= 1")
    if args.motion_anchor_history < 1:
        raise ValueError("motion_anchor_history must be >= 1")
    if not 0.0 <= args.motion_anchor_min_novelty <= 1.0:
        raise ValueError("motion_anchor_min_novelty must be in [0,1]")

    anchor = {
        "frame_index": 1,
        "block": list(initial_anchor),
        "novelty": 0.0,
    }
    history = [anchor]
    recent = []
    generated = []
    prompt_lengths = []
    anchor_updates = []
    novelty_trace = []
    previous_visual = list(initial_anchor[:TOKENS_PER_FRAME])
    started = time.time()

    for step in range(context["num_frames"]):
        prompt_tokens = list(ctx_prefix) + list(anchor["block"])
        for entry in recent:
            prompt_tokens.extend(entry["block"])
        prompt_lengths.append(len(prompt_tokens))

        tokens = sample_one(model, sampling_params, prompt_tokens)
        generated.append(tokens)
        novelty = visual_token_novelty(tokens, previous_visual)
        novelty_trace.append(round(novelty, 6))
        previous_visual = pad_frame_tokens(tokens)

        block = pad_frame_tokens(tokens)
        block.extend(
            action_tokens_at(step + 2, context["all_actions"], context["action_ranges"],
                             context["total_frames"], context["device"])
        )
        entry = {
            "frame_index": step + 2,
            "block": block,
            "novelty": novelty,
        }
        history.append(entry)
        recent.append(entry)
        if len(recent) > args.window_size_w:
            recent = recent[-args.window_size_w:]

        should_refresh = (step + 1) % args.motion_anchor_refresh_interval == 0
        if should_refresh and len(history) > len(recent):
            eligible = history[:-len(recent)] if recent else history
            pool = eligible[-args.motion_anchor_history:]
            candidates = [item for item in pool if item["novelty"] >= args.motion_anchor_min_novelty]
            if candidates:
                denominator = max(len(pool) - 1, 1)
                positions = {item["frame_index"]: index / denominator for index, item in enumerate(pool)}
                selected = max(
                    candidates,
                    key=lambda item: (
                        item["novelty"]
                        + args.motion_anchor_recency_weight * positions[item["frame_index"]],
                        item["frame_index"],
                    ),
                )
                if selected["frame_index"] != anchor["frame_index"]:
                    anchor = selected
                    anchor_updates.append({
                        "after_generated_step": step + 1,
                        "anchor_frame_index": anchor["frame_index"],
                        "novelty": round(anchor["novelty"], 6),
                    })
        progress("recap_motion", step, context["num_frames"], prompt_lengths[-1])

    return {
        "tokens": generated,
        "prompt_tokens_per_step": prompt_lengths,
        "seconds": time.time() - started,
        "notes": {
            "anchor_frames": 1,
            "window_size_W": args.window_size_w,
            "schedule": "motion-aware old anchor + FIFO last-W complete blocks",
            "motion_proxy": "Hamming novelty between consecutive 80-token visual grids",
            "motion_anchor_refresh_interval": args.motion_anchor_refresh_interval,
            "motion_anchor_history": args.motion_anchor_history,
            "motion_anchor_min_novelty": args.motion_anchor_min_novelty,
            "motion_anchor_recency_weight": args.motion_anchor_recency_weight,
            "anchor_updates": anchor_updates,
            "anchor_update_count": len(anchor_updates),
            "final_anchor_frame_index": anchor["frame_index"],
            "novelty_trace": novelty_trace,
            "prompt_tokens_bound": CTX_PREFIX_LEN + BLOCK * (1 + args.window_size_w),
            "pixel_space_round_trips": 0,
            "block_aligned": True,
        },
    }


def run_sliding_window(model, args, context, progress):
    """Sliding Window: keep only the last W complete blocks, no anchor.

    This is ReCAP minus anchor preservation, which makes it the cleanest control
    for the anchor component: the two schedules differ by exactly one block.
    The very first real dynamic frame therefore leaves the prompt once W blocks
    of generated history exist.
    """
    ctx_prefix, anchor_block = context["prompt_parts"]
    sampling_params = context["sampling_params"]
    budget = args.window_size_w

    recent_blocks = [list(anchor_block)]
    generated = []
    prompt_lengths = []
    anchor_dropped_at = None
    started = time.time()

    for step in range(context["num_frames"]):
        prompt_tokens = list(ctx_prefix)
        for block in recent_blocks:
            prompt_tokens.extend(block)
        prompt_lengths.append(len(prompt_tokens))

        tokens = sample_one(model, sampling_params, prompt_tokens)
        generated.append(tokens)

        block = pad_frame_tokens(tokens)
        block.extend(
            action_tokens_at(step + 2, context["all_actions"], context["action_ranges"],
                             context["total_frames"], context["device"])
        )
        recent_blocks.append(block)
        if len(recent_blocks) > budget:
            if anchor_dropped_at is None:
                anchor_dropped_at = step
            recent_blocks = recent_blocks[-budget:]
        progress("sliding_window", step, context["num_frames"], prompt_lengths[-1])

    return {
        "tokens": generated,
        "prompt_tokens_per_step": prompt_lengths,
        "seconds": time.time() - started,
        "notes": {
            "window_size_W": budget,
            "anchor_preserved": False,
            "anchor_dropped_at_step": anchor_dropped_at,
            "prompt_tokens_bound": CTX_PREFIX_LEN + BLOCK * budget,
            "pixel_space_round_trips": 0,
        },
    }


def run_uniform_sampling(model, args, context, progress):
    """Uniform Sampling: spend the same block budget on evenly spaced history.

    Half the budget covers the most recent blocks so local dynamics survive; the
    rest is spread uniformly over the older history. Retained blocks are always
    replayed in their original temporal order, but the gaps between them break
    the step-to-step causal continuity seen during pretraining.
    """
    ctx_prefix, anchor_block = context["prompt_parts"]
    sampling_params = context["sampling_params"]
    budget = 1 + args.window_size_w
    recent_reserved = max(1, budget // 2)

    history = [list(anchor_block)]
    generated = []
    prompt_lengths = []
    started = time.time()

    for step in range(context["num_frames"]):
        if len(history) <= budget:
            selected = list(range(len(history)))
        else:
            tail_start = len(history) - recent_reserved
            older = budget - recent_reserved
            picks = np.linspace(0, tail_start - 1, older).round().astype(int).tolist()
            selected = sorted(set(picks)) + list(range(tail_start, len(history)))

        prompt_tokens = list(ctx_prefix)
        for index in selected:
            prompt_tokens.extend(history[index])
        prompt_lengths.append(len(prompt_tokens))

        tokens = sample_one(model, sampling_params, prompt_tokens)
        generated.append(tokens)

        block = pad_frame_tokens(tokens)
        block.extend(
            action_tokens_at(step + 2, context["all_actions"], context["action_ranges"],
                             context["total_frames"], context["device"])
        )
        history.append(block)
        progress("uniform_sampling", step, context["num_frames"], prompt_lengths[-1])

    return {
        "tokens": generated,
        "prompt_tokens_per_step": prompt_lengths,
        "seconds": time.time() - started,
        "notes": {
            "block_budget": budget,
            "recent_blocks_reserved": recent_reserved,
            "older_blocks_subsampled": budget - recent_reserved,
            "prompt_tokens_bound": CTX_PREFIX_LEN + BLOCK * budget,
            "pixel_space_round_trips": 0,
        },
    }


def run_streaming_llm(model, args, context, progress):
    """StreamingLLM-style: attention-sink tokens plus a token-level recent window.

    Faithful to the language-model formulation, the budget is counted in tokens
    rather than blocks, so the oldest surviving block is generally truncated in
    the middle. That is exactly the block-structure violation ReCAP avoids, and
    the token budget here matches ReCAP's prompt so the comparison is fair.
    """
    ctx_prefix, anchor_block = context["prompt_parts"]
    sampling_params = context["sampling_params"]
    sink_tokens = args.streaming_sink_tokens
    token_budget = BLOCK * (1 + args.window_size_w)
    recent_budget = max(BLOCK, token_budget - sink_tokens)

    stream = list(anchor_block)
    generated = []
    prompt_lengths = []
    mid_block_cuts = 0
    first_cut_step = None
    started = time.time()

    for step in range(context["num_frames"]):
        sink = stream[:sink_tokens]
        tail = stream[len(sink):]
        if len(tail) > recent_budget:
            # Absolute offset in `stream` where the surviving window starts.
            # Block boundaries sit at multiples of BLOCK, so a non-zero
            # remainder means the oldest surviving block is cut in the middle.
            window_start = len(stream) - recent_budget
            tail = tail[len(tail) - recent_budget:]
            if window_start % BLOCK != 0:
                mid_block_cuts += 1
                if first_cut_step is None:
                    first_cut_step = step
        prompt_tokens = list(ctx_prefix) + sink + tail
        prompt_lengths.append(len(prompt_tokens))

        tokens = sample_one(model, sampling_params, prompt_tokens)
        generated.append(tokens)

        stream.extend(pad_frame_tokens(tokens))
        stream.extend(
            action_tokens_at(step + 2, context["all_actions"], context["action_ranges"],
                             context["total_frames"], context["device"])
        )
        progress("streaming_llm", step, context["num_frames"], prompt_lengths[-1])

    return {
        "tokens": generated,
        "prompt_tokens_per_step": prompt_lengths,
        "seconds": time.time() - started,
        "notes": {
            "sink_tokens": sink_tokens,
            "recent_token_budget": recent_budget,
            "total_token_budget": token_budget,
            "block_aligned": recent_budget % BLOCK == 0,
            "steps_with_mid_block_cut": mid_block_cuts,
            "first_mid_block_cut_step": first_cut_step,
            "prompt_tokens_bound": CTX_PREFIX_LEN + sink_tokens + recent_budget,
            "pixel_space_round_trips": 0,
        },
    }


def run_block_kv_selection(model, args, context, progress):
    """Block-KV Selection: keep the highest-scoring past blocks under a budget.

    H2O and SnapKV rank individual KV entries by accumulated attention mass.
    vLLM does not expose attention weights during generation, so we adapt the
    same idea to block granularity using the two signals those methods end up
    tracking in practice: strong recency, plus how much a block overlaps the
    current state in token space. Blocks that share more visual tokens with the
    latest frame stand in for blocks the model keeps attending to. The most
    recent block is always retained and selected blocks are replayed in temporal
    order, so this is a block-level rather than token-level eviction rule.
    """
    ctx_prefix, anchor_block = context["prompt_parts"]
    sampling_params = context["sampling_params"]
    budget = 1 + args.window_size_w
    decay = args.block_kv_recency_decay

    history = [list(anchor_block)]
    generated = []
    prompt_lengths = []
    started = time.time()

    for step in range(context["num_frames"]):
        count = len(history)
        if count <= budget:
            selected = list(range(count))
        else:
            latest = set(history[-1][:TOKENS_PER_FRAME])
            affinity = np.array(
                [len(latest.intersection(block[:TOKENS_PER_FRAME])) / TOKENS_PER_FRAME
                 for block in history],
                dtype=float,
            )
            distance = np.arange(count - 1, -1, -1, dtype=float)
            scores = (affinity + 1e-3) * np.exp(-decay * distance)
            scores[-1] = np.inf
            selected = sorted(np.argsort(-scores)[:budget].tolist())

        prompt_tokens = list(ctx_prefix)
        for index in selected:
            prompt_tokens.extend(history[index])
        prompt_lengths.append(len(prompt_tokens))

        tokens = sample_one(model, sampling_params, prompt_tokens)
        generated.append(tokens)

        block = pad_frame_tokens(tokens)
        block.extend(
            action_tokens_at(step + 2, context["all_actions"], context["action_ranges"],
                             context["total_frames"], context["device"])
        )
        history.append(block)
        progress("block_kv", step, context["num_frames"], prompt_lengths[-1])

    return {
        "tokens": generated,
        "prompt_tokens_per_step": prompt_lengths,
        "seconds": time.time() - started,
        "notes": {
            "block_budget": budget,
            "score": "token_affinity_with_latest_frame x exp(-decay * recency_distance)",
            "recency_decay": decay,
            "surrogate_for": "H2O / SnapKV token importance, lifted to block level",
            "attention_weights_available": False,
            "prompt_tokens_bound": CTX_PREFIX_LEN + BLOCK * budget,
            "pixel_space_round_trips": 0,
        },
    }


def psnr_from_mse(value):
    return float("inf") if value <= 0 else 10.0 * float(np.log10(1.0 / value))


def evaluate(prediction, ground_truth, token_rows):
    """Per-frame fidelity plus the two horizon-sensitive collapse diagnostics."""
    span = min(prediction.shape[0], ground_truth.shape[0])
    prediction = prediction[:span]
    ground_truth = ground_truth[:span]

    squared = (prediction - ground_truth).pow(2).mean(dim=(1, 2, 3))
    absolute = (prediction - ground_truth).abs().mean(dim=(1, 2, 3))
    activity = (prediction[1:] - prediction[:-1]).abs().mean(dim=(1, 2, 3))
    gt_activity = (ground_truth[1:] - ground_truth[:-1]).abs().mean(dim=(1, 2, 3))
    drift = (prediction - prediction[0:1]).abs().mean(dim=(1, 2, 3))
    gt_drift = (ground_truth - ground_truth[0:1]).abs().mean(dim=(1, 2, 3))

    tokens = np.array([pad_frame_tokens(row) for row in token_rows[:span]], dtype=np.int64)
    unique = [int(len(np.unique(row))) for row in tokens]
    repeat = (tokens[1:] == tokens[:-1]).mean(axis=1).tolist() if len(tokens) > 1 else []
    flat_tokens = tokens.reshape(-1)
    token_counts = np.bincount(flat_tokens, minlength=VISUAL_TOKEN_NUM)
    token_probabilities = token_counts[token_counts > 0].astype(np.float64)
    token_probabilities /= token_probabilities.sum()
    token_entropy_bits = float(-(token_probabilities * np.log2(token_probabilities)).sum())
    token_effective_codes = float(2.0 ** token_entropy_bits)
    global_unique_tokens = int((token_counts > 0).sum())
    late_token_rows = tokens[max(0, len(tokens) - 16):]
    late_counts = np.bincount(late_token_rows.reshape(-1), minlength=VISUAL_TOKEN_NUM)
    late_probabilities = late_counts[late_counts > 0].astype(np.float64)
    late_probabilities /= late_probabilities.sum()
    late_entropy_bits = float(-(late_probabilities * np.log2(late_probabilities)).sum())

    count = len(activity)
    early = slice(0, max(1, int(round(count * 0.15))))
    late = slice(min(count - 1, int(round(count * 0.70))), count)
    unique_array = np.array(unique, dtype=float)
    unique_early = slice(0, max(1, int(round(len(unique) * 0.15))))
    unique_late = slice(min(len(unique) - 1, int(round(len(unique) * 0.70))), len(unique))

    # Guard the motion ratio: when the ground-truth clip is nearly still the
    # denominator collapses and the ratio becomes meaningless. Reporting the
    # denominator lets downstream aggregation drop or reweight such windows.
    gt_late = float(gt_activity[late].mean())
    gt_early = float(gt_activity[early].mean())
    floor = 1e-3

    return {
        "frames_scored": int(span),
        "psnr_db_mean": round(psnr_from_mse(float(squared.mean())), 3),
        "mae_mean": round(float(absolute.mean()), 6),
        "per_frame_psnr_db": [round(psnr_from_mse(value), 3) for value in squared.tolist()],
        "per_frame_mae": [round(value, 6) for value in absolute.tolist()],
        "motion_preservation_early": round(
            float(activity[early].mean()) / max(gt_early, floor), 4),
        "motion_preservation_late": round(
            float(activity[late].mean()) / max(gt_late, floor), 4),
        "ground_truth_activity_late": round(gt_late, 6),
        "ground_truth_activity_early": round(gt_early, 6),
        "late_window_is_static": bool(gt_late < 5e-3),
        "predicted_activity_late": round(float(activity[late].mean()), 6),
        "token_retention": round(
            float(unique_array[unique_late].mean() / max(float(unique_array[unique_early].mean()), 1e-8)), 4),
        "unique_tokens_first": unique[0],
        "unique_tokens_last": unique[-1],
        "token_utilization": {
            "global_unique_codes": global_unique_tokens,
            "global_codebook_coverage": round(global_unique_tokens / VISUAL_TOKEN_NUM, 6),
            "entropy_bits": round(token_entropy_bits, 4),
            "normalized_entropy": round(token_entropy_bits / np.log2(VISUAL_TOKEN_NUM), 4),
            "effective_codes": round(token_effective_codes, 2),
            "mean_unique_codes_per_frame": round(float(np.mean(unique)), 3),
            "late_16_unique_codes": int((late_counts > 0).sum()),
            "late_16_entropy_bits": round(late_entropy_bits, 4),
            "mean_temporal_repeat_ratio": round(float(np.mean(repeat)) if repeat else 0.0, 4),
        },
        "drift_peak": round(float(drift.max()), 6),
        "drift_final": round(float(drift[-1]), 6),
        "ground_truth_drift_final": round(float(gt_drift[-1]), 6),
        "prediction_activity": [round(value, 6) for value in activity.tolist()],
        "ground_truth_activity": [round(value, 6) for value in gt_activity.tolist()],
        "prediction_drift_from_first": [round(value, 6) for value in drift.tolist()],
        "ground_truth_drift_from_first": [round(value, 6) for value in gt_drift.tolist()],
        "unique_tokens_per_frame": unique,
        "token_repeat_ratio_vs_previous_frame": [round(float(value), 4) for value in repeat],
    }


def stack_rows(row_tensors, panel_width):
    height = max(1, round(row_tensors[0].shape[-2] * panel_width / row_tensors[0].shape[-1]))
    resized = [
        F.interpolate(row, size=(height, panel_width), mode="bilinear", align_corners=False)
        for row in row_tensors
    ]
    length = min(row.shape[0] for row in resized)
    frames = []
    for index in range(length):
        frames.append(torch.cat([row[index] for row in resized], dim=-2).clamp(0.0, 1.0))
    return torch.stack(frames, dim=0)


def frames_to_gif_bytes(frames, duration=180):
    arrays = [
        (frame.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
        for frame in frames.detach().float().clamp(0.0, 1.0)
    ]
    buffer = io.BytesIO()
    imageio.mimsave(buffer, arrays, format="GIF", duration=duration, loop=0)
    return buffer.getvalue()


def keyframe_grid_png(row_tensors, count, panel_width=190):
    length = min(row.shape[0] for row in row_tensors)
    indices = sorted(set(np.linspace(0, length - 1, min(count, length)).round().astype(int).tolist()))
    height = max(1, round(row_tensors[0].shape[-2] * panel_width / row_tensors[0].shape[-1]))
    rows = []
    for row in row_tensors:
        selected = F.interpolate(row[indices], size=(height, panel_width),
                                 mode="bilinear", align_corners=False)
        rows.append(torch.cat(list(selected), dim=-1))
    grid = torch.cat(rows, dim=-2).clamp(0.0, 1.0)
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
    from vllm import LLM

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    requested_specs = [spec.strip() for spec in args.strategies.split(",") if spec.strip()]
    parsed_specs = [parse_strategy_spec(spec) for spec in requested_specs]
    unknown = [base for base, _, _ in parsed_specs if base not in STRATEGY_NAMES]
    if unknown:
        raise ValueError(f"Unknown strategies: {unknown}. Valid: {sorted(STRATEGY_NAMES)}")
    labels = [label for _, _, label in parsed_specs]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(f"Duplicate strategy specs: {duplicates}")
    requested = labels

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    project_dir = Path(__file__).resolve().parent
    action_ranges_path = Path(args.action_ranges)
    if not action_ranges_path.is_absolute():
        action_ranges_path = project_dir / action_ranges_path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    started = time.time()

    data = np.load(args.input_npz)
    images = data[args.image_key]
    actions = data[args.action_key]
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Expected images [T,H,W,3], got {images.shape}")
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected actions [T,{ACTION_DIM}], got {actions.shape}")
    if len(images) != len(actions):
        raise ValueError(f"Image/action length mismatch: {len(images)} vs {len(actions)}")
    episode_length = len(images)
    if args.context_index < 0 or args.context_index >= episode_length - 3:
        raise ValueError(
            f"Invalid context index {args.context_index} for episode length {episode_length}"
        )
    images = images[args.context_index:]
    actions = actions[args.context_index:]
    total_frames = len(images)
    if total_frames < 4:
        raise ValueError(f"Episode window too short: {total_frames} frames")
    num_frames = min(args.num_frames, total_frames - 2)
    if num_frames < 1:
        raise ValueError(f"Need at least one future frame, resolved {num_frames}")
    max_prompt_len = min(MAX_PROMPT_LEN, args.max_model_len - TOKENS_PER_FRAME)
    if max_prompt_len <= CTX_PREFIX_LEN + BLOCK:
        raise ValueError(
            f"max_model_len={args.max_model_len} leaves insufficient prompt budget"
        )

    pixels = torch.from_numpy(images).permute(0, 3, 1, 2).float()
    if pixels.max() > 1.5:
        pixels = pixels / 255.0
    pixels = F.interpolate(pixels, size=IMAGE_SIZE, mode="bilinear", align_corners=False)
    pixels = pixels.clamp(0.0, 1.0)
    all_actions = torch.from_numpy(actions).float()

    tokenizer_model_path = resolve_model_reference(args.tokenizer_model)
    world_model_path = resolve_model_reference(args.world_model)

    tokenizer_started = time.time()
    tokenizer = CompressiveVQModelFSQ.from_pretrained(tokenizer_model_path)
    tokenizer = tokenizer.to(device).eval()
    tokenizer_load_seconds = time.time() - tokenizer_started

    ctx_pixel = pixels[0]
    pair = torch.stack([ctx_pixel, pixels[1]], dim=0).unsqueeze(0).to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        ctx_tokens, dyn_tokens = tokenizer.tokenize(pair)
        header = tokenizer.detokenize(ctx_tokens, dyn_tokens).float().clamp(0.0, 1.0)
    header_frames = header[0]

    action_ranges = torch.load(str(action_ranges_path), map_location="cpu", weights_only=True).to(device)

    world_started = time.time()
    model = LLM(
        model=world_model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        enforce_eager=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
    )
    world_model_load_seconds = time.time() - world_started

    def progress(strategy, step, total, prompt_length):
        if step % 20 == 0 or step == total - 1:
            print(f"PROGRESS strategy={strategy} step={step + 1}/{total} "
                  f"prompt_tokens={prompt_length}", flush=True)

    context = {
        "prompt_parts": initial_prompt(ctx_tokens, dyn_tokens, all_actions, action_ranges, device),
        "sampling_params": build_sampling_params(args),
        "all_actions": all_actions,
        "action_ranges": action_ranges,
        "total_frames": total_frames,
        "num_frames": num_frames,
        "device": device,
        "ctx_tokens": ctx_tokens,
        "dyn_tokens": dyn_tokens,
        "ctx_pixel": ctx_pixel,
        "pixels": pixels,
        "tokenizer": tokenizer,
        "max_prompt_len": max_prompt_len,
    }

    runners = {
        "full_history": run_full_history,
        "sliding_window": run_sliding_window,
        "sliding_window_eq": run_sliding_window_eq,
        "uniform_sampling": run_uniform_sampling,
        "anchor_uniform": run_anchor_uniform,
        "stale_inject": run_stale_inject,
        "streaming_llm": run_streaming_llm,
        "block_kv": run_block_kv_selection,
        "recap": run_recap,
        "recap_motion": run_recap_motion,
    }

    def run_spec(base, overrides):
        """Invoke one runner, temporarily applying any inline overrides."""
        if not overrides:
            return runners[base](model, args, context, progress)
        saved = {key: getattr(args, key) for key in overrides}
        for key, value in overrides.items():
            setattr(args, key, value)
        try:
            return runners[base](model, args, context, progress)
        finally:
            for key, value in saved.items():
                setattr(args, key, value)

    ground_truth = pixels[: 2 + num_frames].to(device)
    truth_uint8 = (ground_truth.permute(0, 2, 3, 1).clamp(0, 1) * 255).round().to(
        torch.uint8).cpu().numpy()
    outcomes = {}
    for base, overrides, label in parsed_specs:
        print(f"STRATEGY_START {label}", flush=True)
        run = run_spec(base, overrides)
        decoded = decode_frames(tokenizer, ctx_tokens, run["tokens"], device)[0]
        sequence = torch.cat([header_frames, decoded], dim=0)
        prompts = run["prompt_tokens_per_step"]
        # Score generated futures only. The two observed/header frames are shown in
        # visualizations but excluded from fidelity metrics.
        metrics = evaluate(sequence[2:], ground_truth[2:], run["tokens"])

        # Robot trajectory / shape consistency. Segmentation is CPU-only and
        # cheap relative to generation, so it runs inline while the frames are
        # still in memory rather than requiring a second pass over saved video.
        prediction_uint8 = (sequence.permute(0, 2, 3, 1).clamp(0, 1) * 255).round().to(
            torch.uint8).cpu().numpy()
        try:
            trajectory = trajectory_metrics(prediction_uint8[2:], truth_uint8[2:])
        except Exception as error:  # keep the rollout usable if segmentation fails
            trajectory = {"error": f"{type(error).__name__}: {error}"}
        metrics["trajectory"] = trajectory

        outcomes[label] = {
            "sequence": sequence,
            "frames_uint8": prediction_uint8,
            "metrics": metrics,
            "base_strategy": base,
            "overrides": dict(overrides),
            "runtime": {
                "seconds": round(run["seconds"], 3),
                "seconds_per_frame": round(run["seconds"] / max(num_frames, 1), 4),
                "prompt_tokens_first_step": prompts[0],
                "prompt_tokens_last_step": prompts[-1],
                "prompt_tokens_max": max(prompts),
                "prompt_tokens_mean": round(float(np.mean(prompts)), 1),
            },
            "notes": run["notes"],
        }
        print(f"STRATEGY_DONE {label} psnr={metrics['psnr_db_mean']} "
              f"motion_late={metrics['motion_preservation_late']} "
              f"token_ret={metrics['token_retention']} "
              f"ndtw={trajectory.get('ndtw')} arm={trajectory.get('arm_score')} "
              f"iou={trajectory.get('shape_iou')} "
              f"seconds={outcomes[label]['runtime']['seconds']}", flush=True)

    # Deterministic lower bound: repeat the last observed real frame for every
    # future step. This makes it obvious whether a model is merely staying still.
    repeat_future = pixels[1:2].to(device).expand(num_frames, -1, -1, -1).clone()
    repeat_sequence = torch.cat([header_frames, repeat_future], dim=0)
    repeat_token = dyn_tokens[:, 0, :].reshape(-1).detach().cpu().tolist()
    repeat_tokens = [repeat_token for _ in range(num_frames)]
    repeat_metrics = evaluate(repeat_future, ground_truth[2:], repeat_tokens)
    repeat_uint8 = (repeat_sequence.permute(0, 2, 3, 1).clamp(0, 1) * 255).round().to(
        torch.uint8).cpu().numpy()
    try:
        repeat_trajectory = trajectory_metrics(repeat_uint8[2:], truth_uint8[2:])
    except Exception as error:
        repeat_trajectory = {"error": f"{type(error).__name__}: {error}"}
    repeat_metrics["trajectory"] = repeat_trajectory
    outcomes["repeat_frame"] = {
        "sequence": repeat_sequence,
        "frames_uint8": repeat_uint8,
        "metrics": repeat_metrics,
        "runtime": {
            "seconds": 0.0,
            "seconds_per_frame": 0.0,
            "prompt_tokens_first_step": 0,
            "prompt_tokens_last_step": 0,
            "prompt_tokens_max": 0,
            "prompt_tokens_mean": 0.0,
        },
        "notes": {
            "baseline": "repeat the last observed real frame (frame 1)",
            "generated": False,
        },
    }
    print(f"BASELINE_DONE repeat_frame psnr={repeat_metrics['psnr_db_mean']} ",
          f"ndtw={repeat_trajectory.get('ndtw')}", flush=True)

    # Order rows by the canonical strategy order, keeping any parameterized
    # variants (e.g. `stale_inject@k=4+src=generated`) next to their base name
    # and in the order the user requested them. `repeat_frame` is injected
    # directly into `outcomes` below without a `base_strategy` key, so fall back
    # to the label itself rather than assuming the key exists.
    base_order = {name: index for index, name in enumerate(STRATEGY_ROWS)}
    request_order = {label: index for index, (_, _, label) in enumerate(parsed_specs)}
    ordered = sorted(
        (label for label in outcomes if label != "repeat_frame"),
        key=lambda label: (
            base_order.get(outcomes[label].get("base_strategy", label), len(STRATEGY_ROWS)),
            request_order.get(label, 0),
        ),
    )
    row_names = ["ground_truth"] + ordered
    if "repeat_frame" in outcomes:
        row_names.append("repeat_frame")
    row_tensors = [
        ground_truth if name == "ground_truth" else outcomes[name]["sequence"]
        for name in row_names
    ]
    strip_png, strip_indices = keyframe_grid_png(row_tensors, args.strip_frames)
    stacked = stack_rows(row_tensors, args.preview_panel_width)
    comparison_gif = frames_to_gif_bytes(stacked)
    (output_dir / "compare_all_strategies.gif").write_bytes(comparison_gif)
    (output_dir / "keyframe_grid.png").write_bytes(strip_png)

    # Persist the decoded rollouts so downstream evaluation (VLM judging,
    # re-scoring with new metrics) never needs to re-run generation. Streamed
    # back through the job log as a single compressed archive.
    #
    # Every strategy is subsampled at the SAME indices, otherwise a judge would
    # be comparing different moments of the episode across strategies.
    if args.frames_keep > 0 and args.frames_keep < len(truth_uint8):
        keep_indices = np.unique(
            np.linspace(0, len(truth_uint8) - 1, args.frames_keep).round().astype(int))
    else:
        keep_indices = np.arange(len(truth_uint8))

    def archived(array):
        limited = keep_indices[keep_indices < len(array)]
        return array[limited]

    frames_buffer = io.BytesIO()
    np.savez_compressed(
        frames_buffer,
        ground_truth=archived(truth_uint8),
        frame_indices=keep_indices,
        **{name: archived(outcomes[name]["frames_uint8"]) for name in outcomes},
    )
    frames_payload = frames_buffer.getvalue()
    (output_dir / "frames.npz").write_bytes(frames_payload)
    print(f"FRAMES_ARCHIVE_BYTES {len(frames_payload)} frames={len(keep_indices)}", flush=True)

    result = {
        "status": "ok",
        "mode": "strategy_comparison",
        "source": str(args.input_npz),
        "resolved_assets": {
            "tokenizer_model": tokenizer_model_path,
            "world_model": world_model_path,
            "action_ranges": str(action_ranges_path.resolve()),
        },
        "host": socket.gethostname(),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "episode_length": int(episode_length),
        "context_index": int(args.context_index),
        "available_window_frames": int(total_frames),
        "generated_frames": int(num_frames),
        "metrics_scope": "generated future frames only; two observed/header frames excluded",
        "frame_alignment": "frame 0 = visual context, frame 1 = first real dynamic frame, "
                           "generation starts at frame 2",
        "sampling": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "seed": args.seed,
        },
        "config": {
            "max_model_len": args.max_model_len,
            "max_prompt_len": max_prompt_len,
            "ctx_prefix_len": CTX_PREFIX_LEN,
            "block_tokens": BLOCK,
            "recap_window_size_W": args.window_size_w,
            "recap_overlap_K": args.overlap_k,
            "recap_overlap_K_effective": False,
            "recap_anchor_frames": 1,
            "stale_blocks_k": args.stale_blocks_k,
            "stale_source": args.stale_source,
            "motion_anchor_refresh_interval": args.motion_anchor_refresh_interval,
            "motion_anchor_history": args.motion_anchor_history,
            "motion_anchor_min_novelty": args.motion_anchor_min_novelty,
            "motion_anchor_recency_weight": args.motion_anchor_recency_weight,
            "streaming_sink_tokens": args.streaming_sink_tokens,
            "block_kv_recency_decay": args.block_kv_recency_decay,
            "equal_block_budget": 1 + args.window_size_w,
        },
        "row_order": row_names,
        "keyframe_indices": strip_indices,
        "strategies": {
            name: {
                "metrics": payload["metrics"],
                "runtime": payload["runtime"],
                "notes": payload["notes"],
            }
            for name, payload in outcomes.items()
        },
        "tokenizer_load_seconds": round(tokenizer_load_seconds, 3),
        "world_model_load_seconds": round(world_model_load_seconds, 3),
        "total_seconds": round(time.time() - started, 3),
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("COMPARISON_RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)
    if args.emit_keyframe:
        emit_artifact("keyframe_grid.png", strip_png)
    if args.emit_frames:
        emit_artifact("frames.npz", frames_payload)
    if args.emit_preview:
        emit_artifact("compare_all_strategies.gif", comparison_gif)


if __name__ == "__main__":
    main()
