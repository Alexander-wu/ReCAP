#!/usr/bin/env python3
"""Single-stream RQ4 benchmark on one exclusive GPU.

This benchmark isolates context-length cost using real episode/tokenizer prompts.
The prompts at horizon H contain the static 1280-token context, the observed
anchor block, and H-1 real visual-action blocks. Full Context keeps all H blocks;
ReCAP keeps the anchor plus the last W blocks. Quality is not scored here (RQ2
already provides paired quality on model-generated rollouts); this job measures
world-model latency, logical KV footprint, tokenizer cost and device memory.

Main-paper numbers should use --no-prefix-cache. --prefix-cache is a deployment
sensitivity run: prompts are traversed in increasing horizon so vLLM may reuse
the natural prefix within one repeat, while the first context token is varied
between repeats to prevent accidental cross-repeat cache hits.
"""

import argparse
import base64
import hashlib
import json
import os
import platform
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TOKENS_PER_FRAME = 80
ACTION_DIM = 13
ACTION_BINS = 256
VISUAL_TOKEN_NUM = 4375
BLOCK = 93
CTX_PREFIX_LEN = 1280
IMAGE_SIZE = (256, 320)
ARTIFACT_CHUNK_CHARS = 40000


def emit_artifact(name, payload):
    encoded = base64.b64encode(payload).decode("ascii")
    total = max(1, (len(encoded) + ARTIFACT_CHUNK_CHARS - 1) // ARTIFACT_CHUNK_CHARS)
    print(f"ARTIFACT_BEGIN name={name} chunks={total} bytes={len(payload)}", flush=True)
    for index in range(total):
        print(f"ARTIFACT_CHUNK name={name} idx={index} "
              f"data={encoded[index*ARTIFACT_CHUNK_CHARS:(index+1)*ARTIFACT_CHUNK_CHARS]}",
              flush=True)
    print(f"ARTIFACT_END name={name}", flush=True)


def discretize_actions(actions, action_ranges):
    minimum, maximum = action_ranges[:, 0], action_ranges[:, 1]
    scaled = torch.clip((actions - minimum) / (maximum - minimum + 1e-8), 0, 1)
    return torch.floor(scaled * ACTION_BINS).to(torch.int32).clip(0, ACTION_BINS - 1)


def action_tokens(index, actions, ranges, device):
    index = min(index, len(actions) - 1)
    out = discretize_actions(actions[index:index + 1].to(device), ranges)
    return (out + VISUAL_TOKEN_NUM * 2)[0].detach().cpu().tolist()


def physical_gpu_selector():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    return visible or "0"


def gpu_memory_mib():
    selector = physical_gpu_selector()
    try:
        result = subprocess.run(
            ["nvidia-smi", "-i", selector, "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3,
            check=True)
        return float(result.stdout.strip().splitlines()[0])
    except Exception:
        return None


class MemoryMonitor:
    def __init__(self, interval=0.05):
        self.interval = interval
        self.values = []
        self.stop_event = threading.Event()
        self.thread = None

    def __enter__(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def _run(self):
        while not self.stop_event.is_set():
            value = gpu_memory_mib()
            if value is not None:
                self.values.append(value)
            self.stop_event.wait(self.interval)

    def __exit__(self, exc_type, exc, tb):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)
        value = gpu_memory_mib()
        if value is not None:
            self.values.append(value)

    @property
    def peak(self):
        return max(self.values) if self.values else None


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def prompt_variant(prompt, salt):
    """Keep token type valid while making cross-repeat prefixes distinct."""
    out = list(prompt)
    token = out[0]
    if VISUAL_TOKEN_NUM <= token < VISUAL_TOKEN_NUM * 2:
        out[0] = VISUAL_TOKEN_NUM + ((token - VISUAL_TOKEN_NUM + salt) % VISUAL_TOKEN_NUM)
    else:
        out[0] = int((token + salt) % (VISUAL_TOKEN_NUM * 2))
    return out


def theoretical_kv(config, tokens, dtype_bytes=2):
    layers = int(config.num_hidden_layers)
    kv_heads = int(getattr(config, "num_key_value_heads", config.num_attention_heads))
    head_dim = int(config.hidden_size // config.num_attention_heads)
    bytes_per_token = 2 * layers * kv_heads * head_dim * dtype_bytes
    return bytes_per_token, bytes_per_token * tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--context-index", type=int, default=0)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--world-model", required=True)
    parser.add_argument("--action-ranges", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--horizons", default="8,16,32,48,64,80,100")
    parser.add_argument("--window-size-w", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--prefix-cache", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    import transformers
    import vllm
    from vllm import LLM, SamplingParams
    from ivideogpt.ctx_tokenizer import CompressiveVQModelFSQ

    horizons = sorted({int(x) for x in args.horizons.split(",") if x.strip()})
    max_horizon = max(horizons)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")

    data = np.load(args.input_npz)
    images = data["image"][args.context_index:]
    actions_np = data["action"][args.context_index:]
    if len(images) < max_horizon + 2:
        raise ValueError(f"Need {max_horizon + 2} frames, found {len(images)}")
    pixels = torch.from_numpy(images[:max_horizon + 2]).permute(0, 3, 1, 2).float()
    if pixels.max() > 1.5:
        pixels = pixels / 255.0
    pixels = F.interpolate(pixels, size=IMAGE_SIZE, mode="bilinear", align_corners=False).clamp(0, 1)
    actions = torch.from_numpy(actions_np).float()
    ranges = torch.load(args.action_ranges, map_location="cpu", weights_only=True).to(device)

    tokenizer = CompressiveVQModelFSQ.from_pretrained(args.tokenizer_model).to(device).eval()
    torch.cuda.synchronize()
    tokenizer_memory_mib = gpu_memory_mib()

    # Benchmark one realistic 8-frame tokenizer clip separately from world-model latency.
    encode_samples = []
    decode_samples = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        clip = pixels[:8].unsqueeze(0).to(device)
        for _ in range(2):
            c, d = tokenizer.tokenize(clip)
            _ = tokenizer.detokenize(c, d)
        for _ in range(args.repeats):
            torch.cuda.synchronize(); start = time.perf_counter_ns()
            c, d = tokenizer.tokenize(clip)
            torch.cuda.synchronize(); encode_samples.append((time.perf_counter_ns() - start) / 1e6)
            torch.cuda.synchronize(); start = time.perf_counter_ns()
            _ = tokenizer.detokenize(c, d)
            torch.cuda.synchronize(); decode_samples.append((time.perf_counter_ns() - start) / 1e6)

    # Encode every real frame against the same context. Batched pairs keep this
    # preparation out of the world-model timing region.
    ctx_tokens = None
    visual_frames = []
    chunk = 8
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for begin in range(1, max_horizon + 1, chunk):
            end = min(max_horizon + 1, begin + chunk)
            batch = torch.stack([
                torch.stack([pixels[0], pixels[index]], dim=0)
                for index in range(begin, end)
            ], dim=0).to(device)
            c, d = tokenizer.tokenize(batch)
            if ctx_tokens is None:
                ctx_tokens = c[:1].detach().cpu()
            visual_frames.extend(d[:, 0, :].reshape(len(batch), -1).detach().cpu().tolist())

    ctx_prefix = (ctx_tokens.reshape(-1) + VISUAL_TOKEN_NUM).tolist()
    if len(ctx_prefix) != CTX_PREFIX_LEN:
        raise RuntimeError(f"ctx prefix {len(ctx_prefix)} != {CTX_PREFIX_LEN}")
    blocks = []
    for frame_index, visual in enumerate(visual_frames, start=1):
        block = list(visual[:TOKENS_PER_FRAME])
        block.extend(action_tokens(frame_index, actions, ranges, device))
        if len(block) != BLOCK:
            raise RuntimeError(f"block {frame_index} has {len(block)} tokens")
        blocks.append(block)

    prompts = {}
    logical_prompt_lengths = {}
    for horizon in horizons:
        full_blocks = blocks[:horizon]
        recap_blocks = [blocks[0]] + blocks[max(1, horizon - args.window_size_w):horizon]
        for method, selected in (("full_history", full_blocks), ("recap", recap_blocks)):
            prompt = list(ctx_prefix)
            for block in selected:
                prompt.extend(block)
            logical_prompt_lengths[(method, horizon)] = len(prompt)
            # Mirror infer_compare.py exactly: once Full Context exceeds the
            # 7900-token prompt cap, preserve the static context prefix and keep
            # the most recent 6620 tokens. This is reported as truncated Full,
            # never as an unbounded context measurement.
            if method == "full_history" and len(prompt) > 7900:
                prompt = prompt[:CTX_PREFIX_LEN] + prompt[-(7900 - CTX_PREFIX_LEN):]
            prompts[(method, horizon)] = prompt

    # Release tokenizer allocations before creating the persistent vLLM engine.
    del tokenizer, pixels, ranges, c, d, clip
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    before_engine_mib = gpu_memory_mib()

    llm = LLM(
        model=args.world_model,
        dtype="bfloat16",
        max_model_len=8192,
        enforce_eager=True,
        enable_prefix_caching=args.prefix_cache,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=64,
    )
    sampling = SamplingParams(temperature=0.0, ignore_eos=True,
                              max_tokens=TOKENS_PER_FRAME, seed=42)
    engine_idle_mib = gpu_memory_mib()
    config = llm.llm_engine.model_config.hf_config
    bytes_per_token, _ = theoretical_kv(config, 1)

    # Warm both short and long paths. Warmup prompts get unique salts and are not
    # reused by measured repeats.
    for index in range(args.warmups):
        for key in (("recap", max(horizons)), ("full_history", max(h for h in horizons if h <= 64))):
            prompt = prompt_variant(prompts[key], 1000 + 10 * index + len(key[0]))
            llm.generate([{"prompt_token_ids": prompt}], sampling_params=sampling, use_tqdm=False)

    raw = []
    peak_global = engine_idle_mib
    for repeat in range(args.repeats):
        # APC-on traverses natural prefixes in ascending horizon. APC-off uses an
        # alternating ABBA order to reduce drift/order bias.
        if args.prefix_cache:
            sequence = [(m, h) for m in ("full_history", "recap") for h in horizons]
            if repeat % 2:
                sequence = [(m, h) for m in ("recap", "full_history") for h in horizons]
        else:
            left = [("full_history", h) for h in horizons]
            right = [("recap", h) for h in reversed(horizons)]
            sequence = left + right if repeat % 2 == 0 else right + left

        for order_index, (method, horizon) in enumerate(sequence):
            prompt = prompt_variant(prompts[(method, horizon)], repeat + 1)
            prompt_hash = hashlib.sha1(np.asarray(prompt, dtype=np.int32).tobytes()).hexdigest()
            before_mib = gpu_memory_mib()
            with MemoryMonitor() as monitor:
                start_ns = time.perf_counter_ns()
                outputs = llm.generate([{"prompt_token_ids": prompt}],
                                       sampling_params=sampling, use_tqdm=False)
                end_ns = time.perf_counter_ns()
            output_tokens = len(outputs[0].outputs[0].token_ids)
            if output_tokens != TOKENS_PER_FRAME:
                raise RuntimeError(f"Generated {output_tokens} tokens")
            after_mib = gpu_memory_mib()
            peak = monitor.peak
            if peak is not None:
                peak_global = peak if peak_global is None else max(peak_global, peak)
            _, kv_prompt = theoretical_kv(config, len(prompt))
            _, kv_with_output = theoretical_kv(config, len(prompt) + output_tokens)
            raw.append({
                "dataset": args.dataset,
                "case_id": args.case_id,
                "cache_mode": "on" if args.prefix_cache else "off",
                "method": method,
                "horizon": horizon,
                "repeat": repeat,
                "order_index": order_index,
                "prompt_tokens": len(prompt),
                "logical_prompt_tokens_before_truncation": logical_prompt_lengths[(method, horizon)],
                "prompt_sha1": prompt_hash,
                "output_tokens": output_tokens,
                "latency_ms": round((end_ns - start_ns) / 1e6, 6),
                "ms_per_output_token": round((end_ns - start_ns) / 1e6 / output_tokens, 6),
                "visual_tokens_per_second": round(output_tokens / ((end_ns - start_ns) / 1e9), 6),
                "theoretical_kv_prompt_mib": round(kv_prompt / 2**20, 6),
                "theoretical_kv_with_output_mib": round(kv_with_output / 2**20, 6),
                "device_memory_before_mib": before_mib,
                "device_memory_peak_mib": peak,
                "device_memory_after_mib": after_mib,
                "truncated_full_context": (
                    method == "full_history"
                    and logical_prompt_lengths[(method, horizon)] > len(prompt)
                ),
            })
            print(f"RQ4_STEP dataset={args.dataset} cache={'on' if args.prefix_cache else 'off'} "
                  f"method={method} H={horizon} repeat={repeat} "
                  f"tokens={len(prompt)} latency_ms={raw[-1]['latency_ms']}", flush=True)

    summary = []
    for method in ("full_history", "recap"):
        for horizon in horizons:
            selected = [r for r in raw if r["method"] == method and r["horizon"] == horizon]
            lat = [r["latency_ms"] for r in selected]
            summary.append({
                "dataset": args.dataset,
                "case_id": args.case_id,
                "cache_mode": "on" if args.prefix_cache else "off",
                "method": method,
                "horizon": horizon,
                "repeats": len(selected),
                "prompt_tokens": selected[0]["prompt_tokens"],
                "logical_prompt_tokens_before_truncation": selected[0]["logical_prompt_tokens_before_truncation"],
                "latency_ms_mean": round(float(np.mean(lat)), 6),
                "latency_ms_p50": round(percentile(lat, 50), 6),
                "latency_ms_p95": round(percentile(lat, 95), 6),
                "latency_ms_std": round(float(np.std(lat, ddof=1)) if len(lat) > 1 else 0.0, 6),
                "visual_tokens_per_second_p50": round(TOKENS_PER_FRAME / (percentile(lat, 50) / 1000), 6),
                "theoretical_kv_prompt_mib": selected[0]["theoretical_kv_prompt_mib"],
                "theoretical_kv_with_output_mib": selected[0]["theoretical_kv_with_output_mib"],
                "truncated_full_context": selected[0]["truncated_full_context"],
            })

    metadata = {
        "status": "ok",
        "dataset": args.dataset,
        "case_id": args.case_id,
        "source": args.input_npz,
        "context_index": args.context_index,
        "horizons": horizons,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "cache_mode": "on" if args.prefix_cache else "off",
        "world_model_only_timing": True,
        "tokenizer_clip_frames": 8,
        "tokenizer_encode_ms_p50": round(percentile(encode_samples, 50), 6),
        "tokenizer_decode_ms_p50": round(percentile(decode_samples, 50), 6),
        "tokenizer_encode_ms_samples": encode_samples,
        "tokenizer_decode_ms_samples": decode_samples,
        "kv_formula": "2 * layers * kv_heads * head_dim * bytes_per_element",
        "kv_bytes_per_token": bytes_per_token,
        "kv_excludes": ["allocator metadata", "block rounding", "cache sharing", "activations", "workspace"],
        "memory_note": "device memory is NVML total for the exclusive GPU; vLLM preallocates a KV pool, "
                       "so peak-minus-idle is not interpreted as active KV bytes",
        "gpu_memory_mib": {
            "tokenizer_loaded": tokenizer_memory_mib,
            "before_engine": before_engine_mib,
            "engine_idle": engine_idle_mib,
            "peak_workload": peak_global,
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "vllm": vllm.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "model": {
            "layers": int(config.num_hidden_layers),
            "attention_heads": int(config.num_attention_heads),
            "kv_heads": int(getattr(config, "num_key_value_heads", config.num_attention_heads)),
            "hidden_size": int(config.hidden_size),
            "head_dim": int(config.hidden_size // config.num_attention_heads),
            "dtype": "bfloat16",
        },
        "raw": raw,
        "summary": summary,
    }
    name = f"rq4_single_{args.dataset.lower().replace('-', '_')}_{'cache_on' if args.prefix_cache else 'cache_off'}.json"
    payload = json.dumps(metadata, indent=2).encode("utf-8")
    (output_dir / name).write_bytes(payload)
    emit_artifact(name, payload)
    print("RQ4_SINGLE_RESULT=" + json.dumps({
        "dataset": args.dataset,
        "case_id": args.case_id,
        "cache_mode": metadata["cache_mode"],
        "summary": summary,
        "gpu_memory_mib": metadata["gpu_memory_mib"],
        "software": metadata["software"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
