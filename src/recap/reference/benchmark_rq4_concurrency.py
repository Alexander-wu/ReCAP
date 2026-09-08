#!/usr/bin/env python3
"""RQ4 snapshot throughput/concurrency benchmark on one exclusive GPU.

This is a systems microbenchmark, not a quality experiment. It submits batches
of distinct token-typed prompts at the exact H64 lengths observed for Full
Context (7232) and ReCAP W=6 (1931), asks for 80 visual tokens per sequence, and
reports batch wall time, aggregate visual-token throughput, device-memory peak,
and the first failed/OOM concurrency. Prefix caching is off by default so prompt
length, not accidental cross-request reuse, drives the result.
"""

import argparse
import base64
import json
import os
import platform
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch

VISUAL_TOKEN_NUM = 4375
TOKENS_PER_FRAME = 80
ACTION_TOKEN_BASE = VISUAL_TOKEN_NUM * 2
VOCAB_SIZE = 9008
ARTIFACT_CHUNK_CHARS = 40000


def emit_artifact(name, payload):
    encoded = base64.b64encode(payload).decode("ascii")
    chunks = max(1, (len(encoded) + ARTIFACT_CHUNK_CHARS - 1) // ARTIFACT_CHUNK_CHARS)
    print(f"ARTIFACT_BEGIN name={name} chunks={chunks} bytes={len(payload)}", flush=True)
    for index in range(chunks):
        print(f"ARTIFACT_CHUNK name={name} idx={index} "
              f"data={encoded[index*ARTIFACT_CHUNK_CHARS:(index+1)*ARTIFACT_CHUNK_CHARS]}", flush=True)
    print(f"ARTIFACT_END name={name}", flush=True)


def gpu_memory_mib():
    selector = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0"
    try:
        run = subprocess.run(["nvidia-smi", "-i", selector, "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=3, check=True)
        return float(run.stdout.strip().splitlines()[0])
    except Exception:
        return None


class Monitor:
    def __init__(self):
        self.values = []
        self.stop = threading.Event()
        self.thread = None

    def __enter__(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start(); return self

    def run(self):
        while not self.stop.is_set():
            value = gpu_memory_mib()
            if value is not None: self.values.append(value)
            self.stop.wait(0.05)

    def __exit__(self, *args):
        self.stop.set(); self.thread.join(timeout=2)
        value = gpu_memory_mib()
        if value is not None: self.values.append(value)

    @property
    def peak(self):
        return max(self.values) if self.values else None


def token_typed_prompt(length, request_id, seed=20260904):
    """Construct a valid, distinct prompt with the same token-type proportions."""
    rng = np.random.default_rng(seed + request_id * 1009 + length)
    prefix_len = min(1280, length)
    prefix = rng.integers(VISUAL_TOKEN_NUM, VISUAL_TOKEN_NUM * 2,
                          size=prefix_len, dtype=np.int64).tolist()
    rest = []
    while prefix_len + len(rest) < length:
        remaining = length - prefix_len - len(rest)
        visual_n = min(TOKENS_PER_FRAME, remaining)
        rest.extend(rng.integers(0, VISUAL_TOKEN_NUM, size=visual_n,
                                 dtype=np.int64).tolist())
        remaining = length - prefix_len - len(rest)
        action_n = min(13, remaining)
        rest.extend(rng.integers(ACTION_TOKEN_BASE, min(VOCAB_SIZE, ACTION_TOKEN_BASE + 256),
                                 size=action_n, dtype=np.int64).tolist())
    prompt = prefix + rest
    prompt[0] = VISUAL_TOKEN_NUM + (request_id % VISUAL_TOKEN_NUM)
    return prompt[:length]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-model", required=True)
    parser.add_argument("--concurrency", default="1,2,4,8,16,32")
    parser.add_argument("--prompt-specs", default="full_history:7232,recap:1931",
                        help="comma-separated label:length pairs; also supports W-sweep microbenchmarks")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--prefix-cache", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    import transformers
    import vllm
    from vllm import LLM, SamplingParams

    levels = sorted({int(x) for x in args.concurrency.split(",") if x.strip()})
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    specs = {}
    for item in args.prompt_specs.split(","):
        label, sep, length = item.strip().partition(":")
        if not sep or not label or not length:
            raise ValueError(f"Invalid prompt spec: {item}")
        specs[label] = int(length)
    if not specs:
        raise ValueError("No prompt specs")
    before_engine = gpu_memory_mib()
    llm = LLM(model=args.world_model, dtype="bfloat16", max_model_len=8192,
              enforce_eager=True, enable_prefix_caching=args.prefix_cache,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_num_seqs=max(levels))
    sampling = SamplingParams(temperature=0.0, max_tokens=TOKENS_PER_FRAME,
                              ignore_eos=True, seed=42)
    engine_idle = gpu_memory_mib()

    for method, length in specs.items():
        for index in range(args.warmups):
            prompt = token_typed_prompt(length, 900000 + index + length)
            llm.generate([{"prompt_token_ids": prompt}], sampling_params=sampling, use_tqdm=False)

    raw = []
    failed = []
    request_counter = 0
    method_order = list(specs)
    for repeat in range(args.repeats):
        methods = method_order if repeat % 2 == 0 else list(reversed(method_order))
        concurrencies = levels if repeat % 2 == 0 else list(reversed(levels))
        for concurrency in concurrencies:
            for method in methods:
                length = specs[method]
                prompts = []
                for _ in range(concurrency):
                    prompts.append({"prompt_token_ids": token_typed_prompt(length, request_counter)})
                    request_counter += 1
                before = gpu_memory_mib()
                try:
                    with Monitor() as monitor:
                        start = time.perf_counter_ns()
                        outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
                        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                    produced = sum(len(item.outputs[0].token_ids) for item in outputs)
                    if produced != concurrency * TOKENS_PER_FRAME:
                        raise RuntimeError(f"expected {concurrency*TOKENS_PER_FRAME}, got {produced}")
                    after = gpu_memory_mib()
                    row = {
                        "cache_mode": "on" if args.prefix_cache else "off",
                        "method": method,
                        "prompt_tokens_per_request": length,
                        "concurrency": concurrency,
                        "repeat": repeat,
                        "batch_wall_ms": round(elapsed_ms, 6),
                        "sequences_per_second": round(concurrency / (elapsed_ms / 1000), 6),
                        "visual_tokens_per_second": round(produced / (elapsed_ms / 1000), 6),
                        "amortized_ms_per_sequence": round(elapsed_ms / concurrency, 6),
                        "device_memory_before_mib": before,
                        "device_memory_peak_mib": monitor.peak,
                        "device_memory_after_mib": after,
                        "status": "ok",
                    }
                    raw.append(row)
                    print(f"RQ4_CONC method={method} C={concurrency} repeat={repeat} "
                          f"wall_ms={elapsed_ms:.3f} tps={row['visual_tokens_per_second']:.2f}",
                          flush=True)
                except Exception as error:
                    failed.append({"method": method, "concurrency": concurrency,
                                   "repeat": repeat,
                                   "error": f"{type(error).__name__}: {error}"})
                    print(f"RQ4_CONC_FAIL method={method} C={concurrency} {error}", flush=True)

    summary = []
    for method in specs:
        for concurrency in levels:
            selected = [r for r in raw if r["method"] == method and r["concurrency"] == concurrency]
            if not selected:
                continue
            summary.append({
                "cache_mode": "on" if args.prefix_cache else "off",
                "method": method,
                "prompt_tokens_per_request": specs[method],
                "concurrency": concurrency,
                "successful_repeats": len(selected),
                "visual_tokens_per_second_mean": round(float(np.mean([r["visual_tokens_per_second"] for r in selected])), 6),
                "visual_tokens_per_second_p50": round(float(np.median([r["visual_tokens_per_second"] for r in selected])), 6),
                "batch_wall_ms_p50": round(float(np.median([r["batch_wall_ms"] for r in selected])), 6),
                "batch_wall_ms_p95": round(float(np.percentile([r["batch_wall_ms"] for r in selected], 95)), 6),
                "amortized_ms_per_sequence_p50": round(float(np.median([r["amortized_ms_per_sequence"] for r in selected])), 6),
                "peak_device_memory_mib": max(r["device_memory_peak_mib"] for r in selected
                                              if r["device_memory_peak_mib"] is not None),
            })

    result = {
        "status": "ok",
        "benchmark": "snapshot concurrency microbenchmark",
        "prompt_source": "distinct token-typed synthetic prompts at explicitly specified lengths",
        "prompt_specs": specs,
        "quality_claim": False,
        "cache_mode": "on" if args.prefix_cache else "off",
        "concurrency_levels": levels,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "engine_memory_mib": {"before": before_engine, "idle": engine_idle},
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "torch_cuda": torch.version.cuda, "transformers": transformers.__version__,
                     "vllm": vllm.__version__, "gpu": torch.cuda.get_device_name(0)},
        "summary": summary,
        "raw": raw,
        "failed": failed,
    }
    name = f"rq4_concurrency_{'cache_on' if args.prefix_cache else 'cache_off'}.json"
    payload = json.dumps(result, indent=2).encode("utf-8")
    (output_dir / name).write_bytes(payload)
    emit_artifact(name, payload)
    print("RQ4_CONCURRENCY_RESULT=" + json.dumps({"summary": summary, "failed": failed,
          "software": result["software"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
