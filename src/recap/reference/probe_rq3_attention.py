#!/usr/bin/env python3
"""Block-level attention probe for the RQ3 mechanism figure.

Why a separate script: the rollout path uses vLLM, which does not expose
per-layer attention weights. This probe therefore loads the *same* checkpoint
with Hugging Face in eager mode and re-runs the two context schedules itself, so
every attention number comes from a real self-generated context rather than a
synthetic prompt. Generation here is HF-eager rather than vLLM, so treat the
output as a mechanism probe on identical weights, not as the exact rollout used
in the quantitative tables.

Two things this script is careful about:

1. Attention mass is aggregated per *block region* (ctx prefix / anchor /
   stale / recent) and reported as a fraction of the row, which already sums to
   one over keys. Per-token attention is not comparable across schedules because
   ReCAP's context is roughly four times shorter, so a raw per-token value is
   inflated purely by having fewer keys.
2. Full history is split into `stale` (everything older than the last W blocks)
   and `recent` (the last W blocks) using the same W as ReCAP, so the two
   schedules are decomposed on the same axis.
"""

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch

BLOCK = 93
TOKENS_PER_FRAME = 80
ACTION_DIM = 13
ACTION_BINS = 256
VISUAL_TOKEN_NUM = 4375
CTX_PREFIX_LEN = 1280
IMAGE_SIZE = (256, 320)
ARTIFACT_CHUNK_CHARS = 40000


def emit_artifact(name, payload):
    encoded = base64.b64encode(payload).decode("ascii")
    total = max(1, (len(encoded) + ARTIFACT_CHUNK_CHARS - 1) // ARTIFACT_CHUNK_CHARS)
    print(f"ARTIFACT_BEGIN name={name} chunks={total} bytes={len(payload)}", flush=True)
    for index in range(total):
        piece = encoded[index * ARTIFACT_CHUNK_CHARS:(index + 1) * ARTIFACT_CHUNK_CHARS]
        print(f"ARTIFACT_CHUNK name={name} idx={index} data={piece}", flush=True)
    print(f"ARTIFACT_END name={name}", flush=True)


def discretize_actions(actions, action_ranges):
    minimum, maximum = action_ranges[:, 0], action_ranges[:, 1]
    scaled = torch.clip((actions - minimum) / (maximum - minimum + 1e-8), 0, 1)
    return torch.floor(scaled * ACTION_BINS).to(torch.int32).clip(0, ACTION_BINS - 1)


def action_tokens_at(index, all_actions, action_ranges, total_frames, device):
    source = all_actions[index] if index < total_frames else all_actions[-1]
    tokens = discretize_actions(source.unsqueeze(0).to(device), action_ranges)
    return (tokens + VISUAL_TOKEN_NUM * 2)[0].detach().cpu().tolist()


def pad_frame_tokens(tokens):
    if len(tokens) < TOKENS_PER_FRAME:
        filler = tokens[-1] if tokens else 0
        return list(tokens) + [filler] * (TOKENS_PER_FRAME - len(tokens))
    return list(tokens[:TOKENS_PER_FRAME])


@torch.no_grad()
def generate_frame(model, prompt_tokens, generator, temperature, top_k, top_p, device):
    """Sample one 80-token visual frame, mirroring the rollout sampling params."""
    ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    past = None
    produced = []
    step_input = ids
    for _ in range(TOKENS_PER_FRAME):
        out = model(input_ids=step_input, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1].float()
        # Constrain to the visual vocabulary: the rollout asserts every generated
        # token is a visual token, so sampling outside that range would silently
        # diverge from the schedules being probed.
        logits[VISUAL_TOKEN_NUM:] = float("-inf")
        logits = logits / max(temperature, 1e-6)
        if top_k and top_k > 0:
            kth = torch.topk(logits, min(top_k, VISUAL_TOKEN_NUM)).values[-1]
            logits[logits < kth] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        if top_p and top_p < 1.0:
            order = torch.argsort(probs, descending=True)
            cumulative = torch.cumsum(probs[order], dim=0)
            cut = int(torch.searchsorted(cumulative, torch.tensor(top_p, device=device)).item()) + 1
            keep = order[:cut]
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask[keep] = True
            probs = torch.where(mask, probs, torch.zeros_like(probs))
            probs = probs / probs.sum()
        token = int(torch.multinomial(probs, 1, generator=generator).item())
        produced.append(token)
        step_input = torch.tensor([[token]], dtype=torch.long, device=device)
    return produced


def region_spans(schedule, prompt_len, block_count, window):
    """Map a prompt into (region, start, end) spans.

    `block_count` counts the non-anchor blocks currently in the prompt. For
    full history the oldest ones are labelled stale; ReCAP never holds any.
    """
    spans = [("ctx_prefix", 0, CTX_PREFIX_LEN)]
    cursor = CTX_PREFIX_LEN
    spans.append(("anchor", cursor, cursor + BLOCK))
    cursor += BLOCK
    stale_count = max(0, block_count - window)
    if stale_count:
        spans.append(("stale", cursor, cursor + stale_count * BLOCK))
        cursor += stale_count * BLOCK
    recent_count = block_count - stale_count
    if recent_count:
        spans.append(("recent", cursor, cursor + recent_count * BLOCK))
        cursor += recent_count * BLOCK
    if cursor != prompt_len:
        raise RuntimeError(f"{schedule}: span cursor {cursor} != prompt length {prompt_len}")
    return spans


@torch.no_grad()
def attention_by_region(model, prompt_tokens, spans, device):
    """One eager forward pass; aggregate the final query row per region."""
    ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    out = model(input_ids=ids, output_attentions=True, use_cache=False)
    layers = []
    for attn in out.attentions:
        # attn: [batch, heads, query, key]. The row that matters is the last
        # query position, i.e. the state that predicts the next visual token.
        row = attn[0, :, -1, :].float()
        total = row.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        row = row / total
        entry = {}
        for name, begin, end in spans:
            mass = row[:, begin:end].sum(dim=-1)
            entry[name] = {
                "mass_mean_over_heads": float(mass.mean()),
                "mass_std_over_heads": float(mass.std(unbiased=False)),
                "mass_max_head": float(mass.max()),
                "tokens": end - begin,
            }
        layers.append(entry)
    return layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--context-index", type=int, default=0)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--world-model", required=True)
    parser.add_argument("--action-ranges", required=True)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--window-size-w", type=int, default=6)
    parser.add_argument("--probe-steps", type=str, default="8,16,32,48,64")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="CALVIN")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM
    from ivideogpt.ctx_tokenizer import CompressiveVQModelFSQ

    device = torch.device("cuda:0")
    probe_steps = sorted({int(x) for x in args.probe_steps.split(",") if x.strip()})
    if max(probe_steps) > args.num_frames:
        raise ValueError(f"probe step {max(probe_steps)} exceeds num_frames {args.num_frames}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.input_npz)
    images = data["image"][args.context_index:]
    actions = data["action"][args.context_index:]
    total_frames = len(images)
    if total_frames < args.num_frames + 2:
        raise ValueError(f"window has {total_frames} frames, need {args.num_frames + 2}")

    pixels = torch.from_numpy(images).permute(0, 3, 1, 2).float()
    if pixels.max() > 1.5:
        pixels = pixels / 255.0
    pixels = F.interpolate(pixels, size=IMAGE_SIZE, mode="bilinear", align_corners=False).clamp(0, 1)
    all_actions = torch.from_numpy(actions).float()

    tokenizer = CompressiveVQModelFSQ.from_pretrained(args.tokenizer_model).to(device).eval()
    pair = torch.stack([pixels[0], pixels[1]], dim=0).unsqueeze(0).to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        ctx_tokens, dyn_tokens = tokenizer.tokenize(pair)
    action_ranges = torch.load(args.action_ranges, map_location="cpu", weights_only=True).to(device)

    ctx_prefix = (ctx_tokens.reshape(1, -1) + VISUAL_TOKEN_NUM)[0].detach().cpu().tolist()
    if len(ctx_prefix) != CTX_PREFIX_LEN:
        raise RuntimeError(f"ctx prefix length {len(ctx_prefix)} != {CTX_PREFIX_LEN}")
    anchor_block = pad_frame_tokens(dyn_tokens[:, 0, :].reshape(-1).detach().cpu().tolist())
    anchor_block += discretize_actions(all_actions[1].unsqueeze(0).to(device), action_ranges).add(
        VISUAL_TOKEN_NUM * 2)[0].detach().cpu().tolist()

    model = AutoModelForCausalLM.from_pretrained(
        args.world_model, dtype=torch.float32, attn_implementation="eager").to(device).eval()

    records = []
    for schedule in ("full_history", "recap"):
        generator = torch.Generator(device=device).manual_seed(args.seed)
        history = []
        for step in range(1, args.num_frames + 1):
            if schedule == "recap":
                blocks = history[-args.window_size_w:]
            else:
                blocks = list(history)
            prompt = list(ctx_prefix) + list(anchor_block)
            for block in blocks:
                prompt.extend(block)

            if step in probe_steps:
                spans = region_spans(schedule, len(prompt), len(blocks), args.window_size_w)
                layers = attention_by_region(model, prompt, spans, device)
                records.append({
                    "dataset": args.dataset,
                    "schedule": schedule,
                    "future_step": step,
                    "prompt_tokens": len(prompt),
                    "blocks_in_prompt": len(blocks) + 1,
                    "spans": [{"region": n, "begin": b, "end": e} for n, b, e in spans],
                    "layers": layers,
                })
                print(f"PROBE {schedule} step={step} prompt={len(prompt)} "
                      f"regions={[n for n, _, _ in spans]}", flush=True)

            tokens = generate_frame(model, prompt, generator, args.temperature,
                                    args.top_k, args.top_p, device)
            block = pad_frame_tokens(tokens)
            block.extend(action_tokens_at(step + 1, all_actions, action_ranges,
                                          total_frames, device))
            history.append(block)

    summary = {}
    for record in records:
        key = (record["schedule"], record["future_step"])
        per_region = {}
        for region in {name for layer in record["layers"] for name in layer}:
            values = [layer[region]["mass_mean_over_heads"]
                      for layer in record["layers"] if region in layer]
            per_region[region] = {
                "layer_mean": round(float(np.mean(values)), 6),
                "layer_std": round(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, 6),
                "layer_min": round(float(np.min(values)), 6),
                "layer_max": round(float(np.max(values)), 6),
                "layers": len(values),
            }
        summary[f"{key[0]}@step{key[1]}"] = {
            "prompt_tokens": record["prompt_tokens"],
            "blocks_in_prompt": record["blocks_in_prompt"],
            "regions": per_region,
        }

    payload = {
        "status": "ok",
        "dataset": args.dataset,
        "source": str(args.input_npz),
        "context_index": args.context_index,
        "num_frames": args.num_frames,
        "window_size_w": args.window_size_w,
        "probe_steps": probe_steps,
        "generation_backend": "huggingface-eager",
        "attention_note": "mass is a fraction of the final query row, aggregated per block "
                          "region and averaged over heads then layers; per-token attention "
                          "is not comparable across schedules with different context lengths",
        "layers": int(model.config.num_hidden_layers),
        "heads": int(model.config.num_attention_heads),
        "summary": summary,
        "records": records,
    }
    (output_dir / "attention_probe.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    emit_artifact("attention_probe.json", json.dumps(payload).encode("utf-8"))
    print("ATTENTION_PROBE_RESULT=" + json.dumps(
        {"summary": summary, "layers": payload["layers"], "heads": payload["heads"]},
        sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
