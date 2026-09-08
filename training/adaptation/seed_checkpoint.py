#!/usr/bin/env python
"""Turn a released HuggingFace-style world model into an accelerate state dir,
so the *unmodified* train_vgpt.py can fine-tune from it.

Why this exists
---------------
train_vgpt.py has exactly one live weight-loading path: `--resume_from_checkpoint`,
which calls `accelerator.load_state(path)`. That path works - the earlier claim
that the repo cannot fine-tune was wrong. What it needs is an accelerate *state
directory*, and a released checkpoint is not one:

    released                       accelerate state dir
    --------                       --------------------
    config.json                    model.safetensors
    model.safetensors              optimizer.bin
    generation_config.json         scheduler.bin
    tokenizer.json                 random_states_0.pkl

Measured on the cluster (accelerate 1.9.0, transformers 4.57.1):
  * pointing load_state at the released dir  -> FileNotFoundError: optimizer.bin
  * the weights themselves are a perfect fit -> load_state_dict(strict=True)
    accepts all 111 tensors, 0 missing / 0 unexpected

So the only missing pieces are the optimizer/scheduler/RNG side files. This
script builds the model, loads the released weights into it, and lets accelerate
write a complete state directory with save_state().

The optimizer state written here is fresh (zero moments). That is the correct
thing for fine-tuning onto a new dataset: inheriting Adam moments accumulated on
RT-1 would apply RT-1's curvature estimate to CALVIN gradients.

The output directory is named `checkpoint_<N>` because train_vgpt.py parses the
step count out of the directory name:

    resume_step = int(training_difference.replace("checkpoint_", "")) * grad_accum

`checkpoint_0` therefore starts the LR schedule from the beginning, which is what
fine-tuning wants. A non-zero name would silently skip that many steps of the
schedule *and* that many batches of the first epoch.

Round trip is verified before exit: a freshly initialised model is loaded from
the written directory and every tensor is compared against the released
checkpoint. Without that check a partially written directory looks identical to a
working one until training silently starts from noise.

Usage (on the cluster, where the weights are cached):
    python seed_checkpoint.py \
        --released ./weights/rt1_world \
        --config configs/vgpt/ctx_llama_small.json \
        --output ./checkpoints/rt1_seed
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, get_scheduler

# Files accelerate's load_state() looks for. Checked against
# accelerate.utils.constants on the cluster: SAFE_WEIGHTS_NAME is
# "model.safetensors", which is why the released file name already matches.
REQUIRED = ("model.safetensors", "model_1.safetensors", "optimizer.bin",
            "scheduler.bin", "random_states_0.pkl")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--released", required=True,
                        help="released world model dir (config.json + model.safetensors)")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]),
                        help="iVideoGPT repo; needed to construct the Evaluator "
                             "that upstream registers as a second model")
    parser.add_argument("--config", required=True,
                        help="model config train_vgpt.py will use, e.g. "
                             "configs/vgpt/ctx_llama_small.json")
    parser.add_argument("--output", required=True,
                        help="parent dir; a checkpoint_<step> subdir is created inside")
    parser.add_argument("--step", type=int, default=0,
                        help="step number in the directory name. Keep 0 for "
                             "fine-tuning so the LR schedule restarts (default: 0)")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="must match the initial fine-tuning LR")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler", default="constant_with_warmup")
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument("--training-steps", type=int, default=1000000,
                        help="scheduler construction horizon")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing checkpoint dir")
    return parser.parse_args()


def load_released_weights(config_path, released_dir):
    """Build the model train_vgpt.py builds, then load the released weights.

    strict=True is deliberate. A silent partial load is indistinguishable from a
    successful one in the logs, and the failure mode is a model that trains from
    noise while reporting a plausible loss.
    """
    config = AutoConfig.from_pretrained(str(config_path))
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)

    weights_file = released_dir / "model.safetensors"
    if not weights_file.exists():
        raise FileNotFoundError(f"no model.safetensors in {released_dir}")
    reference = load_file(str(weights_file))

    # Confirm the fresh init actually differs, so the post-load comparison
    # cannot pass trivially.
    probe_key = next(k for k in reference
                     if k.endswith("self_attn.q_proj.weight"))
    fresh_differs = not torch.allclose(model.state_dict()[probe_key],
                                       reference[probe_key].float())

    info = model.load_state_dict(reference, strict=True)
    missing = list(getattr(info, "missing_keys", []))
    unexpected = list(getattr(info, "unexpected_keys", []))
    if missing or unexpected:
        raise RuntimeError(
            f"released checkpoint does not fit this config.\n"
            f"  missing:    {missing[:6]}\n"
            f"  unexpected: {unexpected[:6]}\n"
            f"Check that --config matches the released config.json.")

    return model, reference, config, fresh_differs


def build_training_state(model, lr, weight_decay, scheduler_name,
                         warmup_steps, training_steps):
    """Mirror train_vgpt.py's optimizer and scheduler structure exactly.

    Even with embed_no_wd=False, upstream creates *two* optimizer groups; the
    second is empty. A seed checkpoint made from AdamW(model.parameters()) has
    one group and fails during resume with "different number of parameter
    groups" before the first batch. The scheduler must also have the same class
    as the real run.
    """
    groups = [
        {"params": [parameter for _, parameter in model.named_parameters()],
         "weight_decay": weight_decay},
        {"params": [], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=lr)
    scheduler = get_scheduler(
        name=scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=training_steps)
    return optimizer, scheduler


def build_evaluator(repo):
    """Construct the second model registered by upstream train_vgpt.py.

    `accelerator.prepare(model, evaluator, optimizer, ...)` makes Evaluator part
    of every accelerate checkpoint. A seed containing only the transformer loads
    model.safetensors, then fails looking for pytorch_model_1.bin. Reproducing
    this object is therefore required even though it has no optimizer role.
    """
    repo = str(Path(repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from ivideogpt.utils.video_metric import Evaluator
    return Evaluator(i3d_path=None, max_batchsize=None)


def verify_round_trip(destination, config, reference, args):
    """Load the written directory into a fresh model and compare every tensor."""
    fresh = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    evaluator = build_evaluator(args.repo)
    optimizer, scheduler = build_training_state(
        fresh, args.lr, args.weight_decay, args.scheduler,
        args.warmup_steps, args.training_steps)
    accelerator = Accelerator()
    fresh, evaluator, optimizer, scheduler = accelerator.prepare(
        fresh, evaluator, optimizer, scheduler)
    accelerator.load_state(str(destination))

    live = accelerator.unwrap_model(fresh).state_dict()
    matched, mismatched = 0, []
    for key, tensor in reference.items():
        if key not in live:
            mismatched.append(f"{key} (absent)")
            continue
        if torch.allclose(live[key].cpu().float(), tensor.float()):
            matched += 1
        else:
            mismatched.append(key)
    return matched, mismatched


def main():
    args = parse_args()
    released = Path(args.released).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    destination = Path(args.output).expanduser().resolve() / f"checkpoint_{args.step}"

    if not released.is_dir():
        sys.exit(f"released dir not found: {released}")
    if not config_path.is_file():
        sys.exit(f"config not found: {config_path}")

    if destination.exists():
        if not args.overwrite:
            sys.exit(f"{destination} exists; pass --overwrite to replace it")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    print(f"released weights : {released}")
    print(f"model config     : {config_path}")
    print(f"writing state to : {destination}")

    model, reference, config, fresh_differs = load_released_weights(
        config_path, released)
    print(f"  loaded {len(reference)} tensors, strict=True, 0 missing / 0 unexpected")
    if not fresh_differs:
        # Not fatal, but it means the comparison below proves nothing, so say so.
        print("  WARNING: fresh init already equals the checkpoint; the "
              "round-trip check below is vacuous")

    evaluator = build_evaluator(args.repo)
    optimizer, scheduler = build_training_state(
        model, args.lr, args.weight_decay, args.scheduler,
        args.warmup_steps, args.training_steps)
    accelerator = Accelerator()
    model, evaluator, optimizer, scheduler = accelerator.prepare(
        model, evaluator, optimizer, scheduler)
    accelerator.save_state(str(destination))

    written = sorted(p.name for p in destination.iterdir())
    print(f"  wrote: {written}")
    absent = [name for name in REQUIRED if not (destination / name).exists()]
    if absent:
        sys.exit(f"accelerate did not write {absent}; load_state would fail")

    matched, mismatched = verify_round_trip(destination, config, reference, args)
    print(f"  round trip: {matched}/{len(reference)} tensors match the release")
    if mismatched:
        sys.exit(f"round trip failed for {len(mismatched)} tensors, "
                 f"e.g. {mismatched[:5]}")

    size_mib = sum(p.stat().st_size for p in destination.rglob("*")
                   if p.is_file()) / 2**20
    manifest = {
        "released": str(released),
        "config": str(config_path),
        "checkpoint": str(destination),
        "tensors": len(reference),
        "round_trip_exact": True,
        "files": written,
        "size_MiB": round(size_mib, 1),
        "optimizer_state": "fresh (zero moments) - correct for fine-tuning",
        "optimizer_groups": 2,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "warmup_steps": args.warmup_steps,
        "training_steps": args.training_steps,
    }
    (destination.parent / "seed_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"  size: {size_mib:.1f} MiB")
    print("\nOK. Fine-tune from the released weights with the unmodified repo:")
    print(f"    accelerate launch train_vgpt.py \\")
    print(f"        --resume_from_checkpoint {destination} \\")
    print(f"        --config_name {config_path.name} ...")


if __name__ == "__main__":
    main()
