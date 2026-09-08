#!/usr/bin/env python3
"""Aggregate a W sweep produced by run_wsweep.sh.

The sweep answers one question: does the frozen anchor block buy anything, and
if so at which recent-window size W? At W=6 every bounded schedule keeps full
token diversity, so the comparison saturates. Small W is where a thin recent
buffer might fail and the anchor might matter.

The headline comparison is ReCAP vs sliding_window_eq, because those two spend
an identical token budget and differ only in how one block is spent: frozen
first real frame versus one more recent frame.
"""
import argparse
import json
from pathlib import Path

import numpy as np

DISPLAY = {
    "full_history": "Full Context",
    "sliding_window": "Sliding Window (W blocks)",
    "sliding_window_eq": "Sliding Window (W+1, equal budget)",
    "uniform_sampling": "Uniform Sampling",
    "streaming_llm": "StreamingLLM",
    "block_kv": "Block-KV Selection",
    "recap": "ReCAP (anchor + W)",
}
ORDER = ["full_history", "sliding_window", "sliding_window_eq",
         "uniform_sampling", "streaming_llm", "block_kv", "recap"]
# The equal-budget control is the only fair anchor ablation.
CONTROL = "sliding_window_eq"


def metric(entry, key):
    metrics = entry["metrics"]
    if key in metrics:
        return float(metrics[key])
    trajectory = metrics.get("trajectory") or {}
    return float(trajectory.get(key, float("nan")))


def paired_ci(diffs, iterations=10000, seed=0):
    values = np.asarray(diffs, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return (float(values.mean()) if len(values) else float("nan"), None, None)
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(values), (iterations, len(values)))
    means = values[draws].mean(axis=1)
    return (float(values.mean()),
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def load(sweep_dir):
    groups = {}
    for w_dir in sorted(Path(sweep_dir).glob("W*")):
        if not w_dir.is_dir():
            continue
        try:
            w = int(w_dir.name[1:])
        except ValueError:
            continue
        runs = {}
        for path in sorted(w_dir.glob("train_eps_*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("status") == "ok" and "strategies" in data:
                runs[path.stem] = data
        if runs:
            groups[w] = runs
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_dir")
    parser.add_argument("--metrics", default="ndtw,ndtw_late_half,token_retention,"
                                             "arm_score,motion_preservation_late")
    parser.add_argument("--markdown-out", default=None)
    args = parser.parse_args()

    groups = load(args.sweep_dir)
    if not groups:
        raise SystemExit(f"No usable results under {args.sweep_dir}")
    keys = [k.strip() for k in args.metrics.split(",") if k.strip()]

    lines = []

    def emit(text=""):
        print(text)
        lines.append(text)

    # Only episodes present at every W can be compared across W values.
    common = set.intersection(*(set(runs) for runs in groups.values()))
    emit(f"W values: {sorted(groups)}")
    emit(f"Episodes per W: {{{', '.join(f'{w}: {len(r)}' for w, r in sorted(groups.items()))}}}")
    emit(f"Episodes common to all W: {len(common)}")
    if common:
        sample = next(iter(groups.values()))
        frames = [sample[e]["generated_frames"] for e in sorted(common) if e in sample]
        if frames:
            emit(f"Generated frames: mean {np.mean(frames):.1f}, "
                 f"range {min(frames)}-{max(frames)}")
    emit()

    present = [s for s in ORDER
               if any(s in run["strategies"]
                      for runs in groups.values() for run in runs.values())]

    for key in keys:
        emit("=" * 78)
        emit(f"{key} by W  (mean over episodes common to all W)")
        emit("=" * 78)
        header = f"{'W':>3s}" + "".join(f"{DISPLAY[s].split(' (')[0][:13]:>15s}" for s in present)
        emit(header)
        emit("-" * len(header))
        for w in sorted(groups):
            runs = groups[w]
            cells = ""
            for strategy in present:
                values = [metric(runs[e]["strategies"][strategy], key)
                          for e in sorted(common)
                          if e in runs and strategy in runs[e]["strategies"]]
                cells += f"{np.nanmean(values):>15.4f}" if values else f"{'--':>15s}"
            emit(f"{w:>3d}{cells}")
        emit()

    # Headline: the equal-budget anchor ablation.
    emit("=" * 78)
    emit(f"Anchor ablation: ReCAP - {DISPLAY[CONTROL]}")
    emit("Identical token budget; the only difference is how one block is spent.")
    emit("=" * 78)
    for key in keys:
        emit(f"\n{key}")
        for w in sorted(groups):
            runs = groups[w]
            diffs = []
            for episode in sorted(common):
                run = runs.get(episode)
                if not run:
                    continue
                strategies = run["strategies"]
                if "recap" not in strategies or CONTROL not in strategies:
                    continue
                diffs.append(metric(strategies["recap"], key)
                             - metric(strategies[CONTROL], key))
            if not diffs:
                emit(f"  W={w:<2d}  control absent")
                continue
            mean, low, high = paired_ci(diffs)
            wins = int(sum(1 for d in diffs if d > 0))
            if low is None:
                verdict = "n/a"
            elif low > 0:
                verdict = "ReCAP WINS*"
            elif high < 0:
                verdict = "ReCAP LOSES*"
            else:
                verdict = "ns"
            span = f"[{low:+.4f}, {high:+.4f}]" if low is not None else "n/a"
            emit(f"  W={w:<2d}  {mean:+.4f}  {span}  win {wins}/{len(diffs)}  {verdict}")
    emit()
    emit("* = paired bootstrap 95% CI excludes zero")

    if args.markdown_out:
        Path(args.markdown_out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nSaved {args.markdown_out}")


if __name__ == "__main__":
    main()
