"""Aggregate strategy-comparison results across episodes.

Reports, per strategy, the mean of every headline metric plus paired
bootstrap 95% confidence intervals for the ReCAP-minus-baseline differences.
Pairing matters: the same episode is rolled out under every strategy with the
same actions and seed, so paired statistics remove episode difficulty as a
confound.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

HEADLINE = [
    ("psnr_db_mean", "PSNR dB", "higher"),
    ("mae_mean", "MAE", "lower"),
    ("ndtw", "nDTW", "higher"),
    ("arm_score", "Arm Score", "higher"),
    ("shape_iou", "Shape IoU", "higher"),
    ("motion_preservation_late", "motion late", "higher"),
    ("motion_preservation_early", "motion early", "higher"),
    ("token_retention", "token ret", "higher"),
    ("drift_final", "drift end", "n/a"),
]
BASELINE = "full_history"
ORDER = [
    "full_history",
    "sliding_window",
    "uniform_sampling",
    "streaming_llm",
    "block_kv",
    "recap",
]
DISPLAY = {
    "full_history": "Full Context",
    "sliding_window": "Sliding Window",
    "uniform_sampling": "Uniform Sampling",
    "streaming_llm": "StreamingLLM",
    "block_kv": "Block-KV Selection",
    "recap": "ReCAP (ours)",
}


def load(paths, min_frames=0, drop_static_late=True):
    runs = []
    dropped_static = 0
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("status") != "ok" or "strategies" not in data:
            continue
        if data["generated_frames"] < min_frames:
            continue
        if drop_static_late:
            # The motion ratio divides by ground-truth motion in the late window.
            # If that clip is nearly still the ratio is dominated by noise, so we
            # exclude the episode from motion statistics rather than report it.
            flags = [payload["metrics"].get("late_window_is_static", False)
                     for payload in data["strategies"].values()]
            if any(flags):
                dropped_static += 1
                continue
        runs.append(data)
    return runs, dropped_static


def metric_value(payload, key):
    """Look up a headline metric, transparently reaching into `trajectory`.

    Trajectory metrics (nDTW, Arm Score, Shape IoU) live in a nested dict
    because they come from a separate segmentation pass. Returning NaN for a
    missing key keeps older result files loadable.
    """
    metrics = payload.get("metrics", {})
    if key in metrics:
        value = metrics[key]
    else:
        value = (metrics.get("trajectory") or {}).get(key)
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def mean_ignoring_nan(values):
    array = np.array(values, dtype=float)
    valid = array[~np.isnan(array)]
    return float(valid.mean()) if len(valid) else float("nan")


def paired_bootstrap(differences, iterations=10000, seed=0):
    values = np.array(differences, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return float(values.mean()) if len(values) else 0.0, None, None
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(values), size=(iterations, len(values)))
    means = values[draws].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--markdown-out", default=None)
    parser.add_argument("--min-frames", type=int, default=0,
                        help="skip episodes shorter than this many generated frames")
    parser.add_argument("--keep-static-late", action="store_true",
                        help="keep episodes whose late window is nearly still; by default they are "
                             "dropped because the motion ratio divides by a near-zero denominator")
    args = parser.parse_args()

    paths = []
    for pattern in args.results:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])
    runs, dropped_static = load(paths, args.min_frames, not args.keep_static_late)
    if not runs:
        raise SystemExit("No usable result files")

    names = []
    for run in runs:
        for name in run["strategies"]:
            if name not in names:
                names.append(name)
    names = [name for name in ORDER if name in names] + [name for name in names if name not in ORDER]
    complete = [run for run in runs if all(name in run["strategies"] for name in names)]

    lines = []

    def emit(text=""):
        print(text)
        lines.append(text)

    emit(f"Episodes loaded: {len(runs)}   fully paired across all strategies: {len(complete)}")
    if dropped_static:
        emit(f"Dropped {dropped_static} episode(s) whose late window is nearly static "
             f"(motion ratio denominator too small)")
    emit(f"Strategies: {', '.join(DISPLAY.get(name, name) for name in names)}")
    frames = [run["generated_frames"] for run in runs]
    lengths = [run["episode_length"] for run in runs]
    emit(f"Episode length  mean {np.mean(lengths):.1f}  range {min(lengths)}-{max(lengths)}")
    emit(f"Frames generated mean {np.mean(frames):.1f}  range {min(frames)}-{max(frames)}")
    emit()

    header = f"{'strategy':>20s}" + "".join(f"{label:>15s}" for _, label, _ in HEADLINE)
    header += f"{'prompt max':>12s}{'sec/frame':>11s}"
    emit(header)
    emit("-" * len(header))

    table = {}
    for name in names:
        row = {}
        for key, _, _ in HEADLINE:
            row[key] = mean_ignoring_nan([metric_value(run["strategies"][name], key)
                                          for run in runs if name in run["strategies"]])
        row["prompt_tokens_max"] = float(np.mean(
            [run["strategies"][name]["runtime"]["prompt_tokens_max"] for run in runs
             if name in run["strategies"]]))
        row["seconds_per_frame"] = float(np.mean(
            [run["strategies"][name]["runtime"]["seconds_per_frame"] for run in runs
             if name in run["strategies"]]))
        table[name] = row
        cells = "".join(f"{row[key]:>15.4f}" for key, _, _ in HEADLINE)
        emit(f"{DISPLAY.get(name, name):>20s}{cells}"
             f"{row['prompt_tokens_max']:>12.0f}{row['seconds_per_frame']:>11.4f}")

    differences = {}
    if BASELINE in names and len(complete) >= 2:
        emit()
        emit(f"Paired differences vs {DISPLAY.get(BASELINE, BASELINE)} "
             f"(n={len(complete)}, bootstrap 95% CI):")
        for name in names:
            if name == BASELINE:
                continue
            emit(f"  {DISPLAY.get(name, name)}")
            differences[name] = {}
            for key, label, direction in HEADLINE:
                values = [
                    metric_value(run["strategies"][name], key)
                    - metric_value(run["strategies"][BASELINE], key)
                    for run in complete
                ]
                mean, low, high = paired_bootstrap(values)
                wins = sum(1 for value in values
                           if not np.isnan(value) and (value > 0) == (direction == "higher"))
                significant = low is not None and (low > 0 or high < 0)
                marker = " *" if significant else ""
                win_note = f"  win {wins}/{len(values)}" if direction != "n/a" else ""
                span = f"[{low:+.4f}, {high:+.4f}]" if low is not None else "n/a"
                emit(f"    {label:>14s}  {mean:+.4f}  {span}{marker}{win_note}")
                differences[name][key] = {
                    "mean": mean, "ci95_low": low, "ci95_high": high,
                    "wins": wins, "n": len(values), "significant": bool(significant),
                }
        emit("  * = 95% CI excludes zero")

    if "recap" in names and len(complete) >= 2:
        emit()
        emit(f"Paired differences vs ReCAP (n={len(complete)}, negative favours ReCAP "
             f"on higher-is-better metrics):")
        for name in names:
            if name == "recap":
                continue
            parts = []
            for key, label, direction in HEADLINE:
                if direction == "n/a":
                    continue
                values = [
                    metric_value(run["strategies"][name], key)
                    - metric_value(run["strategies"]["recap"], key)
                    for run in complete
                ]
                mean, low, high = paired_bootstrap(values)
                star = "*" if low is not None and (low > 0 or high < 0) else " "
                parts.append(f"{label} {mean:+.4f}{star}")
            emit(f"  {DISPLAY.get(name, name):>20s}  " + "  ".join(parts))

    emit()
    emit("Context and cost notes:")
    for name in names:
        sample = next(run["strategies"][name]["notes"] for run in runs if name in run["strategies"])
        emit(f"  {DISPLAY.get(name, name):>20s}  {json.dumps(sample, sort_keys=True)}")

    if args.json_out:
        payload = {
            "episodes": len(runs),
            "paired_episodes": len(complete),
            "strategies": names,
            "means": table,
            "paired_differences_vs_baseline": differences,
            "sources": [run["source"] for run in runs],
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved {args.json_out}")
    if args.markdown_out:
        Path(args.markdown_out).write_text("```\n" + "\n".join(lines) + "\n```\n", encoding="utf-8")
        print(f"Saved {args.markdown_out}")


if __name__ == "__main__":
    main()
