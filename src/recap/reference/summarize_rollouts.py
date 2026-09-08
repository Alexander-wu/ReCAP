"""Aggregate collapse statistics across rollout runs.

Reports two horizon-sensitive metrics that plain PSNR misses:
  activity ratio   = predicted frame-to-frame motion / ground-truth motion
  token retention  = late-stage unique visual tokens / early-stage unique tokens
Both are computed on an early window (first 15% of steps) and a late window
(last 30% of steps), so they are comparable across rollouts of different length.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

EARLY_FRACTION = 0.15
LATE_FRACTION = 0.30


def windows(count):
    early_end = max(1, int(round(count * EARLY_FRACTION)))
    late_start = min(count - 1, int(round(count * (1.0 - LATE_FRACTION))))
    return slice(0, early_end), slice(late_start, count)


def summarize(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    diagnostics = data["collapse_diagnostics"]
    prediction = np.array(diagnostics["prediction_activity"], dtype=float)
    truth = np.array(diagnostics["ground_truth_activity"], dtype=float)
    unique = np.array(diagnostics["unique_tokens_per_frame"], dtype=float)
    drift = np.array(diagnostics["prediction_drift_from_first"], dtype=float)
    gt_drift = np.array(diagnostics["ground_truth_drift_from_first"], dtype=float)
    early, late = windows(len(prediction))

    return {
        "run": Path(path).parent.name,
        "episode_length": data["episode_length"],
        "steps": data["predicted_frames"],
        "truncated": data["truncated_by_context_budget"],
        "activity_ratio_early": float(prediction[early].mean() / truth[early].mean()),
        "activity_ratio_late": float(prediction[late].mean() / truth[late].mean()),
        "token_retention": float(unique[late].mean() / unique[early].mean()),
        "unique_tokens_first": int(unique[0]),
        "unique_tokens_last": int(unique[-1]),
        "drift_peak": float(drift.max()),
        "drift_final": float(drift[-1]),
        "drift_shrink_from_peak": float(1.0 - drift[-1] / drift.max()) if drift.max() > 0 else 0.0,
        "ground_truth_drift_final": float(gt_drift[-1]),
        "psnr_db_mean": data["metrics"]["psnr_db_mean"],
        "static_psnr_db_mean": data["static_frame_baseline"]["psnr_db_mean"],
        "prompt_tokens_last_step": data["context"]["prompt_tokens_last_step"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern", nargs="?", default="outputs/rollout-*/result.json")
    parser.add_argument("--min-steps-for-long", type=int, default=60)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No result files matched {args.pattern!r}")
    rows = sorted((summarize(path) for path in paths), key=lambda row: -row["steps"])

    header = (
        f"{'run':>42s} {'len':>4s} {'steps':>5s} {'act%early':>9s} {'act%late':>8s} "
        f"{'tok_ret':>7s} {'uniq':>9s} {'PSNR':>6s} {'static':>6s} {'drift pk>end':>13s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        uniq = f"{row['unique_tokens_first']}>{row['unique_tokens_last']}"
        print(
            f"{row['run'][-42:]:>42s} {row['episode_length']:>4d} {row['steps']:>5d} "
            f"{row['activity_ratio_early'] * 100:>8.0f}% {row['activity_ratio_late'] * 100:>7.0f}% "
            f"{row['token_retention'] * 100:>6.0f}% {uniq:>9s} "
            f"{row['psnr_db_mean']:>6.2f} {row['static_psnr_db_mean']:>6.2f} "
            f"{row['drift_peak']:>5.3f}>{row['drift_final']:<6.3f}"
        )

    long_rows = [row for row in rows if row["steps"] >= args.min_steps_for_long]
    if long_rows:
        def mean(key):
            return float(np.mean([row[key] for row in long_rows]))

        print(f"\nlong rollouts (>= {args.min_steps_for_long} steps), n={len(long_rows)}:")
        print(f"  activity ratio  early {mean('activity_ratio_early') * 100:5.1f}%  "
              f"-> late {mean('activity_ratio_late') * 100:5.1f}%")
        print(f"  token retention {mean('token_retention') * 100:5.1f}%")
        print(f"  drift  peak {mean('drift_peak'):.4f} -> final {mean('drift_final'):.4f}  "
              f"(shrink {mean('drift_shrink_from_peak') * 100:.1f}%, "
              f"ground truth final {mean('ground_truth_drift_final'):.4f})")
        print(f"  PSNR  model {mean('psnr_db_mean'):.2f} dB  vs  static baseline "
              f"{mean('static_psnr_db_mean'):.2f} dB  "
              f"(+{mean('psnr_db_mean') - mean('static_psnr_db_mean'):.2f})")

    if args.json_out:
        payload = {
            "windows": {"early_fraction": EARLY_FRACTION, "late_fraction": LATE_FRACTION},
            "runs": rows,
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved {args.json_out}")


if __name__ == "__main__":
    main()
