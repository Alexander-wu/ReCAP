#!/usr/bin/env python3
"""Score the RQ3 component ablation and stale-history intervention.

Reads frame archives from three sources and joins them by case id:
  --rq2-root    already holds ground_truth / full_history / recap
  --rq3-root    holds sliding_window_eq / anchor_uniform / stale_inject@...
Ground truth always comes from the RQ2 archive so every method is scored against
the identical reference frames.

Two reporting rules the paper depends on:
  * Full context is reported as an unbounded reference, NOT as a member of the
    equal-budget group. Its prompt grows without bound and truncates late in the
    rollout, so pooling it with the 1931-token variants would contradict the
    "same context budget" claim.
  * nDTW / arm / shape scores come from heuristic pixel segmentation and are
    emitted with a `proxy_` prefix so they cannot be mistaken for calibrated
    kinematics.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch

HORIZONS = (8, 16, 32, 48, 64)
EQUAL_BUDGET = ("sliding_window_eq", "anchor_uniform", "recap")
REFERENCE = ("full_history",)


def csv_write(path, rows):
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(values, rng, draws=5000):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return float(values.mean()), float(values.mean())
    idx = rng.integers(0, len(values), size=(draws, len(values)))
    boot = values[idx].mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def normalized_auc(values):
    values = np.asarray(values, dtype=np.float64)
    x = np.asarray(HORIZONS, dtype=np.float64) / HORIZONS[-1]
    area = np.sum((values[1:] + values[:-1]) * 0.5 * np.diff(x))
    return float(area / (x[-1] - x[0]))


def frame_metrics(gt_uint8, pred_uint8, device, lpips_metric, ssim_metric, batch_size):
    output = {key: [] for key in ("lpips", "ssim", "psnr_db", "mae")}
    for begin in range(0, len(gt_uint8), batch_size):
        end = min(begin + batch_size, len(gt_uint8))
        gt = torch.from_numpy(gt_uint8[begin:end]).permute(0, 3, 1, 2).float().div_(255).to(device)
        pred = torch.from_numpy(pred_uint8[begin:end]).permute(0, 3, 1, 2).float().div_(255).to(device)
        with torch.inference_mode():
            output["lpips"].extend(lpips_metric(gt * 2 - 1, pred * 2 - 1).flatten().cpu().tolist())
            output["ssim"].extend(ssim_metric(gt, pred).flatten().cpu().tolist())
            mse = (gt - pred).pow(2).mean(dim=(1, 2, 3))
            output["psnr_db"].extend((-10 * torch.log10(mse.clamp_min(1e-12))).cpu().tolist())
            output["mae"].extend((gt - pred).abs().mean(dim=(1, 2, 3)).cpu().tolist())
    return {key: np.asarray(value, dtype=np.float64) for key, value in output.items()}


def load_archive(path, horizon):
    with np.load(path) as archive:
        return {name: archive[name][2:2 + horizon] for name in archive.files
                if name not in ("frame_indices",)}


def parse_stale_label(label):
    match = re.fullmatch(r"stale_inject@k=(\d+)\+src=(generated|ground_truth)", label)
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def proxy_scores(result, label):
    """Pull heuristic trajectory scores, clearly namespaced as proxies."""
    payload = result.get("strategies", {}).get(label, {}).get("metrics", {}).get("trajectory", {})
    if not isinstance(payload, dict) or "error" in payload:
        return {}
    out = {}
    for key in ("ndtw", "arm_score", "shape_iou"):
        if payload.get(key) is not None:
            out[f"proxy_{key}"] = round(float(payload[key]), 6)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--rq2-root", required=True)
    parser.add_argument("--rq3-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    import lpips
    import piqa

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    lpips_metric = lpips.LPIPS(net="vgg").to(device).eval()
    ssim_metric = piqa.SSIM(window_size=11, sigma=1.5, n_channels=3,
                            reduction="none").to(device).eval()

    rq2_root = Path(args.rq2_root)
    rq3_root = Path(args.rq3_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    horizon = HORIZONS[-1]

    horizon_rows = []
    paired_rows = []
    stale_rows = []
    skipped = []
    methods_seen = set()

    for rq2_case in sorted(p for p in rq2_root.iterdir() if (p / "frames.npz").is_file()):
        case_id = rq2_case.name
        rq3_case = rq3_root / case_id
        if not (rq3_case / "frames.npz").is_file():
            skipped.append({"case_id": case_id, "reason": "missing rq3 frames"})
            continue

        base = load_archive(rq2_case / "frames.npz", horizon)
        extra = load_archive(rq3_case / "frames.npz", horizon)
        gt = base.get("ground_truth")
        if gt is None or len(gt) < horizon:
            skipped.append({"case_id": case_id, "reason": "short ground truth"})
            continue

        rq2_result = json.loads((rq2_case / "result.json").read_text(encoding="utf-8"))
        rq3_result = json.loads((rq3_case / "result.json").read_text(encoding="utf-8"))

        predictions = {}
        origin = {}
        for label, frames in base.items():
            if label in ("ground_truth",):
                continue
            predictions[label] = frames
            origin[label] = rq2_result
        for label, frames in extra.items():
            if label in ("ground_truth",):
                continue
            predictions[label] = frames
            origin[label] = rq3_result

        curves = {}
        for label, frames in predictions.items():
            if len(frames) < horizon:
                continue
            values = frame_metrics(gt, frames, device, lpips_metric, ssim_metric, args.batch_size)
            curves[label] = values
            methods_seen.add(label)
            stale = parse_stale_label(label)
            for h in HORIZONS:
                row = {
                    "dataset": args.dataset,
                    "case_id": case_id,
                    "method": label,
                    "group": ("equal_budget" if label in EQUAL_BUDGET or stale
                              else "unbounded_reference" if label in REFERENCE else "other"),
                    "horizon": h,
                }
                for metric, vector in values.items():
                    row[metric] = round(float(vector[:h].mean()), 7)
                    row[f"late8_{metric}"] = round(float(vector[max(0, h - 8):h].mean()), 7)
                if h == horizon:
                    row.update(proxy_scores(origin[label], label))
                    runtime = origin[label]["strategies"][label]["runtime"]
                    row["prompt_tokens_max"] = runtime["prompt_tokens_max"]
                horizon_rows.append(row)

            if stale:
                k, source = stale
                stale_rows.append({
                    "dataset": args.dataset,
                    "case_id": case_id,
                    "stale_blocks_k": k,
                    "stale_source": source,
                    "lpips_auc": round(normalized_auc(
                        [values["lpips"][:h].mean() for h in HORIZONS]), 7),
                    "lpips_h64": round(float(values["lpips"][:horizon].mean()), 7),
                    "ssim_h64": round(float(values["ssim"][:horizon].mean()), 7),
                    "psnr_h64": round(float(values["psnr_db"][:horizon].mean()), 7),
                    "prompt_tokens_max": origin[label]["strategies"][label]["runtime"]["prompt_tokens_max"],
                })

        if "recap" not in curves:
            skipped.append({"case_id": case_id, "reason": "missing recap curve"})
            continue
        recap_auc = {m: normalized_auc([curves["recap"][m][:h].mean() for h in HORIZONS])
                     for m in ("lpips", "ssim", "psnr_db")}
        for label, values in curves.items():
            if label == "recap":
                continue
            other = {m: normalized_auc([values[m][:h].mean() for h in HORIZONS])
                     for m in ("lpips", "ssim", "psnr_db")}
            paired_rows.append({
                "dataset": args.dataset,
                "case_id": case_id,
                "baseline": label,
                "lpips_auc_baseline": round(other["lpips"], 7),
                "lpips_auc_recap": round(recap_auc["lpips"], 7),
                "lpips_auc_gain_baseline_minus_recap": round(other["lpips"] - recap_auc["lpips"], 7),
                "ssim_auc_gain_recap_minus_baseline": round(recap_auc["ssim"] - other["ssim"], 7),
                "psnr_auc_gain_recap_minus_baseline": round(recap_auc["psnr_db"] - other["psnr_db"], 7),
            })
        print(f"RQ3_EVAL_DONE {args.dataset} {case_id} methods={len(curves)}", flush=True)

    if not paired_rows:
        raise RuntimeError(f"No scorable cases for {args.dataset}")

    rng = np.random.default_rng(20260903)
    summary_rows = []
    for method in sorted(methods_seen):
        for h in HORIZONS:
            selected = [r for r in horizon_rows if r["method"] == method and r["horizon"] == h]
            if not selected:
                continue
            row = {"dataset": args.dataset, "method": method, "group": selected[0]["group"],
                   "horizon": h, "case_count": len(selected)}
            for metric in ("lpips", "ssim", "psnr_db", "mae"):
                vector = np.asarray([item[metric] for item in selected], dtype=np.float64)
                low, high = bootstrap_ci(vector, rng)
                row[f"{metric}_mean"] = round(float(vector.mean()), 7)
                row[f"{metric}_std"] = round(float(vector.std(ddof=1)) if len(vector) > 1 else 0.0, 7)
                row[f"{metric}_ci95_low"] = round(low, 7)
                row[f"{metric}_ci95_high"] = round(high, 7)
            for proxy in ("proxy_ndtw", "proxy_arm_score", "proxy_shape_iou"):
                vals = [item[proxy] for item in selected if proxy in item]
                if vals:
                    row[f"{proxy}_mean"] = round(float(np.mean(vals)), 6)
            summary_rows.append(row)

    paired_summary = {}
    for baseline in sorted({r["baseline"] for r in paired_rows}):
        selected = [r for r in paired_rows if r["baseline"] == baseline]
        entry = {"cases": len(selected)}
        for key, label in (("lpips_auc_gain_baseline_minus_recap", "lpips_auc_gain"),
                           ("ssim_auc_gain_recap_minus_baseline", "ssim_auc_gain")):
            vector = np.asarray([r[key] for r in selected], dtype=np.float64)
            low, high = bootstrap_ci(vector, rng)
            entry[label] = {
                "mean": round(float(vector.mean()), 7),
                "std": round(float(vector.std(ddof=1)) if len(vector) > 1 else 0.0, 7),
                "ci95": [round(low, 7), round(high, 7)],
                "recap_wins": int(np.sum(vector > 0)),
            }
        paired_summary[baseline] = entry

    stale_summary = {}
    for source in sorted({r["stale_source"] for r in stale_rows}):
        for k in sorted({r["stale_blocks_k"] for r in stale_rows if r["stale_source"] == source}):
            selected = [r for r in stale_rows
                        if r["stale_source"] == source and r["stale_blocks_k"] == k]
            vector = np.asarray([r["lpips_h64"] for r in selected], dtype=np.float64)
            low, high = bootstrap_ci(vector, rng)
            stale_summary[f"{source}@k={k}"] = {
                "cases": len(selected),
                "lpips_h64_mean": round(float(vector.mean()), 7),
                "lpips_h64_ci95": [round(low, 7), round(high, 7)],
                "lpips_auc_mean": round(float(np.mean([r["lpips_auc"] for r in selected])), 7),
                "ssim_h64_mean": round(float(np.mean([r["ssim_h64"] for r in selected])), 7),
                "prompt_tokens_max": max(r["prompt_tokens_max"] for r in selected),
            }

    report = {
        "dataset": args.dataset,
        "horizons": list(HORIZONS),
        "cases": len({r["case_id"] for r in paired_rows}),
        "methods": sorted(methods_seen),
        "budget_note": "full_history is an unbounded reference; sliding_window_eq, "
                       "anchor_uniform, recap and every stale_inject variant share the "
                       "same 1931-token block budget",
        "proxy_note": "proxy_ndtw / proxy_arm_score / proxy_shape_iou come from heuristic "
                      "pixel segmentation and are not calibrated kinematics",
        "paired_vs_recap": paired_summary,
        "stale_intervention": stale_summary,
        "skipped": skipped,
    }

    prefix = args.dataset.lower().replace("-", "_").replace(" ", "_")
    csv_write(output_dir / f"{prefix}_rq3_per_case_horizon.csv", horizon_rows)
    csv_write(output_dir / f"{prefix}_rq3_summary.csv", summary_rows)
    csv_write(output_dir / f"{prefix}_rq3_paired.csv", paired_rows)
    if stale_rows:
        csv_write(output_dir / f"{prefix}_rq3_stale_sweep.csv", stale_rows)
    (output_dir / f"{prefix}_rq3_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print("RQ3_EVAL_RESULT=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
