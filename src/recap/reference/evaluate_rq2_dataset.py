#!/usr/bin/env python3
"""Evaluate Full History versus ReCAP on fixed long-rollout frame archives."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

HORIZONS = (8, 16, 32, 48, 64)
METHODS = ("full_history", "recap")


def csv_write(path, rows):
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(values, rng, draws=5000):
    values = np.asarray(values, dtype=np.float64)
    idx = rng.integers(0, len(values), size=(draws, len(values)))
    boot = values[idx].mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def normalized_auc(values):
    values = np.asarray(values, dtype=np.float64)
    x = np.asarray(HORIZONS, dtype=np.float64) / HORIZONS[-1]
    area = np.sum((values[1:] + values[:-1]) * 0.5 * np.diff(x))
    return float(area / (x[-1] - x[0]))


def metrics(gt_uint8, pred_uint8, device, lpips_metric, ssim_metric, batch_size):
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


def resolve_case_dirs(root):
    dirs = sorted(path for path in root.iterdir() if path.is_dir() and (path / "frames.npz").is_file())
    if not dirs:
        raise RuntimeError(f"No case directories with frames.npz under {root}")
    return dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--root", required=True)
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
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_dirs = resolve_case_dirs(root)

    per_frame_rows = []
    horizon_rows = []
    paired_rows = []
    arrays = {}
    case_meta = {}
    for case_dir in case_dirs:
        case_id = case_dir.name
        with np.load(case_dir / "frames.npz") as archive:
            required = {"ground_truth", *METHODS}
            missing = required - set(archive.files)
            if missing:
                raise RuntimeError(f"{case_id} missing arrays: {sorted(missing)}")
            gt = archive["ground_truth"][2:2 + HORIZONS[-1]]
            predictions = {method: archive[method][2:2 + HORIZONS[-1]] for method in METHODS}
        if len(gt) < HORIZONS[-1]:
            continue
        result_path = case_dir / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
        case_meta[case_id] = {
            "source": result.get("source", case_id),
            "generated_frames": result.get("generated_frames", len(gt)),
        }
        for method in METHODS:
            values = metrics(gt, predictions[method], device, lpips_metric, ssim_metric, args.batch_size)
            arrays[(case_id, method)] = values
            for index in range(HORIZONS[-1]):
                per_frame_rows.append({
                    "dataset": args.dataset,
                    "case_id": case_id,
                    "method": method,
                    "future_step": index + 1,
                    "lpips": round(float(values["lpips"][index]), 7),
                    "ssim": round(float(values["ssim"][index]), 7),
                    "psnr_db": round(float(values["psnr_db"][index]), 7),
                    "mae": round(float(values["mae"][index]), 7),
                })
            for horizon in HORIZONS:
                row = {
                    "dataset": args.dataset,
                    "case_id": case_id,
                    "method": method,
                    "horizon": horizon,
                    "aggregation": "prefix_mean_t1_to_tH",
                }
                for metric, vector in values.items():
                    row[metric] = round(float(vector[:horizon].mean()), 7)
                    row[f"late8_{metric}"] = round(float(vector[max(0, horizon - 8):horizon].mean()), 7)
                horizon_rows.append(row)
        recap_lpips = [arrays[(case_id, "recap")]["lpips"][:h].mean() for h in HORIZONS]
        full_lpips = [arrays[(case_id, "full_history")]["lpips"][:h].mean() for h in HORIZONS]
        recap_ssim = [arrays[(case_id, "recap")]["ssim"][:h].mean() for h in HORIZONS]
        full_ssim = [arrays[(case_id, "full_history")]["ssim"][:h].mean() for h in HORIZONS]
        recap_psnr = [arrays[(case_id, "recap")]["psnr_db"][:h].mean() for h in HORIZONS]
        full_psnr = [arrays[(case_id, "full_history")]["psnr_db"][:h].mean() for h in HORIZONS]
        paired_rows.append({
            "dataset": args.dataset,
            "case_id": case_id,
            "recap_lpips_auc": round(normalized_auc(recap_lpips), 7),
            "full_lpips_auc": round(normalized_auc(full_lpips), 7),
            "lpips_auc_gain_full_minus_recap": round(normalized_auc(full_lpips) - normalized_auc(recap_lpips), 7),
            "recap_ssim_auc": round(normalized_auc(recap_ssim), 7),
            "full_ssim_auc": round(normalized_auc(full_ssim), 7),
            "ssim_auc_gain_recap_minus_full": round(normalized_auc(recap_ssim) - normalized_auc(full_ssim), 7),
            "recap_psnr_auc": round(normalized_auc(recap_psnr), 7),
            "full_psnr_auc": round(normalized_auc(full_psnr), 7),
            "psnr_auc_gain_recap_minus_full": round(normalized_auc(recap_psnr) - normalized_auc(full_psnr), 7),
        })
        print(f"RQ2_DATASET_DONE {args.dataset} {case_id}", flush=True)

    if not paired_rows:
        raise RuntimeError(f"No cases with at least {HORIZONS[-1]} future frames under {root}")

    rng = np.random.default_rng(20260903)
    summary_rows = []
    for method in METHODS:
        for horizon in HORIZONS:
            selected = [row for row in horizon_rows if row["method"] == method and row["horizon"] == horizon]
            row = {"dataset": args.dataset, "method": method, "horizon": horizon, "case_count": len(selected)}
            for metric in ("lpips", "ssim", "psnr_db", "mae"):
                vector = np.asarray([item[metric] for item in selected], dtype=np.float64)
                low, high = bootstrap_ci(vector, rng)
                row[f"{metric}_mean"] = round(float(vector.mean()), 7)
                row[f"{metric}_std"] = round(float(vector.std(ddof=1)), 7)
                row[f"{metric}_ci95_low"] = round(low, 7)
                row[f"{metric}_ci95_high"] = round(high, 7)
            summary_rows.append(row)

    report = {"dataset": args.dataset, "case_count": len(paired_rows), "horizons": list(HORIZONS), "paired": {}}
    for key in ("lpips_auc_gain_full_minus_recap", "ssim_auc_gain_recap_minus_full", "psnr_auc_gain_recap_minus_full"):
        vector = np.asarray([row[key] for row in paired_rows], dtype=np.float64)
        low, high = bootstrap_ci(vector, rng)
        report["paired"][key] = {
            "mean": round(float(vector.mean()), 7),
            "std": round(float(vector.std(ddof=1)), 7),
            "ci95": [round(low, 7), round(high, 7)],
            "wins": int(np.sum(vector > 0)),
            "cases": len(vector),
        }

    prefix = args.dataset.lower().replace("-", "_").replace(" ", "_")
    csv_write(output_dir / f"{prefix}_rq2_per_frame.csv", per_frame_rows)
    csv_write(output_dir / f"{prefix}_rq2_per_case_horizon.csv", horizon_rows)
    csv_write(output_dir / f"{prefix}_rq2_summary.csv", summary_rows)
    csv_write(output_dir / f"{prefix}_rq2_paired_auc.csv", paired_rows)
    (output_dir / f"{prefix}_rq2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("RQ2_DATASET_RESULT=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
