#!/usr/bin/env python3
"""Evaluate ReCAP W-sweep quality for RQ4 Pareto material."""

import argparse
import base64
import csv
import io
import json
import re
from pathlib import Path

import numpy as np
import torch

HORIZONS = (8, 16, 32, 48, 64)
ARTIFACT_CHUNK_CHARS = 40000


def emit(name, payload):
    s = base64.b64encode(payload).decode("ascii")
    n = max(1, (len(s) + ARTIFACT_CHUNK_CHARS - 1) // ARTIFACT_CHUNK_CHARS)
    print(f"ARTIFACT_BEGIN name={name} chunks={n} bytes={len(payload)}", flush=True)
    for i in range(n):
        print(f"ARTIFACT_CHUNK name={name} idx={i} data={s[i*ARTIFACT_CHUNK_CHARS:(i+1)*ARTIFACT_CHUNK_CHARS]}", flush=True)
    print(f"ARTIFACT_END name={name}", flush=True)


def csv_bytes(rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    stream = io.StringIO(); writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def bootstrap(values, rng, draws=5000):
    v = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(v), size=(draws, len(v)))
    b = v[idx].mean(axis=1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def auc(values):
    x = np.asarray(HORIZONS, dtype=float) / HORIZONS[-1]
    y = np.asarray(values, dtype=float)
    return float(np.sum((y[1:] + y[:-1]) * .5 * np.diff(x)) / (x[-1] - x[0]))


def score(gt8, pred8, device, lpips_metric, ssim_metric, batch=8):
    result = {k: [] for k in ("lpips", "ssim", "psnr_db", "mae")}
    for start in range(0, len(gt8), batch):
        g = torch.from_numpy(gt8[start:start+batch]).permute(0,3,1,2).float().div(255).to(device)
        p = torch.from_numpy(pred8[start:start+batch]).permute(0,3,1,2).float().div(255).to(device)
        with torch.inference_mode():
            result["lpips"].extend(lpips_metric(g*2-1,p*2-1).flatten().cpu().tolist())
            result["ssim"].extend(ssim_metric(g,p).flatten().cpu().tolist())
            mse=(g-p).pow(2).mean((1,2,3)); result["psnr_db"].extend((-10*torch.log10(mse.clamp_min(1e-12))).cpu().tolist())
            result["mae"].extend((g-p).abs().mean((1,2,3)).cpu().tolist())
    return {k:np.asarray(v,dtype=float) for k,v in result.items()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",required=True); parser.add_argument("--root",required=True)
    parser.add_argument("--output-dir",required=True); parser.add_argument("--batch-size",type=int,default=8)
    args=parser.parse_args()
    import lpips,piqa
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lp=lpips.LPIPS(net="vgg").to(device).eval()
    ss=piqa.SSIM(window_size=11,sigma=1.5,n_channels=3,reduction="none").to(device).eval()
    root=Path(args.root); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    rows=[]; auc_rows=[]
    for case in sorted(p for p in root.iterdir() if p.is_dir() and (p/"frames.npz").is_file()):
        with np.load(case/"frames.npz") as z:
            gt=z["ground_truth"][2:66]
            labels=[x for x in z.files if re.fullmatch(r"recap@w=\d+",x)]
            for label in labels:
                pred=z[label][2:66]; vals=score(gt,pred,device,lp,ss,args.batch_size); w=int(label.split("=")[1])
                curves={}
                for h in HORIZONS:
                    item={"dataset":args.dataset,"case_id":case.name,"window_w":w,"horizon":h}
                    for m,v in vals.items(): item[m]=round(float(v[:h].mean()),7)
                    rows.append(item)
                for m,v in vals.items(): curves[m]=auc([v[:h].mean() for h in HORIZONS])
                auc_rows.append({"dataset":args.dataset,"case_id":case.name,"window_w":w,
                                 "context_tokens":1280+93*(1+w),
                                 "theoretical_kv_mib":round((1280+93*(1+w))*36864/2**20,6),
                                 "lpips_auc":round(curves["lpips"],7),"ssim_auc":round(curves["ssim"],7),
                                 "psnr_auc":round(curves["psnr_db"],7),"mae_auc":round(curves["mae"],7)})
        print(f"RQ4_W_EVAL_DONE {args.dataset} {case.name}",flush=True)
    rng=np.random.default_rng(20260904); summary=[]
    for w in sorted({r["window_w"] for r in auc_rows}):
        selected=[r for r in auc_rows if r["window_w"]==w]
        row={"dataset":args.dataset,"window_w":w,"cases":len(selected),
             "context_tokens":selected[0]["context_tokens"],"theoretical_kv_mib":selected[0]["theoretical_kv_mib"]}
        for m in ("lpips_auc","ssim_auc","psnr_auc","mae_auc"):
            v=[r[m] for r in selected]; lo,hi=bootstrap(v,rng); row[f"{m}_mean"]=round(float(np.mean(v)),7); row[f"{m}_std"]=round(float(np.std(v,ddof=1)),7); row[f"{m}_ci95_low"]=round(lo,7); row[f"{m}_ci95_high"]=round(hi,7)
        summary.append(row)
    prefix=args.dataset.lower().replace("-","_").replace(" ","_")
    files={f"{prefix}_rq4_w_per_case.csv":csv_bytes(auc_rows),f"{prefix}_rq4_w_per_horizon.csv":csv_bytes(rows),f"{prefix}_rq4_w_summary.csv":csv_bytes(summary),f"{prefix}_rq4_w_report.json":json.dumps({"dataset":args.dataset,"cases":len({r['case_id'] for r in auc_rows}),"summary":summary},indent=2).encode()}
    for name,payload in files.items(): (out/name).write_bytes(payload); emit(name,payload)
    print("RQ4_W_RESULT="+json.dumps({"dataset":args.dataset,"summary":summary},sort_keys=True),flush=True)

if __name__=="__main__": main()
