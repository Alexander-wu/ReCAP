# Experiment entry points

All paper reference scripts are in `src/recap/reference`. They are launched with their directory on PYTHONPATH by `recap run`; advanced users may execute them directly with explicit arguments.

| Experiment | Script | Inputs / outputs |
|---|---|---|
| Context schedules and RQ1 | infer_compare.py | matched episode, tokenizer, world, ranges → paired rollouts, result.json, GIF/NPZ |
| RQ2 quality | evaluate_rq2_dataset.py | case directories containing frames.npz → frame/horizon/paired-AUC CSVs |
| RQ3 components/staleness | infer_compare.py + evaluate_rq3.py | generated or matched-GT stale controls, fixed cases |
| RQ3 attention | probe_rq3_attention.py | eager HF attention, fixed steps → attention statistics |
| RQ4 latency | benchmark_rq4_single.py | model + data → per-frame latency and logical KV accounting |
| RQ4 concurrency | benchmark_rq4_concurrency.py | vLLM prompt benchmark → throughput, concurrency/APC conditions |
| W sweep | evaluate_rq4_w_sweep.py | fixed sweep output → quality/cost tradeoff |
| Bridge H32 | bridge_pair.py | one observed frame, FP32, top-k 100, seed 20260906 |

The H64 config files are portable reference recipes, not a claim that their default seed reproduces every historical run. Run-specific original result metadata is authoritative. Preserve per-run sampling, case windows, precision, horizon and backend when rerunning published tables.

At long horizons, historical Full History caps the prompt at 7900 tokens with a tail that can cut a block. The implementation records truncation. H64 fits the 8192-position model; interpret H100 separately. `streaming_llm` and `block_kv` in the archived experiments are prompt scheduling/proxy baselines, not drop-in implementations of all mechanisms in the corresponding systems papers. `overlap_k` is recorded but has no effect in the original fixed-anchor ReCAP schedule.

## Training

`training/` is a source snapshot of the existing fine-tuning implementation, not CALVIN VAR. It includes the original tokenizer/world trainers, configuration files and dataset modules. `training/adaptation` contains converters, a strict HF→Accelerate seed builder, and idempotent dataset/trainer patches. Apply patches only to an isolated copy. Launch recipes in `training/recipes/*.txt` are historical evidence with machine paths redacted; they are intentionally not advertised as executable portable training commands.

CALVIN and LIBERO used tokenizer 20k + world 20k. Bridge used B.8.1 warm-start tokenizer 60k (GAN starts at 10k, weight 0.1, LR 5e-5) then a separately initialized world model 100k. GAN belongs to the tokenizer image-reconstruction stage, not ReCAP and not the world-model objective. Bridge's world seed must start with fresh optimizer/scheduler state paired to its chosen tokenizer. `--weighted_mse` was ineffective in the historical trainer (`weights=None`); do not describe it as an active objective.

A generic industrial training launcher, full dependency locking on a clean GPU machine, and distributed resume recovery are outside the validated release wrapper and must not be claimed complete. The exact source snapshot is provided so the experiment implementation is retained without silently redesigning the training method.
