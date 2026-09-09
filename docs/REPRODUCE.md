# Reproduce a paired rollout

[← README](../README.md) · [Data contracts](DATA.md) · [Environment](ENVIRONMENT.md) · [Experiment protocols](EXPERIMENTS.md)

The [README quick start](../README.md#quick-start) runs CALVIN with two observed frames and 64 predicted frames. This guide covers data preparation and the other released recipes. Run commands from the repository root.

## Prepare CALVIN data

**Already have a converted episode?** Place it at `data/calvin_example/episode.npz` and skip to [Run and inspect](#run-and-inspect). It must contain `image: uint8[T,256,320,3]` and `action: float32[T,13]`, with **T ≥ 66**, using the [CALVIN action mapping](DATA.md).

Otherwise:

1. Download **CALVIN D** using the [official instructions](https://github.com/mees/calvin#-quick-start). The historical release uses **D→D**. The extracted input directory must contain `training/` and `validation/`, each with `ep_start_end_ids.npy` and `episode_*.npz` files.
2. Convert it using the included converter. Set `CALVIN_RAW` to your extracted D directory:

```bash
export CALVIN_RAW=/path/to/calvin/task_D_D
python training/adaptation/convert_calvin.py \
  --input "$CALVIN_RAW" --output ./data/converted \
  --dataset-name calvin_d --action-key rel_actions \
  --frame-stride 1 --clip-length 128 --min-clip-length 66
```

The converter needs NumPy, Pillow and PyTorch, included in the inference environment. It preserves source rollout boundaries, maps the seven action values into 13 slots, and resizes/pads images to 256×320. Full conversion can take time and disk space; it is a one-time step. The generated `manifest.json` records each clip's source episode.

3. Copy the first validation clip into the example location:

```bash
mkdir -p data/calvin_example
cp data/converted/calvin_d/val_eps_00000000.npz \
  data/calvin_example/episode.npz
```

Use the **downloaded** `weights/calvin/action_ranges.pth` with the released checkpoint. The converter also fits a range table in its output directory; that table is for training workflows and does not replace the checkpoint's matched ranges.

This clip is a convenience example. To reproduce an archived aggregate, use its recorded case list, seed and settings under [`artifacts/results/`](../artifacts/results/).

## Run and inspect

Install the [README environment](../README.md#quick-start), download the weights, then launch:

```bash
recap download --dataset calvin --root .
recap run --config configs/calvin_h64.json \
  --assets ./weights --data ./data/calvin_example \
  --output ./runs/calvin_demo --verify-manifest weights_manifest.json
```

The CALVIN output directory contains:

| File | What to inspect |
| :--- | :--- |
| `compare_all_strategies.gif` | Vertically stacked reference and prediction rows; see the row order in `result.json` |
| `frames.npz` | Ground-truth, strategy and repeat-frame arrays with frame indices |
| `result.json` | Strategy settings, timing and rollout metadata |
| `release_launch.json` | Resolved command, configuration and Python version |

The launcher refuses to overwrite an existing run. Pick a new output directory for each comparison. To inspect the command without loading models or requiring data, add `--dry-run`. `recap doctor` checks the installed environment; it does not certify GPU inference.

## Choose another dataset

Download the matched assets with `recap download --dataset NAME --root .`, then choose the corresponding config and converted input:

| `NAME` | Config | Input passed to `--data` | Protocol | Backend |
| :--- | :--- | :--- | :--- | :--- |
| `calvin` | [`calvin_h64.json`](../configs/calvin_h64.json) | Directory containing `episode.npz`, T ≥ 66 | 2 observed + 64 predicted | vLLM |
| `rt1` | [`rt1_h64.json`](../configs/rt1_h64.json) | Directory containing `episode.npz`, T ≥ 66 | 2 observed + 64 predicted | vLLM |
| `libero` | [`libero_h64.json`](../configs/libero_h64.json) | Directory containing `episode.npz`, T ≥ 66 | 2 observed + 64 predicted | vLLM |
| `bridge` | [`bridge_h32.json`](../configs/bridge_h32.json) | Converted Bridge validation directory | 1 observed + 32 predicted | Hugging Face |

Dataset sources and converters are listed in [DATA.md](DATA.md). RT-1 raw-to-NPZ conversion follows the upstream pipeline. LIBERO and Bridge require their dataset-specific converters; renaming a raw file is insufficient.

### Bridge example

Use a separate environment with a CUDA-compatible PyTorch installation and the HF extras; vLLM is unnecessary for this backend:

```bash
python -m pip install -e '.[inference,metrics,assets]'
recap download --dataset bridge --root .
recap run --config configs/bridge_h32.json \
  --assets ./weights --data ./data/bridge_v2 \
  --output ./runs/bridge_demo --verify-manifest weights_manifest.json
```

The Bridge recipe runs the first three eligible clips. It duplicates the single observation as the initial dynamic anchor, predicts 32 frames and checks that the first seven predictions agree before ReCAP evicts any history. See [`bridge_showcase_h32.json`](../configs/bridge_showcase_h32.json) for the separately selected illustration cases.

## Comparison protocol

- Supply the same observed images, future actions, checkpoint pair, seed, precision and sampling parameters to both methods.
- Feed generated tokens back into the world model. Do not use true future images or decode/re-encode generated pixels as rollout context.
- Treat tokenizer reconstruction as a separate codec reference. Prediction quality is an open-loop visual metric conditioned on recorded actions, not a closed-loop robot success rate.
- Use each archived run's original metadata when reproducing a table; the portable default configs do not encode every historical seed or case selection. Report aggregate metrics with the actual case count.

## Troubleshooting

| Symptom | Next step |
| :--- | :--- |
| Missing checkpoint or checksum mismatch | Re-run `recap download` for the chosen dataset; keep its tokenizer, world model and ranges together. |
| `episode.npz` missing, too short or invalid | Check the converted path, frame count and [input contract](DATA.md). |
| Output directory already exists | Choose a new `--output` path. |
| torch / vLLM dependency conflict | Use the separate backend environments described in [ENVIRONMENT.md](ENVIRONMENT.md). |
| Metrics differ from an archived result | Match its input window, seed, precision, backend and sampling settings; compare the [validation status](VALIDATION.md). |

The release has passed CPU and checkpoint-integrity checks. Fresh public-package GPU rollouts and the manuscript's expanded backbone sweep remain outside those checks; see [coverage](COVERAGE.md) and [validation](VALIDATION.md).
