# Data and protocol contracts

For a step-by-step CALVIN conversion and first rollout, start with the [reproduction guide](REPRODUCE.md#prepare-calvin-data).

Inputs contain `image: uint8[T,256,320,3]` and finite `action: float32[T,13]`. Use converted episodes and their original split manifests. No cross-episode sampling; fit action ranges on the training split only. Do not use object-array pickles in NPZ inputs.

| Dataset | Action conversion | Protocol |
|---|---|---|
| RT-1 | Original 13-slot format and its own range table | x0 context, x1 observed dynamics; generate x2 onward |
| CALVIN D→D | translation→4–6, rotation→10–12, gripper→3, remaining zero | two observations, H64 recipe |
| LIBERO-90 | Preserve the existing converter and held-out task split | two observations; display-only vertical flip where specified |
| Bridge V2 | translation 0–2, rotation 3–5, gripper 6, slots 7–11 zero, terminate 12 | one observation in showcased H32 protocol |

A full block is 80 visual tokens + 13 action tokens. Context IDs are offset by 4375, action IDs by 8750. Action `a_t` follows visual `x_t` and conditions generation of `x_{t+1}`. Never reuse another dataset's action ranges.

CALVIN static-camera images are resized to 256×256 and edge-padded 32 pixels left/right. Preserve the existing preprocessing for the other datasets. Do not automatically flip LIBERO inputs or metric tensors because its display rendering uses a flip.

Converters live in `training/adaptation/`. Read each `--help`; Bridge conversion verifies records and can fit ranges, but validation must reuse training ranges. Dataset acquisition and redistribution are governed by their upstream terms. The packaged GIFs are research illustrations, not substitutes for the source datasets.

Historical CALVIN here is **D→D**, not ABC→D. LIBERO-90 uses 81 training tasks / 9 held-out validation tasks. Maintain original episode/source grouping when computing uncertainty; frames and adjacent clips are not independent samples. Archived historical statistics retain their original case-level aggregation rather than being silently recalculated under a different protocol.

## Obtain source data

Use the publishers' download instructions and preserve their dataset terms:

| Dataset | Official source | Reference subset |
|---|---|---|
| RT-1 | [TFDS fractal20220817_data](https://www.tensorflow.org/datasets/catalog/fractal20220817_data), [Open X-Embodiment](https://github.com/google-deepmind/open_x_embodiment) | `fractal20220817_data` episodes; record the sampling/split manifest |
| CALVIN | [Official repository and download instructions](https://github.com/mees/calvin#-quick-start) | `dataset/download_data.sh D`; retain official training/validation splits |
| LIBERO | [Official dataset downloader](https://github.com/Lifelong-Robot-Learning/LIBERO#download-datasets) | LIBERO-90 from the official download; code MIT, dataset CC BY 4.0 per publisher |
| Bridge V2 | [BridgeData V2 downloads](https://rail-berkeley.github.io/bridgedata/) | Converted Bridge V2 episodes and explicit split manifests |

The source release links the full datasets rather than redistributing them. Code licenses and website-footer licenses must not be assumed to govern a dataset. Retain upstream attribution and the terms accompanying each actual download.

Use `python training/adaptation/convert_calvin.py --help`, `convert_libero.py --help`, or `convert_bridge.py --help` for the included converters. Converters require dataset-specific dependencies (such as h5py or TensorFlow); use a separate conversion environment when these conflict with inference. RT-1 follows the upstream RLVR-World/Open X-Embodiment conversion pipeline; an RT-1 raw-to-NPZ converter is not included here.
