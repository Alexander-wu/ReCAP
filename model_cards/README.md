# Robot model inventory

ReCAP adds no trained parameters. Inference needs a matched tokenizer, world model and action-range table for each dataset. The four world models are approximately 127M parameters each. Eight model exports and four range pairs total **4,141,635,395 bytes**; `weights_manifest.json` records 36 exact files.

| Dataset | Tokenizer | World model | Action ranges |
|---|---|---|---|
| RT-1 | [Card](rt1_tokenizer.md) | [Card](rt1_world.md) | `weights/rt1/` |
| CALVIN D→D | [Card](calvin_tokenizer.md) | [Card](calvin_world.md) | `weights/calvin/` |
| LIBERO-90 | [Card](libero_tokenizer.md) | [Card](libero_world.md) | `weights/libero/` |
| Bridge V2 | [Card](bridge_tokenizer.md) | [Card](bridge_world.md) | `weights/bridge/` |

All eight exports and their action ranges are publicly available in **[easylearning/recap-robot-models](https://huggingface.co/easylearning/recap-robot-models)**. The manifest pins immutable revision [`61f3c350349f`](https://huggingface.co/easylearning/recap-robot-models/commit/61f3c350349f58ba8619c2454b5fac5802756282) and preserves every published asset's SHA-256 and byte count. Public downloads require no Hugging Face login.

From the ReCAP checkout, download a matched dataset pair:

```bash
python -m pip install -e '.[assets]'
recap download --dataset calvin --root .
```

Choose `rt1`, `calvin`, `libero`, or `bridge`. The command downloads the tokenizer, world model and action-range files together and verifies all nine files immediately. Existing valid files are reused. To download and verify the full bundle:

```bash
for dataset in rt1 calvin libero bridge; do
  recap download --dataset "$dataset" --root .
done
recap verify --root . --manifest weights_manifest.json
```

The official starting checkpoints remain available at [RT-1 tokenizer](https://huggingface.co/thuml/rt1-compressive-tokenizer) and [RT-1 world model](https://huggingface.co/thuml/rt1-world-model-multi-step-rlvr). They are provenance references; use the matched fine-tuned pair and action ranges for each dataset.

The resulting structure is `weights/calvin_tokenizer/`, `weights/calvin_world/`, `weights/calvin/action_ranges.pth` and `.json`, with analogous paths for the other datasets. Full model tensors are excluded from Git. All exports omit optimizer, discriminator and RNG states and are intended for inference, not training resume.

Checkpoint exports retain MIT starting-model terms. New ReCAP source code uses Apache-2.0; dataset terms remain with each data publisher. See [third-party notices](../THIRD_PARTY_NOTICES.md).
