# Robot model inventory

ReCAP adds no trained parameters. Inference needs a matched tokenizer, world model and action-range table for each dataset. The four world models are approximately 127M parameters each. Eight model exports and four range pairs total **4,141,635,395 bytes**; `weights_manifest.json` records 36 exact files.

| Dataset | Tokenizer | World model | Action ranges |
|---|---|---|---|
| RT-1 | [Card](rt1_tokenizer.md) | [Card](rt1_world.md) | `weights/rt1/` |
| CALVIN D→D | [Card](calvin_tokenizer.md) | [Card](calvin_world.md) | `weights/calvin/` |
| LIBERO-90 | [Card](libero_tokenizer.md) | [Card](libero_world.md) | `weights/libero/` |
| Bridge V2 | [Card](bridge_tokenizer.md) | [Card](bridge_world.md) | `weights/bridge/` |

The verified fine-tuned exports are prepared for separate Hugging Face hosting; upload is deferred to a subsequent release. **A public bundle URL has not yet been assigned.** The manifest contains no invented model ID. The official starting checkpoints are available at [RT-1 tokenizer](https://huggingface.co/thuml/rt1-compressive-tokenizer) and [RT-1 world model](https://huggingface.co/thuml/rt1-world-model-multi-step-rlvr); these do not replace dataset-specific fine-tuned checkpoints or their action ranges.

Once a real bundle and revision are provided, download only your matched dataset assets:

```bash
python -m pip install -e '.[assets]'
recap download --dataset calvin --root . --manifest weights_manifest.json \
  --repo-id OWNER/PUBLISHED_BUNDLE --revision IMMUTABLE_COMMIT
recap verify --root . --manifest weights_manifest.json
```

`OWNER/PUBLISHED_BUNDLE` and `IMMUTABLE_COMMIT` are placeholders. Download verifies the selected dataset immediately; `recap verify` above checks the full manifest and requires all four datasets to be downloaded. Valid files are reused on subsequent downloads. An unpublished manifest without an explicit destination fails clearly rather than fetching substitute weights.

The resulting structure is `weights/calvin_tokenizer/`, `weights/calvin_world/`, `weights/calvin/action_ranges.pth` and `.json`, with analogous paths for the other datasets. Full model tensors are excluded from Git. All exports omit optimizer, discriminator and RNG states and are intended for inference, not training resume.

## Maintainer publication

`scripts/publish_weights.py --root RELEASE_ROOT --manifest weights_manifest.json --repo-id OWNER/BUNDLE` verifies the exact bundle and prints a plan. Add `--upload` after authenticating with `hf auth login` on the machine containing the exports. It uploads only manifest-listed assets, reviewed model cards and license files, and prints the resulting commit. Record that immutable commit under `distribution.repo_id` / `distribution.revision` in the manifest before advertising the default download command.

Checkpoint exports retain MIT starting-model terms. New ReCAP source code uses Apache-2.0; dataset terms remain with each data publisher. See [third-party notices](../THIRD_PARTY_NOTICES.md).
