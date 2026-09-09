# Integrate the context scheduler

[← README](../README.md) · [Reference backends](EXPERIMENTS.md)

The scheduler requires only Python 3.10+: `python -m pip install -e .`. It holds token IDs and has no tensor, model or GPU dependency.

## Reference token layout

| Component | Tokens |
| :--- | ---: |
| Static visual context | 1,280 |
| First dynamic anchor: 80 visual + 13 action | 93 |
| Each retained recent visual–action block | 93 |
| Maximum prompt with W = 6 | **1,931** |

Each action block follows its visual frame and conditions the next prediction. Generated visual tokens remain in token space throughout the rollout.

## Decoder integration

This sketch assumes your tokenizer has produced `context_ids` and `anchor_ids`, and your decoder provides `sample_frame` and `rebuild_cache`. These functions belong to your backend, not the ReCAP package:

```python
from recap import ReCAPContext

scheduler = ReCAPContext(context_ids, anchor_ids, window=6)
for action_ids in future_actions:
    # Prefill the selected prompt with contiguous, rebased positions.
    # Rebuilding on every step is the simple reference integration.
    cache = rebuild_cache(model, scheduler.prompt())
    visual_ids = sample_frame(model, cache)  # exactly 80 visual tokens
    scheduler.append(visual_ids, action_ids)  # exactly 13 action tokens
```

`append()` returns whether a recent block was evicted. An optimized backend may extend a valid cache between evictions, but it must rebuild with rebased positions after eviction. Slicing raw KV tensors without repairing positional state is not equivalent. `ReCAPContext` manages tokens only; the [provided inference backends](../src/recap/reference/) implement prompt rebuilding.

The [CPU regression suite](../tests/test_release.py) compares each scheduler prompt against the preserved reference over 64 prediction steps. Development checks are listed in [CONTRIBUTING.md](../CONTRIBUTING.md).
