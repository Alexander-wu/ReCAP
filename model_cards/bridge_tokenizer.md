---
license: mit
language:
  - en
tags:
  - recap
  - robotics
  - video-prediction
  - tokenizer
base_model:
  - thuml/rt1-compressive-tokenizer
  - thuml/rt1-world-model-multi-step-rlvr
---

# bridge_tokenizer

Dataset-specific tokenizer for ReCAP visual open-loop experiments.

- Training: 60,000 tokenizer steps + 100,000 world-model steps. BridgeData V2.
- Initialization: thuml/rt1-compressive-tokenizer (codec) and thuml/rt1-world-model-multi-step-rlvr (world model).
- Format: diffusers; full strict-load weights, no optimizer/RNG state in inference release.
- Pair only with `bridge_tokenizer`, `bridge_world`, and `bridge_action_ranges` from the same manifest.
- Download: [matched model files](https://huggingface.co/easylearning/recap-robot-models/tree/61f3c350349f58ba8619c2454b5fac5802756282/weights/bridge_tokenizer).
- License: MIT, retaining the starting model terms and attribution. This card does not grant rights over the training dataset. The Apache-2.0 code license is separate from checkpoint licensing.
- Intended use: research video prediction with supplied actions. Not validated as a deployed robot controller.
- Metrics: see artifacts/results; selected GIFs are not representative aggregate estimates.
- Integrity: exact per-file SHA-256 and byte counts are in weights_manifest.json.

ReCAP is training-free context scheduling and has no additional model parameters.
