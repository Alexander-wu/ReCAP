---
license: mit
language:
  - en
tags:
  - recap
  - robotics
  - video-prediction
  - world-model
base_model:
  - thuml/rt1-compressive-tokenizer
  - thuml/rt1-world-model-multi-step-rlvr
---

# calvin_world

Dataset-specific world model for ReCAP visual open-loop experiments.

- Training: 20,000 tokenizer steps + 20,000 world-model steps. CALVIN D→D.
- Initialization: thuml/rt1-compressive-tokenizer (codec) and thuml/rt1-world-model-multi-step-rlvr (world model).
- Format: huggingface; full strict-load weights, no optimizer/RNG state in inference release.
- Pair only with `calvin_tokenizer`, `calvin_world`, and `calvin_action_ranges` from the same manifest.
- Download: [matched model files](https://huggingface.co/easylearning/recap-robot-models/tree/61f3c350349f58ba8619c2454b5fac5802756282/weights/calvin_world).
- License: MIT, retaining the starting model terms and attribution. This card does not grant rights over the training dataset. The Apache-2.0 code license is separate from checkpoint licensing.
- Intended use: research video prediction with supplied actions. Not validated as a deployed robot controller.
- Metrics: see artifacts/results; selected GIFs are not representative aggregate estimates.
- Integrity: exact per-file SHA-256 and byte counts are in weights_manifest.json.

ReCAP is training-free context scheduling and has no additional model parameters.
