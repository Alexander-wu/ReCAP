# Robot model adaptation

ReCAP scheduling is training-free. This directory preserves the tokenizer and world-model adaptation code used to obtain the dataset-specific frozen models.

- `train_ctx_tokenizer.py`: compressive tokenizer training, including adversarial reconstruction.
- `train_vgpt.py` / `train_vgpt_finetune.py`: action-conditioned autoregressive world-model training.
- `adaptation/`: dataset conversion, explicit split patches and a fresh Accelerate seed builder.
- `recipes/*.py.txt`: historical run recipes; path placeholders must be configured before use. They are evidence of settings, not ready-to-run deployment scripts.

Read [experiment protocols](../docs/EXPERIMENTS.md) before adapting these scripts. The inference exports omit optimizer and RNG state. To start fine-tuning, use `adaptation/seed_checkpoint.py --help` to construct a fresh seed, and use the corresponding dataset's action ranges. Apply patch scripts only to an isolated copy of this training snapshot. Do not treat an inference export as a resumable training checkpoint.

Use `configs/ctx_vae256/config.json` for the compressive tokenizer and `configs/vgpt/ctx_llama_small.json` for the released world-model layout. Paired inference and evaluation live in [`src/recap/reference/`](../src/recap/reference/); start with the [reproduction guide](../docs/REPRODUCE.md).

This snapshot needs additional training and conversion dependencies beyond the inference package. Complete retraining and distributed resume tests are outside this release's validated scope. Upstream code attribution is in [third-party notices](../THIRD_PARTY_NOTICES.md).
