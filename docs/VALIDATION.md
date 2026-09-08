# Validation for the robot reference release

Completed 2026-09-09. These are release integrity and CPU execution checks, not a new GPU quality benchmark.

## Local package checks

- A clean Python 3.12 virtual environment installs the editable package successfully.
- **10 CPU regression tests pass**: 64-frame reference scheduler comparison, input/action mapping, output overwrite guard, complete checkpoint pairing, download resume, corruption detection and path traversal/symlink protection.
- Source audit checks Python/JSON validity and scans public text, config, CSV and HTML for credentials, private endpoints and excluded experiment identifiers. No findings.
- Source distribution and wheel build successfully. A second isolated environment installs the wheel with no dependencies and imports the public scheduler and bundled inference backends from outside the source checkout.
- Documentation relative links resolve. Upstream and embedded component license texts are bundled.

## Server CPU checks

- The same 10 regression tests pass on the existing server Python 3.13 runtime, using an isolated source copy.
- **36 of 36 asset files** match their recorded size and SHA-256.
- **Eight of eight model exports load with `strict=True`**, without missing or unexpected state-dict keys.
- All four world models produce finite logits in a CPU forward pass. Each has 127,101,696 parameters and vocabulary size 9,008. Each tokenizer has 129,448,016 parameters.
- All four action-range tables have shape `(13, 2)`, finite values and valid min/max ordering.
- No GPU was requested or used for these checks.

Machine-readable results: [checkpoint_validation.json](checkpoint_validation.json). Reproduce with:

```bash
python scripts/validate_checkpoints.py --root ASSET_BUNDLE_ROOT \
  --manifest weights_manifest.json --output checkpoint_validation.json
```

## Public Hugging Face release

The complete bundle is public at [easylearning/recap-robot-models](https://huggingface.co/easylearning/recap-robot-models), with asset revision `61f3c350349f58ba8619c2454b5fac5802756282`.

- All **36 asset files** match their expected byte counts and SHA-256 values on the Hub. Large files use Hub LFS SHA-256 metadata; ordinary files were downloaded anonymously and hashed.
- A fresh anonymous download of the complete RT-1 tokenizer/world-model/action-range pair succeeded through the release downloader; all **nine files** passed integrity verification.
- Machine-readable results: [hf_public_verification.json](hf_public_verification.json).

## Historical evidence and remaining limits

Archived experiment outputs come from the original reference scripts. Earlier preparation checked reference entry-point imports on the existing H20 runtime. These checks do not establish that a fresh CUDA environment exactly reproduces all recorded rollouts or timings.

A fresh public-package GPU paired rollout for each dataset, complete retraining, distributed resume recovery, and reproductions of expanded manuscript backbones have not been run for this release. Report the actual runtime, seed, precision and case selection when reproducing a table. The Bridge reference asserts identical predictions through the first seven frames before eviction and records out-of-range visual token counts.
