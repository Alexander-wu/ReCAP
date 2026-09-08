# Release status

Version 0.1.0 is the robot reference code release. New ReCAP additions use Apache-2.0; third-party notices are preserved. See `docs/COVERAGE.md` for the supported experiment scope.

- Code destination: https://github.com/Alexander-wu/ReCAP
- Project page: https://alexander-wu.github.io/recap-project-page/
- Checkpoints: eight inference exports plus four matched action-range pairs, checksummed in `weights_manifest.json`. Published at [easylearning/recap-robot-models](https://huggingface.co/easylearning/recap-robot-models), pinned to `61f3c350349f58ba8619c2454b5fac5802756282`.
- Verification: see `docs/VALIDATION.md` for completed release checks and limits.
- Raw datasets, optimizer states, credentials and deployment infrastructure are not part of the code distribution.

The code and model weights are distributed separately. `recap download --dataset calvin --root .` reads the pinned Hub revision from the release manifest and verifies every downloaded file. No login is required for these public weights.
