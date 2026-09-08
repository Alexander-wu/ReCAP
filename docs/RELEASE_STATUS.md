# Release status

Version 0.1.0 is the robot reference code release. New ReCAP additions use Apache-2.0; third-party notices are preserved. See `docs/COVERAGE.md` for the supported experiment scope.

- Code destination: https://github.com/Alexander-wu/ReCAP
- Project page: https://alexander-wu.github.io/recap-project-page/
- Checkpoints: eight inference exports plus four matched action-range pairs, checksummed in `weights_manifest.json`. Fine-tuned Hub upload is deferred to a subsequent release; the verified bundle remains staged separately.
- Verification: see `docs/VALIDATION.md` for completed release checks and limits.
- Raw datasets, optimizer states, credentials and deployment infrastructure are not part of the code distribution.

Hosting status for weights is intentionally independent of code publication. `recap download` requires an actual Hub bundle and revision and verifies every downloaded file. Once hosting is available, its immutable commit will be recorded in the manifest.
