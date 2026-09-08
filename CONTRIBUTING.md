# Contributing

Use an isolated environment. Run `python -m unittest discover -s tests -v`, `python scripts/audit_release.py`, and `python -m compileall -q src` before submitting changes. Add regression coverage for changes to block boundaries, action alignment, eviction, cache positions, or metrics. Record dataset protocol and seed in bug reports; do not upload private episodes, credentials or checkpoints in issues.

Do not change archived quantitative results when modifying code. New runs belong in a separate output directory with their own manifest and runtime. Keep inference and training APIs separate; benchmark claims require measured hardware/runtime provenance.
