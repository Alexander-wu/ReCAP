# Runtime and dependency policy

The CPU scheduling, launcher, checksums and source-audit utilities require only Python 3.10+. The original experiment environment used Python 3.13, PyTorch 2.9.1 / CUDA 12.9 and vLLM 0.13.0. Inspect `runtime_verified.json` for the actual release-preparation environment once recorded. `pyproject.toml` pins central experiment APIs but is not a full transitive CUDA lockfile.

Use separate environments for HF/Bridge reference inference and vLLM experiments: vLLM may constrain/replace torch. Install a CUDA-compatible torch build before the other extras, and keep existing training environments untouched. Do not treat the CPU GitHub workflow as a GPU validation result.

Required GPU tests before tagging a stable release: clean installation; one real paired sample per dataset; first seven Bridge predictions identical before eviction; zero illegal visual tokens; metrics and archive schemas; bounded context length; controlled seed/precision; explicit CUDA synchronization for timing. RQ4 performance must be measured on its stated hardware and concurrency/APC setting.
