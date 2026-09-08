# Release coverage

This release contains the verified robot reference implementation and archived experiments. All four exported world models use the approximately **127M-parameter** RLVR-World-derived architecture (12 layers, width 768). Tokenizers and world models are paired per dataset.

| Dataset | Released protocol | Model provenance |
|---|---|---|
| RT-1 | Two observed frames, H64 | Official RT-1 tokenizer and RLVR world model |
| CALVIN | D→D, two observed frames, H64 | RT-1 initialization; tokenizer 20k + world model 20k adaptation |
| LIBERO | LIBERO-90, 81 training / 9 held-out tasks, H64 | RT-1 initialization; tokenizer 20k + world model 20k adaptation |
| BridgeData V2 | One observed frame, H32 | RT-1 initialization; tokenizer 60k + world model 100k adaptation |

The manuscript's broader 0.5B–8B model sweep, RobotAlign-R1 backbone and associated expanded results are **not reproduced by this release**. The archived numbers here describe this reference experiment set; they must not be substituted for those broader claims. No large-backbone checkpoint is bundled or advertised as available.

## Archived results

The RQ2 H64 reference sample includes 9 RT-1, 16 CALVIN and 18 LIBERO cases. Example LPIPS means (Full History → ReCAP) are RT-1 0.2329 → 0.2042 and CALVIN 0.1591 → 0.1061. Consult the CSVs and run metadata under `artifacts/results/` for metrics, seeds and case identities.

RQ4 records the original hardware/runtime benchmark: H64 prompt size 7,232 → 1,931 tokens; calculated KV storage 254.25 → 67.89 MiB. Throughput measurements apply to the archived concurrency and caching conditions, not arbitrary hardware. See `docs/EXPERIMENTS.md` for proxy baseline and truncation details.

Training files are an archived implementation with documented recipes. The release checks do not establish distributed resume correctness or a complete retraining reproduction. ReCAP scheduling itself requires no training.
