# Qualitative cases

`gallery.html` contains the 12 project-page GIFs: three each for RT-1, CALVIN, LIBERO and Bridge. These are chosen illustrations. `artifacts/cases/website_cases.json` provides the existing website metadata; Bridge selection and all candidate metrics are in `artifacts/results/bridge/`. `appendix_selection.json` records the strong/representative/boundary selection definitions for the earlier multi-dataset appendix.

Bridge selected cases are 18, 5 and 15 (the exact order and source filenames are preserved in `selected_cases.json` and `configs/bridge_showcase_h32.json`). They are selected after evaluation for illustrative gain, not randomly selected test results. Consult the selection manifest rather than guessing the episode from the GIF filename.

Raw NPZ illustration archives are excluded from the source release; the original project retains them separately. A keyframe NPZ contains sampled rendered frames and cannot necessarily regenerate a full rollout: it may omit actions or intermediate frames. Full reruns require the original converted episode with actions and the matched weights. Do not infer temporal metrics from a subsampled illustration.

To regenerate the gallery: `python scripts/preview_cases.py`. For paper layout, use nine evenly spaced display frames (Bridge H32: t=0,4,8,12,16,20,24,28,32; H64: t=0,8,...,64). Preserve the manuscript's EB Garamond typography and instruction labels when generating publication figures. The gallery intentionally retains the already exported original GIF pixels.
