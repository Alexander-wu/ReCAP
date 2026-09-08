"""Robot trajectory and shape metrics for video world-model rollouts.

Implements the three trajectory-consistency metrics used in the ReCAP main
table, all derived from a single arm segmentation step:

  arm_score   Fraction of frames where the predicted rollout still contains a
              plausible robot arm, and that arm sits near the ground-truth arm.
              Directly measures the "the arm vanished" failure mode.

  shape_iou   Mean intersection-over-union between the predicted and the
              ground-truth arm masks.

  ndtw        Normalized dynamic time warping between the predicted and the
              ground-truth arm-centroid trajectories, following the
              vision-language-navigation formulation of Ilharco et al. (2019):

                  nDTW = exp( -DTW(P, R) / (threshold * |R|) )

              DTW is the minimum cumulative Euclidean distance over monotone
              alignments; `threshold` is a success radius in normalized image
              units. nDTW is 1.0 for a perfect match and decays toward 0.

Segmentation approach
---------------------
RT-1 / BridgeV2 episodes are recorded from a fixed camera, so the temporal
median of the *ground-truth* episode is a good static-background estimate. A
pixel is foreground when it deviates from that background; the largest
connected component is taken as the arm.

The single most important detail: the **same background reference and the same
threshold are used for the prediction and for the ground truth**. Segmenting
each sequence with its own statistics would make the masks incomparable, since
a collapsed rollout has almost no temporal variation and would produce a
degenerate background.

No learned detector is involved, so these metrics carry no extra model bias,
but they do assume a static camera and a robot that is visually distinct from
the scene. `mask_quality` in the output reports how often segmentation found a
component of plausible size, so unreliable episodes can be filtered.
"""

import numpy as np

try:
    from scipy import ndimage
    HAVE_SCIPY = True
except ImportError:  # pragma: no cover - exercised only on minimal envs
    HAVE_SCIPY = False

# Foreground percentile of the |frame - background| distribution. The arm plus
# the objects it moves occupy roughly 5-15% of the RT-1 frame, so 88 keeps the
# arm while rejecting sensor noise and floor texture.
FOREGROUND_PERCENTILE = 88.0
MIN_ABS_DIFF = 12.0          # 0-255 scale; below this a pixel is never foreground
MIN_AREA_FRACTION = 0.004    # smaller components are treated as "no arm found"
MAX_AREA_FRACTION = 0.45     # larger ones mean the segmentation degenerated
# Success radius for nDTW, in normalized image units (fraction of frame
# diagonal). 0.10 means "within 10% of the diagonal counts as on-track".
NDTW_THRESHOLD = 0.10
# Arm Score counts a frame as correct when the predicted centroid lands within
# this normalized distance of the ground-truth centroid.
ARM_SCORE_RADIUS = 0.12


def _to_uint8_sequence(frames):
    """Accept float [0,1] or uint8 [0,255] arrays shaped [T, H, W, 3]."""
    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"expected [T, H, W, 3], got {array.shape}")
    if array.dtype == np.uint8:
        return array.astype(np.float32)
    array = array.astype(np.float32)
    if float(array.max()) <= 1.5:
        array = array * 255.0
    return array


def estimate_background(reference_frames):
    """Static-background estimate from the ground-truth episode."""
    return np.median(_to_uint8_sequence(reference_frames), axis=0)


def _clean_mask(mask):
    """Drop speckle, close gaps, keep the largest component."""
    if not HAVE_SCIPY:
        return mask
    mask = ndimage.binary_opening(mask, np.ones((3, 3)))
    mask = ndimage.binary_closing(mask, np.ones((7, 7)))
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    return labels == (int(np.argmax(sizes)) + 1)


def segment_arms(frames, background, threshold):
    """Return (masks [T, H, W] bool, centroids [T, 2] normalized, valid [T] bool).

    Centroids are (y, x) divided by (height, width), so both lie in [0, 1].
    Invalid frames (no plausible component) get NaN centroids.
    """
    sequence = _to_uint8_sequence(frames)
    count, height, width = sequence.shape[:3]
    deviation = np.abs(sequence - background[None]).mean(axis=3)

    masks = np.zeros((count, height, width), dtype=bool)
    centroids = np.full((count, 2), np.nan, dtype=np.float64)
    valid = np.zeros(count, dtype=bool)

    for index in range(count):
        raw = (deviation[index] > threshold) & (deviation[index] > MIN_ABS_DIFF)
        mask = _clean_mask(raw)
        area = float(mask.mean())
        if MIN_AREA_FRACTION <= area <= MAX_AREA_FRACTION:
            rows, cols = np.nonzero(mask)
            centroids[index] = (rows.mean() / height, cols.mean() / width)
            valid[index] = True
            masks[index] = mask
        else:
            # Keep the mask for IoU bookkeeping but mark the frame unreliable.
            masks[index] = mask if area <= MAX_AREA_FRACTION else False
    return masks, centroids, valid


def dtw_distance(path_a, path_b):
    """Minimum cumulative Euclidean distance over monotone alignments."""
    a = np.asarray(path_a, dtype=np.float64)
    b = np.asarray(path_b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    cost = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
    rows, cols = cost.shape
    table = np.full((rows + 1, cols + 1), np.inf)
    table[0, 0] = 0.0
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            table[i, j] = cost[i - 1, j - 1] + min(
                table[i - 1, j], table[i, j - 1], table[i - 1, j - 1]
            )
    return float(table[rows, cols])


def normalized_dtw(prediction_path, reference_path, threshold=NDTW_THRESHOLD):
    """nDTW = exp(-DTW / (threshold * |reference|)); 1.0 is perfect."""
    if len(prediction_path) == 0 or len(reference_path) == 0:
        return 0.0
    distance = dtw_distance(prediction_path, reference_path)
    if not np.isfinite(distance):
        return 0.0
    return float(np.exp(-distance / (threshold * len(reference_path))))


def _interpolate_gaps(centroids, valid):
    """Fill short invalid runs so DTW sees a continuous path."""
    if not valid.any():
        return np.zeros((0, 2))
    filled = centroids.copy()
    indices = np.arange(len(centroids))
    for axis in range(2):
        filled[:, axis] = np.interp(indices, indices[valid], centroids[valid, axis])
    return filled


def trajectory_metrics(prediction_frames, ground_truth_frames,
                       ndtw_threshold=NDTW_THRESHOLD,
                       arm_radius=ARM_SCORE_RADIUS):
    """Arm Score, Shape IoU and nDTW for one rollout against its ground truth.

    Both sequences are segmented with a background and threshold derived from
    the ground truth, so the masks are directly comparable.
    """
    prediction = _to_uint8_sequence(prediction_frames)
    truth = _to_uint8_sequence(ground_truth_frames)
    span = min(len(prediction), len(truth))
    prediction, truth = prediction[:span], truth[:span]

    background = np.median(truth, axis=0)
    deviation = np.abs(truth - background[None]).mean(axis=3)
    threshold = float(np.percentile(deviation, FOREGROUND_PERCENTILE))

    pred_masks, pred_centroids, pred_valid = segment_arms(prediction, background, threshold)
    true_masks, true_centroids, true_valid = segment_arms(truth, background, threshold)

    both_valid = pred_valid & true_valid
    distances = np.full(span, np.nan)
    if both_valid.any():
        delta = pred_centroids[both_valid] - true_centroids[both_valid]
        distances[both_valid] = np.sqrt((delta ** 2).sum(axis=1))

    # Arm Score: over frames where the ground truth has a visible arm, how
    # often does the prediction also show one, close to the right place?
    scorable = true_valid.sum()
    if scorable:
        hits = both_valid & (distances <= arm_radius)
        arm_score = float(hits.sum() / scorable)
        detection_rate = float((pred_valid & true_valid).sum() / scorable)
    else:
        arm_score = float("nan")
        detection_rate = float("nan")

    intersection = np.logical_and(pred_masks, true_masks).sum(axis=(1, 2))
    union = np.logical_or(pred_masks, true_masks).sum(axis=(1, 2))
    per_frame_iou = np.where(union > 0, intersection / np.maximum(union, 1), 0.0)
    shape_iou = float(per_frame_iou[true_valid].mean()) if scorable else float("nan")

    pred_path = _interpolate_gaps(pred_centroids, pred_valid)
    true_path = _interpolate_gaps(true_centroids, true_valid)
    ndtw = normalized_dtw(pred_path, true_path, ndtw_threshold) if len(true_path) else float("nan")

    # Late-horizon slice, where collapse shows up.
    late = slice(int(round(span * 0.70)), span)
    late_true = true_valid[late]
    if late_true.sum():
        late_hits = (both_valid[late] & (distances[late] <= arm_radius))
        arm_score_late = float(late_hits.sum() / late_true.sum())
        shape_iou_late = float(per_frame_iou[late][late_true].mean())
    else:
        arm_score_late = float("nan")
        shape_iou_late = float("nan")

    if len(pred_path) and len(true_path):
        half = span // 2
        ndtw_late = normalized_dtw(pred_path[half:], true_path[half:], ndtw_threshold)
    else:
        ndtw_late = float("nan")

    return {
        "ndtw": round(ndtw, 4),
        "ndtw_late_half": round(ndtw_late, 4),
        "arm_score": round(arm_score, 4),
        "arm_score_late": round(arm_score_late, 4),
        "shape_iou": round(shape_iou, 4),
        "shape_iou_late": round(shape_iou_late, 4),
        "arm_detection_rate": round(detection_rate, 4),
        "frames_scored": int(span),
        "ground_truth_arm_frames": int(scorable),
        "mask_quality": round(float(true_valid.mean()), 4),
        "centroid_distance_mean": (
            round(float(np.nanmean(distances)), 5) if both_valid.any() else None),
        "per_frame_centroid_distance": [
            None if np.isnan(value) else round(float(value), 5) for value in distances],
        "per_frame_shape_iou": [round(float(value), 4) for value in per_frame_iou],
        "prediction_centroids": [
            None if not ok else [round(float(y), 5), round(float(x), 5)]
            for ok, (y, x) in zip(pred_valid, pred_centroids)],
        "ground_truth_centroids": [
            None if not ok else [round(float(y), 5), round(float(x), 5)]
            for ok, (y, x) in zip(true_valid, true_centroids)],
        "segmentation": {
            "foreground_percentile": FOREGROUND_PERCENTILE,
            "threshold": round(threshold, 3),
            "min_abs_diff": MIN_ABS_DIFF,
            "ndtw_threshold": ndtw_threshold,
            "arm_score_radius": arm_radius,
            "background_source": "ground_truth_temporal_median",
            "scipy_available": HAVE_SCIPY,
        },
    }


def self_test():
    """Sanity checks with synthetic sequences of known behaviour."""
    height, width, count = 64, 80, 24
    rng = np.random.default_rng(0)

    def scene(positions):
        frames = np.full((count, height, width, 3), 40, dtype=np.uint8)
        frames += rng.integers(0, 6, frames.shape, dtype=np.uint8)
        for index, (cy, cx) in enumerate(positions):
            y0, x0 = int(cy) - 6, int(cx) - 6
            frames[index, max(y0, 0):y0 + 12, max(x0, 0):x0 + 12] = 230
        return frames

    moving = [(20 + i, 20 + i) for i in range(count)]
    truth = scene(moving)

    identical = trajectory_metrics(truth, truth)
    assert identical["ndtw"] > 0.99, identical["ndtw"]
    assert identical["arm_score"] > 0.99, identical["arm_score"]
    assert identical["shape_iou"] > 0.99, identical["shape_iou"]

    # Frozen rollout: arm stops at the first position (the collapse mode).
    frozen = trajectory_metrics(scene([moving[0]] * count), truth)
    assert frozen["ndtw"] < identical["ndtw"], (frozen["ndtw"], identical["ndtw"])
    assert frozen["arm_score"] < 0.6, frozen["arm_score"]

    # Vanished arm: nothing but background.
    empty = np.full((count, height, width, 3), 40, dtype=np.uint8)
    gone = trajectory_metrics(empty, truth)
    assert gone["arm_score"] == 0.0, gone["arm_score"]
    assert gone["shape_iou"] < 0.05, gone["shape_iou"]

    # Small jitter should stay close to perfect.
    jitter = trajectory_metrics(scene([(y + 1, x - 1) for y, x in moving]), truth)
    assert jitter["ndtw"] > frozen["ndtw"], (jitter["ndtw"], frozen["ndtw"])

    # DTW must be symmetric and zero on identical paths.
    path = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.3]])
    assert dtw_distance(path, path) == 0.0
    other = np.array([[0.0, 0.0], [0.2, 0.2]])
    assert abs(dtw_distance(path, other) - dtw_distance(other, path)) < 1e-9

    print("traj_metrics self-test passed")
    print(f"  identical : ndtw={identical['ndtw']:.4f} arm={identical['arm_score']:.3f} "
          f"iou={identical['shape_iou']:.3f}")
    print(f"  jitter    : ndtw={jitter['ndtw']:.4f} arm={jitter['arm_score']:.3f} "
          f"iou={jitter['shape_iou']:.3f}")
    print(f"  frozen    : ndtw={frozen['ndtw']:.4f} arm={frozen['arm_score']:.3f} "
          f"iou={frozen['shape_iou']:.3f}")
    print(f"  vanished  : ndtw={gone['ndtw']:.4f} arm={gone['arm_score']:.3f} "
          f"iou={gone['shape_iou']:.3f}")


if __name__ == "__main__":
    self_test()
