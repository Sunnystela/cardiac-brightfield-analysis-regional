"""Diagnostic comparison of VIDEO morphology reference-frame strategies
-- how sensitive is a video's morphology measurement to WHICH frame (or
frame-blend) it is computed on, given the organoid contracts/relaxes
throughout the recording?

Per explicit project instruction: no reference strategy here is tuned to
match any manual value -- every strategy is purely self-relative to the
video's own frames, reusing `motion.reference_frame_strategies.py`
(round 34) UNCHANGED for the actual reference-construction primitives.
Segmentation reuses `segmentation.sam2_segmenter.segment_frame` and the
SAME fixed provisional candidate index this project uses everywhere else
(never re-picked by a morphology metric). Morphology feature extraction
reuses `morphology.features.compute_morphology_features` UNCHANGED.

Three candidate strategies compared (never silently picking a "best" one
-- see `compare_reference_strategies`, which reports every strategy's
own morphology values side by side for human review):

  A. minimum-global-motion / resting frame: the single REAL captured
     frame whose local (smoothed) frame-to-frame motion is smallest.
  B. temporal median: a per-pixel median image over a subsample of the
     video's own frames -- a synthetic (never actually captured) blended
     image, robust to any single frame's transient contraction phase.
  C. robust median of several low-motion frames: median over the frames
     with the K smallest motion-proxy values (not just the single
     minimum) -- a middle ground between A (one frame, real but
     potentially noisy) and B (every frame, blended but never
     phase-selective).

Also reports the CURRENT PIPELINE DEFAULT (round-1-era `pipeline/
analyze.py`'s own convention: the video's midpoint frame,
frame_count // 2) as a fourth reference point for comparison -- this is
what morphology has been computed on so far project-wide, per
`morphology/features.py`'s own PILOT_DISCLAIMER.
"""

from __future__ import annotations

import cv2
import numpy as np

from cardio_organoid.pipeline.segmentation_interface import assert_legacy_fixed_index_allowed
from cardio_organoid.morphology.features import compute_morphology_features
from cardio_organoid.motion.reference_frame_strategies import (
    select_minimum_motion_frame_index,
    temporal_median_reference,
)


def _temporal_median_reference_rgb(frames_rgb: np.ndarray) -> np.ndarray:
    """`reference_frame_strategies.temporal_median_reference` strictly
    validates a (N, H, W) grayscale-shaped input; this project's
    morphology reference needs an RGB image (SAM2 expects 3 channels).
    Applies the SAME UNCHANGED per-pixel-median formula one channel at a
    time (mathematically identical to a single call over (N, H, W, 3) --
    numpy's median along axis 0 does not mix channels either way) rather
    than modifying that function's shape contract."""
    channels = [temporal_median_reference(frames_rgb[..., c]) for c in range(frames_rgb.shape[-1])]
    return np.stack(channels, axis=-1)


from cardio_organoid.segmentation.sam2_segmenter import provisional_primary_qc, segment_frame

STRATEGY_MIDPOINT_CURRENT_DEFAULT = "midpoint_frame_current_pipeline_default"
STRATEGY_MINIMUM_MOTION = "minimum_motion_resting_frame"
STRATEGY_TEMPORAL_MEDIAN = "temporal_median_image"
STRATEGY_ROBUST_LOW_MOTION_MEDIAN = "robust_low_motion_frames_median"


def read_video_frames_rgb(video_path, stride: int = 1, max_frames: int | None = None) -> np.ndarray:
    """Reads every `stride`-th frame as RGB uint8, up to `max_frames`
    (None = whole video) -- a documented, traceable subsample, same
    convention as `reference_frame_strategies.py`'s own module
    docstring. Returns shape (N, H, W, 3)."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            if max_frames and len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    return np.stack(frames) if frames else np.zeros((0, 0, 0, 3), dtype=np.uint8)


def compute_motion_proxy_series(frames_rgb: np.ndarray) -> np.ndarray:
    """Simple, Farneback-INDEPENDENT motion proxy (round-34 philosophy):
    mean absolute grayscale frame-to-frame difference. Length is
    `len(frames_rgb) - 1`; index i is the motion between frame i and
    frame i+1."""
    if len(frames_rgb) < 2:
        return np.zeros(0)
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float64) for f in frames_rgb])
    return np.mean(np.abs(np.diff(gray, axis=0)), axis=(1, 2))


def select_reference_candidates(
    video_path, stride: int = 3, smoothing_window_frames: int = 5, n_low_motion_frames_for_robust_median: int = 5,
) -> dict:
    """Builds all 4 reference-frame candidates for one video (see module
    docstring). Returns {"frames_used_for_selection": N, "candidates":
    {strategy_name: {"image_rgb", "source_frame_indices"}}}."""
    frames = read_video_frames_rgb(video_path, stride=stride)
    n = len(frames)
    if n == 0:
        return {"frames_used_for_selection": 0, "candidates": {}}

    motion = compute_motion_proxy_series(frames)
    candidates = {
        STRATEGY_MIDPOINT_CURRENT_DEFAULT: {"image_rgb": frames[n // 2], "source_frame_indices": [n // 2]},
    }
    if len(motion) > 0:
        min_motion_idx = select_minimum_motion_frame_index(motion, smoothing_window_frames)
        # motion[i] is the transition frame[i]->frame[i+1]; use frame i as
        # the representative "resting" frame for that low-motion transition.
        candidates[STRATEGY_MINIMUM_MOTION] = {"image_rgb": frames[min_motion_idx], "source_frame_indices": [min_motion_idx]}

        k = min(n_low_motion_frames_for_robust_median, len(motion))
        low_motion_indices = sorted(np.argsort(motion)[:k].tolist())
        candidates[STRATEGY_ROBUST_LOW_MOTION_MEDIAN] = {
            "image_rgb": _temporal_median_reference_rgb(frames[low_motion_indices]).astype(np.uint8),
            "source_frame_indices": low_motion_indices,
        }

    candidates[STRATEGY_TEMPORAL_MEDIAN] = {
        "image_rgb": _temporal_median_reference_rgb(frames).astype(np.uint8),
        "source_frame_indices": list(range(n)),
    }
    return {"frames_used_for_selection": n, "motion_proxy_series": motion.tolist(), "candidates": candidates}


def compare_reference_strategies(video_path, config: dict, predictor) -> dict:
    """Runs SAM2 segmentation + morphology feature extraction on EVERY
    reference-frame candidate for one video, and reports every strategy's
    own values side by side -- no strategy is picked as "correct" here;
    that judgment is left to the human reviewing this diagnostic output.
    Also reports each non-default strategy's RELATIVE difference from
    the current-pipeline-default (midpoint frame) strategy, as a direct
    answer to "how sensitive is video morphology to reference choice"."""
    sam2_cfg = config["segmentation"]["provisional_primary"]
    selection = select_reference_candidates(video_path)
    results = {}
    for strategy, cand in selection["candidates"].items():
        assert_legacy_fixed_index_allowed("pipeline.video_morphology_reference")
        seg_candidates = segment_frame(predictor, cand["image_rgb"])
        qc = provisional_primary_qc(seg_candidates, sam2_cfg["candidate_index"], sam2_cfg["min_dominant_component_fraction"])
        if qc.get("flag") == "invalid_index":
            results[strategy] = {"morphology": None, "qc": qc, "source_frame_indices": cand["source_frame_indices"]}
            continue
        mask = seg_candidates[sam2_cfg["candidate_index"]].mask
        features = compute_morphology_features(mask)
        results[strategy] = {
            "morphology": {
                "area_px": features.pixel.area_px, "perimeter_px": features.pixel.perimeter_px,
                "equivalent_diameter_px": features.pixel.equivalent_diameter_px,
                "circularity": features.pixel.circularity, "solidity": features.pixel.solidity,
                "eccentricity": features.pixel.eccentricity, "aspect_ratio": features.pixel.aspect_ratio,
                "radial_cv": features.pixel.radial_cv,
            },
            "qc": qc, "source_frame_indices": cand["source_frame_indices"],
        }

    baseline = results.get(STRATEGY_MIDPOINT_CURRENT_DEFAULT, {}).get("morphology")
    relative_diff_from_default = {}
    if baseline:
        for strategy, r in results.items():
            if strategy == STRATEGY_MIDPOINT_CURRENT_DEFAULT or not r["morphology"]:
                continue
            relative_diff_from_default[strategy] = {
                field: (abs(r["morphology"][field] - baseline[field]) / baseline[field])
                for field in baseline if baseline[field] not in (None, 0) and r["morphology"].get(field) is not None
            }

    return {
        "video": str(video_path), "n_frames_used_for_selection": selection["frames_used_for_selection"],
        "results_by_strategy": results, "relative_diff_from_current_default": relative_diff_from_default,
    }
