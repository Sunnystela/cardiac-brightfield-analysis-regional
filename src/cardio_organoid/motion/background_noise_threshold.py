"""Background-patch-based, noise-adaptive threshold estimation for the
reference-frame binary movement channel (round 34).

Motivation: a reference-difference "moving pixel" diagnostic (Devarasetty
et al. 2017-style bright-field movement analysis) needs a threshold on
abs(frame - reference) to call a pixel "moving". This project does not
replicate that paper's fixed threshold -- every video here has its own
illumination, sensor noise, and compression characteristics, so a single
literature constant would not be self-relative to this data (the same
convention already used throughout this project's other threshold
choices, e.g. beat_events.robust_noise_floor, activity_significance.py).

Per-video noise is instead estimated from PIXELS THAT SHOULD NOT BE MOVING
(background patches measurably outside the organoid, screened by
background_reference.select_background_patches_multi_ring -- the SAME
already-decided, unrelated-to-Farneback utility this project already uses
for its own noise-floor estimates). The resulting median/MAD are a robust
(outlier-resistant) summary of "how much do genuinely static pixels appear
to move, purely from sensor/compression/illumination noise" -- never
derived from any organoid-region pixel.

This module contains ONLY the small numeric primitives (robust stats +
threshold-candidate construction). It does not select background patches
or read video frames -- see scripts/reference_binary_motion_diagnostic.py
for how it is wired together with background_reference.py.
"""

from __future__ import annotations

import numpy as np

# Gaussian-equivalent scale factor for MAD -> std, same constant already
# used elsewhere in this project (beat_events.robust_noise_floor,
# boundary_motion.robust_mean) -- kept consistent, not re-derived.
MAD_TO_STD_SCALE = 1.4826


def robust_background_stats(diff_values: np.ndarray) -> dict:
    """Robust (median/MAD-based) summary of a sample of background-patch
    abs-difference values. Both the raw MAD and the Gaussian-equivalent
    scaled MAD are reported (per project instruction) so a threshold can
    be built from either without re-deriving the scale factor.

    Returns None fields (not zeros) when `diff_values` is empty -- an
    empty background sample is a QC condition to report, not silently
    default to a threshold of 0.
    """
    values = np.asarray(diff_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "n_samples": 0, "median": None, "mad_raw": None, "mad_scaled": None,
            "mean": None, "std": None, "p95": None, "max": None,
        }
    median = float(np.median(values))
    mad_raw = float(np.median(np.abs(values - median)))
    return {
        "n_samples": int(values.size),
        "median": median,
        "mad_raw": mad_raw,
        "mad_scaled": mad_raw * MAD_TO_STD_SCALE,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def threshold_candidates(stats: dict, k_values: list[float]) -> dict[str, float | None]:
    """threshold_k = stats['median'] + k * stats['mad_scaled'] for each k
    in `k_values` -- a self-relative, per-video/per-reference-strategy
    threshold, never a fixed literature constant. Uses the
    Gaussian-equivalent scaled MAD (not raw MAD) as the dispersion term,
    consistent with this project's other robust-z-style thresholds (e.g.
    activity_significance.significance_threshold's null_mean + z*null_std
    convention). Returns {f"k={k}": None, ...} for every k if
    `stats['median']`/`stats['mad_scaled']` is None (empty background
    sample) rather than silently computing a threshold of 0.
    """
    median, mad_scaled = stats.get("median"), stats.get("mad_scaled")
    if median is None or mad_scaled is None:
        return {f"k={k}": None for k in k_values}
    return {f"k={k}": float(median + k * mad_scaled) for k in k_values}


def moving_fraction_sensitivity(
    diff_values: np.ndarray, thresholds: dict[str, float | None]
) -> dict[str, float | None]:
    """For a SAMPLE of diff values (e.g. background patches, or an
    organoid ROI at one representative frame), the fraction exceeding
    each candidate threshold -- lets a caller see, before committing to
    any candidate, whether a k is so low it flags a large fraction of
    KNOWN-STATIC background as 'moving' (too sensitive -- picking up pure
    noise) or so high that almost nothing ever exceeds it (too
    conservative). Purely descriptive; does not select a threshold."""
    values = np.asarray(diff_values, dtype=float)
    values = values[np.isfinite(values)]
    result: dict[str, float | None] = {}
    for label, threshold in thresholds.items():
        if threshold is None or values.size == 0:
            result[label] = None
        else:
            result[label] = float(np.mean(values > threshold))
    return result
