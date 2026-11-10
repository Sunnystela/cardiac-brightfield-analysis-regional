"""Whole-organoid boundary-band motion waveform (round 30).

Independent of the patch-grid-based per-region analysis
(spatial_patches.py / pipeline/analyze.py, UNCHANGED by this module),
this builds ONE pooled radial-motion waveform from a configurable band
straddling the organoid's morphology-mask CONTOUR (spatial_gate.
boundary_band_mask, already existing/unchanged), intended to capture
whole-organoid coordinated boundary contraction as a signal independent
from local/regional patch-based beating.

Per project decision, the morphology mask/contour is taken ONCE from the
video's existing representative-frame SAM2 segmentation (the same single
inference pipeline/analyze.py already performs) -- this module never
re-segments per frame. The same static contour/ROI/centroid is reused for
every frame pair for the whole video, exactly like the existing patch-
grid pipeline already does with its own dilated analysis-region mask.

Every per-frame-pair optical-flow step (Farneback, global-drift removal,
centroid-relative radial projection) reuses raft_flow.py/radial_flow.py
UNCHANGED -- this module only adds a NEW spatial pooling gate (the
boundary band) and NEW per-frame summary statistics on top of the
already-decided motion pipeline; it does not introduce a new flow method
or change any existing threshold.
"""

from __future__ import annotations

import numpy as np


def robust_mean(values: np.ndarray, mad_z: float = 3.0) -> float:
    """Mean after excluding values farther than `mad_z` normal-consistent
    MADs from the median -- self-relative (per-frame, per-region) outlier
    trimming, never an absolute magic value. Falls back to the plain mean
    if the MAD is zero (all values identical) or too few values remain
    for a meaningful median (fewer than 2)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return float(np.mean(arr)) if arr.size else float("nan")
    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    robust_std = 1.4826 * mad
    if robust_std == 0:
        return float(np.mean(arr))
    kept = arr[np.abs(arr - median) <= mad_z * robust_std]
    return float(np.mean(kept)) if kept.size else float(np.mean(arr))


def boundary_band_frame_stats(gated_radial_field: np.ndarray) -> dict:
    """Per-frame-pair summary of a radial-motion field ALREADY restricted
    to the boundary band (non-band pixels expected to be NaN, e.g. via
    spatial_gate.apply_gate). Signed convention unchanged from
    radial_flow.py: negative = inward, positive = outward -- this
    function does not decide which is contraction.

    Returns median, a robust (MAD-trimmed) mean, and the mean magnitude
    of the inward-only and outward-only subsets (each None if that
    subset is empty for this frame pair)."""
    values = gated_radial_field[~np.isnan(gated_radial_field)]
    if values.size == 0:
        return {"median_radial_flow": None, "robust_mean_radial_flow": None,
                "inward_magnitude": None, "outward_magnitude": None}
    inward = values[values < 0]
    outward = values[values > 0]
    return {
        "median_radial_flow": float(np.median(values)),
        "robust_mean_radial_flow": robust_mean(values),
        "inward_magnitude": float(np.mean(np.abs(inward))) if inward.size else None,
        "outward_magnitude": float(np.mean(outward)) if outward.size else None,
    }


def compute_boundary_motion_waveforms(frame_stats: list[dict]) -> dict[str, np.ndarray]:
    """Assembles a list of per-frame boundary_band_frame_stats dicts
    (one per processed frame pair, in order) into 4 parallel waveform
    arrays. NaN-fills any frame pair where the band happened to be empty
    (should not normally happen for a non-degenerate band) so downstream
    code always gets fixed-length arrays aligned with every other
    per-frame signal in this project."""
    def _series(key: str) -> np.ndarray:
        return np.array([s[key] if s[key] is not None else np.nan for s in frame_stats], dtype=float)

    return {
        "median_radial_flow": _series("median_radial_flow"),
        "robust_mean_radial_flow": _series("robust_mean_radial_flow"),
        "inward_magnitude": _series("inward_magnitude"),
        "outward_magnitude": _series("outward_magnitude"),
    }
