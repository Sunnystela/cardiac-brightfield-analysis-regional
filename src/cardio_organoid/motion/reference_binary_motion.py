"""Reference-frame-difference binary movement channel (round 34).

An independent, second motion channel alongside (never a replacement for)
this project's existing Farneback-based optical-flow pipeline
(raft_flow.py/radial_flow.py/spatial_patches.py/boundary_motion.py, all
UNCHANGED). Where Farneback estimates a dense velocity FIELD, this module
only asks a much coarser binary question per pixel per frame: has this
pixel's brightness moved far enough from a reference image to call it
"moving" at all -- similar in spirit to Devarasetty et al. (2017)'s
bright-field movement analysis, but with the threshold made per-video and
noise-adaptive (see background_noise_threshold.py) rather than a fixed
literature constant.

    diff_t(x, y)   = abs(frame_t(x, y) - reference(x, y))
    binary_t(x, y) = 1 if diff_t(x, y) > threshold else 0

Every function here is a small, pure primitive operating on
already-computed arrays -- no video I/O, no SAM2, no Farneback -- so the
same functions apply unchanged to any reference strategy (see
reference_frame_strategies.py) or ROI/region mask.
"""

from __future__ import annotations

import numpy as np


def absolute_difference(frame_gray: np.ndarray, reference_gray: np.ndarray) -> np.ndarray:
    """abs(frame - reference), promoted to float64 so callers never
    silently wrap/clip on uint8 subtraction."""
    return np.abs(np.asarray(frame_gray, dtype=np.float64) - np.asarray(reference_gray, dtype=np.float64))


def binary_mask_from_diff(diff: np.ndarray, threshold: float) -> np.ndarray:
    """diff > threshold, boolean. `threshold=None` (e.g. an unavailable
    background estimate) raises rather than silently comparing against
    None."""
    if threshold is None:
        raise ValueError("threshold is None -- no valid background-adaptive threshold was computed")
    return diff > threshold


def moving_fraction(binary_mask: np.ndarray, gate_mask: np.ndarray | None = None) -> float | None:
    """Fraction of True pixels in `binary_mask`, restricted to
    `gate_mask` (e.g. the dilated-organoid analysis-region mask, or one
    region's patch-cell mask) if given. Returns None (not 0.0) if the
    gate has zero pixels -- an empty region is a data problem to surface,
    not a silent zero."""
    if gate_mask is None:
        gated = binary_mask
        denom = binary_mask.size
    else:
        gated = binary_mask[gate_mask]
        denom = int(gate_mask.sum())
    if denom == 0:
        return None
    return float(np.count_nonzero(gated) / denom)


def connected_component_stats(binary_mask: np.ndarray, gate_mask: np.ndarray | None = None) -> dict:
    """Descriptive spatial-structure summary of one binary movement frame
    (via cv2.connectedComponentsWithStats, 8-connectivity): number of
    connected "moving" components, their area distribution, and the
    largest component's share of the gated area -- distinguishes a FEW
    COMPACT high-motion blobs (localized strong motion) from MANY
    scattered small components or one large diffuse blob spanning most of
    the gate (broad weak/incoherent motion). Purely descriptive; no
    threshold or classification decision is made here.
    """
    import cv2

    mask_u8 = binary_mask.astype(np.uint8)
    if gate_mask is not None:
        mask_u8 = mask_u8 & gate_mask.astype(np.uint8)
        gate_area = int(gate_mask.sum())
    else:
        gate_area = mask_u8.size

    n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    # Label 0 is always the background component -- excluded.
    areas = stats[1:, cv2.CC_STAT_AREA] if n_labels > 1 else np.array([], dtype=int)
    if areas.size == 0 or gate_area == 0:
        return {
            "n_components": 0, "total_moving_area_px": 0, "median_component_area_px": None,
            "max_component_area_px": None, "largest_component_area_fraction_of_gate": 0.0,
        }
    return {
        "n_components": int(areas.size),
        "total_moving_area_px": int(areas.sum()),
        "median_component_area_px": float(np.median(areas)),
        "max_component_area_px": int(areas.max()),
        "largest_component_area_fraction_of_gate": float(areas.max() / gate_area),
    }


def accumulate_participation(accumulator: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    """Add one frame's binary movement mask into a running per-pixel
    participation-count accumulator (same shape, float). Returns a NEW
    array (does not mutate `accumulator` in place) so callers can use
    this as a plain reduction step."""
    return accumulator + binary_mask.astype(np.float64)


def participation_frequency(accumulator: np.ndarray, n_frames: int) -> np.ndarray:
    """Normalize a participation accumulator (see `accumulate_participation`)
    into a per-pixel frequency in [0, 1]: how often that pixel was
    classified 'moving' across `n_frames` processed frames."""
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    return accumulator / n_frames
