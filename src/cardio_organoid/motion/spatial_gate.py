"""Spatial gating strategies for reducing a motion ROI's radial-flow field
to a single waveform value (Phase 2 signal-extraction comparison).

The rectangular motion ROI (raft_flow.compute_motion_roi) is deliberately
coarse and includes non-organoid material (dish background, peripheral
debris). Averaging radial motion over the *whole* ROI can let that
peripheral motion dominate the signal. This module provides three ways to
restrict which pixels of a computed flow/radial-motion field are averaged
when building a waveform value, so they can be compared against each other
(scripts/compare_motion_signal_extraction.py) without changing how optical
flow itself is computed (still over the whole coarse ROI, per the
segmentation-independence decision in raft_flow.py):

1. "rectangular_roi": no gating -- every pixel in the ROI contributes
   (the current baseline).
2. "exact_mask": only pixels inside the precise morphology mask contribute.
   This reintroduces a dependency on segmentation precision for the
   *aggregation* step only (not for ROI construction) -- exactly what is
   being evaluated here, not assumed.
3. "dilated_mask": pixels inside the morphology mask expanded by a
   configurable margin contribute, as a middle ground intended to include
   protruding contractile tissue just outside the compact mask while still
   excluding most distant peripheral debris.
4. "boundary_band": only a band straddling the mask contour (inward and
   outward) contributes. Motivated by an observed pattern (not assumed a
   priori): optical flow signal was consistently much stronger near the
   organoid boundary than deep in its optically uniform interior, plausibly
   because contraction shows up as boundary displacement more readily than
   interior texture deformation, and/or because a gradient-based method
   like Farneback needs an edge to lock onto. The band's inner/outer
   margins are configurable *ratios of the organoid's own equivalent
   diameter* (never a fixed pixel count), so the same config value scales
   across organoids of different sizes.
"""

from __future__ import annotations

import numpy as np


def crop_mask_to_roi(
    mask: np.ndarray, roi_bbox_xyxy: tuple[int, int, int, int]
) -> np.ndarray:
    """Crop a full-frame mask to a motion ROI's bounding box, so it aligns
    pixel-for-pixel with a flow field computed on that same crop."""
    x0, y0, x1, y1 = roi_bbox_xyxy
    return mask[y0:y1, x0:x1]


def dilate_mask(mask: np.ndarray, dilation_px: int) -> np.ndarray:
    """Expand `mask` by `dilation_px` pixels (elliptical structuring
    element). `dilation_px <= 0` returns a copy unchanged."""
    import cv2

    if dilation_px <= 0:
        return mask.copy()
    kernel_size = 2 * dilation_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def erode_mask(mask: np.ndarray, erosion_px: int) -> np.ndarray:
    """Shrink `mask` by `erosion_px` pixels (elliptical structuring
    element). `erosion_px <= 0` returns a copy unchanged."""
    import cv2

    if erosion_px <= 0:
        return mask.copy()
    kernel_size = 2 * erosion_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.erode(mask.astype(np.uint8), kernel).astype(bool)


def boundary_band_mask(
    mask: np.ndarray,
    equivalent_diameter_px: float,
    inner_ratio: float,
    outer_ratio: float,
) -> np.ndarray:
    """A band straddling `mask`'s contour: everything within
    `outer_ratio * equivalent_diameter_px` pixels outside the mask, plus
    everything within `inner_ratio * equivalent_diameter_px` pixels inside
    it. Margins are ratios of the mask's own equivalent diameter, not fixed
    pixel counts, so the band scales with organoid size across videos.
    """
    outer_margin_px = int(round(outer_ratio * equivalent_diameter_px))
    inner_margin_px = int(round(inner_ratio * equivalent_diameter_px))
    outer_bound = dilate_mask(mask, outer_margin_px)
    inner_bound = erode_mask(mask, inner_margin_px)
    return outer_bound & ~inner_bound


def apply_gate(field: np.ndarray, gate_mask: np.ndarray | None) -> np.ndarray:
    """Return a copy of `field` with pixels outside `gate_mask` set to NaN,
    so it can be reduced with the same nanmean-based waveform builder used
    for the ungated case. `gate_mask=None` returns `field` unchanged
    (the "rectangular_roi" / no-gating strategy).
    """
    if gate_mask is None:
        return field
    return np.where(gate_mask, field, np.nan)
