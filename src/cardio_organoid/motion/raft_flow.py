"""Dense optical flow over a motion ROI (Phase 2 prototype).

Architecture decision (see docs/analysis_design.md #6 and
configs/default.yaml: motion.roi): motion analysis is deliberately
decoupled from precise segmentation. It does not use the morphology mask's
exact contour -- only its coarse centroid and bounding box (see
`cardio_organoid.segmentation.sam2_segmenter.mask_centroid` /
`mask_bounding_box`) -- expanded into a generously sized, configurable
rectangular ROI. Optical flow is then computed over the *whole* ROI, not
masked to any organoid silhouette, so an imperfect or overly tight
segmentation cannot silently cut off real contractile motion (e.g.
protruding tissue during beating).

Baseline vs RAFT: this module started with OpenCV's classical Farneback
dense optical flow as the simplest possible baseline (deterministic, no
model weights, easy to sanity-check). RAFT is now implemented as well --
see `raft_backend.py` -- but behind its own function
(`raft_backend.compute_raft_flow`), not this module's
`compute_dense_optical_flow`, because its inputs and preprocessing differ
too much to share a signature (RGB frames, a loaded model/transforms/
device, and a downscale-then-upscale step RAFT needs at this project's ROI
resolution -- see that module's docstring for why). Farneback and RAFT are
compared directly on the same frame pairs and spatial patches in
scripts/compare_raft_vs_farneback_spatial.py.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from cardio_organoid.segmentation.sam2_segmenter import (
    mask_bounding_box,
    mask_centroid,
)

FARNEBACK_DEFAULT_PARAMS: dict = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)


@dataclasses.dataclass
class MotionROI:
    """A coarse, deliberately imprecise region used for optical flow.

    Derived only from a morphology mask's centroid and bounding box, never
    its exact contour -- see the module docstring. Coordinates are in the
    full-frame reference frame.
    """

    centroid_yx: tuple[float, float]
    organoid_bbox_xyxy: tuple[int, int, int, int]
    roi_bbox_xyxy: tuple[int, int, int, int]

    def centroid_xy_in_roi(self) -> tuple[float, float]:
        """Centroid expressed as (x, y) local to `roi_bbox_xyxy`'s crop,
        for use with a flow field that was computed on that crop.
        """
        roi_x0, roi_y0, _, _ = self.roi_bbox_xyxy
        cy, cx = self.centroid_yx
        return cx - roi_x0, cy - roi_y0


def compute_motion_roi(
    morphology_mask: np.ndarray,
    expansion_px: int,
    frame_shape: tuple[int, int],
) -> MotionROI:
    """Derive a MotionROI from `morphology_mask`'s centroid and bounding
    box, expanded by `expansion_px` on each side and clipped to
    `frame_shape`. Does not use the mask's exact contour.

    `expansion_px` must come from configs/default.yaml (motion.roi.expansion_px)
    -- it is an unvalidated placeholder, not tuned against real data yet.
    """
    centroid = mask_centroid(morphology_mask)
    bbox = mask_bounding_box(morphology_mask)
    if centroid is None or bbox is None:
        raise ValueError("morphology_mask is empty; cannot derive a motion ROI")

    x0, y0, x1, y1 = bbox
    h, w = frame_shape[:2]
    roi_bbox = (
        max(0, x0 - expansion_px),
        max(0, y0 - expansion_px),
        min(w, x1 + expansion_px),
        min(h, y1 + expansion_px),
    )
    return MotionROI(centroid_yx=centroid, organoid_bbox_xyxy=bbox, roi_bbox_xyxy=roi_bbox)


def _crop_roi(frame_gray: np.ndarray, roi_bbox_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = roi_bbox_xyxy
    return frame_gray[y0:y1, x0:x1]


def compute_dense_optical_flow(
    frame_t_gray: np.ndarray,
    frame_t1_gray: np.ndarray,
    roi: MotionROI,
    method: str = "opencv_farneback",
    **method_kwargs,
) -> np.ndarray:
    """Compute a dense (vx, vy) flow field between two consecutive
    grayscale frames, restricted to `roi.roi_bbox_xyxy` (the coarse
    rectangular ROI, not any mask). Returns an array of shape
    (roi_h, roi_w, 2).

    method="opencv_farneback" (default, Phase 2 baseline): OpenCV's
    classical dense optical flow (cv2.calcOpticalFlowFarneback). No model
    weights, deterministic, fast -- meant as the reference baseline before
    RAFT is evaluated.

    method="raft": not implemented yet. Intended future backend behind
    this same interface, for direct before/after comparison against the
    Farneback baseline on the same clips.
    """
    if method == "opencv_farneback":
        import cv2

        crop_t = _crop_roi(frame_t_gray, roi.roi_bbox_xyxy)
        crop_t1 = _crop_roi(frame_t1_gray, roi.roi_bbox_xyxy)
        params = {**FARNEBACK_DEFAULT_PARAMS, **method_kwargs}
        return cv2.calcOpticalFlowFarneback(crop_t, crop_t1, None, **params)
    if method == "raft":
        raise NotImplementedError(
            "RAFT is implemented, but not behind this function -- its "
            "inputs differ (RGB frames, a loaded model/transforms/device, "
            "and an internal downscale-then-upscale step; see the module "
            "docstring in raft_backend.py for why). Call "
            "raft_backend.compute_raft_flow(...) directly instead."
        )
    raise ValueError(f"Unknown optical flow method: {method!r}")
