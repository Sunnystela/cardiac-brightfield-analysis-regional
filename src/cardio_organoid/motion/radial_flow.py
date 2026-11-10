"""Global-drift-corrected radial motion relative to the organoid centroid
(Phase 2 prototype).

Per docs/analysis_design.md #7-8, raw optical flow over the motion ROI
(see raft_flow.py) can contain organoid translation, plate vibration,
camera movement, and focus artifacts on top of true contractile
deformation. Pipeline:

1. Take the dense flow field from `raft_flow.compute_dense_optical_flow`.
2. `remove_global_drift`: subtract an estimated global drift/translation
   vector, isolating residual/internal motion. Two estimators are compared
   (Phase 2 validation, not yet a settled choice): the whole-ROI median
   (default, `reference_gate=None`) and a median restricted to a
   `reference_gate` of screened static background patches (see
   `background_reference.py`) -- the latter exists specifically to check
   whether the whole-ROI estimate was accidentally absorbing some of the
   organoid's own net motion into "drift" and subtracting it away.
3. `radial_motion_from_centroid`: project the drift-corrected flow onto the
   radial direction from the organoid centroid at every pixel. Inward
   (negative) = contraction, outward (positive) = relaxation.

Both raw motion magnitude and radial motion are retained by callers (not
just the radial signal), per docs/analysis_design.md #8 -- this module
only computes the radial signal; magnitude is a trivial norm of the input
flow field if a caller also wants it.
"""

from __future__ import annotations

import numpy as np


def remove_global_drift(
    flow_field: np.ndarray, reference_gate: np.ndarray | None = None
) -> tuple[np.ndarray, tuple[float, float]]:
    """Subtract an estimated global drift vector from every pixel in
    `flow_field` (shape (H, W, 2)), returning (corrected_flow, drift_xy).

    `reference_gate=None` (default, preserves prior behavior): drift is the
    per-component median over the *whole* field. Rationale
    (docs/analysis_design.md #7, "potential strategy: estimate global
    organoid translation, subtract translation component"): if the
    organoid's own contraction is approximately radially symmetric, its
    contribution to the ROI's spatial median flow should be close to zero,
    so the whole-field median approximates translation shared by the whole
    scene (plate/stage/camera drift) rather than contraction itself. This
    is a first-pass heuristic, not validated against ground truth.

    `reference_gate=<boolean mask>`: drift is instead the median over only
    the gated pixels (e.g. screened static background patches, see
    `background_reference.select_background_patches`) -- a Phase 2
    validation check for whether the whole-field estimate above was
    partly picking up the organoid's own net motion (which would then get
    incorrectly subtracted away as if it were drift).
    """
    if reference_gate is None:
        vx, vy = flow_field[..., 0], flow_field[..., 1]
    else:
        vx = flow_field[..., 0][reference_gate]
        vy = flow_field[..., 1][reference_gate]
    drift_x = float(np.median(vx))
    drift_y = float(np.median(vy))
    corrected = flow_field.copy()
    corrected[..., 0] -= drift_x
    corrected[..., 1] -= drift_y
    return corrected, (drift_x, drift_y)


def radial_motion_from_centroid(
    flow_field: np.ndarray, centroid_xy: tuple[float, float]
) -> np.ndarray:
    """Project `flow_field` (shape (H, W, 2), ideally already drift-
    corrected) onto the radial direction from `centroid_xy` (local pixel
    coordinates matching the flow field's crop) at every pixel.

    Returns an (H, W) array; negative = inward (contraction), positive =
    outward (relaxation), NaN at the centroid pixel itself (undefined
    direction).
    """
    h, w = flow_field.shape[:2]
    cx, cy = centroid_xy
    ys, xs = np.mgrid[0:h, 0:w]
    dx = xs - cx
    dy = ys - cy
    radius = np.hypot(dx, dy)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit_x = dx / radius
        unit_y = dy / radius
    return flow_field[..., 0] * unit_x + flow_field[..., 1] * unit_y


def tangential_motion_from_centroid(
    flow_field: np.ndarray, centroid_xy: tuple[float, float]
) -> np.ndarray:
    """Project `flow_field` (shape (H, W, 2), ideally already drift-
    corrected) onto the TANGENTIAL direction (perpendicular to the radial
    direction, counter-clockwise-positive) from `centroid_xy` at every
    pixel -- the complement of `radial_motion_from_centroid`, added
    (round-33 diagnostic) so a flow field can be decomposed into
    radial + tangential components without re-deriving the unit vectors.

    Sign convention: at a pixel whose radial unit vector is (unit_x,
    unit_y), the tangential unit vector is (-unit_y, unit_x) -- a
    counter-clockwise 90-degree rotation of the radial direction (image
    y-axis pointing down, per this project's pixel-coordinate convention).
    This is purely a geometric decomposition; no biological meaning
    (contraction/relaxation/rotation) is assigned to either sign.

    Returns an (H, W) array, NaN at the centroid pixel itself (undefined
    direction), same as `radial_motion_from_centroid`.
    """
    h, w = flow_field.shape[:2]
    cx, cy = centroid_xy
    ys, xs = np.mgrid[0:h, 0:w]
    dx = xs - cx
    dy = ys - cy
    radius = np.hypot(dx, dy)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit_x = dx / radius
        unit_y = dy / radius
    return flow_field[..., 0] * (-unit_y) + flow_field[..., 1] * unit_x


def radial_tangential_decomposition(
    flow_field: np.ndarray, centroid_xy: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper returning (radial, tangential) fields together
    (see `radial_motion_from_centroid` / `tangential_motion_from_centroid`
    for conventions), so a caller decomposing flow into both components
    does not have to recompute the shared unit-vector geometry twice."""
    return radial_motion_from_centroid(flow_field, centroid_xy), tangential_motion_from_centroid(flow_field, centroid_xy)
