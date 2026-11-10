"""Morphology feature extraction from a single organoid mask (Phase 1 pilot).

Implements the initial measurements from docs/analysis_design.md #5 and
CLAUDE.md #7: area, equivalent diameter, circularity, solidity,
eccentricity/aspect ratio, and radial irregularity.

PILOT STATUS -- read before using any output of this module:

- The mask this operates on is the segmentation module's *provisional*
  primary candidate (a fixed SAM2 multimask_output index, currently
  configured in configs/default.yaml: segmentation.provisional_primary).
  It matched a compact organoid body on 9/9 videos reviewed so far, but is
  documented and QC-checked as PROVISIONAL, not confirmed -- see
  `sam2_segmenter.provisional_primary_qc`. Do not describe it elsewhere as
  a confirmed/validated organoid mask.
- The frame this operates on is an arbitrary representative frame (the
  current placeholder: each video's midpoint frame), not a cardiac-phase-
  normalized one. Cardiac-phase normalization -- choosing a relaxed/
  end-diastolic frame from the contraction waveform, per
  docs/analysis_design.md #4 -- requires the motion module, which does not
  exist yet. Every result this module produces will be recomputed once
  that frame selection exists, and every output this module writes must
  carry a copy of `PILOT_DISCLAIMER` so a reader of the raw artifact (not
  just the code) sees this without cross-referencing.
- Consequently: values produced now must not be used for condition
  comparison (e.g. slit vs. conventional) or any other biological
  conclusion. They exist to validate that the measurement code itself is
  correct and sane on real images.

Reuses circularity/solidity from the segmentation module rather than
reimplementing the same formulas, so the two never silently diverge (see
segmentation/sam2_segmenter.py's note on why those are computed there in
the first place: descriptive diagnostics, never used to select a mask).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from cardio_organoid.segmentation.sam2_segmenter import circularity, solidity

PILOT_DISCLAIMER = (
    "PILOT result: computed on an arbitrary representative frame (video "
    "midpoint, not phase-normalized) using a PROVISIONAL primary organoid "
    "mask (see segmentation.provisional_primary_qc). Not for condition "
    "comparison or biological conclusions. Will be recomputed once a "
    "relaxed-phase frame can be selected from the motion/contraction "
    "waveform (docs/analysis_design.md #4)."
)


@dataclasses.dataclass
class PixelMorphologyFeatures:
    """Dimensionless / pixel-unit measurements -- always available, since
    they need no calibration."""

    area_px: int
    perimeter_px: float | None
    equivalent_diameter_px: float
    circularity: float | None
    solidity: float | None
    eccentricity: float | None
    aspect_ratio: float | None
    radial_cv: float | None


@dataclasses.dataclass
class PhysicalMorphologyFeatures:
    """Physical-unit measurements. Only constructed when a pixel_size_um
    calibration is supplied -- see MorphologyFeatures.physical."""

    pixel_size_um: float
    area_um2: float
    equivalent_diameter_um: float


@dataclasses.dataclass
class MorphologyFeatures:
    """`physical` is None until pixel-to-micron calibration is available
    (io/metadata.py calibration parsing is not implemented yet) -- kept as
    a separate, optional block rather than mixed into `pixel` so a future
    calibration source can populate it without changing the pixel-unit
    fields' meaning or names.
    """

    pixel: PixelMorphologyFeatures
    physical: PhysicalMorphologyFeatures | None
    pilot_disclaimer: str = PILOT_DISCLAIMER


def area(mask: np.ndarray) -> int:
    """Projected organoid area, in pixels."""
    return int(mask.sum())


def perimeter(mask: np.ndarray) -> float | None:
    """Largest-contour perimeter, in pixels (`cv2.arcLength`, closed
    contour) -- exposed as its own field since `segmentation.
    sam2_segmenter.circularity` already computes this internally but did
    not previously return it. None if the mask is empty."""
    import cv2

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    return float(cv2.arcLength(largest, closed=True))


def equivalent_diameter(area_px: float) -> float:
    """Diameter of a circle with the same area as `area_px`."""
    return float(2.0 * np.sqrt(area_px / np.pi))


def eccentricity_and_aspect_ratio(
    mask: np.ndarray,
) -> tuple[float | None, float | None]:
    """Fit an ellipse to the mask's largest contour and return
    (eccentricity, aspect_ratio = major_axis / minor_axis). None if the
    mask is empty or too small to fit an ellipse (cv2 requires >= 5 points).
    """
    import cv2

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, None
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return None, None
    _center, (axis_a, axis_b), _angle = cv2.fitEllipse(largest)
    major_axis, minor_axis = max(axis_a, axis_b), min(axis_a, axis_b)
    if major_axis == 0:
        return None, None
    aspect_ratio = major_axis / minor_axis if minor_axis > 0 else None
    eccentricity = float(np.sqrt(max(0.0, 1.0 - (minor_axis / major_axis) ** 2)))
    return eccentricity, aspect_ratio


def radial_profile(mask: np.ndarray, num_angles: int = 360) -> np.ndarray | None:
    """Distance from the mask centroid to the mask boundary, sampled at
    `num_angles` evenly spaced angles (docs/analysis_design.md #5, "Radial
    Irregularity": "sample distances from centroid to boundary across
    angular positions"). None if the mask is empty.

    Ray-intersection definition (precise): for each angle theta_i =
    2*pi*i/num_angles, a ray is cast from the mask's centroid (cy, cx) in
    1-pixel steps: (x, y) = (cx + step*cos(theta_i), cy + step*sin(theta_i))
    for step = 1, 2, ... up to the frame diagonal. The recorded radius at
    that angle is the *outermost* step for which the ray point still lies
    inside the mask -- i.e. the last True sample along the ray, not
    necessarily the first mask boundary crossing. For a star-convex mask
    (any ray from the centroid crosses the boundary exactly once, true for
    a compact organoid without holes or a concavity that fully separates
    an outer ring from the centroid) this is equivalent to the true
    boundary distance in that direction. For a mask where a ray could exit
    and later re-enter (e.g. a hole, or a concavity that briefly leaves and
    rejoins the mask), this takes the outer envelope distance rather than
    the distance to the first crossing -- an intentional choice (it
    captures the organoid's outer extent along that direction) that
    readers should be aware of for non-star-convex shapes. An angle whose
    ray never enters the mask at all is recorded as NaN.
    """
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    cy, cx = float(ys.mean()), float(xs.mean())
    h, w = mask.shape[:2]
    max_r = float(np.hypot(h, w))
    steps = np.arange(1.0, max_r, 1.0)

    angles = np.linspace(0.0, 2 * np.pi, num_angles, endpoint=False)
    radii = np.full(num_angles, np.nan)
    for i, theta in enumerate(angles):
        xs_ray = np.clip((cx + np.cos(theta) * steps).astype(int), 0, w - 1)
        ys_ray = np.clip((cy + np.sin(theta) * steps).astype(int), 0, h - 1)
        inside = mask[ys_ray, xs_ray]
        if inside.any():
            radii[i] = steps[inside].max()
    return radii


def radial_cv(mask: np.ndarray, num_angles: int = 360) -> float | None:
    """Radial CV = SD(radius) / Mean(radius) across the angular samples
    from `radial_profile` (docs/analysis_design.md #5). Higher values
    indicate a more irregular shape. None if the mask is empty or has zero
    mean radius.
    """
    radii = radial_profile(mask, num_angles=num_angles)
    if radii is None:
        return None
    valid = radii[~np.isnan(radii)]
    if valid.size == 0 or valid.mean() == 0:
        return None
    return float(valid.std() / valid.mean())


def compute_morphology_features(
    mask: np.ndarray,
    pixel_size_um: float | None = None,
    num_radial_angles: int = 360,
) -> MorphologyFeatures:
    """Compute all Phase-1 pilot morphology measurements for one mask --
    see the module docstring for why these are PILOT values, not final
    biological measurements.

    `pixel_size_um` (microns per pixel) is optional -- when available (not
    yet, since io/metadata.py calibration parsing is unimplemented),
    `.physical` is populated per docs/analysis_design.md #5. When None,
    `.physical` stays None rather than fabricating a physical value.
    """
    area_px = area(mask)
    equiv_diam_px = equivalent_diameter(area_px)

    pixel_features = PixelMorphologyFeatures(
        area_px=area_px,
        perimeter_px=perimeter(mask),
        equivalent_diameter_px=equiv_diam_px,
        circularity=circularity(mask),
        solidity=solidity(mask),
        eccentricity=(ecc_ar := eccentricity_and_aspect_ratio(mask))[0],
        aspect_ratio=ecc_ar[1],
        radial_cv=radial_cv(mask, num_angles=num_radial_angles),
    )

    physical_features = None
    if pixel_size_um:
        physical_features = PhysicalMorphologyFeatures(
            pixel_size_um=pixel_size_um,
            area_um2=area_px * (pixel_size_um**2),
            equivalent_diameter_um=equiv_diam_px * pixel_size_um,
        )

    return MorphologyFeatures(pixel=pixel_features, physical=physical_features)
