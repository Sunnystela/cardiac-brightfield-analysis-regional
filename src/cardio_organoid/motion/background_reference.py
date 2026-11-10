"""Multi-patch static-background reference selection (Phase 2 validation).

Replaces a single "everything outside a dilated mask" background region
with several small patches placed around the organoid, screened to avoid
obvious debris or moving material, so the background/noise estimate isn't
itself contaminated by real (if uninteresting) motion. All patch geometry
is expressed as ratios of the organoid's own equivalent diameter, never
fixed pixel counts, so it scales across organoids of different sizes.

Patch screening is self-relative (percentile comparisons among the
candidate patches of a single video/frame), not an absolute magic
threshold: candidates are kept only if they are brighter than the median
candidate (bright-field background is lighter than the organoid/debris)
AND less temporally variable than the median candidate (debris drifting
or a moving object shows up as elevated frame-to-frame intensity change).
This is a heuristic filter, not a validated debris detector -- always
inspect the saved QC visualization rather than trusting it blindly.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class PatchCandidate:
    index: int
    angle_deg: float
    bbox_xyxy: tuple[int, int, int, int]  # in ROI-local coordinates
    mean_intensity: float
    temporal_std: float
    kept: bool
    # Optional fields, only populated by select_background_patches_multi_ring
    # (default None keeps select_background_patches's PatchCandidate(**c)
    # construction working unchanged).
    ring_radius_ratio: float | None = None
    local_max_pixel_temporal_std: float | None = None
    reject_reason: str | None = None


def _candidate_patch_bboxes(
    centroid_xy: tuple[float, float],
    radius_px: float,
    patch_size_px: float,
    n_angles: int,
    roi_shape: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    cx, cy = centroid_xy
    h, w = roi_shape
    half = patch_size_px / 2
    bboxes = []
    for i in range(n_angles):
        theta = 2 * np.pi * i / n_angles
        px = cx + radius_px * np.cos(theta)
        py = cy + radius_px * np.sin(theta)
        x0, y0 = int(round(px - half)), int(round(py - half))
        x1, y1 = int(round(px + half)), int(round(py + half))
        bboxes.append((x0, y0, x1, y1))
    return bboxes


def select_background_patches(
    frame_gray_roi: np.ndarray,
    frames_gray_roi_buffer: list[np.ndarray],
    centroid_xy: tuple[float, float],
    equivalent_diameter_px: float,
    n_angles: int,
    radius_ratio: float,
    patch_size_ratio: float,
) -> tuple[list[PatchCandidate], np.ndarray]:
    """Return (all candidates with kept-flags, boolean gate mask = union of
    kept patches) in the same ROI-local coordinate system as `frame_gray_roi`.

    Candidates entirely outside the ROI crop are dropped before screening
    (not counted as "rejected", just infeasible).
    """
    h, w = frame_gray_roi.shape[:2]
    radius_px = radius_ratio * equivalent_diameter_px
    patch_size_px = patch_size_ratio * equivalent_diameter_px
    bboxes = _candidate_patch_bboxes(centroid_xy, radius_px, patch_size_px, n_angles, (h, w))

    raw_candidates = []
    for i, (x0, y0, x1, y1) in enumerate(bboxes):
        angle_deg = 360.0 * i / n_angles
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h or x1 <= x0 or y1 <= y0:
            continue
        patch_stack = np.stack([f[y0:y1, x0:x1] for f in frames_gray_roi_buffer]).astype(np.float32)
        mean_intensity = float(frame_gray_roi[y0:y1, x0:x1].mean())
        # Temporal variability: std, over time, of the patch's own mean
        # intensity -- cheap (no optical flow needed) proxy for "something
        # is drifting/changing here" that can screen candidates up front.
        temporal_std = float(patch_stack.mean(axis=(1, 2)).std())
        raw_candidates.append(
            {
                "index": i,
                "angle_deg": angle_deg,
                "bbox_xyxy": (x0, y0, x1, y1),
                "mean_intensity": mean_intensity,
                "temporal_std": temporal_std,
            }
        )

    if not raw_candidates:
        return [], np.zeros((h, w), dtype=bool)

    intensities = np.array([c["mean_intensity"] for c in raw_candidates])
    temporal_stds = np.array([c["temporal_std"] for c in raw_candidates])

    # Percentile-based OR-exclusion (not a hard AND of two independent
    # median cutoffs, which can have an empty intersection in practice --
    # e.g. the brightest candidates turning out to also be the most
    # temporally variable ones left zero survivors on real data). A
    # candidate is rejected if it is unusually dark (bottom quartile of
    # intensity among *these* candidates) OR unusually variable over time
    # (top quartile of temporal std) -- each criterion alone can disqualify
    # a candidate, but at most ~25% of candidates can be rejected by each,
    # so at least ~50% survive even in the worst case where the two
    # criteria disagree completely. Still self-relative (percentiles among
    # this frame's own candidates), not an absolute magic threshold.
    intensity_cutoff = float(np.percentile(intensities, 25))
    temporal_std_cutoff = float(np.percentile(temporal_stds, 75))

    candidates = []
    gate = np.zeros((h, w), dtype=bool)
    for c in raw_candidates:
        kept = c["mean_intensity"] >= intensity_cutoff and c["temporal_std"] <= temporal_std_cutoff
        candidates.append(PatchCandidate(kept=kept, **c))
        if kept:
            x0, y0, x1, y1 = c["bbox_xyxy"]
            gate[y0:y1, x0:x1] = True

    return candidates, gate


def select_background_patches_multi_ring(
    frame_gray_roi: np.ndarray,
    frames_gray_roi_buffer: list[np.ndarray],
    centroid_xy: tuple[float, float],
    equivalent_diameter_px: float,
    analysis_region_mask: np.ndarray,
    n_angles: int,
    radius_ratios: list[float],
    patch_size_ratio: float,
) -> tuple[list[PatchCandidate], np.ndarray]:
    """Like `select_background_patches`, but samples candidates across
    MULTIPLE concentric rings (`radius_ratios`) x `n_angles` each instead
    of a single ring -- a single ring of ~8 candidates was found, on real
    data, to leave as few as 2-4 surviving patches after screening, too
    few for a stable organoid-vs-background correlation baseline. This is
    purely a larger, more carefully screened candidate pool -- geometry
    and screening philosophy (self-relative percentiles, OR-exclusion so
    no single criterion can reject everyone) unchanged.

    Two exclusion criteria beyond the original screening:
      - any candidate overlapping `analysis_region_mask` (the dilated
        organoid mask used to build the analysis grid) at all is dropped
        outright -- an explicit "outside the organoid" check that doesn't
        rely on radius alone, since an irregularly-shaped organoid isn't
        well approximated by a circle at every angle.
      - "local texture movement" not caught by the whole-patch MEAN
        intensity's temporal std: the same self-relative top-quartile
        exclusion is also applied to each candidate's own 90th-percentile
        per-pixel temporal std -- catches a small moving/flickering
        region (e.g. debris, a floating cell) within an otherwise-static
        patch, whose spatial mean can still look stable.

    No explicit "well edge" detector exists (nothing in this codebase
    currently models where the well edge is) -- mitigated only by
    default `radius_ratios` staying within the range
    configs/default.yaml's `motion.roi.expansion_px`/`radius_ratio`
    comment already empirically validated as clear of the peripheral
    debris ring; always inspect the saved QC overlay.
    """
    h, w = frame_gray_roi.shape[:2]
    patch_size_px = patch_size_ratio * equivalent_diameter_px

    raw = []
    idx = 0
    for radius_ratio in radius_ratios:
        radius_px = radius_ratio * equivalent_diameter_px
        bboxes = _candidate_patch_bboxes(centroid_xy, radius_px, patch_size_px, n_angles, (h, w))
        for i, (x0, y0, x1, y1) in enumerate(bboxes):
            angle_deg = 360.0 * i / n_angles
            if x0 < 0 or y0 < 0 or x1 > w or y1 > h or x1 <= x0 or y1 <= y0:
                idx += 1
                continue
            if analysis_region_mask[y0:y1, x0:x1].any():
                raw.append({
                    "index": idx, "angle_deg": angle_deg, "bbox_xyxy": (x0, y0, x1, y1),
                    "ring_radius_ratio": radius_ratio, "mean_intensity": None, "temporal_std": None,
                    "local_max_pixel_temporal_std": None, "reject_reason": "overlaps_organoid_region",
                })
                idx += 1
                continue
            patch_stack = np.stack([f[y0:y1, x0:x1] for f in frames_gray_roi_buffer]).astype(np.float32)
            mean_intensity = float(frame_gray_roi[y0:y1, x0:x1].mean())
            temporal_std = float(patch_stack.mean(axis=(1, 2)).std())
            local_max_pixel_temporal_std = float(np.percentile(patch_stack.std(axis=0), 90))
            raw.append({
                "index": idx, "angle_deg": angle_deg, "bbox_xyxy": (x0, y0, x1, y1),
                "ring_radius_ratio": radius_ratio, "mean_intensity": mean_intensity,
                "temporal_std": temporal_std, "local_max_pixel_temporal_std": local_max_pixel_temporal_std,
                "reject_reason": None,
            })
            idx += 1

    scoreable = [c for c in raw if c["reject_reason"] is None]
    if not scoreable:
        return (
            [PatchCandidate(
                index=c["index"], angle_deg=c["angle_deg"], bbox_xyxy=c["bbox_xyxy"],
                mean_intensity=float("nan"), temporal_std=float("nan"), kept=False,
                ring_radius_ratio=c["ring_radius_ratio"], reject_reason=c["reject_reason"],
            ) for c in raw],
            np.zeros((h, w), dtype=bool),
        )

    intensities = np.array([c["mean_intensity"] for c in scoreable])
    temporal_stds = np.array([c["temporal_std"] for c in scoreable])
    local_stds = np.array([c["local_max_pixel_temporal_std"] for c in scoreable])
    intensity_cutoff = float(np.percentile(intensities, 25))
    temporal_std_cutoff = float(np.percentile(temporal_stds, 75))
    local_std_cutoff = float(np.percentile(local_stds, 75))

    candidates = []
    gate = np.zeros((h, w), dtype=bool)
    for c in raw:
        if c["reject_reason"] is not None:
            candidates.append(PatchCandidate(
                index=c["index"], angle_deg=c["angle_deg"], bbox_xyxy=c["bbox_xyxy"],
                mean_intensity=float("nan"), temporal_std=float("nan"), kept=False,
                ring_radius_ratio=c["ring_radius_ratio"], reject_reason=c["reject_reason"],
            ))
            continue
        dark = c["mean_intensity"] < intensity_cutoff
        flickery = c["temporal_std"] > temporal_std_cutoff
        locally_textured = c["local_max_pixel_temporal_std"] > local_std_cutoff
        kept = not (dark or flickery or locally_textured)
        reject_reason = None
        if not kept:
            reasons = []
            if dark:
                reasons.append("dark")
            if flickery:
                reasons.append("flickery_mean")
            if locally_textured:
                reasons.append("local_texture_movement")
            reject_reason = "+".join(reasons)
        candidates.append(PatchCandidate(
            index=c["index"], angle_deg=c["angle_deg"], bbox_xyxy=c["bbox_xyxy"],
            mean_intensity=c["mean_intensity"], temporal_std=c["temporal_std"], kept=kept,
            ring_radius_ratio=c["ring_radius_ratio"],
            local_max_pixel_temporal_std=c["local_max_pixel_temporal_std"], reject_reason=reject_reason,
        ))
        if kept:
            x0, y0, x1, y1 = c["bbox_xyxy"]
            gate[y0:y1, x0:x1] = True

    return candidates, gate


def draw_patch_qc(frame_bgr_roi: np.ndarray, candidates: list[PatchCandidate]) -> np.ndarray:
    """Overlay all candidate patches on the ROI crop: green box + kept
    stats for kept patches, red box for rejected ones (labeled with why,
    when available), for a human to visually confirm the background
    reference isn't sitting on debris, the well edge, or other moving
    material.
    """
    import cv2

    vis = frame_bgr_roi.copy()
    for c in candidates:
        x0, y0, x1, y1 = c.bbox_xyxy
        color = (0, 200, 0) if c.kept else (0, 0, 220)
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
        if c.reject_reason is not None:
            label = f"{c.index}:X {c.reject_reason}"
        else:
            label = f"{c.index}:{'K' if c.kept else 'X'} I={c.mean_intensity:.0f} s={c.temporal_std:.2f}"
        cv2.putText(
            vis, label, (x0, max(0, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
            color, 1, cv2.LINE_AA,
        )
    return vis
