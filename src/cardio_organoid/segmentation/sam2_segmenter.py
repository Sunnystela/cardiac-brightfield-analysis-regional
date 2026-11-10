"""SAM2-based cardiac organoid segmentation (Phase 1).

Given a single representative frame, this module runs SAM2 with a single
center point prompt (Strategy A -- see scripts/compare_sam2_prompt_strategies.py
for why a box/negative-point strategy was tried and dropped) and returns
*all* of SAM2's candidate masks (its native ambiguity-aware output).

A fixed candidate index is used as a *provisional* primary organoid
(morphology) mask based on an empirical convention observed across the
initial sample videos -- see `provisional_primary_qc` below for why this is
not hardcoded as a permanent assumption, and the module note near the
bottom for why solidity/circularity must never drive this selection.
`compute_motion_roi_mask` derives a separate, deliberately larger ROI for
motion analysis from that same morphology mask.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch

DEFAULT_CHECKPOINT_PATH = Path("checkpoints/sam2/sam2.1_hiera_small.pt")
DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"


def get_device(preference: str = "auto") -> torch.device:
    """Resolve the torch device to run SAM2 on.

    "auto" selects CUDA when available and falls back to CPU otherwise.
    An explicit value ("cuda" or "cpu") is passed through as-is.
    """
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


@dataclasses.dataclass
class MaskCandidate:
    """One SAM2 candidate mask plus the raw measurements a human reviewer
    needs to judge it -- no attempt is made to rank or select among these.
    """

    index: int
    mask: np.ndarray  # HxW bool array
    score: float
    area_px: int
    area_fraction: float


def summarize_masks(
    masks: np.ndarray, scores: np.ndarray
) -> list[MaskCandidate]:
    """Turn SAM2's raw (masks, scores) output into MaskCandidate records.

    Kept separate from `segment_frame` so the area/area_fraction math is
    testable without loading the SAM2 model.
    """
    if masks.shape[0] != scores.shape[0]:
        raise ValueError(
            f"masks and scores disagree on candidate count: "
            f"{masks.shape[0]} vs {scores.shape[0]}"
        )
    total_px = int(masks.shape[-2] * masks.shape[-1])
    candidates = []
    for i in range(masks.shape[0]):
        mask_bool = masks[i].astype(bool)
        area_px = int(mask_bool.sum())
        candidates.append(
            MaskCandidate(
                index=i,
                mask=mask_bool,
                score=float(scores[i]),
                area_px=area_px,
                area_fraction=area_px / total_px,
            )
        )
    return candidates


def load_predictor(
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH,
    model_cfg: str = DEFAULT_MODEL_CFG,
    device: torch.device | str | None = None,
):
    """Build a SAM2ImagePredictor from a local checkpoint.

    Never downloads a checkpoint -- raises if the given path is missing.
    """
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found at {checkpoint_path}. "
            "This function does not download checkpoints -- fetch it manually."
        )
    if device is None:
        device = get_device()
    sam2_model = build_sam2(model_cfg, str(checkpoint_path), device=str(device))
    return SAM2ImagePredictor(sam2_model)


def predict_candidates(
    predictor,
    frame_rgb: np.ndarray,
    point_coords: np.ndarray | None = None,
    point_labels: np.ndarray | None = None,
    box: np.ndarray | None = None,
    multimask_output: bool = True,
) -> list[MaskCandidate]:
    """Run SAM2 on one RGB frame with an arbitrary prompt (points and/or a
    box). This is the general primitive -- different prompting strategies
    (single center point, point + negative points + box, ...) are built on
    top of it by callers, so the strategies can be compared against each
    other rather than baked into this function.
    """
    predictor.set_image(frame_rgb)
    masks, scores, _logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=multimask_output,
    )
    return summarize_masks(masks, scores)


def segment_frame(
    predictor,
    frame_rgb: np.ndarray,
    point_xy: tuple[int, int] | None = None,
) -> list[MaskCandidate]:
    """Run SAM2 on one RGB frame using a single center point prompt.

    Defaults to the frame center, since the current sample videos frame the
    organoid roughly in the middle. Returns all multimask_output candidates.
    This is "Strategy A" (single point, SAM2's native candidates) -- see
    scripts/compare_sam2_prompt_strategies.py for a comparison against a
    point+box+negative-points strategy.
    """
    h, w = frame_rgb.shape[:2]
    if point_xy is None:
        point_xy = (w // 2, h // 2)
    point_coords = np.array([[point_xy[0], point_xy[1]]])
    point_labels = np.array([1])
    return predict_candidates(
        predictor, frame_rgb, point_coords, point_labels, multimask_output=True
    )


def make_overlay(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend `mask` over `frame_rgb` as a semi-transparent fill with an
    outline, for visual QC of a single candidate.
    """
    import cv2

    overlay = frame_rgb.copy()
    color_arr = np.array(color, dtype=np.float32)
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = (
        (1 - alpha) * overlay[mask_bool].astype(np.float32) + alpha * color_arr
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, color, thickness=2)
    return overlay


# ---------------------------------------------------------------------------
# Candidate-mask diagnostics.
#
# These are for comparing SAM2 candidates / prompting strategies against
# each other, NOT for automatically choosing the primary organoid mask.
# `solidity` and `circularity` in particular are shape descriptors that this
# project later reports as morphology outcomes (see docs/analysis_design.md).
# Using them to select which mask becomes the organoid mask would bias those
# downstream measurements toward whatever shape the selection rule favors
# (e.g. always picking the most circular candidate would erase genuine
# organoid irregularity/protrusions before it is ever measured). Keep these
# strictly descriptive/diagnostic.
# ---------------------------------------------------------------------------


def connected_component_stats(mask: np.ndarray) -> tuple[int, float]:
    """Return (num_components, largest_component_area_fraction_of_mask).

    A structural sanity signal, independent of any shape/morphology
    property: a compact organoid should be one dominant blob, not a
    scattering of disconnected fragments.
    """
    import cv2

    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:  # only the background label
        return 0, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    total = int(areas.sum())
    if total == 0:
        return 0, 0.0
    return int(num_labels - 1), float(areas.max() / total)


def center_distance_fraction(
    mask: np.ndarray, frame_shape: tuple[int, int]
) -> float | None:
    """Distance from the mask's centroid to the frame center, normalized by
    the frame diagonal (0 = exactly centered, ~0.7 = a corner).

    A weak spatial prior only -- organoids in the current sample videos tend
    to be framed near the center, but this should be used as a secondary /
    tie-breaking signal, not a hard filter.
    """
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    h, w = frame_shape[:2]
    centroid_y, centroid_x = float(ys.mean()), float(xs.mean())
    center_y, center_x = h / 2.0, w / 2.0
    dist = float(np.hypot(centroid_y - center_y, centroid_x - center_x))
    diagonal = float(np.hypot(h, w))
    return dist / diagonal


def boundary_gradient_alignment(
    frame_gray: np.ndarray, mask: np.ndarray
) -> float | None:
    """How well the mask boundary coincides with real intensity edges in the
    raw frame, as a percentile rank of the boundary's mean gradient
    magnitude against the whole frame's gradient magnitude distribution.

    Self-normalized per image (percentile rank, not an absolute intensity
    threshold), so it generalizes across videos with different brightness,
    contrast, or magnification. 1.0 means the boundary sits on some of the
    strongest edges in the frame; low values suggest the boundary was drawn
    through a low-contrast region rather than a real structure edge.
    """
    import cv2

    gx = cv2.Sobel(frame_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(frame_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    contour_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, contours, -1, 1, thickness=2)
    boundary_vals = grad_mag[contour_mask.astype(bool)]
    if boundary_vals.size == 0:
        return None

    boundary_mean = float(boundary_vals.mean())
    return float((grad_mag.ravel() < boundary_mean).mean())


def solidity(mask: np.ndarray) -> float | None:
    """Area / convex-hull area of the largest contour. Descriptive only --
    see the module-level note above. Not used for mask selection."""
    import cv2

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    hull_area = cv2.contourArea(cv2.convexHull(largest))
    if hull_area == 0:
        return None
    return float(area / hull_area)


def circularity(mask: np.ndarray) -> float | None:
    """4*pi*Area / Perimeter^2 of the largest contour (1.0 = perfect
    circle). Descriptive only -- see the module-level note above. Not used
    for mask selection."""
    import cv2

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, closed=True)
    if perimeter == 0:
        return None
    return float(4 * np.pi * area / perimeter**2)


# ---------------------------------------------------------------------------
# Provisional primary (morphology) mask selection.
#
# Across the sample videos reviewed so far, the Strategy-A candidate at a
# fixed index (see configs/default.yaml: segmentation.provisional_primary)
# consistently matched the compact organoid body the user wants morphology
# and motion analysis anchored on. This is an *empirical convention observed
# on the current sample set*, not a guaranteed property of the SAM2 API --
# nothing about multimask_output promises a stable semantic ordering. Treat
# `candidate_index` as provisional: keep saving all candidates + QC flags so
# index-consistency can be validated as more videos are processed, and never
# select it using solidity/circularity (see the module note above).
# ---------------------------------------------------------------------------


def provisional_primary_qc(
    candidates: list[MaskCandidate],
    candidate_index: int,
    min_dominant_component_fraction: float = 0.9,
) -> dict:
    """Bias-free sanity checks on whether `candidate_index` still looks like
    the well-formed, high-confidence, single-blob mask it was picked to be.

    Does not use solidity/circularity, and does not *prove* the candidate is
    the organoid -- it only flags when the observed convention (index has
    the highest SAM2 score and is a single dominant connected component)
    breaks down, so a human can review that frame/video specifically.
    """
    if not (0 <= candidate_index < len(candidates)):
        return {"candidate_index": candidate_index, "flag": "invalid_index"}

    provisional = candidates[candidate_index]
    scores_desc = sorted((c.score for c in candidates), reverse=True)
    score_rank = scores_desc.index(provisional.score) + 1  # 1 = highest
    num_components, largest_component_fraction = connected_component_stats(
        provisional.mask
    )
    is_single_dominant_component = (
        largest_component_fraction >= min_dominant_component_fraction
    )
    is_highest_score = score_rank == 1
    flag = "ok" if (is_highest_score and is_single_dominant_component) else "review_needed"

    return {
        "candidate_index": candidate_index,
        "score": provisional.score,
        "score_rank_among_candidates": score_rank,
        "is_highest_score": is_highest_score,
        "num_components": num_components,
        "largest_component_fraction": largest_component_fraction,
        "is_single_dominant_component": is_single_dominant_component,
        "flag": flag,
    }


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    """(y, x) centroid of `mask`, or None if empty. This is the only
    geometric information the motion module is meant to take from the
    morphology mask -- see mask_bounding_box and motion/raft_flow.py.
    """
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return float(ys.mean()), float(xs.mean())


def mask_bounding_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) bounding box of `mask` (x1/y1 exclusive), or None if
    empty. Deliberately coarse -- the motion module builds its ROI from
    this and the centroid, not from the mask's exact contour, so motion
    analysis does not depend on segmentation precision.
    """
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
