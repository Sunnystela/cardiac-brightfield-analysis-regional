"""Per-patch image-observability features (round 36, Phase 1).

Answers a question this project has repeatedly bumped into (round 33's
D10 texture diagnostic, round 35's block-matching LOW_CONFIDENCE zone):
before trusting ANY motion estimate at a patch, is the underlying image
content there even suitable for tracking motion at all? A patch over a
flat, textureless region of a bright-field organoid can show near-zero
measured motion for two entirely different reasons -- genuinely still
tissue, or a tracker with nothing to lock onto -- and this module only
answers the second question (observability), never the first (motion).

Every feature below is computed and returned SEPARATELY, both as raw
(physical-unit) values and as within-video population-normalized
(z-score) values -- deliberately NOT collapsed into one "observability
score" here. A single provisional composite for visualization only exists
in `reliable_hotspot_candidates.py`, clearly labeled provisional.

LOW_TEXTURE/LOW_OBSERVABILITY is a statement about the IMAGE, not about
the tissue's motion -- never conflate a low-observability patch with a
non-moving one (see reliable_hotspot_candidates.classify_state).
"""

from __future__ import annotations

import numpy as np


def compute_pixel_level_maps(frame_gray: np.ndarray) -> dict[str, np.ndarray]:
    """Full-frame (or full-ROI-crop) per-pixel feature maps, computed
    once and reduced per patch afterward (see `reduce_patch_features`) --
    cheap, single-frame, vectorized (no per-patch Python loop for the
    expensive parts).

    - gradient_magnitude: Sobel gradient magnitude.
    - edges: boolean Canny edge map, thresholds derived from the frame's
      OWN gradient-magnitude percentiles (self-relative, not a fixed
      absolute intensity threshold).
    - lambda1/lambda2: the two eigenvalues (per pixel, lambda1 >= lambda2)
      of the local 2x2 structure tensor [[Ix^2, IxIy], [IxIy, Iy^2]]
      (box-averaged over a small window) -- the standard Shi-Tomasi /
      Harris-adjacent trackability measure. `lambda2` (the smaller
      eigenvalue) IS the Shi-Tomasi "min eigenvalue" corner-strength
      criterion -- reported directly as `corner_strength` in
      `reduce_patch_features` rather than recomputed a second way.
    - laplacian: for texture-energy (variance of Laplacian) at the patch
      level.
    """
    import cv2

    gray = frame_gray.astype(np.float32)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = np.hypot(sobel_x, sobel_y)

    grad_median = float(np.median(gradient_magnitude))
    grad_p90 = float(np.percentile(gradient_magnitude, 90))
    low_thresh = max(1.0, 0.5 * grad_median)
    high_thresh = max(low_thresh + 1.0, grad_p90)
    edges = cv2.Canny(frame_gray.astype(np.uint8), low_thresh, high_thresh) > 0

    window = 5
    ixx = cv2.boxFilter(sobel_x * sobel_x, -1, (window, window))
    iyy = cv2.boxFilter(sobel_y * sobel_y, -1, (window, window))
    ixy = cv2.boxFilter(sobel_x * sobel_y, -1, (window, window))
    trace = ixx + iyy
    det = ixx * iyy - ixy * ixy
    disc = np.sqrt(np.clip((trace / 2) ** 2 - det, 0, None))
    lambda1 = trace / 2 + disc
    lambda2 = trace / 2 - disc

    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)

    return {
        "gradient_magnitude": gradient_magnitude, "edges": edges,
        "lambda1": lambda1, "lambda2": lambda2, "laplacian": laplacian,
    }


def reduce_patch_features(
    maps: dict[str, np.ndarray], frame_gray: np.ndarray, cells: list,
) -> dict[tuple[int, int], dict[str, float]]:
    """Reduce the full-frame maps (see `compute_pixel_level_maps`) to one
    raw-feature record per patch cell (`cells`: a list of
    `spatial_patches.PatchCell`, i.e. the SAME analyzed grid production
    already uses -- never a manual-region-derived list)."""
    features: dict[tuple[int, int], dict[str, float]] = {}
    for cell in cells:
        x0, y0, x1, y1 = cell.bbox_xyxy
        patch_gray = frame_gray[y0:y1, x0:x1].astype(np.float64)
        mean_intensity = float(patch_gray.mean()) if patch_gray.size else 0.0
        intensity_std = float(patch_gray.std()) if patch_gray.size else 0.0
        local_contrast = (intensity_std / mean_intensity) if mean_intensity > 0 else 0.0
        lam1 = maps["lambda1"][y0:y1, x0:x1]
        lam2 = maps["lambda2"][y0:y1, x0:x1]
        features[(cell.row, cell.col)] = {
            "intensity_std": intensity_std,
            "local_contrast": local_contrast,
            "gradient_magnitude": float(np.mean(maps["gradient_magnitude"][y0:y1, x0:x1])),
            "edge_density": float(np.mean(maps["edges"][y0:y1, x0:x1])),
            "structure_tensor_lambda1": float(np.mean(lam1)) if lam1.size else 0.0,
            "structure_tensor_lambda2": float(np.mean(lam2)) if lam2.size else 0.0,
            "corner_strength": float(np.mean(lam2)) if lam2.size else 0.0,  # Shi-Tomasi min-eigenvalue criterion
            "texture_energy": float(np.var(maps["laplacian"][y0:y1, x0:x1])) if patch_gray.size else 0.0,
        }
    return features


FEATURE_NAMES = [
    "intensity_std", "local_contrast", "gradient_magnitude", "edge_density",
    "structure_tensor_lambda1", "structure_tensor_lambda2", "corner_strength", "texture_energy",
]


def normalize_features_within_video(
    raw_features: dict[tuple[int, int], dict[str, float]],
) -> dict[tuple[int, int], dict[str, float]]:
    """Within-video z-score normalization, per feature, across all
    analyzed patches -- self-relative to this video's own patch
    population, never an absolute cross-video cutoff. A feature with zero
    population variance normalizes to all-zeros (not NaN/inf)."""
    patch_ids = list(raw_features)
    normalized: dict[tuple[int, int], dict[str, float]] = {pid: {} for pid in patch_ids}
    for name in FEATURE_NAMES:
        values = np.array([raw_features[pid][name] for pid in patch_ids], dtype=float)
        mean, std = values.mean(), values.std()
        z = (values - mean) / std if std > 0 else np.zeros_like(values)
        for pid, v in zip(patch_ids, z):
            normalized[pid][name] = float(v)
    return normalized
