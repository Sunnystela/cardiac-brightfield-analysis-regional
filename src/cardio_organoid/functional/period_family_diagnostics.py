"""Round 40: period-family mode detection -- do reliable patches actually
share ONE fundamental rhythm, or several?

Round 39 assumed a single shared fundamental period across a video's whole
reliable-patch population and estimated ONE P via harmonic-aware weighted
consensus (`fundamental_period_consensus.py`). That assumption held up well
for D10 (0/36 patches needed harmonic correction, all three named regions
independently converged to ~32-33 BPM) but clearly failed for early/conv_a1
and Beating_D9_slit_a1 (round 39's own known failures). This module tests
the single-rhythm assumption itself, BEFORE committing to one P: clusters
reliable patches purely by their own PERIOD (harmonic-aware distance,
reusing `harmonic_aware_timing.harmonic_aware_period_similarity` UNCHANGED
-- never timing/phase/coincidence features, never spatial adjacency, never
manual region labels), using the SAME generic `average_linkage_clusters` +
`cut_at_relative_gap` machinery already validated in rounds 37-39. If more
than one period family survives this self-relative cut, a single P is not
an appropriate model for that video without further work -- reported
honestly rather than forced.
"""

from __future__ import annotations

import numpy as np

from cardio_organoid.functional.adaptive_timing_clustering import cut_at_relative_gap
from cardio_organoid.functional.harmonic_aware_timing import harmonic_aware_period_similarity
from cardio_organoid.functional.patch_timing_clustering import average_linkage_clusters


def build_period_distance_matrix(
    patch_ids: list[tuple[int, int]],
    own_period_by_patch: dict[tuple[int, int], float | None],
    confidence_by_patch: dict[tuple[int, int], float],
) -> np.ndarray:
    """(n, n) distance in [0, 1] built ONLY from each patch's own period
    estimate (`fundamental_period_consensus.patch_own_period_estimate`'s
    `own_period_estimate_s`) via `harmonic_aware_period_similarity`
    (1x/2x/0.5x-aware), weighted by the geometric mean of the two patches'
    confidence (same posture as every other confidence-weighted distance
    matrix in this project). A patch with no valid own period, or a pair
    with no valid comparison, gets the maximum distance 1.0."""
    n = len(patch_ids)
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = patch_ids[i], patch_ids[j]
            result = harmonic_aware_period_similarity(own_period_by_patch.get(a), own_period_by_patch.get(b))
            if result["similarity"] is None:
                d = 1.0
            else:
                conf_a, conf_b = confidence_by_patch.get(a, 1.0), confidence_by_patch.get(b, 1.0)
                weight = float(np.sqrt(max(conf_a, 0.0) * max(conf_b, 0.0)))
                d = 1.0 - result["similarity"] * weight
            dist[i, j] = d
            dist[j, i] = d
    return dist


def detect_period_families(
    patch_ids: list[tuple[int, int]],
    own_period_by_patch: dict[tuple[int, int], float | None],
    confidence_by_patch: dict[tuple[int, int], float],
) -> tuple[list[list[tuple[int, int]]], dict]:
    """End-to-end period-only family discovery: distance matrix (see
    `build_period_distance_matrix`) + round 37's `average_linkage_clusters`
    (UNCHANGED) + round 38's `cut_at_relative_gap` (UNCHANGED). Same
    `(clusters, diagnostic_info)` shape as every other clustering entry
    point in this project. `len(clusters) == 1` (or one large cluster plus
    only singletons) is consistent with a single shared rhythm; several
    comparably-sized clusters is not."""
    ordered = sorted(patch_ids)
    dist = build_period_distance_matrix(ordered, own_period_by_patch, confidence_by_patch)
    merge_history, merge_heights = average_linkage_clusters(ordered, dist)
    clusters, cut_info = cut_at_relative_gap(merge_history, merge_heights, ordered)
    return clusters, {"distance_matrix": dist, **cut_info}


def family_period_summary(
    clusters: list[list[tuple[int, int]]],
    own_period_by_patch: dict[tuple[int, int], float | None],
    confidence_by_patch: dict[tuple[int, int], float],
) -> list[dict]:
    """Descriptive per-family summary (median own period/BPM, mean
    confidence, spatial bounding box) -- sorted the same way `clusters`
    already is (largest first)."""
    summaries = []
    for c in clusters:
        periods = [own_period_by_patch[p] for p in c if own_period_by_patch.get(p)]
        confidences = [confidence_by_patch.get(p, 1.0) for p in c]
        rows, cols = [p[0] for p in c], [p[1] for p in c]
        summaries.append({
            "n_patches": len(c),
            "patch_ids": [list(p) for p in c],
            "median_own_period_s": float(np.median(periods)) if periods else None,
            "median_own_bpm": (60.0 / float(np.median(periods))) if periods else None,
            "mean_confidence": float(np.mean(confidences)) if confidences else None,
            "spatial_bbox": {"row_min": min(rows), "row_max": max(rows), "col_min": min(cols), "col_max": max(cols)},
        })
    return summaries
