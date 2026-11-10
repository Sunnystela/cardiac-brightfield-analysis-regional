"""Round 38: a scale-normalized cut rule and a density-based (DBSCAN-
style) clustering alternative, both applied over `harmonic_aware_timing`'s
robust distance matrix. Reuses round 37's `average_linkage_clusters`
UNCHANGED (see `patch_timing_clustering.py`) -- this module only adds (a)
a modestly improved cut rule for that SAME merge sequence, and (b) an
independent density-based method for cross-checking, per explicit project
instruction not to rely on a single clustering method's cut heuristic
alone.

Cut-rule diagnosis (round 38, D10/slit/a1): re-checking round 37's OWN
merge-height sequence for the secondary_region-overlapping mega-cluster
(`results/_scratch_runs/reliable_patch_phase_consensus_diagnostic_
20260910T075005Z/summary.json`) found round 37's `cut_at_gap` (absolute
gap, second-half-only search) actually cut in a REASONABLE place given
its own inputs: heights climb smoothly from ~0.80 to ~0.93 over 6 merges,
then jump to a degenerate ceiling of 1.0 for the remaining forced merges
-- `cut_at_gap` correctly stopped right at that 0.926 -> 1.0 boundary.
The real problem (see `harmonic_aware_timing.py`'s module docstring) is
upstream, in the similarity metric itself: secondary_region and most of
tertiary_region's patches had ALREADY merged together, below that
ceiling, before round 37's own cut ever got a chance to separate them --
no cut RULE can recover a separation the underlying distances never
encoded in the first place.

Given that, this module's cut-rule change is a smaller, complementary
improvement, not the primary fix: `cut_at_relative_gap` normalizes each
jump by the height it starts from (so a jump from 0.05->0.10, doubling
the distance, counts the same as one from 0.50->1.00) instead of
`cut_at_gap`'s raw absolute difference, which should generalize better
across this project's videos (their overall pairwise-distance scale
varies a lot). Round 38 found empirically (all 5 validation videos) that
naively searching a relative-jump measure over the WHOLE merge sequence
systematically over-fragments every video into near-singletons -- a
structural bias of ANY ratio-based jump measure toward the early,
small-height part of a smoothly climbing sequence, not a numerical
instability `floor_fraction_of_max_height` alone can fix. `cut_at_
relative_gap` therefore keeps round 37's OWN already-established,
empirically-justified "search only the second half of the merge
sequence" restriction as its default search window, and only replaces
the absolute-vs-relative magnitude measure used WITHIN that window.
"""

from __future__ import annotations

import numpy as np

from cardio_organoid.functional.harmonic_aware_timing import get_edge, robust_pairwise_timing_distance


def build_robust_confidence_weighted_distance_matrix(
    patch_ids: list[tuple[int, int]], graph: dict, confidence_by_patch: dict[tuple[int, int], float],
) -> np.ndarray:
    """Same confidence-weighting convention as round 37's
    `patch_timing_clustering.build_confidence_weighted_distance_matrix`
    (effective_similarity = raw_similarity * sqrt(conf_i * conf_j)), but
    reading edges built by `harmonic_aware_timing.build_robust_patch_
    timing_graph` / `robust_pairwise_timing_distance` instead of round
    37's fragile coincidence-fraction/native-frequency features."""
    n = len(patch_ids)
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = patch_ids[i], patch_ids[j]
            raw_similarity = 1.0 - robust_pairwise_timing_distance(get_edge(graph, a, b))
            conf_a = confidence_by_patch.get(a, 1.0)
            conf_b = confidence_by_patch.get(b, 1.0)
            weight = float(np.sqrt(max(conf_a, 0.0) * max(conf_b, 0.0)))
            d = 1.0 - raw_similarity * weight
            dist[i, j] = d
            dist[j, i] = d
    return dist


def cut_at_relative_gap(
    merge_history: list[dict[int, list[int]]], merge_heights: list[float], patch_ids: list[tuple[int, int]],
    min_merges_before_cut: int | None = None,
    floor_fraction_of_max_height: float = 0.05,
) -> tuple[list[list[tuple[int, int]]], dict]:
    """Self-relative cut using the RELATIVE (scale-normalized) jump at
    each merge -- see module docstring for round 37's `cut_at_gap`
    (absolute gap, second-half-only search) and why a pure relative-gap
    replacement over the WHOLE sequence was tried first and rejected.

    relative_jump[i] = (h[i+1]-h[i]) / max(h[i], floor); cuts right before
    the single largest relative jump found within the SEARCHED window.

    Round-38 finding (all 5 validation videos): searching the relative
    jump over the WHOLE merge sequence (no minimum-consolidation floor)
    systematically over-fragments every video into near-singletons. This
    is not the same failure mode `floor_fraction_of_max_height` guards
    against (a numerically tiny denominator) -- it is a structural
    property of ANY ratio-based jump measure: for a smoothly, gradually
    climbing sequence of heights, a fixed-size absolute step counts as a
    LARGER relative jump the smaller the starting height is, so relative-
    jump search is inherently biased toward the EARLY part of the
    sequence even with no numerical instability at all. Round 37's own
    "search only the second half" restriction (empirically justified
    there for the same reason, in the opposite direction: an absolute-gap
    search is biased toward the LATE part of the sequence) is therefore
    kept here as the default `min_merges_before_cut` (half of all
    possible merges, rounded down) -- this is round 37's OWN already-
    established, empirically-justified convention, not a new per-video
    tuning knob; only the magnitude measure used WITHIN that window
    (relative instead of absolute) is this round's change. An explicit
    `min_merges_before_cut` may still be passed (e.g. by tests) to probe
    the raw relative-jump behavior directly.
    `floor_fraction_of_max_height` (default 0.05, i.e. 5% of THIS run's
    own largest merge height) remains as a numerical-stability floor on
    every denominator within the searched window, adapting to each
    video's own distance scale like every other threshold in this
    project."""
    if len(merge_heights) < 2:
        cut_index = len(merge_heights)
        return _clusters_from_cut(merge_history, cut_index, patch_ids), {
            "n_merges_total": len(merge_heights), "n_merges_kept": cut_index,
            "merge_heights": merge_heights, "relative_jumps": [],
            "cut_merge_height": merge_heights[cut_index - 1] if cut_index > 0 else None,
            "next_merge_height_after_cut": merge_heights[cut_index] if cut_index < len(merge_heights) else None,
        }

    heights = np.asarray(merge_heights, dtype=float)
    floor = max(float(np.max(heights)) * floor_fraction_of_max_height, 1e-9)
    relative_jumps = (heights[1:] - heights[:-1]) / np.maximum(heights[:-1], floor)
    if min_merges_before_cut is None:
        min_merges_before_cut = len(merge_heights) // 2
    start = min(min_merges_before_cut, len(relative_jumps) - 1) if len(relative_jumps) > 0 else 0
    search = relative_jumps[start:]
    if search.size == 0:
        cut_index = len(merge_heights)
    else:
        cut_index = start + int(np.argmax(search)) + 1  # keep merges [0, cut_index)

    return _clusters_from_cut(merge_history, cut_index, patch_ids), {
        "n_merges_total": len(merge_heights), "n_merges_kept": cut_index,
        "merge_heights": merge_heights, "relative_jumps": relative_jumps.tolist(), "denominator_floor": floor,
        "cut_merge_height": merge_heights[cut_index - 1] if cut_index > 0 else None,
        "next_merge_height_after_cut": merge_heights[cut_index] if cut_index < len(merge_heights) else None,
    }


def _clusters_from_cut(merge_history, cut_index, patch_ids):
    membership = merge_history[cut_index]
    clusters = [sorted(patch_ids[i] for i in members) for members in membership.values()]
    clusters.sort(key=len, reverse=True)
    return clusters


def cluster_patches_by_robust_timing_similarity_hierarchical(
    patch_ids: list[tuple[int, int]], graph: dict, confidence_by_patch: dict[tuple[int, int], float],
) -> tuple[list[list[tuple[int, int]]], dict]:
    """End-to-end: robust distance matrix (harmonic/dropout-tolerant) +
    round 37's UNCHANGED `average_linkage_clusters` + this module's
    relative-gap cut. Same `(clusters, diagnostic_info)` shape as round
    37's hierarchical entry point, so every downstream consensus function
    is reused unchanged."""
    from cardio_organoid.functional.patch_timing_clustering import average_linkage_clusters

    ordered = sorted(patch_ids)
    dist = build_robust_confidence_weighted_distance_matrix(ordered, graph, confidence_by_patch)
    merge_history, merge_heights = average_linkage_clusters(ordered, dist)
    clusters, cut_info = cut_at_relative_gap(merge_history, merge_heights, ordered)
    return clusters, {"distance_matrix": dist, **cut_info}


# --------------------------------------------------------------------------
# Density-based (DBSCAN-style) clustering -- an independent cross-check
# method over the SAME distance matrix, self-relative eps (k-NN elbow),
# fixed small min_pts, identical parameters for every video (never
# hand-tuned per video/dataset).
# --------------------------------------------------------------------------


def estimate_eps_via_knn_elbow(distance_matrix: np.ndarray, k: int) -> float:
    """Self-relative DBSCAN `eps`: for each point, its distance to its
    k-th nearest neighbor; sort those ascending (the standard k-distance
    plot); pick the "elbow" as the point of maximum perpendicular
    distance from the straight line joining the first and last sorted
    values (a standard, parameter-free knee-detection heuristic -- same
    self-relative spirit as `cut_at_gap`/`cut_at_relative_gap`, never an
    absolute distance value, and computed identically for every video).
    Falls back to the median k-NN distance if there are too few points
    for a stable elbow (n <= k+1)."""
    n = distance_matrix.shape[0]
    if n <= k + 1:
        off_diag = distance_matrix[~np.eye(n, dtype=bool)] if n > 1 else np.array([0.0])
        return float(np.median(off_diag)) if off_diag.size else 0.0

    k_distances = np.sort(distance_matrix, axis=1)[:, k]  # column 0 is self (distance 0)
    sorted_k_distances = np.sort(k_distances)
    n_points = len(sorted_k_distances)

    # Standard kneedle preprocessing: normalize BOTH axes to [0, 1] before
    # measuring perpendicular distance from the chord. Without this, the
    # index axis (range 0..n_points-1, e.g. up to several dozen) and the
    # distance axis (range 0..~1.0) are on wildly different numeric
    # scales, so the raw perpendicular-distance computation is dominated
    # by whichever axis happens to have the larger absolute range and can
    # degenerate to picking an endpoint rather than the true knee (found
    # empirically, round 38, early/conv_a1: unnormalized, this function
    # returned eps=1.0, the maximum possible distance, collapsing every
    # patch into one cluster).
    x = np.arange(n_points, dtype=float)
    y = sorted_k_distances
    x_range = (x[-1] - x[0]) or 1.0
    y_range = (y[-1] - y[0]) or 1.0
    x_norm = (x - x[0]) / x_range
    y_norm = (y - y[0]) / y_range

    x0, y0, x1, y1 = x_norm[0], y_norm[0], x_norm[-1], y_norm[-1]
    line_vec = np.array([x1 - x0, y1 - y0])
    line_len = np.hypot(*line_vec) or 1.0
    line_unit = line_vec / line_len
    diffs = np.stack([x_norm - x0, y_norm - y0], axis=1)
    proj_len = diffs @ line_unit
    proj_points = np.outer(proj_len, line_unit)
    perp_dist = np.linalg.norm(diffs - proj_points, axis=1)
    elbow_idx = int(np.argmax(perp_dist))
    return float(sorted_k_distances[elbow_idx])


def dbscan_cluster_from_distance_matrix(
    patch_ids: list[tuple[int, int]], distance_matrix: np.ndarray, eps: float, min_pts: int,
) -> tuple[list[list[tuple[int, int]]], dict]:
    """Standard DBSCAN (precomputed-distance variant), implemented from
    scratch (no scipy/sklearn in this project's environment -- see round
    37's `patch_timing_clustering.py` module docstring for the same
    constraint). Noise points (density-unreachable from any core point)
    are returned as their OWN singleton clusters -- never dropped, same
    convention as every prior round's clustering (a patch with no
    qualifying match becomes its own cluster, not excluded)."""
    n = len(patch_ids)
    neighbors = [set(np.where(distance_matrix[i] <= eps)[0].tolist()) - {i} for i in range(n)]
    is_core = [len(neighbors[i]) + 1 >= min_pts for i in range(n)]  # +1 counts the point itself

    labels = [-1] * n  # -1 = unassigned/noise
    cluster_id = 0
    for i in range(n):
        if labels[i] != -1 or not is_core[i]:
            continue
        labels[i] = cluster_id
        seeds = list(neighbors[i])
        while seeds:
            j = seeds.pop()
            if labels[j] == -1:
                labels[j] = cluster_id
                if is_core[j]:
                    for k in neighbors[j]:
                        if labels[k] == -1:
                            seeds.append(k)
        cluster_id += 1

    clusters_by_label: dict[int, list[tuple[int, int]]] = {}
    next_noise_id = cluster_id
    for i, lbl in enumerate(labels):
        if lbl == -1:
            lbl = next_noise_id
            next_noise_id += 1
        clusters_by_label.setdefault(lbl, []).append(patch_ids[i])

    clusters = sorted((sorted(v) for v in clusters_by_label.values()), key=len, reverse=True)
    n_noise = sum(1 for lbl in labels if lbl == -1)  # (kept for reporting only; labels list is consumed above)
    return clusters, {
        "eps": eps, "min_pts": min_pts, "n_core_points": sum(is_core),
        "n_clusters_incl_singletons": len(clusters), "n_noise_singletons": n_noise,
    }


def cluster_patches_by_density(
    patch_ids: list[tuple[int, int]], distance_matrix: np.ndarray, k_for_eps: int = 4, min_pts: int = 3,
) -> tuple[list[list[tuple[int, int]]], dict]:
    """DBSCAN-style clustering wrapper with a self-relative `eps` (see
    `estimate_eps_via_knn_elbow`) and a FIXED, small, undifferentiated
    `min_pts` (default 3 -- the standard DBSCAN rule of thumb of
    minPts >= dimensionality+1 for low-dimensional data, here a single
    scalar timing-distance space; NOT tuned per video/dataset). Returns
    the same `(clusters, diagnostic_info)` shape as the hierarchical
    entry points above."""
    n = len(patch_ids)
    if n == 0:
        return [], {"eps": None, "min_pts": min_pts}
    if n == 1:
        return [[patch_ids[0]]], {"eps": None, "min_pts": min_pts}
    k = min(k_for_eps, n - 1)
    eps = estimate_eps_via_knn_elbow(distance_matrix, k)
    return dbscan_cluster_from_distance_matrix(patch_ids, distance_matrix, eps, min_pts)
