"""Reliability-gated, confidence-weighted, timing-similarity hierarchical
clustering of patches (round 37).

Round 35's `patch_beat_consensus.cluster_patches_by_timing_similarity`
(connected components after a self-relative percentile cutoff) is
effectively single-linkage clustering -- known, on this project's own
data, to chain most of a video's patches into one mega-cluster (D10:
53/69 patches at the default 75th-percentile cutoff), which fully
contains secondary_region but does not separate it from tertiary_region
or the rest of the organoid.

This module adds AVERAGE-linkage agglomerative clustering (a standard,
interpretable alternative that resists single-link chaining, per this
project's stated preference for simple/interpretable methods before any
ML classifier) over the SAME timing-similarity edge features
`patch_timing_graph.py` already computes -- it does not recompute timing
features itself.

Per explicit project instruction this round:
  - Spatial adjacency is NEVER part of the clustering distance -- only
    `patch_timing_graph.py`'s TIMING features (event-time coincidence
    fraction, best-lag correlation, dominant-frequency similarity).
  - Observability/reliability is used ONLY as a pre-clustering GATE
    (drop patches already labeled NOT_OBSERVABLE by
    `reliable_hotspot_candidates.py`, round 36) and as a continuous
    per-patch WEIGHT that discounts (never zeroes, never hard-thresholds
    beyond the gate) low-confidence patches' influence on distance --
    never a clustering FEATURE in its own right, and categorical state
    labels (A/B/C/D) are never used as the clustering key.
  - No merge-cutoff, weight formula, or distance combination here is
    tuned to reproduce any manual observed BPM/count -- the cut point is
    derived from THIS graph's own merge-height sequence (a gap/elbow
    heuristic), never a fixed number.
"""

from __future__ import annotations

import numpy as np

from cardio_organoid.functional.patch_timing_graph import get_edge


def pairwise_timing_distance(edge_features: dict | None) -> float:
    """Combine an edge's TIMING-ONLY features (event_time_coincidence_
    fraction, best_correlation rescaled from [-1,1] to [0,1], frequency_
    similarity) into one distance in [0, 1] -- simple, transparent mean of
    whichever of the three are available (never spatial_adjacency, per
    module docstring). `edge_features=None` (e.g. a pair never computed,
    or self-distance) returns 1.0 (maximally dissimilar) rather than
    guessing a value -- callers are expected to pass 0.0 for the
    diagonal explicitly, not rely on this function for it.
    """
    if edge_features is None:
        return 1.0
    similarities = []
    coincidence = edge_features.get("event_time_coincidence_fraction")
    if coincidence is not None:
        similarities.append(coincidence)
    correlation = edge_features.get("best_correlation")
    if correlation is not None:
        similarities.append((correlation + 1.0) / 2.0)
    freq_sim = edge_features.get("frequency_similarity")
    if freq_sim is not None:
        similarities.append(freq_sim)
    if not similarities:
        return 1.0
    return float(1.0 - np.mean(similarities))


def build_confidence_weighted_distance_matrix(
    patch_ids: list[tuple[int, int]], graph: dict, confidence_by_patch: dict[tuple[int, int], float],
) -> np.ndarray:
    """(n, n) distance matrix over `patch_ids` (already the GATED,
    reliable subset -- this function does not gate anything itself),
    using ONLY `patch_timing_graph.py`'s timing-similarity edge features
    (see `pairwise_timing_distance`), each pair's raw distance then
    WEIGHTED (never hard-thresholded) by the geometric mean of the two
    patches' own `confidence_by_patch` values (each expected in [0, 1],
    e.g. round-36's `final_provisional_confidence`): effective_similarity
    = raw_similarity * sqrt(conf_i * conf_j), so two low-confidence
    patches must show a stronger raw timing match than two high-
    confidence patches to end up at the same effective distance --
    confidence discounts influence, it never by itself forces a merge or
    a split. Missing/absent confidence values default to 1.0 (no
    discount) rather than silently excluding the patch (gating already
    happened before this function is called).
    """
    n = len(patch_ids)
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = patch_ids[i], patch_ids[j]
            raw_similarity = 1.0 - pairwise_timing_distance(get_edge(graph, a, b))
            conf_a = confidence_by_patch.get(a, 1.0)
            conf_b = confidence_by_patch.get(b, 1.0)
            weight = float(np.sqrt(max(conf_a, 0.0) * max(conf_b, 0.0)))
            effective_similarity = raw_similarity * weight
            d = 1.0 - effective_similarity
            dist[i, j] = d
            dist[j, i] = d
    return dist


def average_linkage_clusters(
    patch_ids: list[tuple[int, int]], distance_matrix: np.ndarray,
) -> tuple[list[list[tuple[int, int]]], list[float]]:
    """Agglomerative AVERAGE-linkage clustering (no external dependency --
    scipy is not installed in this project's environment; this is a
    small, standard, from-scratch implementation, fine for this
    project's per-video patch counts of a few dozen). Runs the full
    agglomeration down to one single cluster, returning EVERY
    intermediate membership snapshot (`merge_history[k]` = clusters after
    k merges, `merge_history[0]` = all singletons) alongside the parallel
    `merge_heights` list (the average-linkage distance at each merge, in
    merge order) so a caller (see `cut_at_gap`) can decide where to cut;
    this function itself does not decide a cut point. Average-linkage
    distance between two clusters is the mean of every ORIGINAL pairwise
    point distance across their members (recomputed from
    `distance_matrix` each merge, not incrementally updated -- exact,
    simple, and cheap at this project's scale).
    """
    n = len(patch_ids)
    if n == 0:
        return [{}], []
    if n == 1:
        return [{0: [0]}], []

    members = {i: [i] for i in range(n)}
    active = list(range(n))
    merge_heights: list[float] = []
    merge_history: list[dict[int, list[int]]] = [dict(members)]

    while len(active) > 1:
        best_pair, best_dist = None, None
        for a_idx in range(len(active)):
            for b_idx in range(a_idx + 1, len(active)):
                ci, cj = active[a_idx], active[b_idx]
                mi, mj = members[ci], members[cj]
                d = float(np.mean(distance_matrix[np.ix_(mi, mj)]))
                if best_dist is None or d < best_dist:
                    best_dist, best_pair = d, (ci, cj)
        ci, cj = best_pair
        members[ci] = members[ci] + members[cj]
        del members[cj]
        active.remove(cj)
        merge_heights.append(best_dist)
        merge_history.append({k: list(v) for k, v in members.items()})

    # merge_history[k] is the cluster membership right after k merges
    # (k=0: all singletons). Callers combine this with merge_heights to
    # cut at any number of merges.
    return merge_history, merge_heights


def cut_at_gap(merge_history: list[dict[int, list[int]]], merge_heights: list[float], patch_ids: list[tuple[int, int]]) -> tuple[list[list[tuple[int, int]]], dict]:
    """Self-relative cut: find the LARGEST gap between consecutive merge
    heights (a standard, simple 'elbow' heuristic for hierarchical
    clustering -- never a fixed distance value, never tuned to any
    manual target) and cut the dendrogram just BEFORE that gap (i.e. keep
    every merge up to and including the one before the biggest jump,
    reject the rest). Returns (clusters_as_patch_ids, diagnostic_info).
    If there are fewer than 2 merges (e.g. n<=2 patches), returns the
    finest available clustering (no cut needed).

    IMPORTANT (found empirically on this project's own data, round 37):
    naively searching the WHOLE merge sequence for the single biggest gap
    almost always picks the very FIRST gap (between the closest pair and
    the second-closest pair), because merge heights typically start very
    small and climb gradually and near-monotonically afterward -- the
    first step up is often numerically the largest single jump even
    though it reflects almost no real consolidation (e.g. this was
    confirmed to collapse mature/D10_slit_a1's 36 reliable patches into
    35 near-singleton clusters, keeping only 1 of 34 possible merges,
    which is not a meaningful clustering). This function therefore only
    searches for the gap in the SECOND HALF of the merge sequence (i.e.
    only after at least half of all possible pairwise-closest merges have
    already happened) -- a structural, self-relative requirement that
    real consolidation must occur before a 'natural break' is declared,
    not a value tuned to any manual target.
    """
    if len(merge_heights) < 2:
        cut_index = len(merge_heights)
    else:
        skip = len(merge_heights) // 2
        gaps = np.diff(merge_heights)
        search_gaps = gaps[skip:]
        if search_gaps.size == 0:
            cut_index = len(merge_heights)
        else:
            cut_index = skip + int(np.argmax(search_gaps)) + 1  # keep merges [0, cut_index)
    membership = merge_history[cut_index]
    clusters = [sorted(patch_ids[i] for i in members) for members in membership.values()]
    clusters.sort(key=len, reverse=True)
    return clusters, {
        "n_merges_total": len(merge_heights), "n_merges_kept": cut_index,
        "merge_heights": merge_heights, "cut_merge_height": merge_heights[cut_index - 1] if cut_index > 0 else None,
        "next_merge_height_after_cut": merge_heights[cut_index] if cut_index < len(merge_heights) else None,
    }


def cluster_patches_by_timing_similarity_hierarchical(
    patch_ids: list[tuple[int, int]], graph: dict, confidence_by_patch: dict[tuple[int, int], float],
) -> tuple[list[list[tuple[int, int]]], dict]:
    """End-to-end convenience wrapper: gate is the caller's job (pass
    only already-reliable `patch_ids`, e.g. after excluding round-36's
    NOT_OBSERVABLE state) -- build the confidence-weighted timing-only
    distance matrix, agglomerate, and cut at the largest merge-height
    gap. Returns (clusters, diagnostic_info) in the SAME
    `list[list[tuple[int,int]]]` shape as round 35's
    `patch_beat_consensus.cluster_patches_by_timing_similarity`, so every
    downstream consensus-building function (`build_consensus_beats_for_
    cluster`, `consensus_beat_cycle_metrics`, `cluster_regional_summary`)
    is reused completely unchanged.
    """
    ordered = sorted(patch_ids)
    dist = build_confidence_weighted_distance_matrix(ordered, graph, confidence_by_patch)
    merge_history, merge_heights = average_linkage_clusters(ordered, dist)
    clusters, cut_info = cut_at_gap(merge_history, merge_heights, ordered)
    return clusters, {"distance_matrix": dist, **cut_info}
