"""Phase-aware patch-cluster beat consensus (round 35).

Builds regional beat reconstruction the OTHER way around from the
existing pipeline (pipeline/analyze.py: patches -> region-AVERAGE
waveform -> peak detection -> regional BPM): here, per-patch events are
detected and kept first (beat_events.py, UNCHANGED), patches are grouped
into clusters purely from their own event-timing similarity (see
patch_timing_graph.py -- NEVER from manual_reference region coordinates),
and only THEN is a per-cluster "consensus beat" built from the member
patches' own raw events -- a region-average waveform is never computed
before detection for this pipeline.

Clustering (this round's v1): connected components over the patch-timing
graph after keeping only edges whose `event_time_coincidence_fraction`
clears a SELF-RELATIVE percentile cutoff among that video's OWN computed
pairwise values (never an absolute magic number, same posture as every
other threshold in this project) -- the simplest interpretable choice.
Hierarchical clustering or a DBSCAN-style density rule over the same
edge-feature space are reasonable alternatives for a later round; this
round deliberately does not reach for a trained/ML classifier.

Consensus beat construction reuses cardio_organoid.functional.
beat_episodes.group_events_into_episodes UNCHANGED, treating each PATCH
as a "region" in that function's existing region-first abstraction --
this is not a new grouping algorithm, it is the SAME self-relative
time-chain grouping already used for manual-region beat episodes, run
one level down (patches instead of named regions). The grouping
tolerance is `tolerance_period_fraction * cluster_dominant_period_s`
(the cluster's OWN pooled dominant period, from
beat_episodes.pooled_dominant_period_s over member patches' own
frequencies) -- never a fixed absolute millisecond value.

For a per-cluster BPM/IBI/CV comparable to the existing region-level
pipeline's numbers, each polarity's consensus-beat-episode
representative times are wrapped as synthetic per-cluster
"positive"/"negative" events and passed through
cardio_organoid.functional.beat_cycles.build_beat_cycles /
cardio_organoid.functional.beat_cycle_metrics.compute_region_cycle_metrics
UNCHANGED -- the exact same positive/negative pairing and BPM-from-
median-IBI convention already used everywhere else in this project,
applied one level up (cluster-of-patches instead of one waveform).

No detection threshold, pairing window fraction, or clustering cutoff in
this module is tuned to reproduce any manual observed count/BPM -- every
number reported is self-relative to the video's/cluster's own data.
"""

from __future__ import annotations

import numpy as np

from cardio_organoid.functional.beat_cycle_metrics import compute_region_cycle_metrics
from cardio_organoid.functional.beat_cycles import build_beat_cycles
from cardio_organoid.functional.beat_episodes import (
    BeatEpisode,
    RegionEvent,
    group_events_into_episodes,
    pooled_dominant_period_s,
    suggest_episode_tolerance_s,
)
from cardio_organoid.functional.patch_timing_graph import get_edge


# --------------------------------------------------------------------------
# 1. Clustering -- connected components over a self-relative similarity cutoff.
# --------------------------------------------------------------------------


def cluster_patches_by_timing_similarity(
    graph: dict, similarity_key: str = "event_time_coincidence_fraction", similarity_percentile: float = 75.0,
) -> list[list[tuple[int, int]]]:
    """Connected components of `graph["patch_ids"]` after keeping only
    edges whose `similarity_key` value is at or above the
    `similarity_percentile`-th percentile among ALL of this graph's own
    computed (non-None) edge values for that key -- self-relative to this
    video, never an absolute cutoff. A patch with no qualifying edge to
    any other patch becomes its own singleton cluster (not dropped).
    Returns clusters as sorted lists of patch ids, sorted by cluster size
    (largest first) for convenient reporting.
    """
    patch_ids = graph["patch_ids"]
    edges = graph["edges"]
    values = [v[similarity_key] for v in edges.values() if v.get(similarity_key) is not None]
    if not values:
        return [[pid] for pid in patch_ids]
    cutoff = float(np.percentile(values, similarity_percentile))

    adjacency: dict[tuple[int, int], set] = {pid: set() for pid in patch_ids}
    for (a, b), feats in edges.items():
        val = feats.get(similarity_key)
        if val is not None and val >= cutoff:
            adjacency[a].add(b)
            adjacency[b].add(a)

    visited: set = set()
    clusters: list[list[tuple[int, int]]] = []
    for pid in patch_ids:
        if pid in visited:
            continue
        stack, component = [pid], []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            component.append(cur)
            stack.extend(adjacency[cur] - visited)
        clusters.append(sorted(component))
    clusters.sort(key=len, reverse=True)
    return clusters


# --------------------------------------------------------------------------
# 2. Phase-aware consensus beats (per polarity, per cluster) -- built from
#    beat_episodes.group_events_into_episodes UNCHANGED, patches as
#    "regions".
# --------------------------------------------------------------------------


def episode_to_consensus_beat_record(episode: BeatEpisode, graph: dict) -> dict:
    """Enrich one already-built BeatEpisode (see module docstring -- built
    with patches as 'regions') with the extra fields this round's
    phase-aware consensus explicitly asks for: median event time, event-
    time dispersion, median amplitude, spatial extent (bounding box of
    participating patch (row, col) ids), and the within-cluster pairwise
    lag structure (looked up from the ALREADY-COMPUTED timing graph, not
    recomputed here)."""
    times = [e.event_time for e in episode.member_events]
    amplitudes = [e.amplitude for e in episode.member_events]
    patch_ids = sorted({tuple(int(x) for x in e.region_id.split("_")) for e in episode.member_events})
    rows, cols = [p[0] for p in patch_ids], [p[1] for p in patch_ids]

    lag_structure = []
    for i in range(len(patch_ids)):
        for j in range(i + 1, len(patch_ids)):
            a, b = patch_ids[i], patch_ids[j]
            feats = get_edge(graph, a, b)
            if feats is not None:
                lag_structure.append({
                    "patch_a": list(a), "patch_b": list(b),
                    "best_lag_s": feats["best_lag_s"], "median_event_delay_s": feats["median_event_delay_s"],
                })

    return {
        "episode_id": episode.episode_id,
        "polarity": episode.polarity,
        "event_time": float(np.mean(times)),
        "median_event_time": float(np.median(times)),
        "event_time_dispersion_s": float(np.std(times)) if len(times) > 1 else 0.0,
        "median_amplitude": float(np.median(amplitudes)) if amplitudes else None,
        "participating_patch_ids": [list(p) for p in patch_ids],
        "participating_patch_fraction": episode.spatial_participation_fraction,
        "spatial_extent": {
            "row_min": min(rows), "row_max": max(rows), "col_min": min(cols), "col_max": max(cols),
            "n_distinct_patches": len(patch_ids),
        },
        "within_cluster_lag_structure": lag_structure,
    }


def build_consensus_beats_for_cluster(
    cluster_patch_ids: list[tuple[int, int]],
    positive_events_by_patch: dict[tuple[int, int], list[RegionEvent]],
    negative_events_by_patch: dict[tuple[int, int], list[RegionEvent]],
    dominant_freq_by_patch: dict[tuple[int, int], float | None],
    graph: dict,
    tolerance_period_fraction: float,
    fallback_tolerance_s: float,
    global_min_participation_fraction: float,
) -> dict:
    """Per-polarity consensus-beat-episode construction for ONE cluster
    (see module docstring). Returns positive/negative consensus-beat
    records (enriched, see episode_to_consensus_beat_record) plus the
    cluster's own pooled dominant period and the tolerance actually used
    -- both fully traceable, never hidden defaults."""
    pooled_period_s = pooled_dominant_period_s([dominant_freq_by_patch.get(pid) for pid in cluster_patch_ids])
    tolerance_s = suggest_episode_tolerance_s(pooled_period_s, tolerance_period_fraction) or fallback_tolerance_s

    pos_by_patch = {f"{p[0]}_{p[1]}": positive_events_by_patch.get(p, []) for p in cluster_patch_ids}
    neg_by_patch = {f"{p[0]}_{p[1]}": negative_events_by_patch.get(p, []) for p in cluster_patch_ids}
    positive_episodes = group_events_into_episodes(pos_by_patch, tolerance_s, global_min_participation_fraction)
    negative_episodes = group_events_into_episodes(neg_by_patch, tolerance_s, global_min_participation_fraction)

    return {
        "cluster_patch_ids": [list(p) for p in cluster_patch_ids],
        "pooled_period_s": pooled_period_s,
        "tolerance_period_fraction": tolerance_period_fraction,
        "tolerance_s": tolerance_s,
        "positive_consensus_beats": [episode_to_consensus_beat_record(e, graph) for e in positive_episodes],
        "negative_consensus_beats": [episode_to_consensus_beat_record(e, graph) for e in negative_episodes],
    }


# --------------------------------------------------------------------------
# 3. Cluster-level BPM/IBI/CV -- the SAME positive/negative beat_cycle
#    pairing + compute_region_cycle_metrics convention used everywhere
#    else in this project, applied to consensus-beat representative
#    times instead of raw single-waveform events.
# --------------------------------------------------------------------------


def consensus_beat_cycle_metrics(
    cluster_id: str, consensus_result: dict, pairing_window_period_fraction: float, fallback_pairing_window_s: float,
) -> dict:
    """Wraps each polarity's consensus-beat representative times
    (`event_time`, the mean of that beat's member patch-event times) as
    synthetic RegionEvent instances for `cluster_id`, then pairs them
    with build_beat_cycles UNCHANGED and reduces with
    compute_region_cycle_metrics UNCHANGED -- giving a consensus BPM/
    median-IBI/CV directly comparable, apples-to-apples, to every
    existing manual-region's Farneback region-average BPM (same pairing
    algorithm and same metric definitions, just fed a different, phase-
    aware event stream)."""
    pos_beats = consensus_result["positive_consensus_beats"]
    neg_beats = consensus_result["negative_consensus_beats"]
    pooled_period_s = consensus_result["pooled_period_s"]
    pairing_window_s = (
        pooled_period_s * pairing_window_period_fraction if pooled_period_s else fallback_pairing_window_s
    )

    pos_events = [
        RegionEvent(region_id=cluster_id, event_time=b["event_time"], original_frame=0,
                    amplitude=b["median_amplitude"] or 0.0, polarity="positive", prominence=0.0)
        for b in pos_beats
    ]
    neg_events = [
        RegionEvent(region_id=cluster_id, event_time=b["event_time"], original_frame=0,
                    amplitude=b["median_amplitude"] or 0.0, polarity="negative", prominence=0.0)
        for b in neg_beats
    ]
    cycles = build_beat_cycles(pos_events, neg_events, pairing_window_s)
    pos_amp_by_time = {round(e.event_time, 6): e.amplitude for e in pos_events}
    neg_amp_by_time = {round(e.event_time, 6): e.amplitude for e in neg_events}
    metrics = compute_region_cycle_metrics([c.to_dict() for c in cycles], pos_amp_by_time, neg_amp_by_time)
    return {"pairing_window_s": pairing_window_s, "cycles": [c.to_dict() for c in cycles], "metrics": metrics}


# --------------------------------------------------------------------------
# 4. Per-cluster regional-output summary (A4) -- descriptive fields not
#    already covered by consensus_beat_cycle_metrics's BPM/IBI/CV.
# --------------------------------------------------------------------------


def cluster_regional_summary(
    cluster_patch_ids: list[tuple[int, int]],
    consensus_result: dict,
    cycle_metrics: dict,
    per_patch_amplitude: dict[tuple[int, int], float],
    waveforms_by_patch: dict[tuple[int, int], np.ndarray],
) -> dict:
    """Descriptive per-cluster summary: spatial extent, dominant patch
    (highest own waveform amplitude), and a representative waveform --
    DELIBERATELY the dominant patch's OWN raw waveform, not a fresh
    region-mean recomputation, since re-averaging here would silently
    reintroduce the exact phase-cancellation effect this whole module was
    built to avoid measuring around."""
    rows = [p[0] for p in cluster_patch_ids]
    cols = [p[1] for p in cluster_patch_ids]
    dominant_patch = max(cluster_patch_ids, key=lambda p: per_patch_amplitude.get(p, 0.0))

    all_beats = consensus_result["positive_consensus_beats"] + consensus_result["negative_consensus_beats"]
    participation_fraction = (
        float(np.mean([b["participating_patch_fraction"] for b in all_beats if b["participating_patch_fraction"] is not None]))
        if all_beats else None
    )
    phase_dispersion_s = float(np.mean([b["event_time_dispersion_s"] for b in all_beats])) if all_beats else None
    pooled_period_s = consensus_result["pooled_period_s"]
    phase_dispersion_fraction_of_period = (
        phase_dispersion_s / pooled_period_s if (phase_dispersion_s is not None and pooled_period_s) else None
    )

    high = cycle_metrics["metrics"]["tiers"]["high_confidence_only"]
    return {
        "n_patches": len(cluster_patch_ids),
        "patch_ids": [list(p) for p in cluster_patch_ids],
        "spatial_extent": {"row_min": min(rows), "row_max": max(rows), "col_min": min(cols), "col_max": max(cols)},
        "dominant_patch": list(dominant_patch),
        "representative_waveform_source": "dominant_patch_own_waveform",
        "representative_waveform": waveforms_by_patch[dominant_patch],
        "pooled_dominant_period_s": pooled_period_s,
        "consensus_beat_count_high_confidence": high["beat_cycle_count"],
        "consensus_bpm_high_confidence": high["provisional_bpm_from_median_ibi"],
        "median_ibi_s_high_confidence": high["interval_stats"]["median_ibi_s"],
        "cv_ibi_high_confidence": high["interval_stats"]["cv_ibi"],
        "participation_fraction_mean_across_consensus_beats": participation_fraction,
        "phase_dispersion_s_mean": phase_dispersion_s,
        "phase_dispersion_fraction_of_pooled_period": phase_dispersion_fraction_of_period,
    }
