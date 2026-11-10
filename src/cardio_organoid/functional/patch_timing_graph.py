"""Patch-level event-timing graph (round 35).

Motivation: the existing patch/region pipeline
(pipeline/analyze.py -> spatial_patches.py -> beat_events.py) averages a
region's member patches into ONE waveform *before* any event detection
happens. Round 33's D10/slit/a1 secondary_region diagnostic
(scripts/d10_secondary_cause_diagnostic.py) found that region's 5
individual patches each run at ~31-38 BPM (matching primary/tertiary),
not the ~66.8 BPM the region-AVERAGED waveform produces -- "multiple
patch phase mixture" was the strongest surviving hypothesis. This module
builds the machinery to test that hypothesis directly and generally
(not just for one hand-picked region): instead of averaging first, keep
every patch's own detected events, and describe how pairs of patches'
event timing relate to each other (correlation at the best lag, how
often their events coincide, typical delay, how similar their own
dominant frequencies are, and how spatially close they are) -- a graph
over patches, built ENTIRELY from each patch's own already-detected
events. Manual reference region coordinates (results/manual_reference/
*.json) play NO role in constructing this graph; they are used only
downstream, for reporting, to label which auto-discovered patch cluster
happens to spatially overlap a named manual region.

Every function here is read-only over already-computed patch waveforms/
events (from the UNCHANGED beat_events.py detector) -- no Farneback call,
no threshold change, no video I/O.
"""

from __future__ import annotations

import numpy as np

from cardio_organoid.functional.regional_lag_diagnostics import cross_correlation_curve


def chebyshev_adjacency(patch_a: tuple[int, int], patch_b: tuple[int, int]) -> int:
    """Grid (row, col) Chebyshev distance -- 1 for 8-connected neighbors,
    0 only if the same patch. A descriptive EDGE FEATURE only (see module
    docstring): two spatially distant patches with strong timing
    similarity must still be allowed to form a graph edge -- this value
    is never used to forbid an edge, only to describe it."""
    return max(abs(patch_a[0] - patch_b[0]), abs(patch_a[1] - patch_b[1]))


def event_time_coincidence_fraction(
    events_a_s: list[float], events_b_s: list[float], tolerance_s: float,
) -> float | None:
    """Symmetric fraction of patch A's and patch B's own (already
    detected, unchanged) event times that have a matching event in the
    OTHER patch within +/-tolerance_s (nearest-unused-neighbor matching,
    same greedy style as beat_events.regional_activation_delay). Returns
    (n_matched_a_to_b + n_matched_b_to_a) / (n_events_a + n_events_b) --
    None if both patches have zero events (nothing to compare)."""
    if not events_a_s and not events_b_s:
        return None

    def _match_count(times_x: list[float], times_y: list[float]) -> int:
        used_y = set()
        n_matched = 0
        for tx in times_x:
            candidates = [(j, ty) for j, ty in enumerate(times_y) if j not in used_y and abs(ty - tx) <= tolerance_s]
            if not candidates:
                continue
            j, _ty = min(candidates, key=lambda jt: abs(jt[1] - tx))
            used_y.add(j)
            n_matched += 1
        return n_matched

    matched_a_to_b = _match_count(events_a_s, events_b_s)
    matched_b_to_a = _match_count(events_b_s, events_a_s)
    denom = len(events_a_s) + len(events_b_s)
    return (matched_a_to_b + matched_b_to_a) / denom if denom else None


def median_event_delay_s(events_a_s: list[float], events_b_s: list[float], max_match_window_s: float) -> float | None:
    """Median SIGNED delay (b - a) over nearest-unused-neighbor-matched
    event pairs within `max_match_window_s` -- a per-event-pair
    complement to `best_lag_correlation`'s single aggregate lag (which
    assumes one fixed delay for the whole recording). None if no pair
    matches."""
    if not events_a_s or not events_b_s:
        return None
    used_b = set()
    delays = []
    for ta in events_a_s:
        candidates = [(j, tb) for j, tb in enumerate(events_b_s) if j not in used_b and abs(tb - ta) <= max_match_window_s]
        if not candidates:
            continue
        j, tb = min(candidates, key=lambda jt: abs(jt[1] - ta))
        used_b.add(j)
        delays.append(tb - ta)
    return float(np.median(delays)) if delays else None


def frequency_similarity(freq_a_hz: float | None, freq_b_hz: float | None) -> float | None:
    """1 - |fa - fb| / max(fa, fb), in [0, 1] (1 = identical dominant
    frequency) -- a RELATIVE similarity, never an absolute Hz difference,
    so it means the same thing whether two patches both run near 0.5 Hz
    or both near 2 Hz. None if either patch has no valid dominant
    frequency estimate."""
    if not freq_a_hz or not freq_b_hz or freq_a_hz <= 0 or freq_b_hz <= 0:
        return None
    return float(1.0 - abs(freq_a_hz - freq_b_hz) / max(freq_a_hz, freq_b_hz))


def compute_pairwise_timing_features(
    patch_a: tuple[int, int],
    patch_b: tuple[int, int],
    events_a_s: list[float],
    events_b_s: list[float],
    dominant_freq_a_hz: float | None,
    dominant_freq_b_hz: float | None,
    waveform_a: np.ndarray,
    waveform_b: np.ndarray,
    sample_interval_s: float,
    coincidence_tolerance_s: float,
    max_lag_samples: int,
) -> dict:
    """One graph edge's full feature bundle between two patches, built
    ENTIRELY from each patch's own waveform/events -- no manual region
    coordinate is read here. `events_a_s`/`events_b_s` are each patch's
    OWN pooled (positive + negative) raw event times -- pooling both
    polarities here is a coarse timing-similarity signal only; the actual
    beat reconstruction (see patch_beat_consensus.py) keeps polarities
    separate, matching this project's established convention.
    """
    curve = cross_correlation_curve(waveform_a, waveform_b, max_lag_samples)
    best_lag_samples = curve["best_lag_samples"]
    return {
        "best_lag_samples": best_lag_samples,
        "best_lag_s": best_lag_samples * sample_interval_s if best_lag_samples is not None else None,
        "best_correlation": curve["best_correlation"],
        "event_time_coincidence_fraction": event_time_coincidence_fraction(events_a_s, events_b_s, coincidence_tolerance_s),
        "median_event_delay_s": median_event_delay_s(events_a_s, events_b_s, coincidence_tolerance_s),
        "frequency_similarity": frequency_similarity(dominant_freq_a_hz, dominant_freq_b_hz),
        "spatial_adjacency_chebyshev": chebyshev_adjacency(patch_a, patch_b),
    }


def build_patch_timing_graph(
    patch_ids: list[tuple[int, int]],
    waveforms_by_patch: dict[tuple[int, int], np.ndarray],
    pooled_events_by_patch_s: dict[tuple[int, int], list[float]],
    dominant_freq_by_patch: dict[tuple[int, int], float | None],
    sample_interval_s: float,
    coincidence_tolerance_s: float,
    max_lag_s: float,
) -> dict:
    """Full pairwise graph over `patch_ids` (every pair -- O(n^2), fine
    for this project's per-video patch counts of a few dozen to ~100).
    Returns {"patch_ids": [...], "edges": {(a, b): features, ...},
    "coincidence_tolerance_s":, "max_lag_s":} -- edges keyed by the
    (patch_a, patch_b) tuple pair with patch_a < patch_b (lexicographic),
    each looked up in EITHER order by callers.
    """
    max_lag_samples = max(1, int(round(max_lag_s / sample_interval_s)))
    edges: dict[tuple, dict] = {}
    ordered = sorted(patch_ids)
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            a, b = ordered[i], ordered[j]
            edges[(a, b)] = compute_pairwise_timing_features(
                a, b,
                pooled_events_by_patch_s.get(a, []), pooled_events_by_patch_s.get(b, []),
                dominant_freq_by_patch.get(a), dominant_freq_by_patch.get(b),
                waveforms_by_patch[a], waveforms_by_patch[b],
                sample_interval_s, coincidence_tolerance_s, max_lag_samples,
            )
    return {
        "patch_ids": ordered, "edges": edges,
        "coincidence_tolerance_s": coincidence_tolerance_s, "max_lag_s": max_lag_s,
    }


def get_edge(graph: dict, patch_a: tuple[int, int], patch_b: tuple[int, int]) -> dict | None:
    """Order-independent edge lookup (edges are stored with the
    lexicographically smaller patch id first -- see build_patch_timing_graph)."""
    key = (patch_a, patch_b) if patch_a < patch_b else (patch_b, patch_a)
    return graph["edges"].get(key)
