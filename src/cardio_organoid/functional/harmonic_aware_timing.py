"""Robust, dropout- and harmonic-tolerant patch-timing similarity (round 38).

Diagnosis (round 38, reproducing/reading round 33/35/37's own already-saved
output -- no canonical file touched): round 37's clustering
(`patch_timing_clustering.pairwise_timing_distance`) combines three
edge features -- event-time coincidence FRACTION (exclusive nearest-
neighbor match over a fixed tolerance), lagged cross-correlation, and
`frequency_similarity` (built from each patch's own
`beat_events.region_beat_summary` -> `dominant_spectral_peak`, an FFT/
periodogram peak-pick over that ONE patch's own short, often-noisy
waveform). Two structural weaknesses were found in D10/slit/a1:

  1. **Harmonic lock**: per-patch `dominant_frequency_hz` for several of
     D10's own secondary_region AND tertiary_region patches is
     ~1.97-2.02x round 33's independently-derived (direct event
     inspection, not periodogram peak-picking) per-patch rate estimate
     for the SAME patches (e.g. patch (3,11): round 33 estimated ~31.65
     BPM; round 35/37's own `dominant_frequency_hz`-derived BPM for that
     same patch is 62.27 BPM -- a 1.97x ratio, and 3 more secondary/
     tertiary patches show the same ~2.0x ratio). This is the SAME
     harmonic-lock failure mode round 33 already diagnosed for the
     REGION-AVERAGE waveform (66.8 vs ~33-38 BPM) -- it turns out to
     ALSO corrupt the per-patch spectral estimate that round 35/37's
     `frequency_similarity` and episode-tolerance computation both
     depend on. Two patches that are truly at the same biological rate
     therefore often show LOW `frequency_similarity` to each other
     whenever one patch's own periodogram happened to lock onto the
     fundamental and the other's locked onto the 2nd harmonic --
     inflating their pairwise distance for a reason that has nothing to
     do with their real timing relationship.
  2. **Event-count dropout**: D10 secondary/tertiary patches' own pooled
     (positive+negative) event counts range from 21 to 54 over the same
     ~30s recording (round 35's `per_patch_features`) -- some patches
     detect noticeably more or fewer raw events than others (weaker-
     amplitude patches drop legitimate beats; noisier patches pick up
     extra spurious peaks). Round 37's `event_time_coincidence_fraction`
     divides the matched-pair count by the SUM of both patches' own
     event counts -- so any asymmetric dropout/over-detection directly
     lowers the fraction even when every event that WAS matched lines up
     well, conflating "these patches are not the same population" with
     "one of them has patchier detection."

  3. **Lag-invariant correlation conflates "same rate" with "same
     population"**: found empirically (round 38, D10) AFTER fixing (1)
     and (2) above changed nothing for D10 -- because when EVERY patch
     in a group shares the SAME harmonic-lock bias (as D10's secondary
     AND tertiary patches mostly do), the relative-similarity formula
     `1-|x-y|/max(x,y)` is exactly scale-invariant to a uniform harmonic
     multiplier, so (1) alone cannot fix a case where the whole group is
     consistently mis-locked the same way. The actual remaining driver:
     round 35/37's `best_correlation` searches for the BEST correlation
     at ANY lag up to +/-1.0s (`patch_timing_graph.compute_pairwise_
     timing_features`'s `max_lag_samples`) -- but D10's own manual
     reference explicitly documents primary_region -> secondary_region
     -> tertiary_region as ONE sequentially-propagating beat (not 3
     independently-paced oscillators), so two patches from genuinely
     DIFFERENT, real, differently-phased regions sharing the SAME
     underlying rate will, correctly, show a high correlation once
     shifted to their own true propagation lag -- best-lag correlation
     cannot tell "these are literally the same local motion" apart from
     "these are two different regions of one coordinated organoid beating
     at the same rate with a real regional delay between them." This
     module therefore computes correlation TWICE: `narrow_lag_
     correlation` (searched only within +/-`coincidence_tolerance_s`,
     the SAME natural timescale already used for event matching --
     synchrony, not mere same-rate periodicity) feeds the distance;
     `best_correlation`/`best_lag_s` (full `max_lag_s` window, unchanged
     from round 35/37) remain available as a SEPARATE diagnostic field
     only, per this project's convention of keeping distinct evidence
     lines separately reportable rather than conflating them (see
     `motion_evidence.py`).

This module does not change `beat_events.py`, `patch_timing_graph.py`,
or `patch_timing_clustering.py` -- it adds a drop-in-compatible, more
robust alternative edge-feature/distance function for a NEW clustering
entry point (see `adaptive_timing_clustering.py`). Every function here
is read-only over already-detected events/waveforms; no detection
threshold changes, no new Farneback pass, no manual-region or manual-
beat-count input.
"""

from __future__ import annotations

import numpy as np

from cardio_organoid.functional.patch_timing_graph import median_event_delay_s
from cardio_organoid.functional.regional_lag_diagnostics import cross_correlation_curve

# Harmonic candidate ratios checked when comparing two patches' own
# dominant periods -- 1.0 (same rate), 2.0 / 0.5 (one patch's estimate is
# a 2nd-harmonic lock of the other's fundamental, per the round-38
# diagnosis above). Fixed, symmetric, and applied IDENTICALLY to every
# patch pair in every video -- never chosen per-pair to force a
# particular answer.
HARMONIC_RATIO_CANDIDATES: tuple[float, ...] = (0.5, 1.0, 2.0)

# A non-1.0 harmonic ratio is only preferred over the native (1.0)
# comparison if it improves the self-relative similarity by more than
# this margin -- prevents flip-flopping between 1x/2x on noise-level
# differences when the two periods already agree reasonably well at 1x.
DEFAULT_HARMONIC_PREFERENCE_MARGIN = 0.02


def harmonic_aware_period_similarity(
    period_a_s: float | None,
    period_b_s: float | None,
    harmonic_ratios: tuple[float, ...] = HARMONIC_RATIO_CANDIDATES,
    preference_margin: float = DEFAULT_HARMONIC_PREFERENCE_MARGIN,
) -> dict:
    """Relative period similarity, but checking whether B's period at
    1x, 2x, or 0.5x scale best matches A's -- see module docstring. For
    each candidate ratio r, compares period_a against (period_b * r)
    with the SAME relative-similarity formula frequency_similarity
    already used elsewhere in this project (1 - |x-y|/max(x,y), in
    [0, 1]). Returns {"similarity": best value, "best_ratio": the r that
    achieved it (1.0 unless a strictly-better-by-`preference_margin`
    harmonic match was found), "similarity_at_native_ratio": the plain
    1x-only comparison, for transparency/comparison}. None-safe: returns
    all-None if either period is missing or non-positive.
    """
    if not period_a_s or not period_b_s or period_a_s <= 0 or period_b_s <= 0:
        return {"similarity": None, "best_ratio": None, "similarity_at_native_ratio": None}

    def _sim(x: float, y: float) -> float:
        return 1.0 - abs(x - y) / max(x, y)

    native_sim = _sim(period_a_s, period_b_s)
    best_ratio, best_sim = 1.0, native_sim
    for r in harmonic_ratios:
        if r == 1.0:
            continue
        candidate_sim = _sim(period_a_s, period_b_s * r)
        if candidate_sim > best_sim + preference_margin:
            best_ratio, best_sim = r, candidate_sim
    return {"similarity": float(best_sim), "best_ratio": float(best_ratio), "similarity_at_native_ratio": float(native_sim)}


def soft_event_match_scores(
    events_a_s: list[float], events_b_s: list[float], tolerance_s: float,
) -> dict:
    """Dropout-tolerant timing-coincidence score between two patches' own
    (already-detected, unchanged) pooled event times.

    Unlike round 37's `event_time_coincidence_fraction` (exclusive
    nearest-unused-neighbor matching, one fraction over the SUM of both
    patches' event counts -- see module docstring, weakness 2), this:
      - matches each event in A to its nearest event in B NON-exclusively
        (B's events may be reused -- a missing/dropped event in one
        patch never "steals" a match slot from a correctly-detected one),
      - scores each match with a smooth triangular decay (1.0 at zero
        gap, 0.0 at/beyond `tolerance_s`) instead of a hard boolean
        inside/outside the tolerance window, so a slightly-late detection
        is not scored identically to a total miss,
      - reports each direction's own recall (`recall_a_given_b`: how well
        A's events are covered by B; `recall_b_given_a`: the reverse)
        SEPARATELY, then combines them with the harmonic mean (an F1-like
        combination: a population where one patch drops half its beats
        but every retained beat lines up precisely still scores
        moderately well, rather than being penalized by a shrinking
        shared denominator).

    Returns {"recall_a_given_b", "recall_b_given_a", "harmonic_mean",
    "n_events_a", "n_events_b"}; harmonic_mean is None only if BOTH
    patches have zero events (nothing to compare), matching round 37's
    None-means-nothing-to-compare convention -- a patch with zero events
    compared against a non-empty patch scores 0.0 (fully mismatched,
    not "unknown").
    """
    n_a, n_b = len(events_a_s), len(events_b_s)
    if n_a == 0 and n_b == 0:
        return {"recall_a_given_b": None, "recall_b_given_a": None, "harmonic_mean": None, "n_events_a": 0, "n_events_b": 0}

    def _recall(times_x: list[float], times_y: list[float]) -> float:
        if not times_x:
            return 0.0
        if not times_y:
            return 0.0
        scores = []
        for tx in times_x:
            nearest_gap = min(abs(ty - tx) for ty in times_y)
            scores.append(max(0.0, 1.0 - nearest_gap / tolerance_s) if tolerance_s > 0 else float(nearest_gap == 0.0))
        return float(np.mean(scores))

    recall_a_given_b = _recall(events_a_s, events_b_s)
    recall_b_given_a = _recall(events_b_s, events_a_s)
    denom = recall_a_given_b + recall_b_given_a
    harmonic_mean = (2.0 * recall_a_given_b * recall_b_given_a / denom) if denom > 0 else 0.0
    return {
        "recall_a_given_b": recall_a_given_b, "recall_b_given_a": recall_b_given_a,
        "harmonic_mean": harmonic_mean, "n_events_a": n_a, "n_events_b": n_b,
    }


def temporal_coverage_overlap(
    events_a_s: list[float], events_b_s: list[float], duration_s: float, window_s: float,
) -> float | None:
    """Coarse continuity/coverage check (diagnostic field, NOT part of
    the core distance -- see module docstring): bins [0, duration_s) into
    `window_s`-wide windows and computes the Jaccard overlap of "this
    patch has at least one event in this window" between A and B. Two
    patches that are only coincidentally correlated because each is only
    active in a disjoint portion of the recording will score low here
    even if their few overlapping events happen to line up -- a
    corroborating sanity check reported alongside the similarity, never
    folded into the clustering distance itself (this project's
    convention of keeping distinct evidence lines separately reportable,
    per e.g. motion_evidence.py). None if duration_s/window_s are
    invalid or both patches have zero events."""
    if duration_s <= 0 or window_s <= 0 or (not events_a_s and not events_b_s):
        return None
    n_bins = max(1, int(np.ceil(duration_s / window_s)))
    active_a = np.zeros(n_bins, dtype=bool)
    active_b = np.zeros(n_bins, dtype=bool)
    for t in events_a_s:
        active_a[min(n_bins - 1, int(t / window_s))] = True
    for t in events_b_s:
        active_b[min(n_bins - 1, int(t / window_s))] = True
    union = np.logical_or(active_a, active_b).sum()
    if union == 0:
        return None
    return float(np.logical_and(active_a, active_b).sum() / union)


def robust_pairwise_timing_features(
    events_a_s: list[float],
    events_b_s: list[float],
    period_a_s: float | None,
    period_b_s: float | None,
    waveform_a: np.ndarray,
    waveform_b: np.ndarray,
    sample_interval_s: float,
    coincidence_tolerance_s: float,
    max_lag_samples: int,
    duration_s: float,
    coverage_window_s: float,
) -> dict:
    """One robust graph edge's full feature bundle -- the round-38
    drop-in replacement for `patch_timing_graph.compute_pairwise_timing_
    features`'s coincidence/frequency/correlation features. Built
    entirely from each patch's own already-detected events/waveform; no
    manual region coordinate, no manual beat count. See module docstring
    weakness 3 for why correlation is computed at TWO different lag
    windows here."""
    wide_curve = cross_correlation_curve(waveform_a, waveform_b, max_lag_samples)
    narrow_lag_samples = max(1, int(round(coincidence_tolerance_s / sample_interval_s)))
    narrow_curve = cross_correlation_curve(waveform_a, waveform_b, min(narrow_lag_samples, max_lag_samples))
    match = soft_event_match_scores(events_a_s, events_b_s, coincidence_tolerance_s)
    period_sim = harmonic_aware_period_similarity(period_a_s, period_b_s)
    coverage = temporal_coverage_overlap(events_a_s, events_b_s, duration_s, coverage_window_s)
    return {
        "narrow_lag_correlation": narrow_curve["best_correlation"],
        "narrow_lag_s": narrow_curve["best_lag_samples"] * sample_interval_s if narrow_curve["best_lag_samples"] is not None else None,
        # Full-window best-lag correlation -- DIAGNOSTIC ONLY (see module
        # docstring weakness 3), never fed into robust_pairwise_timing_
        # distance. Reused UNCHANGED shape/semantics from round 35/37.
        "best_lag_samples": wide_curve["best_lag_samples"],
        "best_lag_s": wide_curve["best_lag_samples"] * sample_interval_s if wide_curve["best_lag_samples"] is not None else None,
        "best_correlation": wide_curve["best_correlation"],
        # Reused UNCHANGED from round 35's patch_timing_graph.py -- kept
        # here only so this module's edge dict stays a superset-compatible
        # drop-in for patch_beat_consensus.episode_to_consensus_beat_record
        # (which reads this field for its within-cluster lag-structure QC
        # report); it plays no role in robust_pairwise_timing_distance.
        "median_event_delay_s": median_event_delay_s(events_a_s, events_b_s, coincidence_tolerance_s),
        "soft_event_match": match,
        "harmonic_period_similarity": period_sim,
        "temporal_coverage_overlap": coverage,
    }


def robust_pairwise_timing_distance(edge_features: dict | None) -> float:
    """Combine an edge's ROBUST timing-only features into one distance in
    [0, 1] -- same transparent-mean-of-whichever-are-available style as
    `patch_timing_clustering.pairwise_timing_distance`, but built from
    `soft_event_match.harmonic_mean` (dropout-tolerant),
    `narrow_lag_correlation` (synchrony at THIS video's own coincidence
    timescale, not full-window best-lag correlation -- see module
    docstring weakness 3), and `harmonic_period_similarity.similarity`
    (harmonic-tolerant) instead of round 37's fragile exclusive-match
    fraction, full-window best-lag correlation, and native-only frequency
    comparison. `best_correlation`/`best_lag_s` (full window) and
    `temporal_coverage_overlap` are reported diagnostic fields only (see
    module docstring) and are deliberately NOT included in the distance.
    Spatial adjacency is never a feature here, matching every prior
    round. `edge_features=None` returns 1.0 (max distance), matching
    round 37's convention."""
    if edge_features is None:
        return 1.0
    similarities = []
    match = edge_features.get("soft_event_match") or {}
    if match.get("harmonic_mean") is not None:
        similarities.append(match["harmonic_mean"])
    narrow_correlation = edge_features.get("narrow_lag_correlation")
    if narrow_correlation is not None:
        similarities.append((narrow_correlation + 1.0) / 2.0)
    period_sim = edge_features.get("harmonic_period_similarity") or {}
    if period_sim.get("similarity") is not None:
        similarities.append(period_sim["similarity"])
    if not similarities:
        return 1.0
    return float(1.0 - np.mean(similarities))


def build_robust_patch_timing_graph(
    patch_ids: list[tuple[int, int]],
    waveforms_by_patch: dict[tuple[int, int], np.ndarray],
    pooled_events_by_patch_s: dict[tuple[int, int], list[float]],
    dominant_period_by_patch_s: dict[tuple[int, int], float | None],
    sample_interval_s: float,
    coincidence_tolerance_s: float,
    max_lag_s: float,
    video_duration_s: float,
    coverage_window_s: float,
) -> dict:
    """Full pairwise robust-feature graph over `patch_ids` -- same shape
    (`{"patch_ids", "edges"}`, order-independent edge keys) as
    `patch_timing_graph.build_patch_timing_graph`, so `get_edge` and every
    downstream clustering function can look it up identically; the
    difference is only WHICH features are computed (see module
    docstring). `dominant_period_by_patch_s` is each patch's own pooled
    dominant PERIOD (seconds) -- pass `1.0 / dominant_frequency_hz`,
    never the frequency directly, so `harmonic_aware_period_similarity`
    receives periods on both sides."""
    max_lag_samples = max(1, int(round(max_lag_s / sample_interval_s)))
    edges: dict[tuple, dict] = {}
    ordered = sorted(patch_ids)
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            a, b = ordered[i], ordered[j]
            edges[(a, b)] = robust_pairwise_timing_features(
                pooled_events_by_patch_s.get(a, []), pooled_events_by_patch_s.get(b, []),
                dominant_period_by_patch_s.get(a), dominant_period_by_patch_s.get(b),
                waveforms_by_patch[a], waveforms_by_patch[b],
                sample_interval_s, coincidence_tolerance_s, max_lag_samples,
                video_duration_s, coverage_window_s,
            )
    return {
        "patch_ids": ordered, "edges": edges,
        "coincidence_tolerance_s": coincidence_tolerance_s, "max_lag_s": max_lag_s,
    }


def get_edge(graph: dict, patch_a: tuple[int, int], patch_b: tuple[int, int]) -> dict | None:
    """Order-independent edge lookup -- identical convention to
    `patch_timing_graph.get_edge`."""
    key = (patch_a, patch_b) if patch_a < patch_b else (patch_b, patch_a)
    return graph["edges"].get(key)
