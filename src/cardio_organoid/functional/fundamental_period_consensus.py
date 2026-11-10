"""Round 39: population-level fundamental-period estimation, explicitly
harmonic-ambiguity-aware.

Round 38 found that D10/slit/a1's secondary_region and most of
tertiary_region are robustly merged into one timing-similarity mega-
cluster under THREE independent similarity formulations, while a
narrow-lag cross-correlation diagnostic showed a clean, additive
activation-order lag chain (primary -> secondary ~0.10s -> tertiary
~0.10s, primary -> tertiary ~0.20s) matching the manual reference's
stated `activation_order`. This is the signature of ONE shared rhythm
propagating through spatially distinct regions with a consistent phase
delay, NOT three (or more) independently-paced oscillators -- i.e. the
region-average "66.8 BPM" and round 37/38's mega-cluster "49.1 BPM" may
both be artifacts of treating phase-shifted copies of the SAME cycle as
separate beats.

This module estimates ONE fundamental period P for such a rhythm family
from reliable patches' own event trains -- never from the region-average
waveform's peak count (round 33/35's own diagnosis: peak-picking on an
averaged, phase-mixed waveform is exactly the mechanism suspected of
inflating the apparent rate) -- and explicitly resolves 1x/2x/0.5x
harmonic disagreement BETWEEN patches (not within a single patch's own
estimate, which `harmonic_aware_timing.py` already handles pairwise) via
a maximum-similarity spanning tree, so the final population estimate is
not silently dominated by whichever patches happen to be harmonic-locked
the same way.

No manual BPM or manual region label is used as an input anywhere in
this module -- both are compared against the OUTPUT only, downstream, by
the caller.
"""

from __future__ import annotations

import numpy as np

from cardio_organoid.functional.harmonic_aware_timing import harmonic_aware_period_similarity

# Broad, physiologically-generic sanity bounds for the autocorrelation
# period search (0.2-3.0s = 20-300 BPM) -- NOT derived from or tuned to
# this project's specific videos/manual counts, just wide enough to
# exclude trivial zero-lag/DC-drift artifacts on one side and numerically
# unstable long lags on the other.
DEFAULT_AUTOCORR_SEARCH_BOUNDS_S = (0.2, 3.0)


def median_ibi_from_events(event_times_s: list[float]) -> float | None:
    """Median inter-event interval from ONE polarity's own (unchanged)
    detected events -- each successive same-polarity event is
    approximately one full cycle apart, so this is a direct period
    estimate independent of any spectral peak-pick. None if fewer than 2
    events."""
    if len(event_times_s) < 2:
        return None
    ibis = np.diff(sorted(event_times_s))
    return float(np.median(ibis))


def autocorrelation_period_estimate(
    waveform: np.ndarray, sample_interval_s: float,
    search_bounds_s: tuple[float, float] = DEFAULT_AUTOCORR_SEARCH_BOUNDS_S,
) -> float | None:
    """First-principles period estimate from the patch's own waveform
    autocorrelation (independent of `beat_events.py`'s event detector and
    of `dominant_spectral_peak`'s FFT peak-pick -- a third, differently-
    biased candidate source). Returns the lag (in seconds, within
    `search_bounds_s`) of the highest autocorrelation value, or None if
    the waveform is too short for the search window or is constant."""
    x = np.asarray(waveform, dtype=float)
    x = x - np.mean(x)
    if np.allclose(x, 0.0):
        return None
    n = len(x)
    full_ac = np.correlate(x, x, mode="full")
    ac = full_ac[n - 1:]  # lag >= 0 only
    min_lag = max(1, int(round(search_bounds_s[0] / sample_interval_s)))
    max_lag = min(n - 1, int(round(search_bounds_s[1] / sample_interval_s)))
    if max_lag <= min_lag:
        return None
    window = ac[min_lag:max_lag + 1]
    peak_offset = int(np.argmax(window))
    return (min_lag + peak_offset) * sample_interval_s


def patch_own_period_estimate(
    positive_event_times_s: list[float],
    negative_event_times_s: list[float],
    dominant_frequency_hz: float | None,
    waveform: np.ndarray | None,
    sample_interval_s: float | None,
) -> dict:
    """Combines every available per-patch period-candidate source (own
    positive-event median IBI, own negative-event median IBI, own
    spectral dominant period, own waveform autocorrelation) into one
    robust per-patch estimate (median of available candidates -- a single
    candidate source being harmonic-confused does not, by itself, decide
    the patch's estimate). Returns {"own_period_estimate_s", "candidates":
    {source: value_or_None}} -- every candidate kept for transparency,
    never silently discarded."""
    candidates = {
        "median_ibi_positive_s": median_ibi_from_events(positive_event_times_s),
        "median_ibi_negative_s": median_ibi_from_events(negative_event_times_s),
        "spectral_period_s": (1.0 / dominant_frequency_hz) if dominant_frequency_hz and dominant_frequency_hz > 0 else None,
        "autocorrelation_period_s": (
            autocorrelation_period_estimate(waveform, sample_interval_s) if waveform is not None and sample_interval_s else None
        ),
    }
    values = [v for v in candidates.values() if v is not None]
    return {"own_period_estimate_s": float(np.median(values)) if values else None, "candidates": candidates}


def resolve_harmonic_scales(
    patch_ids: list[tuple[int, int]],
    own_period_by_patch: dict[tuple[int, int], float | None],
    weight_by_patch: dict[tuple[int, int], float],
) -> dict:
    """Reconciles every reliable patch's own (possibly 1x/2x/0.5x
    harmonic-ambiguous) period estimate onto ONE common rhythm-family
    scale, via a maximum-similarity spanning tree over
    `harmonic_aware_period_similarity` edges (Prim's algorithm -- cheap
    at this project's per-video patch counts).

    The root is the highest-`weight_by_patch` patch (round 36's
    reliability confidence, e.g.) -- its own raw period defines the
    family's scale (cumulative multiplier 1.0). Every other patch is then
    attached, one at a time, via whichever ALREADY-ATTACHED patch gives
    it the single best harmonic-aware match found so far anywhere in the
    remaining graph (so the most confident agreements propagate first,
    and a noisy/ambiguous patch is only ever anchored through its best
    available evidence, never forced through a fixed pairing order).

    For an edge (already-attached parent `u`, new patch `v`) with
    `harmonic_aware_period_similarity(own_period[u], own_period[v])` ->
    `best_ratio` r (meaning own_period[u] best matches r * own_period[v]),
    v's cumulative scale is `scale[u] * r` and its resolved period is
    `own_period[v] * scale[u] * r` -- this corrects v's harmonic
    ambiguity relative to the tree's root WITHOUT forcing v's resolved
    value to exactly equal u's (real inter-patch period variability is
    preserved; only the discrete 1x/2x/0.5x multiple is corrected).

    Returns {"resolved_period_by_patch", "applied_scale_by_patch"
    (cumulative multiplier relative to each patch's OWN raw estimate --
    1.0 means no harmonic correction was needed), "root_patch"}. Patches
    with no valid own_period are omitted entirely (nothing to resolve)."""
    valid_ids = [p for p in patch_ids if own_period_by_patch.get(p)]
    if not valid_ids:
        return {"resolved_period_by_patch": {}, "applied_scale_by_patch": {}, "root_patch": None}

    root = max(valid_ids, key=lambda p: weight_by_patch.get(p, 0.0))
    scale = {root: 1.0}
    resolved = {root: own_period_by_patch[root]}
    in_tree = {root}
    remaining = set(valid_ids) - in_tree

    while remaining:
        best = None  # (similarity, parent, child, ratio_parent_over_child)
        for u in in_tree:
            for v in remaining:
                r = harmonic_aware_period_similarity(own_period_by_patch[u], own_period_by_patch[v])
                if r["similarity"] is None:
                    continue
                if best is None or r["similarity"] > best[0]:
                    best = (r["similarity"], u, v, r["best_ratio"])
        if best is None:
            # No comparable edge to any remaining patch (should not
            # happen for valid positive periods) -- attach with no
            # correction rather than silently dropping the patch.
            v = next(iter(remaining))
            scale[v], resolved[v] = 1.0, own_period_by_patch[v]
            in_tree.add(v)
            remaining.discard(v)
            continue
        _similarity, u, v, ratio = best
        scale[v] = scale[u] * ratio
        resolved[v] = own_period_by_patch[v] * scale[v]
        in_tree.add(v)
        remaining.discard(v)

    return {"resolved_period_by_patch": resolved, "applied_scale_by_patch": scale, "root_patch": list(root)}


def weighted_median(values_and_weights: list[tuple[float, float]]) -> float | None:
    """Standard weighted median: sort by value, find the value at which
    cumulative weight first reaches half the total weight. Non-positive
    weights are treated as 0 (excluded from the cumulative sum, kept in
    the input for transparency). None if there is no positive-weight
    value."""
    pairs = [(v, max(w, 0.0)) for v, w in values_and_weights if v is not None]
    total_weight = sum(w for _, w in pairs)
    if total_weight <= 0:
        return None
    pairs.sort(key=lambda vw: vw[0])
    cumulative = 0.0
    half = total_weight / 2.0
    for v, w in pairs:
        cumulative += w
        if cumulative >= half:
            return float(v)
    return float(pairs[-1][0])


def estimate_fundamental_period(
    patch_ids: list[tuple[int, int]],
    positive_events_by_patch: dict[tuple[int, int], list[float]],
    negative_events_by_patch: dict[tuple[int, int], list[float]],
    dominant_frequency_by_patch: dict[tuple[int, int], float | None],
    waveforms_by_patch: dict[tuple[int, int], np.ndarray],
    sample_interval_s: float,
    weight_by_patch: dict[tuple[int, int], float],
) -> dict:
    """End-to-end Part-A pipeline: per-patch multi-candidate own-period
    estimate -> population-wide harmonic-scale reconciliation ->
    confidence-weighted median -> ONE fundamental period P. Every
    intermediate value is returned for transparency/QC, never hidden."""
    own_period_by_patch, candidate_detail_by_patch = {}, {}
    for p in patch_ids:
        detail = patch_own_period_estimate(
            positive_events_by_patch.get(p, []), negative_events_by_patch.get(p, []),
            dominant_frequency_by_patch.get(p), waveforms_by_patch.get(p), sample_interval_s,
        )
        candidate_detail_by_patch[p] = detail
        own_period_by_patch[p] = detail["own_period_estimate_s"]

    harmonic = resolve_harmonic_scales(patch_ids, own_period_by_patch, weight_by_patch)
    resolved = harmonic["resolved_period_by_patch"]
    fundamental_period_s = weighted_median([(resolved[p], weight_by_patch.get(p, 1.0)) for p in resolved])

    n_harmonic_corrected = sum(1 for s in harmonic["applied_scale_by_patch"].values() if abs(s - 1.0) > 1e-6)
    return {
        "fundamental_period_s": fundamental_period_s,
        "fundamental_bpm": (60.0 / fundamental_period_s) if fundamental_period_s else None,
        "own_period_by_patch": own_period_by_patch,
        "resolved_period_by_patch": resolved,
        "applied_harmonic_scale_by_patch": harmonic["applied_scale_by_patch"],
        "root_patch": harmonic["root_patch"],
        "n_patches_harmonic_corrected": n_harmonic_corrected,
        "n_patches_total": len(resolved),
        "candidate_detail_by_patch": candidate_detail_by_patch,
    }
