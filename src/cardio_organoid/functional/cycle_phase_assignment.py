"""Round 39: cycle-index assignment and per-patch phase estimation, given
a population fundamental period P (`fundamental_period_consensus.py`).

Instead of counting every patch's own detected event as an independent
beat (round 35/37/38's `beat_episodes.group_events_into_episodes`, which
chains events across patches purely by absolute-time proximity -- exactly
how round 37/38's mega-cluster's 23 "beats" were built), this module
assigns every reliable patch's own (unchanged, already-detected) events
to an integer CYCLE INDEX of the shared rhythm, then describes each
patch's PHASE within that cycle (median offset, dispersion, stability,
participation) -- so events from primary/secondary/tertiary that belong
to the SAME underlying beat, merely phase-shifted, land in the SAME
cycle index instead of being counted as separate beats.

Circular statistics (phase wraps at the cycle boundary) are used
throughout for phase aggregation -- a plain arithmetic mean/median of
phase fractions would be wrong for events that straddle the +-0.5 wrap
point.
"""

from __future__ import annotations

import numpy as np


def choose_anchor_time_s(pooled_events_by_patch: dict[tuple[int, int], list[float]]) -> float:
    """Arbitrary time origin for cycle-index counting -- the earliest
    pooled event across every reliable patch. The exact choice does not
    matter for any RELATIVE phase comparison between patches (shifting
    the anchor adds the same constant to every patch's phase within a
    cycle, which cancels out in every pairwise/relative statistic this
    module computes); it only fixes cycle index 0's absolute position."""
    all_times = [t for times in pooled_events_by_patch.values() for t in times]
    return float(min(all_times)) if all_times else 0.0


def assign_events_to_cycles(
    pooled_events_by_patch: dict[tuple[int, int], list[float]],
    fundamental_period_s: float,
    anchor_time_s: float,
) -> dict[tuple[int, int], list[dict]]:
    """For every reliable patch's own pooled (positive+negative, already-
    detected, UNCHANGED) event times, assigns each event to the nearest
    integer multiple of `fundamental_period_s` from `anchor_time_s`
    (`cycle_index = round((t - anchor) / P)`), and records how far off
    that nearest multiple the event actually landed
    (`phase_offset_s`/`phase_fraction = phase_offset_s / P`, in
    (-0.5, 0.5]). Positive and negative events are assigned via the SAME
    period P but are not otherwise distinguished here -- a full cycle's
    positive and negative excursions land at their own, generally
    different, phase offsets within the SAME cycle index (their offset
    magnitude, roughly half a period apart, is well within the +-P/2
    rounding window), which is exactly the per-patch PHASE this module
    characterizes next; downstream cycle-level aggregation
    (`cycle_aware_consensus.py`) pools both polarities per cycle, same as
    `event_time_coincidence_fraction`'s established pooled-events
    convention elsewhere in this project.

    Missing/dropped events for a patch in a given cycle are handled
    implicitly: a patch simply contributes no record for a cycle it
    didn't detect an event in -- never imputed, never required."""
    if fundamental_period_s is None or fundamental_period_s <= 0:
        return {p: [] for p in pooled_events_by_patch}
    assignments = {}
    for p, times in pooled_events_by_patch.items():
        records = []
        for t in times:
            n = int(round((t - anchor_time_s) / fundamental_period_s))
            offset_s = t - (anchor_time_s + n * fundamental_period_s)
            records.append({
                "event_time": t, "cycle_index": n,
                "phase_offset_s": offset_s, "phase_fraction": offset_s / fundamental_period_s,
            })
        assignments[p] = records
    return assignments


def circular_phase_mean_and_dispersion(phase_fractions: list[float]) -> tuple[float | None, float | None]:
    """Circular mean phase fraction (in (-0.5, 0.5]) and circular
    dispersion (1 - mean resultant length, in [0, 1]; 0 = perfectly
    concentrated, 1 = uniformly spread around the whole cycle) -- the
    standard circular-statistics analog of a mean/variance, needed
    because phase wraps (an event at +0.49 and one at -0.49 are actually
    close together, not far apart). None, None if the input is empty."""
    if not phase_fractions:
        return None, None
    angles = 2.0 * np.pi * np.asarray(phase_fractions, dtype=float)
    c, s = float(np.mean(np.cos(angles))), float(np.mean(np.sin(angles)))
    resultant_length = float(np.hypot(c, s))
    mean_angle = float(np.arctan2(s, c))
    return mean_angle / (2.0 * np.pi), 1.0 - resultant_length


def per_patch_phase_stats(
    assignments_by_patch: dict[tuple[int, int], list[dict]],
) -> dict[tuple[int, int], dict]:
    """Part C: per-patch median (circular) phase offset, phase
    dispersion, participation fraction (this patch's own distinct cycle
    count divided by the TOTAL number of distinct cycles observed
    ANYWHERE in the population -- self-relative, never an absolute count
    or a manual-region-derived denominator), and a simple, transparently-
    labeled `phase_stability` (`1 - dispersion`, PROVISIONAL, for ranking/
    QC only -- same posture as `reliable_hotspot_candidates.py`'s
    provisional composites)."""
    all_cycle_indices = {r["cycle_index"] for recs in assignments_by_patch.values() for r in recs}
    n_total_cycles = len(all_cycle_indices)

    stats = {}
    for p, recs in assignments_by_patch.items():
        fracs = [r["phase_fraction"] for r in recs]
        mean_frac, dispersion = circular_phase_mean_and_dispersion(fracs)
        n_cycles_participated = len({r["cycle_index"] for r in recs})
        stats[p] = {
            "median_phase_fraction": mean_frac,
            "phase_dispersion": dispersion,
            "phase_stability": (1.0 - dispersion) if dispersion is not None else None,
            "n_events": len(recs),
            "n_cycles_participated": n_cycles_participated,
            "participation_fraction": (n_cycles_participated / n_total_cycles) if n_total_cycles else None,
        }
    return stats


def signed_phase_lag_seconds(phase_a: float | None, phase_b: float | None, fundamental_period_s: float) -> float | None:
    """Signed lag (seconds) of patch B relative to patch A, wrapped to
    the shortest circular direction ((-P/2, P/2]) -- positive means B's
    phase falls AFTER A's within the shared cycle. None-safe."""
    if phase_a is None or phase_b is None or not fundamental_period_s:
        return None
    diff = phase_b - phase_a
    wrapped = ((diff + 0.5) % 1.0) - 0.5
    return wrapped * fundamental_period_s


def signed_phase_lag_matrix(
    patch_ids: list[tuple[int, int]], phase_stats: dict[tuple[int, int], dict], fundamental_period_s: float,
) -> np.ndarray:
    """(n, n) signed lag matrix (seconds) -- `matrix[i, j]` is patch j's
    lag relative to patch i (see `signed_phase_lag_seconds`);
    antisymmetric by construction, diagonal 0, NaN where either patch has
    no phase estimate."""
    n = len(patch_ids)
    matrix = np.full((n, n), np.nan)
    for i in range(n):
        matrix[i, i] = 0.0
        for j in range(i + 1, n):
            a, b = patch_ids[i], patch_ids[j]
            lag = signed_phase_lag_seconds(
                phase_stats.get(a, {}).get("median_phase_fraction"),
                phase_stats.get(b, {}).get("median_phase_fraction"),
                fundamental_period_s,
            )
            if lag is not None:
                matrix[i, j] = lag
                matrix[j, i] = -lag
    return matrix
