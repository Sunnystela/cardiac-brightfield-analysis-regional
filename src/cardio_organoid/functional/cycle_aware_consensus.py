"""Round 39, Part E: cycle-index-based (not event-union-based) consensus
BPM/IBI/CV.

Round 35/37/38's `patch_beat_consensus.build_consensus_beats_for_cluster`
pools a cluster's member patches' raw events and re-chains them in time
(`beat_episodes.group_events_into_episodes`) -- an improvement over
region-averaging, but it still counts EVERY resulting time-cluster as one
beat, with no explicit model of "these several patches' events, though
several tens of milliseconds apart, are the SAME underlying cycle seen at
different phases." For a rhythm family spanning a real regional
propagation delay (D10: ~0.1s primary->secondary->tertiary, round 38),
that delay can occasionally be larger than the grouping tolerance and
split what should be one beat into two nearby episodes, or -- more
importantly for THIS round's hypothesis -- the resulting per-episode
`spatial_participation_fraction` and timing spread already computed for
those merged episodes are exactly what inflated round 37/38's mega-
cluster BPM if phase-shifted sub-events were, in effect, over-segmented.

This module instead uses the EXPLICIT cycle index already assigned by
`cycle_phase_assignment.assign_events_to_cycles` (from ONE shared
fundamental period P): a "consensus beat" is a cycle index, not a time-
proximity chain, so patches at genuinely different phases (but the SAME
cycle) are correctly pooled into ONE beat by construction, while missed/
dropped events for some patches in a given cycle simply reduce that
cycle's participation fraction -- never invalidate it, never require
imputation.
"""

from __future__ import annotations

import numpy as np


def cycle_participation_summary(
    assignments_by_patch: dict[tuple[int, int], list[dict]],
    n_total_patches: int,
) -> list[dict]:
    """One summary record per distinct cycle index observed anywhere in
    the population: which/how many reliable patches contributed an event
    to it (`participation_fraction` = that count / `n_total_patches`,
    the FULL reliable-patch population size, not just patches that ever
    fire at all -- a cycle only a minority of patches ever reach is
    exactly the signal a `min_participation_fraction` gate downstream is
    for), the cycle's representative time (median across contributing
    patches' own event times), and activation spread (max-min event time
    within the cycle -- same descriptive convention as
    `beat_episodes.BeatEpisode.activation_spread_s`, never called
    conduction velocity)."""
    cycles: dict[int, list[tuple[tuple[int, int], float, float]]] = {}
    for p, recs in assignments_by_patch.items():
        for r in recs:
            cycles.setdefault(r["cycle_index"], []).append((p, r["event_time"], r["phase_fraction"]))

    summaries = []
    for n in sorted(cycles):
        members = cycles[n]
        participating_patches = sorted({p for p, _t, _f in members})
        times = [t for _p, t, _f in members]
        summaries.append({
            "cycle_index": n,
            "n_patches_participating": len(participating_patches),
            "participation_fraction": (len(participating_patches) / n_total_patches) if n_total_patches else None,
            "representative_time_s": float(np.median(times)),
            "activation_spread_s": float(max(times) - min(times)) if len(times) > 1 else 0.0,
            "participating_patches": [list(p) for p in participating_patches],
            "n_events_total": len(members),
        })
    return summaries


def qualifying_cycles(cycle_summaries: list[dict], min_participation_fraction: float) -> list[dict]:
    """Cycles reaching at least `min_participation_fraction` of the
    reliable-patch population -- the SAME config-provided threshold
    (`functional.beat_episodes.global_min_participation_fraction`)
    already used elsewhere in this project for the analogous decision
    (`beat_episodes._provisional_pattern_label`), reused here rather than
    inventing a new one. Cycles below this are NOT deleted from the
    summary list (still available for QC/reporting), just excluded from
    `cycle_aware_consensus_metrics`."""
    return [c for c in cycle_summaries if (c["participation_fraction"] or 0.0) >= min_participation_fraction]


def cycle_aware_consensus_metrics(qualifying: list[dict]) -> dict:
    """Consensus beat count / BPM / median IBI / CV from the QUALIFYING
    cycles' own representative times -- same median-IBI-based BPM
    definition used everywhere else in this project
    (`beat_cycle_metrics.compute_region_cycle_metrics`), applied to cycle
    indices instead of raw or region-averaged event times. Fewer than 2
    qualifying cycles -> BPM/median_ibi/cv are None (nothing to compute a
    rate from), matching this project's convention of reporting
    insufficient data as None rather than a degenerate number."""
    times = sorted(c["representative_time_s"] for c in qualifying)
    n = len(times)
    if n < 2:
        return {
            "consensus_beat_count": n, "bpm": None, "median_ibi_s": None, "cv_ibi": None,
            "mean_participation_fraction": (float(np.mean([c["participation_fraction"] for c in qualifying])) if qualifying else None),
            "mean_activation_spread_s": (float(np.mean([c["activation_spread_s"] for c in qualifying])) if qualifying else None),
        }
    ibis = np.diff(times)
    median_ibi = float(np.median(ibis))
    return {
        "consensus_beat_count": n,
        "bpm": (60.0 / median_ibi) if median_ibi > 0 else None,
        "median_ibi_s": median_ibi,
        "cv_ibi": float(np.std(ibis, ddof=1) / np.mean(ibis)) if n > 2 and np.mean(ibis) > 0 else 0.0,
        "mean_participation_fraction": float(np.mean([c["participation_fraction"] for c in qualifying])),
        "mean_activation_spread_s": float(np.mean([c["activation_spread_s"] for c in qualifying])),
    }


def cycle_aware_consensus_for_group(
    patch_ids: list[tuple[int, int]],
    pooled_events_by_patch: dict[tuple[int, int], list[float]],
    fundamental_period_s: float,
    anchor_time_s: float,
    min_participation_fraction: float,
) -> dict:
    """Convenience end-to-end wrapper for ONE phase group / cluster: cycle
    assignment (`cycle_phase_assignment.assign_events_to_cycles`) ->
    per-cycle participation summary -> qualifying-cycle consensus
    metrics. Kept separate from `cycle_phase_assignment.py` (which
    computes assignments/phase stats for the WHOLE reliable population
    once) so a caller can also build this for arbitrary SUBSETS (e.g. one
    named-region-overlapping phase group) without recomputing
    assignments for patches outside it."""
    from cardio_organoid.functional.cycle_phase_assignment import assign_events_to_cycles

    group_events = {p: pooled_events_by_patch.get(p, []) for p in patch_ids}
    assignments = assign_events_to_cycles(group_events, fundamental_period_s, anchor_time_s)
    summaries = cycle_participation_summary(assignments, n_total_patches=len(patch_ids))
    qualifying = qualifying_cycles(summaries, min_participation_fraction)
    metrics = cycle_aware_consensus_metrics(qualifying)
    return {
        "cycle_summaries": summaries, "qualifying_cycle_summaries": qualifying, "metrics": metrics,
        "n_patches": len(patch_ids), "min_participation_fraction": min_participation_fraction,
    }
