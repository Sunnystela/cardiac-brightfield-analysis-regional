"""Beat-cycle-based provisional temporal functional metrics (round 25).

Manual QC of the round-21/22 beat-episode/beat-cycle reconstruction found
that region-level beat counting should use the PAIRED beat_cycle
representation (cardio_organoid.functional.beat_cycles) as the basic
unit, not raw positive_events/negative_events counted independently or
summed together. This module computes provisional inter-beat-interval
(IBI) and beat-rate (BPM) metrics FROM ALREADY-BUILT BeatCycle records
(read from results/beat_event_episodes*/.../episodes.json) -- it does
NOT re-run detection, re-tune any threshold, or change beat_cycles.py's
pairing algorithm in any way. This is purely a downstream, read-only
aggregation layer.

Explicitly NOT decided here:
  - which polarity (positive/negative) is contraction vs relaxation --
    "positive_amplitude"/"negative_amplitude"/"peak_to_trough_amplitude"
    are used, never "contraction_amplitude".
  - a single collapsed whole-organoid BPM for multi-region videos --
    every metric here is computed PER REGION; a multi-region video's
    regions are never averaged/collapsed into one number.
  - arrhythmia classification, conduction velocity, or any other final
    biological conclusion.
  - "unpaired" beat-cycles (a positive or negative event with no partner
    within the pairing window -- see beat_cycles.py) are EXCLUDED from
    every interval/BPM/amplitude-summary computation below (they have
    no cycle_reference_time built from two matched excursions), but are
    still counted (n_cycles_unpaired) and never silently dropped from
    the input entirely -- the caller's full `cycles` list, unpaired
    included, is expected to still be persisted upstream.

Every metric is reported per PAIRING-CONFIDENCE TIER
("high_confidence_only" vs "high_and_medium") side by side -- never
collapsed into one number -- so a human can see how sensitive a given
region's provisional BPM/IBI is to including medium-confidence
(ambiguous-alternation or doublet-involved) cycles.
"""

from __future__ import annotations

import numpy as np

# Which pairing_confidence values belong to each reporting tier.
# "unpaired" is never included in either tier.
CONFIDENCE_TIERS: dict[str, set[str]] = {
    "high_confidence_only": {"high"},
    "high_and_medium": {"high", "medium"},
}


def _interval_stats(times: list[float]) -> dict:
    """Consecutive-interval (IBI) descriptive stats over `times` (assumed
    already representing one region+tier's ordered beat-cycle reference
    times). Fewer than 2 times -> no interval can be formed; all fields
    None/0."""
    if len(times) < 2:
        return {
            "n_intervals": 0, "mean_ibi_s": None, "median_ibi_s": None,
            "sd_ibi_s": None, "cv_ibi": None,
        }
    ordered = sorted(times)
    intervals = np.diff(ordered)
    mean = float(np.mean(intervals))
    median = float(np.median(intervals))
    sd = float(np.std(intervals, ddof=1)) if len(intervals) > 1 else 0.0
    cv = (sd / mean) if mean > 0 else None
    return {
        "n_intervals": int(len(intervals)), "mean_ibi_s": mean, "median_ibi_s": median,
        "sd_ibi_s": sd, "cv_ibi": cv,
    }


def _provisional_bpm(ibi_s: float | None) -> float | None:
    """BPM = 60 / inter-beat-interval (seconds). Uses whatever timebase
    (file-recorded fps -> sample_interval_s -> event_time seconds)
    already went into the beat-cycle's cycle_reference_time upstream --
    no fps/timebase value is re-derived or assumed here."""
    if not ibi_s or ibi_s <= 0:
        return None
    return 60.0 / ibi_s


def _amplitude_for(event_time: float | None, amplitude_by_time: dict[float, float]) -> float | None:
    if event_time is None:
        return None
    return amplitude_by_time.get(round(event_time, 6))


def compute_region_cycle_metrics(
    cycles: list[dict],
    positive_amplitude_by_time: dict[float, float],
    negative_amplitude_by_time: dict[float, float],
) -> dict:
    """Computes provisional temporal metrics for ONE region from its
    already-built beat_cycle dicts (as saved in episodes.json's
    "beat_cycles"[region] list -- each with beat_cycle_id,
    pairing_confidence, cycle_reference_time, positive_event_time,
    negative_event_time, polarity_interval). `positive_amplitude_by_time`
    / `negative_amplitude_by_time` map a rounded (6-decimal) event_time
    to that event's `amplitude` (from the region's raw positive/negative
    RegionEvent records) -- amplitude is not stored on BeatCycle itself,
    so it is looked up here rather than by modifying beat_cycles.py.

    Returns per-cycle records (ALL cycles, unpaired included, for full
    traceability) plus, for each CONFIDENCE_TIERS entry, IBI/BPM/
    amplitude-summary/cycle-timing computed with unpaired cycles (and,
    for the "high_confidence_only" tier, medium-confidence ones too)
    excluded.
    """
    ordered_all = sorted(cycles, key=lambda c: c["cycle_reference_time"])

    per_cycle = []
    for c in ordered_all:
        pos_amp = _amplitude_for(c["positive_event_time"], positive_amplitude_by_time)
        neg_amp = _amplitude_for(c["negative_event_time"], negative_amplitude_by_time)
        peak_to_trough = (pos_amp - neg_amp) if (pos_amp is not None and neg_amp is not None) else None
        per_cycle.append({
            "beat_cycle_id": c["beat_cycle_id"],
            "pairing_confidence": c["pairing_confidence"],
            "cycle_reference_time": c["cycle_reference_time"],
            "positive_event_time": c["positive_event_time"],
            "negative_event_time": c["negative_event_time"],
            "positive_amplitude": pos_amp,
            "negative_amplitude": neg_amp,
            "peak_to_trough_amplitude": peak_to_trough,
            "positive_to_negative_interval_s": c.get("polarity_interval"),
        })

    tiers = {}
    for tier_name, allowed_confidences in CONFIDENCE_TIERS.items():
        subset = [c for c in ordered_all if c["pairing_confidence"] in allowed_confidences]
        ref_times = [c["cycle_reference_time"] for c in subset]
        interval_stats = _interval_stats(ref_times)

        cycle_timing = []
        for i, c in enumerate(subset):
            neg_t = c["negative_event_time"]
            next_pos_t = subset[i + 1]["positive_event_time"] if i + 1 < len(subset) else None
            interval = (next_pos_t - neg_t) if (neg_t is not None and next_pos_t is not None) else None
            cycle_timing.append({
                "beat_cycle_id": c["beat_cycle_id"],
                "negative_to_next_positive_interval_s": interval,
            })

        pos_amps, neg_amps, ptt_amps = [], [], []
        for c in subset:
            pa = _amplitude_for(c["positive_event_time"], positive_amplitude_by_time)
            na = _amplitude_for(c["negative_event_time"], negative_amplitude_by_time)
            if pa is not None:
                pos_amps.append(pa)
            if na is not None:
                neg_amps.append(na)
            if pa is not None and na is not None:
                ptt_amps.append(pa - na)

        tiers[tier_name] = {
            "beat_cycle_count": len(subset),
            "interval_stats": interval_stats,
            "provisional_bpm_from_median_ibi": _provisional_bpm(interval_stats["median_ibi_s"]),
            # Reference-only, per project decision -- median is primary.
            "provisional_bpm_from_mean_ibi_reference_only": _provisional_bpm(interval_stats["mean_ibi_s"]),
            "amplitude_summary": {
                "mean_positive_amplitude": float(np.mean(pos_amps)) if pos_amps else None,
                "mean_negative_amplitude": float(np.mean(neg_amps)) if neg_amps else None,
                "mean_peak_to_trough_amplitude": float(np.mean(ptt_amps)) if ptt_amps else None,
            },
            "cycle_timing_negative_to_next_positive": cycle_timing,
        }

    return {
        "n_cycles_total": len(ordered_all),
        "n_cycles_unpaired": sum(1 for c in ordered_all if c["pairing_confidence"] == "unpaired"),
        "tiers": tiers,
        "per_cycle": per_cycle,
    }


def compare_region_bpms(region_metrics: dict[str, dict]) -> list[dict]:
    """Descriptive, NON-decisive pairwise comparison of regions' provisional
    BPMs within one multi-region video -- reports the raw values, absolute
    difference, and ratio for EACH confidence tier side by side. Does NOT
    classify region pairs as "same" or "different" (that would require a
    new threshold this module deliberately does not introduce)."""
    names = list(region_metrics)
    comparisons = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            entry = {"region_a": a, "region_b": b, "tiers": {}}
            for tier_name in CONFIDENCE_TIERS:
                bpm_a = region_metrics[a]["tiers"][tier_name]["provisional_bpm_from_median_ibi"]
                bpm_b = region_metrics[b]["tiers"][tier_name]["provisional_bpm_from_median_ibi"]
                entry["tiers"][tier_name] = {
                    "bpm_a": bpm_a, "bpm_b": bpm_b,
                    "abs_diff_bpm": (abs(bpm_a - bpm_b) if (bpm_a is not None and bpm_b is not None) else None),
                    "ratio_a_over_b": (bpm_a / bpm_b if (bpm_a is not None and bpm_b) else None),
                }
            comparisons.append(entry)
    return comparisons
