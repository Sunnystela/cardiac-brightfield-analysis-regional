"""Biological-beat reconstruction on top of beat_cycles (round 29).

Manual QC of round 25-28's beat-cycle-based provisional BPM found two
concrete, diagnosed problems that are NOT about beat_events.py's raw
detector (positive/negative event timing already matches manual beat
counts well for early/slit_a1 -- ~17 vs ~17) but about how
beat_cycles.py's positive/negative PAIRING reconstructs those raw events
into cycles:

  1. early/slit_a1: a single local "doublet" motion (a genuine extra
     event, not a detector error -- see round 21/26 manual confirmation)
     makes beat_cycles.py's greedy nearest-neighbor pairing leave SEVERAL
     surrounding cycles unpaired/medium-confidence, not just the doublet
     event itself, because an early greedy match can claim the "wrong"
     (locally-nearest but not biologically-corresponding) partner and
     starve a later event of its true partner. This fragments what is
     otherwise a very regular beat-to-beat rhythm into a high-CV mess.
  2. mature/D10_slit_a1's secondary_region: the ROBUST MEDIAN gap between
     consecutive same-polarity raw events (0.898-0.948s) is almost
     EXACTLY half of primary_region's/tertiary_region's (1.796s, matching
     each other to 3 decimal places) -- a much cleaner, more robust
     signal than the FFT-based dominant-frequency harmonic check (which
     used a different, noisier waveform-preprocessing pathway and gave a
     less clean ~1.4-1.5x ratio). This points to secondary_region's raw
     event stream itself containing close to 2 detected same-polarity
     events per one "biological" cycle-length interval, independent of
     any positive/negative pairing artifact.

This module does NOT change beat_events.py's detection thresholds or
beat_cycles.py's positive/negative pairing algorithm -- it adds a
READ-ONLY, ADDITIONAL layer on top of already-built BeatCycle records
that merges cycles (and unpaired singleton cycles) which sit unusually
close together, relative to the region's OWN robust typical spacing,
into one "biological beat". Every merge is fully traceable (which
beat_cycle_ids were merged, into which biological_beat_id) and no raw
event or beat_cycle is ever deleted -- member_cycles are kept verbatim
inside each BiologicalBeat.

Reference-period estimation is DELIBERATELY the MEDIAN of consecutive
same-polarity raw-event gaps (not the FFT-based dominant_frequency_hz
used elsewhere in this project) -- a median is robust to a MINORITY of
short "doublet" gaps pulling a mean or a spectral peak toward a harmonic,
as long as most gaps reflect the true typical spacing (true for
early/slit_a1: only 2 of 16 positive-to-positive gaps are short). This
median is NOT reliable when a MAJORITY of a region's own gaps are
already the short/doubled kind (true for D10's secondary_region, where
the "doubling" is pervasive, not a minority) -- callers needing a
trustworthy reference period for such a region should prefer another,
independently regular region in the same video (e.g. D10's primary_region
or tertiary_region) rather than the contaminated region's own median.
This module does not choose that fallback for the caller; it only
implements the merge given whatever reference_period_s is passed in, and
callers must be explicit about which source they used (see
scripts/biological_beat_reconstruction.py).

No fixed/absolute refractory time is hardcoded anywhere in this module --
every merge window is `merge_period_fraction * reference_period_s`, and
using this module is a no-op (nothing merged) when no valid
reference_period_s can be estimated.
"""

from __future__ import annotations

import dataclasses

import numpy as np


def median_consecutive_gap(times: list[float]) -> float | None:
    """Median of consecutive-event gaps in a single, already-sorted-or-not
    list of event times. Returns None if fewer than 2 gaps exist (i.e.
    fewer than 3 times) -- a median of 0 or 1 gap is not a robust
    estimate of "typical" spacing."""
    if len(times) < 3:
        return None
    ordered = sorted(times)
    gaps = np.diff(ordered)
    return float(np.median(gaps))


def robust_reference_period_s(positive_times: list[float], negative_times: list[float]) -> float | None:
    """Self-relative "typical biological period" estimate for one region:
    the median of the positive-only and negative-only robust median gaps
    (see median_consecutive_gap). Returns None if neither polarity has
    enough events for a robust estimate."""
    candidates = [g for g in (median_consecutive_gap(positive_times), median_consecutive_gap(negative_times)) if g is not None]
    if not candidates:
        return None
    return float(np.median(candidates))


@dataclasses.dataclass
class BiologicalBeat:
    biological_beat_id: str
    region_id: str
    member_cycle_ids: list[str]
    member_cycles: list[dict]
    reference_time: float
    first_cycle_time: float
    last_cycle_time: float
    spread_s: float
    n_member_cycles: int
    # "high" if any member cycle is high-confidence, else "medium" if any
    # member is medium-confidence, else "unpaired" -- see module
    # docstring: a biological beat anchored by even one well-paired cycle
    # is still usable even if a lower-quality neighbor was merged into it.
    confidence: str
    flags: list[str]
    reference_period_s_used: float | None
    merge_period_fraction_used: float

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


_CONFIDENCE_RANK = {"high": 2, "medium": 1, "unpaired": 0}


def _best_confidence(cycles: list[dict]) -> str:
    return max((c["pairing_confidence"] for c in cycles), key=lambda c: _CONFIDENCE_RANK.get(c, -1))


def merge_cycles_into_biological_beats(
    cycles: list[dict],
    reference_period_s: float | None,
    merge_period_fraction: float,
) -> list[BiologicalBeat]:
    """Greedy chain-merge (same adjacency style as
    beat_episodes.group_events_into_episodes): consecutive beat_cycles
    (sorted by cycle_reference_time) are merged into one biological beat
    if the gap to the PREVIOUS beat already in the group is less than
    `merge_period_fraction * reference_period_s`. If
    `reference_period_s` is None (no robust estimate available), every
    cycle becomes its own standalone biological beat (no merging) --
    never falls back to a fixed/absolute time window.
    """
    if not cycles:
        return []
    ordered = sorted(cycles, key=lambda c: c["cycle_reference_time"])
    merge_window_s = (merge_period_fraction * reference_period_s) if reference_period_s else None

    groups: list[list[dict]] = [[ordered[0]]]
    for c in ordered[1:]:
        if merge_window_s is not None and c["cycle_reference_time"] - groups[-1][-1]["cycle_reference_time"] <= merge_window_s:
            groups[-1].append(c)
        else:
            groups.append([c])

    region_id = ordered[0]["region_id"]
    beats = []
    for i, group in enumerate(groups):
        times = [c["cycle_reference_time"] for c in group]
        flags = ["merged_from_multiple_cycles"] if len(group) > 1 else []
        beats.append(BiologicalBeat(
            biological_beat_id=f"{region_id}_biobeat_{i:03d}",
            region_id=region_id,
            member_cycle_ids=[c["beat_cycle_id"] for c in group],
            member_cycles=group,
            reference_time=float(np.mean(times)),
            first_cycle_time=min(times),
            last_cycle_time=max(times),
            spread_s=max(times) - min(times),
            n_member_cycles=len(group),
            confidence=_best_confidence(group),
            flags=flags,
            reference_period_s_used=reference_period_s,
            merge_period_fraction_used=merge_period_fraction,
        ))
    return beats


def _interval_stats(times: list[float]) -> dict:
    if len(times) < 2:
        return {"n_intervals": 0, "mean_ibi_s": None, "median_ibi_s": None, "sd_ibi_s": None, "cv_ibi": None}
    ordered = sorted(times)
    intervals = np.diff(ordered)
    mean = float(np.mean(intervals))
    median = float(np.median(intervals))
    sd = float(np.std(intervals, ddof=1)) if len(intervals) > 1 else 0.0
    cv = (sd / mean) if mean > 0 else None
    return {"n_intervals": int(len(intervals)), "mean_ibi_s": mean, "median_ibi_s": median, "sd_ibi_s": sd, "cv_ibi": cv}


def _provisional_bpm(ibi_s: float | None) -> float | None:
    if not ibi_s or ibi_s <= 0:
        return None
    return 60.0 / ibi_s


def compute_biological_beat_metrics(beats: list[BiologicalBeat]) -> dict:
    """Three-tier reporting: "high_confidence_only" / "high_and_medium"
    match beat_cycle_metrics.compute_region_cycle_metrics's convention at
    the biological-beat level. A third "all_beats" tier additionally
    INCLUDES standalone "unpaired" biological beats (ones beat_cycles.py
    could not pair AND that did not merge with any neighbor) as one beat
    occurrence each -- appropriate once the merge step above has already
    absorbed genuinely-adjacent fragments, since a standalone unpaired
    cycle spaced roughly one typical period from its neighbors is more
    likely a real beat with only one polarity confidently detected than a
    spurious extra count. All three tiers are reported side by side --
    "all_beats" is not asserted as more correct than the other two."""
    tiers = {}
    for tier_name, allowed in (
        ("high_confidence_only", {"high"}), ("high_and_medium", {"high", "medium"}),
        ("all_beats", {"high", "medium", "unpaired"}),
    ):
        subset = [b for b in beats if b.confidence in allowed]
        ref_times = [b.reference_time for b in subset]
        stats = _interval_stats(ref_times)
        tiers[tier_name] = {
            "n_biological_beats": len(subset),
            "interval_stats": stats,
            "provisional_bpm_from_median_ibi": _provisional_bpm(stats["median_ibi_s"]),
            "provisional_bpm_from_mean_ibi_reference_only": _provisional_bpm(stats["mean_ibi_s"]),
        }
    return {
        "n_biological_beats_total": len(beats),
        "n_merged_beats": sum(1 for b in beats if b.n_member_cycles > 1),
        "n_member_cycles_total": sum(b.n_member_cycles for b in beats),
        "tiers": tiers,
        "per_beat": [b.to_dict() for b in beats],
    }
