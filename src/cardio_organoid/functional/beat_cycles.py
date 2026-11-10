"""Positive/negative beat-cycle pairing (round 21).

Manual cross-check of round 20's time-synchronized QC videos against the
raw videos found that treating `positive_events` and `negative_events`
(see beat_events.detect_beat_events_both_polarities) as two independent
beat counts over-counts: one real beating oscillation naturally produces
one excursion in each signed direction, detected separately because this
project has not yet decided which direction is contraction vs
relaxation. This module adds a region-level PAIRING layer on top of the
UNCHANGED raw positive/negative event detection (no threshold retuning,
no change to beat_events.py or beat_episodes.py's episode grouping) that
pairs each positive event with its nearest still-unused negative event
within a per-region, self-relative time window, into a "beat cycle".

Explicitly NOT decided here:
  - which polarity is contraction vs relaxation -- no biological label
    anywhere in this module or its output.
  - a single final "beat count" -- both the raw positive/negative event
    counts and the beat-cycle count are meant to be reported side by
    side by the caller (see beat_cycle_counts), never collapsed into one
    number by simple positive_count + negative_count summation.
  - whether an unpaired or doublet-involved cycle is a detector artifact
    or genuine local repetitive/compound motion (e.g. the "double-peak"
    local doublet observed in early/slit/a1, or the tremor-like
    repetitive motion observed in some mature-stage regions) -- flagged
    (`ambiguous_cycle_pairing`, `possible_compound_local_motion`) for
    human review, never silently dropped or merged. Every input
    RegionEvent (including ones carrying beat_episodes.py's own
    doublet_flag) is preserved in exactly one BeatCycle.

Pairing is a greedy nearest-unused-neighbor match (same heuristic style
as beat_events.regional_activation_delay / global_beat_candidates), not
a globally optimal assignment -- documented, not hidden.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from cardio_organoid.functional.beat_episodes import RegionEvent


@dataclasses.dataclass
class BeatCycle:
    beat_cycle_id: str
    region_id: str
    positive_event_time: float | None
    negative_event_time: float | None
    positive_original_frame: int | None
    negative_original_frame: int | None
    cycle_reference_time: float
    # negative_event_time - positive_event_time (signed); None if unpaired.
    # NOT a biological contraction/relaxation duration.
    polarity_interval: float | None
    # Gap to the NEXT cycle's cycle_reference_time; None for the last cycle.
    next_cycle_interval: float | None
    # "high" (paired, exactly one negative in the surrounding positive-
    # positive gap, neither event doublet-flagged), "medium" (paired but
    # the alternation isn't clean and/or a doublet is involved), or
    # "unpaired" (no matching event found within the pairing window).
    pairing_confidence: str
    flags: list[str]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _pair_positive_negative(
    positive_events: list[RegionEvent], negative_events: list[RegionEvent], max_pairing_window_s: float,
) -> list[dict]:
    pos_sorted = sorted(positive_events, key=lambda e: e.event_time)
    neg_sorted = sorted(negative_events, key=lambda e: e.event_time)
    used_neg_idx: set[int] = set()
    pairs = []
    for pe in pos_sorted:
        candidates = [
            (j, ne) for j, ne in enumerate(neg_sorted)
            if j not in used_neg_idx and abs(ne.event_time - pe.event_time) <= max_pairing_window_s
        ]
        if not candidates:
            pairs.append({"positive": pe, "negative": None})
            continue
        j, ne = min(candidates, key=lambda jt: abs(jt[1].event_time - pe.event_time))
        used_neg_idx.add(j)
        pairs.append({"positive": pe, "negative": ne})
    for j, ne in enumerate(neg_sorted):
        if j not in used_neg_idx:
            pairs.append({"positive": None, "negative": ne})
    return pairs


def _alternation_is_clean(
    pe: RegionEvent, ne: RegionEvent, pos_sorted: list[RegionEvent], neg_sorted: list[RegionEvent],
) -> bool:
    """True if `pe` and `ne` are each other's MUTUAL nearest opposite-
    polarity event (ne is the closest negative to pe, AND pe is the
    closest positive to ne) -- a clean 1:1 alternation candidate. False
    whenever a closer event exists on either side (e.g. two negatives
    clustered near the same positive-positive gap), which is exactly the
    ambiguous case this module flags rather than silently accepting."""
    nearest_neg_to_pe = min(neg_sorted, key=lambda n: abs(n.event_time - pe.event_time))
    nearest_pos_to_ne = min(pos_sorted, key=lambda p: abs(p.event_time - ne.event_time))
    return nearest_neg_to_pe.event_time == ne.event_time and nearest_pos_to_ne.event_time == pe.event_time


def build_beat_cycles(
    positive_events: list[RegionEvent], negative_events: list[RegionEvent], max_pairing_window_s: float,
) -> list[BeatCycle]:
    """Builds the beat-cycle sequence for ONE region (positive_events and
    negative_events must all share that region_id -- not checked/enforced
    here, see the caller). Returns cycles sorted by cycle_reference_time,
    with sequential beat_cycle_id and next_cycle_interval filled in."""
    if not positive_events and not negative_events:
        return []
    pos_sorted = sorted(positive_events, key=lambda e: e.event_time)
    neg_sorted = sorted(negative_events, key=lambda e: e.event_time)
    region_id = pos_sorted[0].region_id if pos_sorted else neg_sorted[0].region_id

    raw_pairs = _pair_positive_negative(positive_events, negative_events, max_pairing_window_s)

    prepared = []
    for pair in raw_pairs:
        pe, ne = pair["positive"], pair["negative"]
        reference_time = float(np.mean([x.event_time for x in (pe, ne) if x is not None]))
        flags: list[str] = []
        if pe is None or ne is None:
            pairing_confidence = "unpaired"
            flags.append("ambiguous_cycle_pairing")
        else:
            clean = _alternation_is_clean(pe, ne, pos_sorted, neg_sorted)
            doublet_involved = bool(pe.doublet_flag) or bool(ne.doublet_flag)
            if doublet_involved:
                flags.append("possible_compound_local_motion")
            if not clean:
                flags.append("ambiguous_cycle_pairing")
            pairing_confidence = "high" if (clean and not doublet_involved) else "medium"
        prepared.append({"positive": pe, "negative": ne, "reference_time": reference_time,
                          "pairing_confidence": pairing_confidence, "flags": flags})

    prepared.sort(key=lambda c: c["reference_time"])

    cycles = []
    for i, c in enumerate(prepared):
        pe, ne = c["positive"], c["negative"]
        cycles.append(BeatCycle(
            beat_cycle_id=f"{region_id}_cycle_{i:03d}",
            region_id=region_id,
            positive_event_time=pe.event_time if pe else None,
            negative_event_time=ne.event_time if ne else None,
            positive_original_frame=pe.original_frame if pe else None,
            negative_original_frame=ne.original_frame if ne else None,
            cycle_reference_time=c["reference_time"],
            polarity_interval=(ne.event_time - pe.event_time) if (pe and ne) else None,
            next_cycle_interval=None,
            pairing_confidence=c["pairing_confidence"],
            flags=c["flags"],
        ))
    for a, b in zip(cycles[:-1], cycles[1:]):
        a.next_cycle_interval = b.cycle_reference_time - a.cycle_reference_time
    return cycles


def beat_cycle_counts(cycles: list[BeatCycle]) -> dict:
    """Descriptive counts only -- NOT a decision about which count is
    "the" beat count. Reported alongside raw positive/negative event
    counts by the caller (see beat_episode_reconstruction.py)."""
    return {
        "n_beat_cycles_total": len(cycles),
        "n_paired_high_confidence": sum(1 for c in cycles if c.pairing_confidence == "high"),
        "n_paired_medium_confidence": sum(1 for c in cycles if c.pairing_confidence == "medium"),
        "n_unpaired": sum(1 for c in cycles if c.pairing_confidence == "unpaired"),
        "n_possible_compound_local_motion": sum(1 for c in cycles if "possible_compound_local_motion" in c.flags),
    }
