"""PROTOTYPE trajectory-aware developmental-stage model -- NOT wired
into any production pipeline path.

Per explicit project instruction, replaces the earlier single-frame
3-way (COMPACT/EARLY_DIFFUSE/AMBIGUOUS) gate for THIS purpose with a
5-way stage vocabulary that reflects the expected biological
trajectory, and classifies each day using the SAME organoid's own
date-ordered TRAJECTORY (not a single-frame threshold):

  EARLY_DIFFUSE        -- single-cell-rich, loose aggregate, diffuse boundary
  INITIAL_COMPACT       -- round/compact, BEFORE any detected remodeling episode and before functional onset
  TRANSITIONAL_SHEDDING -- strong shape deformation / asymmetry / lobulation, often around functional onset
  MATURE_RECOMPACT      -- round/compact again, AFTER a detected remodeling episode and/or functional onset
  UNCERTAIN             -- evidence insufficient or conflicting; never guessed

CRITICAL DESIGN POINT: INITIAL_COMPACT and MATURE_RECOMPACT can look
morphologically IDENTICAL (both round/compact) -- they are told apart
ONLY by TRAJECTORY POSITION (has a transitional/shedding episode
already occurred earlier in THIS SAME organoid's own sequence?) and by
FUNCTIONAL ONSET (has beating been functionally detected yet?), used as
auxiliary evidence, never morphology alone.

Day numbers are NEVER hardcoded. "Early in the trajectory" and "before/
after" are always computed relative to THIS organoid's own observed
day sequence and its own baseline (e.g. the median of its own first
available days), so different batches/organoids with different
transition timing are handled the same way.

Every threshold is a generic, documented constant, NOT tuned to any
one organoid -- this module is explicitly EXPLORATORY (see the
diagnostic script's README): the goal is to check whether the
hypothesized 4-stage order is CONSISTENT with real trajectories, not to
ship a validated classifier.
"""

from __future__ import annotations

import dataclasses

STAGE_EARLY_DIFFUSE = "EARLY_DIFFUSE"
STAGE_INITIAL_COMPACT = "INITIAL_COMPACT"
STAGE_TRANSITIONAL_SHEDDING = "TRANSITIONAL_SHEDDING"
STAGE_MATURE_RECOMPACT = "MATURE_RECOMPACT"
STAGE_UNCERTAIN = "UNCERTAIN"

# Generic, fixed constants -- see module docstring. Not tuned per organoid.
EARLY_BASELINE_N_DAYS = 2  # this organoid's own first N successfully-measured days define its "early" baseline
LOW_COMPACTNESS_RELATIVE_FRACTION = 0.85  # solidity/circularity below this fraction of the organoid's OWN trajectory median counts as "loose"
HIGH_COMPACTNESS_RELATIVE_FRACTION = 0.95  # solidity/circularity at or above this fraction of the organoid's OWN trajectory median counts as "compact"
SHEDDING_RELATIVE_DROP_FRACTION = 0.15  # a >=15% relative drop in solidity (or circularity) vs. the immediately preceding measured day flags a candidate remodeling event
EARLY_POSITION_FRACTION = 0.35  # "early in the sequence" = within the first 35% of observed-day positions


@dataclasses.dataclass
class DayStageResult:
    day_number: int
    stage: str
    reasoning: list[str]
    evidence: dict


def _relative_position(day_number: int, day_numbers: list[int]) -> float:
    lo, hi = min(day_numbers), max(day_numbers)
    return (day_number - lo) / (hi - lo) if hi > lo else 0.0


def classify_trajectory_stages(morphology_records: list[dict], functional_onset_day: int | None) -> list[DayStageResult]:
    """`morphology_records`: date-ordered list of dicts with at least
    `day_number`, `segmentation_status` ("MEASURED" or other),
    `solidity`, `circularity`. `functional_onset_day`: this organoid's
    own first day_number with an ESTIMABLE/LOW_CONFIDENCE functional
    verdict, or None if never reached / not evaluated -- used ONLY as
    auxiliary evidence, never a hard cutoff on its own.

    Returns one DayStageResult per input record, in the same order."""
    measured = [r for r in morphology_records if r.get("segmentation_status") == "MEASURED" and r.get("solidity") is not None]
    day_numbers = [r["day_number"] for r in measured]
    if not measured:
        return [DayStageResult(r["day_number"], STAGE_UNCERTAIN, ["no measured days in this trajectory"], {}) for r in morphology_records]

    baseline_days = sorted(measured, key=lambda r: r["day_number"])[:EARLY_BASELINE_N_DAYS]
    solidity_values = [r["solidity"] for r in measured]
    median_solidity = sorted(solidity_values)[len(solidity_values) // 2]

    results: list[DayStageResult] = []
    shedding_detected_so_far = False
    prev_measured: dict | None = None

    for rec in morphology_records:
        day = rec["day_number"]
        if rec.get("segmentation_status") != "MEASURED" or rec.get("solidity") is None:
            results.append(DayStageResult(day, STAGE_UNCERTAIN, [f"segmentation_status={rec.get('segmentation_status')} -- no usable morphology"], {}))
            continue

        solidity = rec["solidity"]
        rel_pos = _relative_position(day, day_numbers)
        rel_to_median = solidity / median_solidity if median_solidity else None
        evidence = {
            "solidity": solidity, "relative_position_in_sequence": rel_pos,
            "solidity_relative_to_own_trajectory_median": rel_to_median,
            "is_before_functional_onset": (functional_onset_day is None or day < functional_onset_day),
            "shedding_already_detected_earlier_in_this_trajectory": shedding_detected_so_far,
        }
        reasoning = []

        is_shedding_candidate = False
        if prev_measured is not None and prev_measured.get("solidity"):
            drop = (prev_measured["solidity"] - solidity) / prev_measured["solidity"]
            evidence["relative_solidity_drop_vs_previous_measured_day"] = drop
            if drop >= SHEDDING_RELATIVE_DROP_FRACTION:
                is_shedding_candidate = True
                reasoning.append(f"solidity dropped {drop:.1%} vs. previous measured day (>= {SHEDDING_RELATIVE_DROP_FRACTION:.0%} threshold)")

        is_loose = rel_to_median is not None and rel_to_median < LOW_COMPACTNESS_RELATIVE_FRACTION
        is_compact = rel_to_median is not None and rel_to_median >= HIGH_COMPACTNESS_RELATIVE_FRACTION
        is_early_position = rel_pos <= EARLY_POSITION_FRACTION

        if is_shedding_candidate:
            stage = STAGE_TRANSITIONAL_SHEDDING
            shedding_detected_so_far = True
        elif is_loose and is_early_position and evidence["is_before_functional_onset"]:
            stage = STAGE_EARLY_DIFFUSE
            reasoning.append(f"low relative compactness ({rel_to_median:.2f}x own trajectory median) + early sequence position ({rel_pos:.2f}) + before functional onset")
        elif is_compact and not shedding_detected_so_far and evidence["is_before_functional_onset"]:
            stage = STAGE_INITIAL_COMPACT
            reasoning.append(f"high relative compactness ({rel_to_median:.2f}x own trajectory median), no shedding episode yet, before functional onset")
        elif is_compact and (shedding_detected_so_far or not evidence["is_before_functional_onset"]):
            stage = STAGE_MATURE_RECOMPACT
            reasoning.append(f"high relative compactness ({rel_to_median:.2f}x own trajectory median), AFTER a detected shedding episode and/or functional onset")
        else:
            stage = STAGE_UNCERTAIN
            reasoning.append("evidence does not clearly satisfy any stage's conditions -- not guessed")

        results.append(DayStageResult(day, stage, reasoning, evidence))
        prev_measured = rec

    return results
