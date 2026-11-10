"""PROTOTYPE segmentation-QC-aware + gradual-transition-aware
trajectory stage model -- v2, additive to (does not replace)
prototype_trajectory_stage_model.py, which stays available for the
"old single-step rule" side of the explicit old-vs-new comparison this
round. NOT wired into any production pipeline path.

Per explicit project instruction, this round's biological trajectory
result must NOT be promoted to a confirmed stage -- v1's D10
TRANSITIONAL_SHEDDING / D12 MATURE_RECOMPACT calls for P39/Conv/a3 were
found to be strongly confounded with ALREADY-DOCUMENTED incomplete
segmentation on those exact days. v2 adds two independent changes:

1. A SEGMENTATION QC GATE, run BEFORE any stage inference. Any day
   whose segmentation is SEVERE_INCOMPLETE, REVIEW_REQUIRED and never
   resolved, or has a CONFIRMED alternative reference mask that
   substantially differs from what this trajectory's own automatic
   pipeline produced, is forced to STAGE_UNCERTAIN_SEGMENTATION --
   no biological stage is inferred for that day, and (critically) that
   day is ALSO EXCLUDED from the "this organoid's own trajectory
   baseline/median" statistics other days are compared against, so one
   badly-confounded day cannot distort the reading of every other day.
   A day that has genuinely never been reviewed (`NOT_REVIEWED`) is NOT
   forced to UNCERTAIN -- it still gets a normal inference, but is
   explicitly flagged as unverified rather than silently treated as
   equivalent to a confirmed-good day.

2. A GRADUAL/CUMULATIVE remodeling detector, applied ONLY across the
   segmentation-QC-CLEAN days of a trajectory (in their own chronological
   order, gaps from excluded days simply skipped -- day numbers are
   never re-indexed or treated as adjacent just because they are
   sequential in this filtered list). For each clean day it computes,
   over a short recent window of up to
   `GRADUAL_WINDOW_MAX_CLEAN_DAYS` clean days ending at that day:
     - a simple linear slope of solidity, circularity, radial_cv, and
       a perimeter/sqrt(area) shape-irregularity proxy, each over that
       window;
     - a cumulative fractional drop of solidity and of circularity from
       the window's own local maximum to the current day (catches a
       multi-day decline even if no single day-to-day step alone
       crosses a threshold).
   MULTI-METRIC CONCORDANCE (never a single metric alone, per explicit
   instruction): each of the 4 slopes is scored as "consistent with
   shedding" (solidity down, circularity down, radial_cv up, perimeter/
   sqrt(area) up) or not; `concordant_metric_count` (0-4) plus whether
   either cumulative drop exceeds `CUMULATIVE_DROP_HIGH_CONFIDENCE`/
   `CUMULATIVE_DROP_MODERATE_CONFIDENCE` sets a `TRANSITIONAL_SHEDDING`
   call's CONFIDENCE ("HIGH"/"MODERATE"/None) -- a day is only actually
   labeled TRANSITIONAL_SHEDDING here if confidence is HIGH or MODERATE;
   otherwise the (QC-gated, baseline-corrected) v1-style compact/loose/
   mature logic is used instead.

Functional onset remains AUXILIARY evidence only: this module never
uses the onset day itself as a transition day. Instead, for every
detected TRANSITIONAL_SHEDDING candidate window, `functional_onset_
temporal_proximity_days` reports how many days separate that window's
trough from the organoid's own functional onset day (None if onset
never occurred) -- purely descriptive, for a human to judge plausibility.

Every threshold is a generic, fixed constant, NOT tuned to either well
examined this round -- see the diagnostic script's README for the
explicit warning against over-fitting a 2-organoid exploratory check.
"""

from __future__ import annotations

import dataclasses

from cardio_organoid.pipeline.prototype_trajectory_stage_model import (
    EARLY_BASELINE_N_DAYS,
    EARLY_POSITION_FRACTION,
    HIGH_COMPACTNESS_RELATIVE_FRACTION,
    LOW_COMPACTNESS_RELATIVE_FRACTION,
    STAGE_EARLY_DIFFUSE,
    STAGE_INITIAL_COMPACT,
    STAGE_MATURE_RECOMPACT,
    STAGE_TRANSITIONAL_SHEDDING,
    STAGE_UNCERTAIN,
)

STAGE_UNCERTAIN_SEGMENTATION = "UNCERTAIN_SEGMENTATION"

# Segmentation QC statuses a caller may pass per day. Anything in
# BLOCKING_QC_STATUSES forces STAGE_UNCERTAIN_SEGMENTATION and excludes
# that day from baseline/trend statistics entirely.
QC_OK_CONFIRMED = "OK_CONFIRMED"
QC_NOT_REVIEWED = "NOT_REVIEWED"
QC_REVIEW_REQUIRED_UNRESOLVED = "REVIEW_REQUIRED_UNRESOLVED"
QC_SEVERE_INCOMPLETE = "SEVERE_INCOMPLETE"
QC_KNOWN_PARTIAL_MASK_REFERENCE_DIFFERS = "KNOWN_PARTIAL_MASK_REFERENCE_DIFFERS"
BLOCKING_QC_STATUSES = {QC_REVIEW_REQUIRED_UNRESOLVED, QC_SEVERE_INCOMPLETE, QC_KNOWN_PARTIAL_MASK_REFERENCE_DIFFERS}

GRADUAL_WINDOW_MAX_CLEAN_DAYS = 4  # "2-4 day rolling slope" per instruction -- uses up to this many clean days, fewer if not yet available
CUMULATIVE_DROP_HIGH_CONFIDENCE = 0.25
CUMULATIVE_DROP_MODERATE_CONFIDENCE = 0.15
MIN_CONCORDANT_METRICS_FOR_MODERATE = 2  # out of 4
MIN_CONCORDANT_METRICS_FOR_HIGH = 3  # out of 4
SLOPE_EPSILON = 1e-9  # anything smaller than this in magnitude counts as "flat", not a direction


@dataclasses.dataclass
class DayStageResultV2:
    day_number: int
    stage: str
    confidence: str | None  # "HIGH"/"MODERATE"/None -- only meaningful for TRANSITIONAL_SHEDDING calls
    segmentation_qc_status: str
    reasoning: list[str]
    evidence: dict


def _slope(xs: list[int], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    return num / den if den else 0.0


def _gradual_evidence(window: list[dict]) -> dict:
    """`window`: up to GRADUAL_WINDOW_MAX_CLEAN_DAYS clean, chronologically
    ordered records ending at the day being evaluated (the last entry)."""
    days = [r["day_number"] for r in window]
    solidity = [r["solidity"] for r in window]
    circularity = [r["circularity"] for r in window]
    radial_cv = [r.get("radial_cv") for r in window]
    perim_over_sqrt_area = [
        (r["perimeter_px"] / (r["area_px"] ** 0.5)) if r.get("perimeter_px") and r.get("area_px") else None
        for r in window
    ]

    def _clean(xs, ys):
        pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
        return ([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else ([], [])

    sx, sy = _clean(days, solidity)
    cx, cy = _clean(days, circularity)
    rx, ry = _clean(days, radial_cv)
    px, py = _clean(days, perim_over_sqrt_area)

    solidity_slope = _slope(sx, sy)
    circularity_slope = _slope(cx, cy)
    radial_cv_slope = _slope(rx, ry)
    perim_area_slope = _slope(px, py)

    cumulative_solidity_drop = ((max(sy) - sy[-1]) / max(sy)) if sy and max(sy) else 0.0
    cumulative_circularity_drop = ((max(cy) - cy[-1]) / max(cy)) if cy and max(cy) else 0.0

    concordant = {
        "solidity_declining": solidity_slope < -SLOPE_EPSILON,
        "circularity_declining": circularity_slope < -SLOPE_EPSILON,
        "radial_cv_increasing": radial_cv_slope > SLOPE_EPSILON,
        "perimeter_over_sqrt_area_increasing": perim_area_slope > SLOPE_EPSILON,
    }
    concordant_count = sum(concordant.values())

    return {
        "window_days": days, "solidity_slope": solidity_slope, "circularity_slope": circularity_slope,
        "radial_cv_slope": radial_cv_slope, "perimeter_over_sqrt_area_slope": perim_area_slope,
        "cumulative_solidity_drop": cumulative_solidity_drop, "cumulative_circularity_drop": cumulative_circularity_drop,
        "concordant_metrics": concordant, "concordant_metric_count": concordant_count,
    }


def _shedding_confidence(gradual: dict) -> str | None:
    max_cum_drop = max(gradual["cumulative_solidity_drop"], gradual["cumulative_circularity_drop"])
    if gradual["concordant_metric_count"] >= MIN_CONCORDANT_METRICS_FOR_HIGH and max_cum_drop >= CUMULATIVE_DROP_HIGH_CONFIDENCE:
        return "HIGH"
    if gradual["concordant_metric_count"] >= MIN_CONCORDANT_METRICS_FOR_MODERATE and max_cum_drop >= CUMULATIVE_DROP_MODERATE_CONFIDENCE:
        return "MODERATE"
    return None


def classify_trajectory_stages_qc_aware(
    morphology_records: list[dict], functional_onset_day: int | None, segmentation_qc_status_by_day: dict[int, str],
) -> list[DayStageResultV2]:
    """`morphology_records`: date-ordered per-day dicts (day_number,
    segmentation_status, solidity, circularity, radial_cv, perimeter_px,
    area_px). `segmentation_qc_status_by_day`: day_number -> one of the
    QC_* constants above; a day missing from this dict is treated as
    QC_NOT_REVIEWED."""
    results: list[DayStageResultV2] = []

    clean_days_in_order: list[dict] = []
    for rec in morphology_records:
        day = rec["day_number"]
        qc = segmentation_qc_status_by_day.get(day, QC_NOT_REVIEWED)
        if rec.get("segmentation_status") != "MEASURED" or rec.get("solidity") is None:
            results.append(DayStageResultV2(day, STAGE_UNCERTAIN, None, qc, ["no usable morphology (segmentation_status != MEASURED)"], {}))
            continue
        if qc in BLOCKING_QC_STATUSES:
            results.append(DayStageResultV2(
                day, STAGE_UNCERTAIN_SEGMENTATION, None, qc,
                [f"segmentation_qc_status={qc} -- day EXCLUDED from biological stage inference AND from this organoid's own trajectory baseline"],
                {"solidity": rec["solidity"], "circularity": rec.get("circularity")},
            ))
            continue
        clean_days_in_order.append(rec)  # QC_OK_CONFIRMED or QC_NOT_REVIEWED only

    if not clean_days_in_order:
        return results  # every day was either unmeasured or QC-blocked

    median_solidity = sorted(r["solidity"] for r in clean_days_in_order)[len(clean_days_in_order) // 2]
    clean_day_numbers = [r["day_number"] for r in clean_days_in_order]

    shedding_detected_so_far = False
    clean_index_by_day = {r["day_number"]: i for i, r in enumerate(clean_days_in_order)}

    for rec in morphology_records:
        day = rec["day_number"]
        qc = segmentation_qc_status_by_day.get(day, QC_NOT_REVIEWED)
        if rec.get("segmentation_status") != "MEASURED" or rec.get("solidity") is None or qc in BLOCKING_QC_STATUSES:
            continue  # already appended above

        i = clean_index_by_day[day]
        window = clean_days_in_order[max(0, i - GRADUAL_WINDOW_MAX_CLEAN_DAYS + 1): i + 1]
        gradual = _gradual_evidence(window) if len(window) >= 2 else None
        confidence = _shedding_confidence(gradual) if gradual else None

        rel_pos = (day - min(clean_day_numbers)) / (max(clean_day_numbers) - min(clean_day_numbers)) if max(clean_day_numbers) > min(clean_day_numbers) else 0.0
        rel_to_median = rec["solidity"] / median_solidity if median_solidity else None
        is_loose = rel_to_median is not None and rel_to_median < LOW_COMPACTNESS_RELATIVE_FRACTION
        is_compact = rel_to_median is not None and rel_to_median >= HIGH_COMPACTNESS_RELATIVE_FRACTION
        is_early_position = rel_pos <= EARLY_POSITION_FRACTION
        is_before_onset = functional_onset_day is None or day < functional_onset_day

        onset_proximity = None
        if confidence and functional_onset_day is not None:
            onset_proximity = abs(day - functional_onset_day)

        evidence = {
            "segmentation_qc_status": qc, "solidity": rec["solidity"], "circularity": rec.get("circularity"),
            "relative_position_in_clean_sequence": rel_pos, "solidity_relative_to_own_clean_trajectory_median": rel_to_median,
            "is_before_functional_onset": is_before_onset, "gradual_evidence": gradual,
            "functional_onset_temporal_proximity_days": onset_proximity,
        }
        reasoning = []
        if qc == QC_NOT_REVIEWED:
            reasoning.append("NOTE: segmentation not independently visually reviewed this round -- inference below is unverified, not confirmed-good")

        if confidence in ("HIGH", "MODERATE"):
            stage = STAGE_TRANSITIONAL_SHEDDING
            shedding_detected_so_far = True
            reasoning.append(
                f"gradual/cumulative shedding confidence={confidence}: {gradual['concordant_metric_count']}/4 metrics concordant, "
                f"max cumulative drop={max(gradual['cumulative_solidity_drop'], gradual['cumulative_circularity_drop']):.1%} "
                f"over window days {gradual['window_days']}"
                + (f"; {onset_proximity} day(s) from this organoid's own functional onset" if onset_proximity is not None else "; functional onset not reached or not evaluated")
            )
        elif is_loose and is_early_position and is_before_onset:
            stage = STAGE_EARLY_DIFFUSE
            reasoning.append(f"low relative compactness ({rel_to_median:.2f}x own CLEAN-trajectory median) + early position + before functional onset")
        elif is_compact and not shedding_detected_so_far and is_before_onset:
            stage = STAGE_INITIAL_COMPACT
            reasoning.append(f"high relative compactness ({rel_to_median:.2f}x own CLEAN-trajectory median), no shedding detected yet, before functional onset")
        elif is_compact and (shedding_detected_so_far or not is_before_onset):
            stage = STAGE_MATURE_RECOMPACT
            reasoning.append(f"high relative compactness ({rel_to_median:.2f}x own CLEAN-trajectory median), AFTER a detected shedding episode and/or functional onset")
        else:
            stage = STAGE_UNCERTAIN
            reasoning.append("evidence does not clearly satisfy any stage's conditions -- not guessed")

        results.append(DayStageResultV2(day, stage, confidence, qc, reasoning, evidence))

    results.sort(key=lambda r: [rec["day_number"] for rec in morphology_records].index(r.day_number))
    return results
