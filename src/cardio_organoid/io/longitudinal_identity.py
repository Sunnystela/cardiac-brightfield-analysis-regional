"""Confirmed longitudinal organoid identity for the P36-P39 tracking
dataset.

Per the project owner's own experimental-design knowledge (authoritative,
not inferred from files): for this dataset, `(batch, condition, well_id)`
alone identifies one physical organoid tracked across every day it has
ANY observation, regardless of which days have data or whether a given
day's observation is a photo or a video. A gap in the day sequence is
just a missing observation, not evidence of dropout or well reuse; a
well never having both a photo and a video on the same day does not
break its trajectory into two identities either. Batches are never
linked to each other (the same well_id in a different batch is a
different organoid) -- see this project's memory note
`longitudinal-organoid-identity-key.md` for the full context.

This module therefore does NOT re-derive or second-guess identity from
day-gap size, metadata timestamps, or modality co-occurrence (all of
that remains available as descriptive QC context, computed and reported
here, but never used to withhold or downgrade `identity_status`).
"""

from __future__ import annotations


def build_longitudinal_trajectory_summary(records_for_well: list[dict]) -> dict:
    """Descriptive-only summary of one (batch, condition, well_id)
    group's observed days -- day numbers, per-day modalities, and the
    day-sequence span/gap size. Purely informational QC context (e.g. so
    a human can see WHICH days lack data); `n_gap_days` is never used
    anywhere in this module (or by `confirmed_longitudinal_identity_
    record` below) to affect identity confidence."""
    day_numbers = sorted({r["day_number"] for r in records_for_well})
    modality_by_day: dict[int, set] = {}
    for r in records_for_well:
        modality_by_day.setdefault(r["day_number"], set()).add(r["modality"])

    n_days_observed = len(day_numbers)
    if n_days_observed >= 2:
        span_days = day_numbers[-1] - day_numbers[0]
        n_gap_days = span_days - (n_days_observed - 1)
    else:
        span_days = 0
        n_gap_days = 0

    modalities_ever_present = sorted({m for mods in modality_by_day.values() for m in mods})
    days_with_image = sorted(d for d, mods in modality_by_day.items() if "IMAGE" in mods)
    days_with_video = sorted(d for d, mods in modality_by_day.items() if "VIDEO" in mods)
    days_with_both_same_day = sorted(d for d, mods in modality_by_day.items() if {"IMAGE", "VIDEO"} <= mods)

    return {
        "n_days_observed": n_days_observed,
        "day_numbers_observed": day_numbers,
        "first_day_number": day_numbers[0] if day_numbers else None,
        "last_day_number": day_numbers[-1] if day_numbers else None,
        "day_sequence_span": span_days,
        "n_gap_days_in_span": n_gap_days,
        "modalities_ever_present": modalities_ever_present,
        "days_with_image": days_with_image,
        "days_with_video": days_with_video,
        "days_with_image_and_video_same_day": days_with_both_same_day,
    }


def confirmed_longitudinal_identity_record(batch: str, condition: str, well_id: str, trajectory_summary: dict) -> dict:
    """One finalized identity record: `longitudinal_id` = batch +
    condition + well_id (this project's confirmed grouping key, see
    module docstring), `identity_status` always "CONFIRMED" for any well
    with at least one observed day, `identity_source` always
    "experimental_design_manual_confirmation" -- never computed from
    file evidence, since the user's own experimental-design knowledge is
    the authority here, not an inference this code makes."""
    return {
        "longitudinal_id": f"{batch}__{condition}__{well_id}",
        "batch": batch, "condition": condition, "well_id": well_id,
        "identity_status": "CONFIRMED",
        "identity_source": "experimental_design_manual_confirmation",
        **trajectory_summary,
    }
