"""Modality-aware observation / organoid-day schema for the P36-P39
longitudinal tracking dataset.

Two table levels, kept SEPARATE per explicit project instruction (never
averaged together silently):

  - **Observation level** (`new_observation_record`): one row per single
    source FILE (one image, or one video). This is the raw, unaggregated
    measurement -- if a day happens to have more than one file for the
    same organoid (a replicate), each gets its own observation row.
  - **Organoid-day level** (`aggregate_observations_to_organoid_day`):
    one row per (longitudinal_id, day) -- built FROM the observation
    rows, preserving `n_replicate_observations` and only aggregating
    (median, across replicates) genuinely numeric fields. A field that
    is NA/NOT_MEASURED on some or all replicate observations is never
    silently coerced into a numeric average.

`longitudinal_id = batch + "__" + condition + "__" + well_id` (see
`cardio_organoid.io.longitudinal_identity` -- CONFIRMED by the project
owner's own experimental-design knowledge, not re-derived here).

Modality rule (CLAUDE.md-equivalent for this dataset, per explicit
project instruction this round):
  - IMAGE observations: morphology only. `functional_available=False`,
    `functional_status=NOT_MEASURED`, every functional numeric field
    (`bpm`, `beat_count`, `median_ibi_s`, `ibi_cv`, ...) is `None` -- an
    image-only day is NEVER recorded as BPM=0.
  - VIDEO observations: both morphology AND functional analysis are
    attempted. `functional_status` becomes one of NOT_ESTIMABLE /
    LOW_CONFIDENCE / ESTIMABLE (see `FUNCTIONAL_STATUS_*` below) --
    `functional_available=True` describes only whether a video exists to
    attempt functional analysis on, not whether that attempt succeeded;
    a video whose functional analysis fails is `NOT_ESTIMABLE`, still
    logically distinct from an image day's `NOT_MEASURED` (no attempt
    was even possible).
"""

from __future__ import annotations

import statistics
from collections import defaultdict

MODALITY_IMAGE = "IMAGE"
MODALITY_VIDEO = "VIDEO"

# Functional-analysis status -- mutually exclusive, never inferred from
# a numeric 0. See module docstring for the IMAGE/VIDEO distinction.
FUNCTIONAL_STATUS_NOT_MEASURED = "NOT_MEASURED"  # no video exists for this observation/day
FUNCTIONAL_STATUS_NOT_ESTIMABLE = "NOT_ESTIMABLE"  # video exists, but evidence does not support a BPM (round 41)
FUNCTIONAL_STATUS_LOW_CONFIDENCE = "LOW_CONFIDENCE"  # video exists, a candidate BPM exists, with named caveats
FUNCTIONAL_STATUS_ESTIMABLE = "ESTIMABLE"  # video exists, evidence supports a candidate BPM

# Morphology-analysis status -- separate from functional_status; a video
# can have MEASURED morphology while functional_status is NOT_ESTIMABLE
# (weak beating does not necessarily mean the organoid outline itself
# could not be segmented), and vice versa.
MORPHOLOGY_STATUS_MEASURED = "MEASURED"
MORPHOLOGY_STATUS_SEGMENTATION_FAILED = "SEGMENTATION_FAILED"
MORPHOLOGY_STATUS_NOT_ATTEMPTED = "NOT_ATTEMPTED"

_MORPHOLOGY_FIELDS = ("area_px", "perimeter_px", "equivalent_diameter_px", "circularity", "solidity", "eccentricity", "aspect_ratio", "radial_cv")
_FUNCTIONAL_NUMERIC_FIELDS = (
    "bpm", "beat_count", "median_ibi_s", "ibi_cv",
    "reliable_active_fraction", "observable_fraction",
    "estimator_pairwise_agreement", "dominant_period_family_fraction",
)

# `observable_fraction` and `reliable_active_fraction` are DIFFERENT
# quantities over the SAME denominator, NEVER interchangeable -- a bug
# found and fixed in this project's own dry-run pipeline once copied one
# into the other:
#   observable_fraction        = (n_total_patches - n_state_C) / n_total_patches
#     numerator: every analyzed patch NOT in round 36's state C
#     ("C_NOT_OBSERVABLE_do_not_interpret_as_no_motion") -- includes
#     states A (reliable+active), B (likely genuinely weak), and D
#     (ambiguous). Answers "how much of the grid could even be trusted
#     to measure motion on at all".
#   reliable_active_fraction   = n_state_A / n_total_patches
#     numerator: patches SPECIFICALLY in round 36's state A
#     ("A_reliable_active_candidate") -- a STRICTER subset of
#     observable_fraction (excludes B and D too). Answers "how much of
#     the grid is both observable AND actually showing reliable
#     activity". If a caller has not actually computed round 36's
#     per-state breakdown for a given observation, this field MUST be
#     left `None` here (schema-level NOT_IMPLEMENTED) rather than
#     populated by copying `observable_fraction` -- they answer different
#     questions and are not proxies for each other.

OBSERVATION_RECORD_FIELDS = (
    # identity / provenance
    "longitudinal_id", "batch", "condition", "well_id", "day", "day_number",
    "modality", "source_file", "observation_id",
    "segmentation_source", "morphology_reference_strategy",
    "actual_fps", "pipeline_version", "git_commit",
    # availability flags
    "morphology_available", "functional_available",
    # morphology
    "morphology_status", *_MORPHOLOGY_FIELDS,
    # functional
    "functional_status", *_FUNCTIONAL_NUMERIC_FIELDS,
    "estimator_confidence_state", "regional_activity_summary",
    "synchrony_metrics",
)

ORGANOID_DAY_RECORD_FIELDS = (*OBSERVATION_RECORD_FIELDS, "n_replicate_observations", "replicate_observation_ids")


def new_observation_record(**kwargs) -> dict:
    """Builds one observation-level row. Every field in
    `OBSERVATION_RECORD_FIELDS` is present (defaulting to `None` if not
    passed) so every row has an identical, complete key set regardless of
    modality -- an IMAGE row simply has every functional field `None`
    with `functional_status=NOT_MEASURED`, rather than the keys being
    absent. Raises on an unrecognized kwarg (typo protection -- this
    schema is meant to be the single source of truth for field names)."""
    unknown = set(kwargs) - set(OBSERVATION_RECORD_FIELDS)
    if unknown:
        raise ValueError(f"Unknown observation field(s): {sorted(unknown)}. See OBSERVATION_RECORD_FIELDS.")
    record = dict.fromkeys(OBSERVATION_RECORD_FIELDS)
    record.update(kwargs)
    return record


def _is_numeric(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def aggregate_observations_to_organoid_day(observations: list[dict]) -> list[dict]:
    """Groups observation rows by (`longitudinal_id`, `day`) and builds
    ONE organoid-day row per group. Replicate observations are preserved
    (`n_replicate_observations`, `replicate_observation_ids`) -- NEVER
    silently averaged away. For each field in `_MORPHOLOGY_FIELDS` /
    `_FUNCTIONAL_NUMERIC_FIELDS`: if every replicate has a numeric value,
    the aggregated value is their MEDIAN (robust to a single bad
    replicate, same convention as every other median-based aggregation
    in this project); if ANY replicate has that field as `None`
    (NOT_MEASURED/NOT_ESTIMABLE/segmentation failure/etc.), the
    aggregated field is `None` too -- a missing value is never dropped
    from the group to force a clean average from the rest, since that
    would silently hide the missingness. Non-numeric/status/provenance
    fields take the single value when all replicates agree, else `None`
    with the disagreement still visible via the underlying observation
    rows (never guessed)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for obs in observations:
        groups[(obs.get("longitudinal_id"), obs.get("day"))].append(obs)

    aggregated = []
    for (_longitudinal_id, _day), group in groups.items():
        row = dict(group[0])
        row["n_replicate_observations"] = len(group)
        row["replicate_observation_ids"] = [g.get("observation_id") for g in group]

        for field in ORGANOID_DAY_RECORD_FIELDS:
            if field in ("n_replicate_observations", "replicate_observation_ids"):
                continue
            values = [g.get(field) for g in group]
            if field in _MORPHOLOGY_FIELDS or field in _FUNCTIONAL_NUMERIC_FIELDS:
                if all(_is_numeric(v) for v in values):
                    row[field] = float(statistics.median(values)) if len(values) > 1 else values[0]
                else:
                    row[field] = None
            else:
                row[field] = values[0] if len(set(map(str, values))) == 1 else None
        aggregated.append(row)
    return aggregated
