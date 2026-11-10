"""IMAGE-modality observation branch: morphology only, per explicit
project instruction this round.

A single image needs no reference-frame selection (unlike a video --
see `video_morphology_reference.py`): the image itself is the one
available observation. Segmentation reuses `segmentation.sam2_segmenter.
segment_frame` + the SAME provisional fixed candidate index this project
already uses everywhere else (`configs/default.yaml`'s
`segmentation.provisional_primary.candidate_index` -- per this project's
standing rule, never re-picked by a morphology metric; see the memory
note on never using shape metrics to select a segmentation mask).
Morphology feature extraction reuses `morphology.features.
compute_morphology_features` UNCHANGED.

Every functional field is explicitly NOT_MEASURED here -- an image
observation is never analyzed for BPM/IBI/synchrony, and never reports
those as 0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cardio_organoid.pipeline.segmentation_interface import assert_legacy_fixed_index_allowed
from cardio_organoid.morphology.features import compute_morphology_features
from cardio_organoid.pipeline.longitudinal_schema import (
    FUNCTIONAL_STATUS_NOT_MEASURED,
    MODALITY_IMAGE,
    MORPHOLOGY_STATUS_MEASURED,
    MORPHOLOGY_STATUS_SEGMENTATION_FAILED,
    new_observation_record,
)
from cardio_organoid.segmentation.sam2_segmenter import provisional_primary_qc, segment_frame


def segment_and_measure_image(image_rgb: np.ndarray, predictor, candidate_index: int, min_dominant_component_fraction: float) -> dict:
    """SAM2-segments `image_rgb` (single-point center prompt, same
    convention as `_recover_mask_and_roi` elsewhere in this project) and
    returns {"mask", "morphology_status", "qc"} -- `morphology_status` is
    SEGMENTATION_FAILED if SAM2 returns no usable candidate at
    `candidate_index` or the QC dominant-component check fails (never
    silently swaps to a different, "better-looking" candidate -- see
    module docstring)."""
    assert_legacy_fixed_index_allowed("pipeline.image_branch.segment_and_measure_image")
    candidates = segment_frame(predictor, image_rgb)
    qc = provisional_primary_qc(candidates, candidate_index, min_dominant_component_fraction)
    if qc.get("flag") == "invalid_index":
        return {"mask": None, "morphology_status": MORPHOLOGY_STATUS_SEGMENTATION_FAILED, "qc": qc}
    mask = candidates[candidate_index].mask
    # QC flag "review_needed" is a signal for human review, not proof the
    # mask is unusable -- morphology is still computed and reported (per
    # this project's convention of never silently discarding a QC-flagged
    # result), with the flag itself carried through in `qc` for the caller.
    return {"mask": mask, "morphology_status": MORPHOLOGY_STATUS_MEASURED, "qc": qc}


def analyze_image_observation(
    image_path: Path,
    image_rgb: np.ndarray,
    predictor,
    config: dict,
    longitudinal_id: str, batch: str, condition: str, well_id: str, day: str, day_number: int,
    observation_id: str, pipeline_version: str, git_commit: str | None,
) -> dict:
    """Full IMAGE-branch observation record for one image file: segment,
    compute morphology, and explicitly fill every functional field as
    NOT_MEASURED (never 0, never omitted -- see `longitudinal_schema.py`
    module docstring)."""
    sam2_cfg = config["segmentation"]["provisional_primary"]
    seg = segment_and_measure_image(image_rgb, predictor, sam2_cfg["candidate_index"], sam2_cfg["min_dominant_component_fraction"])

    morphology_fields = {}
    if seg["mask"] is not None and seg["morphology_status"] == MORPHOLOGY_STATUS_MEASURED:
        features = compute_morphology_features(seg["mask"])
        morphology_fields = {
            "area_px": features.pixel.area_px, "perimeter_px": features.pixel.perimeter_px,
            "equivalent_diameter_px": features.pixel.equivalent_diameter_px,
            "circularity": features.pixel.circularity, "solidity": features.pixel.solidity,
            "eccentricity": features.pixel.eccentricity, "aspect_ratio": features.pixel.aspect_ratio,
            "radial_cv": features.pixel.radial_cv,
        }

    return new_observation_record(
        longitudinal_id=longitudinal_id, batch=batch, condition=condition, well_id=well_id,
        day=day, day_number=day_number, modality=MODALITY_IMAGE,
        source_file=str(image_path), observation_id=observation_id,
        segmentation_source=f"sam2_candidate_index_{sam2_cfg['candidate_index']}",
        morphology_reference_strategy="single_image_no_reference_needed",
        actual_fps=None, pipeline_version=pipeline_version, git_commit=git_commit,
        morphology_available=True, functional_available=False,
        morphology_status=seg["morphology_status"], **morphology_fields,
        functional_status=FUNCTIONAL_STATUS_NOT_MEASURED,
    )
