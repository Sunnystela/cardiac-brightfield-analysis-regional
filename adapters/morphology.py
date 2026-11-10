"""Morphology adapter: one observation -> one row with the first-pass master's morphology columns.

Calculation: engine `p36_conv_first_pass_screening.morph_metrics` ->
`cardio_organoid.morphology.features.compute_morphology_features(mask).pixel` (unchanged).
Row assembly mirrors `stage_morphology` (engine_scripts/p36_conv_first_pass_screening.py:111-140).
"""
from __future__ import annotations

from pathlib import Path

from ._engine import engine
from .records import base_fields, require_mask

MORPH_METRICS = ["area_px", "perimeter_px", "equivalent_diameter_px", "circularity", "solidity",
                 "eccentricity", "aspect_ratio", "radial_cv", "perimeter_over_sqrt_area"]


def calibration_for_image(image_path: Path) -> dict:
    """Mirror of `calibration_for_image_day` (engine_scripts/p39_conv_a3_longitudinal_morphology_pilot.py:153).

    Same sidecar rule, same status strings, same parser (`parse_pixel_calibration_um`). The only difference:
    the original stores the sidecar path relative to its repository and therefore raises for files
    outside it; here the path is stored as given.
    """
    parse = engine("cardio_organoid.io.longitudinal_metadata_xml").parse_pixel_calibration_um
    image_path = Path(image_path)
    xml_path = image_path.with_name(image_path.name + "_metadata.xml")
    if not xml_path.exists():
        return {"calibration_status": "NOT_CALIBRATED_no_metadata_sidecar", "um_per_pixel": None}
    cal = parse(xml_path)
    if cal["um_per_pixel_x"] is None:
        return {"calibration_status": "NOT_CALIBRATED_sidecar_present_but_no_scaling_block", "um_per_pixel": None,
                "xml_sidecar": str(xml_path)}
    return {"calibration_status": "CALIBRATED_from_real_metadata_xml",
            "um_per_pixel_x": cal["um_per_pixel_x"], "um_per_pixel_y": cal["um_per_pixel_y"], "xml_sidecar": str(xml_path)}


def analyze_morphology(rec, mask_root: Path, source_file: Path | None = None, meta: dict | None = None,
                       measurement_caveat: str = "") -> dict:
    """Morphology for one segmentation record. Never raises for QC reasons: a non-ACCEPT or invalid
    record yields a row with metrics None and the reason kept (CLAUDE.md #11)."""
    scr = engine("p36_conv_first_pass_screening")
    si = engine("cardio_organoid.pipeline.segmentation_interface")
    row = {**base_fields(rec, source_file, meta), "measurement_caveat": measurement_caveat}
    empty = {"morphology_unit": None, "calibration_status": None, "um_per_pixel": None, "um_per_pixel_y": None,
             **{k: None for k in MORPH_METRICS}, "area_um2": None, "equivalent_diameter_um": None}
    if rec.segmentation_qc_status != si.QC_ACCEPT:
        return {**row, "morphology_status": "MISSING_" + rec.segmentation_qc_status, **empty}
    try:
        mask = require_mask(rec, mask_root, "morphology")
    except si.InterfaceFailClosed as exc:
        return {**row, "morphology_status": "FAILED_CLOSED", "morphology_reason": str(exc), **empty}
    m = scr.morph_metrics(mask)
    if rec.modality == "IMAGE":
        cal = calibration_for_image(Path(source_file)) if source_file else {"calibration_status": "NOT_CALIBRATED_no_source_file"}
    else:
        cal = {"calibration_status": "NOT_CALIBRATED_no_metadata_source"}
    um = cal.get("um_per_pixel_x")  # X only, as in the original (x == y in all P36/Conv sidecars)
    return {**row, "morphology_status": "MEASURED", "morphology_unit": "px", "calibration_status": cal.get("calibration_status"),
            "um_per_pixel": um, "um_per_pixel_y": cal.get("um_per_pixel_y"), **m,
            "area_um2": m["area_px"] * um ** 2 if um else None,
            "equivalent_diameter_um": m["equivalent_diameter_px"] * um if um else None,
            "mask_shape": list(mask.shape)}
