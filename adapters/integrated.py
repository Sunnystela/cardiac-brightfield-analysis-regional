"""Integrated analysis of one observation, from validated inputs only.

Gate: the segmentation record must pass the engine gate (valid ACCEPT, human provenance, mask file
present, mask shape == reference frame). Otherwise nothing is measured and the row carries the reason.

Columns combine the two conventions that exist in the original code, under their original names:
  * first-pass master: `functional_qc_status`, `bpm` (ESTIMABLE or LOW_CONFIDENCE), `median_ibi_s`,
    `ibi_cv` (NOT gated by the verdict -- as stored in P36_CONV_MASTER_OBSERVATION_TABLE.csv);
  * pixel-first master: `legacy_bpm`, `legacy_ibi_cv`, `legacy_median_ibi_s` (ESTIMABLE only) and
    `legacy_bpm_reported` (= `bpm`).
Pixel-first values never fill any BPM column.
"""
from __future__ import annotations

from pathlib import Path

from ._engine import engine
from .legacy_rhythm import analyze_legacy_rhythm
from .media import check_mask_matches_frame, reference_frame
from .morphology import analyze_morphology
from .pixel_first import analyze_pixel_first
from .records import base_fields, require_mask


def legacy_columns_pixel_first_convention(row: dict) -> dict:
    """Mirror of build_master (pixel_first_motion_aggregate.py:86-91)."""
    est = row.get("functional_qc_status") == "ESTIMABLE"
    return {"legacy_bpm_reported": row.get("bpm"),
            "legacy_bpm": row.get("bpm") if est else None,
            "legacy_ibi_cv": row.get("ibi_cv") if est else None,
            "legacy_median_ibi_s": row.get("median_ibi_s") if est else None}


def _regional(leg: dict, source_path: Path, rec, mask_root: Path, render_figures: bool) -> dict:
    """[added] Regional (multi-rhythm) BPMs on the legacy cache just written. Never alters `bpm`."""
    if leg.get("functional_execution_status") != "RUN" or not leg.get("motion_dir"):
        return {"regional_status": "NOT_RUN_NO_LEGACY_CACHE"}
    try:
        from .regional_rhythm import analyze_regional_rhythm, overlay_inputs
        frame, mask = overlay_inputs(Path(source_path), rec, mask_root)
        out = analyze_regional_rhythm(Path(leg["motion_dir"]).parent.parent, frame_bgr=frame, mask=mask,
                                      render_figures=render_figures)
        out.pop("regional_outputs", None)
        return out
    except Exception as exc:  # recorded, never silently dropped; official columns are unaffected
        return {"regional_status": "FAILED", "regional_reason": repr(exc)}


def analyze_integrated(source_path: Path, rec, mask_root: Path, output_dir: Path, meta: dict | None = None,
                       run_legacy: bool = True, run_pixel_first: bool = True, render_figures: bool = True,
                       run_regional: bool = True) -> dict:
    si = engine("cardio_organoid.pipeline.segmentation_interface")
    row = {**base_fields(rec, source_path, meta)}
    try:
        mask = require_mask(rec, mask_root, "integrated")
        _, info = reference_frame(Path(source_path), rec.modality)
        check_mask_matches_frame(mask, info)
    except si.InterfaceFailClosed as exc:
        return {**row, "integrated_status": "NOT_RUN_INPUT_NOT_VALIDATED", "integrated_reason": str(exc)}
    except (ValueError, IOError, FileNotFoundError) as exc:
        return {**row, "integrated_status": "NOT_RUN_INPUT_PROBLEM", "integrated_reason": repr(exc)}

    out = {**row, "integrated_status": "RUN", **analyze_morphology(rec, mask_root, source_path, meta)}
    if rec.modality == "VIDEO":
        if run_legacy:
            leg = analyze_legacy_rhythm(source_path, rec, mask_root, output_dir, meta)
            out.update({k: v for k, v in leg.items() if k not in row})
            out.update(legacy_columns_pixel_first_convention(leg))
            out.update(_regional(leg, source_path, rec, mask_root, render_figures) if run_regional
                       else {"regional_status": "SKIPPED_BY_USER"})
        else:
            out["functional_execution_status"] = "SKIPPED_BY_USER"
        if run_pixel_first:
            pf = analyze_pixel_first(source_path, rec, mask_root, output_dir, meta, render_figures=render_figures)
            out.update({k: v for k, v in pf.items() if k not in row})
        else:
            out["pixel_first_status"] = "SKIPPED_BY_USER"
    else:
        out.update(functional_execution_status="NOT_MEASURED_IMAGE", pixel_first_status="NOT_APPLICABLE_IMAGE")
    return out
