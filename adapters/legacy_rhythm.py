"""Legacy BPM / IBI / IBI-CV adapter (first-pass functional path).

Calculation (unchanged engine):
  analyze_video_functional (engine_scripts/longitudinal_modality_aware_dry_run.py)
    -> run_motion_pipeline (Farneback, frame_step 3, file-recorded FPS, injected ACCEPT mask)
    -> round-36 observability states (motion_observability_diagnostic.process_video)
    -> post_motion_functional (rounds 35/39/40/41: per-patch events, fundamental period,
       cycle-aware consensus, evidence bundle, confidence verdict)
Row mapping: `functional_record_from_result` (p36_conv_first_pass_screening.py, reused as is) and the
BPM gate of `stage_functional_aggregate` (`bpm` only for ESTIMABLE / LOW_CONFIDENCE).

`bpm`, `median_ibi_s` and `ibi_cv` come from DIFFERENT estimators (see docs/METRIC_DICTIONARY.csv);
they are kept as in the original and never reconciled here.
"""
from __future__ import annotations

import contextlib
import io
import json
import traceback
from pathlib import Path

from ._engine import engine
from .media import check_mask_matches_frame, reference_frame
from .records import base_fields, observation_id, require_mask

BPM_REPORTABLE_STATES = ("ESTIMABLE", "LOW_CONFIDENCE")


def _gate_bpm(row: dict) -> dict:
    # stage_functional_aggregate: "never report a BPM for NOT_ESTIMABLE; never 0"
    row["bpm"] = row.get("candidate_bpm") if row.get("functional_qc_status") in BPM_REPORTABLE_STATES else None
    return row


def _jdump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=False), encoding="utf-8")


def analyze_legacy_rhythm(video_path: Path, rec, mask_root: Path, output_dir: Path, meta: dict | None = None,
                          cv2_threads: int = 2, quiet: bool = True) -> dict:
    """Full legacy functional analysis of one VIDEO (decodes the video, runs Farneback on CPU)."""
    scr = engine("p36_conv_first_pass_screening")
    si = engine("cardio_organoid.pipeline.segmentation_interface")
    video_path = Path(video_path).resolve()
    row = base_fields(rec, video_path, meta)
    if rec.modality != "VIDEO":
        return {**row, "functional_execution_status": "NOT_MEASURED_IMAGE", "functional_qc_status": None, "bpm": None}
    if rec.segmentation_qc_status != si.QC_ACCEPT:
        return {**row, "functional_execution_status": "NOT_RUN", "functional_qc_status": None, "bpm": None,
                "functional_execution_reason": f"segmentation {rec.segmentation_qc_status}; no substitute mask"}
    obs = observation_id(rec)
    out_dir = Path(output_dir).resolve() / obs / "legacy_rhythm"
    try:
        mask = require_mask(rec, mask_root, "legacy_rhythm")
        _, info = reference_frame(video_path, "VIDEO")
        check_mask_matches_frame(mask, info)
        import cv2
        cv2.setNumThreads(cv2_threads)
        dry = engine("longitudinal_modality_aware_dry_run")
        cfg = engine("global_boundary_motion_diagnostic")._load_config()
        with si.fail_closed_mode():
            ctx = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
            with ctx:
                res = dry.analyze_video_functional(video_path, obs, None, cfg, out_dir, morphology_mask=mask,
                                                   mask_provenance={"interface_record": rec.to_dict()})
        _jdump(out_dir / "functional_result_full.json", res)
        rec_row = scr.functional_record_from_result(res)
        _jdump(out_dir / "result.json", rec_row)
        motion_dir = out_dir / "motion_pipeline" / obs
        qc = json.loads((motion_dir / "motion_pipeline_qc.json").read_text())
        row.update(functional_execution_status="RUN", reference_frame=info["reference_frame"],
                   motion_dir=str(motion_dir), round36=str(out_dir / "round36" / f"{obs}_round36_raw_per_patch.json"),
                   motion_segmentation_source=qc.get("segmentation_source"), **rec_row)
    except si.InterfaceFailClosed as exc:
        return {**row, "functional_execution_status": "FAILED_CLOSED", "functional_qc_status": None, "bpm": None,
                "functional_execution_reason": str(exc)}
    except Exception as exc:  # recorded, never silently dropped
        _jdump(out_dir / "failed.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        return {**row, "functional_execution_status": "FAILED", "functional_qc_status": None, "bpm": None,
                "functional_execution_reason": repr(exc)}
    return _gate_bpm(row)


def reaggregate_legacy_rhythm(video_path: Path, rec, motion_dir: Path, round36_json: Path, meta: dict | None = None) -> dict:
    """Re-run only `post_motion_functional` on an existing motion cache (no video decoding, no optical flow).

    Same as the REUSE branch of `stage_functional_aggregate`. `video_path` is only recorded, never opened.
    """
    scr = engine("p36_conv_first_pass_screening")
    dry = engine("longitudinal_modality_aware_dry_run")
    cfg = engine("global_boundary_motion_diagnostic")._load_config()
    motion_dir, round36_json = Path(motion_dir), Path(round36_json)
    with contextlib.redirect_stdout(io.StringIO()):
        res = dry.post_motion_functional(Path(video_path), observation_id(rec), cfg, motion_dir,
                                         json.loads(round36_json.read_text()))
    qc = json.loads((motion_dir / "motion_pipeline_qc.json").read_text())
    row = {**base_fields(rec, video_path, meta), "functional_execution_status": "REAGGREGATED_FROM_CACHE",
           "motion_dir": str(motion_dir), "round36": str(round36_json),
           "motion_segmentation_source": qc.get("segmentation_source"), **scr.functional_record_from_result(res)}
    return _gate_bpm(row)
