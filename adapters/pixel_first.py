"""Pixel-first motion adapter (exploratory descriptor channel; NOT a BPM and never substituted for one).

Calculation (unchanged engine, engine_scripts/pixel_first_motion_batch.py):
  compute_primitives  video -> Farneback on the functional ROI, whole-ROI median drift removal,
                      validity maps (saturated / low texture / forward-backward), int16 flow chunks
  derive              strict-inside / outer-support / background summaries, motion-pattern regions,
                      autocorrelation periods, timing dispersion (+ figures and movie)
Row mapping: `row_from_summary` mirrors the pf_* block of `build_master`
(engine_scripts/pixel_first_motion_aggregate.py:72-136).

The batch functions locate their inputs through the module constants REPO_ROOT and MANIFEST.
`rebind_paths` points them at a per-observation workspace inside the user's output directory
for the duration of the call; no numeric constant is touched.
"""
from __future__ import annotations

import contextlib
import json
import shutil
import traceback
from pathlib import Path

from ._engine import engine, rebind_paths
from .media import check_mask_matches_frame, reference_frame
from .records import base_fields, observation_id, require_mask, resolved_mask_path, write_records

PF_NOTE = "pixel-first |u| descriptors; pf_mag_rate_per_min is an autocorrelation repetition rate, NOT a BPM"


def row_from_summary(s: dict, p: dict) -> dict:
    """pf_* columns exactly as `build_master` assigns them from summary.json / primitives.json."""
    r1 = s["regions"][0] if s["regions"] else {}
    per = (r1.get("mag_period") or {}).get("positive_peak_period_s") if r1 else None
    disp = r1.get("within_region_timing_dispersion") if r1 else None
    ins, sup = s["exclusions"]["INSIDE_REFERENCE_MASK"], s["exclusions"]["OUTER_BOUNDARY_SUPPORT"]
    return {"pixel_first_status": "REUSED" if p["source"] == "REUSED_PILOT" else "COMPUTED",
            "pf_fps": s["fps"], "pf_dt_s": s["dt_s"],
            "pf_measurable_fraction": s["measurable_fraction_inside_mask"],
            "pf_valid_px_inside": ins["n_valid"], "pf_mask_px": ins["n_region_px"],
            "pf_excl_saturated": ins["n_saturated"], "pf_excl_low_texture": ins["n_low_texture"],
            "pf_excl_fb_inconsistent": ins["n_fb_inconsistent"], "pf_excl_unique": ins["n_excluded_unique"],
            "pf_excl_multiple_reasons": ins["n_excluded_multiple_reasons"], "pf_valid_px_outer_support": sup["n_valid"],
            "pf_strict_mag_mean": s["strict_inside"]["mag_mean_px_per_pair"],
            "pf_strict_speed_p95": s["strict_inside"]["speed_p95_px_per_s"],
            "pf_support_incl_speed_p95": s["support_inclusive"]["speed_p95_px_per_s"],
            "pf_support_to_strict_p95_ratio": s["support_to_strict_p95_ratio"],
            "pf_corr_strict_vs_support": s["corr_strict_vs_support_waveform"],
            "pf_corr_strict_with_background": s["corr_strict_with_background_median_mag"],
            "pf_n_regions": s["n_regions"], "pf_single_region": s["n_regions"] == 1,
            "pf_n_active_blocks": s["n_active_blocks"],
            "pf_region1_fraction_outer_support": r1.get("fraction_px_outer_support"),
            "pf_region1_n_mag_peaks": r1.get("n_mag_peaks"),
            "pf_region1_peaks_per_period": r1.get("mag_peaks_per_positive_period"),
            "pf_mag_period_s": per, "pf_mag_rate_per_min": (60.0 / per) if per else None,
            "pf_mag_period_autocorr": (r1.get("mag_period") or {}).get("positive_peak_autocorr") if r1 else None,
            "pf_mag_period_pilot_rule_s": (r1.get("mag_period") or {}).get("pilot_rule_period_s") if r1 else None,
            "pf_radial_fraction_centroid": r1.get("centroid_based_radial_fraction_active"),
            "pf_tangential_fraction_centroid": r1.get("centroid_based_tangential_fraction_active"),
            "pf_inter_region_synchrony": "AVAILABLE" if isinstance(s["inter_region_synchrony"], list) else "NOT_AVAILABLE_single_or_no_region",
            "pf_within_region_frac_lag0_h1": (disp or {}).get("first_half", {}).get("frac_lag0"),
            "pf_within_region_frac_lag0_h2": (disp or {}).get("second_half", {}).get("frac_lag0"),
            "pf_within_region_frac_same_lag": (disp or {}).get("frac_same_lag_both_halves"),
            "pf_pixel_mean_mag_median": s["pixel_distribution_inside"]["median"],
            "pf_pixel_mean_mag_iqr": s["pixel_distribution_inside"]["iqr"],
            "pf_lag_resolution_s": s["lag_resolution_s"]}


@contextlib.contextmanager
def _figures(pfb, render: bool):
    """render=False skips only `figures()` (PNGs + MP4), which derive calls after summary.json is written."""
    if render:
        yield
        return
    orig = pfb.figures
    pfb.figures = lambda *a, **k: None
    try:
        yield
    finally:
        pfb.figures = orig


def analyze_pixel_first(video_path: Path, rec, mask_root: Path, output_dir: Path, meta: dict | None = None,
                        render_figures: bool = True, cv2_threads: int = 3) -> dict:
    pfb = engine("pixel_first_motion_batch")
    si = engine("cardio_organoid.pipeline.segmentation_interface")
    video_path = Path(video_path).resolve()
    row = {**base_fields(rec, video_path, meta), "pf_note": PF_NOTE}
    if rec.modality != "VIDEO":
        return {**row, "pixel_first_status": "NOT_APPLICABLE_IMAGE"}
    if rec.segmentation_qc_status != si.QC_ACCEPT:
        return {**row, "pixel_first_status": "NOT_RUN", "pixel_first_reason": f"segmentation {rec.segmentation_qc_status}"}
    obs_id = observation_id(rec)
    ws = (Path(output_dir).resolve() / obs_id / "pixel_first")
    odir = ws / "observation"
    try:
        mask = require_mask(rec, mask_root, "pixel_first")
        _, info = reference_frame(video_path, "VIDEO")
        check_mask_matches_frame(mask, info)
        rec_abs = si.SegmentationRecord.from_dict({**rec.to_dict(), "mask_path": str(resolved_mask_path(rec, mask_root))})
        man_dir = write_records([rec_abs], ws / "segmentation_interface" / "segmentation_interface_records.json").parent
        obs = {"observation_id": obs_id, "well": rec.well, "day": rec.day, "role": "standalone",
               "source_file": str(video_path), "mask_path": rec_abs.mask_path}
        import cv2
        cv2.setNumThreads(cv2_threads)
        odir.mkdir(parents=True, exist_ok=True)
        with rebind_paths(pfb, REPO_ROOT=ws, MANIFEST=man_dir), _figures(pfb, render_figures):
            prim = pfb.compute_primitives(obs, odir)
            summary = pfb.derive(obs, odir)
    except si.InterfaceFailClosed as exc:
        return {**row, "pixel_first_status": "FAILED_CLOSED", "pixel_first_reason": str(exc)}
    except Exception as exc:
        odir.mkdir(parents=True, exist_ok=True)
        (odir / "pixel_first_failed.json").write_text(json.dumps({"error": repr(exc), "traceback": traceback.format_exc()}, indent=2))
        return {**row, "pixel_first_status": "FAILED", "pixel_first_reason": repr(exc)}
    return {**row, "reference_frame": info["reference_frame"], "pixel_first_dir": str(odir),
            "figures_rendered": render_figures, **row_from_summary(summary, prim)}


def rederive_from_stored_primitives(stored_obs_dir: Path, data_root: Path, output_dir: Path, obs_meta: dict,
                                    render_figures: bool = False) -> dict:
    """Validation helper: re-run only `derive` on primitives stored by an earlier run (no video, no flow).

    `data_root` is the root the stored primitives.json paths are relative to (the original repository);
    it is only read. All outputs go to `output_dir`.
    """
    pfb = engine("pixel_first_motion_batch")
    odir = Path(output_dir).resolve()
    odir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(stored_obs_dir) / "primitives.json", odir / "primitives.json")
    with rebind_paths(pfb, REPO_ROOT=Path(data_root).resolve()), _figures(pfb, render_figures):
        summary = pfb.derive(obs_meta, odir)
    prim = json.loads((odir / "primitives.json").read_text())
    return {"summary": summary, "row": row_from_summary(summary, prim)}
