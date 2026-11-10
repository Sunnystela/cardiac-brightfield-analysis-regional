"""Build docs/METRIC_DICTIONARY.csv.

Source: the code-level metric dictionary of the 2026-09-29 source audit in the original repository
(results/_scratch_runs/code_formula_source_audit_20260929T111805Z/METRIC_DICTIONARY.csv), which links
every stored column to the exact assigning line. This tool
  * splits combined rows ("a / b") into one row per output column,
  * rewrites `scripts/` paths to this project's `engine_scripts/` (byte-identical files, same line numbers)
    and re-checks the cited file hash against the standalone copy,
  * adds the columns the audit did not list (pf counts/status columns, status columns of the adapters),
  * attaches this project's validation result per column (validation/results/VALIDATION_RESULTS.csv).

Usage:
    python tools/build_metric_dictionary.py <original_repo_root>
"""
from __future__ import annotations

import csv
import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
AUDIT = "results/_scratch_runs/code_formula_source_audit_20260929T111805Z/METRIC_DICTIONARY.csv"

SPLIT = {
    "actual_fps / frame_step": ["actual_fps", "frame_step"],
    "legacy_bpm / legacy_median_ibi_s / legacy_ibi_cv": ["legacy_bpm", "legacy_median_ibi_s", "legacy_ibi_cv"],
    "pf_lag_resolution_s / pf_dt_s": ["pf_lag_resolution_s", "pf_dt_s"],
    "pf_pixel_mean_mag_median / pf_pixel_mean_mag_iqr": ["pf_pixel_mean_mag_median", "pf_pixel_mean_mag_iqr"],
    "pf_radial_fraction_centroid / pf_tangential_fraction_centroid": ["pf_radial_fraction_centroid", "pf_tangential_fraction_centroid"],
    "pf_within_region_frac_lag0_h1 / _h2 / _frac_same_lag": ["pf_within_region_frac_lag0_h1", "pf_within_region_frac_lag0_h2",
                                                              "pf_within_region_frac_same_lag"],
    "fundamental_bpm (NOT a master column; in functional_results.json / result.json)": ["fundamental_bpm"],
}

EXCL = "exclusion_counts (src/cardio_organoid/motion/pixel_first.py:27) via derive (engine_scripts/pixel_first_motion_batch.py:328)"
ADDED = [
    # output_column, group, unit, formula, qc, source
    ("pf_fps", "pixel_first", "frames/s", "cv2.CAP_PROP_FPS of the video file (file-recorded, per video)", "-",
     "compute_primitives (engine_scripts/pixel_first_motion_batch.py:146)"),
    ("pf_mask_px", "pixel_first", "px", "INSIDE_REFERENCE_MASK n_region_px = number of static-mask pixels inside the storage crop", "-", EXCL),
    ("pf_valid_px_inside", "pixel_first", "px", "mask px not in (saturated OR low_texture OR fb_inconsistent)", "invalid pixels excluded, never zero-filled", EXCL),
    ("pf_valid_px_outer_support", "pixel_first", "px", "same rule on OUTER_BOUNDARY_SUPPORT = dilate(mask,3x3,20 iter) & ~mask", "support band is reference only, never tissue", EXCL),
    ("pf_excl_saturated", "pixel_first", "px", "mask px with gray<=2 or >=253 in >=50% of every-10th frames (overlaps counted in every reason)", "-", EXCL),
    ("pf_excl_low_texture", "pixel_first", "px", "mask px with Sobel |grad| < 2.0 after Gaussian sigma 3 on the midpoint frame", "-", EXCL),
    ("pf_excl_fb_inconsistent", "pixel_first", "px", "mask px with median forward-backward error > 1.0 px/pair (every 5th pair)", "-", EXCL),
    ("pf_excl_unique", "pixel_first", "px", "mask px excluded for >=1 reason (union)", "-", EXCL),
    ("pf_excl_multiple_reasons", "pixel_first", "px", "mask px excluded for >=2 reasons", "-", EXCL),
    ("pf_single_region", "pixel_first", "bool", "pf_n_regions == 1", "-", "build_master (engine_scripts/pixel_first_motion_aggregate.py:72)"),
    ("pf_inter_region_synchrony", "pixel_first", "category", "AVAILABLE if >=2 motion-pattern regions else NOT_AVAILABLE_single_or_no_region", "-",
     "build_master (engine_scripts/pixel_first_motion_aggregate.py:72)"),
    ("pixel_first_status", "status", "category", "COMPUTED / REUSED (stored primitives of the pilot) / NOT_APPLICABLE_IMAGE / NOT_RUN / FAILED / FAILED_CLOSED",
     "FAILED rows keep pixel_first_reason and pixel_first_failed.json", "adapters/pixel_first.py"),
    ("morphology_status", "status", "category", "MEASURED / MISSING_<segmentation status> / FAILED_CLOSED",
     "non-ACCEPT or invalid record: metrics None, reason kept", "adapters/morphology.py (mirrors stage_morphology)"),
    ("calibration_status", "status", "category",
     "IMAGE: CALIBRATED_from_real_metadata_xml / NOT_CALIBRATED_no_metadata_sidecar / NOT_CALIBRATED_sidecar_present_but_no_scaling_block; "
     "VIDEO: NOT_CALIBRATED_no_metadata_source", "um columns None when not calibrated", "adapters/morphology.py:calibration_for_image"),
    ("functional_execution_status", "status", "category",
     "RUN / REAGGREGATED_FROM_CACHE (standalone); RERUN / REUSED (original first pass); NOT_RUN / NOT_MEASURED_IMAGE / FAILED / FAILED_CLOSED",
     "reason in functional_execution_reason; failed.json kept", "adapters/legacy_rhythm.py"),
    ("integrated_status", "status", "category", "RUN / NOT_RUN_INPUT_NOT_VALIDATED / NOT_RUN_INPUT_PROBLEM",
     "nothing measured unless the record passes the engine gate and the mask matches the reference frame", "adapters/integrated.py"),
]

OUT_COLS = ["metric", "group", "unit", "formula", "qc_condition", "output_column", "output_table_original", "standalone_adapter",
            "engine_function", "engine_assigning_line", "engine_file_sha256_12", "engine_file_matches_standalone", "caller_chain",
            "inputs", "spatial_aggregation", "temporal_aggregation", "denominator", "audit_verification_2026_09_29",
            "standalone_validation", "notes"]

ADAPTER = {"morphology": "adapters.analyze_morphology", "legacy_functional": "adapters.analyze_legacy_rhythm / reaggregate_legacy_rhythm",
           "pixel_first_master_legacy_copy": "adapters.integrated.legacy_columns_pixel_first_convention",
           "pixel_first": "adapters.analyze_pixel_first (row_from_summary)", "status": "see formula"}


def _remap(s: str) -> str:
    s = re.sub(r"(?<![\w/])scripts/", "engine_scripts/", s or "")
    return re.sub(r"\((p36_conv_first_pass_outputs\.py|pixel_first_motion_aggregate\.py)", r"(engine_scripts/\1", s)


def _standalone_sha12() -> dict:
    out = {}
    for p in list((PROJECT / "src").rglob("*.py")) + list((PROJECT / "engine_scripts").glob("*.py")):
        out[hashlib.sha256(p.read_bytes()).hexdigest()[:12]] = str(p.relative_to(PROJECT))
    return out


def _hash_check(audit_field: str, by_sha: dict) -> tuple[str, object]:
    """Every 12-hex hash the audit cites must belong to a byte-identical standalone file."""
    toks = re.findall(r"\b[0-9a-f]{12}\b", audit_field or "")
    if not toks:
        return "", "n/a"
    files = [by_sha.get(t) for t in toks]
    return " / ".join(f"{t} ({f or 'NOT FOUND'})" for t, f in zip(toks, files)), all(files)


def _validation() -> dict:
    p = PROJECT / "validation" / "results" / "VALIDATION_RESULTS.csv"
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    if p.exists():
        for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
            a = agg[r["metric"]][r["check"]]
            a[0] += 1
            a[1] += r["match"] == "True"
    return {m: "; ".join(f"{g}: {ok}/{n} identical" for g, (n, ok) in d.items()) for m, d in agg.items()}


def main() -> None:
    repo = Path(sys.argv[1]).resolve()
    val = _validation()
    by_sha = _standalone_sha12()
    rows = []
    for r in csv.DictReader(open(repo / AUDIT, newline="", encoding="utf-8")):
        for col in SPLIT.get(r["output_column"], [r["output_column"]]):
            func = _remap(r["calculation_function"])
            sha, sha_ok = _hash_check(r["file_sha256_12"], by_sha)
            rows.append({"metric": col, "group": r["group"], "unit": r["unit"], "formula": r["exact_formula"],
                         "qc_condition": r["qc_missing_handling"], "output_column": col,
                         "output_table_original": r["output_table"], "standalone_adapter": ADAPTER.get(r["group"], ""),
                         "engine_function": func, "engine_assigning_line": _remap(r["assigned_at"]),
                         "engine_file_sha256_12": sha, "engine_file_matches_standalone": sha_ok,
                         "caller_chain": _remap(r["caller_chain"]), "inputs": r["inputs"],
                         "spatial_aggregation": r["spatial_aggregation"], "temporal_aggregation": r["temporal_aggregation"],
                         "denominator": r["valid_samples_and_denominator"], "audit_verification_2026_09_29": r["verification"],
                         "standalone_validation": val.get(col, "not compared in this run"), "notes": r["notes"]})
    for col, grp, unit, formula, qc, src in ADDED:
        rows.append({"metric": col, "group": grp, "unit": unit, "formula": formula, "qc_condition": qc, "output_column": col,
                     "output_table_original": "MASTER_9WELL_130.csv" if col.startswith("pf_") else "standalone output",
                     "standalone_adapter": ADAPTER.get(grp, ""), "engine_function": src, "engine_assigning_line": src,
                     "engine_file_sha256_12": "", "engine_file_matches_standalone": "n/a", "caller_chain": "", "inputs": "",
                     "spatial_aggregation": "", "temporal_aggregation": "", "denominator": "",
                     "audit_verification_2026_09_29": "not in audit dictionary (added from source reading)",
                     "standalone_validation": val.get(col, "not compared in this run"), "notes": ""})
    out = PROJECT / "docs" / "METRIC_DICTIONARY.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:  # BOM: opens correctly in Excel
        w = csv.DictWriter(fh, fieldnames=OUT_COLS)
        w.writeheader()
        w.writerows(rows)
    bad = [r["metric"] for r in rows if r["engine_file_matches_standalone"] is False]
    print(f"{len(rows)} rows -> {out}; hash mismatches vs audit: {bad}")


if __name__ == "__main__":
    main()
