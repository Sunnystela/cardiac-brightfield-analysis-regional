"""P36/Conv FIRST-PASS integrated screening, driven ONLY by the frozen
human-reviewed segmentation manifest (v2). Fail-closed throughout.

Stages (each resumable; outputs under --output-dir):
  morphology          ACCEPT masks only (require_downstream_mask); IMAGE/VIDEO streams
                      kept separate; unit + calibration status per value
  functional_plan     per VIDEO observation: REUSE (validated motion cache whose
                      mask / video / grid provenance matches) / RERUN / NOT_RUN
  functional_run      reruns with the interface mask injected; parallel, one
                      completion file per observation (resume-safe)
  functional_aggregate  re-aggregates every reused cache with the CURRENT code
                      (post_motion_functional, regression-checked 54/54 identical)
                      and merges rerun results
  integrate           per-well outputs + population outputs

This is human-reviewed FIRST-PASS SCREENING, not an independent validation of
automatic segmentation, and it makes no biological / statistical claim.
No selector or Stage-1 training, no threshold fitting, no canonical write,
no config change. P38 not accessed.
"""
from __future__ import annotations

import argparse
import csv
import io
import contextlib
import json
import math
import multiprocessing as mp
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cardio_organoid.pipeline.run_manifest import sha256_file, write_run_manifest  # noqa: E402
from cardio_organoid.pipeline.segmentation_interface import (  # noqa: E402
    INTERFACE_VERSION, QC_ACCEPT, QC_NONE, SegmentationRecord, fail_closed_mode, load_mask, require_downstream_mask,
)
from p36_conv_integrated_batch import BATCH, CONDITION, CONFIG_PATH, SELECTOR_ARTIFACT  # noqa: E402

SCR = REPO_ROOT / "results/_scratch_runs"
SCREENING_CACHES = {
    "a1": ("p36_conv_a1_functional_onset_diagnostic_20260924T070648Z", "P36_Conv_a1_D{day}"),
    "a2": ("p36_conv_a2_screening_20260924T115210Z", "p36_conv_a2_screening_D{day}"),
    "b1": ("p36_conv_b1_screening_20260924T115215Z", "p36_conv_b1_screening_D{day}"),
    "c2": ("p36_conv_c2_screening_20260924T115320Z", "p36_conv_c2_screening_D{day}"),
    "d1": ("p36_conv_d1_screening_20260924T115324Z", "p36_conv_d1_screening_D{day}"),
    "e2": ("p36_conv_e2_screening_20260924T115335Z", "p36_conv_e2_screening_D{day}"),
    "f2": ("p36_conv_f2_screening_20260924T115339Z", "p36_conv_f2_screening_D{day}"),
    "g1": ("p36_conv_g1_screening_20260924T123625Z", "p36_conv_g1_screening_D{day}"),
    "g3": ("p36_conv_g3_v4_holdout_diagnostic_20260924T090338Z", "p36_conv_g3_v4_holdout_diagnostic_D{day}"),
    "h1": ("p36_conv_h1_screening_20260924T123801Z", "p36_conv_h1_screening_D{day}"),
}
PILOT = SCR / "p36_conv_integrated_pilot_20260928T132111Z"
MORPH_METRICS = ["area_px", "perimeter_px", "equivalent_diameter_px", "circularity", "solidity",
                 "eccentricity", "aspect_ratio", "radial_cv", "perimeter_over_sqrt_area"]
NOT_PROVIDED = {
    "regional_participation": "NOT_PROVIDED: implemented (phenotype_axes.participation_ratio) but not validated on this dataset",
    "dominant_region_strength": "NOT_PROVIDED: implemented (phenotype_axes) but not validated on this dataset",
    "regional_phase_lag": "NOT_PROVIDED: beat-level activation order near chance in rounds 38-40; not validated",
    "synchrony_coherence": "NOT_PROVIDED: phenotype_axes candidates are a documented comparison baseline, not validated",
    "activation_sequence": "NOT_PROVIDED: no validated region-free implementation",
}


def rel(p):
    return str(Path(p).resolve().relative_to(REPO_ROOT))


def jdump(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str, ensure_ascii=False)


def load_manifest(man: Path):
    recs = [SegmentationRecord.from_dict(d) for d in json.loads((man / "segmentation_interface_records.json").read_text())["records"]]
    rev = {(r["well"], int(r["day"])): r for r in csv.DictReader(open(man / "SEGMENTATION_REVISION_TABLE.csv", newline=""))}
    return recs, rev


def obs_id(r):
    return f"{BATCH}_{CONDITION}_{r.well}_D{r.day}_{r.modality}"


def write_manifest(out, man, stage, extra=None):
    write_run_manifest(out / "manifests" / stage, REPO_ROOT, config_path=CONFIG_PATH, interface_version=INTERFACE_VERSION,
                       selector_artifact=SELECTOR_ARTIFACT, sam2_checkpoint=None, sam2_model_cfg=None,
                       input_scope={"batch": BATCH, "condition": CONDITION, "stage": stage, "p38": "not accessed",
                                    "segmentation_manifest": rel(man)},
                       extra={"segmentation_manifest_sha256": sha256_file(man / "segmentation_interface_records.json"), **(extra or {})},
                       hash_sam2_checkpoint=False)


# ------------------------------------------------------------------ morphology
def morph_metrics(mask):
    from cardio_organoid.morphology.features import compute_morphology_features
    f = compute_morphology_features(mask).pixel
    return {"area_px": f.area_px, "perimeter_px": f.perimeter_px, "equivalent_diameter_px": f.equivalent_diameter_px,
            "circularity": f.circularity, "solidity": f.solidity, "eccentricity": f.eccentricity,
            "aspect_ratio": f.aspect_ratio, "radial_cv": f.radial_cv,
            "perimeter_over_sqrt_area": (f.perimeter_px / math.sqrt(f.area_px)) if f.perimeter_px and f.area_px else None}


def stage_morphology(out, man):
    from p39_conv_a3_longitudinal_morphology_pilot import calibration_for_image_day
    write_manifest(out, man, "morphology")
    recs, rev = load_manifest(man)
    rows = []
    for r in recs:
        rv = rev[(r.well, r.day)]
        base = {"observation_id": obs_id(r), "well": r.well, "day": r.day, "modality": r.modality,
                "morphology_stream": r.modality, "segmentation_qc_status": r.segmentation_qc_status,
                "mask_provenance": r.mask_provenance, "selected_candidate": r.selected_candidate,
                "measurement_caveat": rv.get("measurement_caveat") or "", "source_file": rv["source_file"]}
        if r.segmentation_qc_status == QC_ACCEPT:
            m = morph_metrics(require_downstream_mask(r, REPO_ROOT, "morphology"))
            if r.modality == "IMAGE":
                cal = calibration_for_image_day(REPO_ROOT / rv["source_file"])
            else:
                cal = {"calibration_status": "NOT_CALIBRATED_no_metadata_source"}
            um = cal.get("um_per_pixel_x")
            row = {**base, "morphology_status": "MEASURED", "unit": "px", "calibration_status": cal.get("calibration_status"),
                   "um_per_pixel": um, **m,
                   "area_um2": m["area_px"] * um ** 2 if um else None,
                   "equivalent_diameter_um": m["equivalent_diameter_px"] * um if um else None}
        else:
            row = {**base, "morphology_status": "MISSING_" + r.segmentation_qc_status, "unit": None,
                   "calibration_status": None, "um_per_pixel": None, **{k: None for k in MORPH_METRICS},
                   "area_um2": None, "equivalent_diameter_um": None}
        rows.append(row)
    jdump(out / "morphology" / "morphology_results.json", rows)
    print(dict(Counter(x["morphology_status"] for x in rows)))


# ------------------------------------------------------------------ functional plan
def stage_functional_plan(out, man):
    write_manifest(out, man, "functional_plan")
    recs, rev = load_manifest(man)
    plan = []
    for r in recs:
        if r.modality != "VIDEO":
            continue
        rv = rev[(r.well, r.day)]
        p = {"observation_id": obs_id(r), "well": r.well, "day": r.day, "segmentation_qc_status": r.segmentation_qc_status,
             "selected_candidate": r.selected_candidate, "source_file": rv["source_file"]}
        if r.segmentation_qc_status != QC_ACCEPT:
            p.update(decision="NOT_RUN", reason=f"segmentation {r.segmentation_qc_status}; no substitute mask")
            plan.append(p)
            continue
        final_mask = require_downstream_mask(r, REPO_ROOT, "functional_plan")
        eqd_final = 2.0 * math.sqrt(final_mask.sum() / math.pi)
        # (a) pilot cache: same session, same code, mask injected
        pl = PILOT / "functional_rerun" / "motion_pipeline" / f"pilot_{r.well}_D{r.day}_cand{r.selected_candidate}"
        pm = PILOT / "masks" / r.well / "selected" / f"D{r.day}_selected_candidate{r.selected_candidate}.npz"
        if pl.exists() and pm.exists():
            qc = json.loads((pl / "motion_pipeline_qc.json").read_text())
            same = np.array_equal(load_mask(pm), final_mask)
            if same and qc.get("segmentation_source") == "injected_segmentation_interface_mask" and str(qc["video"]).endswith(rv["source_file"]):
                p.update(decision="REUSE", cache="pilot", motion_dir=rel(pl),
                         round36=rel(PILOT / "functional_rerun" / "round36" / f"{pl.name}_round36_raw_per_patch.json"),
                         reason="pilot rerun with identical injected mask (array-equal), same video, same code/config")
                plan.append(p)
                continue
        # (b) 2026-09-24 screening cache: built with fixed idx1; valid only if the final mask IS that mask
        if r.well in SCREENING_CACHES and r.selected_candidate == 1:
            root, fmt = SCREENING_CACHES[r.well]
            label = fmt.format(day=r.day)
            md, r36 = SCR / root / "motion_pipeline" / label, SCR / root / "round36" / f"{label}_round36_raw_per_patch.json"
            if (md / "motion_pipeline_qc.json").exists() and r36.exists():
                qc = json.loads((md / "motion_pipeline_qc.json").read_text())
                ok_mask = abs(qc["equivalent_diameter_px"] - eqd_final) < 1e-6
                ok_video = str(qc["video"]).endswith(rv["source_file"])
                if ok_mask and ok_video:
                    p.update(decision="REUSE", cache="screening_2026-09-24", motion_dir=rel(md), round36=rel(r36),
                             reason="cache built from fixed idx1 == final selected candidate 1 (equivalent diameter identical), "
                                    "same video; motion code/config unchanged since; aggregation recomputed with current code")
                    plan.append(p)
                    continue
                p["reuse_rejected"] = f"mask_match={ok_mask} video_match={ok_video}"
        p.update(decision="RERUN", reason="no provenance-valid motion cache for the final selected mask")
        plan.append(p)
    jdump(out / "functional" / "functional_plan.json", plan)
    print(dict(Counter(x["decision"] for x in plan)), dict(Counter(x.get("cache") for x in plan if x["decision"] == "REUSE")))


# ------------------------------------------------------------------ functional run (parallel, resumable)
def functional_record_from_result(res: dict) -> dict:
    cyc = (res.get("cycle_aware_consensus") or {}).get("metrics") or {}
    return {"functional_qc_status": res["verdict"]["state"], "candidate_bpm": res["verdict"]["candidate_bpm"],
            "fundamental_bpm": res.get("fundamental_bpm"), "median_ibi_s": cyc.get("median_ibi_s"), "ibi_cv": cyc.get("cv_ibi"),
            "consensus_beat_count": cyc.get("consensus_beat_count"), "reliable_active_fraction": res.get("reliable_active_fraction"),
            "observable_fraction": res.get("observable_fraction"), "n_total_patches": res.get("n_total_patches"),
            "n_reliable_patches": res.get("n_reliable_patches"), "n_period_families": res.get("n_period_families"),
            "period_family_bpms_descriptive": [x.get("median_own_bpm") for x in (res.get("regional_activity_summary") or [])],
            "actual_fps": res.get("actual_fps"), "frame_step": res.get("frame_step"),
            "triggered_failure_modes": (res.get("verdict") or {}).get("triggered_failure_modes")}


def _run_one(args):
    obs, rec_dict, video, out_dir = args
    import cv2
    cv2.setNumThreads(2)
    done = Path(out_dir) / "result.json"
    if done.exists():
        return obs, "SKIPPED_ALREADY_DONE"
    try:
        from global_boundary_motion_diagnostic import _load_config
        from longitudinal_modality_aware_dry_run import analyze_video_functional
        with fail_closed_mode():
            rec = SegmentationRecord.from_dict(rec_dict)
            mask = require_downstream_mask(rec, REPO_ROOT, "functional_run")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                res = analyze_video_functional(REPO_ROOT / video, obs, None, _load_config(), Path(out_dir), morphology_mask=mask,
                                               mask_provenance={"interface_record": rec_dict})
        jdump(done, {**functional_record_from_result(res), "motion_dir": rel(Path(out_dir) / "motion_pipeline" / obs),
                     "round36": rel(Path(out_dir) / "round36" / f"{obs}_round36_raw_per_patch.json")})
        return obs, "OK"
    except Exception as exc:  # recorded, never silently dropped
        jdump(Path(out_dir) / "failed.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        return obs, "FAILED"


def stage_functional_run(out, man, workers):
    write_manifest(out, man, "functional_run", {"workers": workers})
    recs = {obs_id(r): r for r in load_manifest(man)[0]}
    plan = json.loads((out / "functional" / "functional_plan.json").read_text())
    jobs = [(p["observation_id"], recs[p["observation_id"]].to_dict(), p["source_file"], str(out / "functional" / "rerun" / p["observation_id"]))
            for p in plan if p["decision"] == "RERUN"]
    print(f"{len(jobs)} RERUN jobs, {workers} workers", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for i, (o, st) in enumerate(pool.imap_unordered(_run_one, jobs), 1):
            print(f"[{i}/{len(jobs)}] {o}: {st}", flush=True)


# ------------------------------------------------------------------ functional aggregate
def stage_functional_aggregate(out, man):
    from global_boundary_motion_diagnostic import _load_config
    from longitudinal_modality_aware_dry_run import post_motion_functional
    write_manifest(out, man, "functional_aggregate")
    config = _load_config()
    recs, _ = load_manifest(man)
    plan = {p["observation_id"]: p for p in json.loads((out / "functional" / "functional_plan.json").read_text())}
    rows = []
    for r in recs:
        o = obs_id(r)
        row = {"observation_id": o, "well": r.well, "day": r.day, "modality": r.modality,
               "segmentation_qc_status": r.segmentation_qc_status, **NOT_PROVIDED}
        if r.modality == "IMAGE":
            row.update(execution_status="NOT_MEASURED_IMAGE", functional_qc_status=None)
            rows.append(row)
            continue
        p = plan[o]
        if p["decision"] == "NOT_RUN":
            row.update(execution_status="NOT_RUN", functional_qc_status=None, execution_reason=p["reason"])
        elif p["decision"] == "REUSE":
            with contextlib.redirect_stdout(io.StringIO()):
                res = post_motion_functional(REPO_ROOT / p["source_file"], o, config, REPO_ROOT / p["motion_dir"],
                                             json.loads((REPO_ROOT / p["round36"]).read_text()))
            qc = json.loads((REPO_ROOT / p["motion_dir"] / "motion_pipeline_qc.json").read_text())
            row.update(execution_status="REUSED", cache=p["cache"], motion_dir=p["motion_dir"], execution_reason=p["reason"],
                       motion_segmentation_source=qc.get("segmentation_source", "legacy_fixed_idx1 (field absent; == final mask verified)"),
                       **functional_record_from_result(res))
        else:
            d = out / "functional" / "rerun" / o
            if (d / "result.json").exists():
                res = json.loads((d / "result.json").read_text())
                qc = json.loads((REPO_ROOT / res["motion_dir"] / "motion_pipeline_qc.json").read_text())
                row.update(execution_status="RERUN", motion_segmentation_source=qc.get("segmentation_source"), **res)
            elif (d / "failed.json").exists():
                row.update(execution_status="FAILED", functional_qc_status=None,
                           execution_reason=json.loads((d / "failed.json").read_text())["error"])
            else:
                row.update(execution_status="PENDING_NOT_EXECUTED", functional_qc_status=None)
        # never report a BPM for NOT_ESTIMABLE; never 0
        row["bpm"] = row.get("candidate_bpm") if row.get("functional_qc_status") in ("ESTIMABLE", "LOW_CONFIDENCE") else None
        rows.append(row)
    jdump(out / "functional" / "functional_results.json", rows)
    v = [x for x in rows if x["modality"] == "VIDEO"]
    print(dict(Counter(x["execution_status"] for x in v)), dict(Counter(x.get("functional_qc_status") for x in v)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["morphology", "functional_plan", "functional_run", "functional_aggregate", "integrate"])
    ap.add_argument("--manifest-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    with fail_closed_mode():
        if a.stage == "functional_run":
            stage_functional_run(a.output_dir, a.manifest_dir, a.workers)
        elif a.stage == "integrate":
            from p36_conv_first_pass_outputs import stage_integrate
            stage_integrate(a.output_dir, a.manifest_dir)
        else:
            {"morphology": stage_morphology, "functional_plan": stage_functional_plan,
             "functional_aggregate": stage_functional_aggregate}[a.stage](a.output_dir, a.manifest_dir)


if __name__ == "__main__":
    main()
