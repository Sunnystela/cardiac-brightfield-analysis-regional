"""Modality-aware analysis architecture: dry run on a handful of
representative P36-P39 samples, diagnostic-only.

Implements the IMAGE branch (morphology only, `cardio_organoid.pipeline.
image_branch`) and VIDEO branch (morphology reference-strategy
comparison via `cardio_organoid.pipeline.video_morphology_reference`,
PLUS functional analysis reusing rounds 35/36/39/41's already-built
functional pipeline UNCHANGED, orchestrated here the same way rounds
37-41's own scripts already do) against real files from the NEW
`data/total/Tracking` dataset, using the CONFIRMED longitudinal identity
(`longitudinal_id = batch + condition + well_id`,
`cardio_organoid.io.longitudinal_identity`).

Per explicit project instruction this round:
  - Does NOT analyze the full 1,698 videos / 1,081 images -- only a
    small, deliberately chosen representative sample set (see
    `DRY_RUN_SAMPLES` below): early photo-only, a transition day (both
    an image well and a video well on the SAME day), and several videos
    picked to surface the confidence-gating spread (their sync/async/
    NOT_ESTIMABLE status is DISCOVERED by round 41's pipeline here, not
    known in advance the way the old 5-validation-video set's was).
  - IMAGE observations: morphology only; every functional field stays
    NOT_MEASURED, never 0.
  - VIDEO observations: morphology (multi-strategy reference comparison)
    AND functional (round 41's confidence-gated candidate BPM, or
    NOT_ESTIMABLE -- never forced).
  - No canonical/baseline result is modified; everything computed here
    is written to a fresh results/_scratch_runs/ directory.
  - No reference-frame strategy or functional threshold is tuned to any
    manual value.

Usage:
    .venv/bin/python3 scripts/longitudinal_modality_aware_dry_run.py
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cardio_organoid.functional.cycle_aware_consensus import cycle_aware_consensus_for_group  # noqa: E402
from cardio_organoid.functional.cycle_phase_assignment import choose_anchor_time_s  # noqa: E402
from cardio_organoid.functional.fundamental_period_consensus import resolve_harmonic_scales, weighted_median  # noqa: E402
from cardio_organoid.functional.period_evidence_confidence import (  # noqa: E402
    DEFAULT_PROVISIONAL_THRESHOLDS,
    classify_confidence_state,
    evidence_bundle,
)
from cardio_organoid.functional.period_family_diagnostics import detect_period_families  # noqa: E402
from cardio_organoid.io.longitudinal_identity import (  # noqa: E402
    build_longitudinal_trajectory_summary,
    confirmed_longitudinal_identity_record,
)
from cardio_organoid.motion.reliable_hotspot_candidates import STATE_A_RELIABLE_ACTIVE, STATE_C_NOT_OBSERVABLE  # noqa: E402
from cardio_organoid.pipeline.analyze import run_motion_pipeline  # noqa: E402
from cardio_organoid.pipeline.image_branch import analyze_image_observation  # noqa: E402
from cardio_organoid.pipeline.longitudinal_schema import (  # noqa: E402
    FUNCTIONAL_STATUS_ESTIMABLE,
    FUNCTIONAL_STATUS_LOW_CONFIDENCE,
    FUNCTIONAL_STATUS_NOT_ESTIMABLE,
    MODALITY_VIDEO,
    MORPHOLOGY_STATUS_MEASURED,
    MORPHOLOGY_STATUS_SEGMENTATION_FAILED,
    aggregate_observations_to_organoid_day,
    new_observation_record,
)
from cardio_organoid.pipeline.video_morphology_reference import compare_reference_strategies
from cardio_organoid.segmentation.sam2_segmenter import get_device, load_predictor

from _output_safety import add_output_dir_args, resolve_output_dir  # noqa: E402
from global_boundary_motion_diagnostic import _load_config  # noqa: E402
import motion_observability_diagnostic as round36_module  # noqa: E402
from patch_phase_consensus_diagnostic import per_patch_events_and_features  # noqa: E402

TRACKING_ROOT = REPO_ROOT / "data" / "total" / "Tracking"
OUTPUT_DIR = REPO_ROOT / "results" / "longitudinal_modality_aware_dry_run"
PIPELINE_VERSION = "longitudinal_modality_aware_dry_run_v1"

# Deliberately small, hand-picked (never the full 1,698/1,081 set) --
# picked from round 1's inventory / round 3's CONFIRMED mapping, covering
# the categories asked for. sync/async/NOT_ESTIMABLE status for the video
# samples is DISCOVERED below, not assumed.
IMAGE_SAMPLES = [
    {"label": "early_photo_only", "batch": "P39", "condition": "Conv", "well_id": "a1", "day": "D0",
     "path": TRACKING_ROOT / "P39 07.09.2026" / "D0" / "conv" / "a1.tiff"},
    {"label": "transition_day_photo", "batch": "P38", "condition": "Conv", "well_id": "a2", "day": "D8",
     "path": TRACKING_ROOT / "P38 03.09.2026" / "D8" / "conv" / "a2.tiff"},
]
VIDEO_SAMPLES = [
    {"label": "transition_day_video", "batch": "P38", "condition": "Conv", "well_id": "a1", "day": "D8",
     "path": TRACKING_ROOT / "P38 03.09.2026" / "D8" / "conv" / "a1.avi"},
    {"label": "later_video_P39", "batch": "P39", "condition": "Conv", "well_id": "a3", "day": "D11",
     "path": TRACKING_ROOT / "P39 07.09.2026" / "D11" / "conv" / "a3.avi"},
    {"label": "first_video_day_P36", "batch": "P36", "condition": "Slit", "well_id": "a1", "day": "D13",
     "path": TRACKING_ROOT / "P36 26.08.2026" / "D13" / "slit" / "a1.avi"},
]


def _day_number(day: str) -> int:
    return int(day.lstrip("Dd"))


def _new_obs_id() -> str:
    return uuid.uuid4().hex[:12]


# --------------------------------------------------------------------------
# VIDEO functional branch: reuse rounds 35/36/39/41's UNCHANGED functions,
# orchestrated for one arbitrary NEW video (not the old 5-video VIDEOS
# list). PRODUCTION CANDIDATE channel = round 39 fundamental period +
# round 41 confidence gating (the most-validated diagnostic chain to
# date, still explicitly provisional). DIAGNOSTIC-ONLY cross-check
# channels (round 33/34/35/38/40's D10-specific deep-dive tools, block
# matching, binary movement) are NOT run here -- they are investigative
# tools for a specific already-flagged problem, not part of the routine
# per-video functional branch.
# --------------------------------------------------------------------------


def analyze_video_functional(video_path: Path, video_label: str, predictor, config: dict, output_dir: Path,
                             morphology_mask=None, mask_provenance: dict | None = None) -> dict:
    """`morphology_mask` (optional): segmentation-interface mask injected into BOTH the
    motion pipeline and the round-36 observability step, so neither re-runs SAM2 or
    picks a candidate index itself."""
    motion_dir = output_dir / "motion_pipeline" / video_label
    print(f"    [functional] running canonical motion pipeline (SAM2 + Farneback) -> {motion_dir}")
    run_motion_pipeline(video_path, predictor, config, motion_dir, morphology_mask=morphology_mask, mask_provenance=mask_provenance)

    video_cfg = {"group": "dry_run", "name": video_label, "source_video": video_path, "motion_dir": motion_dir}

    print("    [functional] round-36 observability/reliability gating ...")
    # round36's process_video's manual-region validation/labeling step
    # (never a gating input -- purely descriptive) is now genuinely
    # OPTIONAL (see manual_region_validation() in that module) -- this
    # dry-run video, from the brand-new P36-P39 dataset, has no
    # results/manual_reference/*.json entry at all, and no dummy/
    # placeholder file is needed or created here anymore.
    #
    # round36_module.process_video's own render_qc_panel() writes a PNG
    # directly into this directory without creating it -- previously
    # this "worked" only as an accidental side effect of the (now
    # removed) dummy-reference-file hack, which happened to mkdir the
    # same path first. Exposed as a real FileNotFoundError once that
    # hack was removed -- fixed here by creating the directory BEFORE
    # calling process_video, not by restoring the hack.
    round36_dir = output_dir / "round36"
    round36_dir.mkdir(parents=True, exist_ok=True)
    round36_result, *_round36_rest = round36_module.process_video(video_cfg, config, predictor, round36_dir, binary_summary=None, morphology_mask=morphology_mask)
    print(f"    [functional] manual_reference_available={round36_result['manual_region_validation']['manual_reference_available']} (informational only)")

    # Preserve the raw per-patch motion/observability evidence (never
    # dropped -- see this round's explicit instruction) alongside the
    # aggregate fields this branch derives from it.
    with open(round36_dir / f"{video_label}_round36_raw_per_patch.json", "w") as f:
        json.dump(round36_result, f, indent=2, default=str)

    return post_motion_functional(video_path, video_label, config, motion_dir, round36_result)


def post_motion_functional(video_path: Path, video_label: str, config: dict, motion_dir: Path, round36_result: dict) -> dict:
    """Rounds 35/39/40/41 aggregation + confidence gating from an already
    computed motion cache (motion_pipeline_qc.json + patch_waveforms_radial.npz
    in `motion_dir`) and the round-36 per-patch result. Reads no video and
    runs no optical flow -- lets a validated motion cache be re-aggregated
    with the current code."""
    manual_ref_available = round36_result["manual_region_validation"]["manual_reference_available"]
    per_patch36 = round36_result["per_patch"]
    states = {tuple(int(x) for x in k.split("_")): v["state"] for k, v in per_patch36.items()}
    confidences = {tuple(int(x) for x in k.split("_")): v["final_provisional_confidence"] for k, v in per_patch36.items()}
    state_counts = round36_result["state_counts"]

    print("    [functional] round-35 per-patch event extraction ...")
    import numpy as np
    qc = json.loads((motion_dir / "motion_pipeline_qc.json").read_text())
    npz = np.load(motion_dir / "patch_waveforms_radial.npz")
    waveforms = {tuple(int(p) for p in k.split("_")): npz[k] for k in npz.files}
    fps, frame_step = qc["fps"], qc["frame_step"]
    sample_interval_s = frame_step / fps
    patch_ids = sorted(waveforms)
    per_patch = {pid: per_patch_events_and_features(pid, waveforms[pid], sample_interval_s, frame_step, fps, config) for pid in patch_ids}

    reliable_patch_ids = sorted(pid for pid in patch_ids if states.get(pid) != STATE_C_NOT_OBSERVABLE)
    n_total, n_reliable = len(patch_ids), len(reliable_patch_ids)
    print(f"    [functional] reliable patches: {n_reliable}/{n_total}")

    pos_events = {pid: [e.event_time for e in per_patch[pid]["positive_events"]] for pid in reliable_patch_ids}
    neg_events = {pid: [e.event_time for e in per_patch[pid]["negative_events"]] for pid in reliable_patch_ids}
    pooled_events = {pid: per_patch[pid]["pooled_event_times_s"] for pid in reliable_patch_ids}
    freq_by_patch = {pid: per_patch[pid]["dominant_frequency_hz"] for pid in reliable_patch_ids}

    from cardio_organoid.functional.fundamental_period_consensus import patch_own_period_estimate
    candidate_table = {
        p: patch_own_period_estimate(pos_events.get(p, []), neg_events.get(p, []), freq_by_patch.get(p), waveforms.get(p), sample_interval_s)
        for p in reliable_patch_ids
    }
    own_period_by_patch = {p: candidate_table[p]["own_period_estimate_s"] for p in reliable_patch_ids}

    print("    [functional] round-39 fundamental period + round-40 family diagnostic ...")
    harmonic = resolve_harmonic_scales(reliable_patch_ids, own_period_by_patch, confidences)
    P = weighted_median([(harmonic["resolved_period_by_patch"][p], confidences.get(p, 1.0)) for p in harmonic["resolved_period_by_patch"]])
    families, _fdiag = detect_period_families(reliable_patch_ids, own_period_by_patch, confidences)
    dominant_family_fraction = (len(families[0]) / n_reliable) if families and n_reliable else None

    cfg_be = config["functional"]["beat_episodes"]
    cycle_consensus = (
        cycle_aware_consensus_for_group(reliable_patch_ids, pooled_events, P, choose_anchor_time_s(pooled_events), cfg_be["global_min_participation_fraction"])
        if P else None
    )
    cycle_cv = cycle_consensus["metrics"]["cv_ibi"] if cycle_consensus else None

    print("    [functional] round-41 evidence bundle + confidence gating ...")
    evidence = evidence_bundle(
        video_label, reliable_patch_ids, candidate_table, confidences, harmonic["applied_scale_by_patch"],
        dominant_family_fraction, n_reliable, n_total, cycle_cv,
    )
    verdict = classify_confidence_state(evidence)
    print(f"    [functional] VERDICT: {verdict['state']}  candidate_BPM={verdict['candidate_bpm']}")

    # Regional activity summary: descriptive-only, per-family BPM/size (no
    # manual region label involved) -- gives a human a spatial sense of
    # activity without asserting named-region identity.
    from cardio_organoid.functional.period_family_diagnostics import family_period_summary
    family_summaries = family_period_summary(families, own_period_by_patch, confidences)

    # observable_fraction: numerator = patches NOT in round-36 state C
    # (C_NOT_OBSERVABLE_do_not_interpret_as_no_motion); denominator =
    # every analyzed patch in the full grid (n_total). Answers "how much
    # of the organoid's own grid could even be trusted to measure motion
    # on at all" -- includes states A (reliable+active), B (likely
    # genuinely weak), and D (ambiguous), NOT just A.
    n_state_a = state_counts.get(STATE_A_RELIABLE_ACTIVE, 0)
    observable_fraction = (n_reliable / n_total) if n_total else None
    # reliable_active_fraction: numerator = patches SPECIFICALLY in state
    # A; denominator = n_total. A STRICTER subset of observable_fraction
    # (excludes B/D too) -- answers "how much of the grid is both
    # observable AND actually showing reliable activity". Computed here
    # (not copied from observable_fraction, which was this round's known
    # bug) directly from round-36's own state_counts.
    reliable_active_fraction = (n_state_a / n_total) if n_total else None

    return {
        "video_label": video_label, "video_path": str(video_path),
        "actual_fps": fps, "frame_step": frame_step,
        "n_total_patches": n_total, "n_reliable_patches": n_reliable,
        "state_counts": state_counts, "manual_reference_available": manual_ref_available,
        "observable_fraction": observable_fraction,
        "reliable_active_fraction": reliable_active_fraction,
        "fundamental_period_s": P, "fundamental_bpm": (60.0 / P) if P else None,
        "cycle_aware_consensus": cycle_consensus,
        "dominant_period_family_fraction": dominant_family_fraction,
        "n_period_families": len(families),
        "evidence": evidence, "verdict": verdict,
        "regional_activity_summary": family_summaries[:5],
    }


def build_video_observation_record(sample: dict, functional: dict, morphology_result: dict, observation_id: str) -> dict:
    longitudinal_id = f"{sample['batch']}__{sample['condition']}__{sample['well_id']}"
    verdict_state = functional["verdict"]["state"]
    functional_status = {
        "ESTIMABLE": FUNCTIONAL_STATUS_ESTIMABLE, "LOW_CONFIDENCE": FUNCTIONAL_STATUS_LOW_CONFIDENCE,
        "NOT_ESTIMABLE": FUNCTIONAL_STATUS_NOT_ESTIMABLE,
    }[verdict_state]

    # Morphology from the "current pipeline default" (midpoint frame)
    # strategy, for the single canonical morphology value in this
    # observation row -- the FULL multi-strategy comparison lives
    # separately in morphology_result (see reference_comparisons.json).
    default_strategy_morph = morphology_result["results_by_strategy"].get("midpoint_frame_current_pipeline_default", {}).get("morphology")
    morph_status = MORPHOLOGY_STATUS_MEASURED if default_strategy_morph else MORPHOLOGY_STATUS_SEGMENTATION_FAILED

    cyc_metrics = (functional["cycle_aware_consensus"] or {}).get("metrics", {})
    return new_observation_record(
        longitudinal_id=longitudinal_id, batch=sample["batch"], condition=sample["condition"], well_id=sample["well_id"],
        day=sample["day"], day_number=_day_number(sample["day"]), modality=MODALITY_VIDEO,
        source_file=str(sample["path"]), observation_id=observation_id,
        segmentation_source="sam2_candidate_index_1_midpoint_frame",
        morphology_reference_strategy="midpoint_frame_current_pipeline_default (see reference_comparisons.json for full multi-strategy comparison)",
        actual_fps=functional["actual_fps"], pipeline_version=PIPELINE_VERSION, git_commit=None,
        morphology_available=True, functional_available=True,
        morphology_status=morph_status, **(default_strategy_morph or {}),
        functional_status=functional_status,
        # classify_confidence_state already guarantees candidate_bpm=None
        # whenever state=="NOT_ESTIMABLE" -- no need to re-check here.
        bpm=functional["verdict"]["candidate_bpm"],
        beat_count=cyc_metrics.get("consensus_beat_count"), median_ibi_s=cyc_metrics.get("median_ibi_s"),
        ibi_cv=cyc_metrics.get("cv_ibi"),
        reliable_active_fraction=functional["reliable_active_fraction"], observable_fraction=functional["observable_fraction"],
        estimator_pairwise_agreement=functional["evidence"]["6_estimator_pairwise_agreement"],
        dominant_period_family_fraction=functional["dominant_period_family_fraction"],
        estimator_confidence_state=verdict_state,
        regional_activity_summary=functional["regional_activity_summary"],
        synchrony_metrics=None,  # round 38/40's beat-level activation-order stability remains diagnostic-only, not wired into this schema field yet -- see final report item F
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_output_dir_args(parser, OUTPUT_DIR, "longitudinal_modality_aware_dry_run")
    args = parser.parse_args()
    output_dir = resolve_output_dir(args, OUTPUT_DIR, "longitudinal_modality_aware_dry_run")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_config()
    sam2_cfg = config["segmentation"]["sam2"]
    device = get_device(sam2_cfg["device"])
    predictor = load_predictor(checkpoint_path=REPO_ROOT / sam2_cfg["checkpoint_path"], model_cfg=sam2_cfg["model_cfg"], device=device)

    observations = []

    print("=== IMAGE branch dry run ===")
    for sample in IMAGE_SAMPLES:
        print(f"  {sample['label']}: {sample['path'].relative_to(REPO_ROOT)}")
        frame_bgr = cv2.imread(str(sample["path"]))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        longitudinal_id = f"{sample['batch']}__{sample['condition']}__{sample['well_id']}"
        record = analyze_image_observation(
            sample["path"], frame_rgb, predictor, config,
            longitudinal_id, sample["batch"], sample["condition"], sample["well_id"], sample["day"], _day_number(sample["day"]),
            _new_obs_id(), PIPELINE_VERSION, None,
        )
        observations.append(record)
        print(f"    morphology_status={record['morphology_status']} area_px={record['area_px']} "
              f"functional_status={record['functional_status']} bpm={record['bpm']}")

    print("\n=== VIDEO branch dry run ===")
    reference_comparisons = {}
    for sample in VIDEO_SAMPLES:
        print(f"\n  {sample['label']}: {sample['path'].relative_to(REPO_ROOT)}")
        print("  -- morphology reference-strategy comparison --")
        morph_comparison = compare_reference_strategies(sample["path"], config, predictor)
        reference_comparisons[sample["label"]] = morph_comparison
        for strategy, diffs in morph_comparison["relative_diff_from_current_default"].items():
            print(f"    {strategy}: rel_diff_from_default={ {k: round(v, 4) for k, v in diffs.items()} }")

        print("  -- functional branch --")
        functional = analyze_video_functional(sample["path"], sample["label"], predictor, config, output_dir)
        record = build_video_observation_record(sample, functional, morph_comparison, _new_obs_id())
        observations.append(record)

    with open(output_dir / "observations.json", "w") as f:
        json.dump(observations, f, indent=2, default=str)
    with open(output_dir / "reference_comparisons.json", "w") as f:
        json.dump(reference_comparisons, f, indent=2, default=str)

    print("\n=== Organoid-day aggregation ===")
    organoid_day_rows = aggregate_observations_to_organoid_day(observations)
    with open(output_dir / "organoid_day_table.json", "w") as f:
        json.dump(organoid_day_rows, f, indent=2, default=str)
    for row in organoid_day_rows:
        print(f"  {row['longitudinal_id']} / {row['day']}: modality={row['modality']} "
              f"morphology_status={row['morphology_status']} functional_status={row['functional_status']} "
              f"bpm={row['bpm']}")

    print(f"\nWrote dry-run output to {output_dir}")


if __name__ == "__main__":
    main()
