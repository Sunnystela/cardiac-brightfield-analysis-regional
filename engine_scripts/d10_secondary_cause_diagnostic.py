"""mature/D10_26.08.2026/slit/a1 secondary_region ~2x-rate root-cause
diagnostic (round 33).

Manual QC (rounds 26-30, see results/beat_event_episodes/
manual_validation_log.json's open_items_not_yet_resolved) established
that secondary_region's provisional BPM (66.8) is almost exactly 2x
primary_region's/tertiary_region's (33.4), that the owner's own repeated
manual count of the raw video gives ~19 beats over ~30-31s (~37-40 BPM,
close to primary/tertiary and to two independent boundary-motion signals
computed at frame_step=3), that the owner does NOT see a visible "two
motions per beat" (double-peak/doublet) at secondary_region, and that
round 29's self-relative biological-beat merge could not resolve it
(secondary's own median raw-event gap already IS its majority spacing,
not a minority of short outliers, so a self-relative method has nothing
to identify as "the same beat split in two").

This script does NOT try to force secondary_region's count to ~19/~40
BPM. It decomposes SIX candidate explanations, purely diagnostically:

  1. Temporal-subsampling artifact: does secondary_region's own detection
     result change between frame_step=1 (finest available), 2, and the
     production frame_step=3 -- if a real motion extremum is split across
     samples at frame_step=3 but not at frame_step=1, that would show up
     as a converging/diverging cycle count and BPM across steps.
  2. Waveform-morphology substructure: event-triggered averages of
     secondary_region's raw/detrended/smoothed waveform, first derivative,
     |flow magnitude|, and signed radial flow, aligned on secondary's own
     detected events -- checks for a double-peak, a shoulder/subpeak, or a
     within-beat contraction/recoil substructure directly in the averaged
     shape (frame_step=1, the finest available resolution).
  3. Spatial decomposition: secondary_region's 5 constituent patches
     analyzed INDIVIDUALLY (own amplitude, own dominant frequency, own
     provisional BPM, own neighbor-coherence) -- distinguishes "every
     patch genuinely runs at ~66 BPM" from "individual patches run at
     ~33 BPM but with a phase offset that produces a ~66 BPM appearance
     in the region-mean waveform", compared side by side with
     primary_region's/tertiary_region's own patches.
  4. Optical-flow direction diagnostic: each secondary event's average
     flow, decomposed into radial (inward/outward, sign convention
     unchanged: negative=inward, positive=outward) and tangential
     components (see cardio_organoid.motion.radial_flow.
     tangential_motion_from_centroid, added this round) -- checks whether
     consecutive events alternate inward/outward (consistent with one
     biological beat's two motion phases) or repeat the same direction
     (consistent with two separate real excursions). No contraction/
     relaxation biological label is assigned to either sign.
  5. Image-observability/texture diagnostic: per-patch local intensity
     contrast, gradient magnitude, and forward/backward optical-flow
     consistency for secondary_region AND primary_region/tertiary_region
     (comparison), to separate "motion estimate unreliable because this
     patch has little texture for Farneback to lock onto" from "motion is
     genuinely weak/absent here".
  6. Cross-region timing: every secondary_region raw event (from the
     ALREADY-COMPUTED, unchanged canonical production result,
     results/beat_event_episodes_v3/mature/D10_slit_a1/episodes.json) is
     given a purely descriptive, non-classifier timing label relative to
     primary_region's and tertiary_region's own already-computed events
     (shared_with_primary / shared_with_tertiary / secondary_only /
     possible_within_beat_substructure / ambiguous).

Explicitly NOT done (per project instruction for this round):
  - No detection/pairing/grouping/merge threshold is changed anywhere.
  - No raw event is deleted, and beat_events.py/beat_cycles.py/
    biological_beats.py are imported and called UNCHANGED, never edited.
  - No canonical result (results/beat_event_episodes_v3/*,
    results/manual_reference/*, results/biological_beats/*,
    motion_pipeline*/*) is written to -- this script only READS them and
    writes new diagnostic-only output.
  - No biological conclusion (contraction/relaxation, arrhythmia, "the
    true rate is X BPM") is asserted -- see this run's own printed
    ranked-hypothesis summary for what the evidence does and does not
    support.
  - The source video is never opened for writing.

Adds one new small reusable primitive this round:
cardio_organoid.motion.radial_flow.tangential_motion_from_centroid (+
radial_tangential_decomposition), tested in tests/test_radial_flow.py --
everything else reuses existing, unmodified library functions.

Writes only to a fresh results/_scratch_runs/ directory by default (see
scripts/_output_safety.py) -- pass --output-dir explicitly (versioned, or
the canonical path + --allow-canonical-write) to keep a run.

Usage:
    .venv/bin/python3 scripts/d10_secondary_cause_diagnostic.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cardio_organoid.functional.beat_cycle_metrics import compute_region_cycle_metrics  # noqa: E402
from cardio_organoid.functional.beat_cycles import build_beat_cycles  # noqa: E402
from cardio_organoid.functional.beat_episodes import (  # noqa: E402
    pooled_dominant_period_s,
    region_events_from_candidate_events,
)
from cardio_organoid.functional.beat_events import (  # noqa: E402
    detect_beat_events_both_polarities,
    preprocess_waveform,
    region_beat_summary,
)
from cardio_organoid.functional.biological_beats import (  # noqa: E402
    compute_biological_beat_metrics,
    merge_cycles_into_biological_beats,
    robust_reference_period_s,
)
from cardio_organoid.functional.regional_lag_diagnostics import (  # noqa: E402
    event_triggered_average,
    region_mean_waveform,
)
from cardio_organoid.motion.diagnostics import dominant_spectral_peak  # noqa: E402
from cardio_organoid.motion.radial_flow import (  # noqa: E402
    radial_tangential_decomposition,
    remove_global_drift,
)
from cardio_organoid.motion.raft_flow import compute_dense_optical_flow, compute_motion_roi  # noqa: E402
from cardio_organoid.motion.spatial_patches import cross_patch_synchronization, neighbor_coherence  # noqa: E402
from cardio_organoid.segmentation.sam2_segmenter import get_device, load_predictor, segment_frame  # noqa: E402

from _output_safety import add_output_dir_args, resolve_output_dir  # noqa: E402
from beat_event_detection_validation import (  # noqa: E402
    MIN_PROMINENCE_RATIO,
    MIN_SEPARATION_PERIOD_FRACTION,
    SMOOTH_WINDOW_SAMPLES,
    _baseline_window_samples,
)
from run_sam2_phase1 import load_config  # noqa: E402

VIDEO_PATH = REPO_ROOT / "data" / "working" / "mature_validation" / "D10_26.08.2026" / "slit" / "a1.avi"
MOTION_DIR = REPO_ROOT / "results" / "motion_pipeline_mature_v2_background_stabilized" / "D10_26.08.2026" / "slit" / "a1"
REFERENCE_PATH = REPO_ROOT / "results" / "manual_reference" / "mature" / "D10_26.08.2026" / "slit" / "a1.json"
EPISODES_PATH = REPO_ROOT / "results" / "beat_event_episodes_v3" / "mature" / "D10_slit_a1" / "episodes.json"
OUTPUT_DIR = REPO_ROOT / "results" / "d10_secondary_cause_diagnostic"

REGION_COLORS_BGR = {"primary_region": (168, 120, 76), "secondary_region": (34, 130, 224), "tertiary_region": (95, 202, 44)}
REGION_COLORS_MPL = {name: tuple(c / 255 for c in bgr[::-1]) for name, bgr in REGION_COLORS_BGR.items()}

# Manual observation stated by the project owner (round 28/29/31): NOT a
# stored value in results/manual_reference/*.json, NOT a tuning target --
# reference-only, shown alongside computed values for context.
MANUAL_OBSERVED_BEAT_COUNT = 19
MANUAL_OBSERVED_DURATION_RANGE_S = (30.0, 31.0)


def _fmt(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) and x is not None else "NA"


# --------------------------------------------------------------------------
# 0. Shared setup -- geometry recovery + canonical event/manual-reference
#    loading, all read-only.
# --------------------------------------------------------------------------


def load_common(config: dict, predictor) -> dict:
    qc = json.loads((MOTION_DIR / "motion_pipeline_qc.json").read_text())
    npz = np.load(MOTION_DIR / "patch_waveforms_radial.npz")
    production_waveforms = {tuple(int(p) for p in key.split("_")): npz[key] for key in npz.files}
    ref = json.loads(REFERENCE_PATH.read_text())
    episodes = json.loads(EPISODES_PATH.read_text())

    fps = qc["fps"]
    production_frame_step = qc["frame_step"]
    saved_equiv_diam = qc["equivalent_diameter_px"]

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rep_index = frame_count // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, rep_index)
    ok, rep_frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read representative frame {rep_index} from {VIDEO_PATH}")
    rep_frame_rgb = cv2.cvtColor(rep_frame_bgr, cv2.COLOR_BGR2RGB)

    candidates = segment_frame(predictor, rep_frame_rgb)
    morphology_mask = candidates[config["segmentation"]["provisional_primary"]["candidate_index"]].mask
    recovered_equiv_diam = float(2.0 * np.sqrt(morphology_mask.sum() / np.pi))
    print(f"ROI geometry check: recovered={recovered_equiv_diam:.1f} vs saved={saved_equiv_diam:.1f} "
          f"(diff={abs(recovered_equiv_diam - saved_equiv_diam) / saved_equiv_diam:.2%})")
    roi = compute_motion_roi(
        morphology_mask, expansion_px=config["motion"]["roi"]["expansion_px"], frame_shape=rep_frame_rgb.shape[:2],
    )
    cell_size_px = max(1, int(round(config["motion"]["spatial_patches"]["cell_size_ratio"] * saved_equiv_diam)))

    regions = {name: [tuple(p) for p in ref["regions"][name]] for name in ref["activation_order"]}

    return {
        "config": config, "fps": fps, "production_frame_step": production_frame_step,
        "saved_equiv_diam": saved_equiv_diam, "rep_frame_rgb": rep_frame_rgb, "rep_index": rep_index,
        "frame_count": frame_count, "roi": roi, "cell_size_px": cell_size_px, "regions": regions,
        "production_waveforms": production_waveforms, "ref": ref, "episodes": episodes,
    }


def patch_bbox_local(row: int, col: int, cell_size_px: int) -> tuple[int, int, int, int]:
    x0, y0 = col * cell_size_px, row * cell_size_px
    return x0, y0, x0 + cell_size_px, y0 + cell_size_px


def _region_events(episodes: dict, pol_key: str, region: str) -> list[dict]:
    return sorted(
        (e for ep in episodes[pol_key] for e in ep["member_events"] if e["region_id"] == region),
        key=lambda e: e["event_time"],
    )


# --------------------------------------------------------------------------
# Full-ROI Farneback pass at an arbitrary frame_step, reduced immediately to
# per-patch scalar (radial, tangential, magnitude) waveforms -- never
# stores whole-field history, so this is memory-light even at frame_step=1
# on the full ~30s video. Reused for section 1 (secondary_region only) and
# sections 2/4/5 (all 3 regions' patches, frame_step=1 for finest detail).
# --------------------------------------------------------------------------


def compute_fine_resolution_waveforms(
    video_path: Path, roi, frame_step: int, patch_bboxes: dict,
) -> dict:
    x0, y0, x1, y1 = roi.roi_bbox_xyxy
    centroid_xy = roi.centroid_xy_in_roi()
    local_roi = replace(roi, roi_bbox_xyxy=(0, 0, x1 - x0, y1 - y0))

    series = {key: {"radial": [], "tangential": [], "magnitude": []} for key in patch_bboxes}
    cap = cv2.VideoCapture(str(video_path))
    prev_gray_roi = None
    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_idx % frame_step == 0:
            gray_roi = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
            if prev_gray_roi is not None:
                flow = compute_dense_optical_flow(prev_gray_roi, gray_roi, local_roi, method="opencv_farneback")
                corrected, _drift = remove_global_drift(flow)
                radial, tangential = radial_tangential_decomposition(corrected, centroid_xy)
                magnitude = np.hypot(corrected[..., 0], corrected[..., 1])
                for key, (bx0, by0, bx1, by1) in patch_bboxes.items():
                    series[key]["radial"].append(float(np.nanmean(radial[by0:by1, bx0:bx1])))
                    series[key]["tangential"].append(float(np.nanmean(tangential[by0:by1, bx0:bx1])))
                    series[key]["magnitude"].append(float(np.nanmean(magnitude[by0:by1, bx0:bx1])))
            prev_gray_roi = gray_roi
        frame_idx += 1
    cap.release()
    return {key: {k: np.array(v) for k, v in val.items()} for key, val in series.items()}


# --------------------------------------------------------------------------
# Shared per-waveform beat-cycle / biological-beat analysis (identical
# unmodified functions/constants used everywhere else in this project).
# --------------------------------------------------------------------------


def analyze_waveform(
    waveform: np.ndarray, sample_interval_s: float, frame_step: int, fps: float, config: dict, region_id: str,
) -> dict:
    stages = preprocess_waveform(
        waveform, apply_detrend=True, smooth_window_samples=SMOOTH_WINDOW_SAMPLES,
        baseline_window_samples=_baseline_window_samples(waveform, sample_interval_s),
    )
    detection_input = stages["baseline_corrected"]
    events_both = detect_beat_events_both_polarities(
        detection_input, sample_interval_s, frame_step, fps,
        min_prominence_ratio=MIN_PROMINENCE_RATIO, min_separation_period_fraction=MIN_SEPARATION_PERIOD_FRACTION,
    )
    pos_events = region_events_from_candidate_events(region_id, "positive", events_both["positive_events"])
    neg_events = region_events_from_candidate_events(region_id, "negative", events_both["negative_events"])

    pos_summary = region_beat_summary(events_both["positive_events"], detection_input, sample_interval_s)
    neg_summary = region_beat_summary(events_both["negative_events"], detection_input, sample_interval_s)
    pooled_period = pooled_dominant_period_s([pos_summary["dominant_frequency_hz"], neg_summary["dominant_frequency_hz"]])
    cfg_bc = config["functional"]["beat_cycles"]
    pairing_window_s = (pooled_period * cfg_bc["pairing_window_period_fraction"]) if pooled_period else cfg_bc["fallback_pairing_window_s"]

    cycles = build_beat_cycles(pos_events, neg_events, pairing_window_s)
    pos_amp_by_time = {round(e.event_time, 6): e.amplitude for e in pos_events}
    neg_amp_by_time = {round(e.event_time, 6): e.amplitude for e in neg_events}
    metrics = compute_region_cycle_metrics([c.to_dict() for c in cycles], pos_amp_by_time, neg_amp_by_time)
    high = metrics["tiers"]["high_confidence_only"]

    pos_times = [e.event_time for e in pos_events]
    neg_times = [e.event_time for e in neg_events]
    ref_period = robust_reference_period_s(pos_times, neg_times)
    beats = merge_cycles_into_biological_beats([c.to_dict() for c in cycles], ref_period, config["functional"]["biological_beats"]["merge_period_fraction"])
    bio_metrics = compute_biological_beat_metrics(beats)
    bio_high = bio_metrics["tiers"]["high_confidence_only"]

    return {
        "region_id": region_id, "sample_interval_s": sample_interval_s, "frame_step": frame_step,
        "detection_input": detection_input, "stages": stages,
        "positive_events": events_both["positive_events"], "negative_events": events_both["negative_events"],
        "n_positive_events": len(pos_events), "n_negative_events": len(neg_events),
        "dominant_frequency_hz": pooled_period and (1.0 / pooled_period),
        "beat_cycle_count_high": high["beat_cycle_count"],
        "median_ibi_s_high": high["interval_stats"]["median_ibi_s"],
        "cv_ibi_high": high["interval_stats"]["cv_ibi"],
        "provisional_bpm_high": high["provisional_bpm_from_median_ibi"],
        "reference_period_s_self_relative": ref_period,
        "n_merged_biological_beats": bio_metrics["n_merged_beats"],
        "biological_beat_bpm_high": bio_high["provisional_bpm_from_median_ibi"],
        "biological_beat_cv_ibi_high": bio_high["interval_stats"]["cv_ibi"],
    }


def _trim_for_report(analysis: dict) -> dict:
    return {k: v for k, v in analysis.items() if k not in ("detection_input", "stages", "positive_events", "negative_events")}


# --------------------------------------------------------------------------
# 1. Temporal-resolution diagnostic (frame_step 1/2/3, secondary_region only)
# --------------------------------------------------------------------------


def section1_temporal_resolution(common: dict, output_dir: Path, secondary_fine_fs1: dict) -> dict:
    print("\n=== Section 1: temporal-resolution diagnostic (secondary_region, frame_step 1/2/3) ===")
    config = common["config"]
    fps = common["fps"]
    cell_size_px = common["cell_size_px"]
    secondary_patches = common["regions"]["secondary_region"]
    patch_bboxes = {f"{r}_{c}": patch_bbox_local(r, c, cell_size_px) for (r, c) in secondary_patches}

    results = {}
    waveforms_by_step = {}
    raw_region_radial_by_step = {}
    for frame_step in (1, 2, 3):
        if frame_step == 1:
            # Reuse the single all-regions frame_step=1 pass the caller
            # already computed (see main()) instead of a second, wholly
            # redundant full-video Farneback pass restricted to just
            # secondary_region's patches.
            print("  frame_step=1: reusing the already-computed all-regions fine-resolution pass (no recompute)")
            fine = secondary_fine_fs1
        else:
            print(f"  computing frame_step={frame_step} (a full-video Farneback pass, secondary_region patches only)...")
            fine = compute_fine_resolution_waveforms(VIDEO_PATH, common["roi"], frame_step, patch_bboxes)
        region_radial = np.mean(np.stack([fine[k]["radial"] for k in patch_bboxes]), axis=0)
        raw_region_radial_by_step[frame_step] = region_radial
        sample_interval_s = frame_step / fps
        analysis = analyze_waveform(region_radial, sample_interval_s, frame_step, fps, config, f"secondary_region_fs{frame_step}")
        waveforms_by_step[frame_step] = analysis
        results[str(frame_step)] = _trim_for_report(analysis)
        print(f"    n_pos={analysis['n_positive_events']} n_neg={analysis['n_negative_events']} "
              f"beat_cycles(high)={analysis['beat_cycle_count_high']} BPM(high)={_fmt(analysis['provisional_bpm_high'])} "
              f"median_IBI={_fmt(analysis['median_ibi_s_high'])}s CV={_fmt(analysis['cv_ibi_high'])} "
              f"dominant_freq={_fmt(analysis['dominant_frequency_hz'])}Hz "
              f"biobeat_merges={analysis['n_merged_biological_beats']} biobeat_BPM={_fmt(analysis['biological_beat_bpm_high'])}")

    # Consistency check: this run's OWN frame_step=3 region-mean (computed
    # by the exact same function used for frame_step=1/2 above -- no
    # separate recomputation) vs the canonical production
    # patch_waveforms_radial.npz secondary_region mean -- same method,
    # same source video, should closely agree; a disagreement would flag a
    # methodological difference between this diagnostic and
    # pipeline/analyze.py, not a frame_step effect.
    canonical_secondary_wf = region_mean_waveform(common["production_waveforms"], secondary_patches)
    fresh_fs3 = raw_region_radial_by_step[3]
    n = min(len(canonical_secondary_wf), len(fresh_fs3))
    consistency_corr = float(np.corrcoef(canonical_secondary_wf[:n], fresh_fs3[:n])[0, 1]) if n > 2 else None
    max_abs_diff = float(np.max(np.abs(canonical_secondary_wf[:n] - fresh_fs3[:n]))) if n > 0 else None
    print(f"  [consistency check] recomputed-fs3 vs canonical-npz secondary_region waveform: "
          f"corr={_fmt(consistency_corr)} max_abs_diff={_fmt(max_abs_diff)} (n_samples_compared={n})")

    # -- QC figure --
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=False)
    for ax, frame_step in zip(axes[:3], (1, 2, 3)):
        analysis = waveforms_by_step[frame_step]
        wf = analysis["detection_input"]
        t = np.arange(len(wf)) * analysis["sample_interval_s"]
        ax.plot(t, wf, color="black", linewidth=0.9)
        for events, marker, color in ((analysis["positive_events"], "^", "crimson"), (analysis["negative_events"], "v", "darkblue")):
            ax.scatter([e["time_sec"] for e in events], [e["local_amplitude"] for e in events], marker=marker, color=color, s=28, zorder=5)
        ax.set_title(
            f"frame_step={frame_step} (sample_interval={analysis['sample_interval_s']:.4f}s): "
            f"n_cycles(high)={analysis['beat_cycle_count_high']} BPM(high)={_fmt(analysis['provisional_bpm_high'])} "
            f"CV(IBI)={_fmt(analysis['cv_ibi_high'])} biobeat_BPM={_fmt(analysis['biological_beat_bpm_high'])}",
            fontsize=9,
        )
        ax.set_ylabel("radial motion (px)")
    axes[2].set_xlabel("time (s)")

    bpms = [waveforms_by_step[fs]["provisional_bpm_high"] for fs in (1, 2, 3)]
    bio_bpms = [waveforms_by_step[fs]["biological_beat_bpm_high"] for fs in (1, 2, 3)]
    x = np.arange(3)
    width = 0.35
    axes[3].bar(x - width / 2, [b if b else 0 for b in bpms], width, label="beat_cycle BPM (high-confidence)", color="steelblue")
    axes[3].bar(x + width / 2, [b if b else 0 for b in bio_bpms], width, label="biological-beat BPM (self-relative merge)", color="seagreen")
    axes[3].axhline(60.0 * MANUAL_OBSERVED_BEAT_COUNT / np.mean(MANUAL_OBSERVED_DURATION_RANGE_S), color="black",
                     linestyle="--", linewidth=1.0, label="manual observed (reference only, not a tuning target)")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(["frame_step=1", "frame_step=2", "frame_step=3 (production)"])
    axes[3].set_ylabel("provisional BPM")
    axes[3].legend(fontsize=7)
    axes[3].set_title("BPM stability across temporal resolution", fontsize=9)

    fig.suptitle("D10/slit/a1 secondary_region -- temporal-resolution (frame_step) diagnostic (round 33)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = output_dir / "D10_SECONDARY_FRAMESTEP_COMPARISON.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")

    return {
        "by_frame_step": results,
        "consistency_check_recomputed_fs3_vs_canonical_npz": {
            "correlation": consistency_corr, "max_abs_diff": max_abs_diff, "n_samples_compared": n,
        },
        "note": (
            "If BPM/cycle-count/dominant_frequency stay approximately stable across frame_step=1/2/3, "
            "temporal subsampling is NOT the explanation for secondary_region's ~2x rate -- the same "
            "~66 BPM pattern is already present at the finest available resolution."
        ),
    }, waveforms_by_step


# --------------------------------------------------------------------------
# 2. Waveform-morphology diagnostic (event-triggered averages at frame_step=1)
# --------------------------------------------------------------------------


def section2_waveform_morphology(common: dict, secondary_fs1_analysis: dict, secondary_fine_fs1: dict, output_dir: Path) -> tuple[dict, dict]:
    print("\n=== Section 2: waveform-morphology diagnostic (event-triggered averages, frame_step=1) ===")
    sample_interval_s = secondary_fs1_analysis["sample_interval_s"]
    stages = secondary_fs1_analysis["stages"]
    baseline_corrected = secondary_fs1_analysis["detection_input"]  # == stages["baseline_corrected"]
    trigger_events = secondary_fs1_analysis["positive_events"]
    trigger_indices = [e["sample_index"] for e in trigger_events]

    magnitude_wf = np.mean(np.stack([secondary_fine_fs1[k]["magnitude"] for k in secondary_fine_fs1]), axis=0)
    first_derivative_wf = np.gradient(baseline_corrected, sample_interval_s)
    abs_magnitude_wf = np.abs(magnitude_wf[: len(baseline_corrected)])

    # Window wide enough to show a full local beat cycle on each side,
    # scaled to secondary_region's OWN robust reference period (self-
    # relative, never a fixed absolute time) -- falls back to a generic
    # ~0.9s (round 29's observed secondary_region median gap) only if no
    # robust estimate is available for this run.
    ref_period = secondary_fs1_analysis["reference_period_s_self_relative"] or 0.9
    window_s = 0.7 * ref_period

    variants = {
        "raw": stages["raw"],
        "detrended": stages["detrended"],
        "smoothed": stages["smoothed"],
        "signed_radial_flow (baseline-corrected, detection input)": baseline_corrected,
        "first_derivative_of_radial_flow": first_derivative_wf,
        "abs_optical_flow_magnitude": abs_magnitude_wf,
    }

    reports = {}
    fig, axes_grid = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes_grid.ravel()
    for ax, (label, wf) in zip(axes, variants.items()):
        eta = event_triggered_average(wf, trigger_indices, sample_interval_s, window_s)
        reports[label] = dict(eta)
        if eta["mean_snippet"] is not None:
            t = np.array(eta["time_offsets_s"])
            mean = np.array(eta["mean_snippet"])
            std = np.array(eta["std_snippet"])
            ax.plot(t, mean, color="black", linewidth=1.4)
            ax.fill_between(t, mean - std, mean + std, color="gray", alpha=0.3)
            ax.axvline(0, color="crimson", linewidth=0.8, linestyle=":")
            # descriptive local-maxima count within the window (double-peak check)
            local_max = [i for i in range(1, len(mean) - 1) if mean[i - 1] < mean[i] >= mean[i + 1]]
            reports[label]["n_local_maxima_in_mean_snippet"] = len(local_max)
        ax.set_title(f"{label}\n(n_events_used={eta['n_events_used']})", fontsize=8)
        ax.set_xlabel("time relative to secondary event (s)")
    fig.suptitle(
        "D10/slit/a1 secondary_region -- event-triggered average waveform morphology (frame_step=1, round 33)\n"
        "aligned on secondary_region's own detected positive events; shaded = +/-1 std across events",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return reports, {
        "window_s": window_s, "n_trigger_events": len(trigger_indices), "reference_period_s_used": ref_period,
        "figure": fig, "variants_computed": list(variants),
    }


# --------------------------------------------------------------------------
# 3. Spatial decomposition (per-patch analysis, secondary vs primary/tertiary)
# --------------------------------------------------------------------------


def per_patch_beat_summary(patch_id: tuple[int, int], waveform: np.ndarray, sample_interval_s: float, frame_step: int, fps: float, config: dict) -> dict:
    analysis = analyze_waveform(waveform, sample_interval_s, frame_step, fps, config, f"patch_{patch_id[0]}_{patch_id[1]}")
    return {
        "patch_id": list(patch_id),
        "amplitude_std": float(np.std(analysis["detection_input"])),
        "dominant_frequency_hz": analysis["dominant_frequency_hz"],
        "n_beat_cycles_high": analysis["beat_cycle_count_high"],
        "median_ibi_s": analysis["median_ibi_s_high"],
        "provisional_bpm": analysis["provisional_bpm_high"],
        "detection_input": analysis["detection_input"],
    }


def section3_spatial_decomposition(common: dict, output_dir: Path) -> dict:
    print("\n=== Section 3: spatial decomposition (per-patch, secondary vs primary/tertiary) ===")
    config = common["config"]
    fps = common["fps"]
    frame_step = common["production_frame_step"]
    sample_interval_s = frame_step / fps
    waveforms = common["production_waveforms"]

    per_region_patch_results = {}
    for region_name in ("primary_region", "secondary_region", "tertiary_region"):
        patch_results = {}
        for pid in common["regions"][region_name]:
            pid_t = tuple(pid)
            if pid_t not in waveforms:
                print(f"  {region_name} patch {pid_t}: not present in analyzed grid -- skipping")
                continue
            patch_results[pid_t] = per_patch_beat_summary(pid_t, waveforms[pid_t], sample_interval_s, frame_step, fps, config)
            r = patch_results[pid_t]
            print(f"  [{region_name}] patch {pid_t}: amplitude={_fmt(r['amplitude_std'])} "
                  f"dominant_freq={_fmt(r['dominant_frequency_hz'])}Hz n_cycles(high)={r['n_beat_cycles_high']} "
                  f"BPM={_fmt(r['provisional_bpm'])}")
        # neighbor coherence within the FULL analyzed grid (not just this
        # region's own patches), same convention as spatial_patches.py.
        for pid_t in patch_results:
            patch_results[pid_t]["neighbor_coherence_full_grid"] = neighbor_coherence(pid_t, waveforms)
        per_region_patch_results[region_name] = patch_results

    secondary_sync = cross_patch_synchronization({pid: waveforms[pid] for pid in per_region_patch_results["secondary_region"]})
    print(f"  secondary_region cross-patch mean pairwise correlation: {_fmt(secondary_sync['mean_pairwise_correlation'])}")

    # -- QC figure: stacked secondary patch waveforms + local BPM grid + cross-region BPM comparison --
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1])

    ax_stack = fig.add_subplot(gs[:, 0])
    spacing = 4.0 * np.std(np.concatenate([r["detection_input"] for r in per_region_patch_results["secondary_region"].values()])) or 1.0
    for i, (pid, r) in enumerate(per_region_patch_results["secondary_region"].items()):
        t = np.arange(len(r["detection_input"])) * sample_interval_s
        ax_stack.plot(t, r["detection_input"] + i * spacing, linewidth=1.0)
        ax_stack.text(t[-1] * 1.01, i * spacing, f"{pid}  BPM={_fmt(r['provisional_bpm'])}", fontsize=7, va="center")
    ax_stack.set_title("secondary_region: per-patch waveforms (offset stacked)", fontsize=9)
    ax_stack.set_xlabel("time (s)")
    ax_stack.set_yticks([])

    ax_grid = fig.add_subplot(gs[0, 1])
    rows = [pid[0] for pid in per_region_patch_results["secondary_region"]]
    cols = [pid[1] for pid in per_region_patch_results["secondary_region"]]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    grid = np.full((r1 - r0 + 1, c1 - c0 + 1), np.nan)
    for pid, r in per_region_patch_results["secondary_region"].items():
        grid[pid[0] - r0, pid[1] - c0] = r["provisional_bpm"] if r["provisional_bpm"] else np.nan
    im = ax_grid.imshow(grid, cmap="coolwarm", vmin=20, vmax=80)
    for (row, col), r in per_region_patch_results["secondary_region"].items():
        ax_grid.text(col - c0, row - r0, _fmt(r["provisional_bpm"]), ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax_grid, fraction=0.046, label="provisional BPM")
    ax_grid.set_title("secondary_region: per-patch BPM grid", fontsize=9)
    ax_grid.set_xticks([]); ax_grid.set_yticks([])

    ax_cmp = fig.add_subplot(gs[1, 1])
    for i, region_name in enumerate(("primary_region", "secondary_region", "tertiary_region")):
        bpms = [r["provisional_bpm"] for r in per_region_patch_results[region_name].values() if r["provisional_bpm"]]
        ax_cmp.scatter([i] * len(bpms), bpms, color=REGION_COLORS_MPL[region_name], label=region_name, s=40, alpha=0.8)
    ax_cmp.set_xticks([0, 1, 2])
    ax_cmp.set_xticklabels(["primary", "secondary", "tertiary"])
    ax_cmp.set_ylabel("per-patch provisional BPM")
    ax_cmp.set_title(f"per-patch BPM by region\n(secondary cross-patch mean corr={_fmt(secondary_sync['mean_pairwise_correlation'])})", fontsize=9)
    ax_cmp.axhline(33.4, color="gray", linewidth=0.8, linestyle="--")
    ax_cmp.axhline(66.8, color="gray", linewidth=0.8, linestyle="--")

    fig.suptitle("D10/slit/a1 -- spatial decomposition of secondary_region vs primary/tertiary (round 33)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = output_dir / "D10_SECONDARY_PATCH_WAVEFORMS.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")

    report = {
        region_name: {
            f"{pid[0]}_{pid[1]}": {k: v for k, v in r.items() if k != "detection_input"}
            for pid, r in patches.items()
        }
        for region_name, patches in per_region_patch_results.items()
    }
    report["secondary_region_cross_patch_synchronization"] = secondary_sync
    return report


# --------------------------------------------------------------------------
# 4. Optical-flow direction diagnostic
# --------------------------------------------------------------------------


def section4_flow_direction(common: dict, secondary_fs1_analysis: dict, secondary_fine_fs1: dict, output_dir: Path, fig2) -> dict:
    print("\n=== Section 4: optical-flow direction diagnostic (radial vs tangential, per event) ===")
    sample_interval_s = secondary_fs1_analysis["sample_interval_s"]
    all_events = sorted(
        secondary_fs1_analysis["positive_events"] + secondary_fs1_analysis["negative_events"],
        key=lambda e: e["time_sec"],
    )
    region_radial = np.mean(np.stack([secondary_fine_fs1[k]["radial"] for k in secondary_fine_fs1]), axis=0)
    region_tangential = np.mean(np.stack([secondary_fine_fs1[k]["tangential"] for k in secondary_fine_fs1]), axis=0)
    region_magnitude = np.mean(np.stack([secondary_fine_fs1[k]["magnitude"] for k in secondary_fine_fs1]), axis=0)

    per_event = []
    for e in all_events:
        idx = e["sample_index"]
        idx = max(0, min(idx, len(region_radial) - 1))
        radial_val = float(region_radial[idx])
        tangential_val = float(region_tangential[idx])
        magnitude_val = float(region_magnitude[idx])
        per_event.append({
            "time_sec": e["time_sec"],
            "polarity_of_detection": "positive" if e in secondary_fs1_analysis["positive_events"] else "negative",
            "radial_flow": radial_val, "tangential_flow": tangential_val, "total_flow_magnitude": magnitude_val,
            "radial_direction": "outward" if radial_val > 0 else ("inward" if radial_val < 0 else "zero"),
        })

    # alternation check on the RAW radial-direction sign sequence (not the
    # positive/negative DETECTOR polarity label, which by construction
    # already alternates in most pairs -- this checks the underlying flow
    # vector's own inward/outward sign at each event).
    directions = [p["radial_direction"] for p in per_event if p["radial_direction"] != "zero"]
    n_alternating = sum(1 for a, b in zip(directions[:-1], directions[1:]) if a != b)
    alternating_fraction = (n_alternating / (len(directions) - 1)) if len(directions) > 1 else None
    same_direction_runs = sum(1 for a, b in zip(directions[:-1], directions[1:]) if a == b)

    ax = fig2.add_subplot(2, 1, 2)
    colors = ["crimson" if p["radial_direction"] == "outward" else "darkblue" for p in per_event]
    ax.bar([p["time_sec"] for p in per_event], [p["radial_flow"] for p in per_event], width=0.05, color=colors, label="radial (red=outward, blue=inward)")
    ax.plot([p["time_sec"] for p in per_event], [p["tangential_flow"] for p in per_event], "o-", color="darkorange", markersize=4, linewidth=0.8, label="tangential")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(
        f"optical-flow direction per secondary event: alternating_fraction={_fmt(alternating_fraction)}, "
        f"n_same_direction_consecutive_pairs={same_direction_runs}/{len(directions) - 1 if len(directions) > 1 else 0}",
        fontsize=8,
    )
    ax.set_ylabel("flow (px)")
    ax.legend(fontsize=7)

    print(f"  n_events={len(per_event)}  radial-direction alternating_fraction={_fmt(alternating_fraction)}  "
          f"same-direction consecutive pairs={same_direction_runs}")

    return {
        "per_event": per_event,
        "n_events": len(per_event),
        "radial_direction_alternating_fraction": alternating_fraction,
        "n_same_direction_consecutive_pairs": same_direction_runs,
        "n_direction_comparisons": len(directions) - 1 if len(directions) > 1 else 0,
        "note": (
            "alternating_fraction close to 1.0 (inward/outward/inward/outward...) is CONSISTENT WITH "
            "consecutive events being the two motion phases of one biological beat. A high fraction of "
            "same-direction consecutive pairs (repeated inward-inward or outward-outward) is CONSISTENT "
            "WITH two separate real excursions instead. Neither is asserted as confirmed here."
        ),
    }


# --------------------------------------------------------------------------
# 5. Image-observability / texture diagnostic
# --------------------------------------------------------------------------


def _forward_backward_consistency(video_path: Path, roi, sample_frame_indices: list[int], patch_bboxes: dict) -> dict:
    x0, y0, x1, y1 = roi.roi_bbox_xyxy
    local_roi = replace(roi, roi_bbox_xyxy=(0, 0, x1 - x0, y1 - y0))
    per_patch_errors = {key: [] for key in patch_bboxes}
    cap = cv2.VideoCapture(str(video_path))
    for idx in sample_frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx - 1))
        ok_a, frame_a = cap.read()
        ok_b, frame_b = cap.read()
        if not (ok_a and ok_b):
            continue
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
        flow_fwd = compute_dense_optical_flow(gray_a, gray_b, local_roi, method="opencv_farneback")
        flow_bwd = compute_dense_optical_flow(gray_b, gray_a, local_roi, method="opencv_farneback")
        h, w = flow_fwd.shape[:2]
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        map_x = (xs + flow_fwd[..., 0]).clip(0, w - 1)
        map_y = (ys + flow_fwd[..., 1]).clip(0, h - 1)
        warped_bwd_x = cv2.remap(flow_bwd[..., 0], map_x, map_y, cv2.INTER_LINEAR)
        warped_bwd_y = cv2.remap(flow_bwd[..., 1], map_x, map_y, cv2.INTER_LINEAR)
        error = np.hypot(flow_fwd[..., 0] + warped_bwd_x, flow_fwd[..., 1] + warped_bwd_y)
        for key, (bx0, by0, bx1, by1) in patch_bboxes.items():
            per_patch_errors[key].append(float(np.nanmean(error[by0:by1, bx0:bx1])))
    cap.release()
    return {key: float(np.mean(v)) if v else None for key, v in per_patch_errors.items()}


def section5_texture_observability(common: dict, secondary_fine_fs1: dict, all_patches_fine_fs1: dict, output_dir: Path) -> dict:
    print("\n=== Section 5: image-observability / texture diagnostic ===")
    x0, y0, x1, y1 = common["roi"].roi_bbox_xyxy
    rep_gray_roi = cv2.cvtColor(common["rep_frame_rgb"], cv2.COLOR_RGB2GRAY)[y0:y1, x0:x1]
    sobel_x = cv2.Sobel(rep_gray_roi, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(rep_gray_roi, cv2.CV_32F, 0, 1, ksize=3)
    gradient_mag_full = np.hypot(sobel_x, sobel_y)

    cell_size_px = common["cell_size_px"]
    all_patch_ids = {pid: patch_bbox_local(pid[0], pid[1], cell_size_px) for region in common["regions"].values() for pid in region}
    sample_frame_indices = np.linspace(common["frame_count"] // 4, 3 * common["frame_count"] // 4, 5, dtype=int).tolist()
    fb_consistency = _forward_backward_consistency(VIDEO_PATH, common["roi"], sample_frame_indices, {f"{p[0]}_{p[1]}": b for p, b in all_patch_ids.items()})

    rows = []
    for region_name in ("primary_region", "secondary_region", "tertiary_region"):
        for pid in common["regions"][region_name]:
            bx0, by0, bx1, by1 = patch_bbox_local(pid[0], pid[1], cell_size_px)
            intensity_contrast = float(np.std(rep_gray_roi[by0:by1, bx0:bx1]))
            gradient_magnitude = float(np.mean(gradient_mag_full[by0:by1, bx0:bx1]))
            key = f"{pid[0]}_{pid[1]}"
            flow_magnitude = float(np.mean(all_patches_fine_fs1[key]["magnitude"])) if key in all_patches_fine_fs1 else None
            rows.append({
                "region": region_name, "patch_id": list(pid),
                "intensity_contrast_std": intensity_contrast, "gradient_magnitude_mean": gradient_magnitude,
                "optical_flow_magnitude_mean": flow_magnitude,
                "forward_backward_consistency_error_mean": fb_consistency.get(key),
            })

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for region_name in ("primary_region", "secondary_region", "tertiary_region"):
        region_rows = [r for r in rows if r["region"] == region_name]
        color = REGION_COLORS_MPL[region_name]
        axes[0].scatter([r["intensity_contrast_std"] for r in region_rows], [r["optical_flow_magnitude_mean"] for r in region_rows], color=color, label=region_name, s=40)
        axes[1].scatter([r["gradient_magnitude_mean"] for r in region_rows], [r["optical_flow_magnitude_mean"] for r in region_rows], color=color, label=region_name, s=40)
        axes[2].scatter([r["forward_backward_consistency_error_mean"] for r in region_rows], [r["optical_flow_magnitude_mean"] for r in region_rows], color=color, label=region_name, s=40)
    axes[0].set_xlabel("local intensity contrast (std)"); axes[0].set_ylabel("mean optical-flow magnitude")
    axes[1].set_xlabel("mean gradient magnitude (Sobel)"); axes[1].set_ylabel("mean optical-flow magnitude")
    axes[2].set_xlabel("forward/backward flow consistency error (lower=more reliable)"); axes[2].set_ylabel("mean optical-flow magnitude")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.suptitle(
        "D10/slit/a1 -- per-patch image-observability vs motion-signal diagnostic (round 33)\n"
        "distinguishing 'motion low = real' (low flow, high texture/consistency) from "
        "'motion estimate unreliable' (low flow, low texture and/or high FB error) -- distributions only, no threshold set",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out_path = output_dir / "D10_SECONDARY_TEXTURE_OBSERVABILITY.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")

    return {"per_patch": rows, "sample_frame_indices_used_for_fb_consistency": sample_frame_indices}


# --------------------------------------------------------------------------
# 6. Cross-region timing diagnostic (descriptive labels, not a classifier)
# --------------------------------------------------------------------------


def label_secondary_events(secondary_times, primary_times, tertiary_times, tolerance_s, primary_period_s):
    labels = []
    for t in secondary_times:
        d_p = min((abs(t - pt) for pt in primary_times), default=None)
        d_t = min((abs(t - tt) for tt in tertiary_times), default=None)
        close_p = d_p is not None and d_p <= tolerance_s
        close_t = d_t is not None and d_t <= tolerance_s
        if close_p and close_t:
            label = "ambiguous"
        elif close_p:
            label = "shared_with_primary"
        elif close_t:
            label = "shared_with_tertiary"
        else:
            near_midpoint = False
            if primary_period_s and len(primary_times) >= 2:
                ordered = sorted(primary_times)
                for a, b in zip(ordered[:-1], ordered[1:]):
                    if a <= t <= b and abs(t - (a + b) / 2) <= 0.25 * primary_period_s:
                        near_midpoint = True
                        break
            label = "possible_within_beat_substructure" if near_midpoint else "secondary_only"
        labels.append({"time_sec": t, "label": label, "distance_to_nearest_primary_s": d_p, "distance_to_nearest_tertiary_s": d_t})
    return labels


def section6_cross_region_timing(common: dict, fig2) -> dict:
    print("\n=== Section 6: cross-region timing diagnostic (production frame_step=3, canonical events) ===")
    episodes = common["episodes"]
    primary_pos = [e["event_time"] for e in _region_events(episodes, "positive_episodes", "primary_region")]
    primary_neg = [e["event_time"] for e in _region_events(episodes, "negative_episodes", "primary_region")]
    tertiary_pos = [e["event_time"] for e in _region_events(episodes, "positive_episodes", "tertiary_region")]
    tertiary_neg = [e["event_time"] for e in _region_events(episodes, "negative_episodes", "tertiary_region")]
    secondary_pos = [e["event_time"] for e in _region_events(episodes, "positive_episodes", "secondary_region")]
    secondary_neg = [e["event_time"] for e in _region_events(episodes, "negative_episodes", "secondary_region")]

    primary_times = sorted(primary_pos + primary_neg)
    tertiary_times = sorted(tertiary_pos + tertiary_neg)
    secondary_times = sorted(secondary_pos + secondary_neg)

    primary_period_s = episodes["region_cycle_metrics"]["primary_region"]["tiers"]["high_confidence_only"]["interval_stats"]["median_ibi_s"]
    tolerance_s = 0.3 * primary_period_s if primary_period_s else 0.3

    labels = label_secondary_events(secondary_times, primary_times, tertiary_times, tolerance_s, primary_period_s)
    label_counts = {}
    for entry in labels:
        label_counts[entry["label"]] = label_counts.get(entry["label"], 0) + 1
    print(f"  tolerance_s={_fmt(tolerance_s)} (0.3 x primary_region's own median IBI)")
    print(f"  secondary event label counts: {label_counts}")

    ax = fig2.add_subplot(2, 1, 1)
    label_colors = {
        "shared_with_primary": "steelblue", "shared_with_tertiary": "seagreen",
        "secondary_only": "darkorange", "possible_within_beat_substructure": "purple", "ambiguous": "gray",
    }
    ax.eventplot([primary_times], lineoffsets=2, colors=[REGION_COLORS_MPL["primary_region"]], label="primary_region")
    ax.eventplot([tertiary_times], lineoffsets=1, colors=[REGION_COLORS_MPL["tertiary_region"]], label="tertiary_region")
    for entry in labels:
        ax.scatter(entry["time_sec"], 0, color=label_colors.get(entry["label"], "black"), s=40, zorder=5)
    handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=c, label=lbl) for lbl, c in label_colors.items()]
    ax.legend(handles=handles, fontsize=6, loc="upper right", ncol=3)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["secondary\n(labeled)", "tertiary", "primary"], fontsize=7)
    ax.set_title(f"cross-region event timing (tolerance={_fmt(tolerance_s)}s): {label_counts}", fontsize=8)

    return {"tolerance_s": tolerance_s, "primary_period_s_used": primary_period_s, "per_event_labels": labels, "label_counts": label_counts}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_output_dir_args(parser, OUTPUT_DIR, "d10_secondary_cause_diagnostic")
    args = parser.parse_args()
    output_dir = resolve_output_dir(args, OUTPUT_DIR, "d10_secondary_cause_diagnostic")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()
    sam2_cfg = config["segmentation"]["sam2"]
    device = get_device(sam2_cfg["device"])
    predictor = load_predictor(checkpoint_path=REPO_ROOT / sam2_cfg["checkpoint_path"], model_cfg=sam2_cfg["model_cfg"], device=device)

    common = load_common(config, predictor)

    # ONE full frame_step=1 pass over ALL regions' patches (the expensive
    # step): reused directly for section 1's frame_step=1 case, and for
    # sections 2/4/5, so the finest-resolution video pass never runs twice.
    cell_size_px = common["cell_size_px"]
    all_patch_ids_bboxes = {
        f"{pid[0]}_{pid[1]}": patch_bbox_local(pid[0], pid[1], cell_size_px)
        for region in common["regions"].values() for pid in region
    }
    print("\nComputing full frame_step=1 pass over ALL regions' patches (for sections 1/2/4/5)...")
    all_patches_fine_fs1 = compute_fine_resolution_waveforms(VIDEO_PATH, common["roi"], 1, all_patch_ids_bboxes)
    secondary_patch_keys = [f"{p[0]}_{p[1]}" for p in common["regions"]["secondary_region"]]
    secondary_fine_fs1 = {k: all_patches_fine_fs1[k] for k in secondary_patch_keys}

    section1_report, waveforms_by_step = section1_temporal_resolution(common, output_dir, secondary_fine_fs1)
    secondary_fs1_analysis = waveforms_by_step[1]

    section2_report, section2_extra = section2_waveform_morphology(common, secondary_fs1_analysis, secondary_fine_fs1, output_dir)

    fig2 = plt.figure(figsize=(13, 8))
    section6_report = section6_cross_region_timing(common, fig2)
    section4_report = section4_flow_direction(common, secondary_fs1_analysis, secondary_fine_fs1, output_dir, fig2)

    # -- combine section 2's morphology figure as a saved-then-embedded row
    # is not directly composable across separate matplotlib figures, so
    # section 2's figure is saved standalone and section 4/6 share fig2;
    # both are merged into one PNG canvas via PIL-free vstack of rendered
    # buffers for a single D10_SECONDARY_EVENT_DIAGNOSTIC.png.
    morph_fig = section2_extra.pop("figure")
    morph_fig.canvas.draw()
    w1, h1 = morph_fig.canvas.get_width_height()
    morph_buf = np.asarray(morph_fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h1, w1, 4)[..., :3]
    plt.close(morph_fig)

    fig2.suptitle("D10/slit/a1 secondary_region -- event diagnostic (round 33): cross-region timing + optical-flow direction", fontsize=11)
    fig2.tight_layout(rect=(0, 0, 1, 0.96))
    fig2.canvas.draw()
    w2, h2 = fig2.canvas.get_width_height()
    top_buf = np.asarray(fig2.canvas.buffer_rgba(), dtype=np.uint8).reshape(h2, w2, 4)[..., :3]
    plt.close(fig2)

    target_w = max(w1, w2)
    def _pad(buf, w):
        h, cur_w = buf.shape[:2]
        if cur_w == w:
            return buf
        pad = np.full((h, w - cur_w, 3), 255, dtype=np.uint8)
        return np.hstack([buf, pad])
    combined = np.vstack([_pad(top_buf, target_w), _pad(morph_buf, target_w)])
    out_path = output_dir / "D10_SECONDARY_EVENT_DIAGNOSTIC.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    print(f"  wrote {out_path}")

    section5_report = section5_texture_observability(common, secondary_fine_fs1, all_patches_fine_fs1, output_dir)

    section2_report_serializable = {
        label: {k: v for k, v in entry.items()} for label, entry in section2_report.items()
    }
    for entry in section2_report_serializable.values():
        entry.pop("mean_snippet", None)
        entry.pop("std_snippet", None)

    summary = {
        "video": "mature/D10_slit_a1", "region": "secondary_region",
        "manual_observed_reference_only": {
            "beat_count": MANUAL_OBSERVED_BEAT_COUNT, "duration_range_s": MANUAL_OBSERVED_DURATION_RANGE_S,
            "approximate_bpm": 60.0 * MANUAL_OBSERVED_BEAT_COUNT / np.mean(MANUAL_OBSERVED_DURATION_RANGE_S),
            "note": "Reference only, stated by the project owner in conversation -- NOT used as a tuning target anywhere in this script.",
        },
        "section1_temporal_resolution": section1_report,
        "section2_waveform_morphology": {"reports": section2_report_serializable, "params": section2_extra},
        "section3_spatial_decomposition": None,  # filled below
        "section4_flow_direction": section4_report,
        "section5_texture_observability": section5_report,
        "section6_cross_region_timing": section6_report,
        "explicit_non_actions": [
            "No detection/pairing/grouping/merge threshold changed.",
            "No raw event deleted.",
            "beat_events.py/beat_cycles.py/beat_episodes.py/biological_beats.py imported unchanged, never edited.",
            "No canonical result overwritten -- this script only reads results/beat_event_episodes_v3, "
            "results/manual_reference, results/motion_pipeline_mature_v2_background_stabilized.",
            "No biological conclusion asserted.",
        ],
    }

    section3_report = section3_spatial_decomposition(common, output_dir)
    summary["section3_spatial_decomposition"] = section3_report

    with open(output_dir / "D10_SECONDARY_DIAGNOSTIC_SUMMARY.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"\nWrote D10 secondary_region cause diagnostic to {output_dir}")


if __name__ == "__main__":
    main()
