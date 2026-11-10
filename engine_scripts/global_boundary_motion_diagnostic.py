"""Whole-organoid boundary-motion global BPM diagnostic (round 30).

Independent of the existing patch-grid-based per-region pipeline
(src/cardio_organoid/pipeline/analyze.py, UNCHANGED by this script), this
builds ONE pooled radial-motion waveform from a configurable band
straddling the organoid's morphology-mask contour
(cardio_organoid.motion.boundary_motion, using
spatial_gate.boundary_band_mask -- both new/unchanged existing code), to
check whether whole-organoid coordinated boundary contraction gives an
independent functional signal from local/regional patch-based beating.

Per project decision:
  1. The morphology mask/contour is taken ONCE from a single
     representative-frame SAM2 segmentation (same candidate_index/model
     as pipeline/analyze.py) -- NEVER re-segmented per frame.
  2. The boundary band is built around that ONE static contour
     (configs/default.yaml: motion.global_boundary_motion), and every
     frame pair's optical flow (Farneback, drift-corrected, centroid-
     relative radial projection -- ALL unchanged, reusing raft_flow.py /
     radial_flow.py exactly as pipeline/analyze.py does) is gated to that
     band before pooling.
  3. Per frame pair: median, robust (MAD-trimmed) mean, inward magnitude,
     outward magnitude are all computed and kept (see boundary_motion.py).
  4. The MEDIAN series is fed through beat_events.py's preprocessing +
     detection and beat_cycles.py's pairing -- BOTH COMPLETELY UNCHANGED,
     same MIN_PROMINENCE_RATIO/MIN_SEPARATION_PERIOD_FRACTION/
     SMOOTH_WINDOW_SAMPLES imported from beat_event_detection_validation.py
     -- to get a provisional whole-organoid-boundary BPM/median-IBI/CV,
     directly comparable to the existing patch/region-level metrics
     already stored in each video's episodes.json.
  5. Boundary BPM is compared against patch/region BPM for 5 videos, with
     D10_slit_a1's secondary_region (66.8 BPM, see round 28/29) singled
     out for comparison against the whole-boundary signal.

No detection threshold, optical-flow method, or patch-grid parameter is
changed. Writes only to results/global_boundary_motion/ (new).

Usage:
    .venv/bin/python3 scripts/global_boundary_motion_diagnostic.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from dataclasses import replace

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cardio_organoid.functional.beat_cycle_metrics import compute_region_cycle_metrics  # noqa: E402
from cardio_organoid.functional.beat_cycles import build_beat_cycles  # noqa: E402
from cardio_organoid.functional.beat_events import (  # noqa: E402
    detect_beat_events_both_polarities,
    preprocess_waveform,
    region_beat_summary,
)
from cardio_organoid.functional.beat_episodes import (  # noqa: E402
    pooled_dominant_period_s,
    region_events_from_candidate_events,
)
from cardio_organoid.motion.boundary_motion import compute_boundary_motion_waveforms, boundary_band_frame_stats  # noqa: E402
from cardio_organoid.motion.radial_flow import radial_motion_from_centroid, remove_global_drift  # noqa: E402
from cardio_organoid.motion.raft_flow import compute_dense_optical_flow, compute_motion_roi  # noqa: E402
from cardio_organoid.motion.spatial_gate import apply_gate, boundary_band_mask, crop_mask_to_roi  # noqa: E402
from cardio_organoid.pipeline.analyze import resolve_representative_mask  # noqa: E402
from cardio_organoid.segmentation.sam2_segmenter import get_device, load_predictor, segment_frame  # noqa: E402

from _output_safety import add_output_dir_args, resolve_output_dir  # noqa: E402
from beat_event_detection_validation import (  # noqa: E402
    MIN_PROMINENCE_RATIO,
    MIN_SEPARATION_PERIOD_FRACTION,
    SMOOTH_WINDOW_SAMPLES,
    _baseline_window_samples,
)

OUTPUT_DIR = REPO_ROOT / "results" / "global_boundary_motion"

VIDEOS = [
    {
        "group": "early", "name": "conv_a1",
        "source_video": REPO_ROOT / "data" / "D8 beating" / "conv" / "a1.avi",
        "motion_dir": REPO_ROOT / "results" / "motion_pipeline_v2_background_stabilized" / "conv" / "a1",
        "episodes_json": REPO_ROOT / "results" / "beat_event_episodes" / "early" / "conv_a1" / "episodes.json",
    },
    {
        "group": "early", "name": "slit_a1",
        "source_video": REPO_ROOT / "data" / "D8 beating" / "slit" / "a1.avi",
        "motion_dir": REPO_ROOT / "results" / "motion_pipeline_v2_background_stabilized" / "slit" / "a1",
        "episodes_json": REPO_ROOT / "results" / "beat_event_episodes" / "early" / "slit_a1" / "episodes.json",
    },
    {
        "group": "mature", "name": "Beating_D9_conv_a1",
        "source_video": REPO_ROOT / "data" / "working" / "mature_validation" / "Beating_D9" / "conv" / "a1.avi",
        "motion_dir": REPO_ROOT / "results" / "motion_pipeline_mature_v2_background_stabilized" / "Beating_D9" / "conv" / "a1",
        "episodes_json": REPO_ROOT / "results" / "beat_event_episodes_v2" / "mature" / "Beating_D9_conv_a1" / "episodes.json",
    },
    {
        "group": "mature", "name": "Beating_D9_slit_a1",
        "source_video": REPO_ROOT / "data" / "working" / "mature_validation" / "Beating_D9" / "slit" / "a1.avi",
        "motion_dir": REPO_ROOT / "results" / "motion_pipeline_mature_v2_background_stabilized" / "Beating_D9" / "slit" / "a1",
        "episodes_json": REPO_ROOT / "results" / "beat_event_episodes_v4" / "mature" / "Beating_D9_slit_a1" / "episodes.json",
    },
    {
        "group": "mature", "name": "D10_slit_a1",
        "source_video": REPO_ROOT / "data" / "working" / "mature_validation" / "D10_26.08.2026" / "slit" / "a1.avi",
        "motion_dir": REPO_ROOT / "results" / "motion_pipeline_mature_v2_background_stabilized" / "D10_26.08.2026" / "slit" / "a1",
        "episodes_json": REPO_ROOT / "results" / "beat_event_episodes_v3" / "mature" / "D10_slit_a1" / "episodes.json",
    },
]


def _load_config() -> dict:
    with open(REPO_ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def _fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) and x is not None else "NA"


def _recover_mask_and_roi(video_path: Path, config: dict, predictor, morphology_mask=None):
    """Single-frame SAM2 segmentation + ROI construction, EXACTLY as
    pipeline/analyze.py / beat_episode_time_synced_qc_video.py already do
    -- never repeated per frame. An injected `morphology_mask` (from a
    segmentation-interface record) replaces the internal SAM2 call."""
    sam2_cfg = config["segmentation"]["sam2"]
    motion_cfg = config["motion"]

    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rep_index = frame_count // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, rep_index)
    ok, rep_frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read representative frame {rep_index} from {video_path}")
    rep_frame_rgb = cv2.cvtColor(rep_frame_bgr, cv2.COLOR_BGR2RGB)

    morphology_mask, _source = resolve_representative_mask(predictor, rep_frame_rgb, config, morphology_mask)
    equivalent_diameter_px = float(2.0 * np.sqrt(morphology_mask.sum() / np.pi))
    roi = compute_motion_roi(
        morphology_mask, expansion_px=motion_cfg["roi"]["expansion_px"], frame_shape=rep_frame_rgb.shape[:2],
    )
    return morphology_mask, roi, equivalent_diameter_px, frame_count


def compute_boundary_waveform_for_video(video_cfg: dict, config: dict, predictor) -> dict:
    qc = json.loads((video_cfg["motion_dir"] / "motion_pipeline_qc.json").read_text())
    fps, frame_step = qc["fps"], qc["frame_step"]
    saved_equiv_diam = qc["equivalent_diameter_px"]
    sample_interval_s = frame_step / fps

    morphology_mask, roi, recovered_equiv_diam, frame_count = _recover_mask_and_roi(
        video_cfg["source_video"], config, predictor,
    )
    print(f"  ROI geometry check: recovered={recovered_equiv_diam:.1f} vs saved={saved_equiv_diam:.1f} "
          f"(diff={abs(recovered_equiv_diam - saved_equiv_diam) / saved_equiv_diam:.2%})")

    x0, y0, x1, y1 = roi.roi_bbox_xyxy
    centroid_xy = roi.centroid_xy_in_roi()
    local_roi = replace(roi, roi_bbox_xyxy=(0, 0, x1 - x0, y1 - y0))

    exact_mask_roi = crop_mask_to_roi(morphology_mask, roi.roi_bbox_xyxy)
    gbm_cfg = config["motion"]["global_boundary_motion"]
    band = boundary_band_mask(exact_mask_roi, saved_equiv_diam, gbm_cfg["inner_ratio"], gbm_cfg["outer_ratio"])
    band_coverage = float(band.mean())
    print(f"  boundary band: inner_ratio={gbm_cfg['inner_ratio']} outer_ratio={gbm_cfg['outer_ratio']} "
          f"coverage_fraction_of_ROI={band_coverage:.4f}")

    cap = cv2.VideoCapture(str(video_cfg["source_video"]))
    prev_gray_roi = None
    frame_idx = 0
    frame_stats = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_idx % frame_step == 0:
            gray_roi = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
            if prev_gray_roi is not None:
                flow = compute_dense_optical_flow(prev_gray_roi, gray_roi, local_roi, method="opencv_farneback")
                corrected, _drift = remove_global_drift(flow)
                radial = radial_motion_from_centroid(corrected, centroid_xy)
                gated = apply_gate(radial, band)
                frame_stats.append(boundary_band_frame_stats(gated))
            prev_gray_roi = gray_roi
        frame_idx += 1
    cap.release()

    waveforms = compute_boundary_motion_waveforms(frame_stats)
    return {
        "fps": fps, "frame_step": frame_step, "sample_interval_s": sample_interval_s,
        "band_coverage_fraction_of_roi": band_coverage,
        "n_frame_pairs": len(frame_stats),
        "waveforms": waveforms,
    }


def process_video(video_cfg: dict, config: dict, predictor, output_dir: Path) -> dict:
    group, name = video_cfg["group"], video_cfg["name"]
    print(f"\n=== {group}/{name} ===")
    boundary = compute_boundary_waveform_for_video(video_cfg, config, predictor)
    wf_raw = boundary["waveforms"]["median_radial_flow"]
    sample_interval_s, fps, frame_step = boundary["sample_interval_s"], boundary["fps"], boundary["frame_step"]

    # -- same preprocessing/detection/pairing as the patch-based pipeline,
    # UNCHANGED constants, applied here to the NEW boundary waveform --
    stages = preprocess_waveform(
        wf_raw, apply_detrend=True, smooth_window_samples=SMOOTH_WINDOW_SAMPLES,
        baseline_window_samples=_baseline_window_samples(wf_raw, sample_interval_s),
    )
    detection_input = stages["baseline_corrected"]
    events_both = detect_beat_events_both_polarities(
        detection_input, sample_interval_s, frame_step, fps,
        min_prominence_ratio=MIN_PROMINENCE_RATIO, min_separation_period_fraction=MIN_SEPARATION_PERIOD_FRACTION,
    )
    pos_events = region_events_from_candidate_events("whole_organoid_boundary", "positive", events_both["positive_events"])
    neg_events = region_events_from_candidate_events("whole_organoid_boundary", "negative", events_both["negative_events"])

    pos_summary = region_beat_summary(events_both["positive_events"], detection_input, sample_interval_s)
    neg_summary = region_beat_summary(events_both["negative_events"], detection_input, sample_interval_s)
    pooled_period = pooled_dominant_period_s([pos_summary["dominant_frequency_hz"], neg_summary["dominant_frequency_hz"]])
    cfg_bc = config["functional"]["beat_cycles"]
    pairing_window_s = (pooled_period * cfg_bc["pairing_window_period_fraction"]) if pooled_period else cfg_bc["fallback_pairing_window_s"]

    cycles = build_beat_cycles(pos_events, neg_events, pairing_window_s)
    pos_amp_by_time = {round(e.event_time, 6): e.amplitude for e in pos_events}
    neg_amp_by_time = {round(e.event_time, 6): e.amplitude for e in neg_events}
    metrics = compute_region_cycle_metrics([c.to_dict() for c in cycles], pos_amp_by_time, neg_amp_by_time)

    # -- comparison against existing patch/region-based BPM --
    episodes = json.loads(video_cfg["episodes_json"].read_text())
    region_comparisons = {}
    for region_name, region_metrics in episodes["region_cycle_metrics"].items():
        region_comparisons[region_name] = {
            "bpm_high": region_metrics["tiers"]["high_confidence_only"]["provisional_bpm_from_median_ibi"],
            "bpm_high_and_medium": region_metrics["tiers"]["high_and_medium"]["provisional_bpm_from_median_ibi"],
        }

    high = metrics["tiers"]["high_confidence_only"]
    hm = metrics["tiers"]["high_and_medium"]
    print(f"  n_pos_events={len(pos_events)} n_neg_events={len(neg_events)}  n_beat_cycles={len(cycles)}")
    print(f"  BOUNDARY BPM(high)={_fmt(high['provisional_bpm_from_median_ibi'])}  "
          f"median_IBI={_fmt(high['interval_stats']['median_ibi_s'])}s  CV={_fmt(high['interval_stats']['cv_ibi'])}  "
          f"BPM(high+med)={_fmt(hm['provisional_bpm_from_median_ibi'])}")
    for region_name, comp in region_comparisons.items():
        print(f"    vs patch-region [{region_name}]: BPM(high)={_fmt(comp['bpm_high'])}  BPM(high+med)={_fmt(comp['bpm_high_and_medium'])}")

    out_dir = output_dir / group / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- QC plot --
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
    t = np.arange(len(detection_input)) * sample_interval_s
    axes[0].plot(t, detection_input, color="black", linewidth=1.0)
    for e, marker, color in ((pos_events, "^", "crimson"), (neg_events, "v", "darkblue")):
        axes[0].scatter([ev.event_time for ev in e], [ev.amplitude for ev in e], marker=marker, color=color, s=30, zorder=5)
    axes[0].set_title(f"{group}/{name} -- whole-organoid boundary-band median radial flow (detection input)", fontsize=9)
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("boundary radial flow (px)")

    labels = ["boundary"] + list(region_comparisons)
    bpms = [high["provisional_bpm_from_median_ibi"]] + [region_comparisons[r]["bpm_high"] for r in region_comparisons]
    colors = ["black"] + [f"C{i}" for i in range(len(region_comparisons))]
    bars = axes[1].bar(labels, [b if b is not None else 0 for b in bpms], color=colors)
    for bar, b in zip(bars, bpms):
        axes[1].annotate(_fmt(b), (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha="center", va="bottom", fontsize=8)
    axes[1].set_ylabel("provisional BPM (high-confidence)")
    axes[1].set_title("boundary vs patch/region BPM (provisional, not final)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "boundary_vs_region_bpm.png", dpi=130)
    plt.close(fig)

    result = {
        "video": f"{group}/{name}",
        "fps": fps, "frame_step": frame_step, "sample_interval_s": sample_interval_s,
        "band_coverage_fraction_of_roi": boundary["band_coverage_fraction_of_roi"],
        "n_positive_events": len(pos_events), "n_negative_events": len(neg_events),
        "boundary_cycle_metrics": metrics,
        "patch_region_comparison": region_comparisons,
    }
    with open(out_dir / "global_boundary_motion.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    add_output_dir_args(parser, OUTPUT_DIR, "global_boundary_motion")
    args = parser.parse_args()
    output_dir = resolve_output_dir(args, OUTPUT_DIR, "global_boundary_motion")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_config()
    sam2_cfg = config["segmentation"]["sam2"]
    device = get_device(sam2_cfg["device"])
    predictor = load_predictor(
        checkpoint_path=REPO_ROOT / sam2_cfg["checkpoint_path"], model_cfg=sam2_cfg["model_cfg"], device=device,
    )

    all_results = []
    for video_cfg in VIDEOS:
        all_results.append(process_video(video_cfg, config, predictor, output_dir))

    with open(output_dir / "global_boundary_motion_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote global boundary-motion diagnostic for {len(all_results)} video(s) to {output_dir}")


if __name__ == "__main__":
    main()
