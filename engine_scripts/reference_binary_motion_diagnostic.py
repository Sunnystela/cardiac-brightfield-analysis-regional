"""Reference-frame pixel-difference binary movement diagnostic (round 34).

A SECOND, INDEPENDENT motion channel alongside (never a replacement for)
this project's existing Farneback-based optical-flow pipeline
(raft_flow.py / radial_flow.py / spatial_patches.py / boundary_motion.py,
beat_events.py / beat_cycles.py -- ALL completely unchanged, still the
primary analysis). Where Farneback estimates a dense velocity field per
frame PAIR, this channel asks a much coarser question per frame: how much
of the organoid's analyzed region differs, in raw pixel brightness, from a
reference image -- in the spirit of Devarasetty et al. (2017)'s
bright-field movement analysis, but explicitly NOT reproducing that
paper's fixed threshold. Every threshold here is derived per-video from
this project's own static-background pixel noise (see
cardio_organoid.motion.background_noise_threshold), because a literature
constant is not self-relative to this project's own illumination/sensor/
compression characteristics (the same posture as every other threshold in
this codebase, e.g. beat_events.robust_noise_floor).

    diff_t(x, y)   = abs(frame_t(x, y) - reference(x, y))
    binary_t(x, y) = 1 if diff_t(x, y) > threshold else 0

Three reference-frame strategies are compared, not chosen a priori (this
project's own manual-reference annotations record non-synchronous,
spatially localized regional beating -- e.g. D10/slit/a1's
"sequential_multiregional_beating" -- so a single resting frame is not
assumed sufficient):
  A. minimum-global-motion resting frame (a real captured frame).
  B. temporal median over a subsample of the recording (a blended image).
  C. adaptive EMA reference (causal: a frame's own diff always uses the
     PRE-update reference state -- see reference_frame_strategies.py's
     module docstring for the leakage caveat).

Threshold candidates: k in {2, 3, 4, 5, 6, 8} x background MAD (Gaussian-
scaled), for every (video, reference strategy) combination -- a grid, not
a single hand-picked value, so sensitivity can be inspected rather than
asserted.

NOT done here (per project instruction for this round):
  - beat_events.py/beat_cycles.py/boundary_motion.py/pipeline/analyze.py
    are imported for read-only reuse (dominant_spectral_peak,
    detect_candidate_events, region_mean_waveform, cross_correlation_curve)
    but NEVER modified, and their existing outputs are never overwritten.
  - No threshold, reference strategy, or smoothing parameter is tuned so
    that D10/slit/a1's secondary_region reproduces ~19 beats / ~31s or any
    other manual count. The manual count is used ONLY as a labeled,
    post-hoc reference line in output plots/JSON, never as an optimization
    target.
  - Writes only to a fresh results/_scratch_runs/ directory by default
    (see scripts/_output_safety.py) -- never a canonical path.

Compute-budget design note (documented, not hidden): generating a full
binary QC video for every (video x reference x threshold) combination
(5 x 3 x 6 = 90 videos) is neither necessary nor a reasonable use of disk/
compute for a diagnostic. Every video/reference/threshold combination gets
the full QUANTITATIVE treatment (moving-fraction waveforms, beat-like
periodicity, threshold QC) -- only ONE representative threshold per
reference ("k=4", the middle of the candidate grid) gets a rendered heatmap
+ connected-component summary, and actual binary QC VIDEOS are rendered
for mature/D10_slit_a1 (all 3 reference strategies, the round's primary
focus) plus one default-reference video per other video (for a visual
sanity check), not the full 90-way grid.

Usage:
    .venv/bin/python3 scripts/reference_binary_motion_diagnostic.py
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cardio_organoid.functional.beat_events import detect_candidate_events, light_smooth  # noqa: E402
from cardio_organoid.functional.regional_lag_diagnostics import cross_correlation_curve, region_mean_waveform  # noqa: E402
from cardio_organoid.motion.background_noise_threshold import (  # noqa: E402
    moving_fraction_sensitivity,
    robust_background_stats,
    threshold_candidates,
)
from cardio_organoid.motion.background_reference import select_background_patches_multi_ring  # noqa: E402
from cardio_organoid.motion.reference_binary_motion import (  # noqa: E402
    absolute_difference,
    accumulate_participation,
    binary_mask_from_diff,
    connected_component_stats,
    moving_fraction,
    participation_frequency,
)
from cardio_organoid.motion.reference_frame_strategies import (  # noqa: E402
    ema_alpha_from_time_constant,
    select_minimum_motion_frame_index,
    temporal_median_reference,
    update_ema_reference,
)
from cardio_organoid.motion.spatial_gate import crop_mask_to_roi, dilate_mask  # noqa: E402

from _output_safety import add_output_dir_args, resolve_output_dir  # noqa: E402
from beat_event_detection_validation import MIN_PROMINENCE_RATIO, MIN_SEPARATION_PERIOD_FRACTION  # noqa: E402
from global_boundary_motion_diagnostic import VIDEOS as GB_VIDEOS, _fmt, _load_config, _recover_mask_and_roi  # noqa: E402
from run_sam2_phase1 import load_config  # noqa: E402
from cardio_organoid.segmentation.sam2_segmenter import get_device, load_predictor  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "results" / "reference_binary_motion_diagnostic"

REFERENCE_PATHS = {
    ("early", "conv_a1"): REPO_ROOT / "results" / "manual_reference" / "conv" / "a1.json",
    ("early", "slit_a1"): REPO_ROOT / "results" / "manual_reference" / "slit" / "a1.json",
    ("mature", "Beating_D9_conv_a1"): REPO_ROOT / "results" / "manual_reference" / "mature" / "Beating_D9" / "conv" / "a1.json",
    ("mature", "Beating_D9_slit_a1"): REPO_ROOT / "results" / "manual_reference" / "mature" / "Beating_D9" / "slit" / "a1.json",
    ("mature", "D10_slit_a1"): REPO_ROOT / "results" / "manual_reference" / "mature" / "D10_26.08.2026" / "slit" / "a1.json",
}
GLOBAL_BOUNDARY_PATHS = {
    key: REPO_ROOT / "results" / "global_boundary_motion" / key[0] / key[1] / "global_boundary_motion.json"
    for key in REFERENCE_PATHS
}
# Canonical (unchanged, read-only) per-region Farneback beat-cycle BPM, for
# numeric comparison alongside the waveform-level time-axis comparison plot.
CANONICAL_EPISODES_PATHS = {
    ("early", "conv_a1"): REPO_ROOT / "results" / "beat_event_episodes" / "early" / "conv_a1" / "episodes.json",
    ("early", "slit_a1"): REPO_ROOT / "results" / "beat_event_episodes" / "early" / "slit_a1" / "episodes.json",
    ("mature", "Beating_D9_conv_a1"): REPO_ROOT / "results" / "beat_event_episodes_v2" / "mature" / "Beating_D9_conv_a1" / "episodes.json",
    ("mature", "Beating_D9_slit_a1"): REPO_ROOT / "results" / "beat_event_episodes_v4" / "mature" / "Beating_D9_slit_a1" / "episodes.json",
    ("mature", "D10_slit_a1"): REPO_ROOT / "results" / "beat_event_episodes_v3" / "mature" / "D10_slit_a1" / "episodes.json",
}

K_VALUES = [2, 3, 4, 5, 6, 8]
DEFAULT_K_LABEL = "k=4"
SUBSAMPLE_TARGET_N = 150  # frames sampled for temporal-median reference + background-noise estimation
RESTING_SMOOTH_WINDOW_S = 0.5  # smoothing window (seconds) for resting-frame selection (strategy A)
EMA_TAU_S = 5.0  # adaptive-reference (strategy C) time constant -- several x expected beat period; see caveat below
SMOOTH_WINDOW_SAMPLES_FOR_BEATS = 3  # light smoothing before beat-like peak detection, matches project convention

# Manual observation stated by the project owner (rounds 28-33): NOT a
# stored value in results/manual_reference/*.json, NOT a tuning target --
# reference-only, reported alongside computed values for context.
MANUAL_OBSERVED_BEAT_COUNT = 19
MANUAL_OBSERVED_DURATION_RANGE_S = (30.0, 31.0)
MANUAL_OBSERVED_BPM_REFERENCE_ONLY = 60.0 * MANUAL_OBSERVED_BEAT_COUNT / np.mean(MANUAL_OBSERVED_DURATION_RANGE_S)

REFERENCE_STRATEGY_LABELS = {
    "A_resting_frame": "A: minimum-global-motion resting frame",
    "B_temporal_median": "B: temporal median (subsampled)",
    "C_adaptive_ema": "C: adaptive EMA reference",
}

# Which (group, name) videos get a rendered binary QC video, and with which
# reference strategy/strategies -- see module docstring's compute-budget note.
RENDER_VIDEO_PLAN = {
    ("mature", "D10_slit_a1"): ["A_resting_frame", "B_temporal_median", "C_adaptive_ema"],
    ("early", "conv_a1"): ["A_resting_frame"],
    ("early", "slit_a1"): ["A_resting_frame"],
    ("mature", "Beating_D9_conv_a1"): ["A_resting_frame"],
    ("mature", "Beating_D9_slit_a1"): ["A_resting_frame"],
}


def _regions_from_reference(ref: dict) -> dict[str, list[tuple[int, int]]]:
    if ref.get("regions") and ref.get("activation_order"):
        return {name: [tuple(p) for p in ref["regions"][name]] for name in ref["activation_order"]}
    dominant = ref.get("dominant_beating_region") or []
    return {"dominant_beating_region": [tuple(p) for p in dominant]}


def patch_bbox_local(row: int, col: int, cell_size_px: int) -> tuple[int, int, int, int]:
    x0, y0 = col * cell_size_px, row * cell_size_px
    return x0, y0, x0 + cell_size_px, y0 + cell_size_px


def region_mask_from_patches(patch_ids: list[tuple[int, int]], cell_size_px: int, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    h, w = shape
    for row, col in patch_ids:
        x0, y0, x1, y1 = patch_bbox_local(row, col, cell_size_px)
        x1, y1 = min(x1, w), min(y1, h)
        if x0 < w and y0 < h:
            mask[y0:y1, x0:x1] = True
    return mask


# --------------------------------------------------------------------------
# Video I/O helpers (all read-only; source video is never opened for writing)
# --------------------------------------------------------------------------


def read_gray_roi_at(video_path: Path, frame_index: int, roi_bbox_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = roi_bbox_xyxy
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]


def pass1_motion_proxy_and_subsample(
    video_path: Path, roi_bbox_xyxy: tuple[int, int, int, int], analysis_region: np.ndarray, subsample_stride: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """One full-video pass (frame_step=1): whole-analysis-region mean abs
    frame-to-frame pixel difference (an independent-of-Farneback 'global
    motion' proxy, for reference strategy A), plus a memory-bounded
    subsample of raw grayscale ROI crops (for strategy B's temporal
    median and for background-noise sampling in pass 2/3).
    """
    x0, y0, x1, y1 = roi_bbox_xyxy
    cap = cv2.VideoCapture(str(video_path))
    prev_gray = None
    motion_proxy = []
    subsampled_frames = []
    subsample_indices = []
    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
        if prev_gray is not None:
            diff = np.abs(gray.astype(np.float64) - prev_gray.astype(np.float64))
            motion_proxy.append(float(np.mean(diff[analysis_region])))
        if idx % subsample_stride == 0:
            subsampled_frames.append(gray.copy())
            subsample_indices.append(idx)
        prev_gray = gray
        idx += 1
    cap.release()
    # Frame 0 has no frame-to-frame diff (no previous frame) -- duplicate
    # the first REAL motion-proxy value into that slot rather than
    # inserting an artificial 0.0, which would otherwise make frame 0 win
    # select_minimum_motion_frame_index by construction regardless of its
    # actual content.
    if motion_proxy:
        motion_proxy = [motion_proxy[0]] + motion_proxy
    return np.array(motion_proxy), np.stack(subsampled_frames), subsample_indices


def pass2_background_diff_collection(
    video_path: Path, roi_bbox_xyxy: tuple[int, int, int, int], background_gate: np.ndarray,
    ref_a: np.ndarray, ref_b: np.ndarray, ema_alpha: float, subsample_indices: set[int],
) -> dict[str, list[float]]:
    """One full-video pass, streaming the adaptive (strategy C) reference
    forward from `ref_a` -- but ONLY recording background-patch diff
    VALUES at `subsample_indices` (memory-bounded), against whichever
    reference is current at that frame, for all 3 strategies. Strategy C's
    own diff at frame t is computed against its PRE-update state (see
    reference_frame_strategies.update_ema_reference's causality warning).
    """
    x0, y0, x1, y1 = roi_bbox_xyxy
    cap = cv2.VideoCapture(str(video_path))
    ref_c = ref_a.astype(np.float64).copy()
    collected = {"A_resting_frame": [], "B_temporal_median": [], "C_adaptive_ema": []}
    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
        if idx in subsample_indices:
            for label, ref_img in (("A_resting_frame", ref_a), ("B_temporal_median", ref_b), ("C_adaptive_ema", ref_c)):
                diff = absolute_difference(gray, ref_img)
                collected[label].extend(diff[background_gate].tolist())
        ref_c = update_ema_reference(ref_c, gray, ema_alpha)
        idx += 1
    cap.release()
    return collected


def pass3_moving_fraction_waveforms(
    video_path: Path, roi_bbox_xyxy: tuple[int, int, int, int],
    ref_a: np.ndarray, ref_b: np.ndarray, ema_alpha: float,
    thresholds_by_strategy: dict[str, dict[str, float]],
    gates: dict[str, np.ndarray],
) -> tuple[dict, dict, dict, int]:
    """Third full-video pass: for every (strategy, threshold-candidate,
    gate) combination, accumulate the moving-fraction time series;
    additionally accumulate a per-pixel participation heatmap and
    connected-component-stats summary at DEFAULT_K_LABEL for each
    strategy (see module docstring's compute-budget note). Re-derives
    strategy C's EMA state fresh from `ref_a` (deterministic given the
    frame sequence -- no dependency on anything computed in pass 2).
    """
    x0, y0, x1, y1 = roi_bbox_xyxy
    cap = cv2.VideoCapture(str(video_path))
    ref_c = ref_a.astype(np.float64).copy()

    strategies = ["A_resting_frame", "B_temporal_median", "C_adaptive_ema"]
    refs = {"A_resting_frame": ref_a, "B_temporal_median": ref_b}  # C looked up live below

    moving_fraction_series: dict[str, list[float]] = {
        f"{strategy}__{k_label}__{gate_name}": []
        for strategy in strategies for k_label in thresholds_by_strategy[strategy] for gate_name in gates
    }
    participation_acc = {strategy: np.zeros((y1 - y0, x1 - x0), dtype=np.float64) for strategy in strategies}
    component_stats_accum = {strategy: [] for strategy in strategies}

    n_frames = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
        for strategy in strategies:
            ref_img = ref_c if strategy == "C_adaptive_ema" else refs[strategy]
            diff = absolute_difference(gray, ref_img)
            for k_label, threshold in thresholds_by_strategy[strategy].items():
                if threshold is None:
                    for gate_name in gates:
                        moving_fraction_series[f"{strategy}__{k_label}__{gate_name}"].append(float("nan"))
                    continue
                binary = binary_mask_from_diff(diff, threshold)
                for gate_name, gate in gates.items():
                    frac = moving_fraction(binary, gate)
                    moving_fraction_series[f"{strategy}__{k_label}__{gate_name}"].append(frac if frac is not None else float("nan"))
                if k_label == DEFAULT_K_LABEL:
                    participation_acc[strategy] = accumulate_participation(participation_acc[strategy], binary & gates["whole_roi"])
                    component_stats_accum[strategy].append(connected_component_stats(binary, gates["whole_roi"]))
        ref_c = update_ema_reference(ref_c, gray, ema_alpha)
        n_frames += 1
    cap.release()

    waveforms = {key: np.array(vals) for key, vals in moving_fraction_series.items()}
    participation = {strategy: participation_frequency(acc, n_frames) for strategy, acc in participation_acc.items()}
    component_summary = {}
    for strategy, records in component_stats_accum.items():
        n_comp = [r["n_components"] for r in records]
        largest_frac = [r["largest_component_area_fraction_of_gate"] for r in records]
        component_summary[strategy] = {
            "mean_n_components": float(np.mean(n_comp)) if n_comp else None,
            "median_n_components": float(np.median(n_comp)) if n_comp else None,
            "mean_largest_component_area_fraction": float(np.mean(largest_frac)) if largest_frac else None,
            "median_largest_component_area_fraction": float(np.median(largest_frac)) if largest_frac else None,
        }
    return waveforms, participation, component_summary, n_frames


# --------------------------------------------------------------------------
# Beat-like periodicity assessment (independent of Farneback) -- reuses
# beat_events.detect_candidate_events UNCHANGED (single-polarity peak
# detection; a moving-fraction signal is non-negative, so only positive-
# going bumps are meaningful -- no "negative excursion" concept applies).
# --------------------------------------------------------------------------


def beat_like_summary(waveform: np.ndarray, fps: float) -> dict:
    if np.all(np.isnan(waveform)) or len(waveform) < 5:
        return {"n_candidate_beats": 0, "provisional_bpm": None, "median_ibi_s": None, "cv_ibi": None,
                "median_moving_fraction": None, "peak_amplitude": None}
    clean = np.where(np.isnan(waveform), np.nanmedian(waveform), waveform)
    smoothed = light_smooth(clean, SMOOTH_WINDOW_SAMPLES_FOR_BEATS)
    sample_interval_s = 1.0 / fps
    events = detect_candidate_events(
        smoothed, sample_interval_s, frame_step=1, fps=fps,
        min_prominence_ratio=MIN_PROMINENCE_RATIO, min_separation_period_fraction=MIN_SEPARATION_PERIOD_FRACTION,
    )
    times = [e["time_sec"] for e in events]
    ibi = None
    cv = None
    bpm = None
    if len(times) >= 2:
        gaps = np.diff(sorted(times))
        ibi = float(np.median(gaps))
        mean_gap = float(np.mean(gaps))
        cv = float(np.std(gaps, ddof=1) / mean_gap) if len(gaps) > 1 and mean_gap > 0 else None
        bpm = 60.0 / ibi if ibi > 0 else None
    return {
        "n_candidate_beats": len(events), "provisional_bpm": bpm, "median_ibi_s": ibi, "cv_ibi": cv,
        "median_moving_fraction": float(np.nanmedian(clean)), "peak_amplitude": float(np.nanmax(clean)),
        "event_times_s": times,
    }


# --------------------------------------------------------------------------
# Per-video orchestration
# --------------------------------------------------------------------------


def process_video(video_cfg: dict, config: dict, predictor, output_dir: Path) -> dict:
    group, name = video_cfg["group"], video_cfg["name"]
    print(f"\n=== {group}/{name} ===")
    out_dir = output_dir / "per_video" / f"{group}_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_json = json.loads(REFERENCE_PATHS[(group, name)].read_text())
    regions = _regions_from_reference(ref_json)

    morphology_mask, roi, equiv_diam, frame_count = _recover_mask_and_roi(video_cfg["source_video"], config, predictor)
    roi_bbox = roi.roi_bbox_xyxy
    x0, y0, x1, y1 = roi_bbox
    exact_mask_roi = crop_mask_to_roi(morphology_mask, roi_bbox)
    analysis_region = dilate_mask(exact_mask_roi, config["motion"]["signal_extraction"]["gate_dilation_px"])
    cell_size_px = max(1, int(round(config["motion"]["spatial_patches"]["cell_size_ratio"] * equiv_diam)))
    region_masks = {rname: region_mask_from_patches(patches, cell_size_px, analysis_region.shape) for rname, patches in regions.items()}

    cap = cv2.VideoCapture(str(video_cfg["source_video"]))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # -- background/null patches: SAME already-decided utility production uses --
    null_cfg = config["motion"]["pipeline"]["null_reference"]
    sp_cfg = config["motion"]["spatial_patches"]
    screen_frames = []
    cap = cv2.VideoCapture(str(video_cfg["source_video"]))
    for _ in range(int(round(config["motion"]["signal_extraction"]["buffer_duration_s"] * fps)) + 1):
        ok, f = cap.read()
        if not ok:
            break
        screen_frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1])
    cap.release()
    centroid_xy = roi.centroid_xy_in_roi()
    null_candidates, background_gate = select_background_patches_multi_ring(
        frame_gray_roi=screen_frames[0], frames_gray_roi_buffer=screen_frames, centroid_xy=centroid_xy,
        equivalent_diameter_px=equiv_diam, analysis_region_mask=analysis_region,
        n_angles=null_cfg["n_angles"], radius_ratios=null_cfg["radius_ratios"], patch_size_ratio=sp_cfg["cell_size_ratio"],
    )
    print(f"  background patches: {sum(1 for c in null_candidates if c.kept)}/{len(null_candidates)} kept")

    # -- Pass 1: motion proxy (strategy A) + subsample (strategy B) --
    subsample_stride = max(1, frame_count // SUBSAMPLE_TARGET_N)
    motion_proxy, subsampled_frames, subsample_idx_list = pass1_motion_proxy_and_subsample(
        video_cfg["source_video"], roi_bbox, analysis_region, subsample_stride,
    )
    smooth_window_samples = max(1, int(round(RESTING_SMOOTH_WINDOW_S * fps)))
    ref_a_index = select_minimum_motion_frame_index(motion_proxy, smooth_window_samples)
    ref_a = read_gray_roi_at(video_cfg["source_video"], ref_a_index, roi_bbox).astype(np.float64)
    ref_b = temporal_median_reference(subsampled_frames)
    ema_alpha = ema_alpha_from_time_constant(EMA_TAU_S, 1.0 / fps)
    print(f"  strategy A resting frame index={ref_a_index} (t={ref_a_index / fps:.2f}s)  "
          f"strategy B median over {len(subsampled_frames)} subsampled frames  ema_alpha={ema_alpha:.5f} (tau={EMA_TAU_S}s)")

    # -- Pass 2: background-noise distribution per strategy -> threshold candidates --
    background_diffs = pass2_background_diff_collection(
        video_cfg["source_video"], roi_bbox, background_gate, ref_a, ref_b, ema_alpha, set(subsample_idx_list),
    )
    background_stats = {s: robust_background_stats(np.array(v)) for s, v in background_diffs.items()}
    thresholds_by_strategy = {s: threshold_candidates(background_stats[s], K_VALUES) for s in background_stats}
    for s in thresholds_by_strategy:
        print(f"  [{s}] background median={_fmt(background_stats[s]['median'])} "
              f"mad_scaled={_fmt(background_stats[s]['mad_scaled'])}  thresholds={[_fmt(v) for v in thresholds_by_strategy[s].values()]}")

    # -- sensitivity of the ORGANOID region itself to each threshold, at the
    # representative frame, for the threshold-QC report (does k=2 flag
    # nearly all of the organoid as "moving" even in a quiet frame?) --
    rep_frame = read_gray_roi_at(video_cfg["source_video"], frame_count // 2, roi_bbox)
    organoid_sensitivity = {
        s: moving_fraction_sensitivity(
            absolute_difference(rep_frame, ref_a if s == "A_resting_frame" else (ref_b if s == "B_temporal_median" else ref_a))[analysis_region],
            thresholds_by_strategy[s],
        )
        for s in thresholds_by_strategy
    }

    # -- Pass 3: moving-fraction waveforms (all strategies x thresholds x gates), heatmaps, component stats --
    gates = {"whole_roi": analysis_region, **region_masks}
    waveforms, participation, component_summary, n_frames = pass3_moving_fraction_waveforms(
        video_cfg["source_video"], roi_bbox, ref_a, ref_b, ema_alpha, thresholds_by_strategy, gates,
    )
    np.savez(out_dir / "moving_fraction_waveforms.npz", **waveforms)
    print(f"  n_frames_processed={n_frames}  n_waveforms_saved={len(waveforms)}")

    # -- beat-like periodicity, every (strategy, k, gate) combination --
    beat_summaries = {key: beat_like_summary(wf, fps) for key, wf in waveforms.items()}

    # -- Farneback comparison (canonical, unchanged; read-only) --
    npz = np.load(video_cfg["motion_dir"] / "patch_waveforms_radial.npz")
    farneback_waveforms = {tuple(int(p) for p in k.split("_")): npz[k] for k in npz.files}
    qc = json.loads((video_cfg["motion_dir"] / "motion_pipeline_qc.json").read_text())
    farneback_fps, farneback_frame_step = qc["fps"], qc["frame_step"]
    farneback_region_waveforms = {
        rname: region_mean_waveform(farneback_waveforms, patches) for rname, patches in regions.items()
    }
    gb_path = GLOBAL_BOUNDARY_PATHS.get((group, name))
    boundary_bpm = None
    if gb_path and gb_path.exists():
        gb = json.loads(gb_path.read_text())
        boundary_bpm = gb["boundary_cycle_metrics"]["tiers"]["high_confidence_only"]["provisional_bpm_from_median_ibi"]

    farneback_region_bpm_canonical = {}
    episodes_path = CANONICAL_EPISODES_PATHS.get((group, name))
    if episodes_path and episodes_path.exists():
        canonical_episodes = json.loads(episodes_path.read_text())
        for rname in regions:
            rc = canonical_episodes.get("region_cycle_metrics", {}).get(rname)
            if rc:
                farneback_region_bpm_canonical[rname] = rc["tiers"]["high_confidence_only"]["provisional_bpm_from_median_ibi"]

    # -- temporal lag: Farneback region waveform vs binary moving-fraction (default k, strategy A) --
    lag_report = {}
    for rname in regions:
        key = f"A_resting_frame__{DEFAULT_K_LABEL}__{rname}"
        if key in waveforms and rname in farneback_region_waveforms:
            fb_wf = farneback_region_waveforms[rname]
            bin_wf = waveforms[key]
            # resample the (coarser, frame_step=farneback_frame_step) Farneback
            # waveform onto the binary channel's frame_step=1 time grid via
            # linear interpolation, purely for a shared-grid cross-correlation.
            fb_t = np.arange(len(fb_wf)) * (farneback_frame_step / farneback_fps)
            bin_t = np.arange(len(bin_wf)) * (1.0 / fps)
            common_t = np.linspace(0, min(fb_t[-1], bin_t[-1]), min(len(fb_wf), len(bin_wf)))
            fb_i = np.interp(common_t, fb_t, fb_wf)
            bin_i = np.interp(common_t, bin_t, np.where(np.isnan(bin_wf), np.nanmedian(bin_wf), bin_wf))
            sample_interval = common_t[1] - common_t[0] if len(common_t) > 1 else 1.0
            curve = cross_correlation_curve(np.abs(fb_i - fb_i.mean()), bin_i, max_lag_samples=min(50, len(common_t) // 4))
            lag_report[rname] = {
                "best_lag_samples": curve["best_lag_samples"],
                "best_lag_s": curve["best_lag_samples"] * sample_interval if curve["best_lag_samples"] is not None else None,
                "best_correlation": curve["best_correlation"],
                "note": "cross-correlation of |Farneback region waveform| (rectified) vs binary moving-fraction (strategy A, k=4), interpolated onto a shared time grid.",
            }

    # -- QC plots --
    _plot_threshold_qc(background_diffs, thresholds_by_strategy, organoid_sensitivity, group, name, out_dir)
    _plot_moving_fraction_waveforms(waveforms, fps, group, name, out_dir)
    for strategy in participation:
        _plot_participation_heatmap(rep_frame, participation[strategy], analysis_region, strategy, group, name, out_dir)
    _plot_time_axis_comparison(waveforms, farneback_region_waveforms, farneback_fps, farneback_frame_step, fps, boundary_bpm, group, name, out_dir)

    return {
        "video": f"{group}/{name}", "fps": fps, "frame_count": frame_count, "n_frames_processed": n_frames,
        "equivalent_diameter_px": equiv_diam, "regions_analyzed": list(regions),
        "reference_a_frame_index": ref_a_index, "reference_a_time_s": ref_a_index / fps,
        "ema_alpha": ema_alpha, "ema_tau_s": EMA_TAU_S, "subsample_stride": subsample_stride,
        "background_noise_stats": background_stats, "threshold_candidates": thresholds_by_strategy,
        "organoid_representative_frame_sensitivity": organoid_sensitivity,
        "connected_component_summary_default_k": component_summary,
        "beat_like_summary": beat_summaries,
        "farneback_region_bpm_high_confidence": farneback_region_bpm_canonical,
        "boundary_bpm_high_confidence_reference_only": boundary_bpm,
        "temporal_lag_vs_farneback": lag_report,
        "manual_observed_reference_only": (
            {"beat_count": MANUAL_OBSERVED_BEAT_COUNT, "duration_range_s": MANUAL_OBSERVED_DURATION_RANGE_S,
             "approximate_bpm": MANUAL_OBSERVED_BPM_REFERENCE_ONLY}
            if (group, name) == ("mature", "D10_slit_a1") else None
        ),
    }


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------


def _plot_threshold_qc(background_diffs, thresholds_by_strategy, organoid_sensitivity, group, name, out_dir) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, strategy in zip(axes, thresholds_by_strategy):
        values = np.array(background_diffs[strategy])
        ax.hist(values, bins=40, color="steelblue", alpha=0.75, density=True)
        for k_label, threshold in thresholds_by_strategy[strategy].items():
            if threshold is not None:
                ax.axvline(threshold, linestyle="--", linewidth=1.0, label=f"{k_label}={threshold:.2f}")
        ax.set_title(f"{REFERENCE_STRATEGY_LABELS[strategy]}\nbackground diff distribution", fontsize=8)
        ax.set_xlabel("abs(frame - reference), background patches")
        ax.legend(fontsize=6)
    fig.suptitle(
        f"{group}/{name} -- background-adaptive threshold QC (round 34)\n"
        f"organoid-frame moving-fraction at k=2/{DEFAULT_K_LABEL}/k=8 (sensitivity check): "
        + "; ".join(f"{s}: {_fmt(organoid_sensitivity[s].get('k=2'))}/{_fmt(organoid_sensitivity[s].get(DEFAULT_K_LABEL))}/{_fmt(organoid_sensitivity[s].get('k=8'))}" for s in organoid_sensitivity),
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out_dir / "threshold_qc.png", dpi=130)
    plt.close(fig)


def _plot_moving_fraction_waveforms(waveforms, fps, group, name, out_dir) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, strategy in zip(axes, ("A_resting_frame", "B_temporal_median", "C_adaptive_ema")):
        for k_label in [f"k={k}" for k in K_VALUES]:
            key = f"{strategy}__{k_label}__whole_roi"
            if key not in waveforms:
                continue
            wf = waveforms[key]
            t = np.arange(len(wf)) / fps
            ax.plot(t, wf, linewidth=0.8, alpha=0.8, label=k_label)
        ax.set_title(f"{REFERENCE_STRATEGY_LABELS[strategy]} -- whole-organoid moving fraction, all k candidates", fontsize=8)
        ax.set_ylabel("moving fraction")
        ax.legend(fontsize=6, ncol=6, loc="upper right")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"{group}/{name} -- moving-fraction waveforms by reference strategy & threshold (round 34)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "moving_fraction_waveforms.png", dpi=130)
    plt.close(fig)


def _plot_participation_heatmap(rep_frame_gray, participation, analysis_region, strategy, group, name, out_dir) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(rep_frame_gray, cmap="gray")
    masked = np.where(analysis_region, participation, np.nan)
    im = ax.imshow(masked, cmap="inferno", alpha=0.6, vmin=0, vmax=max(0.05, float(np.nanmax(masked)) if np.isfinite(np.nanmax(masked)) else 0.05))
    plt.colorbar(im, ax=ax, fraction=0.046, label="movement participation frequency")
    ax.set_title(f"{group}/{name} -- {REFERENCE_STRATEGY_LABELS[strategy]}\ncumulative movement participation ({DEFAULT_K_LABEL})", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / f"participation_heatmap_{strategy}.png", dpi=130)
    plt.close(fig)


def _plot_time_axis_comparison(waveforms, farneback_region_waveforms, farneback_fps, farneback_frame_step, fps, boundary_bpm, group, name, out_dir) -> None:
    n_regions = len(farneback_region_waveforms)
    fig, axes = plt.subplots(n_regions, 1, figsize=(12, 3.2 * n_regions), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (rname, fb_wf) in zip(axes, farneback_region_waveforms.items()):
        fb_t = np.arange(len(fb_wf)) * (farneback_frame_step / farneback_fps)
        ax2 = ax.twinx()
        ax.plot(fb_t, fb_wf, color="black", linewidth=1.0, label="Farneback radial motion (canonical, unchanged)")
        key = f"A_resting_frame__{DEFAULT_K_LABEL}__{rname}"
        if key in waveforms:
            bin_wf = waveforms[key]
            bin_t = np.arange(len(bin_wf)) / fps
            ax2.plot(bin_t, bin_wf, color="crimson", linewidth=1.0, alpha=0.85, label=f"binary moving fraction ({DEFAULT_K_LABEL}, strategy A)")
        if boundary_bpm:
            ax.text(0.01, 0.95, f"boundary BPM (reference only): {boundary_bpm:.1f}", transform=ax.transAxes, fontsize=7, va="top")
        ax.set_title(rname, fontsize=8)
        ax.set_ylabel("Farneback radial motion (px)", fontsize=7, color="black")
        ax2.set_ylabel("moving fraction", fontsize=7, color="crimson")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc="upper right")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"{group}/{name} -- Farneback vs binary-movement, same time axis (round 34)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "time_axis_comparison.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# Binary QC video rendering (selected combinations only -- see compute-
# budget note in the module docstring)
# --------------------------------------------------------------------------


def render_binary_video(
    video_cfg: dict, config: dict, roi, cell_size_px: int, strategy: str, waveform_key: str, waveforms: dict,
    regions: dict, ref_a, ref_b, ema_alpha: float, threshold: float, fps: float, output_dir: Path,
) -> None:
    group, name = video_cfg["group"], video_cfg["name"]
    x0, y0, x1, y1 = roi.roi_bbox_xyxy
    ref_c = ref_a.astype(np.float64).copy()
    wf = waveforms[waveform_key]

    panel_w, panel_h = 480, 400
    plot_h = 220
    canvas_w = panel_w * 2
    canvas_h = panel_h + plot_h

    fig, ax = plt.subplots(figsize=(canvas_w / 100, plot_h / 100), dpi=100)
    ax.plot(np.arange(len(wf)) / fps, wf, color="crimson", linewidth=1.0)
    ax.set_xlabel("time (s)", fontsize=7)
    ax.set_ylabel("moving fraction (whole ROI)", fontsize=7)
    ax.set_title(f"{group}/{name} -- {REFERENCE_STRATEGY_LABELS[strategy]}, threshold={threshold:.2f}", fontsize=8)
    fig.tight_layout()
    fig.canvas.draw()
    w_px, h_px = fig.canvas.get_width_height()
    plot_img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h_px, w_px, 4), cv2.COLOR_RGBA2BGR)
    plot_img = cv2.resize(plot_img, (canvas_w, plot_h))

    def time_to_px(t_val: float) -> int:
        return int(round(t_val / (len(wf) / fps) * canvas_w)) if len(wf) > 0 else 0

    plt.close(fig)

    out_dir = output_dir / "binary_videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{group}_{name}_{strategy}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas_w, canvas_h))

    cap = cv2.VideoCapture(str(video_cfg["source_video"]))
    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        gray_roi = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
        ref_img = ref_c if strategy == "C_adaptive_ema" else (ref_a if strategy == "A_resting_frame" else ref_b)
        diff = absolute_difference(gray_roi, ref_img)
        binary = binary_mask_from_diff(diff, threshold)

        raw_panel = cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2BGR)
        for rname, patches in regions.items():
            for (row, col) in patches:
                px0, py0 = col * cell_size_px, row * cell_size_px
                cv2.rectangle(raw_panel, (px0, py0), (px0 + cell_size_px, py0 + cell_size_px), (0, 255, 255), 1)
        raw_panel = cv2.resize(raw_panel, (panel_w, panel_h))

        binary_panel = cv2.cvtColor((binary.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
        binary_panel = cv2.resize(binary_panel, (panel_w, panel_h))

        top_row = np.hstack([raw_panel, binary_panel])
        cv2.putText(top_row, "raw frame (yellow=region overlay)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(top_row, "binary movement mask", (panel_w + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(top_row, f"t={frame_idx / fps:6.2f}s", (10, panel_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        plot_frame = plot_img.copy()
        px = time_to_px(frame_idx / fps)
        cv2.line(plot_frame, (px, 0), (px, plot_h), (0, 255, 255), 1)

        canvas = np.vstack([top_row, plot_frame])
        writer.write(canvas)

        if strategy == "C_adaptive_ema":
            ref_c = update_ema_reference(ref_c, gray_roi, ema_alpha)
        frame_idx += 1
    cap.release()
    writer.release()
    print(f"  wrote {out_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_output_dir_args(parser, OUTPUT_DIR, "reference_binary_motion_diagnostic")
    args = parser.parse_args()
    output_dir = resolve_output_dir(args, OUTPUT_DIR, "reference_binary_motion_diagnostic")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_config()
    sam2_cfg = config["segmentation"]["sam2"]
    device = get_device(sam2_cfg["device"])
    predictor = load_predictor(checkpoint_path=REPO_ROOT / sam2_cfg["checkpoint_path"], model_cfg=sam2_cfg["model_cfg"], device=device)

    all_results = {}
    render_cache = {}
    for video_cfg in GB_VIDEOS:
        key = (video_cfg["group"], video_cfg["name"])
        result = process_video(video_cfg, config, predictor, output_dir)
        all_results[f"{key[0]}/{key[1]}"] = result
        render_cache[key] = video_cfg

    with open(output_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"\nWrote summary for {len(all_results)} video(s) to {output_dir}")

    # -- render selected binary QC videos (see compute-budget note) --
    for key, strategies in RENDER_VIDEO_PLAN.items():
        video_cfg = render_cache[key]
        group, name = key
        out_dir = output_dir / "per_video" / f"{group}_{name}"
        npz = np.load(out_dir / "moving_fraction_waveforms.npz")
        waveforms = {k: npz[k] for k in npz.files}
        ref_json = json.loads(REFERENCE_PATHS[key].read_text())
        regions = _regions_from_reference(ref_json)
        result = all_results[f"{group}/{name}"]
        morphology_mask, roi, equiv_diam, frame_count = _recover_mask_and_roi(video_cfg["source_video"], config, predictor)
        cell_size_px = max(1, int(round(config["motion"]["spatial_patches"]["cell_size_ratio"] * equiv_diam)))
        ref_a = read_gray_roi_at(video_cfg["source_video"], result["reference_a_frame_index"], roi.roi_bbox_xyxy).astype(np.float64)
        subsample_stride = result["subsample_stride"]
        _mp, subsampled_frames, _idx = pass1_motion_proxy_and_subsample(
            video_cfg["source_video"], roi.roi_bbox_xyxy,
            dilate_mask(crop_mask_to_roi(morphology_mask, roi.roi_bbox_xyxy), config["motion"]["signal_extraction"]["gate_dilation_px"]),
            subsample_stride,
        )
        ref_b = temporal_median_reference(subsampled_frames)
        cap = cv2.VideoCapture(str(video_cfg["source_video"]))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        for strategy in strategies:
            threshold = result["threshold_candidates"][strategy][DEFAULT_K_LABEL]
            if threshold is None:
                print(f"  skipping video render for {group}/{name}/{strategy}: no valid threshold")
                continue
            waveform_key = f"{strategy}__{DEFAULT_K_LABEL}__whole_roi"
            render_binary_video(
                video_cfg, config, roi, cell_size_px, strategy, waveform_key, waveforms, regions,
                ref_a, ref_b, result["ema_alpha"], threshold, fps, output_dir,
            )


if __name__ == "__main__":
    main()
