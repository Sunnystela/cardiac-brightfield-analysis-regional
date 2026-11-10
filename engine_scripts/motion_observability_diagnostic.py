"""Per-patch motion-evidence + image-observability diagnostic (round 36).

Answers, BEFORE any new phase-clustering work, the question round 35's
patch-phase consensus exposed but could not itself answer: which patches'
motion can actually be trusted? Round 35's connected-components
clustering produced a 53/69-patch mega-cluster for D10/slit/a1 because it
had no way to tell "this patch's timing looks similar to its neighbors
because it is genuinely co-active" apart from "this patch's timing looks
similar because neither patch has enough signal/texture to say anything
reliable at all." This round builds that missing evidence layer and stops
there -- it does NOT modify, extend, or re-run
`patch_timing_graph.py`/`patch_beat_consensus.py` (round 35's clustering
code), and it does NOT use manual ROI coordinates as an input anywhere in
detection/ranking (only afterward, read-only, to LABEL/evaluate the
output for the validation questions this round was asked to answer).

Five phases, kept deliberately un-combined until Phase 4/5's explicitly
PROVISIONAL composites:
  1. image_observability.py -- is the IMAGE here trackable at all?
  2. motion_confidence.py -- given a Farneback vector exists, how much
     should it be trusted?
  3. motion_evidence.py -- what does each independent motion channel
     (Farneback, reference-frame binary movement, round 34) actually see?
  4. A self-relative 2x2 (motion x observability) state map -- NEVER
     "low observability" == "no motion".
  5. reliable_hotspot_candidates.py -- interpretable ranking (not
     clustering) with an explicit rejection_reason per patch.

Compute-budget scope decisions (documented, not hidden, per this
project's established practice):
  - Per-patch Farneback radial/tangential/magnitude waveforms are
    computed at each video's OWN PRODUCTION frame_step (read from its
    `motion_pipeline_qc.json`), not a finer frame_step=1 pass -- this
    round's question is about RELIABILITY across the whole grid, not
    peak temporal resolution, so reusing the frame_step this project's
    canonical results already use keeps the comparison apples-to-apples
    and the compute bounded (one full-grid Farneback pass per video
    instead of round 33's single-region frame_step=1 pass).
  - Forward/backward consistency is sampled at a small number (8) of
    evenly-spaced frame pairs per video, not every frame pair.
  - Binary-movement evidence reuses round 34's ALREADY-COMPUTED resting-
    frame index and k=4 threshold for strategy A (never re-derived here)
    and is computed at the SAME production frame_step, restricted to the
    full analyzed patch grid instead of round 34's whole-ROI/named-region
    gates.
  - Periodicity significance uses n_shuffles=50 (vs pipeline/analyze.py's
    200) and the phase_randomized null only (vs all 3 methods there) --
    a per-patch, all-5-video diagnostic needs to be cheaper per patch
    than the one-off production pipeline; this is corroborating evidence
    here, not a hard gate, so a coarser p-value is an acceptable
    trade-off (documented, not silently done).

Usage:
    .venv/bin/python3 scripts/motion_observability_diagnostic.py
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

from cardio_organoid.functional.motion_evidence import (  # noqa: E402
    binary_movement_evidence,
    farneback_evidence,
    periodic_motion_strength,
)
from cardio_organoid.motion.image_observability import (  # noqa: E402
    compute_pixel_level_maps,
    normalize_features_within_video,
    reduce_patch_features,
)
from cardio_organoid.motion.motion_confidence import (  # noqa: E402
    direction_stability,
    forward_backward_consistency_field,
    motion_persistence_from_waveform,
    neighbor_vector_agreement,
    temporal_flow_consistency,
)
from cardio_organoid.motion.reference_binary_motion import (  # noqa: E402
    absolute_difference,
    binary_mask_from_diff,
    moving_fraction,
)
from cardio_organoid.motion.reliable_hotspot_candidates import (  # noqa: E402
    STATE_A_RELIABLE_ACTIVE,
    STATE_C_NOT_OBSERVABLE,
    classify_state,
    provisional_flow_confidence_composite,
    provisional_observability_composite,
    rank_hotspot_candidates,
)
from cardio_organoid.motion.spatial_gate import crop_mask_to_roi, dilate_mask  # noqa: E402
from cardio_organoid.motion.spatial_patches import generate_patch_grid  # noqa: E402
from cardio_organoid.segmentation.sam2_segmenter import get_device, load_predictor  # noqa: E402

from _output_safety import add_output_dir_args, resolve_output_dir  # noqa: E402
from d10_secondary_cause_diagnostic import compute_fine_resolution_waveforms  # noqa: E402
from global_boundary_motion_diagnostic import VIDEOS, _fmt, _load_config, _recover_mask_and_roi  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "results" / "motion_observability_diagnostic"

REFERENCE_PATHS = {
    ("early", "conv_a1"): REPO_ROOT / "results" / "manual_reference" / "conv" / "a1.json",
    ("early", "slit_a1"): REPO_ROOT / "results" / "manual_reference" / "slit" / "a1.json",
    ("mature", "Beating_D9_conv_a1"): REPO_ROOT / "results" / "manual_reference" / "mature" / "Beating_D9" / "conv" / "a1.json",
    ("mature", "Beating_D9_slit_a1"): REPO_ROOT / "results" / "manual_reference" / "mature" / "Beating_D9" / "slit" / "a1.json",
    ("mature", "D10_slit_a1"): REPO_ROOT / "results" / "manual_reference" / "mature" / "D10_26.08.2026" / "slit" / "a1.json",
}
# Round 34's already-computed resting-frame index + k=4 threshold for
# strategy A, reused read-only -- never re-derived here.
BINARY_MOVEMENT_SUMMARY_PATH = REPO_ROOT / "results" / "_scratch_runs" / "reference_binary_motion_diagnostic_20260909T113349Z" / "summary.json"

N_FB_CONSISTENCY_SAMPLE_FRAMES = 8
N_PERIODICITY_SHUFFLES = 50
N_PERSISTENCE_WINDOWS = 10


def _regions_from_reference(ref: dict) -> dict[str, list[tuple[int, int]]]:
    if ref.get("regions") and ref.get("activation_order"):
        return {name: [tuple(p) for p in ref["regions"][name]] for name in ref["activation_order"]}
    dominant = ref.get("dominant_beating_region") or []
    return {"dominant_beating_region": [tuple(p) for p in dominant]}


def manual_region_validation(group: str, name: str, state: dict[tuple[int, int], str]) -> dict:
    """OPTIONAL, read-only, POST-HOC labeling of already-computed
    per-patch `state` against a manual reference -- never a detection/
    gating input (see module docstring). Separated into its own function
    (round: modality-aware architecture follow-up) so a video with NO
    manual reference on file -- e.g. every video in the new P36-P39
    longitudinal dataset, which has no `results/manual_reference/*.json`
    entry at all -- can still run the full observability/reliability
    pass without needing a placeholder/dummy reference file. Returns
    `{"manual_reference_available": False, "regions": {}}` in that case,
    never raising and never silently treating "no reference" as "zero
    manual regions confirmed" (a materially different, false, claim)."""
    ref_path = REFERENCE_PATHS.get((group, name))
    if ref_path is None or not ref_path.exists():
        print(f"  [manual validation] no manual reference on file for {group}/{name} -- skipping (not required; gating/state above is unaffected)")
        return {"manual_reference_available": False, "regions": {}}

    ref = json.loads(ref_path.read_text())
    manual_regions = _regions_from_reference(ref)
    regions = {}
    for rname, patches in manual_regions.items():
        analyzed = [p for p in patches if p in state]
        region_states = [state[p] for p in analyzed]
        regions[rname] = {
            "n_patches_in_manual_region": len(patches), "n_analyzed": len(analyzed),
            "state_counts": {s: region_states.count(s) for s in set(region_states)} if region_states else {},
            "fraction_state_A": (region_states.count(STATE_A_RELIABLE_ACTIVE) / len(region_states)) if region_states else None,
        }
        print(f"    manual region [{rname}]: {regions[rname]['state_counts']}")
    return {"manual_reference_available": True, "regions": regions}


def read_gray_roi_at(video_path: Path, frame_index: int, roi_bbox_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = roi_bbox_xyxy
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]


def sample_forward_backward_consistency(video_path: Path, roi, sample_indices: list[int], cells) -> dict[tuple[int, int], float]:
    x0, y0, x1, y1 = roi.roi_bbox_xyxy
    local_roi = replace(roi, roi_bbox_xyxy=(0, 0, x1 - x0, y1 - y0))
    from cardio_organoid.motion.raft_flow import compute_dense_optical_flow

    per_patch_errors = {(c.row, c.col): [] for c in cells}
    cap = cv2.VideoCapture(str(video_path))
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx - 1))
        ok_a, frame_a = cap.read()
        ok_b, frame_b = cap.read()
        if not (ok_a and ok_b):
            continue
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
        flow_fwd = compute_dense_optical_flow(gray_a, gray_b, local_roi, method="opencv_farneback")
        flow_bwd = compute_dense_optical_flow(gray_b, gray_a, local_roi, method="opencv_farneback")
        error_field = forward_backward_consistency_field(flow_fwd, flow_bwd)
        for c in cells:
            bx0, by0, bx1, by1 = c.bbox_xyxy
            per_patch_errors[(c.row, c.col)].append(float(np.nanmean(error_field[by0:by1, bx0:bx1])))
    cap.release()
    return {pid: (float(np.mean(v)) if v else None) for pid, v in per_patch_errors.items()}


def compute_binary_movement_series(video_path: Path, roi_bbox_xyxy, ref_frame: np.ndarray, threshold: float, frame_step: int, cells) -> dict[tuple[int, int], list[float]]:
    x0, y0, x1, y1 = roi_bbox_xyxy
    series = {(c.row, c.col): [] for c in cells}
    cap = cv2.VideoCapture(str(video_path))
    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if idx % frame_step == 0:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
            diff = absolute_difference(gray, ref_frame)
            binary = binary_mask_from_diff(diff, threshold)
            for c in cells:
                bx0, by0, bx1, by1 = c.bbox_xyxy
                series[(c.row, c.col)].append(moving_fraction(binary[by0:by1, bx0:bx1]))
        idx += 1
    cap.release()
    return series


def _zscore_dict(values: dict) -> dict:
    keys = [k for k, v in values.items() if v is not None]
    arr = np.array([values[k] for k in keys], dtype=float)
    mean, std = arr.mean(), arr.std()
    z = (arr - mean) / std if std > 0 else np.zeros_like(arr)
    result = {k: float(v) for k, v in zip(keys, z)}
    for k, v in values.items():
        if v is None:
            result[k] = None
    return result


def process_video(video_cfg: dict, config: dict, predictor, output_dir: Path, binary_summary: dict | None, morphology_mask=None) -> dict:
    group, name = video_cfg["group"], video_cfg["name"]
    print(f"\n=== {group}/{name} ===")

    morphology_mask, roi, equiv_diam, frame_count = _recover_mask_and_roi(video_cfg["source_video"], config, predictor, morphology_mask)
    roi_bbox = roi.roi_bbox_xyxy
    exact_mask_roi = crop_mask_to_roi(morphology_mask, roi_bbox)
    analysis_region = dilate_mask(exact_mask_roi, config["motion"]["signal_extraction"]["gate_dilation_px"])
    cell_size_px = max(1, int(round(config["motion"]["spatial_patches"]["cell_size_ratio"] * equiv_diam)))
    cells = generate_patch_grid(analysis_region, cell_size_px, config["motion"]["spatial_patches"]["min_coverage_fraction"])
    print(f"  n_patches_analyzed={len(cells)} (full grid, NOT manual-region-derived)")

    qc = json.loads((video_cfg["motion_dir"] / "motion_pipeline_qc.json").read_text())
    fps, frame_step = qc["fps"], qc["frame_step"]
    sample_interval_s = frame_step / fps

    # -- Phase 1: image observability (single representative frame) --
    rep_index = frame_count // 2
    rep_gray = read_gray_roi_at(video_cfg["source_video"], rep_index, roi_bbox)
    maps = compute_pixel_level_maps(rep_gray)
    raw_obs_features = reduce_patch_features(maps, rep_gray, cells)
    normalized_obs_features = normalize_features_within_video(raw_obs_features)
    observability_composite = {pid: provisional_observability_composite(feat) for pid, feat in normalized_obs_features.items()}

    # -- Phase 2/3 shared data: full-grid Farneback pass at PRODUCTION frame_step --
    patch_bboxes = {(c.row, c.col): c.bbox_xyxy for c in cells}
    print(f"  computing full-grid Farneback pass at production frame_step={frame_step} ...")
    fine = compute_fine_resolution_waveforms(video_cfg["source_video"], roi, frame_step, patch_bboxes)

    fb_ev = {pid: farneback_evidence(fine[pid]["radial"], fine[pid]["tangential"], fine[pid]["magnitude"]) for pid in patch_bboxes}
    periodicity = {
        pid: periodic_motion_strength(fine[pid]["radial"], sample_interval_s, n_shuffles=N_PERIODICITY_SHUFFLES, rng=np.random.default_rng(hash(pid) % (2**31)))
        for pid in patch_bboxes
    }
    temporal_consistency = {pid: temporal_flow_consistency(fine[pid]["radial"]) for pid in patch_bboxes}
    direction_stab = {pid: direction_stability(fine[pid]["radial"], fine[pid]["tangential"]) for pid in patch_bboxes}
    persistence = {pid: motion_persistence_from_waveform(fine[pid]["radial"], N_PERSISTENCE_WINDOWS) for pid in patch_bboxes}
    radial_waveforms = {pid: fine[pid]["radial"] for pid in patch_bboxes}
    neighbor_agreement = neighbor_vector_agreement(radial_waveforms)

    # -- Phase 2: forward/backward consistency (sampled frame pairs) --
    sample_indices = np.linspace(frame_count // 6, 5 * frame_count // 6, N_FB_CONSISTENCY_SAMPLE_FRAMES, dtype=int).tolist()
    fb_consistency_error = sample_forward_backward_consistency(video_cfg["source_video"], roi, sample_indices, cells)
    fb_consistency_z = _zscore_dict(fb_consistency_error)  # lower error -> should CONTRIBUTE POSITIVELY once negated below

    # -- Phase 3: binary-movement evidence (reuses round-34 threshold, read-only) --
    binary_ev = {pid: None for pid in patch_bboxes}
    if binary_summary is not None:
        ref_a_index = binary_summary["reference_a_frame_index"]
        threshold = binary_summary["threshold_candidates"]["A_resting_frame"]["k=4"]
        if threshold is not None:
            ref_frame = read_gray_roi_at(video_cfg["source_video"], ref_a_index, roi_bbox)
            binary_series = compute_binary_movement_series(video_cfg["source_video"], roi_bbox, ref_frame, threshold, frame_step, cells)
            binary_ev = {pid: binary_movement_evidence(np.array(v)) for pid, v in binary_series.items()}
        else:
            print("  [binary evidence] round-34 threshold unavailable for this video -- skipping (documented, not silently guessed)")
    else:
        print("  [binary evidence] round-34 summary.json not found for this video -- skipping")

    # -- Phase 4: self-relative motion x observability state map --
    motion_score = {pid: fb_ev[pid]["farneback_magnitude_mean"] or 0.0 for pid in patch_bboxes}
    motion_median = float(np.median(list(motion_score.values())))
    observability_median = float(np.median(list(observability_composite.values())))
    state = {
        pid: classify_state(motion_score[pid], observability_composite[pid], motion_median, observability_median)
        for pid in patch_bboxes
    }

    flow_confidence = {
        pid: provisional_flow_confidence_composite(
            (-fb_consistency_z[pid] if fb_consistency_z.get(pid) is not None else None),
            temporal_consistency[pid], neighbor_agreement[pid], direction_stab[pid],
        )
        for pid in patch_bboxes
    }

    # -- Phase 5: ranking --
    records = []
    for pid in patch_bboxes:
        records.append({
            "patch_id": list(pid), "motion_score": motion_score[pid], "image_observability": observability_composite[pid],
            "flow_confidence": flow_confidence[pid], "persistence": persistence[pid]["temporal_persistence_fraction"],
            "periodicity": periodicity[pid]["spectral_concentration"], "state": state[pid],
        })
    ranked = rank_hotspot_candidates([dict(r, patch_id=tuple(r["patch_id"])) for r in records])
    for r in ranked:
        r["patch_id"] = list(r["patch_id"])

    # -- assemble full per-patch record for JSON --
    per_patch = {}
    for pid in patch_bboxes:
        key = f"{pid[0]}_{pid[1]}"
        per_patch[key] = {
            "raw_observability_features": raw_obs_features[pid],
            "normalized_observability_features": normalized_obs_features[pid],
            "observability_composite_provisional": observability_composite[pid],
            "farneback_evidence": fb_ev[pid],
            "periodicity_evidence": periodicity[pid],
            "temporal_flow_consistency": temporal_consistency[pid],
            "direction_stability": direction_stab[pid],
            "motion_persistence": persistence[pid],
            "neighbor_vector_agreement": neighbor_agreement[pid],
            "forward_backward_consistency_error": fb_consistency_error[pid],
            "binary_movement_evidence": binary_ev[pid],
            "flow_confidence_composite_provisional": flow_confidence[pid],
            "motion_score": motion_score[pid],
            "state": state[pid],
        }
    ranked_by_key = {f"{r['patch_id'][0]}_{r['patch_id'][1]}": r for r in ranked}
    for key, r in ranked_by_key.items():
        per_patch[key]["final_provisional_confidence"] = r["final_provisional_confidence"]
        per_patch[key]["rejection_reason"] = r["rejection_reason"]
        per_patch[key]["spatial_support"] = r["spatial_support"]

    state_counts = {s: sum(1 for v in state.values() if v == s) for s in set(state.values())}
    print(f"  state counts: {state_counts}")

    validation = manual_region_validation(group, name, state)

    result = {
        "video": f"{group}/{name}", "fps": fps, "frame_step": frame_step, "n_patches": len(cells),
        "motion_median": motion_median, "observability_median": observability_median,
        "state_counts": state_counts, "manual_region_validation": validation,
        "top_10_hotspot_candidates": ranked[:10],
        "per_patch": per_patch,
    }

    # -- QC panel --
    render_qc_panel(video_cfg, rep_gray, morphology_mask, roi_bbox, cells, motion_score, observability_composite, flow_confidence, state, ranked_by_key, output_dir)

    return result, cells, roi_bbox, cell_size_px, motion_score, observability_composite, state, ranked_by_key, fine, sample_interval_s


def _paint(cells, values: dict, shape: tuple[int, int], default=np.nan) -> np.ndarray:
    canvas = np.full(shape, default, dtype=float)
    for c in cells:
        v = values.get((c.row, c.col))
        if v is None:
            continue
        x0, y0, x1, y1 = c.bbox_xyxy
        canvas[y0:y1, x0:x1] = v
    return canvas


def render_qc_panel(video_cfg, rep_gray, morphology_mask, roi_bbox, cells, motion_score, observability, flow_confidence, state, ranked_by_key, output_dir) -> None:
    group, name = video_cfg["group"], video_cfg["name"]
    x0, y0, x1, y1 = roi_bbox
    shape = rep_gray.shape
    mask_roi = crop_mask_to_roi(morphology_mask, roi_bbox)

    state_code = {"A_reliable_active_candidate": 3, "B_likely_genuinely_weak": 2, "D_ambiguous_needs_independent_confirmation": 1, "C_NOT_OBSERVABLE_do_not_interpret_as_no_motion": 0}
    state_map = _paint(cells, {pid: state_code[s] for pid, s in state.items()}, shape)
    not_observable_map = _paint(cells, {pid: (1.0 if s == STATE_C_NOT_OBSERVABLE else 0.0) for pid, s in state.items()}, shape)
    hotspot_conf = {tuple(r["patch_id"]): r["final_provisional_confidence"] for r in ranked_by_key.values()}
    hotspot_map = _paint(cells, hotspot_conf, shape)
    rejected = {tuple(r["patch_id"]): (r["rejection_reason"] is not None) for r in ranked_by_key.values()}

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    axes[0].imshow(rep_gray, cmap="gray"); axes[0].set_title("original frame (ROI crop)", fontsize=9)
    axes[1].imshow(rep_gray, cmap="gray"); axes[1].imshow(np.ma.masked_where(~mask_roi, mask_roi), cmap="autumn", alpha=0.5)
    axes[1].set_title("SAM2 mask overlay", fontsize=9)
    im2 = axes[2].imshow(_paint(cells, motion_score, shape), cmap="viridis"); axes[2].set_title("motion magnitude (Farneback)", fontsize=9); plt.colorbar(im2, ax=axes[2], fraction=0.046)
    im3 = axes[3].imshow(_paint(cells, observability, shape), cmap="plasma"); axes[3].set_title("image observability (provisional composite)", fontsize=9); plt.colorbar(im3, ax=axes[3], fraction=0.046)
    flow_conf_plot = {pid: v for pid, v in flow_confidence.items() if v is not None}
    im4 = axes[4].imshow(_paint(cells, flow_conf_plot, shape), cmap="cividis"); axes[4].set_title("flow confidence (provisional composite)", fontsize=9); plt.colorbar(im4, ax=axes[4], fraction=0.046)
    im5 = axes[5].imshow(state_map, cmap="RdYlGn", vmin=0, vmax=3); axes[5].set_title("reliable-motion state map\n(green=A reliable_active, yellow=B weak,\norange=D ambiguous, red=C NOT_OBSERVABLE)", fontsize=8)
    im6 = axes[6].imshow(not_observable_map, cmap="Reds", vmin=0, vmax=1); axes[6].set_title("LOW_OBSERVABILITY map (state C)\n-- NOT an assertion of no motion", fontsize=8)
    axes[7].imshow(rep_gray, cmap="gray")
    for pid, r in ranked_by_key.items():
        c = next((c for c in cells if (c.row, c.col) == tuple(r["patch_id"])), None)
        if c is None:
            continue
        cx0, cy0, cx1, cy1 = c.bbox_xyxy
        color = "lime" if r["rejection_reason"] is None else ("orange" if r["rejection_reason"] == "AMBIGUOUS" else ("red" if r["rejection_reason"] == "NOT_OBSERVABLE" else "gray"))
        rect = plt.Rectangle((cx0, cy0), cx1 - cx0, cy1 - cy0, fill=False, edgecolor=color, linewidth=1.0)
        axes[7].add_patch(rect)
    axes[7].set_title("provisional hotspot candidates\n(green=kept, gray=LOW_CONF, orange=AMBIGUOUS, red=NOT_OBSERVABLE)", fontsize=8)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{group}/{name} -- motion-evidence + observability QC (round 36)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    safe_name = f"{group}_{name}".replace("/", "_")
    fig.savefig(output_dir / f"{safe_name}_QC_PANEL.png", dpi=120)
    plt.close(fig)
    print(f"  wrote {output_dir / f'{safe_name}_QC_PANEL.png'}")


def render_d10_qc_video(video_cfg, roi_bbox, cells, motion_score, observability, state, ranked_by_key, fine, sample_interval_s, output_dir) -> None:
    x0, y0, x1, y1 = roi_bbox
    cap = cv2.VideoCapture(str(video_cfg["source_video"]))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    hotspot_pids = {tuple(r["patch_id"]) for r in ranked_by_key.values() if r["rejection_reason"] is None}
    reliable_wf = np.mean(np.stack([fine[pid]["radial"] for pid in hotspot_pids]), axis=0) if hotspot_pids else None

    panel_w, panel_h = 640, 480
    canvas_w = panel_w * 2
    plot_h_px = 220

    dpi = 100
    # figsize matched EXACTLY to (canvas_w, plot_h_px) so no post-hoc resize
    # is needed before vstack-ing with the top row (a resize here would
    # also invalidate ax.transData's pixel mapping used by time_to_px).
    fig, ax = plt.subplots(figsize=(canvas_w / dpi, plot_h_px / dpi), dpi=dpi)
    if reliable_wf is not None:
        t = np.arange(len(reliable_wf)) * sample_interval_s
        ax.plot(t, reliable_wf, color="seagreen", linewidth=1.0)
    ax.set_title("mean radial waveform across kept hotspot candidates", fontsize=8)
    ax.set_xlabel("time (s)", fontsize=7)
    fig.tight_layout()
    fig.canvas.draw()
    w_px, h_px = fig.canvas.get_width_height()
    plot_img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h_px, w_px, 4), cv2.COLOR_RGBA2BGR)
    assert plot_img.shape[1] == canvas_w, f"plot_img width {plot_img.shape[1]} != canvas_w {canvas_w}"

    def time_to_px(t_val):
        disp_x, _ = ax.transData.transform((t_val, 0))
        return int(round(disp_x))
    plt.close(fig)

    canvas_h = panel_h + plot_img.shape[0]
    out_path = output_dir / "D10_MOTION_OBSERVABILITY_QC.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas_w, canvas_h))

    state_color = {
        "A_reliable_active_candidate": (0, 255, 0), "B_likely_genuinely_weak": (0, 255, 255),
        "D_ambiguous_needs_independent_confirmation": (0, 165, 255), "C_NOT_OBSERVABLE_do_not_interpret_as_no_motion": (0, 0, 255),
    }
    cap = cv2.VideoCapture(str(video_cfg["source_video"]))
    for frame_idx in range(frame_count):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        panel_a = frame_bgr.copy()
        cv2.rectangle(panel_a, (x0, y0), (x1, y1), (0, 255, 255), 2)
        h0, w0 = panel_a.shape[:2]
        scale = min(panel_w / w0, panel_h / h0)
        panel_a = cv2.resize(panel_a, (int(w0 * scale), int(h0 * scale)))
        canvas_a = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        canvas_a[: panel_a.shape[0], : panel_a.shape[1]] = panel_a

        roi_crop = frame_bgr[y0:y1, x0:x1].copy()
        for c in cells:
            pid = (c.row, c.col)
            cx0, cy0, cx1, cy1 = c.bbox_xyxy
            color = state_color[state[pid]]
            thickness = 2 if pid in hotspot_pids else 1
            cv2.rectangle(roi_crop, (cx0, cy0), (cx1, cy1), color, thickness)
        hr, wr = roi_crop.shape[:2]
        scale_r = min(panel_w / wr, panel_h / hr)
        roi_resized = cv2.resize(roi_crop, (int(wr * scale_r), int(hr * scale_r)))
        canvas_b = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        canvas_b[: roi_resized.shape[0], : roi_resized.shape[1]] = roi_resized

        top_row = np.hstack([canvas_a, canvas_b])
        cv2.putText(top_row, "original + ROI", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(top_row, "green=A red=NOT_OBS orange=D yellow=B (thick=kept hotspot)", (panel_w + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(top_row, f"t={frame_idx / fps:6.2f}s", (10, panel_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        plot_frame = plot_img.copy()
        px = time_to_px(frame_idx / fps)
        cv2.line(plot_frame, (px, 0), (px, plot_frame.shape[0]), (0, 255, 255), 1)
        canvas = np.vstack([top_row, plot_frame])
        writer.write(canvas)
    cap.release()
    writer.release()
    print(f"  wrote {out_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_output_dir_args(parser, OUTPUT_DIR, "motion_observability_diagnostic")
    args = parser.parse_args()
    output_dir = resolve_output_dir(args, OUTPUT_DIR, "motion_observability_diagnostic")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_config()
    sam2_cfg = config["segmentation"]["sam2"]
    device = get_device(sam2_cfg["device"])
    predictor = load_predictor(checkpoint_path=REPO_ROOT / sam2_cfg["checkpoint_path"], model_cfg=sam2_cfg["model_cfg"], device=device)

    binary_summaries = {}
    if BINARY_MOVEMENT_SUMMARY_PATH.exists():
        all_binary = json.loads(BINARY_MOVEMENT_SUMMARY_PATH.read_text())
        for key_str, v in all_binary.items():
            group, name = key_str.split("/", 1)
            binary_summaries[(group, name)] = v
    else:
        print(f"[warn] round-34 binary summary not found at {BINARY_MOVEMENT_SUMMARY_PATH} -- binary evidence will be skipped for all videos")

    all_results = {}
    for video_cfg in VIDEOS:
        key = (video_cfg["group"], video_cfg["name"])
        result, cells, roi_bbox, cell_size_px, motion_score, observability, state, ranked_by_key, fine, sample_interval_s = process_video(
            video_cfg, config, predictor, output_dir, binary_summaries.get(key),
        )
        all_results[f"{key[0]}/{key[1]}"] = result
        if key == ("mature", "D10_slit_a1"):
            print("  rendering D10 QC video ...")
            try:
                render_d10_qc_video(video_cfg, roi_bbox, cells, motion_score, observability, state, ranked_by_key, fine, sample_interval_s, output_dir)
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] D10 QC video rendering failed ({e}); QC panel PNG above stands as the documented fallback")

    with open(output_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"\nWrote motion-observability diagnostic for {len(all_results)} video(s) to {output_dir}")


if __name__ == "__main__":
    main()
