"""Main functional/motion analysis pipeline (Farneback-based).

DECISION (validated -- see configs/default.yaml: motion.optical_flow and
results/phase2_raft_validation/): Farneback is the primary optical-flow
method for this project. RAFT's preprocessing was verified correct, but it
shows a texture-dependent artifact on this bright-field organoid interior
and worse forward-backward consistency than Farneback even at the
organoid boundary. RAFT is preserved as a documented comparison baseline
(raft_backend.py and the round 4-5 scripts), not used here.

Because the current (beating-onset) sample videos show contraction
localized to part of the organoid rather than synchronized across the
whole body, this pipeline never reduces to a single whole-organoid
averaged waveform as its primary signal. Instead:

video (full duration) -> coarse organoid ROI (dilated morphology mask) ->
spatial patch grid -> per-patch full-video waveform -> active patches ->
spatially contiguous active regions -> one waveform per region ->
region-to-region synchronization (interface only -- not interpreted as
biology yet).

IMPORTANT correction from an earlier round (round 6 attempted this, but
used organoid-derived data as if it were a "no motion" null): every
sample video in this pilot set is expected to contain *some* real
localized beating -- there is no genuine non-beating organoid recording
available, so no organoid region (interior, "quiet-looking" patches, etc.)
may be treated as a biological null, and "most patches should be
inactive" must not be assumed either. The *only* legitimate null sources
are (a) identical-frame pairs (zero true motion by construction) and (b)
static background regions measurably far from the organoid -- both
estimate pure algorithmic/measurement noise, not biology.

Active-patch classification therefore evaluates multiple independent
lines of evidence together, none of them a fixed-percentage rule:
  1. amplitude vs. the measurement noise floor (background-region
     statistics, see activity_significance.py) -- necessary, not
     sufficient, since noise can also occasionally exceed a fixed cutoff.
  2. periodicity vs. a null built by temporally shuffling *the patch's own
     waveform* (diagnostics.periodicity_shuffle_test) -- since no non-
     beating sample exists to compare against, the shuffle is a
     self-contained surrogate that destroys periodicity while preserving
     the value distribution.
  3. temporal persistence (windowed_std / temporal_persistence_fraction)
     -- recorded and visualized, not a third hard gate (see below).
  4. spatial coherence with 4-connected grid neighbors
     (spatial_patches.neighbor_coherence) -- recorded and visualized.
A patch is called "active" only if (1) AND (2) hold -- two independent
statistical tests against real null distributions. (3) and (4) are always
computed and saved/visualized alongside, and used to rank/qualify active
patches, but are deliberately not additional hard AND-gates: stacking four
independent significance tests would bias the result toward finding
nothing regardless of the true underlying activity, which is exactly the
kind of unjustified assumption this round was asked to remove.

Explicitly not a biological metric: no BPM, no beat/peak detection.
`active_region_fraction` is a provisional diagnostic, not a validated
measurement. The goal for the current sample set is to check whether
detected active regions line up with beating visible in the raw video
(QC overlay). Streams through the video frame-by-frame rather than
buffering it, so memory use stays small regardless of video length.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cardio_organoid.pipeline.segmentation_interface import assert_legacy_fixed_index_allowed
from cardio_organoid.morphology.features import area, equivalent_diameter
from cardio_organoid.motion.activity_significance import (
    null_reference_stats,
    significance_threshold,
)
from cardio_organoid.motion.background_reference import (
    draw_patch_qc,
    select_background_patches_multi_ring,
)
from cardio_organoid.motion.diagnostics import (
    dominant_spectral_peak,
    first_secondary_autocorrelation_peak,
    periodicity_shuffle_test,
    spectral_concentration,
)
from cardio_organoid.motion.radial_flow import radial_motion_from_centroid, remove_global_drift
from cardio_organoid.motion.raft_flow import compute_dense_optical_flow, compute_motion_roi
from cardio_organoid.motion.spatial_gate import crop_mask_to_roi, dilate_mask
from cardio_organoid.motion.spatial_patches import (
    active_region_fraction,
    build_activity_heatmap,
    cluster_active_patches_into_regions,
    cross_patch_synchronization,
    generate_patch_grid,
    neighbor_coherence,
    overlay_heatmap,
    region_waveform,
    temporal_persistence_fraction,
    windowed_std,
)
from cardio_organoid.segmentation.sam2_segmenter import segment_frame


def deterministic_seed(*parts) -> int:
    """A 32-bit RNG seed derived from `parts` that is stable ACROSS
    process runs -- unlike Python's built-in `hash()` on a string (or any
    tuple containing one), which is randomized per-process by default
    (PYTHONHASHSEED) for security and therefore silently changes every
    run. That previously made periodicity_shuffle_test's shuffle draws
    (and therefore periodicity_p_value/periodicity_z, though not any
    other score) non-reproducible from one run of the pipeline to the
    next on identical input -- a reproducibility bug, not a change to the
    periodicity statistic's definition. Uses SHA-256 (stdlib, no new
    dependency) over the parts' string representation.
    """
    key = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:4], "big")


def _region_qc_overlay(frame_bgr_roi: np.ndarray, cells, regions) -> np.ndarray:
    """Distinct color per active region, for visually checking whether
    detected regions line up with beating visible in the raw frame."""
    palette = [
        (255, 0, 255), (0, 165, 255), (255, 255, 0), (0, 255, 0),
        (255, 0, 0), (0, 0, 255), (255, 128, 0), (128, 0, 255),
    ]
    cell_by_id = {(c.row, c.col): c for c in cells}
    vis = frame_bgr_roi.copy()
    for region in regions:
        color = palette[region.region_id % len(palette)]
        for pid in region.patch_ids:
            x0, y0, x1, y1 = cell_by_id[pid].bbox_xyxy
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
        rx0, ry0, _, _ = cell_by_id[region.patch_ids[0]].bbox_xyxy
        cv2.putText(
            vis, f"R{region.region_id}", (rx0 + 3, ry0 + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
    return vis


def resolve_representative_mask(predictor, rep_frame_rgb, config: dict, morphology_mask=None) -> tuple:
    """Return (mask, segmentation_source). An injected mask (from a
    segmentation-interface record) always wins and SAM2 is not called.
    Without one, the legacy behaviour is kept: internal SAM2 + the fixed
    configs candidate_index."""
    if morphology_mask is not None:
        if morphology_mask.shape != rep_frame_rgb.shape[:2]:
            raise ValueError(
                f"injected mask shape {morphology_mask.shape} != representative frame shape {rep_frame_rgb.shape[:2]}"
            )
        return morphology_mask.astype(bool), "injected_segmentation_interface_mask"
    if predictor is None:
        raise ValueError("either a predictor or an injected morphology_mask is required")
    assert_legacy_fixed_index_allowed("pipeline.analyze.resolve_representative_mask")
    idx = config["segmentation"]["provisional_primary"]["candidate_index"]
    candidates = segment_frame(predictor, rep_frame_rgb)
    return candidates[idx].mask, f"legacy_internal_sam2_fixed_candidate_index_{idx}"


def run_motion_pipeline(
    video_path: Path, predictor, config: dict, output_dir: Path, excluded_reason: str | None = None,
    morphology_mask=None, mask_provenance: dict | None = None,
) -> dict:
    """Run the main Farneback-based spatial motion pipeline on the full
    duration of `video_path`. `excluded_reason` should be checked by the
    caller (e.g. run_sam2_phase1.excluded_video_reason) and passed through
    so this function can raise consistently -- kept as a parameter rather
    than importing that check here to avoid a scripts->pipeline dependency.

    `morphology_mask` (optional): mask from a segmentation-interface record,
    defined on the video's representative (midpoint) frame. When given, no
    SAM2 call is made here; `mask_provenance` is recorded in the QC output.
    """
    if excluded_reason is not None:
        raise ValueError(f"{video_path} excluded from analysis: {excluded_reason}")
    output_dir.mkdir(parents=True, exist_ok=True)

    sam2_cfg = config["segmentation"]["sam2"]
    motion_cfg = config["motion"]
    se_cfg = motion_cfg["signal_extraction"]
    sp_cfg = motion_cfg["spatial_patches"]
    pipe_cfg = motion_cfg["pipeline"]
    null_cfg = pipe_cfg["null_reference"]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # file-recorded fps, per the project's timebase decision

    # Representative frame (video midpoint, per docs/analysis_design.md #4
    # -- not cardiac-phase-aware yet) for segmentation / ROI / patch grid.
    rep_index = frame_count // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, rep_index)
    ok, rep_frame_bgr = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read representative frame {rep_index} from {video_path}")
    rep_frame_rgb = cv2.cvtColor(rep_frame_bgr, cv2.COLOR_BGR2RGB)
    morphology_mask, segmentation_source = resolve_representative_mask(predictor, rep_frame_rgb, config, morphology_mask)
    equivalent_diameter_px = equivalent_diameter(area(morphology_mask))

    roi = compute_motion_roi(
        morphology_mask, expansion_px=motion_cfg["roi"]["expansion_px"], frame_shape=rep_frame_rgb.shape[:2]
    )
    x0, y0, x1, y1 = roi.roi_bbox_xyxy
    centroid_xy = roi.centroid_xy_in_roi()
    local_roi = replace(roi, roi_bbox_xyxy=(0, 0, x1 - x0, y1 - y0))

    exact_mask_roi = crop_mask_to_roi(morphology_mask, roi.roi_bbox_xyxy)
    analysis_region = dilate_mask(exact_mask_roi, se_cfg["gate_dilation_px"])
    cell_size_px = max(1, int(round(sp_cfg["cell_size_ratio"] * equivalent_diameter_px)))
    cells = generate_patch_grid(analysis_region, cell_size_px, sp_cfg["min_coverage_fraction"])
    if not cells:
        raise RuntimeError(f"{video_path}: no patch cells passed the coverage threshold")

    # --- Null/background reference cells: SAME size as the organoid grid
    # cells (patch_size_ratio deliberately reuses cell_size_ratio, not an
    # independent config value) so the noise-floor comparison is
    # apples-to-apples, not confounded by averaging-region size. Screened
    # for brightness/temporal-stability using a short buffer (cheap; only
    # used to pick *positions*, not for the activity measurement itself).
    screen_cap = cv2.VideoCapture(str(video_path))
    screen_frames = []
    for _ in range(int(round(se_cfg["buffer_duration_s"] * fps)) + 1):
        ok, f = screen_cap.read()
        if not ok:
            break
        screen_frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1])
    screen_cap.release()
    null_candidates, _null_gate = select_background_patches_multi_ring(
        frame_gray_roi=screen_frames[0],
        frames_gray_roi_buffer=screen_frames,
        centroid_xy=centroid_xy,
        equivalent_diameter_px=equivalent_diameter_px,
        analysis_region_mask=analysis_region,
        n_angles=null_cfg["n_angles"],
        radius_ratios=null_cfg["radius_ratios"],
        patch_size_ratio=sp_cfg["cell_size_ratio"],
    )
    null_cells = [c for c in null_candidates if c.kept]
    if not null_cells:
        raise RuntimeError(f"{video_path}: no null/background reference cells survived screening")
    min_target_patches = null_cfg["min_target_patches"]
    background_reference_qc = {
        "n_candidates_generated": len(null_candidates),
        "n_kept": len(null_cells),
        "min_target_patches": min_target_patches,
        "low_confidence_background_reference": len(null_cells) < min_target_patches,
        "reject_reason_counts": {
            reason: sum(1 for c in null_candidates if c.reject_reason == reason)
            for reason in sorted({c.reject_reason for c in null_candidates if c.reject_reason})
        },
    }

    step = sp_cfg["frame_step"]

    # --- Stream through the FULL video once, reducing to per-cell scalars
    # on the fly (both organoid-grid cells and null cells share this one
    # pass -- never buffering all frames or all flow fields). ---
    patch_radial: dict[tuple[int, int], list[float]] = {(c.row, c.col): [] for c in cells}
    null_radial: dict[int, list[float]] = {i: [] for i in range(len(null_cells))}
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    prev_gray_roi = None
    frame_idx = 0
    n_pairs = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            gray_roi = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
            if prev_gray_roi is not None:
                flow = compute_dense_optical_flow(prev_gray_roi, gray_roi, local_roi, method="opencv_farneback")
                corrected, _drift = remove_global_drift(flow)
                radial = radial_motion_from_centroid(corrected, centroid_xy)
                for cell in cells:
                    cx0, cy0, cx1, cy1 = cell.bbox_xyxy
                    patch_radial[(cell.row, cell.col)].append(float(np.nanmean(radial[cy0:cy1, cx0:cx1])))
                for i, null_cell in enumerate(null_cells):
                    nx0, ny0, nx1, ny1 = null_cell.bbox_xyxy
                    null_radial[i].append(float(np.nanmean(radial[ny0:ny1, nx0:nx1])))
                n_pairs += 1
            prev_gray_roi = gray_roi
        frame_idx += 1
    cap.release()

    patch_waveforms_by_id = {pid: np.array(vals) for pid, vals in patch_radial.items()}
    null_waveforms_by_id = {i: np.array(vals) for i, vals in null_radial.items()}
    sample_interval_s = step / fps

    def _activity_record(wf: np.ndarray) -> dict:
        w_stds = windowed_std(wf, pipe_cfg["n_windows"])
        return {
            "radial_std": float(np.nanstd(wf)),
            "radial_ptp": float(np.nanmax(wf) - np.nanmin(wf)),
            "window_stds": w_stds.tolist(),
            "temporal_persistence_fraction": temporal_persistence_fraction(
                w_stds, pipe_cfg["persistence_ratio_threshold"]
            ),
            "autocorr_secondary_peak": first_secondary_autocorrelation_peak(wf, sample_interval_s),
            "spectral_peak": dominant_spectral_peak(wf, sample_interval_s),
            "spectral_concentration": spectral_concentration(wf, sample_interval_s),
        }

    # --- Per-patch AND per-null-cell activity records (identical pipeline) ---
    patch_records = {pid: _activity_record(wf) for pid, wf in patch_waveforms_by_id.items()}
    null_records = {i: _activity_record(wf) for i, wf in null_waveforms_by_id.items()}

    radial_stds = {pid: rec["radial_std"] for pid, rec in patch_records.items()}
    persistence_scores = {pid: rec["temporal_persistence_fraction"] for pid, rec in patch_records.items()}

    # Pool each null cell's *windowed* stds (not just one whole-video std
    # per cell) as the noise-floor sample set: with only ~3-4 kept
    # background cells, a single scalar each gives far too few samples to
    # estimate a reliable mean/std for a z-based threshold. Using the
    # window_stds already computed for each null cell (n_windows per cell)
    # gives ~3-4x n_windows samples instead -- still drawn only from
    # algorithmic/background-noise sources, never from organoid patches.
    null_stds = [s for rec in null_records.values() for s in rec["window_stds"]]

    magnitude_threshold = significance_threshold(null_stds, pipe_cfg["active_z_threshold"])
    null_mean_std = null_reference_stats(null_stds)
    amplitude_z_scores = {
        pid: (s - null_mean_std["mean"]) / null_mean_std["std"] if null_mean_std["std"] > 0 else None
        for pid, s in radial_stds.items()
    }
    active_by_magnitude = {pid for pid, s in radial_stds.items() if s > magnitude_threshold}

    # --- Periodicity: self-contained surrogate-data tests per patch (no
    # non-beating sample exists to compare against -- see module
    # docstring). Computed under all 3 null-generation methods so their
    # permissiveness can be compared directly rather than trusting one by
    # default -- full_shuffle was found, on this project's own data, to
    # flag implausibly many patches (e.g. 70/70 in one video), consistent
    # with it being too permissive for waveforms with short-range
    # smoothness unrelated to true periodicity. ---
    periodicity_alpha = pipe_cfg["periodicity_p_value_threshold"]
    null_methods = pipe_cfg["periodicity_null_methods"]
    for pid, wf in patch_waveforms_by_id.items():
        patch_records[pid]["periodicity_by_null_method"] = {
            method: periodicity_shuffle_test(
                wf, n_shuffles=pipe_cfg["periodicity_n_shuffles"], max_lag=pipe_cfg["periodicity_max_lag"],
                rng=np.random.default_rng(deterministic_seed(pid, method)),
                null_method=method, block_size=pipe_cfg["periodicity_block_size"],
            )
            for method in null_methods
        }
    primary_periodicity_method = pipe_cfg["periodicity_primary_null_method"]
    periodicity_z_scores = {
        pid: rec["periodicity_by_null_method"][primary_periodicity_method]["z_score"]
        for pid, rec in patch_records.items()
    }
    active_by_periodicity_per_method = {
        method: {
            pid for pid, rec in patch_records.items()
            if rec["periodicity_by_null_method"][method]["p_value"] < periodicity_alpha
        }
        for method in null_methods
    }
    active_by_periodicity = active_by_periodicity_per_method[primary_periodicity_method]

    # --- Spatial coherence with 4-connected neighbors: recorded for every
    # patch, not a hard gate (see module docstring for why). ---
    for pid in patch_waveforms_by_id:
        patch_records[pid]["neighbor_coherence"] = neighbor_coherence(pid, patch_waveforms_by_id)
    coherence_pop = np.array([v for v in (patch_records[pid]["neighbor_coherence"] for pid in patch_records) if v is not None])
    coherence_z_scores = {
        pid: (
            (patch_records[pid]["neighbor_coherence"] - coherence_pop.mean()) / coherence_pop.std()
            if patch_records[pid]["neighbor_coherence"] is not None and coherence_pop.std() > 0
            else None
        )
        for pid in patch_records
    }
    persistence_pop = np.array(list(persistence_scores.values()))
    persistence_z_scores = {
        pid: (v - persistence_pop.mean()) / persistence_pop.std() if persistence_pop.std() > 0 else None
        for pid, v in persistence_scores.items()
    }

    # EXPLORATORY ONLY -- see module docstring: binary active/inactive
    # classification is deliberately NOT finalized this round. This
    # amplitude-AND-periodicity(primary_periodicity_method) combination is
    # kept only for continuity with earlier rounds' visualizations and to
    # give *a* grouping for the region/QC overlay; it is not validated
    # against any ground truth yet and must not be treated as the answer.
    # The authoritative output is the raw + normalized per-patch scores
    # below (all_patch_*_values, patch_records), meant to be compared
    # against manual reference annotations via reference_comparison.py
    # once available.
    active_ids = active_by_magnitude & active_by_periodicity

    cell_area_px = cell_size_px * cell_size_px
    fraction = active_region_fraction(cells, active_ids, cell_area_px)

    # --- Group active patches into spatially contiguous regions ---
    regions = cluster_active_patches_into_regions(active_ids)
    region_waveforms = {r.region_id: region_waveform(r, patch_waveforms_by_id) for r in regions}
    total_analyzed_area = sum(c.coverage_fraction * cell_area_px for c in cells)
    region_records = []
    for r in regions:
        wf = region_waveforms[r.region_id]
        region_records.append({
            "region_id": r.region_id,
            "patch_ids": [list(p) for p in r.patch_ids],
            "n_patches": len(r.patch_ids),
            "area_fraction_of_analyzed_region": len(r.patch_ids) * cell_area_px / total_analyzed_area,
            "waveform_std": float(np.nanstd(wf)),
            "autocorr_secondary_peak": first_secondary_autocorrelation_peak(wf, sample_interval_s),
            "spectral_peak": dominant_spectral_peak(wf, sample_interval_s),
        })
    min_size = pipe_cfg["min_region_patches"]
    region_records_min_size_filtered = [r for r in region_records if r["n_patches"] >= min_size]

    # Interface for region-to-region synchronization (not interpreted here
    # -- see cross_patch_synchronization's docstring for why a low value is
    # the expected baseline at beating onset).
    sync = cross_patch_synchronization({r.region_id: region_waveforms[r.region_id] for r in regions})

    # --- QC visualizations ---
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, rep_index)
    ok, rep_frame_bgr = cap.read()
    cap.release()
    frame_bgr_roi = rep_frame_bgr[y0:y1, x0:x1]

    spectral_concentration_scores = {pid: (rec["spectral_concentration"] or 0.0) for pid, rec in patch_records.items()}
    periodicity_p_values = {
        pid: rec["periodicity_by_null_method"][primary_periodicity_method]["p_value"]
        for pid, rec in patch_records.items()
    }
    periodicity_scores = {pid: 1.0 - p for pid, p in periodicity_p_values.items()}
    coherence_values = {pid: rec["neighbor_coherence"] for pid, rec in patch_records.items() if rec["neighbor_coherence"] is not None}
    coherence_scores = {pid: (rec["neighbor_coherence"] or 0.0) for pid, rec in patch_records.items()}

    # --- Raw-score heatmaps: one per evidence dimension, no combining. ---
    for name, scores in (
        ("std", radial_stds),
        ("persistence", persistence_scores),
        ("periodicity", periodicity_scores),
        ("neighbor_coherence", coherence_scores),
        ("spectral_concentration", spectral_concentration_scores),
    ):
        heatmap = build_activity_heatmap(cells, scores, frame_bgr_roi.shape[:2])
        cv2.imwrite(str(output_dir / f"activity_heatmap_{name}.png"), overlay_heatmap(frame_bgr_roi, heatmap))

    # --- Grid QC labeled with (row, col) ids, for building manual
    # reference annotations (see reference_comparison.py). ---
    grid_vis = frame_bgr_roi.copy()
    for cell in cells:
        cx0, cy0, cx1, cy1 = cell.bbox_xyxy
        cv2.rectangle(grid_vis, (cx0, cy0), (cx1, cy1), (200, 200, 200), 1)
        cv2.putText(
            grid_vis, f"{cell.row},{cell.col}", (cx0 + 2, cy0 + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 0), 1, cv2.LINE_AA,
        )
    cv2.imwrite(str(output_dir / "patch_grid_labeled_qc.png"), grid_vis)

    cv2.imwrite(str(output_dir / "active_regions_qc.png"), _region_qc_overlay(frame_bgr_roi, cells, regions))

    cv2.imwrite(
        str(output_dir / "null_reference_cells_qc.png"),
        draw_patch_qc(frame_bgr_roi, null_candidates),
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(list(radial_stds.values()), bins=20, alpha=0.7, label="organoid grid patches", density=True)
    axes[0].hist(null_stds, bins=10, alpha=0.7, label="null (background, windowed)", density=True)
    axes[0].axvline(magnitude_threshold, color="red", linestyle="--", label=f"threshold={magnitude_threshold:.4f}")
    axes[0].set_title("radial_std distribution")
    axes[0].set_xlabel("px")
    axes[0].legend(fontsize=7)
    for method in null_methods:
        pvals = [rec["periodicity_by_null_method"][method]["p_value"] for rec in patch_records.values()]
        axes[1].hist(pvals, bins=20, alpha=0.5, label=method, density=True)
    axes[1].axvline(periodicity_alpha, color="red", linestyle="--", label=f"alpha={periodicity_alpha}")
    axes[1].set_title("periodicity p-values by null method")
    axes[1].legend(fontsize=7)
    axes[2].hist(list(persistence_scores.values()), bins=20, alpha=0.7, label="persistence_fraction")
    axes[2].hist(list(coherence_values.values()), bins=20, alpha=0.5, label="neighbor_coherence")
    axes[2].set_title("persistence & coherence (descriptive, not gated)")
    axes[2].legend(fontsize=7)
    fig.suptitle(f"{video_path.stem}: activity evidence distributions")
    fig.tight_layout()
    fig.savefig(output_dir / "activity_score_distributions.png", dpi=120)
    plt.close(fig)

    def _keyed(d: dict) -> dict:
        return {f"{pid[0]}_{pid[1]}": v for pid, v in d.items()}

    result = {
        "video": str(video_path),
        "fps": fps,
        "frame_count": frame_count,
        "video_duration_s": frame_count / fps,
        "frame_step": step,
        "n_pairs_processed": n_pairs,
        "representative_frame_index": rep_index,
        "segmentation_source": segmentation_source,
        "mask_provenance": mask_provenance,
        "equivalent_diameter_px": equivalent_diameter_px,
        "n_cells_analyzed": len(cells),
        # --- Authoritative output: raw + normalized per-patch scores for
        # each evidence dimension, no hard threshold, no combining. Meant
        # to be compared against manual reference annotations via
        # reference_comparison.evaluate_scores_against_reference once
        # available -- goal is finding which score (or combination) best
        # reproduces visually-observed beating location, not "what % is
        # active". ---
        "scores_raw": {
            "amplitude_radial_std": _keyed(radial_stds),
            "persistence_fraction": _keyed(persistence_scores),
            "periodicity_p_value_primary": _keyed(periodicity_p_values),
            "spectral_concentration": _keyed(spectral_concentration_scores),
            "neighbor_coherence": _keyed(coherence_scores),
        },
        "scores_normalized_z": {
            "amplitude_z_vs_background_null": _keyed(
                {pid: v for pid, v in amplitude_z_scores.items() if v is not None}
            ),
            "periodicity_z_vs_own_shuffle_null_primary": _keyed(
                {pid: v for pid, v in periodicity_z_scores.items() if v is not None}
            ),
            "persistence_z_vs_video_population": _keyed(
                {pid: v for pid, v in persistence_z_scores.items() if v is not None}
            ),
            "neighbor_coherence_z_vs_video_population": _keyed(
                {pid: v for pid, v in coherence_z_scores.items() if v is not None}
            ),
        },
        "periodicity_null_method_comparison": {
            method: {
                "n_significant_at_alpha": len(active_by_periodicity_per_method[method]),
                "alpha": periodicity_alpha,
            }
            for method in null_methods
        },
        "periodicity_primary_null_method": primary_periodicity_method,
        # --- EXPLORATORY grouping only (see module docstring): a binary
        # active/inactive split and region clustering, kept for continuity
        # and a visual QC overlay, NOT validated against ground truth and
        # NOT the authoritative result. ---
        "exploratory_active_classification": {
            "status": "EXPLORATORY -- not validated, not finalized",
            "method": "amplitude_significance(z) AND periodicity_shuffle_test(primary_null_method)",
            "amplitude": {
                "z_threshold": pipe_cfg["active_z_threshold"],
                "magnitude_threshold_px": magnitude_threshold,
                "null_reference_stats_radial_std_windowed": null_mean_std,
                "null_source": "background regions (measurably far from organoid), windowed full-video stds; "
                                "NOT any organoid-derived data",
                "n_active_by_magnitude": len(active_by_magnitude),
            },
            "periodicity": {
                "null_method": primary_periodicity_method,
                "p_value_threshold": periodicity_alpha,
                "n_shuffles": pipe_cfg["periodicity_n_shuffles"],
                "n_active_by_periodicity": len(active_by_periodicity),
            },
            "n_active_by_both": len(active_ids),
            "n_active_cells": len(active_ids),
            "active_region_fraction": fraction,
            "n_active_regions": len(regions),
            "regions_all": region_records,
            "regions_min_size_filtered": {
                "min_region_patches": min_size,
                "regions": region_records_min_size_filtered,
            },
            "n_single_patch_regions": sum(1 for r in region_records if r["n_patches"] == 1),
            "region_synchronization_interface": sync,
        },
        "patch_records": {f"{pid[0]}_{pid[1]}": rec for pid, rec in patch_records.items()},
        "null_reference_records": {str(i): rec for i, rec in null_records.items()},
        "background_reference_qc": background_reference_qc,
        "disclaimer": (
            "PILOT main motion pipeline output (Farneback, full-video, "
            "per-patch). No whole-organoid averaged waveform used as "
            "primary signal. Binary active/inactive classification is "
            "NOT finalized -- scores_raw/scores_normalized_z are the "
            "authoritative output, meant for comparison against manual "
            "reference annotations (see reference_comparison.py) to find "
            "which score/combination best reproduces visually-observed "
            "beating location. exploratory_active_classification is "
            "kept for continuity only, not validated. No BPM, no beat/"
            "peak detection, no biological conclusion."
        ),
    }
    with open(output_dir / "motion_pipeline_qc.json", "w") as f:
        json.dump(result, f, indent=2)

    # Full per-patch radial-motion waveforms (every valid organoid-grid
    # cell, not just the exploratory active subset) -- not needed by the
    # scores above (all reduced to scalars already), but required by any
    # whole-organoid analysis that must use every valid patch's actual time
    # series (e.g. population-wide synchronization candidates: see
    # cardio_organoid.functional.phenotype_axes). Sampling interval is
    # frame_step / fps, both already recorded in motion_pipeline_qc.json.
    np.savez(
        output_dir / "patch_waveforms_radial.npz",
        **{f"{pid[0]}_{pid[1]}": wf for pid, wf in patch_waveforms_by_id.items()},
    )
    # Background/null-cell waveforms (measurably far from the organoid,
    # never organoid-derived) -- needed to diagnose whether elevated
    # inter-patch correlation reflects real coordinated organoid motion or
    # a shared imaging/drift/common-mode artifact also present off-organoid
    # (see cardio_organoid.functional.synchronization_diagnostics).
    np.savez(
        output_dir / "null_waveforms_radial.npz",
        **{f"null_{i}": wf for i, wf in null_waveforms_by_id.items()},
    )
    return result
