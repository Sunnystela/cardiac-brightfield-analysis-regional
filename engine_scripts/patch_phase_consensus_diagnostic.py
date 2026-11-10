"""Patch-level phase-aware regional beat consensus diagnostic (round 35,
PART A).

Existing pipeline (pipeline/analyze.py -> beat_event_detection_validation.py
-> beat_episode_reconstruction.py, all UNCHANGED, still canonical):

    patch waveforms -> region-AVERAGE waveform -> peak detection -> regional BPM

Round 33's D10/slit/a1 secondary_region diagnostic
(scripts/d10_secondary_cause_diagnostic.py) found the region's 5
individual patches each running at ~31-38 BPM (matching primary/
tertiary), not the region-average waveform's ~66.8 BPM -- "multiple
patch phase mixture" was the strongest surviving hypothesis, but was
only checked for that one hand-picked region. This script builds the
general machinery to test it everywhere, the other way around:

    patch waveform -> per-patch events -> per-patch dominant period/BPM
    -> patch-timing graph (cardio_organoid.functional.patch_timing_graph)
    -> phase-consistent patch clusters (connected components, self-
       relative similarity cutoff -- cardio_organoid.functional.
       patch_beat_consensus)
    -> phase-aware consensus beat episodes PER CLUSTER (patches, not
       manual regions, as beat_episodes.py's "region" unit)
    -> cluster-level consensus BPM (SAME beat_cycles.py pairing +
       beat_cycle_metrics.py convention used everywhere else, applied to
       consensus-beat representative times)

Manual reference region coordinates (results/manual_reference/*.json)
are used ONLY to label, after the fact, which auto-discovered cluster
happens to spatially overlap a named region (e.g. D10's
secondary_region) -- never as an input to detection, graph-building, or
clustering.

NOT done here (per project instruction):
  - beat_events.py/beat_cycles.py/beat_episodes.py/pipeline/analyze.py
    are imported READ-ONLY/reused unchanged, never edited.
  - No canonical results/ file is written to (results/beat_event_episodes*,
    results/manual_reference, patch_waveforms_radial.npz are read-only
    inputs).
  - No detection threshold, tolerance fraction, or clustering percentile
    is tuned to reproduce D10 secondary_region's manual ~19/~31s count or
    ~37-40 BPM -- every number reported is self-relative to each video's/
    cluster's own data, and the manual/boundary/binary-movement numbers
    are read in ONLY for side-by-side comparison in the printed/plotted
    output, never fed back into the algorithm.
  - Source video opened read-only.

Compute-budget note (documented, not hidden): a fresh, frame-step=1
Farneback pass (needed for signed radial/tangential/magnitude flow, since
the existing patch_waveforms_radial.npz only stores the radial scalar
already used for the canonical patch/region BPM) is run for
mature/D10_slit_a1 ONLY (this round's primary focus). The other 4 videos
reuse the existing canonical per-patch radial waveform directly -- no new
Farneback pass for them.

Writes only to a fresh results/_scratch_runs/ directory by default (see
scripts/_output_safety.py).

Usage:
    .venv/bin/python3 scripts/patch_phase_consensus_diagnostic.py
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

from cardio_organoid.functional.beat_cycles import beat_cycle_counts, build_beat_cycles  # noqa: E402
from cardio_organoid.functional.beat_episodes import (  # noqa: E402
    RegionEvent,
    pooled_dominant_period_s,
    region_events_from_candidate_events,
    suggest_episode_tolerance_s,
)
from cardio_organoid.functional.beat_events import (  # noqa: E402
    detect_beat_events_both_polarities,
    preprocess_waveform,
    region_beat_summary,
)
from cardio_organoid.functional.patch_beat_consensus import (  # noqa: E402
    build_consensus_beats_for_cluster,
    cluster_patches_by_timing_similarity,
    cluster_regional_summary,
    consensus_beat_cycle_metrics,
)
from cardio_organoid.functional.patch_timing_graph import build_patch_timing_graph  # noqa: E402
from cardio_organoid.functional.regional_lag_diagnostics import region_mean_waveform  # noqa: E402
from cardio_organoid.motion.spatial_patches import neighbor_coherence  # noqa: E402

from _output_safety import add_output_dir_args, resolve_output_dir  # noqa: E402
from beat_event_detection_validation import (  # noqa: E402
    MIN_PROMINENCE_RATIO,
    MIN_SEPARATION_PERIOD_FRACTION,
    SMOOTH_WINDOW_SAMPLES,
    _baseline_window_samples,
)
from global_boundary_motion_diagnostic import VIDEOS, _fmt, _load_config, _recover_mask_and_roi  # noqa: E402
from reference_binary_motion_diagnostic import CANONICAL_EPISODES_PATHS, REFERENCE_PATHS  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "results" / "patch_phase_consensus_diagnostic"

SIMILARITY_PERCENTILE = 75.0  # self-relative, top-quartile among this video's OWN computed pairwise coincidence values
MAX_LAG_S = 1.0  # same default as regional_lag_diagnostics.region_pair_lag_diagnostic

# Manual observation stated by the project owner (rounds 28-34): NOT a
# stored value in results/manual_reference/*.json, NOT a tuning target --
# reference-only, reported alongside computed values for context.
MANUAL_OBSERVED_BEAT_COUNT = 19
MANUAL_OBSERVED_DURATION_RANGE_S = (30.0, 31.0)
MANUAL_OBSERVED_BPM_REFERENCE_ONLY = 60.0 * MANUAL_OBSERVED_BEAT_COUNT / np.mean(MANUAL_OBSERVED_DURATION_RANGE_S)


def _regions_from_reference(ref: dict) -> dict[str, list[tuple[int, int]]]:
    if ref.get("regions") and ref.get("activation_order"):
        return {name: [tuple(p) for p in ref["regions"][name]] for name in ref["activation_order"]}
    dominant = ref.get("dominant_beating_region") or []
    return {"dominant_beating_region": [tuple(p) for p in dominant]}


def per_patch_events_and_features(patch_id, waveform, sample_interval_s, frame_step, fps, config):
    """Per-patch A1 features: own events, own dominant period/BPM,
    amplitude, local IBI distribution, event confidence (own beat_cycle
    pairing tier counts) -- ALL from the SAME unchanged detection/pairing
    functions/constants used everywhere else in this project."""
    region_id = f"{patch_id[0]}_{patch_id[1]}"
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
    dominant_freq_hz = (1.0 / pooled_period) if pooled_period else None

    cfg_bc = config["functional"]["beat_cycles"]
    pairing_window_s = (pooled_period * cfg_bc["pairing_window_period_fraction"]) if pooled_period else cfg_bc["fallback_pairing_window_s"]
    cycles = build_beat_cycles(pos_events, neg_events, pairing_window_s)
    confidence_counts = beat_cycle_counts(cycles)

    pooled_times = sorted(e.event_time for e in pos_events + neg_events)
    ibis = np.diff(pooled_times).tolist() if len(pooled_times) > 1 else []

    return {
        "patch_id": list(patch_id), "region_id": region_id,
        "detection_input": detection_input,
        "positive_events": pos_events, "negative_events": neg_events,
        "pooled_event_times_s": pooled_times,
        "amplitude_std": float(np.std(detection_input)),
        "dominant_frequency_hz": dominant_freq_hz,
        "provisional_bpm_own": (60.0 * dominant_freq_hz) if dominant_freq_hz else None,
        "local_ibi_distribution_s": ibis,
        "median_ibi_s": float(np.median(ibis)) if ibis else None,
        "cv_ibi": (float(np.std(ibis, ddof=1) / np.mean(ibis)) if len(ibis) > 1 and np.mean(ibis) > 0 else None),
        "event_confidence_own_beat_cycles": confidence_counts,
    }


def process_video(video_cfg: dict, config: dict, output_dir: Path, texture_frame_gray=None) -> dict:
    group, name = video_cfg["group"], video_cfg["name"]
    print(f"\n=== {group}/{name} ===")

    qc = json.loads((video_cfg["motion_dir"] / "motion_pipeline_qc.json").read_text())
    npz = np.load(video_cfg["motion_dir"] / "patch_waveforms_radial.npz")
    waveforms = {tuple(int(p) for p in k.split("_")): npz[k] for k in npz.files}
    fps, frame_step = qc["fps"], qc["frame_step"]
    sample_interval_s = frame_step / fps
    patch_ids = sorted(waveforms)
    print(f"  n_patches_analyzed={len(patch_ids)}")

    per_patch = {pid: per_patch_events_and_features(pid, waveforms[pid], sample_interval_s, frame_step, fps, config) for pid in patch_ids}

    # -- neighbor coherence over the FULL analyzed grid (round-33 convention) --
    for pid in patch_ids:
        per_patch[pid]["neighbor_coherence_full_grid"] = neighbor_coherence(pid, waveforms)

    # -- video-level pooled period / graph tolerance (self-relative, same
    # config value already used by beat_episodes.py elsewhere) --
    cfg_be = config["functional"]["beat_episodes"]
    video_pooled_period = pooled_dominant_period_s([per_patch[pid]["dominant_frequency_hz"] for pid in patch_ids])
    coincidence_tolerance_s = suggest_episode_tolerance_s(video_pooled_period, cfg_be["episode_tolerance_period_fraction"]) or cfg_be["fallback_episode_tolerance_s"]
    print(f"  video pooled_period_s={_fmt(video_pooled_period)}  graph coincidence_tolerance_s={_fmt(coincidence_tolerance_s)}")

    graph = build_patch_timing_graph(
        patch_ids, waveforms,
        {pid: per_patch[pid]["pooled_event_times_s"] for pid in patch_ids},
        {pid: per_patch[pid]["dominant_frequency_hz"] for pid in patch_ids},
        sample_interval_s, coincidence_tolerance_s, MAX_LAG_S,
    )
    clusters = cluster_patches_by_timing_similarity(graph, similarity_percentile=SIMILARITY_PERCENTILE)
    print(f"  n_clusters={len(clusters)}  cluster_sizes={[len(c) for c in clusters]}")

    # -- clustering-percentile sensitivity sweep (transparency diagnostic
    # only -- NOT used to pick a "better" percentile post-hoc; the
    # single-linkage-style connected-components method can chain many
    # patches into one mega-cluster at a low cutoff, so this sweep is
    # reported honestly alongside the SIMILARITY_PERCENTILE default result
    # rather than silently tuning the cutoff to whichever value happens to
    # look most sensible -- see README.txt in the output directory) --
    percentile_sensitivity = []
    for pct in (75.0, 85.0, 90.0, 95.0, 99.0):
        cl = cluster_patches_by_timing_similarity(graph, similarity_percentile=pct)
        sizes = sorted((len(c) for c in cl), reverse=True)
        percentile_sensitivity.append({"percentile": pct, "n_clusters": len(cl), "cluster_sizes_desc": sizes})
    print(f"  clustering-percentile sensitivity: {[(s['percentile'], s['n_clusters'], s['cluster_sizes_desc'][:5]) for s in percentile_sensitivity]}")

    # -- manual-region overlap labels (LABELING ONLY -- never used above) --
    ref_path = REFERENCE_PATHS.get((group, name))
    manual_regions = _regions_from_reference(json.loads(ref_path.read_text())) if ref_path and ref_path.exists() else {}
    cluster_labels = []
    for cluster in clusters:
        cluster_set = set(cluster)
        best_region, best_overlap = None, 0.0
        for rname, rpatches in manual_regions.items():
            rset = set(rpatches)
            overlap = len(cluster_set & rset) / len(rset) if rset else 0.0
            if overlap > best_overlap:
                best_region, best_overlap = rname, overlap
        cluster_labels.append({"overlapping_manual_region": best_region, "overlap_fraction_of_manual_region": best_overlap})

    pos_events_by_patch = {pid: per_patch[pid]["positive_events"] for pid in patch_ids}
    neg_events_by_patch = {pid: per_patch[pid]["negative_events"] for pid in patch_ids}
    dominant_freq_by_patch = {pid: per_patch[pid]["dominant_frequency_hz"] for pid in patch_ids}
    amplitude_by_patch = {pid: per_patch[pid]["amplitude_std"] for pid in patch_ids}

    cluster_results = []
    for cluster, label in zip(clusters, cluster_labels):
        consensus = build_consensus_beats_for_cluster(
            cluster, pos_events_by_patch, neg_events_by_patch, dominant_freq_by_patch, graph,
            tolerance_period_fraction=cfg_be["episode_tolerance_period_fraction"],
            fallback_tolerance_s=cfg_be["fallback_episode_tolerance_s"],
            global_min_participation_fraction=cfg_be["global_min_participation_fraction"],
        )
        cluster_id = f"cluster_{len(cluster_results):02d}"
        cycle_metrics = consensus_beat_cycle_metrics(
            cluster_id, consensus,
            pairing_window_period_fraction=config["functional"]["beat_cycles"]["pairing_window_period_fraction"],
            fallback_pairing_window_s=config["functional"]["beat_cycles"]["fallback_pairing_window_s"],
        )
        summary = cluster_regional_summary(cluster, consensus, cycle_metrics, amplitude_by_patch, waveforms)
        cluster_results.append({
            "cluster_id": cluster_id, **label, "consensus": consensus,
            "cycle_metrics_high_confidence": cycle_metrics["metrics"]["tiers"]["high_confidence_only"],
            "summary": summary,
        })
        print(f"    [{cluster_id}] n_patches={len(cluster)} overlaps={label['overlapping_manual_region']}"
              f"({_fmt(label['overlap_fraction_of_manual_region'])}) "
              f"consensus_BPM={_fmt(summary['consensus_bpm_high_confidence'])} "
              f"n_beats={summary['consensus_beat_count_high_confidence']} "
              f"CV={_fmt(summary['cv_ibi_high_confidence'])}")

    # -- canonical Farneback region-average BPM, for side-by-side comparison --
    episodes_path = CANONICAL_EPISODES_PATHS.get((group, name))
    farneback_region_bpm = {}
    if episodes_path and episodes_path.exists():
        canonical_episodes = json.loads(episodes_path.read_text())
        for rname, rc in canonical_episodes.get("region_cycle_metrics", {}).items():
            farneback_region_bpm[rname] = rc["tiers"]["high_confidence_only"]["provisional_bpm_from_median_ibi"]
    print(f"  canonical Farneback region-average BPM: {farneback_region_bpm}")

    return {
        "video": f"{group}/{name}", "fps": fps, "frame_step": frame_step, "n_patches": len(patch_ids),
        "video_pooled_period_s": video_pooled_period, "graph_coincidence_tolerance_s": coincidence_tolerance_s,
        "n_clusters": len(clusters), "cluster_results": cluster_results,
        "similarity_percentile_used": SIMILARITY_PERCENTILE, "clustering_percentile_sensitivity": percentile_sensitivity,
        "farneback_region_bpm_high_confidence": farneback_region_bpm,
        "manual_regions": {k: [list(p) for p in v] for k, v in manual_regions.items()},
        "per_patch_features": {
            f"{pid[0]}_{pid[1]}": {k: v for k, v in feat.items() if k not in ("detection_input", "positive_events", "negative_events")}
            for pid, feat in per_patch.items()
        },
    }, per_patch, waveforms, graph, clusters, cluster_labels


# --------------------------------------------------------------------------
# D10-specific QC figures (A5)
# --------------------------------------------------------------------------


def _find_secondary_overlap_cluster(cluster_results):
    candidates = [c for c in cluster_results if c["overlapping_manual_region"] == "secondary_region"]
    if not candidates:
        return None
    return max(candidates, key=lambda c: c["overlap_fraction_of_manual_region"])


def render_d10_qc_figures(result, per_patch, waveforms, output_dir, manual_ref, boundary_bpm, binary_bpm, texture_frame_gray=None):
    group, name = "mature", "D10_slit_a1"
    # result["manual_regions"] was JSON-round-tripped (list-of-lists) for
    # summary.json -- convert back to hashable (row, col) tuples for use
    # as dict keys / waveform lookups below.
    manual_regions = {rname: [tuple(p) for p in patches] for rname, patches in result["manual_regions"].items()}
    cluster_results = result["cluster_results"]

    fig_map = {}
    for rname in ("secondary_region", "primary_region", "tertiary_region"):
        candidates = [c for c in cluster_results if c["overlapping_manual_region"] == rname]
        if candidates:
            fig_map[rname] = max(candidates, key=lambda c: c["overlap_fraction_of_manual_region"])

    # -- D10_PATCH_EVENT_RASTER.png --
    fig, axes = plt.subplots(len(fig_map), 1, figsize=(12, 3 * len(fig_map)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (rname, cluster) in zip(axes, fig_map.items()):
        patch_ids = [tuple(p) for p in cluster["consensus"]["cluster_patch_ids"]]
        for row_i, pid in enumerate(patch_ids):
            pos_t = [e.event_time for e in per_patch[pid]["positive_events"]]
            neg_t = [e.event_time for e in per_patch[pid]["negative_events"]]
            ax.scatter(pos_t, [row_i] * len(pos_t), marker="|", color="crimson", s=200)
            ax.scatter(neg_t, [row_i] * len(neg_t), marker="|", color="darkblue", s=200)
        for beat in cluster["consensus"]["positive_consensus_beats"]:
            ax.axvline(beat["event_time"], color="seagreen", alpha=0.25, linewidth=1.5)
        ax.set_yticks(range(len(patch_ids)))
        ax.set_yticklabels([str(p) for p in patch_ids], fontsize=6)
        bpm = cluster["summary"]["consensus_bpm_high_confidence"]
        ax.set_title(f"{rname}-overlapping cluster ({cluster['cluster_id']}, n={len(patch_ids)} patches): "
                     f"consensus BPM={_fmt(bpm)} vs Farneback region-avg={_fmt(result['farneback_region_bpm_high_confidence'].get(rname))}",
                     fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("D10/slit/a1 -- per-patch beat-time raster by auto-discovered cluster (round 35)\n"
                 "red=positive event, blue=negative event, green line=phase-aware consensus beat", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_dir / "D10_PATCH_EVENT_RASTER.png", dpi=130)
    plt.close(fig)
    print(f"  wrote {output_dir / 'D10_PATCH_EVENT_RASTER.png'}")

    # -- D10_PATCH_PHASE_MAP.png (secondary_region-overlapping cluster) --
    sec = fig_map.get("secondary_region")
    if sec is not None:
        patch_ids = [tuple(p) for p in sec["consensus"]["cluster_patch_ids"]]
        rows = [p[0] for p in patch_ids]
        cols = [p[1] for p in patch_ids]
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        bpm_grid = np.full((r1 - r0 + 1, c1 - c0 + 1), np.nan)
        for pid in patch_ids:
            bpm_grid[pid[0] - r0, pid[1] - c0] = per_patch[pid]["provisional_bpm_own"] or np.nan

        all_consensus_beats = sorted(sec["consensus"]["positive_consensus_beats"], key=lambda b: b["event_time"])
        phase_offset_grid = np.full((r1 - r0 + 1, c1 - c0 + 1), np.nan)
        if all_consensus_beats and sec["summary"]["pooled_dominant_period_s"]:
            period = sec["summary"]["pooled_dominant_period_s"]
            for pid in patch_ids:
                own_times = [e.event_time for e in per_patch[pid]["positive_events"]]
                if not own_times:
                    continue
                offsets = []
                for beat in all_consensus_beats:
                    nearest = min(own_times, key=lambda t: abs(t - beat["event_time"]))
                    offsets.append(((nearest - beat["event_time"]) / period) % 1.0)
                phase_offset_grid[pid[0] - r0, pid[1] - c0] = float(np.median(offsets))

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        im0 = axes[0].imshow(bpm_grid, cmap="coolwarm", vmin=20, vmax=80)
        for pid in patch_ids:
            axes[0].text(pid[1] - c0, pid[0] - r0, _fmt(per_patch[pid]["provisional_bpm_own"]), ha="center", va="center", fontsize=7)
        plt.colorbar(im0, ax=axes[0], fraction=0.046, label="own provisional BPM")
        axes[0].set_title("secondary_region-overlapping cluster:\nper-patch own BPM", fontsize=9)

        im1 = axes[1].imshow(phase_offset_grid, cmap="twilight", vmin=0, vmax=1)
        for pid in patch_ids:
            v = phase_offset_grid[pid[0] - r0, pid[1] - c0]
            axes[1].text(pid[1] - c0, pid[0] - r0, _fmt(v), ha="center", va="center", fontsize=7)
        plt.colorbar(im1, ax=axes[1], fraction=0.046, label="median phase offset (fraction of period)")
        axes[1].set_title("per-patch phase offset\nrelative to cluster consensus beats", fontsize=9)

        region_avg_wf = region_mean_waveform(waveforms, manual_regions.get("secondary_region", []))
        t = np.arange(len(region_avg_wf)) * (result["frame_step"] / result["fps"])
        axes[2].plot(t, region_avg_wf, color="black", linewidth=1.0, label="region-AVERAGE waveform (existing Farneback)")
        for beat in all_consensus_beats:
            axes[2].axvline(beat["event_time"], color="seagreen", alpha=0.4, linewidth=1.0)
        axes[2].set_title(
            f"region-average waveform (Farneback BPM={_fmt(result['farneback_region_bpm_high_confidence'].get('secondary_region'))})\n"
            f"vs patch-consensus beats (green, consensus BPM={_fmt(sec['summary']['consensus_bpm_high_confidence'])})",
            fontsize=8,
        )
        axes[2].set_xlabel("time (s)")
        axes[2].legend(fontsize=7)

        fig.suptitle(
            "D10/slit/a1 secondary_region -- phase-mixture test: do ~30-40 BPM patches at mixed phase "
            "produce a ~66.8 BPM region-average? (round 35)", fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        fig.savefig(output_dir / "D10_PATCH_PHASE_MAP.png", dpi=130)
        plt.close(fig)
        print(f"  wrote {output_dir / 'D10_PATCH_PHASE_MAP.png'}")

    # -- D10_PATCH_PHASE_CONSENSUS_QC: combined static figure (compute-
    # budget note: a full time-synced video like rounds 33/34's would
    # require re-reading the whole video for a 5th time this round just
    # for rendering; this round uses a single combined static figure
    # instead, covering all 5 requested panels on one shared time axis) --
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.2, 1.2, 1.2, 1.0])

    ax_grid = fig.add_subplot(gs[0])
    if texture_frame_gray is not None:
        ax_grid.imshow(texture_frame_gray, cmap="gray")
    ax_grid.set_title("representative frame (patch grid overlay omitted -- see D10_SECONDARY_CAUSE_QC.mp4 for ROI/grid detail)", fontsize=8)
    ax_grid.set_xticks([]); ax_grid.set_yticks([])

    ax_raster = fig.add_subplot(gs[1])
    if sec is not None:
        patch_ids = [tuple(p) for p in sec["consensus"]["cluster_patch_ids"]]
        for row_i, pid in enumerate(patch_ids):
            pos_t = [e.event_time for e in per_patch[pid]["positive_events"]]
            ax_raster.scatter(pos_t, [row_i] * len(pos_t), marker="|", color="crimson", s=150)
        ax_raster.set_yticks(range(len(patch_ids)))
        ax_raster.set_yticklabels([str(p) for p in patch_ids], fontsize=6)
    ax_raster.set_title("secondary_region-overlapping cluster: per-patch positive-event raster", fontsize=8)

    ax_stack = fig.add_subplot(gs[2])
    if sec is not None:
        patch_ids = [tuple(p) for p in sec["consensus"]["cluster_patch_ids"]]
        spacing = 4.0 * np.std(np.concatenate([per_patch[pid]["detection_input"] if "detection_input" in per_patch[pid] else waveforms[pid] for pid in patch_ids])) or 1.0
        for i, pid in enumerate(patch_ids):
            wf = waveforms[pid]
            t = np.arange(len(wf)) * (result["frame_step"] / result["fps"])
            ax_stack.plot(t, wf + i * spacing, linewidth=0.9, label=str(pid))
        ax_stack.legend(fontsize=6, ncol=len(patch_ids))
    ax_stack.set_title("secondary_region-overlapping cluster: per-patch waveforms (offset stacked)", fontsize=8)

    ax_bottom = fig.add_subplot(gs[3])
    if sec is not None:
        region_avg_wf = region_mean_waveform(waveforms, manual_regions.get("secondary_region", []))
        t = np.arange(len(region_avg_wf)) * (result["frame_step"] / result["fps"])
        ax_bottom.plot(t, region_avg_wf, color="black", linewidth=1.0, label="region-average waveform")
        for beat in sec["consensus"]["positive_consensus_beats"]:
            ax_bottom.axvline(beat["event_time"], color="seagreen", alpha=0.5, linewidth=1.2, label="_")
        ax_bottom.axvline(sec["consensus"]["positive_consensus_beats"][0]["event_time"] if sec["consensus"]["positive_consensus_beats"] else 0,
                           color="seagreen", alpha=0.5, linewidth=1.2, label="patch-consensus beat")
        ax_bottom.legend(fontsize=7)
    ax_bottom.set_xlabel("time (s)")
    ax_bottom.set_title(
        f"region-average waveform + consensus beat timeline  |  Farneback={_fmt(result['farneback_region_bpm_high_confidence'].get('secondary_region'))} BPM, "
        f"patch-consensus={_fmt(sec['summary']['consensus_bpm_high_confidence']) if sec else 'NA'} BPM, "
        f"manual(ref only)={_fmt(manual_ref)} BPM, boundary(ref only)={_fmt(boundary_bpm)} BPM, "
        f"binary(ref only)={_fmt(binary_bpm)} BPM",
        fontsize=8,
    )
    fig.suptitle("D10/slit/a1 secondary_region -- patch-phase consensus QC (round 35, static combined figure)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "D10_PATCH_PHASE_CONSENSUS_QC.png", dpi=130)
    plt.close(fig)
    print(f"  wrote {output_dir / 'D10_PATCH_PHASE_CONSENSUS_QC.png'} (static figure, not a video -- see module docstring's compute-budget note)")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_output_dir_args(parser, OUTPUT_DIR, "patch_phase_consensus_diagnostic")
    args = parser.parse_args()
    output_dir = resolve_output_dir(args, OUTPUT_DIR, "patch_phase_consensus_diagnostic")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_config()

    all_results = {}
    d10_context = None
    for video_cfg in VIDEOS:
        result, per_patch, waveforms, graph, clusters, cluster_labels = process_video(video_cfg, config, output_dir)
        all_results[result["video"]] = result
        if video_cfg["group"] == "mature" and video_cfg["name"] == "D10_slit_a1":
            d10_context = (result, per_patch, waveforms)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"\nWrote summary for {len(all_results)} video(s) to {output_dir}")

    if d10_context is not None:
        result, per_patch, waveforms = d10_context
        boundary_path = REPO_ROOT / "results" / "global_boundary_motion" / "mature" / "D10_slit_a1" / "global_boundary_motion.json"
        boundary_bpm = None
        if boundary_path.exists():
            gb = json.loads(boundary_path.read_text())
            boundary_bpm = gb["boundary_cycle_metrics"]["tiers"]["high_confidence_only"]["provisional_bpm_from_median_ibi"]
        binary_bpm = None
        binary_summary_path = REPO_ROOT / "results" / "_scratch_runs" / "reference_binary_motion_diagnostic_20260909T113349Z" / "summary.json"
        if binary_summary_path.exists():
            binary_summary = json.loads(binary_summary_path.read_text())
            try:
                binary_bpm = binary_summary["mature/D10_slit_a1"]["beat_like_summary"]["C_adaptive_ema__k=8__secondary_region"]["provisional_bpm"]
            except KeyError:
                binary_bpm = None

        sam2_cfg = config["segmentation"]["sam2"]
        from cardio_organoid.segmentation.sam2_segmenter import get_device, load_predictor
        device = get_device(sam2_cfg["device"])
        predictor = load_predictor(checkpoint_path=REPO_ROOT / sam2_cfg["checkpoint_path"], model_cfg=sam2_cfg["model_cfg"], device=device)
        d10_video_cfg = next(v for v in VIDEOS if v["group"] == "mature" and v["name"] == "D10_slit_a1")
        morphology_mask, roi, equiv_diam, frame_count = _recover_mask_and_roi(d10_video_cfg["source_video"], config, predictor)
        cap = cv2.VideoCapture(str(d10_video_cfg["source_video"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
        ok, rep_frame_bgr = cap.read()
        cap.release()
        x0, y0, x1, y1 = roi.roi_bbox_xyxy
        texture_frame_gray = cv2.cvtColor(rep_frame_bgr, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1] if ok else None

        render_d10_qc_figures(result, per_patch, waveforms, output_dir, MANUAL_OBSERVED_BPM_REFERENCE_ONLY, boundary_bpm, binary_bpm, texture_frame_gray)

        sec = _find_secondary_overlap_cluster(result["cluster_results"])
        print("\n=== D10/slit/a1 secondary_region-overlapping cluster: headline comparison ===")
        print(f"  Farneback region-average BPM: {_fmt(result['farneback_region_bpm_high_confidence'].get('secondary_region'))}")
        print(f"  Patch-level phase-aware consensus BPM: {_fmt(sec['summary']['consensus_bpm_high_confidence']) if sec else 'NA (no cluster overlapped secondary_region)'}")
        print(f"  Manual observed (reference only): {_fmt(MANUAL_OBSERVED_BPM_REFERENCE_ONLY)}")
        print(f"  Boundary motion (reference only): {_fmt(boundary_bpm)}")
        print(f"  Binary movement, adaptive+k=8 (reference only): {_fmt(binary_bpm)}")


if __name__ == "__main__":
    main()
