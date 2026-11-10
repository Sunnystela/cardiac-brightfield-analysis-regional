"""Region-level beat-event detection validation (round 18) across all 6
manual-reference videos: early conv/a1, slit/a1, slit/b1; mature
Beating_D9/slit/a1, Beating_D9/conv/a1, D10_26.08.2026/slit/a1.

Goal: check whether real cardiac beating EVENTS can be reliably detected
from the region-average motion waveforms the existing motion-analysis
architecture already produces (Farneback flow -> spatial activity heatmap
-> regional localization -> region-average waveform). This script adds a
new stage on top of that architecture; it does not change it.

For each video's manually-identified region(s) (named "regions" +
"activation_order" if present; else the single "dominant_beating_region";
else -- ONLY for slit/b1, which has no manually-confirmed dominant region
-- an algorithm-suggested top-8-by-amplitude fallback, explicitly labeled
as not manually confirmed):

  1. preprocessing: raw -> (optional) linear detrend -> light smoothing
     -> rolling-median baseline correction, ALL stages saved to one QC
     plot, raw waveform never discarded.
  2. candidate beat-event detection (prominence relative to the
     waveform's own robust noise floor, minimum separation relative to
     its own estimated dominant period -- not scipy find_peaks defaults,
     not an absolute magic threshold), for BOTH the positive and negative
     excursion direction (no polarity assumed to be "the" contraction --
     see point 3), events marked directly on the waveform plot.
  3. a polarity-alternation QC number (does NOT decide which direction is
     contraction -- a human still needs to check against the raw video).
  4. a per-region summary (n_events, mean amplitude, mean inter-event
     interval, dominant frequency, spectral concentration).
  5. for videos with >1 named region: regional activation delay between
     consecutive regions in the manual activation_order (never called
     conduction velocity/propagation speed).
  6. simultaneous global-beat candidates across all of a video's named
     regions (only constructed where events repeatedly co-occur within a
     narrow window -- never forced, never collapsed into one BPM).

No BPM, arrhythmia classification, or biological conclusion is finalized
here. No threshold is tuned to any manual reference -- the same
detection parameters are used for every video, and results are compared
against manual observation only qualitatively/descriptively.

Every timebase-dependent value carries fps (file-recorded, per video --
NEVER a nominal 30 fps), frame_step, sample_interval_s, frame_index, and
time_sec (see cardio_organoid.functional.beat_events module docstring for
the exact lag_seconds formula this guarantees is reconstructible).

Never modifies motion_pipeline_qc.json, waveform npz files, manual
reference files, or source videos (original or working copy).

Usage:
    .venv/bin/python3 scripts/beat_event_detection_validation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cardio_organoid.functional.beat_events import (  # noqa: E402
    detect_beat_events_both_polarities,
    global_beat_candidates,
    polarity_alternation_check,
    preprocess_waveform,
    region_beat_summary,
    regional_activation_delay,
)
from cardio_organoid.functional.phenotype_axes import top_patches_by_score  # noqa: E402
from cardio_organoid.functional.regional_lag_diagnostics import region_mean_waveform  # noqa: E402
from cardio_organoid.motion.diagnostics import dominant_spectral_peak  # noqa: E402

MOTION_PIPELINE_V2_DIR = REPO_ROOT / "results" / "motion_pipeline_v2_background_stabilized"
MOTION_PIPELINE_MATURE_V2_DIR = REPO_ROOT / "results" / "motion_pipeline_mature_v2_background_stabilized"
REFERENCE_DIR = REPO_ROOT / "results" / "manual_reference"
REFERENCE_MATURE_DIR = REPO_ROOT / "results" / "manual_reference" / "mature"
OUTPUT_DIR = REPO_ROOT / "results" / "beat_event_detection"

# (group, name, motion-pipeline output dir, manual reference json path)
VIDEOS = [
    ("early", "conv_a1", MOTION_PIPELINE_V2_DIR / "conv" / "a1", REFERENCE_DIR / "conv" / "a1.json"),
    ("early", "slit_a1", MOTION_PIPELINE_V2_DIR / "slit" / "a1", REFERENCE_DIR / "slit" / "a1.json"),
    ("early", "slit_b1", MOTION_PIPELINE_V2_DIR / "slit" / "b1", REFERENCE_DIR / "slit" / "b1.json"),
    ("mature", "Beating_D9_slit_a1", MOTION_PIPELINE_MATURE_V2_DIR / "Beating_D9" / "slit" / "a1",
     REFERENCE_MATURE_DIR / "Beating_D9" / "slit" / "a1.json"),
    ("mature", "Beating_D9_conv_a1", MOTION_PIPELINE_MATURE_V2_DIR / "Beating_D9" / "conv" / "a1",
     REFERENCE_MATURE_DIR / "Beating_D9" / "conv" / "a1.json"),
    ("mature", "D10_slit_a1", MOTION_PIPELINE_MATURE_V2_DIR / "D10_26.08.2026" / "slit" / "a1",
     REFERENCE_MATURE_DIR / "D10_26.08.2026" / "slit" / "a1.json"),
]

# Same detection parameters for every video -- never tuned per-video/per-
# manual-reference.
MIN_PROMINENCE_RATIO = 3.0
MIN_SEPARATION_PERIOD_FRACTION = 0.4
SMOOTH_WINDOW_SAMPLES = 3
GLOBAL_BEAT_TOLERANCE_S = 0.3


def _unkey(d: dict) -> dict[tuple[int, int], float]:
    return {tuple(int(p) for p in k.split("_")): v for k, v in d.items()}


def get_regions_to_analyze(ref: dict, amplitude: dict) -> dict[str, list[tuple[int, int]]]:
    if ref.get("regions") and ref.get("activation_order"):
        return {name: [tuple(p) for p in ref["regions"][name]] for name in ref["activation_order"]}
    dominant = ref.get("dominant_beating_region") or []
    if dominant:
        return {"dominant_beating_region": [tuple(p) for p in dominant]}
    top8 = top_patches_by_score(amplitude, top_n=8)
    return {"algorithm_suggested_top8_NOT_manually_confirmed": [pid for pid, _ in top8]}


def _baseline_window_samples(waveform: np.ndarray, sample_interval_s: float, default_samples: int = 31) -> int:
    freq = dominant_spectral_peak(waveform, sample_interval_s)["dominant_frequency_hz"]
    if not freq or freq <= 0:
        return default_samples
    period_samples = 1.0 / (freq * sample_interval_s)
    return max(default_samples, int(round(3 * period_samples)) | 1)  # odd window, >= ~3 periods


def _plot_preprocessing_qc(stages: dict, sample_interval_s: float, title: str, output_path: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    for ax, key in zip(axes, ["raw", "detrended", "smoothed", "baseline_corrected"]):
        wf = stages[key]
        t = np.arange(len(wf)) * sample_interval_s
        ax.plot(t, wf, linewidth=1.0, color="steelblue")
        ax.set_ylabel(key)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def _plot_events_overlay(waveform: np.ndarray, events_both: dict, sample_interval_s: float, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    t = np.arange(len(waveform)) * sample_interval_s
    ax.plot(t, waveform, linewidth=1.0, color="gray", label="waveform (detection input stage)")
    pos_t = [e["time_sec"] for e in events_both["positive_events"]]
    pos_y = [e["local_amplitude"] for e in events_both["positive_events"]]
    neg_t = [e["time_sec"] for e in events_both["negative_events"]]
    neg_y = [e["local_amplitude"] for e in events_both["negative_events"]]
    ax.scatter(pos_t, pos_y, color="crimson", marker="^", s=40, zorder=5, label=f"positive events (n={len(pos_t)})")
    ax.scatter(neg_t, neg_y, color="darkblue", marker="v", s=40, zorder=5, label=f"negative events (n={len(neg_t)})")
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("radial motion (px)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def process_video(group: str, name: str, video_dir: Path, ref_path: Path) -> dict:
    qc = json.loads((video_dir / "motion_pipeline_qc.json").read_text())
    amplitude = _unkey(qc["scores_raw"]["amplitude_radial_std"])
    npz = np.load(video_dir / "patch_waveforms_radial.npz")
    waveforms = {tuple(int(p) for p in key.split("_")): npz[key] for key in npz.files}
    ref = json.loads(ref_path.read_text())

    fps = qc["fps"]
    frame_step = qc["frame_step"]
    sample_interval_s = frame_step / fps

    regions = get_regions_to_analyze(ref, amplitude)
    out_dir = OUTPUT_DIR / group / name
    out_dir.mkdir(parents=True, exist_ok=True)

    region_results = {}
    region_events_for_global = {}
    print(f"\n=== {group}/{name} ===  fps={fps} (file-recorded) frame_step={frame_step} "
          f"sample_interval_s={sample_interval_s:.4f}  manual global_pattern={ref.get('global_pattern')}")

    for region_name, patch_ids in regions.items():
        wf = region_mean_waveform(waveforms, patch_ids)
        if wf is None:
            print(f"  region '{region_name}': no patches found in analyzed grid -- skipping")
            continue

        stages = preprocess_waveform(
            wf, apply_detrend=True, smooth_window_samples=SMOOTH_WINDOW_SAMPLES,
            baseline_window_samples=_baseline_window_samples(wf, sample_interval_s),
        )
        safe_region_name = region_name.replace(" ", "_").replace("(", "").replace(")", "")
        _plot_preprocessing_qc(
            stages, sample_interval_s, f"{group}/{name} :: {region_name} -- preprocessing stages",
            out_dir / f"preprocessing_{safe_region_name}.png",
        )

        detection_input = stages["baseline_corrected"]
        events_both = detect_beat_events_both_polarities(
            detection_input, sample_interval_s, frame_step, fps,
            min_prominence_ratio=MIN_PROMINENCE_RATIO,
            min_separation_period_fraction=MIN_SEPARATION_PERIOD_FRACTION,
        )
        _plot_events_overlay(
            detection_input, events_both, sample_interval_s,
            f"{group}/{name} :: {region_name} -- detected candidate events (both polarities)",
            out_dir / f"events_{safe_region_name}.png",
        )

        alternation = polarity_alternation_check(events_both["positive_events"], events_both["negative_events"])
        pos_summary = region_beat_summary(events_both["positive_events"], detection_input, sample_interval_s)
        neg_summary = region_beat_summary(events_both["negative_events"], detection_input, sample_interval_s)

        print(f"  [{region_name}] n_patches={len(patch_ids)}  "
              f"positive_events={pos_summary['n_events']} (mean_interval_s={pos_summary['mean_inter_event_interval_s']})  "
              f"negative_events={neg_summary['n_events']} (mean_interval_s={neg_summary['mean_inter_event_interval_s']})  "
              f"polarity_alternation_fraction={alternation['fraction_alternating']}")

        region_results[region_name] = {
            "n_patches": len(patch_ids),
            "positive_events_summary": pos_summary, "negative_events_summary": neg_summary,
            "polarity_alternation": alternation,
            "positive_events": events_both["positive_events"], "negative_events": events_both["negative_events"],
        }
        region_events_for_global[region_name] = events_both["positive_events"]

    # 5. regional activation delay between consecutive regions in the
    # manual activation_order (only meaningful when named multi-region
    # structure exists).
    activation_delays = []
    activation_order = ref.get("activation_order")
    if activation_order and len(activation_order) > 1:
        print(f"  activation_order: {' -> '.join(activation_order)}")
        for a, b in zip(activation_order[:-1], activation_order[1:]):
            if a not in region_results or b not in region_results:
                continue
            delay = regional_activation_delay(
                region_results[a]["positive_events"], region_results[b]["positive_events"],
                max_match_window_s=GLOBAL_BEAT_TOLERANCE_S,
            )
            print(f"    regional activation delay {a} -> {b}: {delay}")
            activation_delays.append({"pair": f"{a} -> {b}", **delay})

    # 6. simultaneous global-beat candidates (only if >1 region).
    global_candidates = []
    if len(region_events_for_global) > 1:
        global_candidates = global_beat_candidates(region_events_for_global, tolerance_s=GLOBAL_BEAT_TOLERANCE_S)
        print(f"  global beat candidates (all {len(region_events_for_global)} regions within "
              f"+/-{GLOBAL_BEAT_TOLERANCE_S}s): {len(global_candidates)}")

    return {
        "video": f"{group}/{name}", "manual_global_pattern": ref.get("global_pattern"),
        "manual_overall_strength": ref.get("overall_strength"),
        "fps": fps, "frame_step": frame_step, "sample_interval_s": sample_interval_s,
        "regions_analyzed": list(regions),
        "region_results": {
            k: {kk: vv for kk, vv in v.items() if kk not in ("positive_events", "negative_events")}
            for k, v in region_results.items()
        },
        "activation_delays": activation_delays,
        "n_global_beat_candidates": len(global_candidates),
        "global_beat_candidates": global_candidates,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    for group, name, video_dir, ref_path in VIDEOS:
        if not (video_dir / "motion_pipeline_qc.json").exists():
            print(f"skipping {group}/{name}: no motion_pipeline_qc.json at {video_dir}")
            continue
        result = process_video(group, name, video_dir, ref_path)
        all_results.append(result)

    with open(OUTPUT_DIR / "beat_event_detection_validation.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote beat-event detection validation for {len(all_results)} video(s) to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
