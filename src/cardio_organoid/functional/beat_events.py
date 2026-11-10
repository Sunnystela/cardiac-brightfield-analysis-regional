"""Region-level beat-event detection prototype (round 18).

Goal for this round, stated plainly (per project decision): can a real
cardiac beating EVENT be reliably detected from the region-average
motion waveforms already produced by the existing motion-analysis
architecture (Farneback optical flow -> spatial activity heatmap ->
regional localization -> region-average waveform)? This module does NOT
change that upstream architecture -- it only adds a new stage that
operates on its output waveforms.

Explicitly NOT finalized by this module: BPM, arrhythmia classification,
a single collapsed whole-organoid rhythm, or which signed direction
(positive/negative radial motion) corresponds to true contraction.
Candidate beat events, per-region only, with both waveform excursion
directions reported side by side for a human to check against the raw
video.

FPS/timebase discipline (per project decision, re-verified round 17 for
every video currently in the project -- see results/metadata_qc/):
every waveform sample corresponds to `frame_step` ORIGINAL video frames,
and the file's own file-recorded fps (OpenCV CAP_PROP_FPS) is the only
timebase used -- never a nominal/target fps, and this must hold
unchanged whether a video is 22.09 fps (early pilot), 29.49 fps, or
30.08 fps (both occur in this project's actual sample set). Every stored
record below carries fps, frame_step, sample_interval_s, the
original-video frame index, and time_sec, so:

    lag_seconds = lag_samples * frame_step / fps

is always reconstructible without re-deriving it from scratch or
assuming a project-wide constant.
"""

from __future__ import annotations

import numpy as np

from cardio_organoid.motion.diagnostics import dominant_spectral_peak, spectral_concentration


# --------------------------------------------------------------------------
# 1. Region waveform preprocessing -- each stage separate and independently
#    callable; the raw waveform is never discarded.
# --------------------------------------------------------------------------


def linear_detrend(waveform: np.ndarray) -> np.ndarray:
    """Subtract the best-fit straight line (ordinary least squares against
    sample index) -- removes constant-velocity drift only (e.g. slow
    stage settling), not the fast, real contraction/relaxation motion a
    much longer/nonlinear correction could remove. Optional -- callers
    that don't want this may skip straight to smoothing/baseline
    correction on the raw waveform."""
    n = len(waveform)
    if n < 2:
        return np.asarray(waveform, dtype=float).copy()
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, waveform, deg=1)
    return waveform - (slope * x + intercept)


def light_smooth(waveform: np.ndarray, window_samples: int = 3) -> np.ndarray:
    """A DELIBERATELY small moving-average (default 3 samples) -- only
    enough to suppress single-sample optical-flow measurement noise, not
    to smear a genuine fast contraction. `window_samples` is a parameter,
    never hardcoded larger by this function; callers analyzing a video
    with a much higher effective sampling rate may still want to keep
    this small in absolute time terms, not just sample count -- pass a
    smaller window_samples if `frame_step`/fps make one sample span very
    little time, and vice versa. window_samples <= 1 returns the input
    unchanged (no smoothing applied)."""
    window_samples = max(1, int(window_samples))
    if window_samples <= 1 or len(waveform) < window_samples:
        return np.asarray(waveform, dtype=float).copy()
    kernel = np.ones(window_samples) / window_samples
    return np.convolve(waveform, kernel, mode="same")


def rolling_baseline_correct(waveform: np.ndarray, window_samples: int) -> np.ndarray:
    """Subtract a rolling MEDIAN (robust to brief large excursions, unlike
    a rolling mean) computed over a window centered on each sample --
    corrects slow, potentially NONLINEAR baseline wander that
    `linear_detrend` (a single straight line for the whole recording)
    cannot. `window_samples` should be set by the caller to something
    much longer than one beat cycle (e.g. several periods), so the
    "baseline" being removed is genuinely slow drift, not the beat itself.
    A window_samples >= len(waveform) subtracts one single global median
    (equivalent to simple centering)."""
    n = len(waveform)
    window_samples = max(1, int(window_samples))
    x = np.asarray(waveform, dtype=float)
    half = window_samples // 2
    baseline = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        baseline[i] = np.median(x[lo:hi])
    return x - baseline


def preprocess_waveform(
    waveform: np.ndarray,
    apply_detrend: bool = True,
    smooth_window_samples: int = 3,
    baseline_window_samples: int | None = None,
) -> dict:
    """Runs the three preprocessing stages in a fixed, documented order
    (detrend -> smooth -> baseline-correct), but returns EVERY
    intermediate stage, including the untouched raw waveform, so a QC
    plot can show all of them together and nothing is silently discarded.
    Any stage can be disabled: `apply_detrend=False` skips detrending,
    `smooth_window_samples<=1` skips smoothing, `baseline_window_samples=None`
    skips baseline correction.
    """
    raw = np.asarray(waveform, dtype=float).copy()
    stage = raw
    detrended = linear_detrend(stage) if apply_detrend else stage.copy()
    stage = detrended
    smoothed = light_smooth(stage, smooth_window_samples) if smooth_window_samples > 1 else stage.copy()
    stage = smoothed
    baseline_corrected = (
        rolling_baseline_correct(stage, baseline_window_samples) if baseline_window_samples else stage.copy()
    )
    return {
        "raw": raw, "detrended": detrended, "smoothed": smoothed, "baseline_corrected": baseline_corrected,
        "params": {
            "apply_detrend": apply_detrend, "smooth_window_samples": smooth_window_samples,
            "baseline_window_samples": baseline_window_samples,
        },
    }


# --------------------------------------------------------------------------
# 2. Beat event detection -- not scipy find_peaks with default settings.
#    Prominence relative to the waveform's OWN robust noise floor
#    (self-relative, per this project's established convention), minimum
#    separation derived from the waveform's own estimated dominant period
#    (not an assumed physiological constant).
# --------------------------------------------------------------------------


def robust_noise_floor(waveform: np.ndarray) -> float:
    """Normal-consistent robust std of the waveform's first differences
    (MAD-based) -- a self-relative estimate of sample-to-sample
    measurement noise, used as the reference scale for prominence
    filtering below. Not an absolute magic threshold: this is computed
    fresh from each waveform, never a fixed number reused across videos."""
    diffs = np.diff(np.asarray(waveform, dtype=float))
    if len(diffs) == 0:
        return 0.0
    mad = np.median(np.abs(diffs - np.median(diffs)))
    return float(1.4826 * mad)


def peak_prominence(x: np.ndarray, peak_idx: int) -> float:
    """Standard topographic peak prominence: from the peak, scan outward
    in each direction until a HIGHER point is found (or the array ends),
    tracking the minimum value seen along that scan; prominence is the
    peak's height above the higher of the two scans' minima. Same
    definition scipy.signal.peak_prominences uses (no wlen truncation
    here -- these waveforms are short, a few hundred samples, so an
    unbounded scan is cheap)."""
    v = x[peak_idx]

    def _scan(direction: int) -> float:
        i = peak_idx + direction
        min_val = v
        while 0 <= i < len(x):
            if x[i] > v:
                break
            min_val = min(min_val, x[i])
            i += direction
        return min_val

    return float(v - max(_scan(-1), _scan(1)))


def _local_maxima(x: np.ndarray) -> list[int]:
    return [i for i in range(1, len(x) - 1) if x[i - 1] < x[i] >= x[i + 1]]


def detect_candidate_events(
    waveform: np.ndarray,
    sample_interval_s: float,
    frame_step: int,
    fps: float,
    min_prominence_ratio: float = 3.0,
    min_separation_s: float | None = None,
    min_separation_period_fraction: float = 0.4,
    start_frame_index: int = 0,
) -> list[dict]:
    """Candidate beat events at local maxima of `waveform` whose
    prominence exceeds `min_prominence_ratio` x the waveform's own
    robust_noise_floor -- NOT scipy find_peaks with default settings,
    and not an absolute magic amplitude. `min_separation_s` defaults to
    `min_separation_period_fraction` of the waveform's own estimated
    dominant period (dominant_spectral_peak) when not given explicitly,
    so closely-spaced noisy sub-peaks within one real beat cycle are
    suppressed without assuming a fixed physiological rate; if no clear
    dominant frequency can be estimated, falls back to 0 (no separation
    enforced beyond the two-sample minimum inherent to local-maxima
    detection).

    Each returned event dict carries the timebase fields required by
    every downstream step in this module: fps, frame_step,
    sample_interval_s, the original-video frame_index
    (start_frame_index + sample_index * frame_step), and time_sec
    (sample_index * sample_interval_s) -- so lag_seconds = lag_samples *
    frame_step / fps is always reconstructible from stored fields alone.
    """
    x = np.asarray(waveform, dtype=float)
    n = len(x)
    if n < 3:
        return []

    if min_separation_s is None:
        peak_info = dominant_spectral_peak(x, sample_interval_s)
        freq = peak_info["dominant_frequency_hz"]
        min_separation_s = min_separation_period_fraction / freq if freq and freq > 0 else 0.0
    min_separation_samples = int(round(min_separation_s / sample_interval_s)) if sample_interval_s > 0 else 0

    noise_floor = robust_noise_floor(x)
    prominence_cutoff = min_prominence_ratio * noise_floor if noise_floor > 0 else 0.0

    raw_peaks = _local_maxima(x)
    scored = [(i, peak_prominence(x, i)) for i in raw_peaks]
    candidates = [(i, p) for i, p in scored if p >= prominence_cutoff]
    # Enforce minimum separation, keeping the more prominent of any two
    # candidates closer together than min_separation_samples (same
    # greedy-by-height strategy as regional_lag_diagnostics.simple_peak_indices).
    candidates.sort(key=lambda ip: ip[1], reverse=True)
    kept: list[tuple[int, float]] = []
    for i, prom in candidates:
        if all(abs(i - j) >= min_separation_samples for j, _ in kept):
            kept.append((i, prom))
    kept.sort(key=lambda ip: ip[0])

    events = []
    for k, (idx, prom) in enumerate(kept):
        events.append({
            "sample_index": idx,
            "frame_index": start_frame_index + idx * frame_step,
            "time_sec": idx * sample_interval_s,
            "local_amplitude": float(x[idx]),
            "prominence": prom,
            "preceding_interval_s": (
                (idx - kept[k - 1][0]) * sample_interval_s if k > 0 else None
            ),
            "following_interval_s": (
                (kept[k + 1][0] - idx) * sample_interval_s if k < len(kept) - 1 else None
            ),
            "fps": fps, "frame_step": frame_step, "sample_interval_s": sample_interval_s,
        })
    return events


def detect_beat_events_both_polarities(
    waveform: np.ndarray,
    sample_interval_s: float,
    frame_step: int,
    fps: float,
    **kwargs,
) -> dict:
    """Signed radial motion includes both inward and outward excursions,
    and this module does not assume which one is contraction (see module
    docstring / point 3) -- detects candidate events on the waveform
    as-is ("positive_events") AND on its negation ("negative_events"),
    so both are available for a human to check against the actual video.
    """
    positive_events = detect_candidate_events(waveform, sample_interval_s, frame_step, fps, **kwargs)
    negative_events = detect_candidate_events(-np.asarray(waveform, dtype=float), sample_interval_s, frame_step, fps, **kwargs)
    for e in negative_events:
        e["local_amplitude"] = -e["local_amplitude"]  # report in the ORIGINAL waveform's sign convention
    return {"positive_events": positive_events, "negative_events": negative_events}


def polarity_alternation_check(positive_events: list[dict], negative_events: list[dict]) -> dict:
    """Descriptive check only (no polarity decision made): if positive and
    negative excursions are the contraction/relaxation halves of the same
    oscillation, their event times should roughly ALTERNATE (a negative
    event near the midpoint between consecutive positive events, and vice
    versa) -- reports how often that holds, for a human to weigh
    alongside actually watching the video (see module docstring)."""
    pos_times = [e["time_sec"] for e in positive_events]
    neg_times = [e["time_sec"] for e in negative_events]
    if len(pos_times) < 2 or not neg_times:
        return {"n_positive": len(pos_times), "n_negative": len(neg_times), "fraction_alternating": None}
    n_alternating = 0
    for a, b in zip(pos_times[:-1], pos_times[1:]):
        midpoint_candidates = [t for t in neg_times if a < t < b]
        if midpoint_candidates:
            n_alternating += 1
    return {
        "n_positive": len(pos_times), "n_negative": len(neg_times),
        "n_positive_gaps_checked": len(pos_times) - 1,
        "n_alternating": n_alternating,
        "fraction_alternating": n_alternating / (len(pos_times) - 1),
    }


# --------------------------------------------------------------------------
# 4. Region-level summary (waveform + events + amplitude + interval +
#    dominant frequency + periodicity, all kept PER region).
# --------------------------------------------------------------------------


def region_beat_summary(events: list[dict], waveform: np.ndarray, sample_interval_s: float) -> dict:
    amplitudes = [e["local_amplitude"] for e in events]
    intervals = [e["preceding_interval_s"] for e in events if e["preceding_interval_s"] is not None]
    peak_info = dominant_spectral_peak(waveform, sample_interval_s)
    return {
        "n_events": len(events),
        "mean_amplitude": float(np.mean(amplitudes)) if amplitudes else None,
        "mean_inter_event_interval_s": float(np.mean(intervals)) if intervals else None,
        "std_inter_event_interval_s": float(np.std(intervals)) if intervals else None,
        "dominant_frequency_hz": peak_info["dominant_frequency_hz"],
        "spectral_concentration": spectral_concentration(waveform, sample_interval_s),
    }


# --------------------------------------------------------------------------
# 5. Regional timing interface (activation delay / phase lag -- NEVER
#    conduction velocity or propagation speed).
# --------------------------------------------------------------------------


def regional_activation_delay(
    events_a: list[dict], events_b: list[dict], max_match_window_s: float
) -> dict:
    """For each event in region A, find the nearest event in region B
    within `max_match_window_s`; report matched pairs' (signed) time
    differences (b - a). A consistently positive mean/median is
    descriptive evidence of "region A tends to lead region B" -- reported
    as a "regional activation delay" / "regional phase lag" candidate
    ONLY, never conduction velocity or propagation speed (no distance is
    involved, and this is not asserted as validated or final).
    """
    times_a = [e["time_sec"] for e in events_a]
    times_b = [e["time_sec"] for e in events_b]
    matched = []
    used_b = set()
    for ta in times_a:
        candidates = [(j, tb) for j, tb in enumerate(times_b) if j not in used_b and abs(tb - ta) <= max_match_window_s]
        if not candidates:
            continue
        j, tb = min(candidates, key=lambda jt: abs(jt[1] - ta))
        used_b.add(j)
        matched.append(tb - ta)
    if not matched:
        return {"n_matched": 0, "n_events_a": len(times_a), "n_events_b": len(times_b),
                "mean_delay_s": None, "median_delay_s": None}
    return {
        "n_matched": len(matched), "n_events_a": len(times_a), "n_events_b": len(times_b),
        "mean_delay_s": float(np.mean(matched)), "median_delay_s": float(np.median(matched)),
    }


# --------------------------------------------------------------------------
# 6. Simultaneous global beating candidates -- only constructed where
#    several regions' events repeatedly cluster in a narrow window; never
#    forced, and never collapsed into one BPM.
# --------------------------------------------------------------------------


def global_beat_candidates(
    region_events: dict[str, list[dict]], tolerance_s: float, min_regions: int | None = None
) -> list[dict]:
    """A "global beat candidate" is one event, from ANY region, for which
    every OTHER required region also has a matching event within
    +/-tolerance_s. `min_regions` (default: all regions given) sets how
    many DISTINCT regions must agree for one instance to count -- with
    the default (all regions), a purely localized or purely sequential
    (large, consistent, non-overlapping delay) video should correctly
    produce few or zero candidates, without any threshold tuned to force
    a particular outcome. Does not compute or return a BPM -- only the
    list of joint-event instances and which regions participated in each.
    """
    names = list(region_events)
    if min_regions is None:
        min_regions = len(names)
    if len(names) < 2:
        return []

    candidates = []
    for ref_name in names:
        for e in region_events[ref_name]:
            t = e["time_sec"]
            participating = {ref_name}
            for other_name in names:
                if other_name == ref_name:
                    continue
                if any(abs(oe["time_sec"] - t) <= tolerance_s for oe in region_events[other_name]):
                    participating.add(other_name)
            if len(participating) >= min_regions:
                candidates.append({"time_sec": t, "reference_region": ref_name, "participating_regions": sorted(participating)})
    # De-duplicate near-identical instances (same joint event found once
    # per triggering region) by time, keeping the first.
    candidates.sort(key=lambda c: c["time_sec"])
    deduped = []
    for c in candidates:
        if not deduped or c["time_sec"] - deduped[-1]["time_sec"] > tolerance_s:
            deduped.append(c)
    return deduped
