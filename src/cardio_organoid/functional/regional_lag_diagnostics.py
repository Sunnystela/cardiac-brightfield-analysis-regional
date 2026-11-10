"""Regional activation-delay / phase-lag diagnostics (round 15).

Manual observation on the 3 mature-candidate validation videos found that
some videos show several spatial regions beating in a consistent
SEQUENCE rather than either (a) one whole-organoid simultaneous movement
or (b) a single isolated localized patch -- see
results/manual_reference/mature/*/*/*.json's "regions" +
"activation_order" fields. sync_separation (organoid-organoid minus
organoid-background median |correlation|, using only a small +/-3-sample
lag tolerance) cannot distinguish "no coordinated activity" from "real
but sequentially-organized, non-simultaneous multi-region activation" --
a near-zero value is consistent with either. This module provides the
diagnostics needed to tell them apart, WITHOUT calling this conduction
velocity, propagation speed, BPM, or any biological conclusion -- purely
descriptive regional-activation-delay / phase-lag numbers.

Does not change sync_separation's calculation or any existing threshold.
"""

from __future__ import annotations

import numpy as np

from cardio_organoid.motion.diagnostics import dominant_spectral_peak


def region_mean_waveform(
    waveforms_by_patch: dict[tuple[int, int], np.ndarray], region_patch_ids: list[tuple[int, int]]
) -> np.ndarray | None:
    """Simple mean, across the region's member patches, of their raw
    radial-motion waveforms (already on a shared frame-pair time base --
    no resampling needed, same convention as
    spatial_patches.cross_patch_synchronization). Returns None if none of
    the region's listed patches are present in `waveforms_by_patch`
    (e.g. a coordinate outside the analyzed grid)."""
    found = [waveforms_by_patch[pid] for pid in region_patch_ids if pid in waveforms_by_patch]
    if not found:
        return None
    return np.mean(np.stack(found), axis=0)


def zero_lag_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def cross_correlation_curve(a: np.ndarray, b: np.ndarray, max_lag_samples: int) -> dict:
    """Signed Pearson correlation between `a` and `b` at every integer lag
    in [-max_lag_samples, +max_lag_samples]. Convention: at lag L >= 0,
    a[k] is compared against b[k+L] -- so if the correlation peaks at
    L > 0, `b`'s pattern at time k+L resembles `a`'s pattern at time k,
    i.e. `b` reaches at time k+L what `a` already showed at time k: `b`
    trails `a` by L samples ("a activates first"). This is the same
    slicing convention as
    cardio_organoid.functional.phenotype_axes._best_lag_abs_correlation,
    kept consistent across the project.

    Returns the full curve (for QC plotting/inspecting whether the
    reported best lag is a clean, isolated peak or ambiguous among
    several comparable ones -- periodic waveforms can have multiple
    plausible peaks a full cycle apart) plus the best lag as a single
    summary number.

    Best lag is chosen by the highest SIGNED (not absolute) correlation,
    not abs(correlation) -- for a near-periodic waveform, the lag at
    exactly half a period is anti-correlated (corr ~= -1.0) and ties
    abs(correlation) with the true matching lag (corr ~= +1.0), which a
    max-|corr| rule cannot break (confirmed on a synthetic pure sine in
    tests/test_regional_lag_diagnostics.py). A real sequential activation
    delay should make two regions move in the SAME direction at the
    correct lag (positively correlated), not flip to anti-phase, so
    signed correlation is also the physically appropriate choice here --
    unlike phenotype_axes._best_lag_abs_correlation, which deliberately
    uses abs() for a different question ("are these two patches related
    at all", where an anti-phase relationship is still informative).
    """
    lags = list(range(-max_lag_samples, max_lag_samples + 1))
    correlations = []
    for lag in lags:
        if lag >= 0:
            x, y = a[: len(a) - lag or None], b[lag:]
        else:
            x, y = a[-lag:], b[: len(b) + lag or None]
        if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
            correlations.append(None)
            continue
        correlations.append(float(np.corrcoef(x, y)[0, 1]))

    valid = [(lag, c) for lag, c in zip(lags, correlations) if c is not None]
    if not valid:
        return {"lags_samples": lags, "correlations": correlations, "best_lag_samples": None, "best_correlation": None}
    best_lag, best_corr = max(valid, key=lambda lc: lc[1])
    return {
        "lags_samples": lags, "correlations": correlations,
        "best_lag_samples": best_lag, "best_correlation": best_corr,
    }


def simple_peak_indices(waveform: np.ndarray, min_separation_samples: int) -> list[int]:
    """Local maxima of `waveform` (x[i-1] < x[i] >= x[i+1]), with any two
    reported peaks required to be at least `min_separation_samples` apart
    (keeping the taller of a too-close pair) -- a plain, dependency-free
    peak picker (no scipy), self-relative in the sense that the caller is
    expected to derive `min_separation_samples` from the waveform's own
    estimated dominant period, not an assumed absolute value.
    """
    x = np.asarray(waveform, dtype=float)
    n = len(x)
    raw_peaks = [i for i in range(1, n - 1) if x[i - 1] < x[i] >= x[i + 1]]
    if not raw_peaks:
        return []
    raw_peaks.sort(key=lambda i: x[i], reverse=True)
    kept: list[int] = []
    for i in raw_peaks:
        if all(abs(i - j) >= min_separation_samples for j in kept):
            kept.append(i)
    return sorted(kept)


def matched_peak_timing_differences(
    peaks_a_s: list[float], peaks_b_s: list[float], max_match_window_s: float
) -> dict:
    """For each peak time in `peaks_a_s`, find the nearest peak time in
    `peaks_b_s` within `max_match_window_s`; report the matched pairs'
    (signed) time differences (b_time - a_time), so a consistently
    positive mean means b's peaks consistently follow a's -- an
    independent, per-cycle complement to the single aggregate
    cross_correlation_curve lag (which assumes one fixed delay across the
    whole recording).
    """
    matched = []
    used_b = set()
    for ta in peaks_a_s:
        candidates = [(j, tb) for j, tb in enumerate(peaks_b_s) if j not in used_b and abs(tb - ta) <= max_match_window_s]
        if not candidates:
            continue
        j, tb = min(candidates, key=lambda jt: abs(jt[1] - ta))
        used_b.add(j)
        matched.append(tb - ta)
    if not matched:
        return {"n_matched": 0, "n_peaks_a": len(peaks_a_s), "n_peaks_b": len(peaks_b_s),
                "mean_diff_s": None, "median_diff_s": None, "diffs_s": []}
    return {
        "n_matched": len(matched), "n_peaks_a": len(peaks_a_s), "n_peaks_b": len(peaks_b_s),
        "mean_diff_s": float(np.mean(matched)), "median_diff_s": float(np.median(matched)),
        "diffs_s": matched,
    }


def event_triggered_average(
    target_waveform: np.ndarray,
    trigger_peak_indices: list[int],
    sample_interval_s: float,
    window_s: float,
) -> dict:
    """Average `target_waveform` in a window around each of
    `trigger_peak_indices` (sample indices of already-detected peaks in
    some OTHER, "trigger", waveform), aligned on trigger-peak-relative
    time (0 = the trigger peak). Only peaks whose full +/-window_s window
    fits inside the recording are used.

    This is a stronger diagnostic than a single cross-correlation-derived
    "best lag" scalar: it shows the actual AVERAGED shape of the target
    region's motion relative to every trigger event, across the whole
    recording, so a consistent bump at the same relative delay (not just
    one summary number) is direct, visual evidence of a repeating
    relationship -- and a flat/noisy average is direct evidence AGAINST
    one, rather than relying on a single (possibly noise-driven) peak in
    a cross-correlation curve.
    """
    window_samples = int(round(window_s / sample_interval_s))
    offsets = np.arange(-window_samples, window_samples + 1)
    snippets = [
        target_waveform[idx - window_samples : idx + window_samples + 1]
        for idx in trigger_peak_indices
        if idx - window_samples >= 0 and idx + window_samples < len(target_waveform)
    ]
    if not snippets:
        return {
            "time_offsets_s": (offsets * sample_interval_s).tolist(),
            "mean_snippet": None, "std_snippet": None, "n_events_used": 0,
        }
    stacked = np.stack(snippets)
    return {
        "time_offsets_s": (offsets * sample_interval_s).tolist(),
        "mean_snippet": np.mean(stacked, axis=0).tolist(),
        "std_snippet": np.std(stacked, axis=0).tolist(),
        "n_events_used": len(snippets),
    }


def power_spectrum(waveform: np.ndarray, sample_interval_s: float) -> dict:
    """Full (not just dominant-peak) power spectrum of `waveform`
    (mean-subtracted, Hann-windowed, same convention as
    diagnostics.dominant_spectral_peak) -- needed to check whether a
    region's detected "dominant frequency" is genuinely the strongest
    component or whether a harmonically-related frequency (e.g. half or
    double another region's dominant frequency) also carries substantial
    power, which would suggest the two regions may actually share a
    fundamental frequency and the single-peak dominant-frequency estimate
    for one of them is a peak-detection artifact (aliasing to a
    sub/super-harmonic bin), not a genuine 2:1 (or other ratio) frequency
    difference.
    """
    x = np.asarray(waveform, dtype=float)
    x = np.where(np.isnan(x), np.nanmean(x), x)
    x = x - x.mean()
    n = len(x)
    if n < 4:
        return {"freqs_hz": [], "power": []}
    spectrum = np.fft.rfft(x * np.hanning(n))
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(n, d=sample_interval_s)
    return {"freqs_hz": freqs.tolist(), "power": power.tolist()}


def power_at_frequency(spectrum: dict, target_freq_hz: float, tolerance_hz: float) -> float | None:
    """Total power within +/-tolerance_hz of `target_freq_hz` in a
    `power_spectrum` result, relative to the spectrum's own total non-DC
    power (0-1) -- used to check how much power a region's waveform has
    near ANOTHER region's dominant frequency (harmonic-relation check)."""
    freqs = np.array(spectrum["freqs_hz"])
    power = np.array(spectrum["power"])
    if len(freqs) < 2:
        return None
    non_dc = power[1:].sum()
    if non_dc == 0:
        return None
    band = (freqs >= target_freq_hz - tolerance_hz) & (freqs <= target_freq_hz + tolerance_hz) & (freqs > 0)
    return float(power[band].sum() / non_dc)


def region_pair_lag_diagnostic(
    waveform_a: np.ndarray,
    waveform_b: np.ndarray,
    sample_interval_s: float,
    max_lag_s: float = 1.0,
) -> dict:
    """Bundle of all requested region-pair diagnostics: zero-lag
    correlation, lag-tolerant cross-correlation (full curve + best lag in
    samples and seconds), each region's own dominant frequency, and a
    peak-matching-based activation timing difference. `max_lag_s` (a time
    value, not a fixed sample count, so it means the same thing across
    videos recorded at different frame rates) defaults to 1.0s -- wide
    enough to cover roughly one full beat cycle at the ~1-3 Hz dominant
    frequencies seen so far in this project's sample videos, narrow
    enough to keep the search within one cycle and avoid picking a
    lag from an adjacent cycle by aliasing. This is a diagnostic
    parameter, not a validated or tuned threshold.
    """
    max_lag_samples = max(1, int(round(max_lag_s / sample_interval_s)))
    zero_lag = zero_lag_correlation(waveform_a, waveform_b)
    curve = cross_correlation_curve(waveform_a, waveform_b, max_lag_samples)
    freq_a = dominant_spectral_peak(waveform_a, sample_interval_s)
    freq_b = dominant_spectral_peak(waveform_b, sample_interval_s)

    # Peak-matching: derive the minimum peak separation from whichever
    # region's own dominant frequency is available (falls back to a
    # generous 5-sample minimum if neither could be estimated).
    est_freq = freq_a["dominant_frequency_hz"] or freq_b["dominant_frequency_hz"]
    min_sep_samples = max(2, int(round(0.5 / (est_freq * sample_interval_s)))) if est_freq else 5
    peaks_a = simple_peak_indices(waveform_a, min_sep_samples)
    peaks_b = simple_peak_indices(waveform_b, min_sep_samples)
    peaks_a_s = [i * sample_interval_s for i in peaks_a]
    peaks_b_s = [i * sample_interval_s for i in peaks_b]
    peak_matching = matched_peak_timing_differences(peaks_a_s, peaks_b_s, max_match_window_s=max_lag_s)

    best_lag_samples = curve["best_lag_samples"]
    return {
        "zero_lag_correlation": zero_lag,
        "cross_correlation": {
            "lags_samples": curve["lags_samples"], "correlations": curve["correlations"],
            "best_lag_samples": best_lag_samples,
            "best_lag_s": best_lag_samples * sample_interval_s if best_lag_samples is not None else None,
            "best_correlation": curve["best_correlation"],
        },
        "dominant_frequency_hz": {"region_a": freq_a["dominant_frequency_hz"], "region_b": freq_b["dominant_frequency_hz"]},
        "peak_timing_difference": peak_matching,
        "max_lag_s_searched": max_lag_s,
    }
