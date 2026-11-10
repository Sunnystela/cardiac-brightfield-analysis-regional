"""Waveform diagnostics beyond a single std-ratio SNR (Phase 2 validation).

None of these compute or report BPM, detect beats/peaks, or estimate
contraction velocity -- they are descriptive signal-quality diagnostics
only, to help judge whether a raw waveform looks like it contains real
periodic structure before any of that later work begins.
"""

from __future__ import annotations

import numpy as np


def autocorrelation(waveform: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Normalized autocorrelation of `waveform` (mean-subtracted, NaNs
    replaced with the series mean so a few missing samples don't break the
    correlation), for lags 0..max_lag (default: len(waveform) - 1).
    Returns an array where index 0 is always 1.0 (lag 0).
    """
    x = np.asarray(waveform, dtype=float)
    x = np.where(np.isnan(x), np.nanmean(x), x)
    x = x - x.mean()
    n = len(x)
    if max_lag is None:
        max_lag = n - 1
    denom = np.dot(x, x)
    if denom == 0:
        return np.zeros(max_lag + 1)
    return np.array([np.dot(x[: n - lag], x[lag:]) / denom for lag in range(max_lag + 1)])


def first_secondary_autocorrelation_peak(
    waveform: np.ndarray, sample_interval_s: float, min_lag: int = 1
) -> dict:
    """Find the first local maximum of the autocorrelation function at a
    lag >= min_lag (skipping the trivial lag-0 peak), as a periodicity
    diagnostic. Returns lag_s=None if no local maximum is found (e.g. the
    autocorrelation is monotonically decreasing -- no periodicity evidence
    at the lags examined).
    """
    ac = autocorrelation(waveform)
    for lag in range(min_lag, len(ac) - 1):
        if ac[lag] > ac[lag - 1] and ac[lag] >= ac[lag + 1]:
            return {"lag_samples": lag, "lag_s": lag * sample_interval_s, "autocorr_value": float(ac[lag])}
    return {"lag_samples": None, "lag_s": None, "autocorr_value": None}


def dominant_spectral_peak(waveform: np.ndarray, sample_interval_s: float) -> dict:
    """Periodogram of `waveform` (mean-subtracted, Hann-windowed, NaNs
    filled with the series mean) via numpy's real FFT. Returns the
    non-DC frequency bin with the largest power, in Hz, plus its power's
    ratio to the mean power of all other non-DC bins (a crude prominence
    measure: how much this bin stands out from the rest of the spectrum).

    Explicitly NOT converted to BPM or reported as a heart rate -- this is
    a raw-signal diagnostic pending validation, not a functional
    measurement.
    """
    x = np.asarray(waveform, dtype=float)
    x = np.where(np.isnan(x), np.nanmean(x), x)
    x = x - x.mean()
    n = len(x)
    if n < 4:
        return {"dominant_frequency_hz": None, "power": None, "prominence": None}
    window = np.hanning(n)
    spectrum = np.fft.rfft(x * window)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(n, d=sample_interval_s)

    if len(power) < 3:
        return {"dominant_frequency_hz": None, "power": None, "prominence": None}
    non_dc_power = power[1:]
    non_dc_freqs = freqs[1:]
    peak_idx = int(np.argmax(non_dc_power))
    peak_power = float(non_dc_power[peak_idx])
    other_power = np.delete(non_dc_power, peak_idx)
    mean_other = float(other_power.mean()) if other_power.size else 0.0
    prominence = peak_power / mean_other if mean_other > 0 else float("inf")

    return {
        "dominant_frequency_hz": float(non_dc_freqs[peak_idx]),
        "power": peak_power,
        "prominence": prominence,
    }


def _block_shuffle_surrogate(x: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Shuffle `x` by contiguous block, not individual sample -- preserves
    short-range (within-block) autocorrelation structure (e.g. smoothing
    from overlapping optical-flow windows across nearby frame pairs) while
    still destroying longer-range/periodic structure. A full per-sample
    shuffle can look spuriously "significant" for any waveform with even
    mild short-lag smoothness, since that null is unrealistically
    unstructured compared to the real data.
    """
    n = len(x)
    n_blocks = int(np.ceil(n / block_size))
    pad_len = n_blocks * block_size - n
    padded = np.concatenate([x, np.full(pad_len, x.mean())]) if pad_len > 0 else x
    blocks = padded.reshape(n_blocks, block_size)
    order = rng.permutation(n_blocks)
    return blocks[order].ravel()[:n]


def _phase_randomized_surrogate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Fourier phase-randomized surrogate: keeps `x`'s power spectrum (and
    therefore its full autocorrelation function) exactly, replaces phases
    with independent uniform-random values. Standard surrogate-data method
    (Theiler et al. 1992) for testing whether periodicity evidence goes
    beyond what the linear/autocorrelation structure alone would produce.
    """
    n = len(x)
    spectrum = np.fft.rfft(x - x.mean())
    magnitudes = np.abs(spectrum)
    phases = rng.uniform(0, 2 * np.pi, size=len(spectrum))
    phases[0] = 0.0
    if n % 2 == 0 and len(spectrum) > 1:
        phases[-1] = 0.0
    return np.fft.irfft(magnitudes * np.exp(1j * phases), n=n)


def periodicity_shuffle_test(
    waveform: np.ndarray,
    n_shuffles: int = 200,
    max_lag: int | None = None,
    rng: np.random.Generator | None = None,
    null_method: str = "full_shuffle",
    block_size: int = 5,
) -> dict:
    """Test whether `waveform` is more periodic than expected by chance,
    using a null built from `waveform` itself -- deliberately self-
    contained, since this project has no genuine non-beating organoid
    sample to use as a periodicity null (per project discussion: do not
    assume organoid patches or any available recording represent "no
    motion").

    `null_method` selects how the null surrogates are generated:
      - "full_shuffle": every sample independently reordered. Simplest,
        but destroys ALL temporal structure including short-range
        smoothness the real signal may share with pure noise -- can be
        too permissive (flags near-everything as "periodic") if the real
        waveform has even mild short-lag autocorrelation unrelated to true
        beating (e.g. from overlapping optical-flow windows). Confirmed on
        this project's own data: check per-video null_method comparisons.
      - "block_shuffle": contiguous blocks of `block_size` samples are
        reordered as units, preserving within-block autocorrelation.
      - "phase_randomized": Fourier phase-randomized surrogate, preserving
        the exact power spectrum / autocorrelation function while
        destroying phase relationships (the most conservative of the
        three -- see `_phase_randomized_surrogate`).
    Comparing results across all three is recommended before trusting any
    single one; none is assumed correct by default.

    Statistic: max |autocorrelation| at lag >= 1 (excludes the trivial
    lag-0 peak). `p_value` is the fraction of surrogates whose statistic is
    >= the observed one (a one-sided permutation-test p-value).
    """
    rng = rng or np.random.default_rng()
    x = np.asarray(waveform, dtype=float)
    x = np.where(np.isnan(x), np.nanmean(x), x)

    def _stat(series: np.ndarray) -> float:
        ac = autocorrelation(series, max_lag=max_lag)
        return float(np.max(np.abs(ac[1:]))) if len(ac) > 1 else 0.0

    if null_method == "full_shuffle":
        surrogate = lambda: rng.permutation(x)  # noqa: E731
    elif null_method == "block_shuffle":
        surrogate = lambda: _block_shuffle_surrogate(x, block_size, rng)  # noqa: E731
    elif null_method == "phase_randomized":
        surrogate = lambda: _phase_randomized_surrogate(x, rng)  # noqa: E731
    else:
        raise ValueError(f"Unknown null_method: {null_method!r}")

    observed = _stat(x)
    null_stats = np.array([_stat(surrogate()) for _ in range(n_shuffles)])
    p_value = float((null_stats >= observed).mean())

    result = {
        "null_method": null_method,
        "observed_statistic": observed,
        "p_value": p_value,
        "null_mean": float(null_stats.mean()),
        "null_std": float(null_stats.std()),
        "null_p95": float(np.percentile(null_stats, 95)),
        "n_shuffles": n_shuffles,
        "z_score": (
            float((observed - null_stats.mean()) / null_stats.std())
            if null_stats.std() > 0 else None
        ),
    }
    if null_method == "block_shuffle":
        result["block_size"] = block_size
    return result


def spectral_concentration(waveform: np.ndarray, sample_interval_s: float) -> float | None:
    """Fraction of total non-DC spectral power concentrated in the single
    dominant frequency bin (0 to 1) -- a normalized periodicity measure,
    unlike `dominant_spectral_peak`'s "prominence" which is unbounded.
    None if the waveform is too short or has zero non-DC power.
    """
    x = np.asarray(waveform, dtype=float)
    x = np.where(np.isnan(x), np.nanmean(x), x)
    x = x - x.mean()
    n = len(x)
    if n < 4:
        return None
    spectrum = np.fft.rfft(x * np.hanning(n))
    power = np.abs(spectrum) ** 2
    non_dc_power = power[1:]
    total = non_dc_power.sum()
    if total == 0:
        return None
    return float(non_dc_power.max() / total)


def waveform_consistency_across_steps(
    step_waveforms: dict[int, np.ndarray], fps: float
) -> dict:
    """Resample each step's waveform onto a common fine time grid (linear
    interpolation) covering their shared time range, then return the mean
    pairwise Pearson correlation across all step pairs -- a diagnostic for
    whether independently-subsampled step waveforms agree on *when* large
    excursions happen (strong agreement is evidence of real, temporally
    localized signal rather than independent per-pair noise).
    """
    if len(step_waveforms) < 2:
        return {"mean_pairwise_correlation": None, "pairwise": {}}

    grids = {}
    max_common_t = min(len(wf) * step / fps for step, wf in step_waveforms.items())
    common_t = np.linspace(0, max_common_t, 200)
    for step, wf in step_waveforms.items():
        t = np.arange(len(wf)) * (step / fps)
        grids[step] = np.interp(common_t, t, wf)

    steps = sorted(grids)
    pairwise = {}
    correlations = []
    for i in range(len(steps)):
        for j in range(i + 1, len(steps)):
            a, b = grids[steps[i]], grids[steps[j]]
            if a.std() == 0 or b.std() == 0:
                corr = 0.0
            else:
                corr = float(np.corrcoef(a, b)[0, 1])
            pairwise[f"{steps[i]}_vs_{steps[j]}"] = corr
            correlations.append(corr)

    return {
        "mean_pairwise_correlation": float(np.mean(correlations)),
        "pairwise": pairwise,
    }
