"""Continuous, threshold-free whole-video functional phenotype axes
(round 11; supersedes trying to fit a single binary active/inactive
patch classifier to every beating pattern -- see
scripts/compare_manual_pattern_reference.py's findings, which showed
that classifier cleanly separating localized/synchronized/diffuse
phenotypes with one amplitude+periodicity AND-gate is not appropriate:
a genuine dominant region can fail a periodicity gate, and a genuine
weak-but-coordinated organoid has no compact "active" region for any
percentile rule to find).

Four independent, continuous axes, each computed without any hard
threshold or AND-gate:

  motion_strength    -- how strong is motion anywhere in the organoid
                        (population summary stats of the existing
                        amplitude_radial_std score, already computed
                        upstream -- not a new score).
  spatial_extent      -- how spatially broad the elevated-amplitude
                        region is (participation ratio -- an "effective
                        fraction of patches carrying the activity",
                        continuous in (0, 1], no percentile cutoff).
  regional_dominance  -- how concentrated amplitude is in a small
                        sub-region vs. spread evenly (Gini coefficient
                        of the amplitude distribution, continuous in
                        [0, 1)).
  global_synchronization -- how consistently ALL valid patches (not
                        just ones an amplitude/periodicity gate already
                        flagged "active") move together in time.
                        Reported as five independent candidate metrics,
                        deliberately not combined into one number yet:
                          - mean pairwise zero-lag correlation
                          - mean pairwise correlation allowing a small
                            temporal lag (captures near-simultaneous but
                            not frame-exact timing, e.g. a propagating
                            contraction wave)
                          - mean pairwise Welch spectral coherence
                            (peak over frequency, since no beat-
                            frequency band is fixed project-wide yet)
                          - Kuramoto phase-locking order parameter,
                            using the instantaneous phase of every
                            patch's waveform band-passed around one
                            data-derived reference frequency (the peak
                            of the *pooled* periodogram across all
                            patches -- not an assumed Hz value)
                          - PCA first-principal-component explained
                            variance ratio, computed on per-patch
                            z-scored waveforms (so a shared *timing*
                            pattern is detected even if amplitude is
                            tiny and uneven across patches -- the key
                            property needed to tell a weak-but-globally-
                            synchronized organoid apart from noise).

None of this decides a final phenotype label, computes BPM, or claims a
biological conclusion -- see docstrings below and each function's
"continuous, no threshold" framing. Uses only the per-patch radial-
motion waveforms already produced by
cardio_organoid.pipeline.analyze.run_motion_pipeline (patch_waveforms_radial.npz)
and the amplitude_radial_std score already in motion_pipeline_qc.json --
no score is recomputed with a different definition.
"""

from __future__ import annotations

import hashlib

import numpy as np

from cardio_organoid.motion.spatial_patches import cluster_active_patches_into_regions


_MIN_MEANINGFUL_STD = 1e-9
"""Below this, a waveform's std is floating-point noise from a patch with
no real measured motion at all (e.g. a textureless background-like cell
where optical flow returns ~0 every frame), not a real, if tiny,
biological signal -- confirmed on real data: slit/a1 and slit/b1 each had
roughly HALF their patch grid at amplitude ~1e-15/1e-16 (vs. the smallest
genuinely-measured amplitude anywhere in the sample set, ~2e-4). Treating
`std == 0` as the only degenerate case let these patches' pure floating-
point noise get divided by its own near-zero std and blown up to unit
variance during z-scoring, contaminating consensus_waveform and every
correlation computed against it. 1e-9 is far below any real background-
noise floor (~0.02, see motion_pipeline_qc.json's
null_reference_stats_radial_std_windowed) or real signal in this
dataset -- this excludes only numerically-degenerate patches, never a
real low-amplitude one."""


def _deterministic_seed(*parts) -> int:
    """Same approach as cardio_organoid.pipeline.analyze.deterministic_seed
    (duplicated locally rather than imported, to avoid a functional ->
    pipeline dependency -- see that function's docstring for why
    Python's built-in hash() must not be used for an RNG seed here)."""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def _fill_nan_linear(waveform: np.ndarray) -> np.ndarray:
    """Linearly interpolate interior NaNs and forward/back-fill any
    leading/trailing NaNs. Only used as input pre-processing for the
    synchronization candidates below (correlation/coherence/PCA/phase all
    require a complete series) -- never modifies the stored waveform or
    any already-computed score, which use nanmean/nanstd directly."""
    x = np.asarray(waveform, dtype=float).copy()
    n = len(x)
    valid = ~np.isnan(x)
    if valid.all() or not valid.any():
        return x
    idx = np.arange(n)
    x[~valid] = np.interp(idx[~valid], idx[valid], x[valid])
    return x


# --------------------------------------------------------------------------
# motion_strength
# --------------------------------------------------------------------------


def motion_strength_stats(amplitude_by_patch: dict) -> dict:
    """Population summary of the existing per-patch amplitude score
    (radial_std), continuous, no threshold. Reports several complementary
    statistics rather than collapsing to one number: `median`/`mean`
    describe the "typical" patch (low for a localized-beating video, where
    most of the organoid is quiet even though one patch beats strongly),
    while `p90`/`max` describe the strongest motion observed anywhere
    (informative for exactly that localized case). Which of these best
    matches a manually judged "overall_strength" is left for the
    validation comparison, not assumed here.
    """
    values = np.array(list(amplitude_by_patch.values()), dtype=float)
    if values.size == 0:
        return {"median": None, "mean": None, "p90": None, "max": None, "n_patches": 0}
    return {
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
        "n_patches": int(values.size),
    }


# --------------------------------------------------------------------------
# spatial_extent
# --------------------------------------------------------------------------


def participation_ratio(weights_by_patch: dict) -> float | None:
    """Continuous "effective fraction of patches carrying the activity",
    from the participation ratio / inverse Simpson index:
    PR = (sum w_i)^2 / (n * sum w_i^2), for non-negative weights w_i
    (e.g. amplitude_radial_std). If activity is spread evenly across all
    n patches, PR -> 1 (fully extended); if concentrated in one patch,
    PR -> 1/n (minimal extent). No percentile or magnitude cutoff is
    used to decide which patches "count" -- every valid patch contributes
    in proportion to its own weight.
    """
    values = np.array(list(weights_by_patch.values()), dtype=float)
    values = np.clip(values, 0.0, None)
    n = values.size
    if n == 0:
        return None
    total = values.sum()
    sq_total = np.sum(values**2)
    if sq_total == 0:
        return None  # every weight is zero -- extent undefined, not zero
    return float((total**2) / (n * sq_total))


# --------------------------------------------------------------------------
# regional_dominance
# --------------------------------------------------------------------------


def gini_coefficient(values_by_patch: dict) -> float | None:
    """Gini coefficient of the amplitude distribution across patches, a
    continuous, threshold-free concentration measure: 0 means amplitude is
    identical across every patch (no dominant sub-region), values closer
    to 1 mean amplitude is concentrated in a small subset of patches (a
    clear dominant region). Standard mean-absolute-difference formula;
    values are clipped to non-negative (amplitude scores are already
    non-negative by construction, this guards the formula only).
    """
    values = np.clip(np.array(list(values_by_patch.values()), dtype=float), 0.0, None)
    n = values.size
    if n < 2:
        return None
    if values.sum() == 0:
        return 0.0
    sorted_vals = np.sort(values)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_vals)) / (n * np.sum(sorted_vals)) - (n + 1) / n)


def top_patches_by_score(values_by_patch: dict, top_n: int = 8) -> list:
    """Descriptive top-N patches by score, for visually comparing against
    a manual dominant-region annotation -- NOT an active-region decision
    (top_n is a reporting choice, not a classification cutoff)."""
    return sorted(values_by_patch.items(), key=lambda kv: kv[1], reverse=True)[:top_n]


# --------------------------------------------------------------------------
# global_synchronization candidates
# --------------------------------------------------------------------------


def pairwise_zero_lag_correlation(waveforms_by_patch: dict) -> dict:
    """Mean pairwise Pearson correlation (zero lag) across every valid
    patch pair -- unlike cross_patch_synchronization in spatial_patches.py
    (which only ever receives an already-selected "active" subset), this
    is meant to be called on every patch the analysis region produced,
    so a weak-but-globally-synchronized organoid with no standout-
    amplitude patch is not excluded from the comparison before it starts.
    """
    ids = list(waveforms_by_patch)
    n = len(ids)
    if n < 2:
        return {"mean_pairwise_correlation": None, "n_patches": n, "n_pairs": 0}
    correlations = []
    for i in range(n):
        a = _fill_nan_linear(waveforms_by_patch[ids[i]])
        if np.std(a) < _MIN_MEANINGFUL_STD:
            continue
        for j in range(i + 1, n):
            b = _fill_nan_linear(waveforms_by_patch[ids[j]])
            if np.std(b) < _MIN_MEANINGFUL_STD:
                continue
            correlations.append(float(np.corrcoef(a, b)[0, 1]))
    if not correlations:
        return {"mean_pairwise_correlation": None, "n_patches": n, "n_pairs": 0}
    return {
        "mean_pairwise_correlation": float(np.mean(correlations)),
        "n_patches": n,
        "n_pairs": len(correlations),
    }


def _best_lag_abs_correlation(a: np.ndarray, b: np.ndarray, max_lag_samples: int) -> tuple[float, int]:
    """Best (highest |correlation|) alignment of `a` against `b`, allowing
    a shift of up to +/- max_lag_samples. Shared by
    pairwise_lagged_correlation and the coherent-cluster significance test
    below (patch_consensus_coherence)."""
    best_corr, best_abs, best_lag = 0.0, -1.0, 0
    for lag in range(-max_lag_samples, max_lag_samples + 1):
        if lag >= 0:
            x, y = a[: len(a) - lag or None], b[lag:]
        else:
            x, y = a[-lag:], b[: len(b) + lag or None]
        if len(x) < 3 or np.std(x) < _MIN_MEANINGFUL_STD or np.std(y) < _MIN_MEANINGFUL_STD:
            continue
        corr = float(np.corrcoef(x, y)[0, 1])
        if abs(corr) > best_abs:
            best_corr, best_abs, best_lag = corr, abs(corr), lag
    return best_corr, best_lag


def pairwise_lagged_correlation(waveforms_by_patch: dict, max_lag_samples: int = 3) -> dict:
    """Mean pairwise correlation allowing each pair to be shifted by up to
    +/- `max_lag_samples` before correlating, keeping the best (highest
    absolute) alignment -- a small, fixed tolerance (not fit per video),
    meant to capture near-simultaneous timing that isn't frame-exact (e.g.
    a contraction wave reaching one patch a sample or two after another),
    which a zero-lag-only correlation would understate. Also reports the
    mean magnitude of the lag that gave the best correlation: a value near
    zero across the whole video is itself evidence of true near-
    simultaneous (not just eventually-correlated) motion.
    """
    ids = list(waveforms_by_patch)
    n = len(ids)
    if n < 2:
        return {"mean_best_lag_correlation": None, "mean_abs_best_lag_samples": None, "n_patches": n}
    filled = {pid: _fill_nan_linear(wf) for pid, wf in waveforms_by_patch.items()}
    best_corrs, best_lags = [], []
    for i in range(n):
        a = filled[ids[i]]
        if np.std(a) < _MIN_MEANINGFUL_STD:
            continue
        for j in range(i + 1, n):
            b = filled[ids[j]]
            if np.std(b) < _MIN_MEANINGFUL_STD:
                continue
            best_corr, best_lag = _best_lag_abs_correlation(a, b, max_lag_samples)
            best_corrs.append(best_corr)
            best_lags.append(abs(best_lag))
    if not best_corrs:
        return {"mean_best_lag_correlation": None, "mean_abs_best_lag_samples": None, "n_patches": n}
    return {
        "mean_best_lag_correlation": float(np.mean(best_corrs)),
        "mean_abs_best_lag_samples": float(np.mean(best_lags)),
        "n_patches": n,
        "max_lag_samples": max_lag_samples,
    }


def _welch_cross_spectrum(a: np.ndarray, b: np.ndarray, n_segments: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Segment-averaged (Welch-style) auto/cross power spectra, Hann-
    tapered, 50%-overlap segments. Averaging across segments is required
    for a meaningful coherence estimate -- a single-segment FFT gives
    coherence == 1 everywhere by construction (no averaging)."""
    n = len(a)
    seg_len = max(8, int(round(2 * n / (n_segments + 1))))
    step = max(1, seg_len // 2)
    window = np.hanning(seg_len)
    pxx = pyy = pxy = None
    n_used = 0
    start = 0
    while start + seg_len <= n:
        sa = (a[start : start + seg_len] - a[start : start + seg_len].mean()) * window
        sb = (b[start : start + seg_len] - b[start : start + seg_len].mean()) * window
        fa = np.fft.rfft(sa)
        fb = np.fft.rfft(sb)
        if pxx is None:
            pxx, pyy, pxy = np.abs(fa) ** 2, np.abs(fb) ** 2, fa * np.conj(fb)
        else:
            pxx += np.abs(fa) ** 2
            pyy += np.abs(fb) ** 2
            pxy += fa * np.conj(fb)
        n_used += 1
        start += step
    if n_used == 0:
        return np.array([]), np.array([]), np.array([])
    return pxx / n_used, pyy / n_used, pxy / n_used


def pairwise_spectral_coherence(waveforms_by_patch: dict, n_segments: int = 6) -> dict:
    """Mean, across all valid patch pairs, of the peak (over frequency)
    magnitude-squared Welch coherence. Peak-over-frequency is used
    (rather than coherence at one assumed beat frequency) because no
    project-wide beat-frequency band is fixed yet (see
    configs/default.yaml: no BPM range is configured) -- this keeps the
    metric threshold-free with respect to frequency as well as amplitude.
    """
    ids = list(waveforms_by_patch)
    n = len(ids)
    if n < 2:
        return {"mean_peak_coherence": None, "n_patches": n, "n_pairs": 0}
    filled = {pid: _fill_nan_linear(wf) for pid, wf in waveforms_by_patch.items()}
    peak_cohs = []
    for i in range(n):
        a = filled[ids[i]]
        if np.std(a) < _MIN_MEANINGFUL_STD:
            continue
        for j in range(i + 1, n):
            b = filled[ids[j]]
            if np.std(b) < _MIN_MEANINGFUL_STD:
                continue
            pxx, pyy, pxy = _welch_cross_spectrum(a, b, n_segments)
            if pxx.size == 0:
                continue
            denom = pxx * pyy
            valid = denom > 0
            if not valid.any():
                continue
            coherence = np.zeros_like(denom)
            coherence[valid] = (np.abs(pxy[valid]) ** 2) / denom[valid]
            peak_cohs.append(float(coherence[1:].max()) if coherence.size > 1 else float(coherence.max()))
    if not peak_cohs:
        return {"mean_peak_coherence": None, "n_patches": n, "n_pairs": 0}
    return {"mean_peak_coherence": float(np.mean(peak_cohs)), "n_patches": n, "n_pairs": len(peak_cohs)}


def population_reference_frequency(waveforms_by_patch: dict, sample_interval_s: float) -> float | None:
    """Data-derived shared oscillation frequency: sum the (mean-
    subtracted, Hann-windowed) power spectrum of every valid patch and
    return the non-DC frequency bin with the largest *pooled* power. Used
    only to define one shared frequency band for phase extraction below
    -- not a claim about BPM, and not an assumed/fixed Hz value."""
    ids = list(waveforms_by_patch)
    if not ids:
        return None
    total_power = None
    freqs = None
    for pid in ids:
        x = _fill_nan_linear(waveforms_by_patch[pid])
        n = len(x)
        if n < 4:
            continue
        xw = (x - x.mean()) * np.hanning(n)
        spectrum = np.fft.rfft(xw)
        power = np.abs(spectrum) ** 2
        f = np.fft.rfftfreq(n, d=sample_interval_s)
        if total_power is None:
            total_power, freqs = power.copy(), f
        elif len(power) == len(total_power):
            total_power += power
    if total_power is None or len(total_power) < 3:
        return None
    peak_idx = int(np.argmax(total_power[1:])) + 1
    return float(freqs[peak_idx])


def kuramoto_phase_consistency(
    waveforms_by_patch: dict, sample_interval_s: float, bandwidth_fraction: float = 0.5
) -> dict:
    """Kuramoto phase-locking order parameter r(t) = |mean_i(exp(i *
    theta_i(t)))| across every valid patch i, averaged over time.
    r == 1 means every patch's waveform is perfectly in phase at that
    instant (regardless of amplitude); r == 0 means phases are
    uniformly scattered -- the property this project needs to tell a
    weak-but-globally-synchronized organoid (small amplitude, but high
    mean r) apart from unsynchronized noise (small amplitude AND low
    mean r).

    Each patch's instantaneous phase comes from the analytic signal
    (FFT-domain Hilbert transform) of its waveform band-passed around one
    shared `population_reference_frequency` (+/- bandwidth_fraction of
    that frequency, or +/- 2 FFT bins, whichever is wider, so the band
    never collapses to nothing at low reference frequencies). If no
    reference frequency can be estimated, returns None fields rather than
    guessing one.
    """
    ids = list(waveforms_by_patch)
    n = len(ids)
    if n < 2:
        return {"reference_frequency_hz": None, "order_parameter_mean": None, "order_parameter_median": None, "n_patches": n}
    ref_freq = population_reference_frequency(waveforms_by_patch, sample_interval_s)
    if ref_freq is None or ref_freq <= 0:
        return {"reference_frequency_hz": None, "order_parameter_mean": None, "order_parameter_median": None, "n_patches": n}

    phases = []
    for pid in ids:
        x = _fill_nan_linear(waveforms_by_patch[pid])
        n_samples = len(x)
        if n_samples < 8:
            continue
        freqs = np.fft.rfftfreq(n_samples, d=sample_interval_s)
        bin_width = freqs[1] - freqs[0] if len(freqs) > 1 else ref_freq
        half_band = max(bandwidth_fraction * ref_freq, 2 * bin_width)
        spectrum = np.fft.rfft(x - x.mean())
        band = (freqs >= max(0.0, ref_freq - half_band)) & (freqs <= ref_freq + half_band)
        if not band.any():
            continue
        filtered_spectrum = np.where(band, spectrum, 0)
        # Analytic signal of the band-passed waveform: zero negative
        # frequencies, double positive (non-DC, non-Nyquist) frequencies,
        # inverse FFT -- standard FFT-domain Hilbert transform.
        analytic_spectrum = np.zeros(n_samples, dtype=complex)
        analytic_spectrum[: len(filtered_spectrum)] = filtered_spectrum
        analytic_spectrum[1 : len(filtered_spectrum) - (1 if n_samples % 2 == 0 else 0)] *= 2
        analytic = np.fft.ifft(analytic_spectrum)
        if np.allclose(analytic, 0):
            continue
        phases.append(np.angle(analytic))
    if len(phases) < 2:
        return {"reference_frequency_hz": ref_freq, "order_parameter_mean": None, "order_parameter_median": None, "n_patches": n}

    phase_matrix = np.array(phases)  # (n_valid_patches, n_time)
    order_param_t = np.abs(np.mean(np.exp(1j * phase_matrix), axis=0))
    return {
        "reference_frequency_hz": ref_freq,
        "order_parameter_mean": float(np.mean(order_param_t)),
        "order_parameter_median": float(np.median(order_param_t)),
        "order_parameter_timeseries": order_param_t.tolist(),
        "n_patches": len(phases),
    }


def pca_first_component_variance_ratio(waveforms_by_patch: dict) -> dict:
    """Fraction of total variance explained by the first principal
    component of every valid patch's waveform, after z-scoring each
    patch's own waveform (zero mean, unit variance) so a shared *timing*
    pattern can be detected even when patches differ hugely in amplitude
    (without z-scoring, a single very-high-amplitude patch would dominate
    the variance and inflate this ratio regardless of whether other
    patches share its timing). High ratio: one shared temporal pattern
    dominates across the organoid (consistent with global
    synchronization); low ratio: patches move along largely independent
    temporal patterns.
    """
    ids = list(waveforms_by_patch)
    n = len(ids)
    if n < 2:
        return {"explained_variance_ratio_pc1": None, "n_patches": n}
    columns = []
    for pid in ids:
        x = _fill_nan_linear(waveforms_by_patch[pid])
        std = np.std(x)
        if std < _MIN_MEANINGFUL_STD:
            continue
        columns.append((x - x.mean()) / std)
    if len(columns) < 2:
        return {"explained_variance_ratio_pc1": None, "n_patches": n}
    matrix = np.column_stack(columns)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    total_variance = float(np.sum(singular_values**2))
    if total_variance == 0:
        return {"explained_variance_ratio_pc1": None, "n_patches": len(columns)}
    return {
        "explained_variance_ratio_pc1": float(singular_values[0] ** 2 / total_variance),
        "n_patches": len(columns),
    }


def global_synchronization_candidates(waveforms_by_patch: dict, sample_interval_s: float) -> dict:
    """Bundle of all five global_synchronization candidates, computed on
    the SAME set of valid patches (every patch the analysis region
    produced, not an already-selected "active" subset). Deliberately not
    combined into one score -- see module docstring.

    Round-11 finding: averaged unweighted over the WHOLE patch population,
    every one of these five came out low and nearly indistinguishable
    across a genuinely localized, a genuinely globally-synchronized-with-
    dominant-region, and a genuinely weak-diffuse-synchronized video --
    because most grid patches in any of these videos are low-SNR/non-
    participating tissue or periphery, which dilutes a real but partial
    synchrony signal when every pair counts equally. See
    coherent_cluster_analysis below, which is the redesign this motivated:
    instead of one population-wide average, it finds WHICH patches are
    significantly aligned with the organoid's own shared temporal
    component (amplitude-independent) and reports cluster-level
    statistics on that subset -- the actual axis this project should read
    for global_synchronization / spatial_extent going forward. These five
    whole-population candidates are kept only as a documented comparison
    baseline explaining why that redesign was needed.
    """
    return {
        "pairwise_zero_lag_correlation": pairwise_zero_lag_correlation(waveforms_by_patch),
        "pairwise_lagged_correlation": pairwise_lagged_correlation(waveforms_by_patch),
        "pairwise_spectral_coherence": pairwise_spectral_coherence(waveforms_by_patch),
        "kuramoto_phase_consistency": kuramoto_phase_consistency(waveforms_by_patch, sample_interval_s),
        "pca_first_component": pca_first_component_variance_ratio(waveforms_by_patch),
    }


# --------------------------------------------------------------------------
# Coherent-cluster redesign (round 11): replaces the whole-population
# average above as the actual global_synchronization / spatial_extent
# axes. Finds which patches are significantly aligned with the organoid's
# own shared temporal component, amplitude-independent, then reports
# cluster-level statistics on spatially-contiguous groups of those patches.
# --------------------------------------------------------------------------


def consensus_waveform(waveforms_by_patch: dict) -> np.ndarray | None:
    """The organoid's shared "coherent beating component": the first
    principal component's time-course (leading left singular vector) of
    every valid patch's z-scored waveform. Amplitude-invariant by
    construction (each patch contributes a unit-variance column), so a
    small number of genuinely synchronized patches can still dominate this
    component even among many independent/noisy ones -- PCA finds the
    dominant *shared* linear structure, which is exactly what a small
    correlated block looks like among otherwise-uncorrelated columns.
    Returns None if fewer than 2 patches have non-zero variance.
    """
    columns = []
    for pid in waveforms_by_patch:
        x = _fill_nan_linear(waveforms_by_patch[pid])
        std = np.std(x)
        if std < _MIN_MEANINGFUL_STD:
            continue
        columns.append((x - x.mean()) / std)
    if len(columns) < 2:
        return None
    matrix = np.column_stack(columns)
    u, _s, _vt = np.linalg.svd(matrix, full_matrices=False)
    return u[:, 0]


def patch_consensus_alignment(waveform: np.ndarray, consensus: np.ndarray, max_lag_samples: int = 3) -> dict:
    """Signed, lag-tolerant correlation of one patch's z-scored waveform
    against `consensus` -- the raw per-patch statistic
    coherent_cluster_analysis turns into a population-relative robust
    z-score (see that function's docstring for why a shuffle/phase-
    randomization null is NOT used here, unlike periodicity_shuffle_test).
    """
    x = _fill_nan_linear(waveform)
    std = np.std(x)
    if std < _MIN_MEANINGFUL_STD or len(x) != len(consensus):
        return {"correlation": None, "best_lag_samples": None}
    xz = (x - x.mean()) / std
    corr, lag = _best_lag_abs_correlation(xz, consensus, max_lag_samples)
    return {"correlation": corr, "best_lag_samples": lag}


def _internal_correlation_stats(cluster_ids: set, waveforms_by_patch: dict, max_lag_samples: int) -> dict:
    if len(cluster_ids) < 2:
        return {"mean": None, "median": None}
    corrs = []
    cluster_list = sorted(cluster_ids)
    for i in range(len(cluster_list)):
        a = _fill_nan_linear(waveforms_by_patch[cluster_list[i]])
        if np.std(a) < _MIN_MEANINGFUL_STD:
            continue
        for j in range(i + 1, len(cluster_list)):
            b = _fill_nan_linear(waveforms_by_patch[cluster_list[j]])
            if np.std(b) < _MIN_MEANINGFUL_STD:
                continue
            corr, _ = _best_lag_abs_correlation(a, b, max_lag_samples)
            corrs.append(corr)
    if not corrs:
        return {"mean": None, "median": None}
    return {"mean": float(np.mean(corrs)), "median": float(np.median(corrs))}


def coherent_cluster_analysis(
    waveforms_by_patch: dict,
    z_threshold: float = 3.0,
    max_lag_samples: int = 3,
) -> dict:
    """Find the organoid's temporally-coherent beating region(s) WITHOUT
    gating on amplitude. Every valid patch's alignment with the shared
    consensus_waveform is reduced to one correlation value
    (patch_consensus_alignment, amplitude-independent -- both statistic
    and consensus are z-scored); a patch counts as "coherent" if that
    value is a robust population-relative outlier (>= z_threshold,
    reusing configs/default.yaml's motion.pipeline.active_z_threshold
    value for consistency, not a new arbitrary number) among every OTHER
    patch's own alignment value. Coherent patches are then grouped into
    spatially-contiguous clusters (4-connected grid, reusing
    spatial_patches.cluster_active_patches_into_regions exactly as this
    project already groups amplitude+periodicity "active" patches) -- the
    difference is only what qualifies a patch for grouping (temporal
    alignment with the whole-organoid consensus, not its own raw
    amplitude), so a low-amplitude-but-synchronized organoid is not
    excluded before this even starts.

    Two design decisions, both confirmed necessary by sanity checks on
    synthetic data (see tests/test_phenotype_axes.py) before being used on
    real data:

    1. LEAVE-ONE-OUT consensus: each patch is compared against a consensus
       built from every OTHER valid patch, never including the patch
       under test. Testing a patch against a consensus its own data
       helped build is circular and inflates the observed statistic --
       confirmed on a pure-noise sanity check: with a naive same-data
       consensus, an all-independent-noise grid falsely produced a large
       "coherent cluster".

    2. Population-relative robust z-score, NOT a phase-randomization
       shuffle null (unlike periodicity_shuffle_test, which tests a
       different question -- a waveform's OWN periodicity, not its
       alignment to an external reference). A shuffle null is unsound
       here: for a waveform close to a single dominant frequency (true of
       most patches, per spectral_concentration), correlation with a
       same-frequency external reference is cos(phase difference), whose
       expected |value| stays substantial (~2/pi) even for a uniformly
       random phase -- confirmed empirically (the shuffle null's mean
       |correlation| came out nearly as high as the observed value,
       making the test unable to reject almost anything). The robust
       z-score instead reuses this project's own established convention
       (amplitude_z_vs_background_null, persistence_z_vs_video_population,
       neighbor_coherence_z_vs_video_population all already compare one
       patch's value against a population), using median/MAD rather than
       mean/std so the very cluster being detected contaminates the
       "background" population less.

    This is the actual `global_synchronization` axis (via
    largest_cluster_size / cluster_internal_*_correlation /
    n_coherent_patches) and `spatial_extent` axis (via
    coherent_patch_fraction, the fraction of ALL valid patches -- not just
    the single largest cluster -- found coherent) going forward, replacing
    the whole-population-average candidates in
    global_synchronization_candidates and the raw-amplitude
    participation_ratio respectively.

    Caveat: population-relative detection assumes the coherent region is a
    MINORITY of all valid patches (true so far on this project's sample
    videos, where even "globally synchronized" ones show real motion in
    only a fraction of the coarse grid -- much of the grid sits over
    non-contractile margin/background). If nearly the whole grid were
    truly coherent, the "population" itself would already reflect that and
    this method would under-detect -- flagged here, not fixed, since no
    sample video in this pilot set exercises that case.

    `coherent_patch_fraction`/cluster sizes are simple patch-COUNT
    fractions, not area-weighted (patch cell geometry isn't persisted
    alongside the waveforms) -- an approximation, documented as such;
    reasonable since one video's grid cells are close to uniform size.
    """
    ids = list(waveforms_by_patch)
    n_total = len(ids)
    empty_result = {
        "n_total_patches": n_total, "n_coherent_patches": 0, "coherent_patch_ids": [],
        "coherent_patch_fraction": 0.0, "n_clusters": 0, "n_single_patch_clusters": 0,
        "largest_cluster_patch_ids": [], "largest_cluster_size": 0,
        "largest_cluster_fraction_of_all_patches": 0.0,
        "largest_cluster_fraction_of_coherent_patches": None,
        "cluster_internal_mean_correlation": None, "cluster_internal_median_correlation": None,
        "z_threshold": z_threshold, "per_patch_alignment": {},
    }
    if n_total < 3 or consensus_waveform(waveforms_by_patch) is None:
        return empty_result

    alignment = {}
    for pid in ids:
        leave_one_out_consensus = consensus_waveform({p: wf for p, wf in waveforms_by_patch.items() if p != pid})
        if leave_one_out_consensus is None:
            alignment[pid] = None
            continue
        result = patch_consensus_alignment(waveforms_by_patch[pid], leave_one_out_consensus, max_lag_samples)
        alignment[pid] = result["correlation"]

    valid_alignment = {pid: v for pid, v in alignment.items() if v is not None}
    if len(valid_alignment) < 3:
        return empty_result

    abs_values = np.array([abs(v) for v in valid_alignment.values()])
    median = float(np.median(abs_values))
    mad = float(np.median(np.abs(abs_values - median)))
    robust_std = 1.4826 * mad if mad > 0 else float(np.std(abs_values))

    per_patch = {}
    for pid, v in valid_alignment.items():
        z = (abs(v) - median) / robust_std if robust_std > 0 else None
        per_patch[pid] = {"correlation_to_leave_one_out_consensus": v, "population_z_score": z}

    coherent_ids = {
        pid for pid, rec in per_patch.items()
        if rec["population_z_score"] is not None and rec["population_z_score"] >= z_threshold
    }
    regions = cluster_active_patches_into_regions(coherent_ids)
    regions_sorted = sorted(regions, key=lambda r: len(r.patch_ids), reverse=True)
    largest = regions_sorted[0] if regions_sorted else None
    largest_ids = set(largest.patch_ids) if largest else set()
    internal_stats = _internal_correlation_stats(largest_ids, waveforms_by_patch, max_lag_samples)

    return {
        "n_total_patches": n_total,
        "n_coherent_patches": len(coherent_ids),
        "coherent_patch_ids": [list(p) for p in sorted(coherent_ids)],
        "coherent_patch_fraction": len(coherent_ids) / n_total,
        "n_clusters": len(regions),
        "n_single_patch_clusters": sum(1 for r in regions if len(r.patch_ids) == 1),
        "largest_cluster_patch_ids": [list(p) for p in sorted(largest_ids)],
        "largest_cluster_size": len(largest_ids),
        "largest_cluster_fraction_of_all_patches": len(largest_ids) / n_total,
        "largest_cluster_fraction_of_coherent_patches": (
            len(largest_ids) / len(coherent_ids) if coherent_ids else None
        ),
        "cluster_internal_mean_correlation": internal_stats["mean"],
        "cluster_internal_median_correlation": internal_stats["median"],
        "z_threshold": z_threshold,
        "per_patch_alignment": {f"{pid[0]}_{pid[1]}": rec for pid, rec in per_patch.items()},
    }


def regional_dominance_analysis(
    amplitude_by_patch: dict,
    coherent_ids: set,
    top_n_global: int = 8,
) -> dict:
    """Redesigned regional_dominance: not a single population-wide Gini
    number, which cannot tell a genuine dominant cluster apart from noisy
    spatial heterogeneity with no real dominant region at all (round 11
    finding: a weak, diffusely-synchronized video with no annotated
    dominant region scored the HIGHEST Gini of three test videos, purely
    from low-SNR noise variability). Reports instead:

      - whether the top-N globally-highest-amplitude patches are
        spatially contiguous (a real cluster) or scattered (isolated
        noise peaks) -- `top_n_largest_contiguous_fraction`.
      - within the coherent (temporally-significant, see
        coherent_cluster_analysis) patch set specifically: a self-
        relative median-amplitude split identifies a "stronger half" of
        the coherent region; its largest spatially-contiguous sub-cluster
        is `dominant_cluster` -- so dominance is assessed only among
        genuinely-beating tissue, not contaminated by unrelated high-
        amplitude noise elsewhere in the grid.
      - `dominant_cluster_vs_remaining_coherent_amplitude_ratio`: mean
        amplitude of that dominant sub-cluster divided by mean amplitude
        of the rest of the coherent region. Near 1: no real dominance
        (uniform coherent region -- e.g. weak diffuse global beating).
        Much greater than 1: a clear dominant sub-region within the
        synchronized tissue.
    """
    top_n = top_patches_by_score(amplitude_by_patch, top_n=top_n_global)
    top_n_ids = {pid for pid, _ in top_n}
    top_n_regions = cluster_active_patches_into_regions(top_n_ids)
    largest_top_n_region_size = max((len(r.patch_ids) for r in top_n_regions), default=0)
    top_n_result = {
        "top_n_global": top_n_global,
        "top_n_largest_contiguous_size": largest_top_n_region_size,
        "top_n_largest_contiguous_fraction": (
            largest_top_n_region_size / len(top_n_ids) if top_n_ids else None
        ),
    }

    if len(coherent_ids) < 2:
        return {
            **top_n_result,
            "dominant_cluster_patch_ids": [], "dominant_cluster_mean_amplitude": None,
            "remaining_coherent_patch_ids": [], "remaining_coherent_mean_amplitude": None,
            "dominant_cluster_vs_remaining_coherent_amplitude_ratio": None,
            "note": "fewer than 2 coherent patches -- dominance within the coherent region is undefined",
        }

    coherent_amplitudes = {pid: amplitude_by_patch[pid] for pid in coherent_ids if pid in amplitude_by_patch}
    median_amp = float(np.median(list(coherent_amplitudes.values())))
    stronger_half_ids = {pid for pid, v in coherent_amplitudes.items() if v >= median_amp}
    stronger_regions = cluster_active_patches_into_regions(stronger_half_ids)
    dominant_region = max(stronger_regions, key=lambda r: len(r.patch_ids), default=None)
    dominant_ids = set(dominant_region.patch_ids) if dominant_region else set()
    remaining_ids = coherent_ids - dominant_ids

    dominant_mean = float(np.mean([coherent_amplitudes[p] for p in dominant_ids])) if dominant_ids else None
    remaining_mean = float(np.mean([coherent_amplitudes[p] for p in remaining_ids])) if remaining_ids else None
    ratio = (
        dominant_mean / remaining_mean
        if dominant_mean is not None and remaining_mean not in (None, 0) else None
    )

    return {
        **top_n_result,
        "dominant_cluster_patch_ids": [list(p) for p in sorted(dominant_ids)],
        "dominant_cluster_mean_amplitude": dominant_mean,
        "remaining_coherent_patch_ids": [list(p) for p in sorted(remaining_ids)],
        "remaining_coherent_mean_amplitude": remaining_mean,
        "dominant_cluster_vs_remaining_coherent_amplitude_ratio": ratio,
        "note": None if remaining_ids else "dominant cluster covers the entire coherent region -- ratio undefined",
    }
