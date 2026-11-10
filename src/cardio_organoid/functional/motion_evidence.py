"""Per-patch motion-evidence bundle, kept deliberately un-combined (round
36, Phase 3).

Farneback-derived evidence (magnitude, radial/tangential components,
temporal variance, periodicity strength) and reference-frame binary-
movement evidence (round 34) are two INDEPENDENT motion channels this
project has already built. This module bundles each patch's evidence from
BOTH, as separate, clearly-labeled fields -- never merged into one
"motion score" here (that provisional combination, if needed for
visualization, belongs in `reliable_hotspot_candidates.py`, clearly
labeled provisional).
"""

from __future__ import annotations

import numpy as np

from cardio_organoid.motion.diagnostics import (
    dominant_spectral_peak,
    periodicity_shuffle_test,
    spectral_concentration,
)


def farneback_evidence(radial_waveform: np.ndarray, tangential_waveform: np.ndarray, magnitude_waveform: np.ndarray) -> dict:
    """Separate Farneback-derived evidence fields for one patch -- no
    combination into a single score."""
    radial = np.asarray(radial_waveform, dtype=float)
    tangential = np.asarray(tangential_waveform, dtype=float)
    magnitude = np.asarray(magnitude_waveform, dtype=float)
    return {
        "farneback_magnitude_mean": float(np.nanmean(magnitude)) if magnitude.size else None,
        "radial_component_std": float(np.nanstd(radial)) if radial.size else None,
        "tangential_component_std": float(np.nanstd(tangential)) if tangential.size else None,
        "temporal_variance": float(np.nanvar(radial)) if radial.size else None,
    }


def periodic_motion_strength(waveform: np.ndarray, sample_interval_s: float, n_shuffles: int = 50, rng: np.random.Generator | None = None) -> dict:
    """Recurrent/periodic-motion-strength evidence: dominant spectral
    peak + spectral concentration (both UNCHANGED, imported) plus a
    SINGLE, CHEAPER periodicity-significance check (phase_randomized
    null only, reduced n_shuffles=50 vs pipeline/analyze.py's 200 -- a
    documented compute-budget scope decision for a per-patch, all-video
    diagnostic; this is a corroborating-evidence field, not a hard gate,
    so a coarser p-value estimate is an acceptable trade-off here)."""
    peak = dominant_spectral_peak(waveform, sample_interval_s)
    concentration = spectral_concentration(waveform, sample_interval_s)
    shuffle = periodicity_shuffle_test(
        waveform, n_shuffles=n_shuffles, rng=rng or np.random.default_rng(0), null_method="phase_randomized",
    )
    return {
        "dominant_frequency_hz": peak["dominant_frequency_hz"], "spectral_prominence": peak["prominence"],
        "spectral_concentration": concentration,
        "periodicity_p_value": shuffle["p_value"], "periodicity_z_score": shuffle["z_score"],
    }


def binary_movement_evidence(moving_fraction_series: np.ndarray) -> dict:
    """Summary of a patch's reference-frame binary-movement (round 34)
    moving-fraction time series -- kept as separate fields, distinct
    from every Farneback-derived field above; the underlying per-frame
    computation (absolute_difference/binary_mask_from_diff/
    moving_fraction) happens in the calling script, this only summarizes
    the resulting series."""
    values = np.asarray(moving_fraction_series, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"median_moving_fraction": None, "peak_moving_fraction": None, "n_samples": 0}
    return {
        "median_moving_fraction": float(np.median(values)),
        "peak_moving_fraction": float(np.max(values)),
        "n_samples": int(values.size),
    }
