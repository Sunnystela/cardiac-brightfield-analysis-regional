"""Pure helpers for the pixel-first motion screening (P36/Conv exploratory).

Everything here is deterministic numpy; no thresholds are fitted. The
parameters are the ones fixed by the owner-approved a1 D20 pilot
(scripts/pixel_first_motion_diagnostic.py) and are passed in explicitly.
"""
from __future__ import annotations

import numpy as np

INVALID_INT16 = np.int16(-32768)  # reserved code: never produced by quantize_flow (it clips to +-32767)


def quantize_flow(u: np.ndarray, scale: float) -> np.ndarray:
    """px/pair float -> int16 (round(u*scale), clipped to +-32767). NaN -> INVALID_INT16."""
    q = np.clip(np.round(np.nan_to_num(u, nan=0.0) * scale), -32767, 32767).astype(np.int16)
    q[np.isnan(u)] = INVALID_INT16
    return q


def dequantize_flow(q: np.ndarray, scale: float) -> np.ndarray:
    u = q.astype(np.float32) / scale
    u[q == INVALID_INT16] = np.nan
    return u


def exclusion_counts(region: np.ndarray, reasons: dict[str, np.ndarray]) -> dict:
    """Per-reason counts inside `region` (overlaps counted in every reason),
    plus the unique excluded count and the exclusive-only count per reason."""
    out = {"n_region_px": int(region.sum())}
    stack = np.zeros(region.shape, np.int16)
    for name, m in reasons.items():
        r = region & m
        out[f"n_{name}"] = int(r.sum())
        stack += r.astype(np.int16)
    for name, m in reasons.items():
        out[f"n_{name}_only"] = int((region & m & (stack == 1)).sum())
    out["n_excluded_unique"] = int((stack > 0).sum())
    out["n_excluded_multiple_reasons"] = int((stack > 1).sum())
    out["n_valid"] = out["n_region_px"] - out["n_excluded_unique"]
    out["measurable_fraction"] = out["n_valid"] / out["n_region_px"] if out["n_region_px"] else None
    return out


def prominent_peaks(x: np.ndarray, dt: float, k: float = 5.0, min_sep_s: float = 0.3) -> list[int]:
    """Pilot rule, unchanged: simple_peak_indices (>= min_sep_s apart) filtered to
    heights above median + k * MAD (1.4826-scaled) of the same waveform."""
    from cardio_organoid.functional.regional_lag_diagnostics import simple_peak_indices
    x = np.asarray(x, float)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    return [i for i in simple_peak_indices(x, max(1, int(round(min_sep_s / dt)))) if x[i] > med + k * mad]


def autocorr_period_s(x: np.ndarray, dt: float) -> dict:
    """Pilot rule, unchanged (first local autocorrelation maximum, motion.diagnostics),
    reported together with its autocorrelation value, plus a separately labelled
    'first local maximum with positive autocorrelation' (added 2026-09-29: the pilot
    rule can return a spurious negative-valued bump)."""
    from cardio_organoid.motion.diagnostics import autocorrelation, first_secondary_autocorrelation_peak
    x = np.asarray(x, float)
    r = first_secondary_autocorrelation_peak(x, dt)
    ac = autocorrelation(x)
    pos = None
    for lag in range(1, len(ac) - 1):
        if ac[lag] > ac[lag - 1] and ac[lag] >= ac[lag + 1] and ac[lag] > 0:
            pos = (lag * dt, float(ac[lag]))
            break
    return {"pilot_rule_period_s": r.get("lag_s"), "pilot_rule_autocorr": r.get("autocorr_value"),
            "positive_peak_period_s": pos[0] if pos else None, "positive_peak_autocorr": pos[1] if pos else None}


def best_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Lag (samples) maximizing Pearson r of a(t) vs b(t+lag), |lag| <= max_lag; returns (lag, r)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    best = (0, -np.inf)
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = a[: len(a) - lag], b[lag:]
        else:
            x, y = a[-lag:], b[: len(b) + lag]
        if len(x) < 4 or x.std() == 0 or y.std() == 0:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        if r > best[1]:
            best = (lag, r)
    return best if np.isfinite(best[1]) else (0, float("nan"))


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    """Mean over `mask` pixels only (denominator = mask count); NaN when empty. Never fills invalid with 0."""
    n = int(mask.sum())
    return float(values[mask].mean()) if n else float("nan")
