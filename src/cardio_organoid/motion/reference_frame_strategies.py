"""Reference-frame construction strategies for the reference-difference
binary movement channel (round 34).

Because this project's organoids can show non-synchronous, spatially
localized regional beating (see docs/analysis_design.md, and this
project's own manual-reference "sequential_multiregional_beating"
annotations), a SINGLE resting reference frame is not assumed sufficient
-- this module provides three, so
scripts/reference_binary_motion_diagnostic.py can compare them rather than
pick one a priori:

  A. minimum-global-motion resting frame: the single video frame whose
     LOCAL (smoothed) frame-to-frame motion proxy is smallest -- a
     concrete, inspectable single image.
  B. temporal median: a per-pixel median image over a (subsampled, for
     memory) set of frames across the recording -- robust to any single
     frame's transient motion, but is a genuinely temporally-blended
     image, not a real captured frame.
  C. adaptive (exponential moving average) reference: updated every
     frame, `ref_t = alpha*frame_t + (1-alpha)*ref_{t-1}`. Explicitly
     CAUSAL: a frame's own difference must always be computed against
     `ref_{t-1}` (the state BEFORE that frame updates it) -- comparing a
     frame against a reference that already incorporates that same frame
     would partially cancel its own signal (future/self leakage). This
     module only provides the update primitive; callers are responsible
     for using the pre-update state for the current frame's diff (see
     scripts/reference_binary_motion_diagnostic.py).

     Caveat (explicitly requested): if `alpha` is too large (reference
     adapts too fast relative to the beat period), the reference itself
     starts tracking the beat and the diff against it shrinks toward
     zero -- an adaptive reference can erase the very motion it is meant
     to detect. `ema_alpha_from_time_constant` derives alpha from a time
     constant that should be set to several multiples of the organoid's
     own expected beat period, never shorter than it.
"""

from __future__ import annotations

import numpy as np


def light_moving_average(series: np.ndarray, window_samples: int) -> np.ndarray:
    """A small, centered moving average -- used only to avoid picking a
    single noisy-dip frame as the 'resting' frame in
    `select_minimum_motion_frame_index`. Same-length output (edge-padded
    via 'same' convolution mode), `window_samples <= 1` returns the input
    unchanged."""
    window_samples = max(1, int(window_samples))
    if window_samples <= 1 or len(series) < window_samples:
        return np.asarray(series, dtype=float).copy()
    kernel = np.ones(window_samples) / window_samples
    return np.convolve(np.asarray(series, dtype=float), kernel, mode="same")


def select_minimum_motion_frame_index(motion_proxy_series: np.ndarray, smoothing_window_samples: int) -> int:
    """Index of the frame with the smallest LOCAL (smoothed) motion-proxy
    value (e.g. whole-ROI mean abs frame-to-frame pixel difference,
    computed independently of Farneback -- see
    scripts/reference_binary_motion_diagnostic.py). Smoothing avoids
    selecting a single-frame noise dip as if it were a genuinely
    quiescent moment; the smoothing window should be short relative to
    the expected beat period (a few samples), not long enough to blur
    across an entire beat cycle.

    Excludes the first/last `smoothing_window_samples // 2` samples from
    candidacy: `light_moving_average`'s 'same'-mode convolution implicitly
    zero-pads past the array edges, which pulls the smoothed value DOWN
    near the boundaries regardless of the true motion there (a genuine
    edge artifact, not evidence of a quiescent moment) -- without this
    exclusion, argmin would be biased toward picking frame 0 or the last
    frame almost independently of the recording's actual content."""
    smoothed = light_moving_average(motion_proxy_series, smoothing_window_samples)
    half_window = smoothing_window_samples // 2
    if half_window > 0 and len(smoothed) > 2 * half_window:
        candidate_range = slice(half_window, len(smoothed) - half_window)
        return half_window + int(np.argmin(smoothed[candidate_range]))
    return int(np.argmin(smoothed))


def temporal_median_reference(frames: np.ndarray) -> np.ndarray:
    """Per-pixel median image over `frames` (shape (N, H, W)), returned as
    float64. `frames` is expected to already be a (memory-bounded)
    SUBSAMPLE of the full recording, not necessarily every frame -- exact
    per-frame precision is not required for a background/reference image;
    a representative subsample is a documented, traceable trade-off (see
    the calling script's subsampling stride)."""
    arr = np.asarray(frames, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"expected frames with shape (N, H, W), got {arr.shape}")
    return np.median(arr, axis=0)


def ema_alpha_from_time_constant(tau_s: float, sample_interval_s: float) -> float:
    """Exponential-moving-average update weight alpha such that the
    reference's effective memory time constant is `tau_s` seconds, at the
    given sampling interval. `tau_s` MUST be set by the caller to several
    multiples of the organoid's own expected beat period (see module
    docstring's leakage caveat) -- this function does not choose or
    validate that relationship itself."""
    if tau_s <= 0 or sample_interval_s <= 0:
        raise ValueError("tau_s and sample_interval_s must be positive")
    return float(1.0 - np.exp(-sample_interval_s / tau_s))


def update_ema_reference(current_reference: np.ndarray, new_frame: np.ndarray, alpha: float) -> np.ndarray:
    """One EMA reference update: `alpha * new_frame + (1 - alpha) *
    current_reference`. CAUSALITY WARNING (see module docstring): the
    caller must compute `new_frame`'s own difference against
    `current_reference` (the PRE-update state) BEFORE calling this to
    obtain the POST-update state for the next frame -- never diff a frame
    against a reference this same call already folded it into."""
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")
    return alpha * np.asarray(new_frame, dtype=np.float64) + (1.0 - alpha) * np.asarray(current_reference, dtype=np.float64)
