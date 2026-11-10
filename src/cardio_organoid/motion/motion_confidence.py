"""Per-patch Farneback-vector trust diagnostics (round 36, Phase 2).

Independent of `image_observability.py` (which asks whether the IMAGE has
enough texture to track at all): this module asks, given that Farneback
DID produce a vector at a patch, how much should it be trusted? A patch
can have plenty of texture yet still produce an unreliable vector (e.g.
aliased/fast motion, a moving occluder), and conversely a texture-poor
patch's vector, if it happens to look consistent, is still worth reporting
alongside its low observability rather than being silently trusted or
distrusted by texture alone -- these are two separate, cross-referenced
lines of evidence, per the project's own established "act pattern"
(multiple independent significance tests, none of them decisive alone --
see pipeline/analyze.py's module docstring).

Every function operates on already-computed per-patch time series or
frame-pair flow fields -- no video I/O here (that lives in
scripts/motion_observability_diagnostic.py, following this project's
established io-in-scripts / math-in-library convention).
"""

from __future__ import annotations

import numpy as np

from cardio_organoid.motion.diagnostics import autocorrelation
from cardio_organoid.motion.spatial_patches import neighbor_coherence, temporal_persistence_fraction, windowed_std


def forward_backward_consistency_field(flow_fwd: np.ndarray, flow_bwd: np.ndarray) -> np.ndarray:
    """Standard forward/backward optical-flow consistency check: warps
    `flow_bwd` (t+1 -> t) by `flow_fwd` (t -> t+1) and returns the
    per-pixel residual magnitude |flow_fwd(x) + warped_flow_bwd(x)| --
    should be near zero wherever the estimated correspondence is
    self-consistent. Caller reduces this per patch (see
    scripts/motion_observability_diagnostic.py, generalizing the
    per-region pattern already used in
    scripts/d10_secondary_cause_diagnostic.py's
    `_forward_backward_consistency` to the full analyzed grid).
    """
    import cv2

    h, w = flow_fwd.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = (xs + flow_fwd[..., 0]).clip(0, w - 1)
    map_y = (ys + flow_fwd[..., 1]).clip(0, h - 1)
    warped_bwd_x = cv2.remap(flow_bwd[..., 0], map_x, map_y, cv2.INTER_LINEAR)
    warped_bwd_y = cv2.remap(flow_bwd[..., 1], map_x, map_y, cv2.INTER_LINEAR)
    return np.hypot(flow_fwd[..., 0] + warped_bwd_x, flow_fwd[..., 1] + warped_bwd_y)


def temporal_flow_consistency(waveform: np.ndarray) -> float:
    """Lag-1 autocorrelation of a patch's own radial-motion waveform --
    a smoothly-varying (temporally consistent, not sample-to-sample
    erratic) signal has a high positive value here; a patch whose
    'motion' is dominated by independent per-frame noise has a value near
    zero (or negative). Reuses `motion.diagnostics.autocorrelation`
    unchanged."""
    ac = autocorrelation(waveform, max_lag=1)
    return float(ac[1]) if len(ac) > 1 else 0.0


def direction_stability(radial_waveform: np.ndarray, tangential_waveform: np.ndarray) -> float:
    """Circular consistency (mean resultant length, in [0, 1]) of the
    flow-angle sequence formed by treating (radial, tangential) as a 2D
    vector at each sample. 1.0 = the flow direction (in the radial/
    tangential frame) barely varies over time; 0.0 = directions are
    uniformly scattered (as consistent with noise as with genuine
    oscillation reversing direction every sample -- this is a stability
    measure, not a periodicity measure, and is deliberately reported
    alongside, never combined with, `periodic_motion_strength`)."""
    radial = np.asarray(radial_waveform, dtype=float)
    tangential = np.asarray(tangential_waveform, dtype=float)
    valid = np.isfinite(radial) & np.isfinite(tangential)
    if valid.sum() < 2:
        return 0.0
    angles = np.arctan2(tangential[valid], radial[valid])
    mean_resultant = np.hypot(np.mean(np.cos(angles)), np.mean(np.sin(angles)))
    return float(mean_resultant)


def motion_persistence_from_waveform(waveform: np.ndarray, n_windows: int = 10, ratio_threshold: float = 1.5) -> dict:
    """Thin wrapper bundling `spatial_patches.windowed_std` +
    `temporal_persistence_fraction` (both UNCHANGED, imported) into one
    call -- see those functions' own docstrings for the persistence
    definition."""
    window_stds = windowed_std(waveform, n_windows)
    return {
        "window_stds": window_stds.tolist(),
        "temporal_persistence_fraction": temporal_persistence_fraction(window_stds, ratio_threshold),
    }


def neighbor_vector_agreement(patch_waveforms_by_id: dict[tuple[int, int], np.ndarray]) -> dict[tuple[int, int], float | None]:
    """Per-patch neighbor-vector agreement over the FULL analyzed grid --
    a thin convenience wrapper that calls `spatial_patches.
    neighbor_coherence` (UNCHANGED) once per patch id, so callers don't
    have to loop over the grid themselves."""
    return {pid: neighbor_coherence(pid, patch_waveforms_by_id) for pid in patch_waveforms_by_id}
