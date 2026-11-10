"""Statistical-significance-based active-patch classification (Phase 2,
round 6).

Replaces `spatial_patches.classify_active_patches`'s percentile rule for
the main pipeline. That rule always selected the top
`100 - active_percentile` percent of *whatever candidate patches existed*
-- confirmed on the sample set: ~17-18 of ~68-71 patches were "active" in
every single video, which is mechanically just 100-75=25% of the
candidates, not a finding about which patches move. A percentile of a
fixed candidate set can never come back empty and can never flag "all of
them" or "none of them" -- it is not a measurement of activity, only a
guaranteed reshuffling of the same fixed count.

This module instead classifies a patch as active only if its activity
score is a statistical outlier relative to an independently measured
null/background reference (static, non-organoid regions analyzed with the
exact same pipeline and the exact same patch size -- see
scripts' use of `background_reference.select_background_patches` with
`patch_size_ratio` matched to the grid's `cell_size_ratio`). This can
legitimately select zero patches (nothing exceeds the noise floor) or
many (most of the region genuinely moves) -- the answer is not fixed in
advance by the classification rule itself.
"""

from __future__ import annotations

import numpy as np


def null_reference_stats(null_scores: list[float]) -> dict:
    """Summary statistics of an independently measured null/background
    score distribution (e.g. per-null-cell radial-waveform std over the
    full video). Raises if there are no null samples -- a noise floor
    cannot be estimated from zero reference regions.
    """
    if not null_scores:
        raise ValueError("null_scores is empty -- cannot establish a noise floor")
    arr = np.array(null_scores, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
        "values": arr.tolist(),
    }


def significance_threshold(null_scores: list[float], z_threshold: float) -> float:
    """The score value a patch must exceed to be called active:
    null_mean + z_threshold * null_std. Exposed as its own function so
    callers can log/report the exact numeric threshold used, not just the
    resulting active/inactive verdicts.
    """
    stats = null_reference_stats(null_scores)
    return stats["mean"] + z_threshold * stats["std"]


def classify_active_by_significance(
    patch_scores: dict, null_scores: list[float], z_threshold: float
) -> tuple[set, float]:
    """Return (active_ids, threshold_value). A patch is active only if its
    score exceeds `threshold_value` -- not a percentile of `patch_scores`
    itself. Can return an empty set (nothing exceeds the noise floor) or
    all of `patch_scores` (everything does); neither is prevented by the
    classification rule.
    """
    threshold = significance_threshold(null_scores, z_threshold)
    active = {pid for pid, score in patch_scores.items() if score > threshold}
    return active, threshold
