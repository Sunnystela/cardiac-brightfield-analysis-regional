"""Provisional per-patch reliability ranking (round 36, Phase 4 + 5).

This module does the ONLY combining step in this round's diagnostic: it
builds a provisional, clearly-labeled-as-such observability composite and
flow-confidence composite (for visualization and ranking ONLY -- every
underlying raw feature from image_observability.py/motion_confidence.py/
motion_evidence.py remains separately available and is never overwritten),
classifies each patch into one of 4 self-relative observability x motion
states (Phase 4), and produces an interpretable, transparent ranked
hotspot-candidate list (Phase 5).

This module explicitly does NOT cluster patches by timing/phase -- that is
next round's job (see patch_timing_graph.py / patch_beat_consensus.py,
untouched by this round). It also NEVER uses manual ROI coordinates as an
input -- callers may label/evaluate its output against manual regions
afterward, read-only, but this module has no such coordinates in its
signature.
"""

from __future__ import annotations

import numpy as np

STATE_A_RELIABLE_ACTIVE = "A_reliable_active_candidate"
STATE_B_LIKELY_WEAK = "B_likely_genuinely_weak"
STATE_C_NOT_OBSERVABLE = "C_NOT_OBSERVABLE_do_not_interpret_as_no_motion"
STATE_D_AMBIGUOUS = "D_ambiguous_needs_independent_confirmation"


def _zscore(values: dict[tuple[int, int], float]) -> dict[tuple[int, int], float]:
    keys = list(values)
    arr = np.array([values[k] for k in keys], dtype=float)
    mean, std = arr.mean(), arr.std()
    z = (arr - mean) / std if std > 0 else np.zeros_like(arr)
    return dict(zip(keys, z.tolist()))


def provisional_observability_composite(normalized_features: dict[str, float]) -> float:
    """PROVISIONAL, visualization/ranking-only composite: the mean of the
    (already within-video z-scored, see image_observability.
    normalize_features_within_video) gradient_magnitude, edge_density,
    structure_tensor_lambda1/lambda2, corner_strength, and texture_energy
    features. intensity_std/local_contrast are deliberately excluded from
    this particular composite (a uniformly bright but flat patch can have
    high intensity_std from illumination gradient alone without having
    any trackable structure) -- both remain available as raw/normalized
    fields regardless. This is ONE reasonable choice among several,
    explicitly not validated or finalized."""
    keys = ["gradient_magnitude", "edge_density", "structure_tensor_lambda1", "structure_tensor_lambda2", "corner_strength", "texture_energy"]
    return float(np.mean([normalized_features[k] for k in keys]))


def provisional_flow_confidence_composite(
    fb_consistency_error: float | None, temporal_consistency: float | None,
    neighbor_agreement: float | None, direction_stability_value: float | None,
) -> float | None:
    """PROVISIONAL composite: mean of (-fb_consistency_error as a
    'lower error is better' term, NOT z-scored here since the caller is
    expected to have already population-normalized it -- see
    scripts/motion_observability_diagnostic.py), temporal_consistency,
    neighbor_agreement, and direction_stability. None fields are
    excluded from the mean (not treated as 0) so a single missing
    evidence line doesn't silently drag the composite down."""
    terms = [t for t in (fb_consistency_error, temporal_consistency, neighbor_agreement, direction_stability_value) if t is not None]
    return float(np.mean(terms)) if terms else None


def classify_state(motion_score: float, observability_score: float, motion_median: float, observability_median: float) -> str:
    """Self-relative median-split into the 4 Phase-4 states -- `motion_median`/
    `observability_median` MUST be computed by the caller from THIS
    video's own patch population (never a fixed/absolute/manual-derived
    cutoff)."""
    high_motion = motion_score >= motion_median
    high_observability = observability_score >= observability_median
    if high_motion and high_observability:
        return STATE_A_RELIABLE_ACTIVE
    if not high_motion and high_observability:
        return STATE_B_LIKELY_WEAK
    if not high_motion and not high_observability:
        return STATE_C_NOT_OBSERVABLE
    return STATE_D_AMBIGUOUS


def rank_hotspot_candidates(records: list[dict], low_confidence_percentile: float = 25.0) -> list[dict]:
    """Builds the final Phase-5 ranked candidate list from per-patch
    records. Each input record must have: `patch_id`, `motion_score`,
    `image_observability`, `flow_confidence` (may be None),
    `persistence`, `periodicity`, `state` (one of the 4 Phase-4 labels).

    Adds `spatial_support` (fraction of 4-connected neighbors, WITHIN
    `records`, whose own `state` is state A), `final_provisional_confidence`
    (mean of z-scored motion_score/image_observability/persistence/
    periodicity/spatial_support, plus flow_confidence when available --
    PROVISIONAL, for ranking only), and `rejection_reason`:
      - "NOT_OBSERVABLE" if state is C (self-relative low motion AND low
        observability -- never asserted as 'confirmed no motion').
      - "AMBIGUOUS" if state is D (high motion, low observability --
        needs independent confirmation, e.g. from binary movement or
        block matching, never trusted alone).
      - "LOW_CONFIDENCE" if `final_provisional_confidence` falls in the
        bottom `low_confidence_percentile` among states A/B (self-relative,
        never tuned to any manual target).
      - None otherwise.

    Returns records sorted by `final_provisional_confidence` descending
    (ties broken by patch_id for determinism).
    """
    by_id = {tuple(r["patch_id"]): r for r in records}
    reliable_ids = {pid for pid, r in by_id.items() if r["state"] == "A_reliable_active_candidate"}

    for pid, r in by_id.items():
        row, col = pid
        neighbors = [(row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)]
        analyzed_neighbors = [n for n in neighbors if n in by_id]
        r["spatial_support"] = (
            float(sum(1 for n in analyzed_neighbors if n in reliable_ids) / len(analyzed_neighbors))
            if analyzed_neighbors else 0.0
        )

    z_motion = _zscore({pid: r["motion_score"] for pid, r in by_id.items()})
    z_obs = _zscore({pid: r["image_observability"] for pid, r in by_id.items()})
    z_persist = _zscore({pid: r["persistence"] for pid, r in by_id.items()})
    z_period = _zscore({pid: (r["periodicity"] if r["periodicity"] is not None else 0.0) for pid, r in by_id.items()})
    z_support = _zscore({pid: r["spatial_support"] for pid, r in by_id.items()})
    flow_conf_values = {pid: r["flow_confidence"] for pid, r in by_id.items() if r.get("flow_confidence") is not None}
    z_flow = _zscore(flow_conf_values) if flow_conf_values else {}

    for pid, r in by_id.items():
        terms = [z_motion[pid], z_obs[pid], z_persist[pid], z_period[pid], z_support[pid]]
        if pid in z_flow:
            terms.append(z_flow[pid])
        r["final_provisional_confidence"] = float(np.mean(terms))

    ab_confidences = [r["final_provisional_confidence"] for r in by_id.values() if r["state"] in ("A_reliable_active_candidate", "B_likely_genuinely_weak")]
    low_conf_cutoff = float(np.percentile(ab_confidences, low_confidence_percentile)) if ab_confidences else None

    for pid, r in by_id.items():
        if r["state"] == STATE_C_NOT_OBSERVABLE:
            r["rejection_reason"] = "NOT_OBSERVABLE"
        elif r["state"] == STATE_D_AMBIGUOUS:
            r["rejection_reason"] = "AMBIGUOUS"
        elif low_conf_cutoff is not None and r["final_provisional_confidence"] <= low_conf_cutoff:
            r["rejection_reason"] = "LOW_CONFIDENCE"
        else:
            r["rejection_reason"] = None

    return sorted(by_id.values(), key=lambda r: (-r["final_provisional_confidence"], r["patch_id"]))
