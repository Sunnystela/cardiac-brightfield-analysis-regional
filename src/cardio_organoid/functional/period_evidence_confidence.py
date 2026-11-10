"""Round 41: period-evidence aggregation + PROVISIONAL confidence gating.

Round 40 found round 39's fundamental-period estimator fails in at least
two DIFFERENT ways across this project's 5 validation videos:
early/conv_a1 (all 3 candidate sources disagree wildly -- weak/absent
signal) and Beating_D9_slit_a1 (autocorrelation matches canonical but
median-IBI/spectral don't -- no single source is reliable across videos).
Per explicit project instruction this round, the fix is NOT another BPM
estimator that tries to pick the "right" answer -- it is a layer that
quantifies HOW MUCH the existing evidence sources agree, decides whether
a single BPM is even estimable at all, and reports a candidate BPM only
when the evidence actually supports one (never forcing one out of
disagreeing/weak evidence).

This module adds NO new period-estimation logic of its own -- every
period/period-family/harmonic value it consumes comes from
`fundamental_period_consensus.py` and `period_family_diagnostics.py`
(rounds 39-40, UNCHANGED). It only:
  1. Summarizes each of the 3 raw candidate SOURCES (median-IBI,
     autocorrelation, spectral) into one population-level BPM each (see
     `population_candidate_bpm`).
  2. Measures how much those 3 summarized sources agree with each other,
     harmonic-ambiguity-aware (`estimator_pairwise_agreement`).
  3. Bundles 10 evidence metrics per video (`evidence_bundle`).
  4. Classifies 6 distinct FAILURE MODES (A-F) and an overall
     ESTIMABLE / LOW_CONFIDENCE / NOT_ESTIMABLE state
     (`classify_confidence_state`) using EXPLICIT, PASSED-IN thresholds
     -- never hardcoded silently -- so a caller/report can show exactly
     which threshold and which raw value triggered each verdict.

The thresholds in `DEFAULT_PROVISIONAL_THRESHOLDS` are PROVISIONAL,
derived from inspecting this project's own 5-video evidence distribution
(see `scripts/round41_period_evidence_confidence_diagnostic.py`'s
distribution figure) -- some are simple, domain-general cutoffs (e.g. "a
majority, >50%, of patches needed harmonic correction"), others are
gap-based splits in the observed 5-video sample. NEITHER kind was chosen
to reproduce any specific video's manual/canonical answer -- manual/
canonical BPM values are never read by any function in this module. With
only 5 videos, these thresholds are explicitly NOT validated for
unseen data and must be re-derived as more videos are added.
"""

from __future__ import annotations

import numpy as np

from cardio_organoid.functional.fundamental_period_consensus import weighted_median
from cardio_organoid.functional.harmonic_aware_timing import harmonic_aware_period_similarity

# --------------------------------------------------------------------------
# Failure modes (A-F) -- distinct, non-exclusive; a video can trigger
# several at once.
# --------------------------------------------------------------------------

FAILURE_MODE_A_WEAK_SIGNAL = "A_weak_or_no_periodic_signal"
FAILURE_MODE_B_HARMONIC_AMBIGUITY = "B_harmonic_ambiguity"
FAILURE_MODE_C_ESTIMATOR_DISAGREEMENT = "C_estimator_disagreement"
FAILURE_MODE_D_FRAGMENTED_SPATIAL_RHYTHM = "D_fragmented_spatial_rhythm"
FAILURE_MODE_E_HIGH_BEAT_VARIABILITY = "E_high_beat_variability"
FAILURE_MODE_F_INSUFFICIENT_OBSERVABLE_AREA = "F_insufficient_observable_area"

# Structural safety-net reason -- NOT one of the 6 tunable failure modes
# and NOT gated by any threshold in DEFAULT_PROVISIONAL_THRESHOLDS (see
# `classify_confidence_state`'s zero-reliable-patch guard). A dataset
# with literally zero reliable patches has no evidence at all to reason
# about; this is a structural fact, never a confidence judgment call, so
# it is kept in its own field rather than mixed into the failure-mode
# list a threshold-tuning pass might later revisit.
STRUCTURAL_REASON_NO_VALID_PATCHES = "NO_VALID_PATCHES"

FAILURE_MODE_DESCRIPTIONS = {
    FAILURE_MODE_A_WEAK_SIGNAL: "candidate BPMs implausible/wildly spread (e.g. >150 BPM) -- no usable periodicity, not just disagreement about a plausible rate",
    FAILURE_MODE_B_HARMONIC_AMBIGUITY: "most patches needed a 0.5x/2x correction to agree with the population, but still resolve to ONE dominant family",
    FAILURE_MODE_C_ESTIMATOR_DISAGREEMENT: "median-IBI/autocorrelation/spectral population estimates do not harmonically agree with each other",
    FAILURE_MODE_D_FRAGMENTED_SPATIAL_RHYTHM: "no single period family dominates the reliable-patch population",
    FAILURE_MODE_E_HIGH_BEAT_VARIABILITY: "cycle-aware consensus IBI is highly irregular even though a period was estimated",
    FAILURE_MODE_F_INSUFFICIENT_OBSERVABLE_AREA: "too small a fraction of patches passed round-36 reliability gating to trust a population estimate",
}

DEFAULT_PROVISIONAL_THRESHOLDS = {
    # A: a candidate source's population BPM above this is treated as
    # physiologically implausible for this project's organoids (a
    # generic, domain-general ceiling -- not derived from any video's
    # own answer; cardiac organoid literature and this project's own 5
    # canonical BPMs are all well under 100).
    "implausible_bpm_ceiling": 150.0,
    # B: "most" patches needed harmonic correction -- >50% is a plain
    # majority, not a fitted cutoff.
    "harmonic_correction_fraction_ambiguous": 0.5,
    # C: mean pairwise harmonic-aware agreement (in [0, 1], see
    # `estimator_pairwise_agreement`) below this is treated as the 3
    # sources meaningfully disagreeing. 0.85 is a gap-based split
    # observed in this project's own 5-video sample (D10/Beating_D9_
    # conv_a1's agreement ~0.90-0.97; Beating_D9_slit_a1's ~0.55-0.75;
    # early/conv_a1's ~0.1-0.3) -- see the round-41 distribution figure.
    "estimator_agreement_disagreement_ceiling": 0.85,
    # D: largest period family below this fraction of the reliable
    # population is treated as "no dominant family" -- a plain-majority
    # (>50%) cutoff, not fitted.
    "dominant_family_fraction_fragmented_ceiling": 0.5,
    # E: cycle-aware consensus CV(IBI) above this is treated as highly
    # irregular -- reuses this project's own existing qualitative
    # language for CV (e.g. round-29's biological_beats.py reports
    # CV~0.82 as "irregular", ~0.43 as much improved); 0.5 sits between.
    "ibi_cv_high_variability_floor": 0.5,
    # F: below this fraction of ALL analyzed patches surviving round-36
    # reliability gating, the population is too small/sparse to trust --
    # a gap-based split observed in this sample (values cluster at
    # ~0.50-0.52 vs. this project's own round-36 finding that state A/C
    # each cover ~48-51% of patches on average; used only as a floor
    # sanity check here, not the primary driver of any of this sample's
    # 5 verdicts).
    "reliable_patch_fraction_floor": 0.3,
}


# --------------------------------------------------------------------------
# 1-3. Per-source population BPM + pairwise agreement.
# --------------------------------------------------------------------------


def population_candidate_bpm(
    patch_ids: list[tuple[int, int]], candidate_table: dict, confidence_by_patch: dict, source_key: str,
) -> dict:
    """Confidence-weighted median population BPM for ONE raw candidate
    source (`"median_ibi_positive_s"`, `"median_ibi_negative_s"`,
    `"spectral_period_s"`, or `"autocorrelation_period_s"` --
    `fundamental_period_consensus.patch_own_period_estimate`'s own
    candidate keys, read here unchanged). Returns {"bpm", "period_s",
    "n_patches_with_value"}; None fields if no patch has a value for this
    source."""
    values = [
        (candidate_table[p]["candidates"][source_key], confidence_by_patch.get(p, 1.0))
        for p in patch_ids if candidate_table[p]["candidates"].get(source_key) is not None
    ]
    period = weighted_median(values) if values else None
    return {"period_s": period, "bpm": (60.0 / period) if period else None, "n_patches_with_value": len(values)}


def estimator_pairwise_agreement(bpm_by_source: dict[str, float | None]) -> dict:
    """Harmonic-aware pairwise agreement (reuses `harmonic_aware_
    period_similarity` UNCHANGED, converting BPM -> period first) among
    named population-level candidate sources (e.g. "median_ibi",
    "autocorrelation", "spectral"). Returns {"pairwise": {"a__vs__b":
    {"similarity", "best_ratio"}, ...}, "mean_agreement": float or None}
    -- `mean_agreement` is the plain mean of every computable pairwise
    similarity (in [0, 1], 1 = every source harmonically agrees), None if
    fewer than 2 sources have a value."""
    names = [n for n, bpm in bpm_by_source.items() if bpm]
    pairwise = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            result = harmonic_aware_period_similarity(60.0 / bpm_by_source[a], 60.0 / bpm_by_source[b])
            pairwise[f"{a}__vs__{b}"] = result
    similarities = [r["similarity"] for r in pairwise.values() if r["similarity"] is not None]
    return {"pairwise": pairwise, "mean_agreement": float(np.mean(similarities)) if similarities else None}


# --------------------------------------------------------------------------
# 8, 10. Harmonic-correction / low-confidence-outlier fractions.
# --------------------------------------------------------------------------


def harmonic_correction_fraction(applied_scale_by_patch: dict[tuple[int, int], float]) -> float | None:
    """Fraction of patches whose `fundamental_period_consensus.resolve_
    harmonic_scales` cumulative scale differs from 1.0 (i.e. needed a
    0.5x/2x/etc. correction to align with the population). None if the
    population is empty."""
    if not applied_scale_by_patch:
        return None
    n_corrected = sum(1 for s in applied_scale_by_patch.values() if abs(s - 1.0) > 1e-6)
    return n_corrected / len(applied_scale_by_patch)


def low_confidence_outlier_fraction(confidence_by_patch: dict[tuple[int, int], float], threshold: float = 0.0) -> dict:
    """Fraction of reliable patches with confidence below `threshold`
    (default 0.0 -- round 36's `final_provisional_confidence` is a
    z-scored composite, so 0.0 is its own population's mean BY
    CONSTRUCTION, not a fitted cutoff: "below this video's own average
    reliability"). Returns {"fraction", "n_below_threshold", "n_total"}."""
    if not confidence_by_patch:
        return {"fraction": None, "n_below_threshold": 0, "n_total": 0}
    n_below = sum(1 for c in confidence_by_patch.values() if c < threshold)
    return {"fraction": n_below / len(confidence_by_patch), "n_below_threshold": n_below, "n_total": len(confidence_by_patch)}


# --------------------------------------------------------------------------
# Evidence bundle (10 metrics) + failure-mode / confidence-state rules.
# --------------------------------------------------------------------------


def evidence_bundle(
    video_key: str,
    patch_ids: list[tuple[int, int]],
    candidate_table: dict,
    confidence_by_patch: dict[tuple[int, int], float],
    applied_scale_by_patch: dict[tuple[int, int], float],
    dominant_family_fraction: float | None,
    n_reliable: int,
    n_total_patches: int,
    cycle_aware_cv: float | None,
) -> dict:
    """Bundles all 10 evidence metrics for one video. Every sub-value is
    ALSO kept (not just the summary numbers) so a caller can show its
    full derivation, never a bare pass/fail."""
    median_ibi_pos = population_candidate_bpm(patch_ids, candidate_table, confidence_by_patch, "median_ibi_positive_s")
    median_ibi_neg = population_candidate_bpm(patch_ids, candidate_table, confidence_by_patch, "median_ibi_negative_s")
    median_ibi_bpm = weighted_median([(v["bpm"], 1.0) for v in (median_ibi_pos, median_ibi_neg) if v["bpm"]])
    autocorr = population_candidate_bpm(patch_ids, candidate_table, confidence_by_patch, "autocorrelation_period_s")
    spectral = population_candidate_bpm(patch_ids, candidate_table, confidence_by_patch, "spectral_period_s")

    agreement = estimator_pairwise_agreement({"median_ibi": median_ibi_bpm, "autocorrelation": autocorr["bpm"], "spectral": spectral["bpm"]})
    harmonic_frac = harmonic_correction_fraction(applied_scale_by_patch)
    outlier = low_confidence_outlier_fraction(confidence_by_patch)

    return {
        "video": video_key,
        "1_median_ibi_bpm": median_ibi_bpm,
        "2_autocorrelation_bpm": autocorr["bpm"],
        "3_spectral_bpm": spectral["bpm"],
        "4_harmonic_relationships": agreement["pairwise"],
        "5_dominant_period_family_fraction": dominant_family_fraction,
        "6_estimator_pairwise_agreement": agreement["mean_agreement"],
        "7_ibi_cv": cycle_aware_cv,
        "8_harmonic_correction_fraction": harmonic_frac,
        "9_reliable_patch_fraction": (n_reliable / n_total_patches) if n_total_patches else None,
        "10_low_confidence_outlier_fraction": outlier["fraction"],
        "n_reliable_patches": n_reliable, "n_total_patches": n_total_patches,
    }


def classify_failure_modes(evidence: dict, thresholds: dict = DEFAULT_PROVISIONAL_THRESHOLDS) -> list[dict]:
    """Evaluates each of the 6 failure modes against `evidence` using
    `thresholds` (explicit, never hardcoded here) -- returns a list of
    {"mode", "description", "triggered", "rationale"} for EVERY mode
    (triggered or not), so a caller can show the full picture, not just
    the ones that fired."""
    findings = []
    bpms = [evidence["1_median_ibi_bpm"], evidence["2_autocorrelation_bpm"], evidence["3_spectral_bpm"]]
    implausible = [b for b in bpms if b is not None and b > thresholds["implausible_bpm_ceiling"]]
    findings.append({
        "mode": FAILURE_MODE_A_WEAK_SIGNAL, "description": FAILURE_MODE_DESCRIPTIONS[FAILURE_MODE_A_WEAK_SIGNAL],
        "triggered": len(implausible) >= 2,
        "rationale": f"{len(implausible)}/3 candidate sources exceed {thresholds['implausible_bpm_ceiling']} BPM: {implausible}",
    })

    hf = evidence["8_harmonic_correction_fraction"]
    dff = evidence["5_dominant_period_family_fraction"]
    findings.append({
        "mode": FAILURE_MODE_B_HARMONIC_AMBIGUITY, "description": FAILURE_MODE_DESCRIPTIONS[FAILURE_MODE_B_HARMONIC_AMBIGUITY],
        "triggered": (hf is not None and hf > thresholds["harmonic_correction_fraction_ambiguous"]) and (dff is not None and dff > thresholds["dominant_family_fraction_fragmented_ceiling"]),
        "rationale": f"harmonic_correction_fraction={hf}, dominant_family_fraction={dff}",
    })

    agreement = evidence["6_estimator_pairwise_agreement"]
    findings.append({
        "mode": FAILURE_MODE_C_ESTIMATOR_DISAGREEMENT, "description": FAILURE_MODE_DESCRIPTIONS[FAILURE_MODE_C_ESTIMATOR_DISAGREEMENT],
        "triggered": agreement is not None and agreement < thresholds["estimator_agreement_disagreement_ceiling"],
        "rationale": f"mean_estimator_agreement={agreement}",
    })

    findings.append({
        "mode": FAILURE_MODE_D_FRAGMENTED_SPATIAL_RHYTHM, "description": FAILURE_MODE_DESCRIPTIONS[FAILURE_MODE_D_FRAGMENTED_SPATIAL_RHYTHM],
        "triggered": dff is not None and dff <= thresholds["dominant_family_fraction_fragmented_ceiling"],
        "rationale": f"dominant_period_family_fraction={dff}",
    })

    cv = evidence["7_ibi_cv"]
    findings.append({
        "mode": FAILURE_MODE_E_HIGH_BEAT_VARIABILITY, "description": FAILURE_MODE_DESCRIPTIONS[FAILURE_MODE_E_HIGH_BEAT_VARIABILITY],
        "triggered": cv is not None and cv > thresholds["ibi_cv_high_variability_floor"],
        "rationale": f"cycle_aware_consensus_CV={cv}",
    })

    rpf = evidence["9_reliable_patch_fraction"]
    findings.append({
        "mode": FAILURE_MODE_F_INSUFFICIENT_OBSERVABLE_AREA, "description": FAILURE_MODE_DESCRIPTIONS[FAILURE_MODE_F_INSUFFICIENT_OBSERVABLE_AREA],
        "triggered": rpf is not None and rpf < thresholds["reliable_patch_fraction_floor"],
        "rationale": f"reliable_patch_fraction={rpf}",
    })
    return findings


def classify_confidence_state(evidence: dict, thresholds: dict = DEFAULT_PROVISIONAL_THRESHOLDS) -> dict:
    """Overall ESTIMABLE / LOW_CONFIDENCE / NOT_ESTIMABLE verdict, built
    ONLY from the failure-mode findings above (see `classify_failure_
    modes`) -- PROVISIONAL, explicitly not validated beyond this
    project's own 5-video sample:

      - NOT_ESTIMABLE: failure mode A (weak/implausible signal) OR C
        (estimator disagreement) triggered -- these mean the evidence
        itself does not support ANY single candidate number, so no BPM
        is reported.
      - LOW_CONFIDENCE: neither A nor C triggered, but B, D, E, or F is
        -- a candidate number exists but with a named, quantified
        caveat.
      - ESTIMABLE: no failure mode triggered.

    `candidate_bpm`/`candidate_source` are populated ONLY for ESTIMABLE/
    LOW_CONFIDENCE (never for NOT_ESTIMABLE, which reports None/NA
    explicitly rather than a number nobody should trust) -- the
    candidate is the median-IBI-based population BPM (the source found,
    across rounds 39-40, to agree with canonical most often among the 3;
    reported as a candidate for human review, not asserted as correct).

    STRUCTURAL SAFETY NET (checked BEFORE any threshold-based failure
    mode, and never itself a tunable threshold -- see
    `STRUCTURAL_REASON_NO_VALID_PATCHES`): if `evidence["n_reliable_
    patches"]` is 0 (or missing), every one of the 6 failure-mode checks
    above is structurally unable to fire, because each one only compares
    against a non-None value -- with zero reliable patches, EVERY
    population-level metric (median-IBI/autocorrelation/spectral BPM,
    estimator agreement, dominant-family fraction, IBI CV) is None, and
    "None > threshold" / "None < threshold" is always False. Found
    empirically while building the modality-aware dry-run pipeline: a
    video with a functional pass that yields 0 reliable patches would
    otherwise silently fall through to ESTIMABLE with a None candidate
    BPM -- self-contradictory output no downstream consumer should ever
    see. This check is a structural fact about the input (zero pieces of
    evidence), never a confidence judgment, so it is NOT one of the 6
    named failure modes and is reported in its own `structural_reason`
    field."""
    n_reliable = evidence.get("n_reliable_patches")
    if not n_reliable:
        return {
            "state": "NOT_ESTIMABLE",
            "candidate_bpm": None,
            "candidate_source": None,
            "triggered_failure_modes": [],
            "structural_reason": STRUCTURAL_REASON_NO_VALID_PATCHES,
            "all_failure_mode_findings": [],
        }

    findings = classify_failure_modes(evidence, thresholds)
    triggered = {f["mode"]: f for f in findings if f["triggered"]}

    if FAILURE_MODE_A_WEAK_SIGNAL in triggered or FAILURE_MODE_C_ESTIMATOR_DISAGREEMENT in triggered:
        state = "NOT_ESTIMABLE"
    elif triggered:
        state = "LOW_CONFIDENCE"
    else:
        state = "ESTIMABLE"

    candidate_bpm = evidence["1_median_ibi_bpm"] if state != "NOT_ESTIMABLE" else None
    return {
        "state": state,
        "candidate_bpm": candidate_bpm,
        "candidate_source": "median_ibi (population weighted median)" if candidate_bpm else None,
        "triggered_failure_modes": list(triggered),
        "structural_reason": None,
        "all_failure_mode_findings": findings,
    }
