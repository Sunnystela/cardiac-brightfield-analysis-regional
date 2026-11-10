"""PROTOTYPE v4 -- modality-aware trajectory stage diagnostic. Additive
to (does NOT replace, does NOT modify) v1/v2/v3, which remain unchanged
and available for the existing old-vs-new comparisons. NOT wired into
any production pipeline path. Diagnostic-only.

Motivation: the P36/Conv/a1 USER_QC_UPDATE round found that v3's
MATURE_RECOMPACT call at D21/D23 disappears once 5 newly-accepted IMAGE
days (D3/D4/D5/D7/D8) are added to the SAME clean-day pool v3 uses for
its rolling window and per-organoid median baseline -- IMAGE-day
solidity/circularity run systematically higher than VIDEO-day values,
so the IMAGE->VIDEO jump itself gets read as a "shedding" drop. v4
addresses this STRUCTURALLY by never letting an IMAGE observation and a
VIDEO observation share a rolling window or a baseline computation.

This module reuses (imports, does not copy or modify) the same generic,
already-validated pure helper functions v2/v3 use for slope/concordance
evidence (`_gradual_evidence`, `_slope` from prototype_trajectory_
stage_model_v2.py) -- these are unchanged, well-agnostic arithmetic
utilities, not v2's orchestration/threshold logic. All v4-specific
window sizes, confidence tiers, and recovery criteria below are NEW,
independent constants -- not copied from v2/v3, not tuned to P36/Conv/a1
(see the diagnostic script's README for the explicit warning against
over-fitting a single-organoid exploratory check).

Two independent temporal-validity views are computed for every VIDEO day
because they answer different questions and must not be conflated:
  - CAUSAL: uses only that day and days STRICTLY BEFORE it (what could
    have been said in near-real-time, without seeing the future).
  - RETROSPECTIVE: uses a centered window (before AND after) -- a
    smoother, hindsight view, only valid for describing a completed
    trajectory, never for a real-time decision.
"""

from __future__ import annotations

import dataclasses
import statistics

from cardio_organoid.pipeline.prototype_trajectory_stage_model_v2 import _gradual_evidence, _slope  # noqa: F401 (reused unchanged)

# -------------------- diagnostic-only state vocabulary --------------------
STATE_MODALITY_BRIDGE = "MODALITY_BRIDGE"                # the IMAGE->VIDEO transition point itself; never a biology call
STATE_CROSS_MODALITY_UNCERTAIN = "CROSS_MODALITY_UNCERTAIN"
STATE_BASELINE_INSUFFICIENT = "BASELINE_INSUFFICIENT"
STATE_COMPACT = "VIDEO_COMPACT"
STATE_UNCERTAIN = "VIDEO_UNCERTAIN"
STATE_TRANSITIONAL_CANDIDATE = "VIDEO_TRANSITIONAL_CANDIDATE"
STATE_TRANSIENT_SPIKE_CANDIDATE = "VIDEO_TRANSIENT_SPIKE_CANDIDATE"

RECOVERY_NOT_SUPPORTED = "NOT_SUPPORTED"
RECOVERY_TREND = "RECOVERY_TREND"
RECOVERY_PARTIAL_RECOMPACTION = "PARTIAL_RECOMPACTION"
RECOVERY_MATURE_RECOMPACT = "MATURE_RECOMPACT"

# -------------------- v4-own constants (NOT shared with v2/v3, NOT tuned to a1) --------------------
MIN_WINDOW_DAYS_FOR_ANY_CONFIDENCE = 3     # same philosophy as v3's persistence fix, independently declared here
MAX_WINDOW_DAYS = 4
MIN_CONCORDANT_METRICS_FOR_MODERATE = 2    # per explicit instruction: >=2 metrics same direction
MIN_CONCORDANT_METRICS_FOR_HIGH = 3
CUMULATIVE_DROP_MODERATE = 0.12
CUMULATIVE_DROP_HIGH = 0.22
MAX_GAP_DAYS_FOR_NEAR_CONSECUTIVE = 2       # a 1-day-missing gap (e.g. D18 excluded) still counts as "near-consecutive"
RECOVERY_FRACTION_OF_BASELINE_MEDIAN = 0.90  # solidity/circularity considered "recovered" at >=90% of video-only median
RADIAL_CV_REDUCTION_FRACTION = 0.15          # vs this metric's own post-episode peak
PERIMETER_COMPLEXITY_REDUCTION_FRACTION = 0.10
MIN_CONSECUTIVE_RECOVERY_DAYS_FOR_PARTIAL = 2
MIN_RECOVERING_METRICS_FOR_MATURE = 3        # out of 4 tracked recovery metrics
MIN_CONSECUTIVE_RECOVERY_DAYS_FOR_MATURE = 2


@dataclasses.dataclass
class ImageStreamSummary:
    n_days: int
    days: list[int]
    metric_median: dict
    metric_mad: dict
    per_day: list[dict]
    note: str


@dataclasses.dataclass
class VideoOnlyDayResult:
    day_number: int
    qc_state: str
    causal_state: str
    causal_confidence: str | None
    retrospective_state: str
    retrospective_confidence: str | None
    onset_proximity_days: int | None  # descriptive-only distance to functional onset, NEVER used to set state
    evidence: dict
    reasoning: list[str]


@dataclasses.dataclass
class TransitionEpisodeV4:
    days: list[int]
    onset_range: tuple
    peak_deformation_day: int
    duration_days: int
    confidence: str
    status: str  # "VALIDATED" or "TRANSIENT_SPIKE_CANDIDATE"
    basis: str   # "causal" or "retrospective"
    reasoning: list[str]


@dataclasses.dataclass
class RecoveryAssessmentV4:
    episode_days: list[int]
    evaluated_days: list[int]
    per_day_recovery_flags: dict  # day -> {metric: bool}
    max_consecutive_recovery_days: int
    state: str
    reasoning: list[str]


def _median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _mad(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    med = statistics.median(xs)
    return statistics.median([abs(x - med) for x in xs])


def _iqr(xs):
    xs = sorted(x for x in xs if x is not None)
    if len(xs) < 2:
        return (None, None)
    try:
        q1, q3 = statistics.quantiles(xs, n=4)[0], statistics.quantiles(xs, n=4)[2]
    except statistics.StatisticsError:
        return (min(xs), max(xs))
    return (q1, q3)


def _perim_over_sqrt_area(rec):
    return (rec["perimeter_px"] / (rec["area_px"] ** 0.5)) if rec.get("perimeter_px") and rec.get("area_px") else None


METRIC_KEYS = ["area_px", "perimeter_px", "equivalent_diameter_px", "circularity", "solidity", "radial_cv", "eccentricity", "aspect_ratio"]


def _metric_value(rec, key):
    if key == "perimeter_over_sqrt_area":
        return _perim_over_sqrt_area(rec)
    return rec.get(key)


ALL_METRIC_KEYS = METRIC_KEYS + ["perimeter_over_sqrt_area"]


def summarize_image_stream(image_records: list[dict]) -> ImageStreamSummary:
    """IMAGE-only descriptive trend -- NEVER used as a baseline or window
    input for any VIDEO-day stage decision. Purely descriptive of the
    organoid's own early (pre-video) morphology."""
    days = sorted(r["day_number"] for r in image_records)
    per_day = [
        {"day_number": r["day_number"], **{k: _metric_value(r, k) for k in ALL_METRIC_KEYS}}
        for r in sorted(image_records, key=lambda r: r["day_number"])
    ]
    metric_median = {k: _median([_metric_value(r, k) for r in image_records]) for k in ALL_METRIC_KEYS}
    metric_mad = {k: _mad([_metric_value(r, k) for r in image_records]) for k in ALL_METRIC_KEYS}
    note = (
        f"IMAGE-only trend over {len(days)} day(s) ({days}). Descriptive only -- "
        "never mixed into any VIDEO-day rolling window or baseline (see module docstring)."
    )
    return ImageStreamSummary(len(days), days, metric_median, metric_mad, per_day, note)


def _direction_flags(gradual: dict) -> dict:
    return gradual["concordant_metrics"]


def _confidence_from_gradual(gradual: dict) -> str | None:
    count = gradual["concordant_metric_count"]
    max_drop = max(gradual["cumulative_solidity_drop"], gradual["cumulative_circularity_drop"])
    if count >= MIN_CONCORDANT_METRICS_FOR_HIGH and max_drop >= CUMULATIVE_DROP_HIGH:
        return "HIGH"
    if count >= MIN_CONCORDANT_METRICS_FOR_MODERATE and max_drop >= CUMULATIVE_DROP_MODERATE:
        return "MODERATE"
    return None


def analyze_video_stream(video_records: list[dict], functional_onset_day: int | None) -> list[VideoOnlyDayResult]:
    """VIDEO-only stage evidence -- baseline/window computed EXCLUSIVELY
    from the VIDEO days passed in. `video_records` should already be
    filtered to whichever QC variant is being evaluated (e.g. with or
    without D18) by the caller; nothing here excludes days itself.
    functional_onset_day is reported as a descriptive proximity distance
    only -- it never enters any state decision below."""
    ordered = sorted(video_records, key=lambda r: r["day_number"])
    n = len(ordered)
    video_median = {k: _median([_metric_value(r, k) for r in ordered]) for k in ("solidity", "circularity", "radial_cv")}
    video_median["perimeter_over_sqrt_area"] = _median([_perim_over_sqrt_area(r) for r in ordered])

    results: list[VideoOnlyDayResult] = []
    for i, rec in enumerate(ordered):
        day = rec["day_number"]
        onset_prox = (day - functional_onset_day) if functional_onset_day is not None else None
        reasoning = []

        # --- CAUSAL: only this day and days strictly before it ---
        causal_window = ordered[max(0, i - MAX_WINDOW_DAYS + 1): i + 1]
        if len(causal_window) < MIN_WINDOW_DAYS_FOR_ANY_CONFIDENCE:
            causal_state, causal_conf = STATE_BASELINE_INSUFFICIENT, None
            reasoning.append(f"causal: only {len(causal_window)} VIDEO day(s) available up to and including D{day} "
                              f"(<{MIN_WINDOW_DAYS_FOR_ANY_CONFIDENCE} required) -- no IMAGE data substituted")
        else:
            gradual = _gradual_evidence(causal_window)
            conf = _confidence_from_gradual(gradual)
            if conf:
                causal_state, causal_conf = STATE_TRANSITIONAL_CANDIDATE, conf
                reasoning.append(f"causal: {gradual['concordant_metric_count']}/4 metrics concordant, "
                                  f"max cumulative drop={max(gradual['cumulative_solidity_drop'], gradual['cumulative_circularity_drop']):.3f} -> {conf}")
            else:
                sol, circ = rec.get("solidity"), rec.get("circularity")
                is_compact = (
                    video_median["solidity"] and sol is not None and sol >= RECOVERY_FRACTION_OF_BASELINE_MEDIAN * video_median["solidity"]
                    and video_median["circularity"] and circ is not None and circ >= RECOVERY_FRACTION_OF_BASELINE_MEDIAN * video_median["circularity"]
                )
                causal_state = STATE_COMPACT if is_compact else STATE_UNCERTAIN
                causal_conf = None
                reasoning.append("causal: no shedding-evidence bar met; compact/uncertain via video-only median comparison")

        # --- RETROSPECTIVE: centered window, before AND after ---
        half = MAX_WINDOW_DAYS // 2
        retro_window = ordered[max(0, i - half): min(n, i + half + 1)]
        if len(retro_window) < MIN_WINDOW_DAYS_FOR_ANY_CONFIDENCE:
            retro_state, retro_conf = STATE_BASELINE_INSUFFICIENT, None
        else:
            gradual_r = _gradual_evidence(sorted(retro_window, key=lambda r: r["day_number"]))
            conf_r = _confidence_from_gradual(gradual_r)
            if conf_r:
                retro_state, retro_conf = STATE_TRANSITIONAL_CANDIDATE, conf_r
            else:
                sol, circ = rec.get("solidity"), rec.get("circularity")
                is_compact = (
                    video_median["solidity"] and sol is not None and sol >= RECOVERY_FRACTION_OF_BASELINE_MEDIAN * video_median["solidity"]
                    and video_median["circularity"] and circ is not None and circ >= RECOVERY_FRACTION_OF_BASELINE_MEDIAN * video_median["circularity"]
                )
                retro_state = STATE_COMPACT if is_compact else STATE_UNCERTAIN
                retro_conf = None

        results.append(VideoOnlyDayResult(
            day_number=day, qc_state=rec.get("_qc_state", "UNKNOWN"),
            causal_state=causal_state, causal_confidence=causal_conf,
            retrospective_state=retro_state, retrospective_confidence=retro_conf,
            onset_proximity_days=onset_prox,
            evidence={"video_only_median": video_median, "n_video_days_in_pool": n},
            reasoning=reasoning,
        ))
    return results


def group_episodes_v4(day_results: list[VideoOnlyDayResult], basis: str = "retrospective") -> list[TransitionEpisodeV4]:
    """Groups consecutive-or-near-consecutive (gap <= MAX_GAP_DAYS_FOR_
    NEAR_CONSECUTIVE) TRANSITIONAL_CANDIDATE days into episodes. An
    isolated single day with no persistence is a TRANSIENT_SPIKE_
    CANDIDATE, never VALIDATED. `basis` selects which of the two state
    views (causal/retrospective) to group -- callers should report both."""
    state_attr = f"{basis}_state"
    conf_attr = f"{basis}_confidence"
    ordered = sorted(day_results, key=lambda r: r.day_number)
    candidates = [r for r in ordered if getattr(r, state_attr) == STATE_TRANSITIONAL_CANDIDATE]
    episodes: list[TransitionEpisodeV4] = []
    i = 0
    while i < len(candidates):
        group = [candidates[i]]
        j = i + 1
        while j < len(candidates) and (candidates[j].day_number - group[-1].day_number) <= MAX_GAP_DAYS_FOR_NEAR_CONSECUTIVE:
            group.append(candidates[j])
            j += 1
        days = [r.day_number for r in group]
        confidences = [getattr(r, conf_attr) for r in group]
        best_conf = "HIGH" if "HIGH" in confidences else ("MODERATE" if "MODERATE" in confidences else None)
        # peak deformation day: within the episode, the day whose own record has lowest circularity
        peak_day = min(days)  # placeholder overwritten by caller with actual metric lookup where available
        status = "VALIDATED" if len(days) >= 2 else "TRANSIENT_SPIKE_CANDIDATE"
        onset_range = (days[0], days[0]) if len(days) == 1 else (days[0], min(days[0] + 1, days[-1]))
        reasoning = [
            f"{basis} basis: grouped {len(days)} day(s) {days} with gap<= {MAX_GAP_DAYS_FOR_NEAR_CONSECUTIVE}",
            "isolated single day, not persisted -> TRANSIENT_SPIKE_CANDIDATE, not asserted as real transition" if status != "VALIDATED"
            else f">=2 (near-)consecutive candidate days -> VALIDATED episode",
        ]
        episodes.append(TransitionEpisodeV4(
            days=days, onset_range=onset_range, peak_deformation_day=peak_day, duration_days=days[-1] - days[0] + 1,
            confidence=best_conf, status=status, basis=basis, reasoning=reasoning,
        ))
        i = j
    return episodes


def refine_peak_day(episode: TransitionEpisodeV4, records_by_day: dict) -> TransitionEpisodeV4:
    circ_by_day = {d: records_by_day[d].get("circularity") for d in episode.days if records_by_day[d].get("circularity") is not None}
    if circ_by_day:
        episode.peak_deformation_day = min(circ_by_day, key=circ_by_day.get)
    return episode


def assess_recovery_v4(video_records: list[dict], episode: TransitionEpisodeV4 | None) -> RecoveryAssessmentV4:
    """Independently checks solidity recovery, circularity recovery,
    radial_cv reduction, and perimeter-complexity reduction over the
    VIDEO days AFTER a VALIDATED episode -- against the VIDEO-only
    median/peak (never an IMAGE baseline). Distinguishes RECOVERY_TREND
    (weak/short), PARTIAL_RECOMPACTION (moderate), and MATURE_RECOMPACT
    (strong, persistent, multi-metric) -- a single metric recovering is
    NEVER sufficient for MATURE_RECOMPACT."""
    ordered = sorted(video_records, key=lambda r: r["day_number"])
    if episode is None or episode.status != "VALIDATED":
        return RecoveryAssessmentV4([], [], {}, 0, RECOVERY_NOT_SUPPORTED, ["no VALIDATED episode to recover from"])

    video_median = {k: _median([_metric_value(r, k) for r in ordered]) for k in ("solidity", "circularity")}
    post_episode = [r for r in ordered if r["day_number"] > max(episode.days)]
    if not post_episode:
        return RecoveryAssessmentV4(episode.days, [], {}, 0, RECOVERY_NOT_SUPPORTED, ["no VIDEO observations after the episode"])

    # radial_cv / perimeter-complexity "post-episode peak" = their own max value at/soon after the episode
    near_episode_days = [r for r in ordered if max(episode.days) <= r["day_number"] <= max(episode.days) + 2]
    radial_cv_peak = max((r.get("radial_cv") for r in near_episode_days if r.get("radial_cv") is not None), default=None)
    perim_peak = max((_perim_over_sqrt_area(r) for r in near_episode_days if _perim_over_sqrt_area(r) is not None), default=None)

    per_day_flags = {}
    for r in post_episode:
        d = r["day_number"]
        sol, circ = r.get("solidity"), r.get("circularity")
        rcv, perim = r.get("radial_cv"), _perim_over_sqrt_area(r)
        flags = {
            "solidity_recovered": bool(video_median["solidity"] and sol is not None and sol >= RECOVERY_FRACTION_OF_BASELINE_MEDIAN * video_median["solidity"]),
            "circularity_recovered": bool(video_median["circularity"] and circ is not None and circ >= RECOVERY_FRACTION_OF_BASELINE_MEDIAN * video_median["circularity"]),
            "radial_cv_reduced": bool(radial_cv_peak and rcv is not None and rcv <= (1 - RADIAL_CV_REDUCTION_FRACTION) * radial_cv_peak),
            "perimeter_complexity_reduced": bool(perim_peak and perim is not None and perim <= (1 - PERIMETER_COMPLEXITY_REDUCTION_FRACTION) * perim_peak),
        }
        per_day_flags[d] = flags

    # longest run of consecutive (by day-order in post_episode list) days with >=2 recovering metrics
    counts = [sum(per_day_flags[r["day_number"]].values()) for r in post_episode]
    max_consecutive = 0
    run = 0
    for c in counts:
        if c >= 2:
            run += 1
            max_consecutive = max(max_consecutive, run)
        else:
            run = 0

    max_metrics_any_day = max((sum(f.values()) for f in per_day_flags.values()), default=0)
    reasoning = [f"post-episode VIDEO days evaluated: {[r['day_number'] for r in post_episode]}",
                 f"max metrics recovering on any single day: {max_metrics_any_day}/4",
                 f"longest run of days with >=2 recovering metrics: {max_consecutive}"]

    if max_metrics_any_day >= MIN_RECOVERING_METRICS_FOR_MATURE and max_consecutive >= MIN_CONSECUTIVE_RECOVERY_DAYS_FOR_MATURE:
        state = RECOVERY_MATURE_RECOMPACT
    elif max_consecutive >= MIN_CONSECUTIVE_RECOVERY_DAYS_FOR_PARTIAL and max_metrics_any_day >= 2:
        state = RECOVERY_PARTIAL_RECOMPACTION
    elif max_metrics_any_day >= 1:
        state = RECOVERY_TREND
    else:
        state = RECOVERY_NOT_SUPPORTED

    return RecoveryAssessmentV4(episode.days, [r["day_number"] for r in post_episode], per_day_flags, max_consecutive, state, reasoning)


def quantify_modality_shift(image_records: list[dict], video_records: list[dict]) -> dict:
    """Per-metric IMAGE vs VIDEO distributional comparison -- descriptive
    only, reported as 'capture-regime-associated difference', never
    asserted to be 'camera effect' or 'biology effect'."""
    out = {}
    for key in ALL_METRIC_KEYS:
        img_vals = [v for v in (_metric_value(r, key) for r in image_records) if v is not None]
        vid_vals = [v for v in (_metric_value(r, key) for r in video_records) if v is not None]
        img_med, vid_med = _median(img_vals), _median(vid_vals)
        img_mad, vid_mad = _mad(img_vals), _mad(vid_vals)
        img_iqr, vid_iqr = _iqr(img_vals), _iqr(vid_vals)
        pooled_mad = None
        effect_size = None
        if img_mad is not None and vid_mad is not None and img_med is not None and vid_med is not None:
            pooled_mad = ((img_mad ** 2 + vid_mad ** 2) / 2) ** 0.5
            effect_size = (vid_med - img_med) / pooled_mad if pooled_mad else None
        overlap = None
        if None not in img_iqr and None not in vid_iqr:
            lo = max(img_iqr[0], vid_iqr[0])
            hi = min(img_iqr[1], vid_iqr[1])
            union_lo = min(img_iqr[0], vid_iqr[0])
            union_hi = max(img_iqr[1], vid_iqr[1])
            overlap = max(0.0, hi - lo) / (union_hi - union_lo) if union_hi > union_lo else (1.0 if hi >= lo else 0.0)
        out[key] = {
            "n_image": len(img_vals), "n_video": len(vid_vals),
            "image_median": img_med, "video_median": vid_med,
            "image_mad": img_mad, "video_mad": vid_mad,
            "image_iqr": img_iqr, "video_iqr": vid_iqr,
            "robust_effect_size": effect_size, "iqr_overlap_fraction": overlap,
            "raw_value_comparison_valid": key not in ("area_px", "perimeter_px", "equivalent_diameter_px"),
        }
    return out
