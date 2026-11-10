"""Region-first beat-event decomposition and beat-episode grouping (round 19).

Manual cross-check of round-18's region-level beat-event detector
(beat_events.py) against the actual videos (see
results/beat_event_detection/ + the validation write-up) found that beat
COUNT already matches the videos reasonably well for several samples
(early conv/a1, slit/a1, slit/b1), but the *event structure* is more
complex than one waveform per video can represent: some organoids show a
local region beating with a double-peak ("doublet") shape at the same
time as a slower whole-organoid motion (early slit/a1); some show
several spatial regions beating with a consistent lag relative to each
other (mature Beating_D9/conv/a1, Beating_D9/slit/a1, D10/slit/a1); one
region can show repetitive/tremor-like local motion that looks like
"double detection" but may be real local repetitive motion. Per project
decision: DO NOT retune beat_events.py's detection thresholds to try to
collapse this into a single clean waveform. Instead, this module adds a
representation layer ON TOP of the unchanged per-region event detection:
keep every region's events independent, then describe how events from
different regions cluster in time ("beat episodes"), without deciding
what any of it means biologically.

Explicitly NOT decided here (same posture as beat_events.py):

  - which excursion direction (positive/negative radial motion) is
    contraction vs relaxation. Positive and negative events are
    reconstructed into COMPLETELY SEPARATE episode sets
    (`positive_episodes` / `negative_episodes`) -- they are never grouped
    into the same BeatEpisode, and no label in this module ever says
    "contraction" or "relaxation". A positive-episode-to-negative-episode
    timing relationship is available ONLY as an optional, clearly-labeled
    diagnostic (`positive_to_negative_episode_delay_candidates`) that
    does not assert the two form one cardiac cycle.
  - whether a "global_coordinated_candidate" / "localized_candidate" /
    "sequential_regional_candidate" label is biologically correct --
    these are provisional, threshold-based, descriptive labels only (see
    `provisional_pattern_label` on BeatEpisode), not a validated
    classifier.
  - whether a short gap between two events in the same region is a
    detector artifact or a real local doublet/compound motion --
    `flag_within_region_close_pairs` tags candidates for human review
    (`possible_doublet_or_detector_artifact`) and never removes them.
  - BPM, arrhythmia classification, or conduction velocity/propagation
    speed (regional timing is reported as "regional activation
    sequence" / "regional activation delay" only, matching
    regional_lag_diagnostics.py's convention).

Uses only the events already produced by
beat_events.detect_beat_events_both_polarities, called with the SAME,
unmodified detection parameters used in
scripts/beat_event_detection_validation.py -- this module never
re-tunes min_prominence_ratio, min_separation_period_fraction, or any
other detection threshold.
"""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np

from cardio_organoid.motion.diagnostics import dominant_spectral_peak

# --------------------------------------------------------------------------
# 1. Region-first event records -- one RegionEvent per detected candidate
#    event, independent per region and per polarity. Never collapsed into
#    a single whole-organoid event stream.
# --------------------------------------------------------------------------


@dataclasses.dataclass
class RegionEvent:
    region_id: str
    event_time: float
    original_frame: int
    amplitude: float
    # "positive" or "negative" -- a waveform-sign label ONLY, never a
    # contraction/relaxation biological label (see module docstring).
    polarity: str
    prominence: float
    # Set by flag_within_region_close_pairs; None until then, or if the
    # event's gaps to its neighbors were never flagged.
    doublet_flag: str | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def region_events_from_candidate_events(region_id: str, polarity: str, events: list[dict]) -> list[RegionEvent]:
    """Wrap the event dicts produced by
    beat_events.detect_candidate_events / detect_beat_events_both_polarities
    (called with unchanged detection parameters) into RegionEvent records,
    preserving `prominence` per event -- the field
    scripts/beat_event_detection_validation.py's saved JSON does not
    persist, because that script only keeps per-region summaries."""
    return [
        RegionEvent(
            region_id=region_id,
            event_time=e["time_sec"],
            original_frame=e["frame_index"],
            amplitude=e["local_amplitude"],
            polarity=polarity,
            prominence=e["prominence"],
        )
        for e in events
    ]


# --------------------------------------------------------------------------
# 2. Doublet / compound-local-motion QC -- flags, never removes.
# --------------------------------------------------------------------------


def flag_within_region_close_pairs(events: list[RegionEvent], doublet_interval_ratio: float) -> list[RegionEvent]:
    """Within ONE region and ONE polarity, flag event pairs whose gap is
    less than `doublet_interval_ratio` times that region's OWN mean
    consecutive-event gap (self-relative, like beat_events.py's
    prominence/separation conventions -- never an absolute time
    threshold, since different regions/videos have very different
    dominant periods).

    This does NOT decide whether a flagged pair is a detector artifact
    or a true local doublet/compound event (e.g. the "double-peak" local
    motion observed in early/slit/a1, or the tremor-like repetitive
    motion observed in some mature-stage regions) -- both events are
    kept, unmodified except for `doublet_flag`, so a human can check the
    QC waveform+event plot against the raw video.

    Returns a NEW list (input `events` is not mutated), sorted by
    event_time. Fewer than 2 events -> returned as-is (nothing to flag).
    """
    if len(events) < 2:
        return list(events)
    ordered = sorted(events, key=lambda e: e.event_time)
    gaps = [b.event_time - a.event_time for a, b in zip(ordered[:-1], ordered[1:])]
    mean_gap = float(np.mean(gaps))
    flagged = [dataclasses.replace(e) for e in ordered]
    if mean_gap <= 0:
        return flagged
    threshold = doublet_interval_ratio * mean_gap
    for i, gap in enumerate(gaps):
        if gap < threshold:
            flagged[i].doublet_flag = "possible_doublet_or_detector_artifact"
            flagged[i + 1].doublet_flag = "possible_doublet_or_detector_artifact"
    return flagged


# --------------------------------------------------------------------------
# 3. Episode-grouping tolerance -- derived from the video's OWN
#    file-recorded fps / dominant period, never a fixed cross-project
#    number of seconds.
# --------------------------------------------------------------------------


def pooled_dominant_period_s(dominant_frequencies_hz: list[float | None]) -> float | None:
    """Median of the per-region dominant periods (1/frequency) already
    estimated for this video (e.g. beat_events.region_beat_summary's
    `dominant_frequency_hz` for each analyzed region). Using the median
    across regions -- rather than any single region's estimate -- avoids
    anchoring the whole video's episode-grouping tolerance to one
    region's (possibly harmonic-confused) spectral estimate. Returns
    None if no valid positive frequency is available (caller must then
    fall back to a config-provided absolute tolerance)."""
    periods = [1.0 / f for f in dominant_frequencies_hz if f and f > 0]
    if not periods:
        return None
    return float(np.median(periods))


def suggest_episode_tolerance_s(pooled_period_s: float | None, tolerance_period_fraction: float) -> float | None:
    """Episode-grouping tolerance = `tolerance_period_fraction` of the
    video's own pooled dominant period (see pooled_dominant_period_s) --
    configurable via configs/default.yaml
    (functional.beat_episodes.episode_tolerance_period_fraction), never a
    fixed number of seconds shared across every video regardless of its
    fps or beat rate. Returns None if no pooled period is available."""
    if pooled_period_s is None or pooled_period_s <= 0:
        return None
    return tolerance_period_fraction * pooled_period_s


# --------------------------------------------------------------------------
# 4. Beat episodes -- events from DIFFERENT regions (same polarity only)
#    that cluster in time. Positive and negative events are grouped
#    completely separately; see module docstring.
# --------------------------------------------------------------------------


@dataclasses.dataclass
class BeatEpisode:
    episode_id: str
    polarity: str
    member_events: list[RegionEvent]
    participating_regions: list[str]
    first_activation_region: str
    first_activation_time: float
    last_activation_region: str
    last_activation_time: float
    activation_spread_s: float
    spatial_participation_fraction: float | None
    # Earliest event per participating region, ordered by time -- the
    # "regional activation sequence" for this episode. Never called
    # conduction velocity/propagation speed.
    regional_activation_sequence: list[dict]
    regional_activation_delays: list[dict]
    # Provisional/descriptive only -- see module docstring. One of:
    # "single_region", "localized_candidate", "global_coordinated_candidate",
    # "sequential_regional_candidate".
    provisional_pattern_label: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _provisional_pattern_label(
    n_regions_analyzed: int, n_participating: int, global_min_participation_fraction: float
) -> str:
    if n_regions_analyzed <= 1:
        return "single_region"
    if n_participating == 1:
        return "localized_candidate"
    fraction = n_participating / n_regions_analyzed
    if fraction >= global_min_participation_fraction:
        return "global_coordinated_candidate"
    return "sequential_regional_candidate"


def _build_episode(
    episode_id: str,
    polarity: str,
    member_events: list[RegionEvent],
    n_regions_analyzed: int,
    global_min_participation_fraction: float,
) -> BeatEpisode:
    member_events = sorted(member_events, key=lambda e: e.event_time)
    earliest_by_region: dict[str, RegionEvent] = {}
    for e in member_events:
        if e.region_id not in earliest_by_region or e.event_time < earliest_by_region[e.region_id].event_time:
            earliest_by_region[e.region_id] = e
    ordered_regions = sorted(earliest_by_region.items(), key=lambda kv: kv[1].event_time)
    participating_regions = [r for r, _ in ordered_regions]
    first_region, first_event = ordered_regions[0]
    last_region, last_event = ordered_regions[-1]
    regional_activation_sequence = [
        {"region_id": r, "event_time": e.event_time, "original_frame": e.original_frame}
        for r, e in ordered_regions
    ]
    regional_activation_delays = [
        {"pair": f"{a_r} -> {b_r}", "delay_s": b_e.event_time - a_e.event_time}
        for (a_r, a_e), (b_r, b_e) in zip(ordered_regions[:-1], ordered_regions[1:])
    ]
    return BeatEpisode(
        episode_id=episode_id,
        polarity=polarity,
        member_events=member_events,
        participating_regions=participating_regions,
        first_activation_region=first_region,
        first_activation_time=first_event.event_time,
        last_activation_region=last_region,
        last_activation_time=last_event.event_time,
        activation_spread_s=last_event.event_time - first_event.event_time,
        spatial_participation_fraction=(
            len(participating_regions) / n_regions_analyzed if n_regions_analyzed else None
        ),
        regional_activation_sequence=regional_activation_sequence,
        regional_activation_delays=regional_activation_delays,
        provisional_pattern_label=_provisional_pattern_label(
            n_regions_analyzed, len(participating_regions), global_min_participation_fraction
        ),
    )


def group_events_into_episodes(
    events_by_region: dict[str, list[RegionEvent]],
    tolerance_s: float,
    global_min_participation_fraction: float,
) -> list[BeatEpisode]:
    """Greedy time-chain clustering: events (pooled across ALL regions of
    ONE polarity, sorted by time) join the current episode if within
    `tolerance_s` of the PREVIOUS event added to it (chain-linking, same
    greedy-by-adjacency style as beat_events.detect_candidate_events'
    separation enforcement); otherwise a new episode starts. All input
    events must already be the same polarity -- this function does not
    check or enforce that itself (see reconstruct_polarity_episodes,
    which is the intended entry point).

    A region can contribute more than one event to the same episode
    (e.g. a local doublet within the episode's time window); episode-
    level activation ordering then uses each region's EARLIEST event in
    that episode (see _build_episode), while every raw member event
    (doublets included) is still kept in `member_events` for QC.
    """
    all_events = sorted(itertools.chain.from_iterable(events_by_region.values()), key=lambda e: e.event_time)
    if not all_events:
        return []
    n_regions_analyzed = len(events_by_region)

    groups: list[list[RegionEvent]] = [[all_events[0]]]
    for ev in all_events[1:]:
        if ev.event_time - groups[-1][-1].event_time <= tolerance_s:
            groups[-1].append(ev)
        else:
            groups.append([ev])

    polarity = all_events[0].polarity
    return [
        _build_episode(f"{polarity}_episode_{i:03d}", polarity, grp, n_regions_analyzed, global_min_participation_fraction)
        for i, grp in enumerate(groups)
    ]


def reconstruct_polarity_episodes(
    events_by_region: dict[str, list[RegionEvent]],
    tolerance_s: float,
    global_min_participation_fraction: float,
    doublet_interval_ratio: float,
) -> tuple[list[BeatEpisode], dict[str, list[RegionEvent]]]:
    """Entry point for ONE polarity: flags within-region doublet
    candidates first (never removing events), then groups the
    (doublet-flagged, but otherwise unmodified) events across regions
    into episodes. Returns (episodes, flagged_events_by_region) -- the
    latter so callers/QC plots can show every raw event including its
    doublet flag, independent of episode membership."""
    flagged_by_region = {
        region_id: flag_within_region_close_pairs(events, doublet_interval_ratio)
        for region_id, events in events_by_region.items()
    }
    episodes = group_events_into_episodes(flagged_by_region, tolerance_s, global_min_participation_fraction)
    return episodes, flagged_by_region


# --------------------------------------------------------------------------
# 5. Optional cross-polarity diagnostic -- descriptive only, never
#    asserts positive+negative form one cardiac cycle.
# --------------------------------------------------------------------------


def _episode_representative_time(episode: BeatEpisode) -> float:
    return float(np.mean([e.event_time for e in episode.member_events]))


def positive_to_negative_episode_delay_candidates(
    positive_episodes: list[BeatEpisode], negative_episodes: list[BeatEpisode], max_match_window_s: float
) -> list[dict]:
    """OPTIONAL diagnostic only: for each positive episode, the nearest
    negative episode within `max_match_window_s` (nearest-neighbor
    matching, same pattern as beat_events.regional_activation_delay).
    Reports a candidate positive-episode -> negative-episode delay --
    this is NOT a decision that the two form one contraction/relaxation
    cycle, and no BPM or cycle count is derived from it here. Empty
    input on either side returns an empty list."""
    if not positive_episodes or not negative_episodes:
        return []
    neg_times = [_episode_representative_time(ne) for ne in negative_episodes]
    candidates = []
    used = set()
    for pe in positive_episodes:
        pt = _episode_representative_time(pe)
        options = [(j, nt) for j, nt in enumerate(neg_times) if j not in used and abs(nt - pt) <= max_match_window_s]
        if not options:
            continue
        j, nt = min(options, key=lambda jt: abs(jt[1] - pt))
        used.add(j)
        candidates.append({
            "positive_episode_id": pe.episode_id,
            "negative_episode_id": negative_episodes[j].episode_id,
            "delay_s_negative_minus_positive": nt - pt,
        })
    return candidates
