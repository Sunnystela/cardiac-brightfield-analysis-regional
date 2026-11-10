"""[added] Regional (multi-rhythm) BPM on a synthetic legacy_rhythm cache (no video, no optical flow).

Three 3x4-patch areas: A beats at 60 BPM, B at 36 BPM, C at 60 BPM but half a cycle later than A.
Expected: three groups (rate OR timing differs), A/C flagged as a merge suggestion (same rate, same
beat shape after lag alignment), and a manual A+C merge recomputed as one 60 BPM group.
"""
import json

import numpy as np
import pytest

from adapters.regional_rhythm import (RegionalRhythm, analyze_regional_rhythm, load_merge_file, parse_merges,
                                      rank_weights, row_columns)

FPS, FRAME_STEP, DURATION_S = 30.0, 3, 30.0
STATE_A = "A_reliable_active_candidate"
STATE_C = "C_NOT_OBSERVABLE_do_not_interpret_as_no_motion"


def _beat_wave(bpm, phase_s, dt, n, rng):
    t = np.arange(n) * dt
    period = 60.0 / bpm
    w = np.zeros(n)
    for t0 in np.arange(phase_s, t[-1] + period, period):
        w += 1.0 * np.exp(-0.5 * ((t - t0) / 0.06) ** 2)            # contraction (outward)
        w -= 0.8 * np.exp(-0.5 * ((t - t0 - 0.25) / 0.08) ** 2)     # relaxation (inward)
    return w * rng.uniform(0.8, 1.2) * 0.2 + rng.normal(0, 0.01, n)


def make_cache(root, areas, extra_not_observable=2, seed=0):
    rng = np.random.default_rng(seed)
    dt = FRAME_STEP / FPS
    n = int(DURATION_S / dt)
    legacy = root / "OBS_X" / "legacy_rhythm"
    motion = legacy / "motion_pipeline" / "OBS_X"
    motion.mkdir(parents=True)
    (legacy / "round36").mkdir()
    waves, per = {}, {}
    for row0, (bpm, phase) in enumerate(areas):
        for r in range(3):
            for c in range(4):
                pid = f"{row0 * 3 + r}_{c}"
                waves[pid] = _beat_wave(bpm, phase, dt, n, rng)
                per[pid] = {"state": STATE_A, "final_provisional_confidence": float(rng.normal(0, 1))}
    for k in range(extra_not_observable):
        pid = f"{len(areas) * 3 + k}_0"
        waves[pid] = rng.normal(0, 0.01, n)
        per[pid] = {"state": STATE_C, "final_provisional_confidence": -2.0}
    np.savez(motion / "patch_waveforms_radial.npz", **waves)
    (motion / "motion_pipeline_qc.json").write_text(json.dumps({"fps": FPS, "frame_step": FRAME_STEP,
                                                                 "video_duration_s": DURATION_S}))
    (legacy / "round36" / "OBS_X_round36_raw_per_patch.json").write_text(json.dumps(
        {"per_patch": per, "state_counts": {}, "manual_region_validation": {"manual_reference_available": False}}))
    return legacy


@pytest.fixture(scope="module")
def three_area(tmp_path_factory):
    legacy = make_cache(tmp_path_factory.mktemp("rr"), [(60, 0.2), (36, 0.2), (60, 0.7)])
    return RegionalRhythm(legacy)


def _area_of(pid_key):
    return int(pid_key.split("_")[0]) // 3  # 0 = A, 1 = B, 2 = C


def test_rank_weights_keep_order_and_are_positive():
    w = rank_weights({"a": -3.0, "b": 0.5, "c": 0.5, "d": 2.0}, ["a", "b", "c", "d"])
    assert w["a"] < w["b"] == w["c"] < w["d"] and min(w.values()) > 0 and w["d"] == 1.0


def test_three_areas_become_three_groups(three_area):
    s = three_area.summary()
    assert s["n_reliable"] == 36
    groups = [g for g in s["groups_final"] if g["assigned"]]
    assert len(groups) == 3, [(g["label"], g["n_patches"]) for g in s["groups_final"]]
    for g in groups:  # each group is exactly one area
        assert len({_area_of(k) for k in g["patch_ids"]}) == 1
        expected = 36 if _area_of(g["patch_ids"][0]) == 1 else 60
        assert g["bpm"] == pytest.approx(expected, rel=0.08)
    assert s["status"] == "MULTIPLE_RHYTHMS"


def test_same_rate_out_of_phase_pair_is_suggested_not_merged(three_area):
    s = three_area.summary()
    lab = {_area_of(g["patch_ids"][0]): g["label"] for g in s["groups_final"] if g["assigned"]}
    pair = next(p for p in s["pairs_final"] if {p["a"], p["b"]} == {lab[0], lab[2]})
    assert pair["merge_suggested"] and abs(pair["best_lag_s"]) == pytest.approx(0.5, abs=0.15)
    other = next(p for p in s["pairs_final"] if {p["a"], p["b"]} == {lab[0], lab[1]})
    assert not other["merge_suggested"]


def test_manual_merge_recomputes_metrics(three_area, tmp_path):
    auto = {g.patch_ids[0][0] // 3: g.label for g in three_area.auto_groups_raw() if g.assigned}
    merges = [[auto[0], auto[2]]]
    s = three_area.summary(merges)
    groups = [g for g in s["groups_final"] if g["assigned"]]
    assert len(groups) == 2
    merged = next(g for g in groups if len(g["merged_from"]) == 2)
    assert merged["n_patches"] == 24 and merged["bpm"] == pytest.approx(60, rel=0.08)
    assert s["groups_auto"] != s["groups_final"] and s["merges"] == merges
    cols = row_columns(s)
    assert cols["n_rhythm_groups"] == 2 and cols["regional_merge_applied"] is True
    with pytest.raises(ValueError):
        three_area.summary([["G1", "G99"]])


def test_parity_with_official_rule_on_whole_population(three_area):
    """group_metrics over ALL reliable patches uses the official rule -> same number as candidate_bpm."""
    from adapters._engine import engine
    pec = engine("cardio_organoid.functional.period_evidence_confidence")
    fpc = engine("cardio_organoid.functional.fundamental_period_consensus")
    rr = three_area
    pos = pec.population_candidate_bpm(rr.reliable, rr.candidate_table, rr.confidence, "median_ibi_positive_s")["bpm"]
    neg = pec.population_candidate_bpm(rr.reliable, rr.candidate_table, rr.confidence, "median_ibi_negative_s")["bpm"]
    official = fpc.weighted_median([(v, 1.0) for v in (pos, neg) if v])
    assert rr.group_metrics(rr.reliable)["bpm"] == pytest.approx(official)


def test_end_to_end_files_and_merge_file(three_area, tmp_path):
    legacy = three_area.cache["legacy_dir"]
    first = analyze_regional_rhythm(legacy, tmp_path / "out", render_figures=True)
    assert (tmp_path / "out" / "regional_rhythm.json").is_file() and (tmp_path / "out" / "regional_groups.csv").is_file()
    assert first["regional_outputs"]["figures"]["map"]["grid"] == "abstract"
    for k in ("traces", "clustering"):
        assert (tmp_path / "out" / f"regional_rhythm_{k}.png").is_file()
    labels = [g["label"] for g in json.loads((tmp_path / "out" / "regional_rhythm.json").read_text(encoding="utf-8"))["groups_auto"]
              if g["assigned"]]
    merges = parse_merges(f"{labels[0]}+{labels[1]}")
    second = analyze_regional_rhythm(legacy, tmp_path / "out2", merges, render_figures=False)
    assert load_merge_file(second["regional_outputs"]["merge_file"]) == merges
    assert second["n_rhythm_groups"] == first["n_rhythm_groups"] - 1
