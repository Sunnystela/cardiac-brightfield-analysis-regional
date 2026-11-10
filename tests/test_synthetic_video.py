"""Full VIDEO path on a synthetic pulsing disk (plumbing + numeric sanity; NOT biological validation).

Opt-in: CARDIAC_RUN_SLOW=1 (decodes ~330 frames of 1080x1920 and runs Farneback on CPU, ~2-3 min).
"""
import json

import pytest

import adapters
from adapters.records import _si
from synthetic import make_pulsing_disk_video

pytestmark = pytest.mark.slow


@pytest.fixture
def synthetic(work_dir):
    video, mask = make_pulsing_disk_video(work_dir / "syn.avi", n_frames=330, beat_hz=1.0)
    _si().save_mask(mask, work_dir / "mask.npz")
    rec = adapters.make_record(batch="SYN", condition="Test", well="a1", day=1, modality="VIDEO", mask_path="mask.npz",
                               segmentation_qc_status="ACCEPT", mask_provenance="human_confirmed", qc_reason="synthetic",
                               selected_candidate=0)
    return work_dir, video, rec


def test_integrated_on_1hz_pulse(synthetic):
    wd, video, rec = synthetic
    row = adapters.analyze_integrated(video, rec, wd, wd / "out", render_figures=False)
    assert row["integrated_status"] == "RUN" and row["morphology_status"] == "MEASURED"
    assert row["functional_execution_status"] == "RUN"
    assert row["actual_fps"] == pytest.approx(30.0) and row["frame_step"] == 3
    assert row["functional_qc_status"] == "ESTIMABLE"
    assert row["bpm"] == pytest.approx(60.0, abs=1.0)
    assert row["median_ibi_s"] == pytest.approx(1.0, abs=0.05)
    assert row["pixel_first_status"] == "COMPUTED"
    # |u| peaks on expansion AND contraction -> repetition ~2x the beat rate: why it is never a BPM
    assert row["pf_mag_rate_per_min"] == pytest.approx(120.0, abs=5.0)


def test_pixel_first_reproducible(synthetic):
    wd, video, rec = synthetic
    a = adapters.analyze_pixel_first(video, rec, wd, wd / "run_a", render_figures=False)
    b = adapters.analyze_pixel_first(video, rec, wd, wd / "run_b", render_figures=False)
    sa = json.loads((wd / "run_a" / a["observation_id"] / "pixel_first/observation/summary.json").read_text())
    sb = json.loads((wd / "run_b" / b["observation_id"] / "pixel_first/observation/summary.json").read_text())
    assert sa == sb


def test_short_video_fails_and_is_reported(work_dir):
    video, mask = make_pulsing_disk_video(work_dir / "short.avi", n_frames=180)
    _si().save_mask(mask, work_dir / "mask.npz")
    rec = adapters.make_record(batch="SYN", condition="Test", well="a1", day=2, modality="VIDEO", mask_path="mask.npz",
                               segmentation_qc_status="ACCEPT", mask_provenance="human_confirmed", qc_reason="synthetic",
                               selected_candidate=0)
    row = adapters.analyze_legacy_rhythm(video, rec, work_dir, work_dir / "out")
    # 59 pairs < motion.pipeline periodicity_max_lag (100): the engine raises; the adapter records it
    assert row["functional_execution_status"] == "FAILED" and row["bpm"] is None
    assert (work_dir / "out" / row["observation_id"] / "legacy_rhythm" / "failed.json").exists()
