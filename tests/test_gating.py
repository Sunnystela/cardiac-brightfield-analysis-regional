"""Unapproved masks and non-ACCEPT records must never be analysed (fail closed, reason retained)."""
import numpy as np
import pytest

import adapters
from adapters.integrated import legacy_columns_pixel_first_convention
from adapters.legacy_rhythm import _gate_bpm
from adapters.records import _si


def _disk(shape=(200, 300), r=40):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return (yy - shape[0] / 2) ** 2 + (xx - shape[1] / 2) ** 2 <= r * r


def _rec(**over):
    base = dict(batch="SYN", condition="Test", well="a1", day=1, modality="IMAGE", mask_path="m.npz",
                segmentation_qc_status="ACCEPT", mask_provenance="human_confirmed", qc_reason="test", selected_candidate=0)
    base.update(over)
    return _si().SegmentationRecord(**{**base, "selector_top1_probability": None, "selector_margin": None})


def test_make_record_rejects_automatic_accept():
    with pytest.raises(ValueError, match="automatic ACCEPT"):
        adapters.make_record(batch="SYN", condition="T", well="a1", day=1, modality="IMAGE", mask_path="m.npz",
                             segmentation_qc_status="ACCEPT", mask_provenance="automatic", qc_reason="x", selected_candidate=0)


def test_make_record_rejects_mask_on_no_acceptable_candidate():
    with pytest.raises(ValueError):
        adapters.make_record(batch="SYN", condition="T", well="a1", day=1, modality="IMAGE", mask_path="m.npz",
                             segmentation_qc_status="NO_ACCEPTABLE_CANDIDATE", mask_provenance="human_confirmed",
                             qc_reason="x", selected_candidate=None)


@pytest.mark.parametrize("status", ["REVIEW_REQUIRED", "NO_ACCEPTABLE_CANDIDATE"])
def test_non_accept_records_are_not_measured(work_dir, status):
    _si().save_mask(_disk(), work_dir / "m.npz")
    rec = _rec(segmentation_qc_status=status, selected_candidate=None if status == "NO_ACCEPTABLE_CANDIDATE" else 0,
               mask_path=None if status == "NO_ACCEPTABLE_CANDIDATE" else "m.npz")
    row = adapters.analyze_morphology(rec, work_dir, None)
    assert row["morphology_status"] == f"MISSING_{status}"
    assert row["area_px"] is None and row["circularity"] is None
    assert row["qc_reason"] == "test"  # reason kept
    with pytest.raises(_si().InterfaceFailClosed):
        adapters.require_mask(rec, work_dir, "test")


def test_invalid_accept_record_fails_closed_with_reason(work_dir):
    _si().save_mask(_disk(), work_dir / "m.npz")
    row = adapters.analyze_morphology(_rec(mask_provenance="automatic"), work_dir, None)
    assert row["morphology_status"] == "FAILED_CLOSED"
    assert "automatic ACCEPT" in row["morphology_reason"]


def test_missing_mask_file_fails_closed(work_dir):
    row = adapters.analyze_morphology(_rec(mask_path="does_not_exist.npz"), work_dir, None)
    assert row["morphology_status"] == "FAILED_CLOSED"


def test_integrated_refuses_mask_frame_shape_mismatch(work_dir):
    import cv2
    cv2.imwrite(str(work_dir / "img.tif"), np.zeros((210, 300, 3), np.uint8))
    _si().save_mask(_disk(), work_dir / "m.npz")  # 200 x 300 mask vs 210 x 300 image
    row = adapters.analyze_integrated(work_dir / "img.tif", _rec(), work_dir, work_dir / "out")
    assert row["integrated_status"] == "NOT_RUN_INPUT_PROBLEM"
    assert "shape" in row["integrated_reason"]


@pytest.mark.parametrize("state,expect", [("ESTIMABLE", 55.0), ("LOW_CONFIDENCE", 55.0), ("NOT_ESTIMABLE", None), (None, None)])
def test_bpm_reported_only_for_estimable_or_low_confidence(state, expect):
    assert _gate_bpm({"functional_qc_status": state, "candidate_bpm": 55.0})["bpm"] == expect


def test_pixel_first_convention_keeps_legacy_rhythm_only_for_estimable():
    low = legacy_columns_pixel_first_convention({"functional_qc_status": "LOW_CONFIDENCE", "bpm": 50.0, "ibi_cv": 0.1, "median_ibi_s": 1.2})
    assert low == {"legacy_bpm_reported": 50.0, "legacy_bpm": None, "legacy_ibi_cv": None, "legacy_median_ibi_s": None}
    est = legacy_columns_pixel_first_convention({"functional_qc_status": "ESTIMABLE", "bpm": 50.0, "ibi_cv": 0.1, "median_ibi_s": 1.2})
    assert est["legacy_bpm"] == 50.0 and est["legacy_median_ibi_s"] == 1.2
