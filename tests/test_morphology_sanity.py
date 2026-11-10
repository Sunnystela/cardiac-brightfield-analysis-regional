"""Numerical sanity of the (unchanged) morphology definitions on synthetic masks."""
import math

import numpy as np
import pytest

import adapters
from adapters.records import _si


def _record(work_dir, mask, modality="IMAGE"):
    _si().save_mask(mask, work_dir / "m.npz")
    return adapters.make_record(batch="SYN", condition="Test", well="a1", day=1, modality=modality, mask_path="m.npz",
                                segmentation_qc_status="ACCEPT", mask_provenance="human_confirmed", qc_reason="test",
                                selected_candidate=0)


def _ellipse(shape, a, b):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return ((xx - shape[1] / 2) / a) ** 2 + ((yy - shape[0] / 2) / b) ** 2 <= 1


def test_disk(work_dir):
    mask = _ellipse((600, 600), 200, 200)
    row = adapters.analyze_morphology(_record(work_dir, mask), work_dir, None)
    assert row["morphology_status"] == "MEASURED"
    assert row["area_px"] == int(mask.sum())                                   # pixel count, not polygon area
    assert row["equivalent_diameter_px"] == pytest.approx(2 * math.sqrt(mask.sum() / math.pi))
    assert row["equivalent_diameter_px"] == pytest.approx(400, rel=0.01)
    assert 0.85 < row["circularity"] <= 1.0                                   # polygon perimeter of a digitised circle
    assert row["solidity"] == pytest.approx(1.0, abs=0.01)
    assert row["eccentricity"] < 0.1 and row["aspect_ratio"] == pytest.approx(1.0, abs=0.02)
    assert row["radial_cv"] < 0.01
    assert row["calibration_status"] == "NOT_CALIBRATED_no_source_file" and row["area_um2"] is None


def test_ellipse_aspect_ratio_and_eccentricity(work_dir):
    row = adapters.analyze_morphology(_record(work_dir, _ellipse((600, 800), 300, 150)), work_dir, None)
    assert row["aspect_ratio"] == pytest.approx(2.0, rel=0.02)
    assert row["eccentricity"] == pytest.approx(math.sqrt(1 - 0.25), rel=0.01)


def test_pixel_area_vs_contour_area_kept_distinct(work_dir):
    # A hole is excluded from area_px but not from the largest-contour polygon used by circularity/solidity.
    mask = _ellipse((600, 600), 200, 200) & ~_ellipse((600, 600), 50, 50)
    row = adapters.analyze_morphology(_record(work_dir, mask), work_dir, None)
    assert row["area_px"] == int(mask.sum())
    assert row["solidity"] == pytest.approx(1.0, abs=0.01)  # hole invisible to RETR_EXTERNAL contour


def test_video_morphology_is_uncalibrated(work_dir):
    row = adapters.analyze_morphology(_record(work_dir, _ellipse((300, 300), 80, 80), "VIDEO"), work_dir, None)
    assert row["calibration_status"] == "NOT_CALIBRATED_no_metadata_source" and row["um_per_pixel"] is None


def test_reproducible(work_dir):
    rec = _record(work_dir, _ellipse((500, 700), 220, 130))
    assert adapters.analyze_morphology(rec, work_dir, None) == adapters.analyze_morphology(rec, work_dir, None)


def test_calibration_mirror_matches_engine_function(work_dir):
    """adapters.calibration_for_image must agree with the engine's calibration_for_image_day for a sidecar
    inside the engine's REPO_ROOT (the only place the original can be called)."""
    from adapters._engine import PROJECT_ROOT, engine
    pilot = engine("p39_conv_a3_longitudinal_morphology_pilot")
    img = work_dir / "x.tiff"
    img.write_bytes(b"")
    (work_dir / "x.tiff_metadata.xml").write_text(
        '<?xml version="1.0"?><Metadata><Scaling><Items>'
        '<Distance Id="X"><Value>2.2E-07</Value></Distance><Distance Id="Y"><Value>2.2E-07</Value></Distance>'
        '</Items></Scaling></Metadata>')
    eng = pilot.calibration_for_image_day(img)
    ours = adapters.calibration_for_image(img)
    assert eng["calibration_status"] == ours["calibration_status"]
    assert eng.get("um_per_pixel_x") == ours.get("um_per_pixel_x") and eng.get("um_per_pixel_y") == ours.get("um_per_pixel_y")
    assert work_dir.is_relative_to(PROJECT_ROOT)
