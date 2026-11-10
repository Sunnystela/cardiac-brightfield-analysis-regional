"""pf_* column mapping (mirror of pixel_first_motion_aggregate.build_master) on minimal summaries."""
import pytest

import adapters


def _summary(regions):
    excl = {"n_valid": 90, "n_region_px": 100, "n_saturated": 1, "n_low_texture": 8, "n_fb_inconsistent": 2,
            "n_excluded_unique": 10, "n_excluded_multiple_reasons": 1, "measurable_fraction": 0.9}
    return {"fps": 30.0, "dt_s": 0.1, "measurable_fraction_inside_mask": 0.9,
            "exclusions": {"INSIDE_REFERENCE_MASK": excl, "OUTER_BOUNDARY_SUPPORT": {**excl, "n_valid": 40}},
            "strict_inside": {"mag_mean_px_per_pair": 0.2, "speed_p95_px_per_s": 5.0},
            "support_inclusive": {"speed_p95_px_per_s": 6.0}, "support_to_strict_p95_ratio": 1.2,
            "corr_strict_vs_support_waveform": 0.8, "corr_strict_with_background_median_mag": 0.1,
            "n_regions": len(regions), "n_active_blocks": 50,
            "inter_region_synchrony": "NOT_AVAILABLE (fewer than 2 motion-pattern regions)",
            "pixel_distribution_inside": {"median": 0.1, "iqr": 0.05}, "lag_resolution_s": 0.1, "regions": regions}


def test_rate_is_60_over_positive_period_and_never_called_bpm():
    r = {"fraction_px_outer_support": 0.3, "n_mag_peaks": 10, "mag_peaks_per_positive_period": 2.0,
         "mag_period": {"positive_peak_period_s": 0.8, "positive_peak_autocorr": 0.6, "pilot_rule_period_s": 0.8},
         "centroid_based_radial_fraction_active": 0.5, "centroid_based_tangential_fraction_active": 0.4,
         "within_region_timing_dispersion": None}
    row = adapters.row_from_summary(_summary([r]), {"source": "COMPUTED"})
    assert row["pf_mag_rate_per_min"] == pytest.approx(75.0)
    assert row["pixel_first_status"] == "COMPUTED" and row["pf_single_region"] is True
    assert not any("bpm" in k for k in row)


def test_no_region_gives_none_not_zero():
    row = adapters.row_from_summary(_summary([]), {"source": "REUSED_PILOT"})
    assert row["pixel_first_status"] == "REUSED"
    assert row["pf_mag_period_s"] is None and row["pf_mag_rate_per_min"] is None
    assert row["pf_n_regions"] == 0 and row["pf_single_region"] is False
