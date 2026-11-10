"""Aggregation for the P36/Conv 9-well pixel-first screening (exploratory).

Per-well longitudinal tables/timelines and primary-7-well population
descriptive statistics (n, mean, SD, variance, median, IQR, CV where the
metric is ratio-scaled and positive). Gaussian curves are reference lines
only, never a normality claim; real points are always shown. No p-values.
Supplementary wells (e4, f1) are reported separately. Repeated days and
pixels are never counted as independent organoids.
"""
from __future__ import annotations

import csv
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
FIRST_PASS = REPO_ROOT / "results/_scratch_runs/p36_conv_first_pass_screening_20260928T155956Z"
PRIMARY = ["a1", "b2", "c1", "e1", "e3", "h2", "h3"]
SUPP = ["e4", "f1"]
MORPH = ["area_px", "perimeter_px", "equivalent_diameter_px", "circularity", "solidity", "eccentricity", "aspect_ratio",
         "radial_cv", "perimeter_over_sqrt_area"]
SIZE = ["area_px", "perimeter_px", "equivalent_diameter_px"]
RATIO_OK = {"area_px", "perimeter_px", "equivalent_diameter_px", "aspect_ratio", "perimeter_over_sqrt_area", "area_um2",
            "legacy_bpm", "legacy_median_ibi_s", "pf_strict_mag_mean", "pf_strict_speed_p95", "pf_support_incl_speed_p95",
            "pf_mag_rate_per_min"}
FUNC = [("legacy_bpm", "legacy BPM (ESTIMABLE only)"), ("legacy_ibi_cv", "legacy IBI CV (ESTIMABLE only)"),
        ("legacy_median_ibi_s", "legacy median IBI s (ESTIMABLE only)"),
        ("pf_strict_mag_mean", "pixel-first strict-inside mean |u| (px/pair)"),
        ("pf_strict_speed_p95", "pixel-first strict-inside p95 speed (px/s)"),
        ("pf_support_incl_speed_p95", "pixel-first support-inclusive p95 speed (px/s)"),
        ("pf_support_to_strict_p95_ratio", "outer-support / strict p95 |u| ratio"),
        ("pf_measurable_fraction", "measurable fraction of mask (method-defined)"),
        ("pf_mag_rate_per_min", "pixel-first |u| autocorr repetition rate (/min, positive-peak rule)"),
        ("pf_pixel_mean_mag_iqr", "within-video IQR of per-pixel mean |u| (valid inside px)")]


def f(v):
    try:
        return None if v in (None, "", "None", "nan") else float(v)
    except (TypeError, ValueError):
        return None


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def stats(vals, metric):
    v = np.array([x for x in vals if x is not None], float)
    if not len(v):
        return {"n": 0}
    s = {"n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None,
         "variance": float(v.var(ddof=1)) if len(v) > 1 else None, "median": float(np.median(v)),
         "q1": float(np.percentile(v, 25)), "q3": float(np.percentile(v, 75)), "iqr": float(np.subtract(*np.percentile(v, [75, 25])))}
    s["cv"] = (s["sd"] / s["mean"]) if (metric in RATIO_OK and s["sd"] is not None and s["mean"] > 0) else "not applicable"
    return s


def build_master(out):
    inv = json.loads((out / "inventory.json").read_text())
    fp = {(r["well"], int(r["day"])): r for r in csv.DictReader(open(FIRST_PASS / "population/P36_CONV_MASTER_OBSERVATION_TABLE.csv"))}
    rows, notprov = [], []
    for o in inv:
        m = fp[(o["well"], o["day"])]
        est = m["functional_qc_status"] == "ESTIMABLE"
        row = {"observation_id": o["observation_id"], "well": o["well"], "day": o["day"], "modality": o["modality"], "role": o["role"],
               "segmentation_qc_status": m["segmentation_qc_status"], "selected_candidate": m["selected_candidate"],
               "mask_provenance": m["mask_provenance"], "measurement_caveat": m["measurement_caveat"],
               "morphology_status": m["morphology_status"], "calibration_status": m["calibration_status"],
               **{k: f(m[k]) for k in MORPH}, "area_um2": f(m["area_um2"]),
               "legacy_execution": m["functional_execution_status"], "legacy_functional_qc": m["functional_qc_status"],
               "legacy_bpm_reported": f(m["bpm"]),
               "legacy_bpm": f(m["bpm"]) if est else None, "legacy_ibi_cv": f(m["ibi_cv"]) if est else None,
               "legacy_median_ibi_s": f(m["median_ibi_s"]) if est else None,
               "legacy_note": "rhythm columns only for ESTIMABLE; LOW_CONFIDENCE BPM kept in legacy_bpm_reported; NOT_ESTIMABLE IBI never used"}
        odir = out / "observations" / o["observation_id"]
        if o["modality"] == "IMAGE":
            row["pixel_first_status"] = "NOT_APPLICABLE_IMAGE"
        elif o["pixel_first_status"] == "INPUT_REVIEW_REQUIRED":
            row["pixel_first_status"] = "INPUT_REVIEW_REQUIRED"
            notprov.append({"observation_id": o["observation_id"], "item": "pixel-first", "reason": o.get("input_problem")})
        elif (odir / "summary.json").exists():
            s = json.loads((odir / "summary.json").read_text())
            p = json.loads((odir / "primitives.json").read_text())
            r1 = s["regions"][0] if s["regions"] else {}
            per = (r1.get("mag_period") or {}).get("positive_peak_period_s") if r1 else None
            disp = r1.get("within_region_timing_dispersion") if r1 else None
            row.update({"pixel_first_status": "REUSED" if p["source"] == "REUSED_PILOT" else "COMPUTED",
                        "pf_fps": s["fps"], "pf_dt_s": s["dt_s"],
                        "pf_measurable_fraction": s["measurable_fraction_inside_mask"],
                        "pf_valid_px_inside": s["exclusions"]["INSIDE_REFERENCE_MASK"]["n_valid"],
                        "pf_mask_px": s["exclusions"]["INSIDE_REFERENCE_MASK"]["n_region_px"],
                        "pf_excl_saturated": s["exclusions"]["INSIDE_REFERENCE_MASK"]["n_saturated"],
                        "pf_excl_low_texture": s["exclusions"]["INSIDE_REFERENCE_MASK"]["n_low_texture"],
                        "pf_excl_fb_inconsistent": s["exclusions"]["INSIDE_REFERENCE_MASK"]["n_fb_inconsistent"],
                        "pf_excl_unique": s["exclusions"]["INSIDE_REFERENCE_MASK"]["n_excluded_unique"],
                        "pf_excl_multiple_reasons": s["exclusions"]["INSIDE_REFERENCE_MASK"]["n_excluded_multiple_reasons"],
                        "pf_valid_px_outer_support": s["exclusions"]["OUTER_BOUNDARY_SUPPORT"]["n_valid"],
                        "pf_strict_mag_mean": s["strict_inside"]["mag_mean_px_per_pair"],
                        "pf_strict_speed_p95": s["strict_inside"]["speed_p95_px_per_s"],
                        "pf_support_incl_speed_p95": s["support_inclusive"]["speed_p95_px_per_s"],
                        "pf_support_to_strict_p95_ratio": s["support_to_strict_p95_ratio"],
                        "pf_corr_strict_vs_support": s["corr_strict_vs_support_waveform"],
                        "pf_corr_strict_with_background": s["corr_strict_with_background_median_mag"],
                        "pf_n_regions": s["n_regions"], "pf_single_region": s["n_regions"] == 1,
                        "pf_n_active_blocks": s["n_active_blocks"],
                        "pf_region1_fraction_outer_support": r1.get("fraction_px_outer_support"),
                        "pf_region1_n_mag_peaks": r1.get("n_mag_peaks"),
                        "pf_region1_peaks_per_period": r1.get("mag_peaks_per_positive_period"),
                        "pf_mag_period_s": per, "pf_mag_rate_per_min": (60.0 / per) if per else None,
                        "pf_mag_period_autocorr": (r1.get("mag_period") or {}).get("positive_peak_autocorr") if r1 else None,
                        "pf_mag_period_pilot_rule_s": (r1.get("mag_period") or {}).get("pilot_rule_period_s") if r1 else None,
                        "pf_radial_fraction_centroid": r1.get("centroid_based_radial_fraction_active"),
                        "pf_tangential_fraction_centroid": r1.get("centroid_based_tangential_fraction_active"),
                        "pf_inter_region_synchrony": "AVAILABLE" if isinstance(s["inter_region_synchrony"], list) else "NOT_AVAILABLE_single_or_no_region",
                        "pf_within_region_frac_lag0_h1": (disp or {}).get("first_half", {}).get("frac_lag0"),
                        "pf_within_region_frac_lag0_h2": (disp or {}).get("second_half", {}).get("frac_lag0"),
                        "pf_within_region_frac_same_lag": (disp or {}).get("frac_same_lag_both_halves"),
                        "pf_pixel_mean_mag_median": s["pixel_distribution_inside"]["median"],
                        "pf_pixel_mean_mag_iqr": s["pixel_distribution_inside"]["iqr"],
                        "pf_lag_resolution_s": s["lag_resolution_s"]})
            if not s["regions"]:
                notprov.append({"observation_id": o["observation_id"], "item": "motion-pattern region", "reason": "no active group >= 25 blocks"})
            if per is None:
                notprov.append({"observation_id": o["observation_id"], "item": "|u| repetition rate", "reason": "no positive-autocorrelation peak"})
        else:
            failed = (odir / "derive_failed.json").exists() or (odir / "primitives_failed.json").exists()
            row["pixel_first_status"] = "FAILED" if failed else "PENDING_NOT_EXECUTED"
            notprov.append({"observation_id": o["observation_id"], "item": "pixel-first", "reason": row["pixel_first_status"]})
        for k, label in (("legacy_bpm", "legacy BPM"), ("legacy_ibi_cv", "legacy IBI CV")):
            if o["modality"] == "VIDEO" and row.get(k) is None:
                notprov.append({"observation_id": o["observation_id"], "item": label,
                                "reason": f"legacy functional {m['functional_execution_status']}/{m['functional_qc_status'] or '-'}"})
        rows.append(row)
    return rows, notprov


def day_strip(rows, metric, label, path, days):
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 4.2), gridspec_kw={"width_ratios": [3, 1.3]})
    for j, d in enumerate(days):
        v = [r[metric] for r in rows if r["day"] == d and r.get(metric) is not None]
        if v:
            a.scatter(np.full(len(v), j) + np.random.default_rng(j).uniform(-0.12, 0.12, len(v)), v, s=14, color="tab:blue", zorder=3)
            a.hlines(np.median(v), j - 0.3, j + 0.3, color="black", lw=1.5)
            if len(v) > 1:
                m, sd = np.mean(v), np.std(v, ddof=1)
                a.errorbar(j + 0.33, m, yerr=sd, fmt="o", color="tab:red", ms=3, capsize=2)
    a.set_xticks(range(len(days)))
    a.set_xticklabels([f"D{d}" for d in days], fontsize=7)
    a.set_title(f"{label}: primary 7 wells by day (dots = wells; black = median; red = mean +- SD)", fontsize=8)
    allv = np.array([r[metric] for r in rows if r.get(metric) is not None], float)
    if len(allv):
        b.hist(allv, bins=min(20, max(5, len(allv) // 3)), density=True, color="lightgrey", edgecolor="grey")
        b.plot(np.sort(allv), np.zeros(len(allv)), "|", color="tab:blue", ms=12)
        if metric in RATIO_OK and len(allv) > 2 and allv.std() > 0:
            xs = np.linspace(allv.min(), allv.max(), 200)
            mu, sd = allv.mean(), allv.std(ddof=1)
            b.plot(xs, np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)), color="tab:red", lw=1, label="Gaussian reference")
            b.legend(fontsize=6)
        b.set_title(f"pooled well-days (n={len(allv)}; NOT independent organoids)", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=85)
    plt.close(fig)


def ecdf_plot(rows, metric, label, path):
    v = np.sort(np.array([r[metric] for r in rows if r.get(metric) is not None], float))
    if len(v) < 3:
        return
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color="tab:blue", label="ECDF (well-days)")
    ax.plot(v, np.zeros(len(v)), "|", color="tab:blue")
    if metric in RATIO_OK and v.std() > 0:
        from math import erf, sqrt
        mu, sd = v.mean(), v.std(ddof=1)
        xs = np.linspace(v.min(), v.max(), 200)
        ax.plot(xs, [0.5 * (1 + erf((x - mu) / (sd * sqrt(2)))) for x in xs], color="tab:red", lw=1, label="Gaussian reference CDF")
    ax.legend(fontsize=6)
    ax.set_title(f"{label} ECDF (primary 7)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=85)
    plt.close(fig)


def heatmap(rows, wells, days, metric, title, path, cmap="viridis"):
    idx = {(r["well"], r["day"]): r for r in rows}
    g = np.full((len(wells), len(days)), np.nan)
    obs = np.zeros_like(g, bool)
    for i, w in enumerate(wells):
        for j, d in enumerate(days):
            r = idx.get((w, d))
            if r is not None:
                obs[i, j] = True
                if r.get(metric) is not None:
                    g[i, j] = r[metric]
    fig, ax = plt.subplots(figsize=(max(7, len(days) * 0.55), 0.45 * len(wells) + 2))
    for i in range(len(wells)):
        for j in range(len(days)):
            if not obs[i, j]:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch="//", ec="#cfcfcf", lw=0))
            elif np.isnan(g[i, j]):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#bdbdbd"))
    im = ax.imshow(np.ma.masked_invalid(g), cmap=cmap, aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels([f"D{d}" for d in days], fontsize=7, rotation=90)
    ax.set_yticks(range(len(wells)))
    ax.set_yticklabels([w + (" (supp.)" if w in SUPP else "") for w in wells], fontsize=7)
    ax.axhline(len([w for w in wells if w in PRIMARY]) - 0.5, color="black", lw=1)
    ax.set_title(title + "\n(hatched = not imaged; grey = observed, value not available -- never 0)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=85)
    plt.close(fig)


def well_timeline(w, rows, path):
    rows = sorted(rows, key=lambda r: r["day"])
    days = [r["day"] for r in rows]
    fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True, gridspec_kw={"height_ratios": [3, 3, 3, 1.3]})
    for mod, mk, col, a in (("IMAGE", "s", "tab:blue", ax[0].twinx()), ("VIDEO", "o", "tab:orange", ax[0])):
        pts = [r for r in rows if r["modality"] == mod and r["area_px"] is not None]
        if pts:
            a.plot([r["day"] for r in pts], [r["area_px"] for r in pts], ls=":", marker=mk, color=col)
            cav = [r for r in pts if r["measurement_caveat"]]
            a.scatter([r["day"] for r in cav], [r["area_px"] for r in cav], s=90, facecolors="none", edgecolors="red")
        a.set_ylabel(f"area px [{mod}]", color=col, fontsize=7)
    ax[0].set_title(f"P36/Conv/{w} -- morphology (separate IMAGE/VIDEO axes), legacy and pixel-first functional; exploratory", fontsize=9)
    for r in rows:
        if r["legacy_bpm_reported"] is not None and r["legacy_functional_qc"] in ("ESTIMABLE", "LOW_CONFIDENCE"):
            e = r["legacy_functional_qc"] == "ESTIMABLE"
            ax[1].scatter(r["day"], r["legacy_bpm_reported"], color="tab:green" if e else "none", edgecolors="tab:green", s=40,
                          label="legacy BPM (filled ESTIMABLE, hollow LOW_CONF)")
        if r.get("pf_mag_rate_per_min") is not None:
            ax[1].scatter(r["day"], r["pf_mag_rate_per_min"], marker="x", color="black", s=30, label="pixel-first |u| repetition /min")
    h, l = ax[1].get_legend_handles_labels()
    ax[1].legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), fontsize=6)
    ax[1].set_ylabel("per min", fontsize=7)
    for key, col, lab in (("pf_strict_speed_p95", "green", "strict inside p95 speed"), ("pf_support_incl_speed_p95", "orange", "support-inclusive p95 speed")):
        pts = [r for r in rows if r.get(key) is not None]
        ax[2].plot([r["day"] for r in pts], [r[key] for r in pts], ls=":", marker="o", color=col, label=lab)
    b = ax[2].twinx()
    pts = [r for r in rows if r.get("pf_measurable_fraction") is not None]
    b.plot([r["day"] for r in pts], [r["pf_measurable_fraction"] for r in pts], ls=":", marker="^", color="grey", label="measurable fraction")
    b.set_ylim(0, 1)
    b.set_ylabel("measurable fraction", fontsize=7, color="grey")
    ax[2].legend(fontsize=6, loc="upper left")
    ax[2].set_ylabel("px/s", fontsize=7)
    for r in rows:
        seg = "#2e7d32" if r["segmentation_qc_status"] == "ACCEPT" else "#c62828"
        lq = {"ESTIMABLE": "#2e7d32", "LOW_CONFIDENCE": "#f9a825", "NOT_ESTIMABLE": "#c62828"}.get(r["legacy_functional_qc"], "#e3f2fd" if r["modality"] == "IMAGE" else "#9e9e9e")
        pf = {"COMPUTED": "#1565c0", "REUSED": "#6a1b9a"}.get(r["pixel_first_status"], "#e3f2fd" if r["modality"] == "IMAGE" else "#000000")
        for i, c in enumerate((seg, lq, pf)):
            ax[3].add_patch(plt.Rectangle((r["day"] - .45, 2 - i - .4), .9, .8, color=c))
    ax[3].set_yticks([2, 1, 0])
    ax[3].set_yticklabels(["segmentation", "legacy functional QC", "pixel-first"], fontsize=7)
    ax[3].set_ylim(-.6, 2.6)
    ax[3].set_xlim(min(days) - 1, max(days) + 1)
    ax[3].set_xlabel("day (actual position; dotted lines do not imply measurement between days)")
    fig.tight_layout()
    fig.savefig(path, dpi=85)
    plt.close(fig)


def aggregate(out: Path):
    A = out / "aggregate"
    A.mkdir(exist_ok=True)
    rows, notprov = build_master(out)
    write_csv(A / "MASTER_9WELL_130.csv", rows)
    write_csv(A / "NOT_PROVIDED_FAILED_HELD.csv", notprov or [{"observation_id": "", "item": "", "reason": "none"}])
    prim = [r for r in rows if r["role"] == "primary"]
    # baseline-relative change within each stream (first observed day of that stream in the same well; never interpolated)
    for r in rows:
        base = min((x for x in rows if x["well"] == r["well"] and x["modality"] == r["modality"] and x["area_px"] is not None),
                   key=lambda x: x["day"], default=None)
        for m in MORPH:
            if base and r[m] is not None and base[m] is not None:
                r[f"{m}_change"] = (r[m] / base[m]) if m in SIZE else (r[m] - base[m])
                r["baseline_day_stream"] = base["day"]
    write_csv(A / "MASTER_9WELL_130_with_baseline_change.csv", rows)
    # ---- per well
    for w in sorted({r["well"] for r in rows}):
        wr = sorted([r for r in rows if r["well"] == w], key=lambda r: r["day"])
        d = A / "per_well" / w
        write_csv(d / f"{w}_LONGITUDINAL.csv", wr)
        write_csv(d / f"{w}_COVERAGE_QC_BY_DAY.csv", [{k: r.get(k) for k in ("day", "modality", "segmentation_qc_status", "measurement_caveat",
                                                                                "legacy_execution", "legacy_functional_qc", "pixel_first_status",
                                                                                "pf_measurable_fraction", "pf_valid_px_inside", "pf_valid_px_outer_support",
                                                                                "pf_n_regions")} for r in wr])
        well_timeline(w, wr, d / f"{w}_INTEGRATED_TIMELINE.png")
    # ---- population (primary 7)
    P = A / "population_primary7"
    for sub in (P / "morphology", P / "functional", A / "heatmaps", A / "supplementary"):
        sub.mkdir(parents=True, exist_ok=True)
    stats_rows = []
    for mod in ("IMAGE", "VIDEO"):
        pm = [r for r in prim if r["modality"] == mod]
        days = sorted({r["day"] for r in pm})
        for m in MORPH + (["area_um2"] if mod == "IMAGE" else []):
            for dday in days:
                stats_rows.append({"stream": mod, "metric": m, "day": dday, **stats([r[m] for r in pm if r["day"] == dday], m)})
            day_strip(pm, m, f"{m} [{mod}]", P / "morphology" / f"{m}_{mod}_by_day.png", days)
            ecdf_plot(pm, m, f"{m} [{mod}]", P / "morphology" / f"{m}_{mod}_ecdf.png")
            if f"{m}_change" in pm[0] or any(f"{m}_change" in r for r in pm):
                day_strip(pm, f"{m}_change", f"{m} change vs first {mod} day ({'ratio' if m in SIZE else 'difference'})",
                          P / "morphology" / f"{m}_{mod}_change_by_day.png", days)
    vdays = sorted({r["day"] for r in prim if r["modality"] == "VIDEO"})
    pv = [r for r in prim if r["modality"] == "VIDEO"]
    for m, label in FUNC:
        for dday in vdays:
            stats_rows.append({"stream": "VIDEO", "metric": m, "day": dday, **stats([r.get(m) for r in pv if r["day"] == dday], m)})
        day_strip(pv, m, label, P / "functional" / f"{m}_by_day.png", vdays)
        ecdf_plot(pv, m, label, P / "functional" / f"{m}_ecdf.png")
    write_csv(P / "DAY_LEVEL_DESCRIPTIVE_STATS.csv", stats_rows)
    # trajectories of mean/SD/median/IQR
    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    for a, (m, label) in zip(axes.flat, [("area_px", "area px VIDEO"), ("circularity", "circularity VIDEO"), ("solidity", "solidity VIDEO"),
                                         ("legacy_bpm", "legacy BPM"), ("legacy_ibi_cv", "legacy IBI CV"), ("pf_strict_speed_p95", "strict p95 speed px/s"),
                                         ("pf_measurable_fraction", "measurable fraction"), ("pf_support_to_strict_p95_ratio", "support/strict p95"),
                                         ("pf_mag_rate_per_min", "|u| repetition /min")]):
        srs = [s for s in stats_rows if s["metric"] == m and s["stream"] == "VIDEO" and s.get("n", 0) > 0]
        ds = [s["day"] for s in srs]
        for w in PRIMARY:
            pts = [r for r in pv if r["well"] == w and r.get(m) is not None]
            a.plot([r["day"] for r in pts], [r[m] for r in pts], ls=":", color="lightgrey", lw=0.8)
        a.errorbar(ds, [s["mean"] for s in srs], yerr=[s["sd"] or 0 for s in srs], color="tab:red", fmt="-o", ms=3, capsize=2, label="mean +- SD")
        a.plot(ds, [s["median"] for s in srs], color="black", marker="s", ms=3, label="median")
        a.fill_between(ds, [s["q1"] for s in srs], [s["q3"] for s in srs], color="black", alpha=0.1, label="IQR")
        a.set_title(label + " (primary 7; grey = individual wells)", fontsize=8)
        a.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(P / "TRAJECTORIES_mean_SD_median_IQR.png", dpi=85)
    plt.close(fig)
    # heatmaps (all 9, supplementary below the line)
    wells = PRIMARY + SUPP
    alldays = sorted({r["day"] for r in rows})
    H = A / "heatmaps"
    for m in ["area_px", "circularity", "solidity"]:
        for mod in ("IMAGE", "VIDEO"):
            rr = [r for r in rows if r["modality"] == mod]
            heatmap(rr, wells, sorted({r["day"] for r in rr}), m, f"{m} [{mod}]", H / f"morph_{m}_{mod}.png")
    for m, label in FUNC:
        heatmap([r for r in rows if r["modality"] == "VIDEO"], wells, sorted({r["day"] for r in rows if r["modality"] == "VIDEO"}), m, label, H / f"func_{m}.png", "magma")
    # timing availability map
    tm = []
    for r in rows:
        if r["modality"] != "VIDEO":
            continue
        tm.append({**{k: r.get(k) for k in ("observation_id", "well", "day", "pf_n_regions", "pf_inter_region_synchrony",
                                             "pf_within_region_frac_lag0_h1", "pf_within_region_frac_lag0_h2", "pf_within_region_frac_same_lag",
                                             "pf_lag_resolution_s")}})
    write_csv(A / "TIMING_SYNCHRONY_SUMMARY.csv", tm)
    # legacy vs pixel-first repetition comparison (descriptive, not validation)
    comp = []
    for r in rows:
        if r["modality"] == "VIDEO" and r.get("pf_mag_rate_per_min") is not None and r["legacy_bpm_reported"] is not None:
            ratio = r["pf_mag_rate_per_min"] / r["legacy_bpm_reported"]
            comp.append({"observation_id": r["observation_id"], "legacy_qc": r["legacy_functional_qc"], "legacy_bpm": r["legacy_bpm_reported"],
                         "pf_mag_rate_per_min": r["pf_mag_rate_per_min"], "pf_mag_period_autocorr": r.get("pf_mag_period_autocorr"),
                         "ratio_pf_over_legacy": ratio,
                         "class": "within_10pct" if abs(ratio - 1) <= 0.10 else ("near_2x" if abs(ratio - 2) <= 0.2 else
                                                                                 ("near_0.5x" if abs(ratio - 0.5) <= 0.1 else "other_disagreement"))})
    write_csv(A / "LEGACY_VS_PIXEL_FIRST_REPETITION.csv", comp or [{"observation_id": "", "class": "none"}])
    # supplementary
    write_csv(A / "supplementary" / "f1_morphology_only.csv", [r for r in rows if r["well"] == "f1"])
    write_csv(A / "supplementary" / "e4_single_video.csv", [r for r in rows if r["well"] == "e4"])
    summ = {"n_rows": len(rows), "by_role": dict(Counter(r["role"] for r in rows)),
            "pixel_first_status": dict(Counter(r["pixel_first_status"] for r in rows)),
            "n_regions_distribution": dict(Counter(r.get("pf_n_regions") for r in rows if r["modality"] == "VIDEO")),
            "legacy_vs_pf_classes": dict(Counter(c["class"] for c in comp)),
            "n_not_provided_items": len(notprov)}
    (A / "aggregate_summary.json").write_text(json.dumps(summ, indent=2))
    # lightweight review zip (no flow, no npz, no movies)
    zp = A / "PIXEL_FIRST_9WELL_REVIEW.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in [out / "README.txt", out / "config" / "FROZEN_METHOD.json", out / "inventory.json"]:
            if p.exists():
                z.write(p, p.relative_to(out))
        for p in sorted(A.rglob("*")):
            if p.is_file() and p.suffix in (".csv", ".png", ".json") and p != zp:
                z.write(p, p.relative_to(out))
        for p in sorted((out / "observations").glob("*/summary.json")):
            z.write(p, p.relative_to(out))
        for p in sorted((out / "observations").glob("*/0[123]_*.png")):
            z.write(p, p.relative_to(out))
    print(json.dumps(summ, indent=1))
