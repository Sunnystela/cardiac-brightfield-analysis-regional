"""Integrated per-well + population outputs for the P36/Conv first-pass
screening (called by p36_conv_first_pass_screening.py --stage integrate).

Display rules: real day positions; IMAGE and VIDEO never share a y-axis;
dotted connectors only (a line between accepted days does not imply
measurement in between); missing / not-imaged / not-run / NOT_ESTIMABLE are
never drawn as 0 or as a normal value; LOW_CONFIDENCE and caveated
measurements are visually distinct; every summary records its denominator.
No biological or statistical claim.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

from cardio_organoid.pipeline.segmentation_interface import QC_ACCEPT, SegmentationRecord, require_downstream_mask  # noqa: E402
from generic_well_longitudinal_morphology_pilot import load_image_day_frame, load_video_day_midpoint_frame  # noqa: E402
from p36_conv_integrated_batch import BATCH, BATCH_DIR, CONDITION, CONDITION_DIR  # noqa: E402

MORPH = ["area_px", "perimeter_px", "equivalent_diameter_px", "circularity", "solidity",
         "eccentricity", "aspect_ratio", "radial_cv", "perimeter_over_sqrt_area"]
FUNC_Q = {"ESTIMABLE": "#2e7d32", "LOW_CONFIDENCE": "#f9a825", "NOT_ESTIMABLE": "#c62828"}
MOD_COL = {"IMAGE": "#1f77b4", "VIDEO": "#ff7f0e"}


def _f(v):
    return None if v in (None, "", "None") else float(v)


def build_master(out: Path, man: Path):
    recs = [SegmentationRecord.from_dict(d) for d in json.loads((man / "segmentation_interface_records.json").read_text())["records"]]
    rev = {(r["well"], int(r["day"])): r for r in csv.DictReader(open(man / "SEGMENTATION_REVISION_TABLE.csv", newline=""))}
    morph = {(m["well"], m["day"]): m for m in json.loads((out / "morphology" / "morphology_results.json").read_text())}
    func = {(f["well"], f["day"]): f for f in json.loads((out / "functional" / "functional_results.json").read_text())}
    rows = []
    for r in sorted(recs, key=lambda x: (x.well, x.day)):
        rv, m, f = rev[(r.well, r.day)], morph[(r.well, r.day)], func[(r.well, r.day)]
        rows.append({
            "observation_id": rv["observation_id"], "batch": BATCH, "condition": CONDITION, "well": r.well, "day": r.day,
            "modality": r.modality, "source_file": rv["source_file"], "reference_frame": rv["reference_frame"],
            "segmentation_qc_status": r.segmentation_qc_status, "qc_reason": r.qc_reason, "mask_provenance": r.mask_provenance,
            "selected_candidate": r.selected_candidate, "original_selector_candidate": rv["original_selector_candidate"],
            "owner_decision": rv["owner_decision"], "owner_comment": rv["owner_comment"],
            "acceptable_candidate_set": rv["acceptable_candidate_set"], "error_direction": rv["error_direction"],
            "measurement_caveat": rv["measurement_caveat"], "mask_path": r.mask_path,
            "morphology_status": m["morphology_status"], "morphology_unit": m["unit"],
            "calibration_status": m["calibration_status"], "um_per_pixel": m["um_per_pixel"],
            **{k: m[k] for k in MORPH}, "area_um2": m["area_um2"], "equivalent_diameter_um": m["equivalent_diameter_um"],
            "functional_execution_status": f.get("execution_status"), "functional_execution_reason": f.get("execution_reason"),
            "functional_qc_status": f.get("functional_qc_status"), "bpm": f.get("bpm"),
            "median_ibi_s": f.get("median_ibi_s"), "ibi_cv": f.get("ibi_cv"),
            "consensus_beat_count": f.get("consensus_beat_count"),
            "reliable_active_fraction": f.get("reliable_active_fraction"), "observable_fraction": f.get("observable_fraction"),
            "period_family_bpms_descriptive": f.get("period_family_bpms_descriptive"),
            "triggered_failure_modes": f.get("triggered_failure_modes"),
            "actual_fps": f.get("actual_fps"), "frame_step": f.get("frame_step"),
            "functional_cache": f.get("cache"), "motion_dir": f.get("motion_dir"),
            "motion_segmentation_source": f.get("motion_segmentation_source"),
            "regional_participation": f.get("regional_participation"), "dominant_region_strength": f.get("dominant_region_strength"),
            "regional_phase_lag": f.get("regional_phase_lag"), "synchrony_coherence": f.get("synchrony_coherence"),
            "activation_sequence": f.get("activation_sequence"),
        })
    return rows, recs


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------ per-well plots
def _qc_strip(ax, rows, all_days):
    observed = {r["day"]: r for r in rows}
    labels = ["segmentation QC", "functional", "modality"]
    for d in all_days:
        r = observed.get(d)
        cells = []
        if r is None:
            cells = ["#ffffff"] * 3
        else:
            seg = "#2e7d32" if r["segmentation_qc_status"] == QC_ACCEPT else "#c62828"
            if r["segmentation_qc_status"] == QC_ACCEPT and r["measurement_caveat"]:
                seg = "#81c784"
            fx = r["functional_execution_status"]
            fc = "#e3f2fd" if r["modality"] == "IMAGE" else (FUNC_Q.get(r["functional_qc_status"]) if fx in ("REUSED", "RERUN")
                                                             else "#9e9e9e" if fx == "NOT_RUN" else "#000000")
            cells = [seg, fc, MOD_COL[r["modality"]]]
        for i, c in enumerate(cells):
            if r is None:  # not imaged that day: hatched, visibly different from any status colour
                ax.add_patch(plt.Rectangle((d - 0.45, 2 - i - 0.4), 0.9, 0.8, facecolor="#f5f5f5", edgecolor="#9e9e9e",
                                           hatch="////", lw=0.4))
            else:
                ax.add_patch(plt.Rectangle((d - 0.45, 2 - i - 0.4), 0.9, 0.8, facecolor=c, edgecolor="#9e9e9e", lw=0.4))
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_ylim(-0.6, 2.6)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#2e7d32", "#81c784", "#c62828", "#f9a825", "#9e9e9e", "#e3f2fd", "#1f77b4", "#ff7f0e"]]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="#f5f5f5", hatch="////", ec="#9e9e9e"))
    ax.legend(handles, ["ACCEPT / ESTIMABLE", "ACCEPT with caveat", "NONE / NOT_ESTIMABLE", "LOW_CONFIDENCE", "functional NOT_RUN",
                        "IMAGE: functional not measured", "IMAGE", "VIDEO", "not imaged"], fontsize=6, ncol=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.5))


def _morph_panel(ax, rows, metric):
    axes = {"VIDEO": ax, "IMAGE": ax.twinx()}
    for mod, mk in (("IMAGE", "s"), ("VIDEO", "o")):
        pts = [r for r in rows if r["modality"] == mod and r["morphology_status"] == "MEASURED" and r[metric] is not None]
        if pts:
            axes[mod].plot([r["day"] for r in pts], [r[metric] for r in pts], ls=":", marker=mk, color=MOD_COL[mod], ms=4)
            cav = [r for r in pts if r["measurement_caveat"]]
            if cav:
                axes[mod].scatter([r["day"] for r in cav], [r[metric] for r in cav], s=90, facecolors="none", edgecolors="red", zorder=4)
        axes[mod].set_ylabel(f"{metric} [{mod}]", color=MOD_COL[mod], fontsize=6)
        axes[mod].tick_params(labelsize=6)
    for r in rows:
        if r["morphology_status"] != "MEASURED":
            ax.axvline(r["day"], color="#c62828", alpha=0.15, lw=5)


def plot_well(well, rows, wdir, all_days):
    tag = f"{BATCH}/{CONDITION}/{well}"
    fig, axes = plt.subplots(3, 3, figsize=(14, 9), sharex=True)
    for ax, mt in zip(axes.flat, MORPH):
        _morph_panel(ax, rows, mt)
        ax.set_xlim(min(all_days) - 1, max(all_days) + 1)
    fig.suptitle(f"{tag} morphology (ACCEPT masks only; IMAGE and VIDEO on separate axes, pixel units, not linked; "
                 "red ring = caveat; red band = no acceptable mask)", fontsize=9)
    fig.tight_layout()
    fig.savefig(wdir / f"{well}_2_MORPHOLOGY_TRAJECTORY.png", dpi=100)
    plt.close(fig)

    vid = [r for r in rows if r["modality"] == "VIDEO"]
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [3, 3, 3, 1.2]})
    for ax, key, lab in zip(axes[:3], ["bpm", "ibi_cv", "reliable_active_fraction"], ["BPM", "IBI CV", "active-region fraction (round-36 state A)"]):
        for r in vid:
            v = r.get(key)
            if v is None or r["functional_qc_status"] not in ("ESTIMABLE", "LOW_CONFIDENCE"):
                continue
            filled = r["functional_qc_status"] == "ESTIMABLE"
            ax.scatter(r["day"], v, s=40, color=FUNC_Q[r["functional_qc_status"]] if filled else "none",
                       edgecolors=FUNC_Q[r["functional_qc_status"]], linewidths=1.5, zorder=3)
        ax.set_ylabel(lab, fontsize=7)
    _qc_strip(axes[3], rows, all_days)
    axes[0].set_title(f"{tag} functional (VIDEO only; filled = ESTIMABLE, hollow = LOW_CONFIDENCE; "
                      "NOT_ESTIMABLE / NOT_RUN / IMAGE are never plotted as values)", fontsize=8)
    axes[3].set_xlim(min(all_days) - 1, max(all_days) + 1)
    fig.tight_layout()
    fig.savefig(wdir / f"{well}_3_FUNCTIONAL_TRAJECTORY.png", dpi=100)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 3, 1.3])
    ax1 = fig.add_subplot(gs[0])
    _morph_panel(ax1, rows, "area_px")
    ax1.set_title(f"{tag} integrated timeline -- first-pass screening, human-reviewed segmentation; no biological interpretation", fontsize=9)
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    for r in vid:
        if r.get("bpm") is not None:
            e = r["functional_qc_status"] == "ESTIMABLE"
            ax2.scatter(r["day"], r["bpm"], s=40, color=FUNC_Q[r["functional_qc_status"]] if e else "none",
                        edgecolors=FUNC_Q[r["functional_qc_status"]], linewidths=1.5)
    ax2.set_ylabel("BPM (ESTIMABLE filled, LOW_CONF hollow)", fontsize=7)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    _qc_strip(ax3, rows, all_days)
    ax3.set_xlim(min(all_days) - 1, max(all_days) + 1)
    ax3.set_xlabel("day (actual position)")
    fig.tight_layout()
    fig.savefig(wdir / f"{well}_4_INTEGRATED_TIMELINE.png", dpi=100)
    plt.close(fig)


def contact_sheet(well, recs_well, rows_by_day, wdir):
    tiles_rows = []
    for r in sorted(recs_well, key=lambda x: x.day):
        frame = (load_image_day_frame(BATCH_DIR, CONDITION_DIR, well, r.day)[0] if r.modality == "IMAGE"
                 else load_video_day_midpoint_frame(BATCH_DIR, CONDITION_DIR, well, r.day)[0])
        w = 300
        small = cv2.resize(frame, (w, int(frame.shape[0] * w / frame.shape[1])))
        over = small.copy()
        if r.segmentation_qc_status == QC_ACCEPT:
            m = require_downstream_mask(r, REPO_ROOT, "contact_sheet")
            ms = cv2.resize(m.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
            cnts, _ = cv2.findContours(ms, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(over, cnts, -1, (0, 0, 0), 4)
            cv2.drawContours(over, cnts, -1, (0, 220, 255), 2)
        else:
            cv2.putText(over, "NO ACCEPTABLE CANDIDATE", (8, over.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 40, 40), 2)
        row = rows_by_day[r.day]
        txt = (f"D{r.day} {r.modality} | {r.segmentation_qc_status} cand={r.selected_candidate} {r.mask_provenance}"
               + (f" | caveat {row['measurement_caveat']}" if row["measurement_caveat"] else ""))
        band = np.full((26, w * 2, 3), (255, 255, 255) if not row["measurement_caveat"] else (255, 225, 200), np.uint8)
        cv2.putText(band, txt, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        tiles_rows.append(np.vstack([band, np.hstack([small, over])]))
    maxw = max(t.shape[1] for t in tiles_rows)
    sheet = np.vstack([np.hstack([t, np.full((t.shape[0], maxw - t.shape[1], 3), 255, np.uint8)]) for t in tiles_rows])
    cv2.imwrite(str(wdir / f"{well}_1_SEGMENTATION_QC_CONTACT_SHEET.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


# ------------------------------------------------------------------ population plots
def _grid(rows, wells, days, value_fn):
    g = np.full((len(wells), len(days)), np.nan)
    observed = np.zeros_like(g, dtype=bool)
    idx = {(r["well"], r["day"]): r for r in rows}
    for i, w in enumerate(wells):
        for j, d in enumerate(days):
            r = idx.get((w, d))
            if r is not None:
                observed[i, j] = True
                v = value_fn(r)
                if v is not None:
                    g[i, j] = v
    return g, observed


def heatmap_numeric(rows, wells, days, value_fn, title, path, cmap="viridis"):
    g, obs = _grid(rows, wells, days, value_fn)
    fig, ax = plt.subplots(figsize=(max(8, len(days) * 0.55), max(6, len(wells) * 0.32)))
    ax.set_facecolor("white")
    for i in range(len(wells)):
        for j in range(len(days)):
            if not obs[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="//", ec="#cfcfcf", lw=0))
            elif np.isnan(g[i, j]):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#bdbdbd"))
    im = ax.imshow(np.ma.masked_invalid(g), cmap=cmap, aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels([f"D{d}" for d in days], fontsize=7, rotation=90)
    ax.set_yticks(range(len(wells)))
    ax.set_yticklabels(wells, fontsize=7)
    ax.set_title(title + "\n(hatched = not imaged that day; grey = observed but no value -- never shown as 0)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def heatmap_categorical(rows, wells, days, cat_fn, cats, title, path):
    codes = {c: i for i, c in enumerate(cats)}
    g, obs = _grid(rows, wells, days, lambda r: codes.get(cat_fn(r)))
    fig, ax = plt.subplots(figsize=(max(8, len(days) * 0.55), max(6, len(wells) * 0.32)))
    cmap = ListedColormap([c for _, c in cats.items()])
    ax.imshow(np.ma.masked_invalid(g), cmap=cmap, vmin=-0.5, vmax=len(cats) - 0.5, aspect="auto")
    for i in range(len(wells)):
        for j in range(len(days)):
            if not obs[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="//", ec="#cfcfcf", lw=0))
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels([f"D{d}" for d in days], fontsize=7, rotation=90)
    ax.set_yticks(range(len(wells)))
    ax.set_yticklabels(wells, fontsize=7)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in cats.values()] + [plt.Rectangle((0, 0), 1, 1, fill=False, hatch="//", ec="#9e9e9e")],
              list(cats) + ["not imaged"], fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def distributions(rows, days, metric, path, modality_split=True, qc_filter=None):
    fig, ax = plt.subplots(figsize=(max(8, len(days) * 0.5), 4))
    for mod, off in (("IMAGE", -0.15), ("VIDEO", 0.15)):
        for j, d in enumerate(days):
            vals = [r[metric] for r in rows if r["day"] == d and r["modality"] == mod and r[metric] is not None
                    and (qc_filter is None or qc_filter(r))]
            if vals:
                ax.scatter(np.full(len(vals), j + off) + np.random.default_rng(j).uniform(-0.06, 0.06, len(vals)), vals,
                           s=10, color=MOD_COL[mod], alpha=0.7, label=mod if j == 0 else None)
                ax.hlines(np.median(vals), j + off - 0.12, j + off + 0.12, color="black", lw=1.5)
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels([f"D{d}" for d in days], fontsize=7, rotation=90)
    ax.set_ylabel(metric, fontsize=8)
    ax.set_title(f"{metric} by day (dots = wells, bar = median; IMAGE blue / VIDEO orange shown side by side, NOT one scale-comparable series)", fontsize=8)
    h, l = ax.get_legend_handles_labels()
    if h:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def stage_integrate(out: Path, man: Path):
    rows, recs = build_master(out, man)
    pop = out / "population"
    pop.mkdir(parents=True, exist_ok=True)
    write_csv(pop / "P36_CONV_MASTER_OBSERVATION_TABLE.csv", rows)
    wells = sorted({r["well"] for r in rows})
    days = sorted({r["day"] for r in rows})
    all_days = list(range(min(days), max(days) + 1))
    by_well = defaultdict(list)
    for r in rows:
        by_well[r["well"]].append(r)
    recs_by_well = defaultdict(list)
    for r in recs:
        recs_by_well[r.well].append(r)

    # ---- per well
    for w in wells:
        wdir = out / "per_well" / w
        wdir.mkdir(parents=True, exist_ok=True)
        wr = sorted(by_well[w], key=lambda x: x["day"])
        contact_sheet(w, recs_by_well[w], {r["day"]: r for r in wr}, wdir)
        plot_well(w, wr, wdir, all_days)
        write_csv(wdir / f"{w}_5_DAY_LEVEL.csv", wr)
        v = [r for r in wr if r["modality"] == "VIDEO"]
        est = [r["day"] for r in v if r["functional_qc_status"] == "ESTIMABLE"]
        qc = {"well": f"{BATCH}/{CONDITION}/{w}", "n_observations": len(wr),
              "days_observed": [r["day"] for r in wr],
              "segmentation": dict(Counter(r["segmentation_qc_status"] for r in wr)),
              "mask_provenance": dict(Counter(r["mask_provenance"] for r in wr)),
              "caveated_accepts": [r["observation_id"] for r in wr if r["measurement_caveat"]],
              "morphology_measured": {m: sum(1 for r in wr if r["modality"] == m and r["morphology_status"] == "MEASURED") for m in ("IMAGE", "VIDEO")},
              "functional_execution": dict(Counter(r["functional_execution_status"] for r in v)),
              "functional_qc": dict(Counter(r["functional_qc_status"] for r in v if r["functional_qc_status"])),
              "first_ESTIMABLE_video_day_within_observed_range": min(est) if est else None,
              "note": "first ESTIMABLE day is only the first observed video with an estimable BPM -- not a biological onset"}
        with open(wdir / f"{w}_6_QC_SUMMARY.json", "w") as fh:
            json.dump(qc, fh, indent=2, ensure_ascii=False)

    # ---- population heatmaps
    hm = pop / "heatmaps"
    hm.mkdir(exist_ok=True)
    for mt in MORPH:
        for mod in ("IMAGE", "VIDEO"):
            md = [d for d in days if any(r["day"] == d and r["modality"] == mod for r in rows)]
            heatmap_numeric([r for r in rows if r["modality"] == mod], wells, md, lambda r, mt=mt: _f(r[mt]),
                            f"{mt} [{mod} stream only, px]", hm / f"morph_{mt}_{mod}.png")
    vdays = [d for d in days if any(r["day"] == d and r["modality"] == "VIDEO" for r in rows)]
    vrows = [r for r in rows if r["modality"] == "VIDEO"]
    heatmap_numeric(vrows, wells, vdays, lambda r: _f(r["bpm"]), "BPM (ESTIMABLE + LOW_CONFIDENCE; see functional QC heatmap)", hm / "functional_BPM.png", "magma")
    heatmap_numeric(vrows, wells, vdays, lambda r: _f(r["bpm"]) if r["functional_qc_status"] == "ESTIMABLE" else None,
                    "BPM, ESTIMABLE only", hm / "functional_BPM_ESTIMABLE_only.png", "magma")
    heatmap_numeric(vrows, wells, vdays, lambda r: _f(r["ibi_cv"]) if r["functional_qc_status"] in ("ESTIMABLE", "LOW_CONFIDENCE") else None,
                    "IBI CV (ESTIMABLE + LOW_CONFIDENCE)", hm / "functional_IBI_CV.png", "cividis")
    heatmap_numeric(vrows, wells, vdays, lambda r: _f(r["reliable_active_fraction"]) if r["functional_execution_status"] in ("REUSED", "RERUN") else None,
                    "active-region fraction (round-36 state A / all patches; diagnostic)", hm / "functional_active_region_fraction.png")
    heatmap_categorical(rows, wells, days,
                        lambda r: ("ACCEPT_caveat" if r["measurement_caveat"] else "ACCEPT") if r["segmentation_qc_status"] == QC_ACCEPT else r["segmentation_qc_status"],
                        {"ACCEPT": "#2e7d32", "ACCEPT_caveat": "#81c784", "NO_ACCEPTABLE_CANDIDATE": "#c62828", "REVIEW_REQUIRED": "#f9a825"},
                        "segmentation QC (final human-reviewed manifest v2)", hm / "segmentation_QC.png")
    heatmap_categorical(rows, wells, days,
                        lambda r: "IMAGE_not_measured" if r["modality"] == "IMAGE" else (
                            r["functional_qc_status"] if r["functional_execution_status"] in ("REUSED", "RERUN") else r["functional_execution_status"]),
                        {"ESTIMABLE": "#2e7d32", "LOW_CONFIDENCE": "#f9a825", "NOT_ESTIMABLE": "#c62828", "NOT_RUN": "#757575",
                         "FAILED": "#000000", "PENDING_NOT_EXECUTED": "#8e24aa", "IMAGE_not_measured": "#e3f2fd"},
                        "functional QC / execution status", hm / "functional_QC_and_execution.png")
    heatmap_categorical(vrows, wells, vdays, lambda r: r["functional_execution_status"],
                        {"REUSED": "#1565c0", "RERUN": "#6a1b9a", "NOT_RUN": "#757575", "FAILED": "#000000", "PENDING_NOT_EXECUTED": "#ffb300"},
                        "functional execution (VIDEO only)", hm / "functional_execution_status.png")

    # ---- per-day counts with denominators
    counts = []
    for d in days:
        dr = [r for r in rows if r["day"] == d]
        vd = [r for r in dr if r["modality"] == "VIDEO"]
        ex = [r for r in vd if r["functional_execution_status"] in ("REUSED", "RERUN")]
        counts.append({"day": d, "n_wells_imaged": len(dr), "n_IMAGE": sum(r["modality"] == "IMAGE" for r in dr), "n_VIDEO": len(vd),
                       "n_segmentation_ACCEPT": sum(r["segmentation_qc_status"] == QC_ACCEPT for r in dr),
                       "n_morphology_measured": sum(r["morphology_status"] == "MEASURED" for r in dr),
                       "n_functional_executed": len(ex),
                       "n_ESTIMABLE": sum(r["functional_qc_status"] == "ESTIMABLE" for r in ex),
                       "n_LOW_CONFIDENCE": sum(r["functional_qc_status"] == "LOW_CONFIDENCE" for r in ex),
                       "n_NOT_ESTIMABLE": sum(r["functional_qc_status"] == "NOT_ESTIMABLE" for r in ex),
                       "fraction_of_VIDEOs_with_ESTIMABLE_BPM": (sum(r["functional_qc_status"] == "ESTIMABLE" for r in ex) / len(vd)) if vd else None,
                       "denominator_note": "fraction denominator = all VIDEO observations that day (incl. NOT_RUN); "
                                           "NOT a fraction of beating organoids; IMAGE-only / unimaged days are not non-beating"})
    write_csv(pop / "PER_DAY_COUNTS.csv", counts)
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(days))
    for k, (lab, col) in enumerate([("n_wells_imaged", "#bdbdbd"), ("n_morphology_measured", "#1f77b4"), ("n_functional_executed", "#ff7f0e"), ("n_ESTIMABLE", "#2e7d32")]):
        ax.bar(x + (k - 1.5) * 0.2, [c[lab] for c in counts], width=0.2, color=col, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{d}" for d in days], fontsize=7, rotation=90)
    ax.set_title("usable observations per day (counts; denominator = wells imaged that day, see PER_DAY_COUNTS.csv)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(pop / "per_day_usable_counts.png", dpi=100)
    plt.close(fig)

    # ---- distributions
    dist = pop / "distributions"
    dist.mkdir(exist_ok=True)
    num_rows = [{**r, **{k: _f(r[k]) for k in MORPH + ["bpm", "ibi_cv", "reliable_active_fraction"]}} for r in rows]
    for mt in MORPH:
        distributions(num_rows, days, mt, dist / f"morph_{mt}_by_day.png")
    distributions([r for r in num_rows if r["modality"] == "VIDEO"], vdays, "bpm", dist / "BPM_by_day_ESTIMABLE_only.png",
                  qc_filter=lambda r: r["functional_qc_status"] == "ESTIMABLE")
    distributions([r for r in num_rows if r["modality"] == "VIDEO"], vdays, "ibi_cv", dist / "IBI_CV_by_day_ESTIMABLE_only.png",
                  qc_filter=lambda r: r["functional_qc_status"] == "ESTIMABLE")

    # ---- caveats / unresolved list
    issues = []
    for r in rows:
        tags = []
        if r["segmentation_qc_status"] != QC_ACCEPT:
            tags.append(f"segmentation {r['segmentation_qc_status']} ({r['owner_comment'] or 'no reason given'})")
        if r["measurement_caveat"]:
            tags.append(f"measurement caveat {r['measurement_caveat']} ({r['owner_comment']})")
        if r["mask_provenance"] == "human_corrected":
            tags.append("human_corrected candidate")
        if r["acceptable_candidate_set"]:
            tags.append(f"acceptable set {r['acceptable_candidate_set']} (execution rule applied)")
        if r["functional_qc_status"] in ("LOW_CONFIDENCE", "NOT_ESTIMABLE"):
            tags.append(f"functional {r['functional_qc_status']} {r['triggered_failure_modes']}")
        if r["functional_execution_status"] in ("FAILED", "PENDING_NOT_EXECUTED"):
            tags.append(f"functional {r['functional_execution_status']}")
        if tags:
            issues.append({"observation_id": r["observation_id"], "well": r["well"], "day": r["day"], "modality": r["modality"], "items": " | ".join(tags)})
    write_csv(pop / "CAVEATS_AND_UNRESOLVED.csv", issues)
    summary = {"n_master_rows": len(rows), "n_wells": len(wells),
               "segmentation": dict(Counter(r["segmentation_qc_status"] for r in rows)),
               "morphology": dict(Counter(r["morphology_status"] for r in rows)),
               "functional_execution_VIDEO": dict(Counter(r["functional_execution_status"] for r in rows if r["modality"] == "VIDEO")),
               "functional_qc_executed": dict(Counter(r["functional_qc_status"] for r in rows if r["functional_execution_status"] in ("REUSED", "RERUN"))),
               "n_caveat_or_unresolved_rows": len(issues)}
    with open(pop / "population_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=1))
