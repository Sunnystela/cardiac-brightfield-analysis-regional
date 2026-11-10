"""Shared, well-agnostic plotting/reporting functions for the P36/Conv
longitudinal development + cross-well validation work. Extracted from
scripts/p36_conv_a1_development_trajectory_diagnostic.py so the SAME,
UNCHANGED plotting/reporting code is reused for every validation well
-- no per-well duplication, no per-well tuning of how these are built.

Trajectory stage model v3 itself (src/cardio_organoid/pipeline/
prototype_trajectory_stage_model_v3.py) is imported and used UNCHANGED
by callers of this module -- nothing here modifies its logic/thresholds.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cardio_organoid.pipeline.prototype_trajectory_stage_model_v2 import QC_NOT_REVIEWED, QC_OK_CONFIRMED  # noqa: E402

STAGE_COLORS = {
    "EARLY_DIFFUSE": "tab:red", "INITIAL_COMPACT": "tab:blue", "TRANSITIONAL_SHEDDING": "tab:orange",
    "MATURE_RECOMPACT": "tab:green", "UNCERTAIN": "gray", "UNCERTAIN_SEGMENTATION": "black",
    "TRANSIENT_ARTIFACT_CANDIDATE": "purple",
    # v4 modality-aware state names (prototype_trajectory_stage_model_v4_modality_aware.py)
    "VIDEO_COMPACT": "tab:blue", "VIDEO_UNCERTAIN": "gray", "VIDEO_TRANSITIONAL_CANDIDATE": "tab:orange",
    "VIDEO_TRANSIENT_SPIKE_CANDIDATE": "purple", "BASELINE_INSUFFICIENT": "black",
}


def build_qc_contact_sheet(records: list[dict], frames_dir: Path, qc_status: dict[int, str], well_label: str, output_path: Path) -> Path:
    tiles = []
    for rec in records:
        day = rec["day_number"]
        overlay_path = frames_dir / f"D{day}_overlay.png"
        img = cv2.imread(str(overlay_path))
        if img is None:
            continue
        target_w = 320
        scale = target_w / img.shape[1]
        img = cv2.resize(img, (target_w, int(img.shape[0] * scale)))
        qc = qc_status.get(day, QC_NOT_REVIEWED)
        label_color = (0, 150, 0) if qc == QC_OK_CONFIRMED else (0, 0, 200)
        band = np.full((44, target_w, 3), 255, dtype=np.uint8)
        cv2.putText(band, f"D{day} {rec['modality']}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(band, qc, (4, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1, cv2.LINE_AA)
        tile = np.vstack([band, img])
        tiles.append(tile)

    if not tiles:
        raise ValueError(f"{well_label}: no overlay frames found for contact sheet")
    cols = 6
    rows = (len(tiles) + cols - 1) // cols
    tile_h = max(t.shape[0] for t in tiles)
    tile_w = tiles[0].shape[1]
    sheet = np.full((rows * tile_h, cols * tile_w, 3), 255, dtype=np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * tile_h: r * tile_h + t.shape[0], c * tile_w: (c + 1) * tile_w, :] = t
    cv2.imwrite(str(output_path), sheet)
    return output_path


def build_morphology_trajectory(records: list[dict], clean_days: list[int], well_label: str, output_path: Path, day_range=None) -> Path:
    clean = [r for r in records if r["day_number"] in clean_days]
    if day_range is not None:
        clean = [r for r in clean if r["day_number"] in day_range]
    days = [r["day_number"] for r in clean]
    modality_by_day = {r["day_number"]: r["modality"] for r in clean}

    fields = [
        ("area_px", "area (px, raw)"), ("perimeter_px", "perimeter (px, raw)"), ("circularity", "circularity"),
        ("solidity", "solidity"), ("equivalent_diameter_px", "equiv. diameter (px, raw)"), ("radial_cv", "radial CV"),
        ("eccentricity", "eccentricity"), ("aspect_ratio", "aspect ratio"), ("perimeter_over_sqrt_area", "perimeter/sqrt(area)"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for ax, (key, title) in zip(axes.flat, fields):
        ys = [r.get(key) for r in clean]
        colors = ["tab:blue" if modality_by_day[d] == "IMAGE" else "tab:orange" for d in days]
        ax.plot(days, ys, "-", color="lightgray", zorder=1)
        for d, y, c in zip(days, ys, colors):
            if y is not None:
                ax.scatter([d], [y], color=c, s=55, zorder=2)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("day")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", markersize=9, label="IMAGE (raw px; see README for calibration)"),
               plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:orange", markersize=9, label="VIDEO (raw px only, NOT calibrated)")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03), fontsize=9)
    fig.suptitle(f"{well_label} -- morphology trajectory, CLEAN/CONFIRMED days only", y=1.06)
    fig.tight_layout()
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_stage_timeline(stage_results, output_path: Path) -> Path:
    tile_w, h = 110, 90
    canvas = np.full((h, tile_w * len(stage_results), 3), 255, dtype=np.uint8)
    for i, r in enumerate(stage_results):
        color_hex = STAGE_COLORS.get(r.stage, "gray")
        color_bgr = tuple(int(c * 255) for c in matplotlib.colors.to_rgb(color_hex))[::-1]
        cv2.rectangle(canvas, (i * tile_w, 15), ((i + 1) * tile_w - 4, h - 8), color_bgr, -1)
        cv2.putText(canvas, f"D{r.day_number}", (i * tile_w + 4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, r.stage[:12], (i * tile_w + 4, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
        if len(r.stage) > 12:
            cv2.putText(canvas, r.stage[12:], (i * tile_w + 4, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
        if r.confidence:
            cv2.putText(canvas, f"({r.confidence})", (i * tile_w + 4, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), canvas)
    return output_path


def build_morphology_function_overlay(records: list[dict], clean_days: list[int], functional: list[dict], stage_results, well_label: str, output_path: Path) -> Path:
    clean = [r for r in records if r["day_number"] in clean_days]
    days = [r["day_number"] for r in clean]
    stage_by_day = {r.day_number: r.stage for r in stage_results}
    func_by_day = {f["day_number"]: f for f in functional}

    fig, ax = plt.subplots(figsize=(14, 6))
    for key, color in [("solidity", "purple"), ("circularity", "brown")]:
        ax.plot(days, [r.get(key) for r in clean], "-o", label=key, color=color, markersize=6)
    ax2 = ax.twinx()
    bpm_days = [d for d in days if d in func_by_day and func_by_day[d]["bpm"] is not None]
    bpm_vals = [func_by_day[d]["bpm"] for d in bpm_days]
    ax2.scatter(bpm_days, bpm_vals, color="black", marker="^", s=80, label="candidate BPM (functional pipeline)", zorder=3)
    ax2.set_ylabel("BPM")
    for d in days:
        f = func_by_day.get(d)
        if f is None:
            continue
        status = f["functional_status"]
        marker_color = {"ESTIMABLE": "green", "LOW_CONFIDENCE": "orange", "NOT_ESTIMABLE": "red"}.get(status, "gray")
        ax.axvline(d, color=marker_color, alpha=0.15, linewidth=8)
    for d in days:
        stage = stage_by_day.get(d, "UNCERTAIN")
        ax.annotate(stage.replace("_", "\n")[:12], (d, ax.get_ylim()[0]), fontsize=6, ha="center", va="bottom", rotation=0)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right", fontsize=8)
    ax.set_title(f"{well_label} -- morphology (solidity/circularity) + functional onset overlay\n(background shade: green=ESTIMABLE, orange=LOW_CONFIDENCE, red=NOT_ESTIMABLE; never used to set stage)")
    ax.set_xlabel("day")
    fig.tight_layout()
    fig.savefig(output_path, dpi=115)
    plt.close(fig)
    return output_path
