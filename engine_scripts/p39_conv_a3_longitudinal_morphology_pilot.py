"""Longitudinal morphology pilot for ONE CONFIRMED organoid --
P39/Conv/a3 -- across every day it was actually observed (per the
CONFIRMED identity mapping in `results/_scratch_runs/
longitudinal_identity_finalization_diagnostic_20260921T090140Z/
longitudinal_mapping_confirmed.json`: days D0-D12, no gaps, D0-D7
IMAGE-only, D8-D12 VIDEO-only, `days_with_image_and_video_same_day: []`).

Morphology-only, per explicit project instruction this round. Does NOT
touch, recompute, or read the functional motion ROI / BPM pipeline for
any day.

Per-day segmentation is run INDEPENDENTLY for every day using this
project's existing AUTOMATIC convention only (single center-point SAM2
prompt, fixed `candidate_index=1`, `pipeline/image_branch.py` for IMAGE
days / the same convention `pipeline/analyze.py` already uses for a
video's real midpoint frame for VIDEO days). The D11 manual box prompt
confirmed in `results/manual_reference_longitudinal/P39/Conv/a3/D11.json`
is DELIBERATELY NOT reused or copied to any other day here -- per
explicit instruction, every day's whole-organoid extent must be
independently judged, not assumed from one day's correction.

The automatic result is always computed and preserved first. A human
visual-QC pass (via this repo's own image-reading review, not an
automatic metric) then flags `review_required=True` with a plain-text
`review_note` for any day whose automatic mask visibly omits or
over-includes tissue -- it never silently substitutes a different mask;
any future manual correction for a flagged day would be recorded
separately (a new `results/manual_reference_longitudinal/P39/Conv/a3/
D<n>.json`, the same convention as D11), not by editing this script's
automatic output in place.

Calibration: IMAGE days carry a REAL per-file `um_per_pixel` value
parsed from that file's own `*.tiff_metadata.xml` "Scaling" block (see
`io.longitudinal_metadata_xml.parse_pixel_calibration_um` --
cross-checked against the file's own CameraPixelDistance/Magnification
tags, never estimated from image resolution). VIDEO days in this dataset
have NO metadata sidecar at all (confirmed by direct filesystem check --
no `*.avi_metadata.xml` or equivalent exists anywhere under this
project's `.avi` files) and are reported calibration_status=
NOT_CALIBRATED_no_metadata_source, values kept in raw pixels only. Image
(2560x1920) and video (1920x1080) frames in this dataset are also
different resolutions/aspect ratios -- direct evidence the two capture
paths are not the same optical setup, so an image day's um_per_pixel is
NEVER applied to a video day's pixel measurements, and no video-day
value is compared to an image-day value in physical units anywhere in
this pilot's outputs.

Day-to-day changes in any measured value are NOT interpreted here as
real growth/contraction, and the trajectory is NOT smoothed -- this
pilot only reports each day's own original frame + segmentation overlay
and measured values side by side, for human review.

Scope: this ONE organoid only (P39/Conv/a3). Does not start, and must
not be read as starting, full-dataset batch analysis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cardio_organoid.pipeline.segmentation_interface import assert_legacy_fixed_index_allowed  # noqa: E402
from cardio_organoid.io.longitudinal_metadata_xml import parse_pixel_calibration_um  # noqa: E402
from cardio_organoid.morphology.features import compute_morphology_features  # noqa: E402
from cardio_organoid.segmentation.sam2_segmenter import (  # noqa: E402
    get_device,
    load_predictor,
    provisional_primary_qc,
    segment_frame,
)

from _output_safety import add_output_dir_args, resolve_output_dir  # noqa: E402
from global_boundary_motion_diagnostic import _load_config  # noqa: E402

TRACKING_ROOT = REPO_ROOT / "data" / "total" / "Tracking"
BATCH_FOLDER = "P39 07.09.2026"
CONDITION_FOLDER = "conv"
WELL = "a3"
CANONICAL_DIR = REPO_ROOT / "results" / "p39_conv_a3_longitudinal_morphology_pilot"
REVIEW_DIR = REPO_ROOT / "results" / "_REVIEW_LATEST" / "P39_Conv_a3_LONGITUDINAL_MORPHOLOGY"
PIPELINE_VERSION = "p39_conv_a3_longitudinal_morphology_pilot_v1"

# From the CONFIRMED identity mapping (see module docstring) -- not
# re-derived here, just consumed as-is.
IMAGE_DAYS = [0, 1, 2, 3, 4, 5, 6, 7]
VIDEO_DAYS = [8, 9, 10, 11, 12]


def _day_dir(day: int) -> Path:
    return TRACKING_ROOT / BATCH_FOLDER / f"D{day}" / CONDITION_FOLDER


def _git_commit() -> str | None:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "-c", "safe.directory=" + str(REPO_ROOT), "rev-parse", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def load_image_day_frame(day: int) -> tuple[np.ndarray, Path]:
    path = _day_dir(day) / f"{WELL}.tiff"
    frame_bgr = cv2.imread(str(path))
    if frame_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), path


def load_video_day_midpoint_frame(day: int) -> tuple[np.ndarray, Path, int, int]:
    """Same convention as `pipeline/analyze.py`'s own representative-frame
    selection: the video's REAL midpoint frame, `frame_count // 2` --
    never called a cardiac-phase label, never re-derived per strategy."""
    path = _day_dir(day) / f"{WELL}.avi"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rep_index = frame_count // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, rep_index)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read representative frame {rep_index} from {path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), path, rep_index, frame_count


def segment_automatic(image_rgb: np.ndarray, predictor, sam2_cfg: dict) -> dict:
    """The project's existing AUTOMATIC convention only (single
    center-point prompt inside `segment_frame`, fixed
    `candidate_index`) -- no day-specific manual prompt of any kind."""
    assert_legacy_fixed_index_allowed("scripts.p39_conv_a3_longitudinal_morphology_pilot.segment_automatic")
    candidates = segment_frame(predictor, image_rgb)
    qc = provisional_primary_qc(candidates, sam2_cfg["candidate_index"], sam2_cfg["min_dominant_component_fraction"])
    if qc.get("flag") == "invalid_index":
        return {"mask": None, "qc": qc, "status": "SEGMENTATION_FAILED"}
    mask = candidates[sam2_cfg["candidate_index"]].mask
    return {"mask": mask, "qc": qc, "status": "MEASURED"}


def calibration_for_image_day(image_path: Path) -> dict:
    xml_path = image_path.with_name(image_path.name + "_metadata.xml")
    if not xml_path.exists():
        return {"calibration_status": "NOT_CALIBRATED_no_metadata_sidecar", "um_per_pixel": None}
    cal = parse_pixel_calibration_um(xml_path)
    if cal["um_per_pixel_x"] is None:
        return {"calibration_status": "NOT_CALIBRATED_sidecar_present_but_no_scaling_block", "um_per_pixel": None,
                "xml_sidecar": str(xml_path.relative_to(REPO_ROOT))}
    # um_per_pixel_x/y matched exactly in the D0-D7 census done before this
    # script was written; still reported separately here (never averaged
    # silently) so a future day with a differing X/Y value would be visible.
    return {
        "calibration_status": "CALIBRATED_from_real_metadata_xml",
        "um_per_pixel_x": cal["um_per_pixel_x"], "um_per_pixel_y": cal["um_per_pixel_y"],
        "xml_sidecar": str(xml_path.relative_to(REPO_ROOT)),
    }


def render_overlay(image_rgb: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    overlay = image_rgb.copy()
    if mask is not None:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 6)
    return overlay


def process_day(day: int, modality: str, predictor, sam2_cfg: dict, output_dir: Path) -> dict:
    day_label = f"D{day}"
    print(f"  {day_label} ({modality}) ...")
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    record: dict = {"day": day_label, "day_number": day, "modality": modality}
    try:
        if modality == "IMAGE":
            image_rgb, source_path = load_image_day_frame(day)
            record["source_file"] = str(source_path.relative_to(REPO_ROOT))
            record["reference_frame"] = "single_image_no_reference_needed"
            record["calibration"] = calibration_for_image_day(source_path)
        else:
            image_rgb, source_path, rep_index, frame_count = load_video_day_midpoint_frame(day)
            record["source_file"] = str(source_path.relative_to(REPO_ROOT))
            record["reference_frame"] = f"REAL video midpoint frame_index={rep_index} (frame_count={frame_count}, same convention as pipeline/analyze.py)"
            record["calibration"] = {"calibration_status": "NOT_CALIBRATED_no_metadata_source"}
    except (FileNotFoundError, RuntimeError) as exc:
        record["load_error"] = str(exc)
        record["segmentation_status"] = "LOAD_FAILED"
        return record

    seg = segment_automatic(image_rgb, predictor, sam2_cfg)
    record["segmentation_status"] = seg["status"]
    record["segmentation_source"] = f"sam2_candidate_index_{sam2_cfg['candidate_index']}_automatic_center_point"
    record["segmentation_qc_flag"] = seg["qc"].get("flag")

    morphology_fields = {}
    if seg["mask"] is not None and seg["status"] == "MEASURED":
        features = compute_morphology_features(seg["mask"])
        morphology_fields = {
            "area_px": features.pixel.area_px, "perimeter_px": features.pixel.perimeter_px,
            "equivalent_diameter_px": features.pixel.equivalent_diameter_px,
            "circularity": features.pixel.circularity, "solidity": features.pixel.solidity,
            "eccentricity": features.pixel.eccentricity, "aspect_ratio": features.pixel.aspect_ratio,
            "radial_cv": features.pixel.radial_cv,
        }
        cal = record["calibration"]
        if cal.get("calibration_status") == "CALIBRATED_from_real_metadata_xml":
            um_px = cal["um_per_pixel_x"]  # X/Y matched for this dataset; using X as the scalar factor, never averaged silently
            morphology_fields["area_um2"] = features.pixel.area_px * (um_px ** 2)
            morphology_fields["equivalent_diameter_um"] = features.pixel.equivalent_diameter_px * um_px
            morphology_fields["perimeter_um"] = features.pixel.perimeter_px * um_px if features.pixel.perimeter_px is not None else None
    record.update(morphology_fields)

    overlay = render_overlay(image_rgb, seg["mask"])
    orig_path = frames_dir / f"{day_label}_original.png"
    overlay_path = frames_dir / f"{day_label}_overlay.png"
    cv2.imwrite(str(orig_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    record["overlay_image"] = str(overlay_path.relative_to(REPO_ROOT))
    record["original_image"] = str(orig_path.relative_to(REPO_ROOT))

    # Filled in by the separate human visual-QC pass (see
    # apply_visual_review below) -- never auto-set here.
    record["review_required"] = None
    record["review_note"] = None

    return record


def build_contact_sheet(records: list[dict], output_path: Path) -> None:
    tiles = []
    max_h = 0
    for r in records:
        overlay_rel = r.get("overlay_image")
        if not overlay_rel:
            continue
        img = cv2.imread(str(REPO_ROOT / overlay_rel))
        h, w = img.shape[:2]
        target_w = 480
        target_h = int(h * target_w / w)
        img = cv2.resize(img, (target_w, target_h))
        label = f"{r['day']} {r['modality']}"
        if r.get("review_required"):
            label += "  [REVIEW_REQUIRED]"
        canvas = np.full((target_h + 40, target_w, 3), 255, dtype=np.uint8)
        canvas[40:, :, :] = img
        cv2.putText(canvas, label, (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 200) if r.get("review_required") else (0, 0, 0), 2, cv2.LINE_AA)
        tiles.append(canvas)
        max_h = max(max_h, canvas.shape[0])

    if not tiles:
        return
    n = len(tiles)
    cols = 5
    rows = (n + cols - 1) // cols
    tile_w = tiles[0].shape[1]
    sheet = np.full((rows * max_h, cols * tile_w, 3), 255, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        h = tile.shape[0]
        sheet[r * max_h: r * max_h + h, c * tile_w: (c + 1) * tile_w, :] = tile
    cv2.imwrite(str(output_path), sheet)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_output_dir_args(parser, CANONICAL_DIR, "p39_conv_a3_longitudinal_morphology_pilot")
    args = parser.parse_args()
    output_dir = resolve_output_dir(args, CANONICAL_DIR, "p39_conv_a3_longitudinal_morphology_pilot")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_config()
    sam2_cfg_full = config["segmentation"]["sam2"]
    provisional_cfg = config["segmentation"]["provisional_primary"]
    device = get_device(sam2_cfg_full["device"])
    predictor = load_predictor(checkpoint_path=REPO_ROOT / sam2_cfg_full["checkpoint_path"], model_cfg=sam2_cfg_full["model_cfg"], device=device)

    git_commit = _git_commit()

    records = []
    print("=== IMAGE days (D0-D7) ===")
    for day in IMAGE_DAYS:
        rec = process_day(day, "IMAGE", predictor, provisional_cfg, output_dir)
        rec["pipeline_version"] = PIPELINE_VERSION
        rec["git_commit"] = git_commit
        records.append(rec)
        print(f"    status={rec.get('segmentation_status')} area_px={rec.get('area_px')} calibration={rec.get('calibration', {}).get('calibration_status')}")

    print("\n=== VIDEO days (D8-D12) ===")
    for day in VIDEO_DAYS:
        rec = process_day(day, "VIDEO", predictor, provisional_cfg, output_dir)
        rec["pipeline_version"] = PIPELINE_VERSION
        rec["git_commit"] = git_commit
        records.append(rec)
        print(f"    status={rec.get('segmentation_status')} area_px={rec.get('area_px')} calibration={rec.get('calibration', {}).get('calibration_status')} reference={rec.get('reference_frame')}")

    with open(output_dir / "records.json", "w") as f:
        json.dump(records, f, indent=2, default=str)

    print(f"\nWrote per-day records + frames to {output_dir}")
    print("Next: visual review pass (see apply_visual_review.py step) before building the contact sheet.")


if __name__ == "__main__":
    main()
