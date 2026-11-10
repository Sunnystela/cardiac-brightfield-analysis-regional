"""Generic (any batch/condition/well) longitudinal morphology pilot --
same method as scripts/p39_conv_a3_longitudinal_morphology_pilot.py
(existing PRODUCTION AUTOMATIC convention: single center-point SAM2
prompt, fixed candidate_index, no day-specific manual prompt, no D_point
prototype -- D_point is not yet validated broadly, see the multi-well
generalization check), generalized to any (batch, condition, well)
tuple so a SECOND organoid's trajectory can be built for the stage-
trajectory diagnostic without duplicating that script.

Per-day whole-organoid segmentation is independent per day (never a
copied prompt/box from another day or another well). Images use the
day's own real .tiff; videos use the video's own REAL midpoint frame
(frame_count // 2), same as the rest of this project. No functional
ROI/BPM computation here -- morphology only.
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

from cardio_organoid.io.longitudinal_metadata_xml import parse_pixel_calibration_um  # noqa: E402
from cardio_organoid.morphology.features import compute_morphology_features  # noqa: E402
from cardio_organoid.segmentation.sam2_segmenter import get_device, load_predictor  # noqa: E402

from _output_safety import add_output_dir_args, resolve_output_dir  # noqa: E402
from global_boundary_motion_diagnostic import _load_config  # noqa: E402
from p39_conv_a3_longitudinal_morphology_pilot import (  # noqa: E402
    _git_commit,
    calibration_for_image_day,
    render_overlay,
    segment_automatic,
)

TRACKING_ROOT = REPO_ROOT / "data" / "total" / "Tracking"
PIPELINE_VERSION = "generic_well_longitudinal_morphology_pilot_v1"


def _day_dir(batch_folder: str, condition_folder: str, day: int) -> Path:
    return TRACKING_ROOT / batch_folder / f"D{day}" / condition_folder


def load_image_day_frame(batch_folder: str, condition_folder: str, well: str, day: int) -> tuple[np.ndarray, Path]:
    path = _day_dir(batch_folder, condition_folder, day) / f"{well}.tiff"
    frame_bgr = cv2.imread(str(path))
    if frame_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), path


def load_video_day_midpoint_frame(batch_folder: str, condition_folder: str, well: str, day: int) -> tuple[np.ndarray, Path, int, int]:
    path = _day_dir(batch_folder, condition_folder, day) / f"{well}.avi"
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


def process_day(batch_folder, condition_folder, well, day, modality, predictor, sam2_cfg, output_dir) -> dict:
    day_label = f"D{day}"
    print(f"  {day_label} ({modality}) ...")
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    record: dict = {"day": day_label, "day_number": day, "modality": modality}
    try:
        if modality == "IMAGE":
            image_rgb, source_path = load_image_day_frame(batch_folder, condition_folder, well, day)
            record["source_file"] = str(source_path.relative_to(REPO_ROOT))
            record["reference_frame"] = "single_image_no_reference_needed"
            record["calibration"] = calibration_for_image_day(source_path)
        else:
            image_rgb, source_path, rep_index, frame_count = load_video_day_midpoint_frame(batch_folder, condition_folder, well, day)
            record["source_file"] = str(source_path.relative_to(REPO_ROOT))
            record["reference_frame"] = f"REAL video midpoint frame_index={rep_index} (frame_count={frame_count})"
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
            um_px = cal["um_per_pixel_x"]
            morphology_fields["area_um2"] = features.pixel.area_px * (um_px ** 2)
            morphology_fields["equivalent_diameter_um"] = features.pixel.equivalent_diameter_px * um_px
    record.update(morphology_fields)

    overlay = render_overlay(image_rgb, seg["mask"])
    overlay_path = frames_dir / f"{day_label}_overlay.png"
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    record["overlay_image"] = str(overlay_path.relative_to(REPO_ROOT))
    return record


def run_pilot(batch_folder: str, condition_folder: str, well: str, image_days: list[int], video_days: list[int], label: str, output_dir: Path) -> list[dict]:
    config = _load_config()
    sam2_cfg_full = config["segmentation"]["sam2"]
    provisional_cfg = config["segmentation"]["provisional_primary"]
    device = get_device(sam2_cfg_full["device"])
    predictor = load_predictor(checkpoint_path=REPO_ROOT / sam2_cfg_full["checkpoint_path"], model_cfg=sam2_cfg_full["model_cfg"], device=device)
    git_commit = _git_commit()

    records = []
    for day in image_days:
        rec = process_day(batch_folder, condition_folder, well, day, "IMAGE", predictor, provisional_cfg, output_dir)
        rec["pipeline_version"], rec["git_commit"] = PIPELINE_VERSION, git_commit
        records.append(rec)
        print(f"    status={rec.get('segmentation_status')} area_px={rec.get('area_px')}")
    for day in video_days:
        rec = process_day(batch_folder, condition_folder, well, day, "VIDEO", predictor, provisional_cfg, output_dir)
        rec["pipeline_version"], rec["git_commit"] = PIPELINE_VERSION, git_commit
        records.append(rec)
        print(f"    status={rec.get('segmentation_status')} area_px={rec.get('area_px')}")

    with open(output_dir / "records.json", "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"Wrote {output_dir / 'records.json'}")
    return records


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-folder", required=True)
    parser.add_argument("--condition-folder", required=True)
    parser.add_argument("--well", required=True)
    parser.add_argument("--image-days", required=True, help="comma-separated day numbers")
    parser.add_argument("--video-days", required=True, help="comma-separated day numbers")
    parser.add_argument("--label", required=True)
    canonical_dir = REPO_ROOT / "results" / "generic_well_longitudinal_morphology_pilot"
    add_output_dir_args(parser, canonical_dir, "generic_well_longitudinal_morphology_pilot")
    args = parser.parse_args()
    output_dir = resolve_output_dir(args, canonical_dir, args.label)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_days = [int(x) for x in args.image_days.split(",") if x.strip() != ""]
    video_days = [int(x) for x in args.video_days.split(",") if x.strip() != ""]
    run_pilot(args.batch_folder, args.condition_folder, args.well, image_days, video_days, args.label, output_dir)


if __name__ == "__main__":
    main()
