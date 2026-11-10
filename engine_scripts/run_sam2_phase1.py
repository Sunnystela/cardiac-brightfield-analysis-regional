"""Phase 1 CLI: segment one representative frame of one video with SAM2,
save every candidate mask for human visual QC, and compute morphology
features on the provisional primary (compact organoid) mask.

Strategy A only (single center point) -- a point+box+negative-points
alternative was tried and dropped; see
scripts/compare_sam2_prompt_strategies.py for that comparison.

Does not select a final "correct" organoid mask with a validated rule --
the provisional primary candidate is an empirical convention that must keep
being checked via provisional_primary_qc (see sam2_segmenter.py).

Morphology features are computed on the single representative (currently:
video-midpoint placeholder) frame's morphology mask -- not phase-normalized
yet (docs/analysis_design.md #4) and not computed over every frame; that
waits on a relaxed-phase frame chosen from the future contraction waveform.

Motion analysis is a separate, not-yet-implemented pipeline (see
src/cardio_organoid/motion/) that will use only the morphology mask's
coarse centroid/bounding box, not this script's precise mask -- this
script does not produce any motion-ROI artifact. Does not run debris/
dead-cell classification. Never modifies the source video.

Usage:
    .venv/bin/python3 scripts/run_sam2_phase1.py --video "data/D8 beating/slit/a1.avi"
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cardio_organoid.morphology.features import compute_morphology_features  # noqa: E402
from cardio_organoid.segmentation.sam2_segmenter import (  # noqa: E402
    get_device,
    load_predictor,
    make_overlay,
    provisional_primary_qc,
    segment_frame,
)


def load_config() -> dict:
    with open(REPO_ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def read_representative_frame(video_path: Path) -> tuple[int, np.ndarray]:
    """Read the middle frame of the video as a placeholder representative
    frame. Not cardiac-phase-aware yet (see docs/analysis_design.md #4) --
    fine for a segmentation QC pass, not for final morphology measurement.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = frame_count // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    return frame_index, frame_bgr


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={REPO_ROOT}", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def parse_source_metadata(video_path: Path) -> dict:
    """Best-effort experimental metadata from the path, e.g.
    data/D8 beating/slit/a1.avi -> condition=slit, day=D8 beating, well=a1.
    Full metadata parsing belongs in io/metadata.py (not yet implemented).
    """
    parts = video_path.resolve().relative_to(REPO_ROOT).parts
    return {
        "source_file": str(video_path),
        "well_or_organoid_id": video_path.stem,
        "condition": parts[-2] if len(parts) >= 2 else None,
        "day_or_timepoint": parts[-3] if len(parts) >= 3 else None,
    }


def output_dir_for(video_path: Path, results_root: Path) -> Path:
    """Nest by condition (parent dir name) so e.g. conv/a1.avi and
    slit/a1.avi don't collide on a shared "a1" output folder."""
    return results_root / video_path.parent.name / video_path.stem


def excluded_video_reason(video_path: Path, config: dict) -> str | None:
    """Return the recorded QC exclusion reason for `video_path`
    (configs/default.yaml: qc.excluded_videos), or None if it isn't
    excluded. CLAUDE.md #11: exclusions must be reported, not silently
    discarded -- callers should log/record this, not just skip quietly.
    """
    resolved = video_path.resolve()
    for entry in config.get("qc", {}).get("excluded_videos", []):
        if (REPO_ROOT / entry["path"]).resolve() == resolved:
            return entry["reason"]
    return None


def run_phase1_on_video(
    video_path: Path, predictor, config: dict, output_dir: Path
) -> dict:
    reason = excluded_video_reason(video_path, config)
    if reason is not None:
        raise ValueError(f"{video_path} is excluded from analysis (qc.excluded_videos): {reason}")

    sam2_cfg = config["segmentation"]["sam2"]
    provisional_cfg = config["segmentation"]["provisional_primary"]
    morphology_cfg = config["morphology"]
    checkpoint_path = REPO_ROOT / sam2_cfg["checkpoint_path"]

    output_dir.mkdir(parents=True, exist_ok=True)

    frame_index, frame_bgr = read_representative_frame(video_path)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    cv2.imwrite(str(output_dir / "original_frame.png"), frame_bgr)

    candidates = segment_frame(predictor, frame_rgb)

    candidate_records = []
    for c in candidates:
        cv2.imwrite(
            str(output_dir / f"mask_{c.index}.png"), (c.mask.astype("uint8") * 255)
        )
        overlay_rgb = make_overlay(frame_rgb, c.mask)
        cv2.imwrite(
            str(output_dir / f"overlay_{c.index}.png"),
            cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR),
        )
        candidate_records.append(
            {
                "index": c.index,
                "score": c.score,
                "area_px": c.area_px,
                "area_fraction": c.area_fraction,
            }
        )
    with open(output_dir / "candidates.json", "w") as f:
        json.dump(candidate_records, f, indent=2)

    candidate_index = provisional_cfg["candidate_index"]
    qc = provisional_primary_qc(
        candidates,
        candidate_index=candidate_index,
        min_dominant_component_fraction=provisional_cfg["min_dominant_component_fraction"],
    )
    with open(output_dir / "provisional_primary_qc.json", "w") as f:
        json.dump(qc, f, indent=2)

    morphology_features_dict = None
    if qc["flag"] != "invalid_index":
        # "provisional_morphology_mask": candidate_index is an empirically
        # observed convention (see provisional_primary_qc above), not a
        # confirmed/validated organoid mask -- do not describe it as such
        # elsewhere. File/variable names below say "provisional" on purpose.
        provisional_morphology_mask = candidates[candidate_index].mask
        cv2.imwrite(
            str(output_dir / "provisional_morphology_mask.png"),
            provisional_morphology_mask.astype("uint8") * 255,
        )
        overlay_rgb = make_overlay(
            frame_rgb, provisional_morphology_mask, color=(255, 0, 0), alpha=0.45
        )
        cv2.imwrite(
            str(output_dir / "provisional_morphology_mask_overlay.png"),
            cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR),
        )

        morphology_features = compute_morphology_features(
            provisional_morphology_mask,
            pixel_size_um=morphology_cfg["pixel_size_um"],
            num_radial_angles=morphology_cfg["num_radial_angles"],
        )
        morphology_features_dict = dataclasses.asdict(morphology_features)
        with open(output_dir / "morphology_features.json", "w") as f:
            json.dump(morphology_features_dict, f, indent=2)

    run_metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "video_path": str(video_path),
        "frame_index": frame_index,
        "frame_shape_hw": list(frame_rgb.shape[:2]),
        **parse_source_metadata(video_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_of(checkpoint_path),
        "model_cfg": sam2_cfg["model_cfg"],
        "provisional_primary_candidate_index": candidate_index,
        "morphology_pixel_size_um": morphology_cfg["pixel_size_um"],
        "morphology_num_radial_angles": morphology_cfg["num_radial_angles"],
        "device": str(get_device(sam2_cfg["device"])),
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "torch_version": torch.__version__,
        "sam2_package_version": importlib.metadata.version("sam-2"),
        "python_version": sys.version,
        "git_commit": git_commit_hash(),
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(run_metadata, f, indent=2)

    return {
        "video": str(video_path),
        "output_dir": str(output_dir),
        "candidates": candidate_records,
        "provisional_primary_qc": qc,
        "morphology_features": morphology_features_dict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        default=REPO_ROOT / "data" / "D8 beating" / "slit" / "a1.avi",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    video_path = args.video
    output_dir = args.output_dir or output_dir_for(
        video_path, REPO_ROOT / "results" / "phase1"
    )

    config = load_config()
    sam2_cfg = config["segmentation"]["sam2"]
    device = get_device(sam2_cfg["device"])
    predictor = load_predictor(
        checkpoint_path=REPO_ROOT / sam2_cfg["checkpoint_path"],
        model_cfg=sam2_cfg["model_cfg"],
        device=device,
    )

    result = run_phase1_on_video(video_path, predictor, config, output_dir)

    print(f"Wrote {len(result['candidates'])} candidates to {output_dir}")
    for c in result["candidates"]:
        print(c)
    print("provisional_primary_qc:", result["provisional_primary_qc"])
    print("morphology_features:", result["morphology_features"])


if __name__ == "__main__":
    main()
