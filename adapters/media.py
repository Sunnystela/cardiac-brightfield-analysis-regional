"""Reading the representative frame the masks are defined on, and the ROI/mask preview."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def reference_frame(path: Path, modality: str) -> tuple[np.ndarray, dict]:
    """VIDEO: midpoint frame `frame_count // 2` (the frame every engine mask is defined on).
    IMAGE: the image itself. Returns (BGR frame, info)."""
    import cv2
    path = Path(path)
    if modality == "IMAGE":
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"cannot read image {path}")
        return img, {"frame_shape": list(img.shape[:2])}
    cap = cv2.VideoCapture(str(path))
    try:
        fc, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fc // 2)
        ok, fr = cap.read()
    finally:
        cap.release()
    if not ok:
        raise IOError(f"cannot read midpoint frame of {path}")
    return fr, {"frame_count": fc, "fps_recorded": fps, "frame_index": fc // 2, "frame_shape": list(fr.shape[:2]),
                "reference_frame": f"frame_index={fc // 2}/frame_count={fc}"}


def check_mask_matches_frame(mask: np.ndarray, frame_info: dict) -> None:
    if list(mask.shape[:2]) != list(frame_info["frame_shape"]):
        raise ValueError(f"mask shape {list(mask.shape[:2])} != reference frame shape {frame_info['frame_shape']}")


def render_mask_preview(path: Path, modality: str, mask: np.ndarray, out_png: Path, roi_expansion_px: int | None = None) -> dict:
    """Overlay the mask contour (green) and, for VIDEO, the functional ROI box (yellow) on the reference frame.

    The ROI box uses the engine's own `compute_motion_roi` with `motion.roi.expansion_px` from configs/default.yaml.
    """
    import cv2
    from ._engine import engine
    frame, info = reference_frame(path, modality)
    check_mask_matches_frame(mask, info)
    out = frame.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 0), 4)
    if modality == "VIDEO":
        cfg = engine("global_boundary_motion_diagnostic")._load_config()
        exp = roi_expansion_px if roi_expansion_px is not None else cfg["motion"]["roi"]["expansion_px"]
        roi = engine("cardio_organoid.motion.raft_flow").compute_motion_roi(mask, expansion_px=exp, frame_shape=frame.shape[:2])
        x0, y0, x1, y1 = roi.roi_bbox_xyxy
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1) - 1, int(y1) - 1), (0, 255, 255), 4)
        info["functional_roi_xyxy"] = [int(x0), int(y0), int(x1), int(y1)]
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), out)
    return {**info, "preview_png": str(out_png), "n_external_contours": len(contours), "mask_px": int(mask.sum())}
