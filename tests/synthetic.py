"""Tiny synthetic bright-field-like inputs for plumbing tests (not a biological model)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def make_pulsing_disk_video(path: Path, shape: tuple[int, int] = (1080, 1920), radius: int = 300, n_frames: int = 90,
                            fps: float = 30.0, beat_hz: float = 1.0, amplitude: float = 0.06,
                            seed: int = 0) -> tuple[Path, np.ndarray]:
    """Textured disk whose texture is radially scaled by 1 + amplitude*sin(2*pi*beat_hz*t), on a static
    textured background. Default frame shape = the P36 videos (1080x1920), so the functional ROI
    (mask + 400 px) leaves background outside it, as in real data.
    Returns (video path, bool mask of the disk on the midpoint frame)."""
    import cv2
    h, wd = shape
    rng = np.random.default_rng(seed)
    bg = cv2.GaussianBlur(rng.integers(90, 140, (h, wd)).astype(np.float32), (0, 0), 2)
    tex = cv2.GaussianBlur(rng.integers(40, 220, (h, wd)).astype(np.float32), (0, 0), 1.5)
    yy, xx = np.mgrid[:h, :wd]
    cx, cy = wd / 2, h / 2
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (wd, h))
    for f in range(n_frames):
        s = 1.0 + amplitude * np.sin(2 * np.pi * beat_hz * f / fps)
        M = np.float32([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)]])
        t = cv2.warpAffine(tex, M, (wd, h), borderMode=cv2.BORDER_REFLECT)
        inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= (radius * s) ** 2
        frame = np.where(inside, t, bg).clip(0, 255).astype(np.uint8)
        w.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    w.release()
    mid = n_frames // 2
    s_mid = 1.0 + amplitude * np.sin(2 * np.pi * beat_hz * mid / fps)
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= (radius * s_mid) ** 2
    return path, mask
