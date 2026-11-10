"""Spatially resolved motion analysis (Phase 2 validation, round 3).

Motivation (per project discussion): at beating onset, contraction can be
localized to part of the organoid rather than synchronized across the
whole body. A single whole-organoid (or whole-boundary-band) averaged
waveform can wash out or misrepresent that -- an active local region and
an inactive one can partially cancel in a spatial mean. This module tiles
a coarse organoid region (default: the dilated morphology mask -- a
"coarse organoid ROI" per the project's own terminology, not the exact
contour) into a grid of patches and computes per-patch signals
independently, so the *spatial* structure of motion can be inspected
before any single summary waveform is trusted.

This is explicitly NOT a biological metric yet. The goal for the current
(beating-onset) sample videos is only to check whether the motion activity
heatmap this module produces lines up with where beating is actually
visible in the raw video. The same functions are meant to be reusable
unchanged on later, more mature recordings, so local-vs-global
synchronization can be compared over developmental time using
`cross_patch_synchronization` below.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class PatchCell:
    """One grid cell over the analysis region, in ROI-local coordinates."""

    row: int
    col: int
    bbox_xyxy: tuple[int, int, int, int]
    coverage_fraction: float  # fraction of the cell's area inside the analysis region


def generate_patch_grid(
    analysis_region_mask: np.ndarray,
    cell_size_px: int,
    min_coverage_fraction: float = 0.3,
) -> list[PatchCell]:
    """Tile `analysis_region_mask`'s bounding shape into a regular grid of
    `cell_size_px`-sided square cells, keeping only cells whose overlap
    with the mask is at least `min_coverage_fraction`. `cell_size_px`
    should be derived from a ratio of the organoid's equivalent diameter
    by the caller (see configs/default.yaml: motion.spatial_patches),
    never a fixed literal, so the grid scales with organoid size.
    """
    if cell_size_px <= 0:
        raise ValueError("cell_size_px must be positive")
    h, w = analysis_region_mask.shape[:2]
    n_rows = h // cell_size_px
    n_cols = w // cell_size_px
    cells = []
    for row in range(n_rows):
        for col in range(n_cols):
            y0, y1 = row * cell_size_px, (row + 1) * cell_size_px
            x0, x1 = col * cell_size_px, (col + 1) * cell_size_px
            cell_region = analysis_region_mask[y0:y1, x0:x1]
            coverage = float(cell_region.mean()) if cell_region.size else 0.0
            if coverage >= min_coverage_fraction:
                cells.append(
                    PatchCell(row=row, col=col, bbox_xyxy=(x0, y0, x1, y1), coverage_fraction=coverage)
                )
    return cells


def patch_waveforms(
    radial_fields: list[np.ndarray], magnitude_fields: list[np.ndarray], cell: PatchCell
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce each frame-pair's radial-motion and flow-magnitude fields to
    one scalar each within `cell`'s bbox (mean, ignoring NaNs), returning
    (radial_waveform, magnitude_waveform) aligned with `radial_fields`'
    frame-pair order. Reuses already-computed whole-ROI fields (no new
    optical-flow computation) so tiling is cheap.
    """
    x0, y0, x1, y1 = cell.bbox_xyxy
    radial_wf = np.array([np.nanmean(f[y0:y1, x0:x1]) for f in radial_fields])
    magnitude_wf = np.array([np.nanmean(f[y0:y1, x0:x1]) for f in magnitude_fields])
    return radial_wf, magnitude_wf


def patch_field_waveform(fields: list[np.ndarray], cell: PatchCell) -> np.ndarray:
    """General version of `patch_waveforms` for a single arbitrary list of
    per-frame-pair 2D fields (e.g. local divergence or deformation
    magnitude, see deformation.py) -- reduces each to one scalar within
    `cell`'s bbox (mean, ignoring NaNs).
    """
    x0, y0, x1, y1 = cell.bbox_xyxy
    return np.array([np.nanmean(f[y0:y1, x0:x1]) for f in fields])


def classify_active_patches(
    patch_stds: dict[tuple[int, int], float], active_percentile: float = 75.0
) -> set[tuple[int, int]]:
    """Return the patch ids (row, col) whose radial-waveform std is at or
    above the `active_percentile` percentile among all analyzed patches --
    self-relative to this frame/video, not an absolute magic threshold.
    """
    if not patch_stds:
        return set()
    cutoff = float(np.percentile(list(patch_stds.values()), active_percentile))
    return {pid for pid, std in patch_stds.items() if std >= cutoff}


def active_region_fraction(
    cells: list[PatchCell], active_ids: set[tuple[int, int]], cell_area_px: float
) -> float:
    """Fraction of the analyzed organoid-region area (sum of all analyzed
    cells' *covered* area) that belongs to active cells. Purely
    descriptive -- not a validated biological measurement.
    """
    total_covered = sum(c.coverage_fraction * cell_area_px for c in cells)
    if total_covered == 0:
        return 0.0
    active_covered = sum(
        c.coverage_fraction * cell_area_px for c in cells if (c.row, c.col) in active_ids
    )
    return active_covered / total_covered


def build_activity_heatmap(
    cells: list[PatchCell], patch_scores: dict[tuple[int, int], float], roi_shape: tuple[int, int]
) -> np.ndarray:
    """Paint each cell's activity score (e.g. radial-waveform std) into an
    (H, W) float array matching `roi_shape`, zero elsewhere (unanalyzed /
    below-coverage cells)."""
    heatmap = np.zeros(roi_shape[:2], dtype=np.float32)
    for cell in cells:
        score = patch_scores.get((cell.row, cell.col), 0.0)
        x0, y0, x1, y1 = cell.bbox_xyxy
        heatmap[y0:y1, x0:x1] = score
    return heatmap


def overlay_heatmap(frame_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a normalized, colorized `heatmap` over `frame_bgr` for visual
    QC of where motion activity concentrates."""
    import cv2

    max_val = heatmap.max()
    normalized = (heatmap / max_val * 255).astype(np.uint8) if max_val > 0 else heatmap.astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    mask = (heatmap > 0)[..., None]
    blended = frame_bgr.copy()
    blend_region = cv2.addWeighted(frame_bgr, 1 - alpha, colored, alpha, 0)
    blended = np.where(mask, blend_region, blended)
    return blended.astype(np.uint8)


def cross_patch_synchronization(
    patch_waveforms_by_id: dict[tuple[int, int], np.ndarray]
) -> dict:
    """Mean pairwise Pearson correlation among the given patches' radial
    waveforms (already on a shared frame-pair time base -- no resampling
    needed, unlike waveform_consistency_across_steps).

    Intended to be run unchanged on later, more mature recordings: a low
    value here (patches move largely independently, as expected at
    beating onset per the project's own observation) that rises in later
    videos would be evidence of a shift from localized to globally
    synchronized contraction. Not a claim about that shift now -- just the
    metric structured so the comparison is possible later.
    """
    ids = list(patch_waveforms_by_id)
    if len(ids) < 2:
        return {"mean_pairwise_correlation": None, "n_patches": len(ids), "pairwise": {}}

    pairwise = {}
    correlations = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = patch_waveforms_by_id[ids[i]], patch_waveforms_by_id[ids[j]]
            if np.nanstd(a) == 0 or np.nanstd(b) == 0:
                corr = 0.0
            else:
                valid = ~(np.isnan(a) | np.isnan(b))
                corr = float(np.corrcoef(a[valid], b[valid])[0, 1]) if valid.sum() > 1 else 0.0
            pairwise[f"{ids[i]}_vs_{ids[j]}"] = corr
            correlations.append(corr)

    return {
        "mean_pairwise_correlation": float(np.mean(correlations)),
        "n_patches": len(ids),
        "pairwise": pairwise,
    }


def windowed_std(waveform: np.ndarray, n_windows: int) -> np.ndarray:
    """Split `waveform` into `n_windows` contiguous, roughly-equal chunks
    and return each chunk's std. Used to distinguish a patch whose
    variance comes from one isolated burst (few elevated windows) from one
    that is repeatedly active across the recording (most windows
    elevated) -- see `temporal_persistence_fraction`.
    """
    n = len(waveform)
    if n_windows <= 0:
        raise ValueError("n_windows must be positive")
    if n == 0:
        return np.zeros(n_windows)
    edges = np.linspace(0, n, n_windows + 1).astype(int)
    stds = []
    for i in range(n_windows):
        chunk = waveform[edges[i] : edges[i + 1]]
        chunk = chunk[~np.isnan(chunk)]
        stds.append(float(chunk.std()) if chunk.size else 0.0)
    return np.array(stds)


def temporal_persistence_fraction(window_stds: np.ndarray, ratio_threshold: float = 1.5) -> float:
    """Fraction of windows (from `windowed_std`) whose std is at least
    `ratio_threshold` times the patch's own quietest window -- self-
    relative to that patch, not an absolute magic threshold. A patch
    active in only one window (a single burst) scores low; one elevated
    in most windows (recurring activity across the recording) scores high.
    """
    if window_stds.size == 0:
        return 0.0
    baseline = window_stds.min()
    if baseline == 0:
        return float((window_stds > 0).mean())
    return float((window_stds >= ratio_threshold * baseline).mean())


@dataclasses.dataclass
class ActiveRegion:
    """A spatially contiguous cluster of active patch grid cells."""

    region_id: int
    patch_ids: list[tuple[int, int]]


def cluster_active_patches_into_regions(active_ids: set[tuple[int, int]]) -> list[ActiveRegion]:
    """Group active patch grid-ids (row, col) into spatially contiguous
    regions via 4-connectivity (shared-edge) flood fill.

    Motivation: independently averaging all active patches (or correlating
    every pair) treats a coordinated cluster of adjacent active patches the
    same as scattered, unrelated ones. Grouping by adjacency first lets a
    genuine localized beating region be analyzed (and later compared
    across regions or over developmental time) as the coherent unit it
    likely is.
    """
    remaining = set(active_ids)
    regions = []
    region_id = 0
    while remaining:
        start = next(iter(remaining))
        stack = [start]
        component: set[tuple[int, int]] = set()
        while stack:
            pid = stack.pop()
            if pid in component:
                continue
            component.add(pid)
            remaining.discard(pid)
            r, c = pid
            for neighbor in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if neighbor in remaining:
                    stack.append(neighbor)
        regions.append(ActiveRegion(region_id=region_id, patch_ids=sorted(component)))
        region_id += 1
    return regions


def region_waveform(
    region: ActiveRegion, patch_waveforms_by_id: dict[tuple[int, int], np.ndarray]
) -> np.ndarray:
    """Mean waveform across a region's member patches (ignoring NaNs)."""
    stacked = np.stack([patch_waveforms_by_id[pid] for pid in region.patch_ids])
    return np.nanmean(stacked, axis=0)


def neighbor_coherence(
    pid: tuple[int, int], patch_waveforms_by_id: dict[tuple[int, int], np.ndarray]
) -> float | None:
    """Mean Pearson correlation between patch `pid`'s waveform and its
    4-connected (edge-sharing) grid neighbors that were also analyzed.

    A genuinely beating local region should move together with its
    immediate spatial neighbors, not just show elevated activity in
    isolation -- this is independent evidence from amplitude/persistence/
    periodicity, which are all computed per patch with no notion of
    space. Returns None if `pid` has no analyzed neighbors.
    """
    r, c = pid
    wf = patch_waveforms_by_id[pid]
    correlations = []
    for neighbor in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
        if neighbor not in patch_waveforms_by_id:
            continue
        other = patch_waveforms_by_id[neighbor]
        valid = ~(np.isnan(wf) | np.isnan(other))
        if valid.sum() < 2 or np.nanstd(wf) == 0 or np.nanstd(other) == 0:
            correlations.append(0.0)
            continue
        correlations.append(float(np.corrcoef(wf[valid], other[valid])[0, 1]))
    return float(np.mean(correlations)) if correlations else None
