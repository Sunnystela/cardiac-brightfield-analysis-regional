"""Builds an UNBIASED candidate-review dataset for the project owner to
directly compare SAM2's existing candidates 0/1/2 and pick the correct
one, BEFORE any selector/classifier is trained.

STRICT SCOPE, per explicit instruction:
  - No classifier/ML model is trained here.
  - No candidate-selection rule is changed.
  - No production/config/canonical/source data is modified.
  - SAM2 is not re-tuned; no threshold changed.
  - Candidates are regenerated read-only (segment_frame, the same
    single-center-point prompt already used in production) purely to
    render review panels and extract features -- this reproduces, never
    redefines, the existing candidate pool (same determinism property
    already verified in the prior HUMAN_QC_FAILURE_AUDIT round).
  - Review panels deliberately withhold SAM2 score, which candidate was
    originally selected, human score, automatic QC state, and any
    failure-source diagnosis -- to avoid biasing the human reviewer. All
    of that is preserved ONLY in review_index.csv / candidate_features.csv
    metadata, never shown on the images themselves.
  - All 3 candidate overlays use the SAME color/style so color cannot
    imply quality.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cardio_organoid.morphology.features import compute_morphology_features  # noqa: E402
from cardio_organoid.segmentation.sam2_segmenter import connected_component_stats, get_device, load_predictor, segment_frame  # noqa: E402
from generic_well_longitudinal_morphology_pilot import load_image_day_frame, load_video_day_midpoint_frame  # noqa: E402
from global_boundary_motion_diagnostic import _load_config  # noqa: E402
from p36_conv_human_qc_failure_audit import HUMAN_SCORES  # noqa: E402 (reused, not duplicated, to avoid transcription mismatch)
from p36_conv_screening_diagnostic import locate_well_paths  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "results/_REVIEW_LATEST/P36_CONV_SCREENING/CANDIDATE_SELECTOR_HUMAN_LABELING"
PANELS_DIR = OUTPUT_DIR / "individual_panels"
CONTACT_DIR = OUTPUT_DIR / "well_contact_sheets"

DAY_ORDER = [1, 3, 4, 5, 7, 8, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
IMAGE_DAYS = {1, 3, 4, 5, 7, 8}
EARLY_VIDEO_DAYS = {13, 14, 15, 16, 17, 18}
LATE_VIDEO_DAYS = {19, 20, 21, 22, 23, 24}
WELLS = ["a1", "a2", "b1", "c2", "d1", "e2", "f2", "g1", "h1"]
TRANSLUCENT_LOBE_WELLS = {"d1", "g1"}

CANDIDATE_COLOR = (255, 215, 0)  # SAME color for every candidate overlay, on purpose -- gold/amber, neutral, distinct from background
MASK_ALPHA = 0.35


def load_source_frame(well: str, day: int) -> np.ndarray:
    if day in IMAGE_DAYS:
        frame_rgb, _ = load_image_day_frame("P36 26.08.2026", "conv", well, day)
    else:
        frame_rgb, _, _, _ = load_video_day_midpoint_frame("P36 26.08.2026", "conv", well, day)
    return frame_rgb


# ---------------- TASK 1: review targets ----------------

def low_score_cases() -> list[tuple[str, int]]:
    cases = []
    for well in WELLS:
        for day, score in zip(DAY_ORDER, HUMAN_SCORES[well]):
            if score is not None and score <= 3:
                cases.append((well, day))
    return cases


def select_controls() -> list[tuple[str, int]]:
    """~15-20 score>=4 observations, balanced across well / IMAGE-VIDEO /
    early-late video, with EXTRA (image + early-video + late-video)
    coverage guaranteed for d1 and g1 specifically (the translucent-lobe
    wells) so their GOOD days are directly comparable to their BAD days.
    Deterministic (lowest-day-number-in-bucket), not random, so the
    exact selection is reproducible and auditable."""
    controls: list[tuple[str, int]] = []
    for i, well in enumerate(WELLS):
        good_days = [d for d, s in zip(DAY_ORDER, HUMAN_SCORES[well]) if s is not None and s >= 4]
        image_good = sorted(d for d in good_days if d in IMAGE_DAYS)
        early_video_good = sorted(d for d in good_days if d in EARLY_VIDEO_DAYS)
        late_video_good = sorted(d for d in good_days if d in LATE_VIDEO_DAYS)

        picks = []
        if image_good:
            picks.append(image_good[len(image_good) // 2])
        if well in TRANSLUCENT_LOBE_WELLS:
            if early_video_good:
                picks.append(early_video_good[0])
            if late_video_good:
                picks.append(late_video_good[0])
        else:
            # alternate which video window gets sampled, well by well, for overall balance
            pool = early_video_good if (i % 2 == 0) else late_video_good
            other = late_video_good if (i % 2 == 0) else early_video_good
            chosen_pool = pool if pool else other
            if chosen_pool:
                picks.append(chosen_pool[0])
        for d in picks:
            controls.append((well, d))
    return controls


# ---------------- panel rendering (deliberately unbiased) ----------------

def overlay_mask(frame_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = frame_rgb.copy().astype(np.float32)
    color_arr = np.array(CANDIDATE_COLOR, dtype=np.float32)
    overlay[mask] = overlay[mask] * (1 - MASK_ALPHA) + color_arr * MASK_ALPHA
    overlay = overlay.astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, CANDIDATE_COLOR, 4)
    return overlay


def label_band(img: np.ndarray, text: str, band_h: int = 34) -> np.ndarray:
    band = np.full((band_h, img.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(band, text, (4, band_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([band, img])


def build_individual_panel(well: str, day: int, modality: str, frame_rgb: np.ndarray, candidates, out_path: Path) -> None:
    """2x2: [ORIGINAL, candidate 0] / [candidate 1, candidate 2].
    NO SAM2 score, NO selected marker, NO human score, NO QC state, NO
    diagnosis anywhere on this image -- only well/day/modality/candidate
    index, per explicit instruction (reviewer bias reduction)."""
    target_w = 480
    scale = target_w / frame_rgb.shape[1]
    target_h = int(frame_rgb.shape[0] * scale)
    frame_small = cv2.resize(frame_rgb, (target_w, target_h))

    tiles = [label_band(frame_small, f"{well} D{day} {modality} -- ORIGINAL")]
    for c in candidates[:3]:
        mask_small = cv2.resize(c.mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        ov = overlay_mask(frame_small, mask_small)
        tiles.append(label_band(ov, f"{well} D{day} {modality} -- candidate={c.index}"))
    while len(tiles) < 4:
        blank = np.full((target_h + 34, target_w, 3), 255, dtype=np.uint8)
        tiles.append(blank)

    top = np.hstack([tiles[0], tiles[1]])
    bottom = np.hstack([tiles[2], tiles[3]])
    sheet = np.vstack([top, bottom])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def build_well_contact_sheet(well: str, day_panels: list[tuple[int, str, np.ndarray, list]], out_path: Path) -> None:
    """One row per day: [ORIGINAL, candidate0, candidate1, candidate2],
    large enough to judge boundaries, same neutral styling as individual
    panels."""
    target_w = 380
    rows = []
    for day, modality, frame_rgb, candidates in sorted(day_panels, key=lambda x: x[0]):
        scale = target_w / frame_rgb.shape[1]
        target_h = int(frame_rgb.shape[0] * scale)
        frame_small = cv2.resize(frame_rgb, (target_w, target_h))
        row_tiles = [label_band(frame_small, f"D{day} {modality} ORIGINAL", band_h=28)]
        for c in candidates[:3]:
            mask_small = cv2.resize(c.mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            ov = overlay_mask(frame_small, mask_small)
            row_tiles.append(label_band(ov, f"D{day} candidate={c.index}", band_h=28))
        rows.append(np.hstack(row_tiles))
    max_w = max(r.shape[1] for r in rows)
    def pad(r):
        return np.hstack([r, np.full((r.shape[0], max_w - r.shape[1], 3), 255, dtype=np.uint8)]) if r.shape[1] < max_w else r
    sheet = np.vstack([pad(r) for r in rows])
    header = np.full((36, sheet.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(header, f"{well} -- LOW-SCORE days, candidate review (no scores/labels shown)", (6, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    sheet = np.vstack([header, sheet])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


# ---------------- TASK 5: candidate feature extraction ----------------

def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2)


def boundary_band(mask: np.ndarray, width_px: int = 5) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    dil = cv2.dilate(mask_u8, kernel, iterations=width_px).astype(bool)
    ero = cv2.erode(mask_u8, kernel, iterations=width_px).astype(bool)
    return dil & (~ero)


def extract_candidate_features(well: str, day: int, modality: str, frame_rgb: np.ndarray, candidates, selected_index: int) -> list[dict]:
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad = gradient_magnitude(gray)
    grad_flat_sorted = np.sort(grad.flatten())
    grad_p75 = float(np.percentile(grad_flat_sorted, 75))

    areas = [c.area_fraction for c in candidates]
    scores = [c.score for c in candidates]
    area_order = sorted(range(len(candidates)), key=lambda i: -areas[i])  # 0 = largest
    score_order = sorted(range(len(candidates)), key=lambda i: -scores[i])
    max_area = max(areas) if areas else None
    selected_area = candidates[selected_index].area_fraction if 0 <= selected_index < len(candidates) else None

    # "common core" = the SMALLEST candidate this observation -- a neutral,
    # per-observation reference, not tied to any specific well/phenotype
    core_idx = min(range(len(candidates)), key=lambda i: areas[i])
    core_mask = candidates[core_idx].mask
    core_dilated = cv2.dilate(core_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=10).astype(bool)
    far_bg_region = ~cv2.dilate(candidates[area_order[0]].mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=15).astype(bool)
    core_mean_intensity = float(gray[core_mask].mean()) if core_mask.sum() else None
    far_bg_mean_intensity = float(gray[far_bg_region].mean()) if far_bg_region.sum() else None

    rows = []
    for i, c in enumerate(candidates):
        mask = c.mask
        n_comp, largest_frac = connected_component_stats(mask)
        feats = compute_morphology_features(mask)
        pf = feats.pixel

        mean_intensity = float(gray[mask].mean()) if mask.sum() else None
        sd_intensity = float(gray[mask].std()) if mask.sum() else None
        band = boundary_band(mask, width_px=5)
        mean_gradient_boundary = float(grad[band].mean()) if band.sum() else None
        edge_support_fraction = float((grad[band] >= grad_p75).mean()) if band.sum() else None

        ring_outer = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=25).astype(bool)
        ring = ring_outer & (~mask)
        ring_mean_intensity = float(gray[ring].mean()) if ring.sum() else None
        mask_vs_bg_contrast = (ring_mean_intensity - mean_intensity) if (ring_mean_intensity is not None and mean_intensity is not None) else None

        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        texture_local_variance = float(lap[mask].var()) if mask.sum() else None

        border = np.zeros_like(mask, dtype=bool)
        border[0, :] = border[-1, :] = True
        border[:, 0] = border[:, -1] = True
        border_touch_fraction = float((mask & border).sum() / border.sum()) if border.sum() else 0.0

        # extra region relative to this observation's own smallest ("core") candidate
        extra_mask = mask & (~core_mask)
        extra_area_px = int(extra_mask.sum())
        extra_region_mean_intensity = float(gray[extra_mask].mean()) if extra_area_px else None
        extra_contiguous_with_core_fraction = (
            float((extra_mask & core_dilated).sum() / extra_area_px) if extra_area_px else None
        )
        # DIAGNOSTIC-ONLY feature (never a hardcoded biological rule): does the
        # extra region added by this candidate sit at an INTERMEDIATE
        # brightness between the core and the far background -- i.e. a
        # lighter-but-still-distinct attached region, exactly the pattern the
        # project owner described for d1/g1, but computed identically for
        # EVERY well/candidate here, never conditioned on well name.
        added_intermediate_brightness_region = None
        if extra_region_mean_intensity is not None and core_mean_intensity is not None and far_bg_mean_intensity is not None:
            lo, hi = sorted([core_mean_intensity, far_bg_mean_intensity])
            added_intermediate_brightness_region = float(lo < extra_region_mean_intensity < hi)

        # pairwise overlap vs the other two candidates
        overlaps = {}
        centroid_displacements = {}
        my_centroid = np.array(np.nonzero(mask)).mean(axis=1) if mask.sum() else None
        for j, c2 in enumerate(candidates):
            if j == i:
                continue
            inter = (mask & c2.mask).sum()
            union = (mask | c2.mask).sum()
            overlaps[j] = float(inter / union) if union else None
            other_centroid = np.array(np.nonzero(c2.mask)).mean(axis=1) if c2.mask.sum() else None
            if my_centroid is not None and other_centroid is not None:
                centroid_displacements[j] = float(np.linalg.norm(my_centroid - other_centroid))
            else:
                centroid_displacements[j] = None

        rows.append({
            "well": well, "day": day, "modality": modality, "candidate_index": c.index,
            "sam2_confidence_score": c.score,
            "area_fraction": c.area_fraction, "area_px": c.area_px,
            "perimeter_px": pf.perimeter_px, "connected_component_count": n_comp,
            "largest_component_fraction": largest_frac, "solidity": pf.solidity, "circularity": pf.circularity,
            "eccentricity": pf.eccentricity, "aspect_ratio": pf.aspect_ratio, "border_touch_fraction": border_touch_fraction,
            "mean_intensity_inside_mask": mean_intensity, "sd_intensity_inside_mask": sd_intensity,
            "mask_vs_background_contrast": mask_vs_bg_contrast, "mean_gradient_near_boundary": mean_gradient_boundary,
            "texture_local_variance": texture_local_variance, "edge_support_fraction": edge_support_fraction,
            "area_rank": area_order.index(i) + 1, "sam2_score_rank": score_order.index(i) + 1,
            "area_ratio_to_largest_candidate": (c.area_fraction / max_area) if max_area else None,
            "area_ratio_to_current_selected": (c.area_fraction / selected_area) if selected_area else None,
            "overlap_with_candidate_other1": list(overlaps.values())[0] if len(overlaps) > 0 else None,
            "overlap_with_candidate_other2": list(overlaps.values())[1] if len(overlaps) > 1 else None,
            "centroid_displacement_other1_px": list(centroid_displacements.values())[0] if len(centroid_displacements) > 0 else None,
            "centroid_displacement_other2_px": list(centroid_displacements.values())[1] if len(centroid_displacements) > 1 else None,
            "extra_area_px_beyond_core_candidate": extra_area_px,
            "extra_region_mean_intensity": extra_region_mean_intensity,
            "extra_region_contiguous_with_core_fraction": extra_contiguous_with_core_fraction,
            "extra_region_intermediate_brightness_diagnostic_only": added_intermediate_brightness_region,
            "is_core_candidate_this_observation": i == core_idx,
            # preserved metadata only, NEVER shown on the review panel image itself:
            "existing_selected_candidate_index": selected_index, "is_existing_selected": i == selected_index,
        })
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PANELS_DIR.mkdir(parents=True, exist_ok=True)
    CONTACT_DIR.mkdir(parents=True, exist_ok=True)

    low_cases = low_score_cases()
    control_cases = select_controls()
    control_cases = [c for c in control_cases if c not in low_cases]  # avoid double-listing (shouldn't happen, disclosed safety)
    print(f"low-score review targets: {len(low_cases)}")
    print(f"control review targets: {len(control_cases)} -> {control_cases}")

    config = _load_config()
    sam2_cfg = config["segmentation"]["sam2"]
    provisional_cfg = config["segmentation"]["provisional_primary"]
    device = get_device(sam2_cfg["device"])
    predictor = load_predictor(checkpoint_path=REPO_ROOT / sam2_cfg["checkpoint_path"], model_cfg=sam2_cfg["model_cfg"], device=device)
    selected_index = provisional_cfg["candidate_index"]

    all_targets = [(w, d, "LOW_SCORE") for w, d in low_cases] + [(w, d, "CONTROL") for w, d in control_cases]

    review_index_rows = []
    feature_rows = []
    label_template_rows = []
    fewer_than_3_count = 0
    thumbnails = []
    well_day_panels: dict[str, list] = {w: [] for w in WELLS}

    records_cache: dict[str, dict] = {}
    for well in WELLS:
        records_path, _ = locate_well_paths(well)
        records_cache[well] = json.load(open(records_path)) if records_path and records_path.exists() else []

    print(f"Processing {len(all_targets)} total observations ...")
    for i, (well, day, group) in enumerate(all_targets):
        modality = "IMAGE" if day in IMAGE_DAYS else "VIDEO"
        print(f"  [{i+1}/{len(all_targets)}] {well} D{day} ({group}) ...")
        try:
            frame_rgb = load_source_frame(well, day)
        except (FileNotFoundError, RuntimeError) as exc:
            review_index_rows.append({"well": well, "day": day, "modality": modality, "group": group,
                                       "n_candidates": 0, "panel_path": None, "error": str(exc)})
            continue
        candidates = segment_frame(predictor, frame_rgb)
        if len(candidates) < 3:
            fewer_than_3_count += 1

        panel_path = PANELS_DIR / f"P36_Conv_{well}_D{day}_CANDIDATE_REVIEW.png"
        build_individual_panel(well, day, modality, frame_rgb, candidates, panel_path)
        well_day_panels[well].append((day, modality, frame_rgb, candidates))

        feature_rows.extend(extract_candidate_features(well, day, modality, frame_rgb, candidates, selected_index))

        rec_by_day = {r["day_number"]: r for r in records_cache.get(well, [])}
        rec = rec_by_day.get(day, {})
        original_score = HUMAN_SCORES[well][DAY_ORDER.index(day)]
        review_index_rows.append({
            "well": well, "day": day, "modality": modality, "group": group,
            "n_candidates": len(candidates), "panel_path": str(panel_path.relative_to(REPO_ROOT)),
            "original_human_score": original_score,
            "existing_qc_status_metadata_only": rec.get("segmentation_qc_flag"),
            "existing_selected_candidate_metadata_only": selected_index,
        })
        label_template_rows.append({
            "batch": "P36", "condition": "Conv", "well": well, "day": day, "modality": modality,
            "original_human_score": original_score,
            "candidate_0_available": len(candidates) > 0, "candidate_1_available": len(candidates) > 1,
            "candidate_2_available": len(candidates) > 2,
            "human_best_candidate": "", "human_candidate_quality": "",
            # error_direction: UNDER (best candidate misses real tissue) / OVER (best candidate includes
            # extra non-tissue area) -- blank when best_candidate is NONE or the mask is essentially exact.
            "error_direction": "",
            # error_extent: free-form severity bucket the reviewer assigns (e.g. MINOR_1_2_PERCENT) --
            # deliberately not a fixed enum here, since only a handful of shorthand examples exist so far.
            "error_extent": "",
            # phenotype_note: short structured tag for a recurring qualitative pattern (e.g.
            # balloon_like_tissue_missed) -- distinct from human_comment, which stays free text.
            "phenotype_note": "",
            "human_comment": "",
        })

        thumb = cv2.resize(frame_rgb, (140, int(140 * frame_rgb.shape[0] / frame_rgb.shape[1])))
        thumb = label_band(thumb, f"{well} D{day}", band_h=20)
        thumbnails.append(thumb)

    # ---------------- TASK 3: contact sheets ----------------
    for well in WELLS:
        low_panels = [p for p in well_day_panels[well] if any(w == well and d == p[0] for w, d in low_cases)]
        if low_panels:
            out_path = CONTACT_DIR / f"P36_Conv_{well}_LOW_SCORE_CANDIDATES_CONTACT_SHEET.png"
            build_well_contact_sheet(well, low_panels, out_path)

    # overall visual index (best-effort, thumbnails only, ORIGINAL frames)
    if thumbnails:
        max_w = max(t.shape[1] for t in thumbnails)
        max_h = max(t.shape[0] for t in thumbnails)
        padded = []
        for t in thumbnails:
            pad = np.full((max_h, max_w, 3), 255, dtype=np.uint8)
            pad[: t.shape[0], : t.shape[1], :] = t
            padded.append(pad)
        cols = 10
        rows = (len(padded) + cols - 1) // cols
        sheet = np.full((rows * max_h, cols * max_w, 3), 255, dtype=np.uint8)
        for i, t in enumerate(padded):
            r, c = divmod(i, cols)
            sheet[r * max_h: r * max_h + max_h, c * max_w: c * max_w + max_w, :] = t
        cv2.imwrite(str(OUTPUT_DIR / "P36_CONV_ALL_OBSERVATIONS_INDEX_CONTACT_SHEET.png"), sheet)

    # ---------------- TASK 4: human label template ----------------
    with open(OUTPUT_DIR / "P36_CONV_CANDIDATE_HUMAN_LABEL_TEMPLATE.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(label_template_rows[0].keys()))
        w.writeheader()
        w.writerows(label_template_rows)

    # ---------------- TASK 5: feature table ----------------
    with open(OUTPUT_DIR / "candidate_features.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(feature_rows[0].keys()))
        w.writeheader()
        w.writerows(feature_rows)

    # ---------------- review_index.csv ----------------
    with open(OUTPUT_DIR / "review_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(review_index_rows[0].keys()))
        w.writeheader()
        w.writerows(review_index_rows)

    print(f"\nlow-score observations: {len(low_cases)}")
    print(f"control observations: {len(control_cases)}")
    print(f"observations with <3 candidates: {fewer_than_3_count}")
    print(f"individual panels: {PANELS_DIR}")
    print(f"well contact sheets: {CONTACT_DIR}")
    print(f"Wrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
