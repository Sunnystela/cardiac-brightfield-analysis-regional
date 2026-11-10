"""Candidate-ACCEPTABILITY / abstention DIAGNOSTIC (Stage-1 design), per
explicit instruction following the P39 abstention-metric audit:
  - candidate selection itself is essentially solved (23/23, 24/24 on
    P39 once ambiguity/label-correction is accounted for);
  - the real remaining gap is DOES_AN_ACCEPTABLE_CANDIDATE_EXIST? (the 2
    P39 NONE cases), which nothing in the current pipeline detects.

This script is DIAGNOSTIC ONLY:
  - the frozen candidate-selector (Stage 2) is loaded and applied
    READ-ONLY for its existing probability outputs -- NEVER retrained,
    NEVER modified.
  - no abstention threshold is fit anywhere in this script.
  - SAM2 is used only for READ-ONLY candidate generation (the same
    single-center-point forward pass used throughout this project) --
    never fine-tuned.
  - nothing is wired into any production path.
  - P38 is NOT touched anywhere in this script (reserved final holdout).
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cardio_organoid.morphology.features import compute_morphology_features  # noqa: E402
from cardio_organoid.segmentation.sam2_segmenter import connected_component_stats, get_device, load_predictor, segment_frame  # noqa: E402
from generic_well_longitudinal_morphology_pilot import load_image_day_frame, load_video_day_midpoint_frame  # noqa: E402
from global_boundary_motion_diagnostic import _load_config  # noqa: E402
from p36_conv_candidate_selector_feasibility import build_feature_vector, impute_nan, predict_from_scores  # noqa: E402
from p36_conv_candidate_selector_human_labeling import HUMAN_SCORES as P36_HUMAN_SCORES  # noqa: E402
from p36_conv_candidate_selector_human_labeling import (  # noqa: E402
    DAY_ORDER as P36_DAY_ORDER, WELLS as P36_WELLS, IMAGE_DAYS as P36_IMAGE_DAYS,
    label_band, overlay_mask,
)
from p39_blind_validation_setup import IMAGE_DAYS as P39_IMAGE_DAYS, WELLS as P39_WELL_CFGS

FROZEN_MODEL_PATH = REPO_ROOT / "results/_REVIEW_LATEST/CANDIDATE_SELECTOR_ACTIVE_LEARNING_PLAN/frozen_diagnostic_selector/logistic_regression_selector.joblib"
P36_LABEL_DIR = REPO_ROOT / "results/_REVIEW_LATEST/P36_CONV_SCREENING/CANDIDATE_SELECTOR_HUMAN_LABELING"
P39_DIR = REPO_ROOT / "results/_REVIEW_LATEST/CANDIDATE_SELECTOR_P39_BLIND_VALIDATION"

OUTPUT_DIR = REPO_ROOT / "results/_REVIEW_LATEST/CANDIDATE_ACCEPTABILITY_ABSTENTION_DIAGNOSTIC"
PANELS_DIR = OUTPUT_DIR / "individual_panels"
CONTACT_DIR = OUTPUT_DIR / "well_contact_sheets"

MAX_PLAUSIBLE_AREA_FRACTION = 0.40   # reused, generic, disclosed constant from prototype_early_stage_aggregate_segmentation.py
MIN_PLAUSIBLE_AREA_FRACTION = 0.01   # generic floor -- even the largest candidate covering <1% of frame is implausible

frozen = joblib.load(FROZEN_MODEL_PATH)
MODEL, SCALER, FILL_VALUES, FEATURE_COLUMNS = frozen["model"], frozen["scaler"], frozen["fill_values"], frozen["feature_columns"]


# ---------------- generic frame loading across batches ----------------

def load_frame(batch_dir: str, condition: str, well: str, day: int, is_image: bool):
    if is_image:
        frame_rgb, _ = load_image_day_frame(batch_dir, condition, well, day)
    else:
        frame_rgb, _, _, _ = load_video_day_midpoint_frame(batch_dir, condition, well, day)
    return frame_rgb


def mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    inter = (m1 & m2).sum()
    union = (m1 | m2).sum()
    return float(inter / union) if union else 0.0


def mask_centroid(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return np.array([ys.mean(), xs.mean()])


def compute_stage1_features(candidates, frame_rgb, modality: str) -> dict:
    """Observation-level acceptability-diagnostic features -- candidate-
    SET features, not a single top1-probability threshold. No phenotype/
    day-specific hardcoded rule anywhere here."""
    idxs = sorted(c.index for c in candidates)
    cand_by_idx = {c.index: c for c in candidates}

    # --- A: selector uncertainty (frozen model, read-only) ---
    feat_by_idx = {}
    selected_index_placeholder = 1  # only used inside extract_candidate_features for the "extra region" reference; not a decision
    from p36_conv_candidate_selector_human_labeling import extract_candidate_features
    rows = extract_candidate_features("well", 0, modality, frame_rgb, candidates, selected_index_placeholder)
    feat_by_idx = {r["candidate_index"]: r for r in rows}

    probs = {}
    for idx in idxs:
        vec = []
        for col in FEATURE_COLUMNS:
            if col == "modality_is_video":
                vec.append(1.0 if modality == "VIDEO" else 0.0)
            else:
                v = feat_by_idx[idx].get(col)
                vec.append(np.nan if v is None else float(v))
        Xc = np.array([vec], dtype=float)
        Xc_imp, _ = impute_nan(Xc, FILL_VALUES)
        probs[idx] = float(MODEL.predict_proba(SCALER.transform(Xc_imp))[0, 1])

    sorted_probs = sorted(probs.values(), reverse=True)
    top1_prob = sorted_probs[0]
    top1_top2_margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else None
    prob_vals = np.clip(np.array(list(probs.values())), 1e-9, None)
    p_norm = prob_vals / prob_vals.sum()
    entropy = float(-np.sum(p_norm * np.log(p_norm)) / np.log(len(p_norm)))
    prob_spread = float(max(probs.values()) - min(probs.values()))
    predicted_candidate = predict_from_scores(probs)

    # --- B: SAM2 candidate agreement (pairwise, on raw masks) ---
    pair_ious = {}
    pair_centroid_dist = {}
    for i, j in itertools.combinations(idxs, 2):
        pair_ious[f"iou_{i}{j}"] = mask_iou(cand_by_idx[i].mask, cand_by_idx[j].mask)
        c1, c2 = mask_centroid(cand_by_idx[i].mask), mask_centroid(cand_by_idx[j].mask)
        pair_centroid_dist[f"centroid_dist_{i}{j}"] = float(np.linalg.norm(c1 - c2)) if c1 is not None and c2 is not None else None
    areas = {i: cand_by_idx[i].area_fraction for i in idxs}
    area_ratio_maxmin = max(areas.values()) / max(min(areas.values()), 1e-9)
    perim_complexity = {i: (cand_by_idx[i].area_px and (feat_by_idx[i]["perimeter_px"] / (cand_by_idx[i].area_px ** 0.5))) for i in idxs}
    perim_complexity_spread = max(perim_complexity.values()) - min(perim_complexity.values())

    # --- C: structural QC (worst-case across the 3 candidates) ---
    max_border_touch = max(feat_by_idx[i]["border_touch_fraction"] for i in idxs)
    max_component_count = max(feat_by_idx[i]["connected_component_count"] for i in idxs)
    min_largest_component_fraction = min(feat_by_idx[i]["largest_component_fraction"] for i in idxs)
    max_area_fraction = max(areas.values())
    any_implausibly_large = max_area_fraction > MAX_PLAUSIBLE_AREA_FRACTION
    all_implausibly_small = max_area_fraction < MIN_PLAUSIBLE_AREA_FRACTION

    # --- D: candidate-image consistency ---
    min_edge_support_fraction = min(feat_by_idx[i]["edge_support_fraction"] for i in idxs if feat_by_idx[i]["edge_support_fraction"] is not None) if any(feat_by_idx[i]["edge_support_fraction"] is not None for i in idxs) else None
    max_mask_vs_bg_contrast = max((feat_by_idx[i]["mask_vs_background_contrast"] for i in idxs if feat_by_idx[i]["mask_vs_background_contrast"] is not None), default=None)
    min_mask_vs_bg_contrast = min((feat_by_idx[i]["mask_vs_background_contrast"] for i in idxs if feat_by_idx[i]["mask_vs_background_contrast"] is not None), default=None)

    return {
        "top1_prob": round(top1_prob, 4), "top1_top2_margin": round(top1_top2_margin, 4) if top1_top2_margin is not None else None,
        "entropy": round(entropy, 4), "probability_spread": round(prob_spread, 4), "predicted_candidate": predicted_candidate,
        **{k: round(v, 4) for k, v in pair_ious.items()},
        **{k: (round(v, 2) if v is not None else None) for k, v in pair_centroid_dist.items()},
        "area_ratio_maxmin": round(area_ratio_maxmin, 3), "perim_complexity_spread": round(perim_complexity_spread, 3),
        "max_border_touch_fraction": round(max_border_touch, 4), "max_component_count": max_component_count,
        "min_largest_component_fraction": round(min_largest_component_fraction, 4),
        "max_area_fraction": round(max_area_fraction, 4), "any_implausibly_large": any_implausibly_large, "all_implausibly_small": all_implausibly_small,
        "min_edge_support_fraction": round(min_edge_support_fraction, 4) if min_edge_support_fraction is not None else None,
        "max_mask_vs_bg_contrast": round(max_mask_vs_bg_contrast, 2) if max_mask_vs_bg_contrast is not None else None,
        "min_mask_vs_bg_contrast": round(min_mask_vs_bg_contrast, 2) if min_mask_vs_bg_contrast is not None else None,
    }


def build_individual_panel(tag_prefix: str, well, day, modality, frame_rgb, candidates, out_path: Path) -> None:
    target_w = 480
    scale = target_w / frame_rgb.shape[1]
    target_h = int(frame_rgb.shape[0] * scale)
    frame_small = cv2.resize(frame_rgb, (target_w, target_h))
    tiles = [label_band(frame_small, f"{tag_prefix} {well} D{day} {modality} -- ORIGINAL")]
    for c in candidates[:3]:
        mask_small = cv2.resize(c.mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        ov = overlay_mask(frame_small, mask_small)
        tiles.append(label_band(ov, f"{tag_prefix} {well} D{day} {modality} -- candidate={c.index}"))
    while len(tiles) < 4:
        tiles.append(np.full((target_h + 34, target_w, 3), 255, dtype=np.uint8))
    sheet = np.vstack([np.hstack([tiles[0], tiles[1]]), np.hstack([tiles[2], tiles[3]])])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def build_well_contact_sheet(tag, day_panels, out_path: Path) -> None:
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
    cv2.putText(header, f"{tag} -- acceptability review (no predictions/scores shown)", (6, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    sheet = np.vstack([header, sheet])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


if __name__ == "__main__":
    print("module loaded ok")
