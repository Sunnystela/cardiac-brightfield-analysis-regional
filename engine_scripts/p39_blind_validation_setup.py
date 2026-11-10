"""Sets up a GENUINE independent (blind) validation of the frozen
candidate-selector on the 26 ALREADY_AVAILABLE P39 observations
(P39/Conv/a3 D0-D12, P39/Slit/a1 D0-D12) -- per explicit instruction,
this round does NOT proceed to the other 69 proposed new-generation
targets, does NOT retrain/tune the frozen selector, and computes/saves
the frozen model's predictions BEFORE any human label exists, so they
can never be adjusted after the fact.

STRICT SCOPE, per explicit instruction:
  - SAM2 is not "re-run" in the sense of new generation -- candidates
    are reproduced read-only via the SAME deterministic single-center-
    point prompt already used for these two wells' existing records.json
    (determinism re-verified here, same convention as every prior
    candidate-review round this project has done).
  - The 69 new-generation active-learning targets are NOT touched here.
  - The frozen selector (results/_REVIEW_LATEST/
    CANDIDATE_SELECTOR_ACTIVE_LEARNING_PLAN/frozen_diagnostic_selector/)
    is loaded and applied AS-IS -- no retraining, no threshold/
    hyperparameter change.
  - No production/config/canonical/source data modified. No commit.
  - No accuracy/generalization number is computed anywhere in this
    script -- there is no human label yet to compare against. Only the
    frozen model's OWN predictions are saved, once, before any human
    label exists (PRE_HUMAN_FROZEN_PREDICTIONS.csv) -- this file must
    never be recomputed or overwritten once human labels arrive.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cardio_organoid.segmentation.sam2_segmenter import connected_component_stats, get_device, load_predictor, segment_frame  # noqa: E402
from generic_well_longitudinal_morphology_pilot import load_image_day_frame, load_video_day_midpoint_frame  # noqa: E402
from global_boundary_motion_diagnostic import _load_config  # noqa: E402
from p36_conv_candidate_selector_feasibility import build_feature_vector, impute_nan, predict_from_scores  # noqa: E402
from p36_conv_candidate_selector_human_labeling import (  # noqa: E402 (reused unchanged, same neutral styling/feature extraction)
    CANDIDATE_COLOR, extract_candidate_features, label_band, overlay_mask,
)

FROZEN_MODEL_PATH = REPO_ROOT / "results/_REVIEW_LATEST/CANDIDATE_SELECTOR_ACTIVE_LEARNING_PLAN/frozen_diagnostic_selector/logistic_regression_selector.joblib"
OUTPUT_DIR = REPO_ROOT / "results/_REVIEW_LATEST/CANDIDATE_SELECTOR_P39_BLIND_VALIDATION"
PANELS_DIR = OUTPUT_DIR / "individual_panels"

WELLS = [
    {"batch": "P39", "condition": "conv", "condition_label": "Conv", "well": "a3",
     "records": REPO_ROOT / "results/_scratch_runs/p39_conv_a3_longitudinal_morphology_pilot_20260921T155121Z/records.json"},
    {"batch": "P39", "condition": "slit", "condition_label": "Slit", "well": "a1",
     "records": REPO_ROOT / "results/_scratch_runs/p39_slit_a1_longitudinal_morphology_pilot_20260922T124514Z/records.json"},
]
BATCH_DIR = "P39 07.09.2026"
IMAGE_DAYS = set(range(0, 8))  # D0-D7 IMAGE, D8-D12 VIDEO for both a3-conv and a1-slit (confirmed from records.json)


def load_source_frame(condition: str, well: str, day: int):
    if day in IMAGE_DAYS:
        frame_rgb, _ = load_image_day_frame(BATCH_DIR, condition, well, day)
    else:
        frame_rgb, _, _, _ = load_video_day_midpoint_frame(BATCH_DIR, condition, well, day)
    return frame_rgb


def build_individual_panel(batch, condition_label, well, day, modality, frame_rgb, candidates, out_path: Path) -> None:
    target_w = 480
    scale = target_w / frame_rgb.shape[1]
    target_h = int(frame_rgb.shape[0] * scale)
    frame_small = cv2.resize(frame_rgb, (target_w, target_h))

    tiles = [label_band(frame_small, f"{batch} {condition_label} {well} D{day} {modality} -- ORIGINAL")]
    for c in candidates[:3]:
        mask_small = cv2.resize(c.mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        ov = overlay_mask(frame_small, mask_small)
        tiles.append(label_band(ov, f"{batch} {condition_label} {well} D{day} {modality} -- candidate={c.index}"))
    while len(tiles) < 4:
        tiles.append(np.full((target_h + 34, target_w, 3), 255, dtype=np.uint8))

    top = np.hstack([tiles[0], tiles[1]])
    bottom = np.hstack([tiles[2], tiles[3]])
    sheet = np.vstack([top, bottom])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def build_well_contact_sheet(batch, condition_label, well, day_panels: list[tuple], out_path: Path) -> None:
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
    cv2.putText(header, f"{batch} {condition_label} {well} -- ALL D0-D12, candidate review (no predictions/scores shown)", (6, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    sheet = np.vstack([header, sheet])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def entropy_from_probs(probs: dict[int, float]) -> float:
    vals = np.clip(np.array(list(probs.values()), dtype=float), 1e-9, None)
    p = vals / vals.sum()
    ent = -np.sum(p * np.log(p))
    max_ent = np.log(len(p))
    return float(ent / max_ent) if max_ent > 0 else 0.0


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PANELS_DIR.mkdir(parents=True, exist_ok=True)

    frozen = joblib.load(FROZEN_MODEL_PATH)
    model, scaler, fill_values, feature_columns = frozen["model"], frozen["scaler"], frozen["fill_values"], frozen["feature_columns"]

    config = _load_config()
    sam2_cfg = config["segmentation"]["sam2"]
    device = get_device(sam2_cfg["device"])
    predictor = load_predictor(checkpoint_path=REPO_ROOT / sam2_cfg["checkpoint_path"], model_cfg=sam2_cfg["model_cfg"], device=device)

    prediction_rows = []
    label_template_rows = []
    idx_counter = 0

    for well_cfg in WELLS:
        records = json.load(open(well_cfg["records"]))
        records_by_day = {r["day_number"]: r for r in records}
        day_panels = []
        for day in range(0, 13):
            rec = records_by_day[day]
            modality = rec["modality"]
            print(f"  {well_cfg['batch']} {well_cfg['condition_label']} {well_cfg['well']} D{day} ({modality}) ...")
            frame_rgb = load_source_frame(well_cfg["condition"], well_cfg["well"], day)
            candidates = segment_frame(predictor, frame_rgb)

            # determinism check against the already-existing stored area (same check used every prior round)
            selected_index = config["segmentation"]["provisional_primary"]["candidate_index"]
            stored_area = rec.get("area_px")
            regen_area = int(candidates[selected_index].mask.sum()) if 0 <= selected_index < len(candidates) else None
            determinism_match = (abs(regen_area - stored_area) <= max(5, 0.001 * stored_area)) if (stored_area and regen_area is not None) else None

            idx_counter += 1
            out_path = PANELS_DIR / f"P39_{well_cfg['condition_label']}_{well_cfg['well']}_D{day}_CANDIDATE_REVIEW.png"
            build_individual_panel(well_cfg["batch"], well_cfg["condition_label"], well_cfg["well"], day, modality, frame_rgb, candidates, out_path)
            day_panels.append((day, modality, frame_rgb, candidates))

            # ---- frozen model prediction (feature extraction reused unchanged from the human-labeling round) ----
            feat_rows = extract_candidate_features(well_cfg["well"], day, modality, frame_rgb, candidates, selected_index)
            feat_by_idx = {r["candidate_index"]: r for r in feat_rows}
            probs = {}
            for cidx in sorted(feat_by_idx):
                vec = []
                for col in feature_columns:
                    if col == "modality_is_video":
                        vec.append(1.0 if modality == "VIDEO" else 0.0)
                    else:
                        v = feat_by_idx[cidx].get(col)
                        vec.append(np.nan if v is None else float(v))
                Xc = np.array([vec], dtype=float)
                Xc_imp, _ = impute_nan(Xc, fill_values)
                p = model.predict_proba(scaler.transform(Xc_imp))[0, 1]
                probs[cidx] = float(p)

            predicted = predict_from_scores(probs)
            sorted_probs = sorted(probs.values(), reverse=True)
            top1 = sorted_probs[0]
            margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else None
            disagreement = entropy_from_probs(probs)
            sam2_scores = {c.index: c.score for c in candidates}
            sam2_choice = predict_from_scores(sam2_scores)

            prediction_rows.append({
                "batch": well_cfg["batch"], "condition": well_cfg["condition_label"], "well": well_cfg["well"],
                "day": day, "modality": modality,
                "candidate_0_score": round(probs.get(0, float("nan")), 4),
                "candidate_1_score": round(probs.get(1, float("nan")), 4),
                "candidate_2_score": round(probs.get(2, float("nan")), 4),
                "predicted_candidate": predicted, "top1_score": round(top1, 4),
                "top1_top2_margin": round(margin, 4) if margin is not None else None,
                "normalized_candidate_disagreement_entropy": round(disagreement, 4),
                "highest_sam2_score_candidate": sam2_choice,
                "learned_vs_sam2_score_disagreement": predicted != sam2_choice,
                "fixed_index_1_prediction": 1,
                "learned_vs_fixed_index_disagreement": predicted != 1,
                "possible_abstain_flag_diagnostic_only": top1 < 0.5,
                "determinism_check_area_match": determinism_match,
            })
            label_template_rows.append({
                "index": idx_counter, "batch": well_cfg["batch"], "condition": well_cfg["condition_label"],
                "well": well_cfg["well"], "day": day, "modality": modality,
                "human_best_candidate": "", "error_direction": "", "error_extent": "", "phenotype_note": "", "human_comment": "",
            })

        sheet_path = OUTPUT_DIR / f"P39_{well_cfg['condition_label']}_{well_cfg['well']}_CANDIDATE_REVIEW_CONTACT_SHEET.png"
        build_well_contact_sheet(well_cfg["batch"], well_cfg["condition_label"], well_cfg["well"], day_panels, sheet_path)
        print(f"  Wrote contact sheet: {sheet_path}")

    pred_path = OUTPUT_DIR / "PRE_HUMAN_FROZEN_PREDICTIONS.csv"
    if pred_path.exists():
        raise RuntimeError(
            f"{pred_path} already exists -- refusing to overwrite a PRE_HUMAN_FROZEN_PREDICTIONS file. "
            "This file must be written exactly once, before any human label exists, and never recomputed afterward."
        )
    with open(pred_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prediction_rows[0].keys()))
        w.writeheader()
        w.writerows(prediction_rows)
    print(f"\nWrote {pred_path} (26 rows, frozen predictions, saved BEFORE any human label)")

    with open(OUTPUT_DIR / "P39_26_HUMAN_LABEL_TEMPLATE.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(label_template_rows[0].keys()))
        w.writeheader()
        w.writerows(label_template_rows)
    print(f"Wrote {OUTPUT_DIR / 'P39_26_HUMAN_LABEL_TEMPLATE.csv'} (26 blank rows)")

    # ---- verification summary (no accuracy/generalization computed) ----
    n_det_ok = sum(1 for r in prediction_rows if r["determinism_check_area_match"] is True)
    print(f"\nDeterminism check: {n_det_ok}/{len(prediction_rows)} candidate_index=1 areas match existing records.json exactly")
    print(f"Panels written: {len(list(PANELS_DIR.glob('*.png')))}")
    print(f"Contact sheets written: 2")


if __name__ == "__main__":
    main()
