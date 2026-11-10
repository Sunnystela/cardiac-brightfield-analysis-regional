"""Human-QC-driven segmentation failure-source DIAGNOSTIC for P36/Conv.

Connects the project owner's human segmentation scores (1-5 per day, per
well) to the EXISTING SAM2 candidate pool, to diagnose whether a low
score traces to SELECTION_FAILURE (a better candidate already existed in
the pool), PROMPT_LOCALIZATION_FAILURE (the point prompt never landed
near real tissue), GENERATION_FAILURE (SAM2 never produced a usable
candidate for this prompt), MIXED, or UNRESOLVED (needs human visual
confirmation).

STRICT DIAGNOSTIC-ONLY SCOPE, per explicit instruction:
  - No production segmentation rule, SAM2 threshold, or candidate_index
    default is changed anywhere in this script.
  - No morphology/stage threshold is tuned.
  - The functional pipeline is NOT re-run.
  - No canonical result is modified.
  - No parameter is adjusted to make the human scores "come out better."
  - Candidates are REGENERATED read-only (`segment_frame`, the SAME
    single-center-point prompt convention already used in production)
    purely to inspect what was already in the pool at the time of the
    original run -- SAM2 with a fixed point prompt on a fixed frame is
    deterministic, so this reproduces (never redefines) the original
    candidate pool. A determinism check (regenerated candidate_index=1
    area vs the area already stored in records.json) is run for every
    case and disclosed.
  - No "correct" candidate is auto-confirmed as ground truth anywhere --
    clear-cut diagnoses (by disclosed, conservative structural criteria)
    are separated from cases explicitly marked as needing human visual
    review via the generated candidate contact sheets.
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

from cardio_organoid.segmentation.sam2_segmenter import (  # noqa: E402
    connected_component_stats, get_device, load_predictor, provisional_primary_qc, segment_frame,
)
from generic_well_longitudinal_morphology_pilot import load_image_day_frame, load_video_day_midpoint_frame  # noqa: E402
from global_boundary_motion_diagnostic import _load_config  # noqa: E402
from p36_conv_screening_diagnostic import locate_well_paths  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "results/_REVIEW_LATEST/P36_CONV_SCREENING/HUMAN_QC_FAILURE_AUDIT"
PANELS_DIR = OUTPUT_DIR / "representative_candidate_panels"

DAY_ORDER = [1, 3, 4, 5, 7, 8, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
IMAGE_DAYS = {1, 3, 4, 5, 7, 8}

WELLS = ["a1", "a2", "b1", "c2", "d1", "e2", "f2", "g1", "h1"]  # g3 excluded (hold-out well, no human QC this round)

TRANSLUCENT_LOBE_WELLS = {"d1", "g1"}

HUMAN_SCORES = {
    "a2": [5, 1, 5, 5, 5, 5, 5, 4, 5, 5, 5, 3, 3, 4, 4, 5, 5, 2],
    "b1": [5, 5, 5, 5, 5, 5, 5, 4, 3, 5, 5, 4, 4, 4, 4, 4, 3, 3],
    "c2": [4, 3, 5, 3, 5, 5, 5, 1, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5],
    "d1": [5, 3, 5, 5, 4, 5, 3, 3, 5, 5, 3, 3, 4, 3, 3, 3, 3, 2],
    "e2": [5, 5, 5, 4, 4, 5, 5, 4, 5, 4, 4, 5, 5, 4, 5, 4, 5, 1],
    "f2": [5, 5, 5, 4, 5, 5, 4, 5, 5, 5, 5, 4, 5, 4, 4, 3, 3, 3],
    "g1": [5, 5, 5, 4, 5, 5, 4, 3, 3, 3, 3, 5, 3, 3, 3, 3, 3, 3],
    "h1": [5, 5, 5, 5, 5, 5, 4, 4, 5, 4, 5, 3, 5, 5, 5, 5, 5, 4],
    "a1": [5, 5, 5, 5, 5, 5, 4, 5, 3, 3, 4, 3, 4, 5, 5, 3, 5, 4],
}


def score_flags(score) -> dict:
    if score is None or (isinstance(score, str) and score.upper() == "NA"):
        return {"human_usable": None, "human_borderline": None, "human_failure": None}
    return {"human_usable": score >= 4, "human_borderline": score == 3, "human_failure": score <= 2}


# ---------------- TASK 1: human QC dataset ----------------

def build_human_scores_table() -> list[dict]:
    rows = []
    for well in WELLS:
        records_path, _ = locate_well_paths(well)
        records = json.load(open(records_path)) if records_path and records_path.exists() else []
        rec_by_day = {r["day_number"]: r for r in records}
        scores = HUMAN_SCORES[well]
        for day, score in zip(DAY_ORDER, scores):
            rec = rec_by_day.get(day, {})
            modality = "IMAGE" if day in IMAGE_DAYS else "VIDEO"
            row = {
                "batch": "P36", "condition": "Conv", "well": well, "day": day, "modality": modality,
                "human_score": score,
                **score_flags(score),
                "existing_qc_status": rec.get("segmentation_qc_flag"),
                "segmentation_status": rec.get("segmentation_status"),
                "overlay_source_path": rec.get("overlay_image"),
                "area_px": rec.get("area_px"), "circularity": rec.get("circularity"), "solidity": rec.get("solidity"),
            }
            rows.append(row)
    return rows


# ---------------- image helpers ----------------

def load_source_frame(well: str, day: int) -> np.ndarray:
    if day in IMAGE_DAYS:
        frame_rgb, _ = load_image_day_frame("P36 26.08.2026", "conv", well, day)
    else:
        frame_rgb, _, _, _ = load_video_day_midpoint_frame("P36 26.08.2026", "conv", well, day)
    return frame_rgb


def border_touch_fraction(mask: np.ndarray) -> float:
    """Fraction of the image's own border pixels (top+bottom rows,
    left+right columns) that are part of `mask` -- a continuous severity
    proxy for a mask running into (likely truncated by) the frame edge."""
    h, w = mask.shape
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    n_border = int(border.sum())
    return float((mask & border).sum() / n_border) if n_border else 0.0


def render_candidate_overlay(frame_rgb: np.ndarray, mask: np.ndarray | None, color=(0, 255, 0)) -> np.ndarray:
    overlay = frame_rgb.copy()
    if mask is not None:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 5)
    return overlay


def label_panel(img: np.ndarray, text: str, band_h: int = 40) -> np.ndarray:
    band = np.full((band_h, img.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(band, text, (4, band_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([band, img])


def build_candidate_panel(well: str, day: int, frame_rgb: np.ndarray, candidates, selected_index: int, human_score) -> Path:
    target_w = 420
    scale = target_w / frame_rgb.shape[1]
    frame_small = cv2.resize(frame_rgb, (target_w, int(frame_rgb.shape[0] * scale)))
    panels = [label_panel(cv2.resize(render_candidate_overlay(frame_rgb, None), (target_w, int(frame_rgb.shape[0] * scale))), f"D{day} SOURCE (no mask)")]
    for c in candidates:
        mask_small = cv2.resize(c.mask.astype(np.uint8), (target_w, int(frame_rgb.shape[0] * scale)), interpolation=cv2.INTER_NEAREST).astype(bool)
        color = (0, 255, 0) if c.index == selected_index else (0, 165, 255)
        ov = render_candidate_overlay(frame_small, mask_small, color=color)
        tag = f"idx={c.index} score={c.score:.3f} {'[SELECTED]' if c.index == selected_index else ''}"
        panels.append(label_panel(ov, tag))
    max_w = max(p.shape[1] for p in panels)
    max_h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        pad = np.full((max_h, max_w, 3), 255, dtype=np.uint8)
        pad[: p.shape[0], : p.shape[1], :] = p
        padded.append(pad)
    sheet = np.hstack(padded)
    header = np.full((36, sheet.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(header, f"{well} D{day} -- human_score={human_score}", (6, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    sheet = np.vstack([header, sheet])
    PANELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PANELS_DIR / f"{well}_D{day}_candidates.png"
    cv2.imwrite(str(out_path), sheet)
    return out_path


# ---------------- ring-band image statistics (for translucent-lobe subgroup + controls) ----------------

def ring_band_stats(frame_rgb: np.ndarray, mask: np.ndarray, dilate_px: int = 40) -> dict:
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    mask_u8 = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    ring_outer = cv2.dilate(mask_u8, kernel, iterations=dilate_px)
    ring = (ring_outer.astype(bool)) & (~mask)
    far_bg = (~ring_outer.astype(bool))

    def stats_for(region: np.ndarray) -> dict:
        if region.sum() == 0:
            return {"mean_intensity": None, "std_intensity": None, "mean_gradient": None, "n_px": 0}
        vals = gray[region]
        grad_vals = grad_mag[region]
        return {
            "mean_intensity": float(vals.mean()), "std_intensity": float(vals.std()),
            "mean_gradient": float(grad_vals.mean()), "n_px": int(region.sum()),
        }

    core_stats = stats_for(mask)
    ring_stats = stats_for(ring)
    bg_stats = stats_for(far_bg)
    contrast_core_ring = (
        (ring_stats["mean_intensity"] - core_stats["mean_intensity"])
        if core_stats["mean_intensity"] is not None and ring_stats["mean_intensity"] is not None else None
    )
    contrast_ring_bg = (
        (ring_stats["mean_intensity"] - bg_stats["mean_intensity"])
        if ring_stats["mean_intensity"] is not None and bg_stats["mean_intensity"] is not None else None
    )
    return {
        "core_mean_intensity": core_stats["mean_intensity"], "core_std_intensity": core_stats["std_intensity"], "core_mean_gradient": core_stats["mean_gradient"],
        "ring_mean_intensity": ring_stats["mean_intensity"], "ring_std_intensity": ring_stats["std_intensity"], "ring_mean_gradient": ring_stats["mean_gradient"],
        "bg_mean_intensity": bg_stats["mean_intensity"], "bg_std_intensity": bg_stats["std_intensity"], "bg_mean_gradient": bg_stats["mean_gradient"],
        "contrast_core_vs_ring": contrast_core_ring, "contrast_ring_vs_bg": contrast_ring_bg,
        "ring_n_px": ring_stats["n_px"],
    }


def candidate_overlaps_ring(candidate_mask: np.ndarray, selected_mask: np.ndarray, dilate_px: int = 40) -> float:
    """Fraction of the 'ring' band (area just outside the SELECTED mask,
    within dilate_px) that this candidate's OWN mask also covers -- answers
    'does this candidate already extend into the region right around the
    selected boundary' without assuming that region is definitely organoid."""
    kernel = np.ones((3, 3), np.uint8)
    ring_outer = cv2.dilate(selected_mask.astype(np.uint8), kernel, iterations=dilate_px)
    ring = (ring_outer.astype(bool)) & (~selected_mask)
    if ring.sum() == 0:
        return 0.0
    return float((candidate_mask & ring).sum() / ring.sum())


# ---------------- TASK 3/4: candidate audit ----------------

LARGER_CANDIDATE_MARGIN = 1.10  # a candidate must be >=10% larger (relative) to count as "notably larger" -- a generic,
                                 # disclosed margin chosen to exceed ordinary pixel-level noise, NOT fit to these cases


def diagnose_case(selected_qc: dict, candidates, selected_index: int, min_dominant_fraction: float) -> tuple[str, bool, str]:
    """Returns (failure_source, requires_human_visual_confirmation, note).
    Conservative, disclosed criteria only -- no candidate is auto-confirmed
    as the 'correct' organoid mask; every branch below still requires
    visual confirmation via the generated candidate panel.

    Precedence (checked in this order):
      1. PROMPT_LOCALIZATION_FAILURE -- even the largest candidate is tiny.
      2. GENERATION_FAILURE -- no candidate is structurally clean at all.
      3. SELECTION_FAILURE -- a clean candidate other than the selected one
         is either structurally cleaner (selected fails the bar) OR
         notably LARGER in area than the selected one (selected passes the
         bar but a bigger, equally-clean candidate was available and not
         picked) -- this second sub-case is exactly the "candidate exists
         but the fixed candidate_index missed it" question this audit was
         built to answer, and is NOT caught by the structural QC bar alone
         (which only checks single-blob-ness, never completeness/area).
      4. GENERATION_FAILURE (partial) -- selected is already the largest,
         cleanest candidate in the pool; nothing bigger/cleaner exists, so
         a low human score here means no candidate in the pool captured
         the full tissue extent -- a generation-level limit, not a
         selection error.
    """
    max_area_frac = max(c.area_fraction for c in candidates)
    if max_area_frac < 0.02:
        return "PROMPT_LOCALIZATION_FAILURE", True, f"even the largest candidate covers only {max_area_frac:.1%} of the frame -- prompt likely landed away from any large structure"

    def is_clean(c) -> bool:
        return connected_component_stats(c.mask)[1] >= min_dominant_fraction and c.area_fraction >= 0.02

    clean_candidates = [c for c in candidates if is_clean(c)]
    if not clean_candidates:
        return "GENERATION_FAILURE", True, "no candidate in the pool reaches a single-dominant-component structure at all -- SAM2 did not generate a clean single-blob candidate for this prompt"

    selected = candidates[selected_index]
    selected_clean = is_clean(selected)
    other_clean = [c for c in clean_candidates if c.index != selected_index]

    if not selected_clean and other_clean:
        best = max(other_clean, key=lambda c: c.area_fraction)
        return "SELECTION_FAILURE", True, (
            f"selected candidate (idx={selected_index}) fails the single-dominant-component bar "
            f"({selected_qc.get('largest_component_fraction'):.2f}), but candidate idx={best.index} is structurally "
            f"clean ({connected_component_stats(best.mask)[1]:.2f}) and covers {best.area_fraction:.1%} of the frame -- "
            f"NOT auto-confirmed as the correct organoid mask, requires visual check via the candidate panel."
        )

    larger_alternatives = [c for c in other_clean if c.area_fraction >= selected.area_fraction * LARGER_CANDIDATE_MARGIN]
    if larger_alternatives:
        best = max(larger_alternatives, key=lambda c: c.area_fraction)
        pct_larger = (best.area_fraction / selected.area_fraction - 1) * 100 if selected.area_fraction else None
        # STRONG evidence: the larger candidate ALSO has a higher SAM2 confidence score than the
        # selected one -- this means production's OWN existing is_highest_score check would already
        # flag this day "review_needed" (selected is not score-rank-1), yet the FIXED candidate_index
        # convention uses it anyway. WEAK evidence: larger/clean by area+component criteria only, but
        # SAM2 itself still prefers the (smaller) selected candidate -- needs more visual scrutiny,
        # since the larger region could be debris SAM2 correctly down-weighted.
        evidence_strength = "STRONG_matches_existing_score_rank_flag" if best.score > selected.score else "WEAK_area_only_SAM2_still_prefers_selected"
        return "SELECTION_FAILURE", True, (
            f"[{evidence_strength}] selected candidate (idx={selected_index}, area_fraction={selected.area_fraction:.1%}) "
            f"already passes the structural QC bar, but candidate idx={best.index} is ALSO structurally clean "
            f"({connected_component_stats(best.mask)[1]:.2f}) and covers {best.area_fraction:.1%} of the frame -- "
            f"{pct_larger:.0f}% larger, sam2_score {best.score:.3f} vs selected's {selected.score:.3f} -- a bigger, "
            f"equally-clean candidate existed in the pool and was not picked by the fixed candidate_index="
            f"{selected_index} convention. NOT auto-confirmed as the correct organoid mask (could include extra "
            f"debris rather than real tissue) -- requires visual check via the candidate panel."
        )

    return "GENERATION_FAILURE", True, (
        f"selected candidate (idx={selected_index}) is already the largest, cleanest candidate in the pool "
        f"(area_fraction={selected.area_fraction:.1%}) -- no other candidate captures more tissue. A low human "
        f"score here most likely reflects a boundary-completeness issue (e.g. an attached lighter lobe/extension) "
        f"that NO candidate in this prompt's pool includes -- a generation-level limit for this prompt, not a "
        f"pool-selection problem. Requires visual confirmation via the candidate panel."
    )


def run_candidate_audit(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    config = _load_config()
    sam2_cfg = config["segmentation"]["sam2"]
    provisional_cfg = config["segmentation"]["provisional_primary"]
    device = get_device(sam2_cfg["device"])
    predictor = load_predictor(checkpoint_path=REPO_ROOT / sam2_cfg["checkpoint_path"], model_cfg=sam2_cfg["model_cfg"], device=device)
    selected_index = provisional_cfg["candidate_index"]
    min_dominant_fraction = provisional_cfg["min_dominant_component_fraction"]

    audit_rows = []
    case_summaries = []
    low_score_rows = [r for r in rows if r["human_score"] is not None and r["human_score"] <= 3]
    print(f"Auditing {len(low_score_rows)} score<=3 observations ...")
    for i, row in enumerate(low_score_rows):
        well, day = row["well"], row["day"]
        print(f"  [{i+1}/{len(low_score_rows)}] {well} D{day} (human_score={row['human_score']}) ...")
        try:
            frame_rgb = load_source_frame(well, day)
        except (FileNotFoundError, RuntimeError) as exc:
            case_summaries.append({"well": well, "day": day, "human_score": row["human_score"], "failure_source": "UNRESOLVED",
                                    "requires_human_visual_confirmation": True, "note": f"could not reload source frame: {exc}",
                                    "determinism_check_area_match": None})
            continue
        candidates = segment_frame(predictor, frame_rgb)
        qc = provisional_primary_qc(candidates, selected_index, min_dominant_fraction)

        determinism_match = None
        stored_area = row.get("area_px")
        if 0 <= selected_index < len(candidates) and stored_area:
            regen_area = int(candidates[selected_index].mask.sum())
            determinism_match = abs(regen_area - stored_area) <= max(5, 0.001 * stored_area)

        for c in candidates:
            n_comp, largest_frac = connected_component_stats(c.mask)
            overlap_ring = candidate_overlaps_ring(c.mask, candidates[selected_index].mask if 0 <= selected_index < len(candidates) else c.mask)
            audit_rows.append({
                "well": well, "day": day, "human_score": row["human_score"], "candidate_index": c.index,
                "sam2_score": c.score, "area_px": c.area_px, "area_fraction": c.area_fraction,
                "largest_component_fraction": largest_frac, "component_count": n_comp,
                "border_touch_fraction": border_touch_fraction(c.mask),
                "selected": c.index == selected_index,
                "prompt_type": "single_center_point", "prompt_location_xy": f"({frame_rgb.shape[1]//2},{frame_rgb.shape[0]//2})",
                "overlap_with_ring_around_selected": overlap_ring,
            })

        panel_path = build_candidate_panel(well, day, frame_rgb, candidates, selected_index, row["human_score"])
        failure_source, needs_review, note = diagnose_case(qc, candidates, selected_index, min_dominant_fraction)
        evidence_strength = None
        if note.startswith("["):
            evidence_strength = note[1: note.index("]")]

        ring_stats = ring_band_stats(frame_rgb, candidates[selected_index].mask) if 0 <= selected_index < len(candidates) else {}
        case_summaries.append({
            "well": well, "day": day, "human_score": row["human_score"], "modality": row["modality"],
            "failure_source": failure_source, "evidence_strength": evidence_strength,
            "requires_human_visual_confirmation": needs_review, "note": note,
            "determinism_check_area_match": determinism_match, "n_candidates": len(candidates),
            "selected_largest_component_fraction": qc.get("largest_component_fraction"),
            "selected_is_highest_score": qc.get("is_highest_score"),
            "candidate_panel_path": str(panel_path.relative_to(REPO_ROOT)),
            "translucent_lobe_well": well in TRANSLUCENT_LOBE_WELLS,
            **{f"ring_{k}": v for k, v in ring_stats.items()},
        })
    return audit_rows, case_summaries


def run_control_ring_stats(rows: list[dict], n_controls_per_well: int = 3) -> list[dict]:
    """TASK 5: score-5 controls -- same ring-band stats computed for a
    sample of score==5 observations per well (stage/modality-balanced
    where possible), for comparison against the low-score cases -- never
    used to derive a threshold, purely descriptive."""
    config = _load_config()
    sam2_cfg = config["segmentation"]["sam2"]
    provisional_cfg = config["segmentation"]["provisional_primary"]
    device = get_device(sam2_cfg["device"])
    predictor = load_predictor(checkpoint_path=REPO_ROOT / sam2_cfg["checkpoint_path"], model_cfg=sam2_cfg["model_cfg"], device=device)
    selected_index = provisional_cfg["candidate_index"]

    control_rows = []
    by_well: dict[str, list[dict]] = {}
    for r in rows:
        if r["human_score"] == 5:
            by_well.setdefault(r["well"], []).append(r)
    for well, well_rows in by_well.items():
        image_ctrl = [r for r in well_rows if r["modality"] == "IMAGE"][:2]
        video_ctrl = [r for r in well_rows if r["modality"] == "VIDEO"][:2]
        for r in (image_ctrl + video_ctrl)[:n_controls_per_well]:
            day = r["day"]
            try:
                frame_rgb = load_source_frame(well, day)
            except (FileNotFoundError, RuntimeError):
                continue
            candidates = segment_frame(predictor, frame_rgb)
            if not (0 <= selected_index < len(candidates)):
                continue
            ring_stats = ring_band_stats(frame_rgb, candidates[selected_index].mask)
            control_rows.append({"well": well, "day": day, "human_score": 5, "modality": r["modality"],
                                  **{f"ring_{k}": v for k, v in ring_stats.items()}})
    return control_rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- TASK 1 ----------------
    rows = build_human_scores_table()
    with open(OUTPUT_DIR / "human_scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(OUTPUT_DIR / "human_scores.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)

    # ---------------- TASK 2 ----------------
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    def median(xs):
        xs = sorted(x for x in xs if x is not None)
        n = len(xs)
        if n == 0:
            return None
        mid = n // 2
        return xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2

    by_well_summary = []
    for well in WELLS:
        well_rows = [r for r in rows if r["well"] == well]
        scores = [r["human_score"] for r in well_rows if r["human_score"] is not None]
        img_scores = [r["human_score"] for r in well_rows if r["modality"] == "IMAGE" and r["human_score"] is not None]
        vid_scores = [r["human_score"] for r in well_rows if r["modality"] == "VIDEO" and r["human_score"] is not None]
        by_well_summary.append({
            "well": well, "n_days": len(scores), "mean_score": round(mean(scores), 2) if mean(scores) else None,
            "median_score": median(scores), "fraction_4_5": round(sum(1 for s in scores if s >= 4) / len(scores), 2),
            "fraction_le_3": round(sum(1 for s in scores if s <= 3) / len(scores), 2),
            "image_mean_score": round(mean(img_scores), 2) if img_scores and mean(img_scores) else None,
            "video_mean_score": round(mean(vid_scores), 2) if vid_scores and mean(vid_scores) else None,
        })
    with open(OUTPUT_DIR / "score_summary_by_well.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(by_well_summary[0].keys()))
        w.writeheader()
        w.writerows(by_well_summary)

    by_day_summary = []
    for day in DAY_ORDER:
        day_rows = [r for r in rows if r["day"] == day]
        scores = [r["human_score"] for r in day_rows if r["human_score"] is not None]
        by_day_summary.append({
            "day": day, "modality": "IMAGE" if day in IMAGE_DAYS else "VIDEO", "n_wells": len(scores),
            "mean_score": round(mean(scores), 2) if mean(scores) else None, "median_score": median(scores),
            "fraction_4_5": round(sum(1 for s in scores if s >= 4) / len(scores), 2) if scores else None,
            "fraction_le_3": round(sum(1 for s in scores if s <= 3) / len(scores), 2) if scores else None,
            "scores": scores,
        })
    with open(OUTPUT_DIR / "score_summary_by_day.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(by_day_summary[0].keys()))
        w.writeheader()
        w.writerows(by_day_summary)

    # descriptive: does failure fraction increase in later video days? (D13-18 "early video" vs D19-24 "late video")
    early_video_days = [13, 14, 15, 16, 17, 18]
    late_video_days = [19, 20, 21, 22, 23, 24]
    early_scores = [r["human_score"] for r in rows if r["day"] in early_video_days and r["human_score"] is not None]
    late_scores = [r["human_score"] for r in rows if r["day"] in late_video_days and r["human_score"] is not None]
    descriptive_trend = {
        "early_video_D13_18_mean": round(mean(early_scores), 2), "early_video_D13_18_fraction_le_3": round(sum(1 for s in early_scores if s <= 3) / len(early_scores), 2),
        "late_video_D19_24_mean": round(mean(late_scores), 2), "late_video_D19_24_fraction_le_3": round(sum(1 for s in late_scores if s <= 3) / len(late_scores), 2),
        "note": "purely descriptive comparison of two day windows -- no threshold fit, no trend test performed",
    }
    with open(OUTPUT_DIR / "_descriptive_trend_early_vs_late_video.json", "w") as f:
        json.dump(descriptive_trend, f, indent=2)

    # ---------------- TASK 3 + 4 ----------------
    audit_rows, case_summaries = run_candidate_audit(rows)
    with open(OUTPUT_DIR / "low_score_candidate_audit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
        w.writeheader()
        w.writerows(audit_rows)

    with open(OUTPUT_DIR / "_case_diagnoses.json", "w") as f:
        json.dump(case_summaries, f, indent=2, default=str)

    # failure_source_summary.csv (TASK 6) -- SELECTION_FAILURE split by evidence_strength sub-tier
    from collections import Counter
    n_cases = len(case_summaries)

    def group_key(c):
        if c["failure_source"] == "SELECTION_FAILURE" and c.get("evidence_strength"):
            return f"SELECTION_FAILURE ({c['evidence_strength']})"
        return c["failure_source"]

    counts = Counter(group_key(c) for c in case_summaries)
    failure_summary_rows = []
    for source, n in counts.most_common():
        rep = next(c for c in case_summaries if group_key(c) == source)
        failure_summary_rows.append({
            "failure_source": source, "n_cases": n, "fraction": round(n / n_cases, 2),
            "representative_well_day": f"{rep['well']} D{rep['day']}", "evidence": rep["note"],
        })
    with open(OUTPUT_DIR / "failure_source_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["failure_source", "n_cases", "fraction", "representative_well_day", "evidence"])
        w.writeheader()
        w.writerows(failure_summary_rows)

    # translucent_lobe_subgroup.csv (TASK 4) + controls (TASK 5)
    lobe_cases = [c for c in case_summaries if c["translucent_lobe_well"]]
    control_rows = run_control_ring_stats(rows)
    with open(OUTPUT_DIR / "translucent_lobe_subgroup.csv", "w", newline="") as f:
        fieldnames = list(lobe_cases[0].keys()) if lobe_cases else []
        w = csv.DictWriter(f, fieldnames=fieldnames + ["group"])
        w.writeheader()
        for c in lobe_cases:
            w.writerow({**c, "group": "LOW_SCORE_d1_g1"})
    with open(OUTPUT_DIR / "_score5_controls_ring_stats.csv", "w", newline="") as f:
        if control_rows:
            w = csv.DictWriter(f, fieldnames=list(control_rows[0].keys()) + ["group"])
            w.writeheader()
            for c in control_rows:
                w.writerow({**c, "group": "SCORE_5_CONTROL"})

    print(f"\nWrote outputs to {OUTPUT_DIR}")
    print(json.dumps(failure_summary_rows, indent=2, default=str))


if __name__ == "__main__":
    main()
