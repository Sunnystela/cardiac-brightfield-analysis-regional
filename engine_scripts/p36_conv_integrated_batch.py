"""P36/Conv integrated batch runner -- FAIL-CLOSED, segmentation-interface driven.

This round implements only:
  segment       SAM2 candidates + frozen selector (read-only) + interface record
                + owner-approved reconciliation, for every P36/Conv observation
  review_queue  human review materials for REVIEW_REQUIRED observations only
  eligibility   downstream eligibility preview (what morphology/functional
                WOULD consume; nothing is analysed)

Fail-closed: every stage runs inside segmentation_interface.fail_closed_mode(),
so any legacy fixed candidate_index selection raises InterfaceFailClosed
instead of silently using configs candidate_index. Downstream consumers must
call require_downstream_mask(), which only returns a mask for a valid ACCEPT
record. Morphology/functional batch stages are deliberately not implemented
in this runner yet.

Every run writes run_manifest.json (git commit/dirty state + code snapshot,
config hash, interface version, selector artifact hash, SAM2 checkpoint).
Diagnostic namespace only; no canonical write, no config change, no raw-data
change. P38 untouched.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cardio_organoid.pipeline.run_manifest import write_run_manifest  # noqa: E402
from cardio_organoid.pipeline.segmentation_interface import (  # noqa: E402
    INTERFACE_VERSION, QC_ACCEPT, QC_NONE, QC_REVIEW, InterfaceFailClosed, SegmentationRecord,
    fail_closed_mode, reconcile_status, require_downstream_mask, save_mask, validate_record,
)

from _output_safety import add_output_dir_args, resolve_output_dir  # noqa: E402

BATCH, CONDITION = "P36", "Conv"
BATCH_DIR, CONDITION_DIR = "P36 26.08.2026", "conv"
CONFIG_PATH = REPO_ROOT / "configs/default.yaml"
R = REPO_ROOT / "results/_REVIEW_LATEST"
IDENTITY = REPO_ROOT / "results/_scratch_runs/longitudinal_identity_finalization_diagnostic_20260921T090140Z/longitudinal_mapping_confirmed.csv"
HUMAN_SCORES = R / "P36_CONV_SCREENING/HUMAN_QC_FAILURE_AUDIT/human_scores.csv"
LABELING_CANDIDATE_FEATURES = R / "P36_CONV_SCREENING/CANDIDATE_SELECTOR_HUMAN_LABELING/candidate_features.csv"
HARMONIZED_LABELS = REPO_ROOT / "results/_scratch_runs/stage1_feature_harmonization_20260928T112449Z/STAGE1_HARMONIZED_FEATURE_DATASET_124.csv"
SELECTOR_ARTIFACT = R / "CANDIDATE_SELECTOR_ACTIVE_LEARNING_PLAN/frozen_diagnostic_selector/logistic_regression_selector.joblib"
EXISTING_FUNCTIONAL_WELLS = {"a1", "g3", "a2", "b1", "c2", "d1", "e2", "f2", "g1", "h1"}  # P36/Conv screening round


def rel(p) -> str:
    return str(Path(p).resolve().relative_to(REPO_ROOT))


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def observations():
    """Every observed P36/Conv (well, day, modality) from the confirmed identity table."""
    obs = []
    for r in read_csv(IDENTITY):
        if r["batch"] != BATCH or r["condition"] != CONDITION:
            continue
        img, vid = set(json.loads(r["days_with_image"])), set(json.loads(r["days_with_video"]))
        for d in json.loads(r["day_numbers_observed"]):
            obs.append((r["well_id"], int(d), "IMAGE" if d in img else "VIDEO" if d in vid else "UNKNOWN"))
    return sorted(obs)


def manifest(out: Path, stage: str, extra=None):
    import yaml
    cfg = yaml.safe_load(CONFIG_PATH.read_text())["segmentation"]["sam2"]
    write_run_manifest(out / "manifests" / stage, REPO_ROOT, config_path=CONFIG_PATH, interface_version=INTERFACE_VERSION,
                       selector_artifact=SELECTOR_ARTIFACT, sam2_checkpoint=REPO_ROOT / cfg["checkpoint_path"],
                       sam2_model_cfg=cfg["model_cfg"],
                       input_scope={"batch": BATCH, "condition": CONDITION, "n_observations": len(observations()),
                                    "stage": stage, "p38": "untouched"},
                       extra=extra)


# ------------------------------------------------------------------ segment
def stage_segment(out: Path):
    from cardio_organoid.segmentation.sam2_segmenter import get_device, load_predictor, segment_frame
    from generic_well_longitudinal_morphology_pilot import load_image_day_frame, load_video_day_midpoint_frame
    from global_boundary_motion_diagnostic import _load_config
    from p39_abstention_diagnostic import compute_stage1_features  # frozen selector, read-only

    manifest(out, "segment")
    config = _load_config()
    sam2_cfg = config["segmentation"]["sam2"]
    predictor = load_predictor(checkpoint_path=REPO_ROOT / sam2_cfg["checkpoint_path"], model_cfg=sam2_cfg["model_cfg"],
                               device=get_device(sam2_cfg["device"]))
    labels = {(r["well"], int(r["day"])): r for r in read_csv(HARMONIZED_LABELS) if r["batch"] == BATCH and r["condition"] == CONDITION}
    scores = {(r["well"], int(r["day"])): r for r in read_csv(HUMAN_SCORES)}
    lab_areas = defaultdict(dict)
    for r in read_csv(LABELING_CANDIDATE_FEATURES):
        lab_areas[(r["well"], int(r["day"]))][int(r["candidate_index"])] = int(float(r["area_px"]))

    recs, aux, failures = [], {}, []
    for well, day, modality in observations():
        try:
            if modality == "IMAGE":
                frame, src = load_image_day_frame(BATCH_DIR, CONDITION_DIR, well, day)
                ref = "single_image"
            elif modality == "VIDEO":
                frame, src, ri, fc = load_video_day_midpoint_frame(BATCH_DIR, CONDITION_DIR, well, day)
                ref = f"video_midpoint_frame_index={ri}/frame_count={fc}"
            else:
                raise RuntimeError("unknown modality")
        except (FileNotFoundError, RuntimeError) as exc:
            failures.append({"well": well, "day": day, "modality": modality, "error": str(exc)})
            continue
        cands = segment_frame(predictor, frame)
        for c in cands:
            save_mask(c.mask, out / "masks" / well / "candidates" / f"D{day}_candidate{c.index}.npz")
        s1 = compute_stage1_features(cands, frame, modality)
        pred = int(s1["predicted_candidate"])
        areas = {c.index: int(c.mask.sum()) for c in cands}
        lab, sc = labels.get((well, day)), scores.get((well, day))
        same_mask = (areas.get(1) == int(sc["area_px"])) if sc else None
        lab_match = all(areas.get(i) == a for i, a in lab_areas[(well, day)].items()) if lab else None
        sel, status, reason, prov = reconcile_status(
            lab["human_best_candidate"] if lab else None, lab["legacy_category"] if lab else None, pred,
            int(sc["human_score"]) if sc else None, same_mask, lab_match)
        rec = SegmentationRecord(batch=BATCH, condition=CONDITION, well=well, day=day, modality=modality,
                                 selected_candidate=sel, selector_top1_probability=float(s1["top1_prob"]),
                                 selector_margin=float(s1["top1_top2_margin"]), segmentation_qc_status=status,
                                 qc_reason=reason, mask_provenance=prov, mask_path=None)
        if sel is not None:
            p = out / "masks" / well / "selected" / f"D{day}_selected_candidate{sel}.npz"
            save_mask(cands[sel].mask, p)
            rec.mask_path = rel(p)
        errs = validate_record(rec)
        if errs:
            raise InterfaceFailClosed(f"{well} D{day}: invalid record {errs}")
        recs.append(rec)
        aux[f"{well}_D{day}"] = {"source_file": rel(src), "reference_frame": ref, "frame_shape": list(frame.shape[:2]),
                                 "candidate_areas_px": areas, "selector_predicted_candidate": pred,
                                 "historical_idx1_score": int(sc["human_score"]) if sc else None,
                                 "historical_idx1_same_mask": same_mask, "candidates_match_labeling_run": lab_match,
                                 "owner_label": lab["human_best_candidate"] if lab else None,
                                 "owner_category": lab["legacy_category"] if lab else None,
                                 "stage1_features": s1}
        print(f"  {well} D{day} {modality}: {status} sel={sel} ({reason})", flush=True)
    write_json(out / "segmentation_interface_records.json", {"interface_version": INTERFACE_VERSION, "records": [r.to_dict() for r in recs]})
    write_json(out / "segmentation_aux.json", aux)
    write_json(out / "segmentation_failures.json", failures)


def load_records(out):
    return [SegmentationRecord.from_dict(d) for d in json.loads((out / "segmentation_interface_records.json").read_text())["records"]]


# ------------------------------------------------------------------ review queue
BAND = {"owner_label": (230, 120, 120), "historical_other": (245, 190, 110), "no_evidence": (250, 235, 150),
        "mismatch": (190, 160, 230), "none_ref": (200, 60, 60)}


def review_class(rec):
    r = rec.qc_reason
    if r.startswith("owner_label_") and r.endswith("_not_accepted"):
        return "owner_label"  # low-quality-but-best / ambiguous -- hard cases
    if r.startswith("historical_"):
        return "historical_other"
    if "mismatch" in r:
        return "mismatch"
    return "no_evidence"


def _tile(frame, mask, text, w=300, highlight=False):
    small = cv2.resize(frame, (w, int(frame.shape[0] * w / frame.shape[1])))
    if mask is not None:
        m = cv2.resize(mask.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        ov = small.astype(np.float32)
        ov[m] = ov[m] * 0.6 + np.array([0, 200, 255], np.float32) * 0.4
        small = ov.astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(small, cnts, -1, (0, 200, 255), 2)
    if highlight:
        cv2.rectangle(small, (0, 0), (small.shape[1] - 1, small.shape[0] - 1), (220, 0, 220), 6)
    band = np.full((24, small.shape[1], 3), 255, np.uint8)
    cv2.putText(band, text, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([band, small])


def stage_review_queue(out: Path):
    from cardio_organoid.pipeline.segmentation_interface import load_mask
    from generic_well_longitudinal_morphology_pilot import load_image_day_frame, load_video_day_midpoint_frame
    manifest(out, "review_queue")
    recs, aux = load_records(out), json.loads((out / "segmentation_aux.json").read_text())
    queue = [r for r in recs if r.segmentation_qc_status == QC_REVIEW]
    none_ref = [r for r in recs if r.segmentation_qc_status == QC_NONE]
    rows = []
    by_well = defaultdict(list)
    for r in sorted(queue + none_ref, key=lambda x: (x.well, x.day)):
        by_well[r.well].append(r)
    for well, items in by_well.items():
        sheet_rows = []
        for r in items:
            a = aux[f"{r.well}_D{r.day}"]
            frame = (load_image_day_frame(BATCH_DIR, CONDITION_DIR, r.well, r.day)[0] if r.modality == "IMAGE"
                     else load_video_day_midpoint_frame(BATCH_DIR, CONDITION_DIR, r.well, r.day)[0])
            tiles = [_tile(frame, None, f"D{r.day} {r.modality} ORIGINAL")]
            for k in (0, 1, 2):
                m = load_mask(out / "masks" / r.well / "candidates" / f"D{r.day}_candidate{k}.npz")
                tiles.append(_tile(frame, m, f"candidate {k}" + ("  <- selector" if k == a["selector_predicted_candidate"] else ""),
                                   highlight=(k == a["selector_predicted_candidate"])))
            row = np.hstack(tiles)
            cls = "none_ref" if r.segmentation_qc_status == QC_NONE else review_class(r)
            ev = []
            if a["owner_label"] is not None:
                ev.append(f"owner label={a['owner_label']} ({a['owner_category']})")
            if a["historical_idx1_score"] is not None:
                ev.append(f"historical idx1 score={a['historical_idx1_score']} (score of candidate 1 only)")
            line1 = f"{BATCH}/{CONDITION}/{r.well} D{r.day} {r.modality} | {r.segmentation_qc_status} | {r.qc_reason}"
            line2 = (f"selector={a['selector_predicted_candidate']} top1={r.selector_top1_probability:.3f} "
                     f"margin={r.selector_margin:.3f} | evidence: {'; '.join(ev) or 'none'}")
            head = np.full((48, row.shape[1], 3), BAND[cls], np.uint8)
            cv2.putText(head, line1, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(head, line2, (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            sheet_rows.append(np.vstack([head, row]))
            rows.append({"batch": BATCH, "condition": CONDITION, "well": r.well, "day": r.day, "modality": r.modality,
                         "segmentation_qc_status": r.segmentation_qc_status, "review_class": cls, "qc_reason": r.qc_reason,
                         "selector_prediction": a["selector_predicted_candidate"],
                         "selector_top1_probability": r.selector_top1_probability, "selector_margin": r.selector_margin,
                         "owner_label": a["owner_label"], "owner_category": a["owner_category"],
                         "historical_idx1_score": a["historical_idx1_score"],
                         "in_review_queue": r.segmentation_qc_status == QC_REVIEW,
                         "owner_decision": "", "owner_candidate": "", "owner_comment": ""})
        maxw = max(x.shape[1] for x in sheet_rows)
        sheet = np.vstack([np.hstack([x, np.full((x.shape[0], maxw - x.shape[1], 3), 255, np.uint8)]) for x in sheet_rows])
        d = out / "review_queue" / "well_sheets"
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"{BATCH}_{CONDITION}_{well}_REVIEW_QUEUE.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    with open(out / "review_queue" / "REVIEW_QUEUE.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    legend = {"band colours": {"red (owner_label)": "owner best candidate exists but LOW_QUALITY / AMBIGUOUS -- hard case",
                               "orange (historical_other)": "historical idx1 score >=4 but selector picked another candidate",
                               "yellow (no_evidence)": "no human evidence",
                               "purple (mismatch)": "candidate regeneration mismatch",
                               "dark red (none_ref)": "NO_ACCEPTABLE_CANDIDATE -- reference only, NOT in the review queue"},
              "magenta frame": "selector prediction",
              "owner_decision vocabulary": ["ACCEPT_SELECTOR_PICK", "ACCEPT_OTHER_CANDIDATE (fill owner_candidate)", "NONE",
                                            "KEEP_REVIEW (ambiguous)"]}
    write_json(out / "review_queue" / "LEGEND.json", legend)
    print(f"review queue: {len(queue)} REVIEW_REQUIRED + {len(none_ref)} NONE reference rows")


# ------------------------------------------------------------------ eligibility preview
def stage_eligibility(out: Path):
    manifest(out, "eligibility")
    recs = load_records(out)
    rows, blocked = [], Counter()
    for r in recs:
        try:
            require_downstream_mask(r, REPO_ROOT, "morphology")
            morph = "ELIGIBLE"
        except InterfaceFailClosed:
            morph = "BLOCKED"
            blocked[r.segmentation_qc_status] += 1
        func = "NOT_APPLICABLE_IMAGE"
        if r.modality == "VIDEO":
            if morph != "ELIGIBLE":
                func = "BLOCKED_" + r.segmentation_qc_status
            elif r.selected_candidate == 1 and r.well in EXISTING_FUNCTIONAL_WELLS:
                func = "REUSE_CANDIDATE_pending_provenance_check"
            else:
                func = "RERUN_EXPECTED"
        rows.append({"well": r.well, "day": r.day, "modality": r.modality, "status": r.segmentation_qc_status,
                     "selected_candidate": r.selected_candidate, "morphology": morph, "functional_preview": func})
    write_json(out / "downstream_eligibility_preview.json", rows)
    print({"morphology": dict(Counter(x["morphology"] for x in rows)),
           "functional_preview": dict(Counter(x["functional_preview"] for x in rows))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["segment", "review_queue", "eligibility"])
    add_output_dir_args(ap, REPO_ROOT / "results/p36_conv_integrated_batch", "p36_conv_integrated_batch")
    args = ap.parse_args()
    out = resolve_output_dir(args, REPO_ROOT / "results/p36_conv_integrated_batch", "p36_conv_integrated_batch")
    out.mkdir(parents=True, exist_ok=True)
    with fail_closed_mode():
        {"segment": stage_segment, "review_queue": stage_review_queue, "eligibility": stage_eligibility}[args.stage](out)


if __name__ == "__main__":
    main()
