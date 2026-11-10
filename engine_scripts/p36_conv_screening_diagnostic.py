"""P36/Conv cross-well DIAGNOSTIC SCREENING using the FROZEN v4
modality-aware architecture (prototype_trajectory_stage_model_v4_
modality_aware.py, unchanged since the a1/g3 rounds -- no threshold was
edited after seeing any well's own result, including this round's).

Purpose: NOT to produce a production result, but to see (1) what
phenomena repeat across wells, (2) what failure modes repeat, and (3)
how far the current rule-based stage model can be trusted -- per
explicit instruction.

Per explicit instruction:
  - IMAGE and VIDEO morphology streams stay fully separate throughout
    (summarize_image_stream is purely descriptive; VIDEO-only causal/
    retrospective/episode/recovery evidence never uses an IMAGE
    baseline).
  - functional_status/BPM is NEVER fed into the morphology stage
    classifier -- it is computed independently (analyze_video_functional,
    the existing, unchanged functional pipeline) and compared only at
    the final per-well integration step, as descriptive evidence.
  - No threshold in v4 changed. Nothing tuned by looking at any well's
    own result this round.
  - No production/config/canonical change. No commit. No automatic
    segmentation "fix".

QC DISCLOSURE (important limitation of this screening round): a1 and g3
received a FULL, day-by-day MANUAL visual audit of every overlay frame
(see their own dedicated diagnostic rounds). The other 8 wells here
(a2,b1,c2,d1,e2,f2,g1,h1) are screened using the AUTOMATIC SAM2
structural QC flag only (segmentation_qc_flag: "ok" vs "review_needed")
-- a materially weaker signal, known from the a1/g3 rounds to disagree
with manual review on some days in both directions. This is a
deliberate, disclosed scope reduction for a screening round (10 wells
picked from the P36/Conv inventory, one per grid row a-h, at the
project owner's own explicit approval of a representative-subset scope
over a full 21-well run) -- not a claim that automatic-flag screening is
equivalent to manual audit.
"""

from __future__ import annotations

import csv
import dataclasses
import glob
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cardio_organoid.pipeline.prototype_trajectory_stage_model_v4_modality_aware import (  # noqa: E402
    RECOVERY_NOT_SUPPORTED,
    STATE_BASELINE_INSUFFICIENT, STATE_TRANSITIONAL_CANDIDATE,
    analyze_video_stream, assess_recovery_v4, group_episodes_v4, quantify_modality_shift,
    refine_peak_day, summarize_image_stream,
)
from well_trajectory_report_lib import build_stage_timeline  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "results/_REVIEW_LATEST/P36_CONV_SCREENING"
FIGURES_DIR = OUTPUT_DIR / "figures"

IMAGE_DAYS = [1, 3, 4, 5, 7, 8]
VIDEO_DAYS_ALL = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

# a1 and g3's own FULL manual visual QC audits (from their dedicated rounds), reused verbatim, unchanged.
MANUAL_VIDEO_QC = {
    "a1": {13: "OK", 14: "OK", 15: "FLAGGED", 16: "FLAGGED", 17: "OK", 18: "FLAGGED", 19: "OK", 20: "OK", 21: "OK", 22: "FLAGGED", 23: "OK", 24: "FLAGGED"},
    "g3": {13: "OK", 14: "OK", 15: "OK", 16: "FLAGGED", 17: "OK", 18: "OK", 19: "OK", 20: "OK", 21: "FLAGGED", 22: "OK", 23: "OK", 24: "OK"},
}

WELLS = ["a1", "g3", "a2", "b1", "c2", "d1", "e2", "f2", "g1", "h1"]


def find_latest(pattern: str) -> Path | None:
    matches = sorted(glob.glob(str(REPO_ROOT / pattern)))
    return Path(matches[-1]) if matches else None


def locate_well_paths(well: str) -> tuple[Path, Path]:
    if well == "a1":
        records = REPO_ROOT / "results/_scratch_runs/p36_conv_a1_longitudinal_morphology_pilot_20260924T070224Z/records.json"
        functional = REPO_ROOT / "results/_scratch_runs/p36_conv_a1_functional_onset_diagnostic_20260924T070648Z/functional_onset_results.json"
        return records, functional
    if well == "g3":
        records = REPO_ROOT / "results/_scratch_runs/p36_conv_g3_v4_holdout_diagnostic_20260924T090334Z/records.json"
        functional = REPO_ROOT / "results/_scratch_runs/p36_conv_g3_v4_holdout_diagnostic_20260924T090338Z/functional_onset_results.json"
        return records, functional
    morph_dir = find_latest(f"results/_scratch_runs/p36_conv_{well}_screening_*/records.json")
    func_dir = find_latest(f"results/_scratch_runs/p36_conv_{well}_screening_*/functional_onset_results.json")
    return morph_dir, func_dir


def video_qc_for(well: str, records: list[dict]) -> tuple[dict, str]:
    if well in MANUAL_VIDEO_QC:
        return MANUAL_VIDEO_QC[well], "MANUAL_VISUAL_FULL_AUDIT"
    qc = {}
    for r in records:
        if r["modality"] != "VIDEO":
            continue
        qc[r["day_number"]] = "OK" if r.get("segmentation_qc_flag") == "ok" else "FLAGGED"
    return qc, "AUTOMATIC_FLAG_ONLY"


def _adapter(day_result, which):
    @dataclasses.dataclass
    class _Adapted:
        day_number: int
        stage: str
        confidence: str | None
    return _Adapted(day_result.day_number, getattr(day_result, f"{which}_state"), getattr(day_result, f"{which}_confidence"))


def process_well(well: str) -> dict | None:
    records_path, functional_path = locate_well_paths(well)
    if records_path is None or not records_path.exists():
        return {"well": well, "status": "NO_MORPHOLOGY_DATA"}
    records = json.load(open(records_path))
    rec_by_day = {r["day_number"]: r for r in records}
    image_records = [rec_by_day[d] for d in IMAGE_DAYS if d in rec_by_day]
    video_qc, qc_source = video_qc_for(well, records)
    ok_days = sorted(d for d, s in video_qc.items() if s == "OK")
    flagged_days = sorted(d for d, s in video_qc.items() if s == "FLAGGED")

    functional = json.load(open(functional_path)) if functional_path and functional_path.exists() else []
    func_by_day = {f["day_number"]: f for f in functional}
    onset_candidates = sorted(f["day_number"] for f in functional if f["functional_status"] in ("LOW_CONFIDENCE", "ESTIMABLE"))
    functional_onset_day = onset_candidates[0] if onset_candidates else None

    day_rows = []
    for d in IMAGE_DAYS:
        if d not in rec_by_day:
            continue
        r = rec_by_day[d]
        day_rows.append({
            "well": well, "day": d, "modality": "IMAGE", "segmentation_status": r.get("segmentation_status"),
            "qc": "N/A (halo-exclusion pattern, project-wide, not separately screened per well this round)",
            "usable_for_morphology": True,  # descriptive IMAGE stream only, never a VIDEO baseline
            "area_px": r.get("area_px"), "circularity": r.get("circularity"), "solidity": r.get("solidity"), "radial_cv": r.get("radial_cv"),
            "functional_status": None, "bpm": None,
        })
    for d in VIDEO_DAYS_ALL:
        if d not in rec_by_day:
            continue
        r = rec_by_day[d]
        f = func_by_day.get(d, {})
        day_rows.append({
            "well": well, "day": d, "modality": "VIDEO", "segmentation_status": r.get("segmentation_status"),
            "qc": video_qc.get(d, "NOT_SCREENED"), "usable_for_morphology": video_qc.get(d) == "OK",
            "area_px": r.get("area_px"), "circularity": r.get("circularity"), "solidity": r.get("solidity"), "radial_cv": r.get("radial_cv"),
            "functional_status": f.get("functional_status"), "bpm": f.get("bpm"),
        })

    if len(ok_days) < 3:
        return {"well": well, "status": "TOO_FEW_QC_CLEAN_VIDEO_DAYS", "n_ok_video_days": len(ok_days),
                "qc_source": qc_source, "day_rows": day_rows, "functional_onset_day": functional_onset_day}

    video_ok_records = [dict(rec_by_day[d], _qc_state="OK") for d in ok_days]
    results = analyze_video_stream(video_ok_records, functional_onset_day)

    causal_eps = group_episodes_v4(results, basis="causal")
    retro_eps = group_episodes_v4(results, basis="retrospective")
    rbd = {r["day_number"]: r for r in video_ok_records}
    for ep in causal_eps + retro_eps:
        refine_peak_day(ep, rbd)

    def best_episode(eps):
        validated = [e for e in eps if e.status == "VALIDATED"]
        return validated[0] if validated else None

    main_causal, main_retro = best_episode(causal_eps), best_episode(retro_eps)
    recovery_causal = assess_recovery_v4(video_ok_records, main_causal)
    recovery_retro = assess_recovery_v4(video_ok_records, main_retro)

    modality_shift = quantify_modality_shift(image_records, video_ok_records) if image_records else {}

    return {
        "well": well, "status": "ANALYZED", "qc_source": qc_source,
        "n_ok_video_days": len(ok_days), "n_flagged_video_days": len(flagged_days),
        "usable_video_fraction": len(ok_days) / len(VIDEO_DAYS_ALL),
        "functional_onset_day": functional_onset_day,
        "day_rows": day_rows, "video_results": results,
        "causal_episodes": causal_eps, "retrospective_episodes": retro_eps,
        "main_causal_episode": main_causal, "main_retrospective_episode": main_retro,
        "recovery_causal": recovery_causal, "recovery_retrospective": recovery_retro,
        "modality_shift": modality_shift,
    }


def qc_flagged_fraction(days, video_qc):
    return sum(1 for d in days if video_qc.get(d) == "FLAGGED") / len(days) if days else 0.0


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    well_results = {w: process_well(w) for w in WELLS}

    # ---------------- P36_CONV_SCREENING_OVERVIEW.csv ----------------
    with open(OUTPUT_DIR / "P36_CONV_SCREENING_OVERVIEW.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["well", "status", "qc_source", "n_ok_video_days", "n_flagged_video_days", "usable_video_fraction", "functional_onset_day"])
        for well, r in well_results.items():
            w.writerow([well, r.get("status"), r.get("qc_source"), r.get("n_ok_video_days"), r.get("n_flagged_video_days"),
                        round(r["usable_video_fraction"], 2) if r.get("usable_video_fraction") is not None else None, r.get("functional_onset_day")])

    # ---------------- P36_CONV_ORGANOID_DAY_TABLE.csv ----------------
    with open(OUTPUT_DIR / "P36_CONV_ORGANOID_DAY_TABLE.csv", "w", newline="") as f:
        fieldnames = ["well", "day", "modality", "segmentation_status", "qc", "usable_for_morphology",
                      "area_px", "circularity", "solidity", "radial_cv", "functional_status", "bpm"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in well_results.values():
            for row in r.get("day_rows", []):
                w.writerow(row)

    # ---------------- P36_CONV_TRANSITION_EPISODE_SUMMARY.csv ----------------
    episode_rows = []
    for well, r in well_results.items():
        if r.get("status") != "ANALYZED":
            continue
        video_qc, _ = video_qc_for(well, json.load(open(locate_well_paths(well)[0])))
        for basis, eps in [("causal", r["causal_episodes"]), ("retrospective", r["retrospective_episodes"])]:
            for idx, ep in enumerate(eps):
                episode_rows.append({
                    "well": well, "basis": basis, "episode_index": idx, "days": ep.days, "status": ep.status,
                    "confidence": ep.confidence, "duration_days": ep.duration_days, "peak_deformation_day": ep.peak_deformation_day,
                    "qc_flagged_day_fraction": round(qc_flagged_fraction(ep.days, video_qc), 2),
                    "functional_onset_distance_days": (ep.days[0] - r["functional_onset_day"]) if r.get("functional_onset_day") is not None else None,
                })
    if episode_rows:
        with open(OUTPUT_DIR / "P36_CONV_TRANSITION_EPISODE_SUMMARY.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()))
            w.writeheader()
            w.writerows(episode_rows)

    # ---------------- P36_CONV_FUNCTIONAL_SUMMARY.csv ----------------
    with open(OUTPUT_DIR / "P36_CONV_FUNCTIONAL_SUMMARY.csv", "w", newline="") as f:
        fieldnames = ["well", "day", "functional_status", "bpm", "reliable_patch_fraction", "median_ibi_s", "ibi_cv",
                      "n_period_families", "regional_bpm_disagreement_range", "max_region_vs_global_bpm_diff"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for well in WELLS:
            _, functional_path = locate_well_paths(well)
            if not functional_path or not functional_path.exists():
                continue
            functional = json.load(open(functional_path))
            for rec in functional:
                w.writerow({
                    "well": well, "day": rec["day_number"], "functional_status": rec.get("functional_status"), "bpm": rec.get("bpm"),
                    "reliable_patch_fraction": rec.get("reliable_patch_fraction"), "median_ibi_s": rec.get("median_ibi_s"),
                    "ibi_cv": rec.get("ibi_cv"), "n_period_families": rec.get("n_period_families"),
                    "regional_bpm_disagreement_range": rec.get("regional_bpm_disagreement_range"),
                    "max_region_vs_global_bpm_diff": rec.get("max_region_vs_global_bpm_diff"),
                })

    # ---------------- per-well integration (section E) ----------------
    def onset_lag_bin(lag):
        if lag is None:
            return "N/A"
        a = abs(lag)
        if a == 0:
            return "0d"
        if a <= 2:
            return "1-2d"
        if a <= 4:
            return "3-4d"
        return ">=5d"

    integration_rows = []
    for well, r in well_results.items():
        if r.get("status") != "ANALYZED":
            integration_rows.append({"well": well, "status": r.get("status"), "transition_episode": None, "causal_retro_agree": None,
                                      "onset_lag_days": None, "onset_lag_bin": "N/A", "recovery_state_causal": None,
                                      "recovery_state_retro": None, "segmentation_confounded": None, "secondary_episode": None,
                                      "recovery_transition_contradiction": None})
            continue
        mc, mr = r["main_causal_episode"], r["main_retrospective_episode"]
        transition_exists = mc is not None or mr is not None
        agree = None
        if mc is not None and mr is not None:
            agree = bool(set(mc.days) & set(mr.days))
        lag = (mc.days[0] - r["functional_onset_day"]) if (mc is not None and r.get("functional_onset_day") is not None) else \
              ((mr.days[0] - r["functional_onset_day"]) if (mr is not None and r.get("functional_onset_day") is not None) else None)
        secondary = (len([e for e in r["causal_episodes"] if e.status == "VALIDATED"]) >= 2) or \
                    (len([e for e in r["retrospective_episodes"] if e.status == "VALIDATED"]) >= 2)
        segmentation_confounded = r["usable_video_fraction"] < 0.75
        rec_c, rec_r = r["recovery_causal"], r["recovery_retrospective"]
        contradiction = False
        for rec, eps in [(rec_c, r["causal_episodes"]), (rec_r, r["retrospective_episodes"])]:
            candidate_days = {d for e in eps for d in e.days}
            for d, flags in (rec.per_day_recovery_flags or {}).items():
                if sum(flags.values()) >= 2 and d in candidate_days:
                    contradiction = True
        integration_rows.append({
            "well": well, "status": "ANALYZED", "transition_episode": transition_exists,
            "causal_retro_agree": agree, "onset_lag_days": lag, "onset_lag_bin": onset_lag_bin(lag),
            "recovery_state_causal": rec_c.state, "recovery_state_retro": rec_r.state,
            "segmentation_confounded": segmentation_confounded, "secondary_episode": secondary,
            "recovery_transition_contradiction": contradiction,
        })
    with open(OUTPUT_DIR / "per_well_integration.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(integration_rows[0].keys()))
        w.writeheader()
        w.writerows(integration_rows)

    # ---------------- cross-well summary (section F) ----------------
    analyzed = [row for row in integration_rows if row["status"] == "ANALYZED"]
    n_total, n_analyzable = len(WELLS), len(analyzed)
    n_transition = sum(1 for row in analyzed if row["transition_episode"])
    agree_known = [row for row in analyzed if row["causal_retro_agree"] is not None]
    n_agree = sum(1 for row in agree_known if row["causal_retro_agree"])
    lag_bins = {"0d": 0, "1-2d": 0, "3-4d": 0, ">=5d": 0, "N/A": 0}
    for row in analyzed:
        lag_bins[row["onset_lag_bin"]] += 1
    n_recovery = sum(1 for row in analyzed if row["recovery_state_causal"] not in (RECOVERY_NOT_SUPPORTED, None) or row["recovery_state_retro"] not in (RECOVERY_NOT_SUPPORTED, None))
    n_secondary = sum(1 for row in analyzed if row["secondary_episode"])
    n_confounded = sum(1 for row in analyzed if row["segmentation_confounded"])
    all_func_status = []
    for well in WELLS:
        _, fp = locate_well_paths(well)
        if fp and fp.exists():
            all_func_status.extend(rec.get("functional_status") for rec in json.load(open(fp)))
    n_not_estimable = sum(1 for s in all_func_status if s == "NOT_ESTIMABLE")

    cross_well_summary = {
        "n_wells_total_in_screening": n_total, "n_wells_analyzable": n_analyzable,
        "transition_episode_detected_fraction": round(n_transition / n_analyzable, 2) if n_analyzable else None,
        "causal_retrospective_agreement_fraction": round(n_agree / len(agree_known), 2) if agree_known else None,
        "onset_lag_distribution": lag_bins,
        "recovery_evidence_fraction": round(n_recovery / n_analyzable, 2) if n_analyzable else None,
        "secondary_episode_fraction": round(n_secondary / n_analyzable, 2) if n_analyzable else None,
        "segmentation_confounded_fraction": round(n_confounded / n_analyzable, 2) if n_analyzable else None,
        "not_estimable_functional_fraction": round(n_not_estimable / len(all_func_status), 2) if all_func_status else None,
        "n_functional_observations": len(all_func_status),
    }
    with open(OUTPUT_DIR / "cross_well_summary.json", "w") as f:
        json.dump(cross_well_summary, f, indent=2, default=str)
    print(json.dumps(cross_well_summary, indent=2, default=str))

    # ---------------- figures ----------------
    # per-well causal stage timeline montage
    montage_rows = []
    for well in WELLS:
        r = well_results[well]
        if r.get("status") != "ANALYZED":
            continue
        tmp = FIGURES_DIR / f"_tmp_{well}.png"
        build_stage_timeline([_adapter(res, "causal") for res in r["video_results"]], tmp)
        img = cv2.imread(str(tmp))
        band = np.full((24, img.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(band, f"{well} (onset=D{r.get('functional_onset_day')})", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        montage_rows.append(np.vstack([band, img]))
        tmp.unlink(missing_ok=True)
    if montage_rows:
        max_w = max(im.shape[1] for im in montage_rows)
        def pad(im):
            return np.hstack([im, np.full((im.shape[0], max_w - im.shape[1], 3), 255, dtype=np.uint8)]) if im.shape[1] < max_w else im
        cv2.imwrite(str(FIGURES_DIR / "per_well_causal_stage_montage.png"), np.vstack([pad(im) for im in montage_rows]))

    # integrated timeline: onset day + episode range per well
    fig, ax = plt.subplots(figsize=(14, 0.6 * max(len(analyzed), 1) + 2))
    for i, row in enumerate(analyzed):
        well = row["well"]
        r = well_results[well]
        onset = r.get("functional_onset_day")
        if onset is not None:
            ax.scatter([onset], [i], marker="^", color="black", s=100, zorder=3, label="functional onset" if i == 0 else None)
        mc = r["main_causal_episode"]
        if mc is not None:
            ax.plot([mc.days[0], mc.days[-1]], [i, i], color="tab:orange", linewidth=8, alpha=0.7, label="causal episode" if i == 0 else None)
        mr = r["main_retrospective_episode"]
        if mr is not None:
            ax.plot([mr.days[0], mr.days[-1]], [i + 0.25, i + 0.25], color="tab:red", linewidth=4, alpha=0.5, label="retrospective episode" if i == 0 else None)
    ax.set_yticks(range(len(analyzed)))
    ax.set_yticklabels([row["well"] for row in analyzed])
    ax.set_xlabel("day")
    ax.set_title("P36/Conv screening -- integrated timeline: functional onset (triangle) vs morphology transition episode (bars)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "P36_CONV_INTEGRATED_TIMELINE.png", dpi=115)
    plt.close(fig)

    # ---------------- failure-mode table ----------------
    failure_rows = []
    for well, r in well_results.items():
        modes = []
        if r.get("status") == "TOO_FEW_QC_CLEAN_VIDEO_DAYS":
            modes.append("SEGMENTATION (too few QC-clean video days)")
        if r.get("status") == "ANALYZED":
            n_baseline_insufficient = sum(1 for res in r["video_results"] if res.causal_state == STATE_BASELINE_INSUFFICIENT)
            if n_baseline_insufficient >= 3:
                modes.append("BASELINE_INSUFFICIENCY (many days lack enough preceding clean video days)")
            row = next((x for x in integration_rows if x["well"] == well), None)
            if row and row.get("causal_retro_agree") is False:
                modes.append("CAUSAL/RETROSPECTIVE_DISAGREEMENT (episodes do not overlap)")
            if row and row.get("segmentation_confounded"):
                modes.append("SEGMENTATION (usable-video-day fraction < 75%)")
            if row and row.get("recovery_transition_contradiction"):
                modes.append("METRIC_INSTABILITY (same day flagged both transitional and recovering)")
            if not modes:
                modes.append("NONE_OBVIOUS (or genuine biological heterogeneity -- not distinguishable this round)")
        failure_rows.append({"well": well, "status": r.get("status"), "failure_modes": "; ".join(modes) if modes else "N/A"})
    with open(OUTPUT_DIR / "failure_mode_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["well", "status", "failure_modes"])
        w.writeheader()
        w.writerows(failure_rows)

    print("\nWrote outputs to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
