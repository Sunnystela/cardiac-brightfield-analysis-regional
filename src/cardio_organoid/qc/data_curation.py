"""Master observation QC state: experimental / acquisition / segmentation QC and downstream eligibility.

One set of stage definitions serves both workflows (owner decision 2026-10-12):

    FULL_AUTOMATION  experimental -> morphology + functional input -> segmentation -> functional ROI
    BEATING_ONLY     experimental -> functional input -> functional ROI          (a subset of the same stages)

Every stage is a pure function of one observation's evidence dict (built by the curation script from
existing owner labels / manifests / inventories) plus the config, and returns a StageResult.  Nothing
here reads files, trains, segments or measures anything.

Owner rules encoded here (2026-10-12):
- An organoid fused with debris / surrounding tissue so that even a human cannot place its true boundary
  is an EXPERIMENTAL_QC failure, not a segmentation failure.  Debris that is clearly separable is not.
- One bad observation does not fail the well; well-level exclusion is only proposed, never applied.
- Morphology and functional acquisition QC are separate: a still-frame focus failure does not exclude
  the video's functional analysis, and vice versa.
- Eligibility (input / sample usability) is not the post-run functional status
  (ESTIMABLE / LOW_CONFIDENCE / NOT_ESTIMABLE); the latter is never read here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

QC3 = ("PASS", "FAIL", "REVIEW")
INPUT_QC = QC3 + ("NOT_APPLICABLE",)
ROI_QC = QC3 + ("NOT_APPLICABLE", "NOT_EVALUATED")
EXPERIMENTAL_REASONS = ("HUMAN_AMBIGUOUS_ORGANOID_DEBRIS_ADHESION", "ABNORMAL_ATTACHMENT", "ABNORMAL_AGGREGATION",
                        "SEVERE_DEBRIS_CONTAMINATION", "OTHER_SAMPLE_QUALITY", "NONE")
ACQUISITION_REASONS = ("FOCUS", "ILLUMINATION", "LOW_CONTRAST", "VIDEO_QUALITY", "FRAME_CORRUPTION",
                       "SOURCE_PATH_UNRESOLVED", "OTHER")
SEGMENTATION_QC = ("ACCEPT", "REVIEW_REQUIRED", "NO_ACCEPTABLE_CANDIDATE", "NO_VALID_MAIN_BODY_CANDIDATE", "NOT_EVALUATED")
SEGMENTATION_REASONS = ("SUSPECTED_TRANSLUCENT_MISS", "OVERSEGMENTATION_DEBRIS", "UNDERSEGMENTATION", "BOUNDARY_ERROR",
                        "MAIN_BODY_GENERATION_FAILURE", "OTHER")
ELIGIBILITY = ("YES", "NO", "REVIEW")
CATEGORIES = {"A": "EXPERIMENTAL_FAILURE", "B": "ACQUISITION_FAILURE", "C": "SAM2_GENERATION_FAILURE",
              "D": "SEGMENTATION_MASK_FAILURE", "E": "TRANSLUCENT_MISS", "F": "CLEAN_ACCEPTABLE", "G": "UNRESOLVED"}

# pool states (normalised by the script from the round-6 candidate label state + manifest v4)
POOL_STATES = ("EXPLICIT_ACCEPTABLE", "MANIFEST_ACCEPT", "LEGACY_INFERRED_ONLY", "NONE_CONFIRMED", "UNCERTAIN", "NOT_REVIEWED")


@dataclass(frozen=True)
class StageResult:
    status: str
    reasons: tuple[str, ...] = ()
    basis: str = ""
    review_question: str = ""        # set when the status needs a human / owner answer
    notes: tuple[str, ...] = field(default_factory=tuple)

    def reason_str(self) -> str:
        return ";".join(self.reasons)


def _q(ev: Mapping, key: str) -> str:
    return ((ev.get("quick") or {}).get(key) or "").upper()


def owner_accepted(ev: Mapping) -> bool:
    """The owner explicitly accepted a candidate (manifest ACCEPT after owner review, or an explicit acceptable label)."""
    return ev["pool_state"] in ("EXPLICIT_ACCEPTABLE", "MANIFEST_ACCEPT")


def alternative_explanation(ev: Mapping) -> bool:
    """Boundary difficulty already explained by translucent tissue, focus, or a main-body generation failure."""
    l3 = ev.get("l3") or {}
    focus = bool(ev.get("focus_failure_evidence")) and (l3.get("focus_confounded") or "").upper() != "NO"   # L3 supersedes
    if l3:   # latest detailed component labels supersede quick-screen Q1 and older comments
        translucent = l3.get("n_translucent", 0) > 0
    else:
        translucent = bool(ev.get("translucent_comment")) or _q(ev, "q1_translucent_in_mask") == "YES"
    return bool(translucent or focus or ev.get("main_body_label_free_invalid"))


# ------------------------------------------------------------------ stage 1: experimental / sample QC
def experimental_qc(ev: Mapping, cfg: Mapping) -> StageResult:
    debris_amb = _q(ev, "q4_debris_boundary_confusing") == "YES" or bool(ev.get("debris_ambiguity_comment"))
    amb = bool(ev.get("boundary_ambiguity"))
    pool = ev["pool_state"]
    if pool == "NONE_CONFIRMED" and debris_amb and amb and not alternative_explanation(ev) and _q(ev, "q3_focus_illumination") in ("", "GOOD") \
            and not ev.get("pending_owner_question"):     # an open owner question blocks automatic confirmation
        return StageResult("FAIL", ("HUMAN_AMBIGUOUS_ORGANOID_DEBRIS_ADHESION",), "CONSISTENT_OWNER_EVIDENCE",
                           notes=("owner: no acceptable candidate + debris/organoid boundary confusing + boundary ambiguous; "
                                  "no translucent / focus / main-body explanation",))
    if debris_amb:
        return StageResult("REVIEW", ("HUMAN_AMBIGUOUS_ORGANOID_DEBRIS_ADHESION",), "SUSPECTED_FROM_OWNER_DEBRIS_STATEMENT",
                           "ADHESION_AMBIGUITY_UNCLEAR")
    if amb and not owner_accepted(ev) and not alternative_explanation(ev):
        return StageResult("REVIEW", ("HUMAN_AMBIGUOUS_ORGANOID_DEBRIS_ADHESION",), "SUSPECTED_FROM_AMBIGUOUS_BOUNDARY_NO_ACCEPTABLE",
                           "EXPERIMENTAL_VS_SEGMENTATION_UNCLEAR")
    if pool == "NONE_CONFIRMED" and not (amb or alternative_explanation(ev) or ev.get("error_direction")):
        return StageResult("REVIEW", ("OTHER_SAMPLE_QUALITY",), "POOL_NONE_WITHOUT_RECORDED_REASON",
                           "EXPERIMENTAL_VS_SEGMENTATION_UNCLEAR")
    if owner_accepted(ev):
        basis = "OWNER_ACCEPTED_MASK"
    elif _q(ev, "q4_debris_boundary_confusing") == "NO":
        basis = "OWNER_QUICK_SCREEN_NO_DEBRIS_CONFUSION"
    else:
        basis = "NO_ADVERSE_EVIDENCE"
    return StageResult("PASS", ("NONE",), basis)


# ------------------------------------------------------------------ stage 2a: morphology acquisition QC
def _source_result(ev: Mapping) -> StageResult:
    """A moved file whose alias awaits owner confirmation is REVIEW; a file absent from the repository is FAIL."""
    if ev["source_status"] == "ALIAS_PENDING_OWNER_CONFIRMATION":
        return StageResult("REVIEW", ("SOURCE_PATH_UNRESOLVED",), ev["source_status"], "SOURCE_PATH_ALIAS_CONFIRMATION")
    return StageResult("FAIL", ("SOURCE_PATH_UNRESOLVED",), ev["source_status"])


def morphology_input_qc(ev: Mapping, cfg: Mapping) -> StageResult:
    if ev["source_status"] != "OK":
        return _source_result(ev)
    if ev.get("reference_corrupt"):
        return StageResult("FAIL", ("FRAME_CORRUPTION",), "INVENTORY_CORRUPT")
    l3f = ((ev.get("l3") or {}).get("focus_confounded") or "").upper()
    fail_ev = bool(ev.get("focus_failure_evidence"))
    q3 = _q(ev, "q3_focus_illumination")
    q3_reasons = ("FOCUS", "ILLUMINATION") if q3 in ("BAD", "BORDERLINE") else ()
    if l3f == "NO":
        note = ("latest detailed answer (L3) says focus does not confound; older focus statements superseded",) if (fail_ev or q3_reasons) else ()
        return StageResult("PASS", (), "L3_DETAILED_FOCUS_NO", notes=note)
    if l3f == "YES":
        if fail_ev:
            return StageResult("FAIL", ("FOCUS",), "L3_FOCUS_CONFOUNDED_PLUS_OWNER_FOCUS_FAILURE")
        if owner_accepted(ev):
            return StageResult("PASS", (), "L3_FOCUS_CONFOUNDED_BUT_OWNER_ACCEPTED_MASK",
                               notes=("detailed focus answer concerns component judgement; mask was accepted by the owner",))
        return StageResult("REVIEW", ("FOCUS",), "L3_FOCUS_CONFOUNDED", "MORPHOLOGY_INPUT_FOCUS")
    if fail_ev:
        if owner_accepted(ev):
            return StageResult("REVIEW", tuple(sorted(set(("FOCUS",) + q3_reasons))), "OWNER_FOCUS_FAILURE_VS_ACCEPTED_MASK",
                               "OWNER_LABEL_CONFLICT_FOCUS_VS_ACCEPTED_MASK")
        return StageResult("FAIL", tuple(sorted(set(("FOCUS",) + q3_reasons))), "OWNER_FOCUS_FAILURE")
    if q3 == "BORDERLINE":
        if owner_accepted(ev):
            return StageResult("PASS", (), "BORDERLINE_BUT_OWNER_ACCEPTED_MASK")
        return StageResult("REVIEW", q3_reasons, "QUICK_SCREEN_BORDERLINE", "MORPHOLOGY_INPUT_FOCUS")
    return StageResult("PASS", (), "NO_ADVERSE_EVIDENCE" if not q3 else "QUICK_SCREEN_GOOD")


# ------------------------------------------------------------------ stage 2b: functional acquisition QC
def functional_input_qc(ev: Mapping, cfg: Mapping) -> StageResult:
    if ev["modality"] != "VIDEO":
        return StageResult("NOT_APPLICABLE", (), "IMAGE_OBSERVATION_NO_VIDEO")
    if ev["source_status"] != "OK":
        return _source_result(ev)
    v = ev.get("video") or {}
    if not v:
        return StageResult("REVIEW", ("VIDEO_QUALITY",), "NOT_IN_VIDEO_INVENTORY", "FUNCTIONAL_INPUT_METADATA")
    if v.get("corrupt"):
        return StageResult("FAIL", ("FRAME_CORRUPTION",), "INVENTORY_CORRUPT")
    tol = float(cfg["functional_input"]["fps_relative_tolerance"])
    if abs(v["fps"] - v["fps_population_median"]) > tol * v["fps_population_median"]:
        return StageResult("REVIEW", ("VIDEO_QUALITY",), "FPS_OUTSIDE_TOLERANCE", "FUNCTIONAL_INPUT_METADATA")
    if v["duration_s"] < float(cfg["functional_input"]["min_duration_s"]):
        return StageResult("REVIEW", ("VIDEO_QUALITY",), "VIDEO_SHORTER_THAN_MIN", "FUNCTIONAL_INPUT_METADATA")
    notes = ("morphology-frame focus flag not propagated to functional input (owner rule)",) if ev.get("focus_failure_evidence") else ()
    return StageResult("PASS", (), "INVENTORY_OK", notes=notes)


# ------------------------------------------------------------------ stage 3: segmentation QC
def _translucent_omitted(l3: Mapping, cov: Mapping, cfg: Mapping) -> bool:
    t_in, t_n = cov.get("TRANSLUCENT", (0, 0))
    return bool(t_n) and (l3.get("inclusion") or "").upper() in ("YES", "UNCERTAIN") \
        and t_n - t_in >= int(cfg["segmentation"]["translucent_miss_min_uncovered_strokes"])


def _failure_reasons(ev: Mapping, cfg: Mapping) -> list[str]:
    out = []
    l3 = ev.get("l3")
    if l3:   # judged on the owner's preferred candidate when it exists, else on presence alone
        pc = l3.get("preferred_mask_coverage")
        miss = _translucent_omitted(l3, pc, cfg) if pc else l3.get("n_translucent", 0) > 0
    else:
        miss = bool(ev.get("translucent_comment"))
    if miss:
        out.append("SUSPECTED_TRANSLUCENT_MISS")
    ed = set(ev.get("error_direction") or ())
    if "UNDER" in ed:
        out.append("UNDERSEGMENTATION")
    if "OVER" in ed:
        out.append("OVERSEGMENTATION_DEBRIS")
    if ev.get("boundary_ambiguity"):
        out.append("BOUNDARY_ERROR")
    return out


def accepted_mask_flags(ev: Mapping, cfg: Mapping) -> list[str]:
    """Owner-accepted mask vs the latest owner component labels (L3 scribbles, else L2 quick screen)."""
    c = cfg["segmentation"]
    l3 = ev.get("l3")
    out = []
    if l3:
        cov = l3.get("accepted_mask_coverage") or {}
        s_in, s_n = cov.get("SOLID", (0, 0))
        if _translucent_omitted(l3, cov, cfg):
            out.append("SUSPECTED_TRANSLUCENT_MISS")
        if s_n and s_in / s_n < float(c["main_body_min_solid_coverage"]):
            out.append("UNDERSEGMENTATION")
    elif _q(ev, "q1_translucent_in_mask") == "YES":
        out.append("SUSPECTED_TRANSLUCENT_MISS")
    return out


def segmentation_qc(ev: Mapping, cfg: Mapping, exp: StageResult, morph_in: StageResult) -> StageResult:
    if exp.status == "FAIL" or morph_in.status == "FAIL":
        return StageResult("NOT_EVALUATED", (), "UPSTREAM_QC_FAIL (historical segmentation labels kept in registry)")
    pool = ev["pool_state"]
    if pool == "NOT_REVIEWED":
        sus = []
        l3 = ev.get("l3") or {}
        if l3.get("main_body_status", "").startswith("MAIN_BODY_INVALID"):
            sus.append("MAIN_BODY_GENERATION_FAILURE")
        if l3.get("n_translucent", 0) > 0 or _q(ev, "q1_translucent_in_mask") == "YES":
            sus.append("SUSPECTED_TRANSLUCENT_MISS")
        return StageResult("NOT_EVALUATED", tuple(sus), "NO_CANDIDATE_REVIEW (never segmentation-reviewed)", "NOT_YET_SEGMENTATION_REVIEWED")
    if owner_accepted(ev):
        flags = accepted_mask_flags(ev, cfg)
        if flags:
            return StageResult("REVIEW_REQUIRED", tuple(flags), "OWNER_ACCEPTED_MASK_VS_LATEST_COMPONENT_LABELS", "SEGMENTATION_MASK_REVIEW")
        return StageResult("ACCEPT", (), "MANIFEST_V4_ACCEPT" if pool == "MANIFEST_ACCEPT" else "OWNER_EXPLICIT_ACCEPTABLE_LABEL")
    if ev.get("main_body_label_free_invalid"):
        return StageResult("NO_VALID_MAIN_BODY_CANDIDATE", ("MAIN_BODY_GENERATION_FAILURE",), "MAIN_BODY_PROXY_BELOW_MIN_FRAME_FRACTION")
    if pool == "NONE_CONFIRMED":
        return StageResult("NO_ACCEPTABLE_CANDIDATE", tuple(_failure_reasons(ev, cfg) or ["OTHER"]), "OWNER_POOL_NONE")
    if pool == "UNCERTAIN":
        return StageResult("REVIEW_REQUIRED", tuple(_failure_reasons(ev, cfg) or ["OTHER"]), "OWNER_POOL_UNCERTAIN", "SEGMENTATION_MASK_REVIEW")
    if pool == "LEGACY_INFERRED_ONLY":
        return StageResult("REVIEW_REQUIRED", tuple(_failure_reasons(ev, cfg) or ["OTHER"]), "PREFERENCE_ONLY_USABILITY_UNCONFIRMED",
                           "SEGMENTATION_USABILITY_UNCONFIRMED")
    raise ValueError(f"unknown pool state {pool!r}")


# ------------------------------------------------------------------ stage 4: functional ROI QC (main body only)
def functional_roi_qc(ev: Mapping, cfg: Mapping, exp: StageResult, func_in: StageResult) -> StageResult:
    """Is there a main-body region good enough to analyse motion?  Independent of morphology-boundary precision."""
    if ev["modality"] != "VIDEO":
        return StageResult("NOT_APPLICABLE", (), "IMAGE_OBSERVATION_NO_VIDEO")
    if exp.status == "FAIL" or func_in.status == "FAIL":
        return StageResult("NOT_EVALUATED", (), "UPSTREAM_QC_FAIL")
    if owner_accepted(ev):
        cov = ((ev.get("l3") or {}).get("accepted_mask_coverage") or {}).get("SOLID", (0, 0))
        if cov[1] and cov[0] / cov[1] < float(cfg["segmentation"]["main_body_min_solid_coverage"]):
            return StageResult("REVIEW", ("MAIN_BODY_GENERATION_FAILURE",), "ACCEPTED_MASK_MISSES_SOLID_SCRIBBLES",
                               "OWNER_LABEL_CONFLICT_ACCEPTED_MASK_VS_SOLID")
        return StageResult("PASS", (), "OWNER_ACCEPTED_MASK_COVERS_MAIN_BODY")
    if ev.get("main_body_label_free_invalid"):
        return StageResult("FAIL", ("MAIN_BODY_GENERATION_FAILURE",), "NO_MAIN_BODY_CANDIDATE")
    q = "FUNCTIONAL_ROI_NOT_YET_REVIEWED" if ev["pool_state"] == "NOT_REVIEWED" else "FUNCTIONAL_ROI_FROM_NON_ACCEPTED_CANDIDATE"
    return StageResult("REVIEW", (), "NO_OWNER_ACCEPTED_MASK", q)


# ------------------------------------------------------------------ eligibility
def morphology_eligibility(exp: StageResult, morph_in: StageResult, seg: StageResult) -> tuple[str, str]:
    if exp.status == "FAIL":
        return "NO", "EXPERIMENTAL_QC_FAIL"
    if morph_in.status == "FAIL":
        return "NO", "MORPHOLOGY_INPUT_QC_FAIL"
    if seg.status in ("NO_ACCEPTABLE_CANDIDATE", "NO_VALID_MAIN_BODY_CANDIDATE"):
        return "NO", f"SEGMENTATION_{seg.status}"
    if exp.status == "PASS" and morph_in.status == "PASS" and seg.status == "ACCEPT":
        return "YES", "ALL_PASS"
    pend = [n for n, r in (("EXPERIMENTAL_QC", exp), ("MORPHOLOGY_INPUT_QC", morph_in)) if r.status == "REVIEW"]
    if seg.status != "ACCEPT":
        pend.append(f"SEGMENTATION_{seg.status}")
    return "REVIEW", ";".join(pend)


def functional_eligibility(modality: str, exp: StageResult, func_in: StageResult, roi: StageResult) -> tuple[str, str]:
    if modality != "VIDEO":
        return "NO", "NOT_APPLICABLE_IMAGE_OBSERVATION"
    if exp.status == "FAIL":
        return "NO", "EXPERIMENTAL_QC_FAIL"
    if func_in.status == "FAIL":
        return "NO", "FUNCTIONAL_INPUT_QC_FAIL"
    if roi.status == "FAIL":
        return "NO", "FUNCTIONAL_ROI_QC_FAIL"
    if exp.status == "PASS" and func_in.status == "PASS" and roi.status == "PASS":
        return "YES", "ALL_PASS"
    return "REVIEW", ";".join(f"{n}_{r.status}" for n, r in (("EXPERIMENTAL_QC", exp), ("FUNCTIONAL_INPUT_QC", func_in),
                                                             ("FUNCTIONAL_ROI_QC", roi)) if r.status != "PASS")


def full_pipeline_eligibility(morph: str, func: str) -> str:
    if "NO" in (morph, func):
        return "NO"
    return "YES" if (morph, func) == ("YES", "YES") else "REVIEW"


# ------------------------------------------------------------------ workflows (one stage registry, two orderings)
STAGES: dict[str, Callable] = {
    "experimental_qc": lambda ev, cfg, r: experimental_qc(ev, cfg),
    "morphology_input_qc": lambda ev, cfg, r: morphology_input_qc(ev, cfg),
    "functional_input_qc": lambda ev, cfg, r: functional_input_qc(ev, cfg),
    "segmentation_qc": lambda ev, cfg, r: segmentation_qc(ev, cfg, r["experimental_qc"], r["morphology_input_qc"]),
    "functional_roi_qc": lambda ev, cfg, r: functional_roi_qc(ev, cfg, r["experimental_qc"], r["functional_input_qc"]),
}
WORKFLOWS = {
    "FULL_AUTOMATION": ("experimental_qc", "morphology_input_qc", "functional_input_qc", "segmentation_qc", "functional_roi_qc"),
    "BEATING_ONLY": ("experimental_qc", "functional_input_qc", "functional_roi_qc"),
}


def run_workflow(name: str, ev: Mapping, cfg: Mapping) -> dict:
    """Run one workflow's stages in order; returns stage results plus the eligibilities that workflow can decide."""
    res: dict = {}
    for s in WORKFLOWS[name]:
        res[s] = STAGES[s](ev, cfg, res)
    out = {"stages": res}
    out["functional_eligible"] = functional_eligibility(ev["modality"], res["experimental_qc"], res["functional_input_qc"],
                                                        res["functional_roi_qc"])
    if "segmentation_qc" in res:
        out["morphology_eligible"] = morphology_eligibility(res["experimental_qc"], res["morphology_input_qc"], res["segmentation_qc"])
        out["full_pipeline_eligible"] = full_pipeline_eligibility(out["morphology_eligible"][0], out["functional_eligible"][0])
    return out


# ------------------------------------------------------------------ retrospective difficult-case taxonomy
def difficult_case_category(ev: Mapping, res: Mapping) -> tuple[str, str]:
    """Primary category (precedence A > B > G-experimental > C > E > D > F > G) and a subtype."""
    exp, mi, fi, seg = (res[k] for k in ("experimental_qc", "morphology_input_qc", "functional_input_qc", "segmentation_qc"))
    if exp.status == "FAIL":
        return "A", exp.reason_str()
    if mi.status == "FAIL" or fi.status == "FAIL":
        return "B", ";".join(sorted(set(mi.reasons) | set(fi.reasons)))
    if exp.status == "REVIEW":
        return "G", f"EXPERIMENTAL_QC_REVIEW:{exp.review_question}"
    if seg.status == "NO_VALID_MAIN_BODY_CANDIDATE":
        return "C", "ALL_CANDIDATES_FAIL_MAIN_BODY"
    accepted_misses_solid = seg.basis == "OWNER_ACCEPTED_MASK_VS_LATEST_COMPONENT_LABELS" and "UNDERSEGMENTATION" in seg.reasons
    if seg.status in ("NO_ACCEPTABLE_CANDIDATE", "REVIEW_REQUIRED") and "SUSPECTED_TRANSLUCENT_MISS" in seg.reasons \
            and not accepted_misses_solid:
        return "E", f"{seg.status}:{seg.basis}"
    if seg.status == "NO_ACCEPTABLE_CANDIDATE" or (seg.status == "REVIEW_REQUIRED" and seg.basis != "PREFERENCE_ONLY_USABILITY_UNCONFIRMED"
                                                     and set(seg.reasons) & {"OVERSEGMENTATION_DEBRIS", "UNDERSEGMENTATION", "BOUNDARY_ERROR"}):
        return "D", f"UNRESOLVED:{seg.reason_str()}"
    if seg.status == "ACCEPT" and ev.get("selector_pick_explicit_unacceptable"):
        return "D", "RESOLVED_BY_ALTERNATIVE_CANDIDATE"
    if seg.status == "ACCEPT" and mi.status == "PASS" and exp.status == "PASS":
        return "F", seg.basis
    return "G", f"{seg.status}:{seg.basis}" if seg.status != "ACCEPT" else f"MORPHOLOGY_INPUT_{mi.status}"
