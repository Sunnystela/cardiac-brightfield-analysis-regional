"""Segmentation-result interface v1: the single source of truth that
downstream morphology and functional code consume.

Downstream modules must never pick a SAM2 candidate themselves. They read a
SegmentationRecord, check its QC status, and load `mask_path`.

Status semantics (v1, frozen 2026-09-28):
  ACCEPT                   -- mask may enter automatic morphology.
  REVIEW_REQUIRED          -- excluded from automatic morphology; metrics, if
                              computed at all, are diagnostic-only.
  NO_ACCEPTABLE_CANDIDATE  -- no usable mask; selected_candidate and
                              mask_path are None; morphology is missing (NaN).
There is no validated automatic acceptability gate yet, so ACCEPT requires
human provenance (human_confirmed / human_corrected) and status is never
derived from a single confidence threshold.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

INTERFACE_VERSION = "1.0"

MODALITIES = ("IMAGE", "VIDEO")
QC_ACCEPT = "ACCEPT"
QC_REVIEW = "REVIEW_REQUIRED"
QC_NONE = "NO_ACCEPTABLE_CANDIDATE"
QC_STATUSES = (QC_ACCEPT, QC_REVIEW, QC_NONE)
PROV_AUTOMATIC = "automatic"
PROV_HUMAN_CONFIRMED = "human_confirmed"
PROV_HUMAN_CORRECTED = "human_corrected"
PROVENANCES = (PROV_AUTOMATIC, PROV_HUMAN_CONFIRMED, PROV_HUMAN_CORRECTED)


@dataclasses.dataclass
class SegmentationRecord:
    batch: str
    condition: str
    well: str
    day: int
    modality: str
    selected_candidate: int | None
    selector_top1_probability: float | None
    selector_margin: float | None
    segmentation_qc_status: str
    qc_reason: str
    mask_provenance: str
    mask_path: str | None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SegmentationRecord":
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: d[k] for k in fields})


def validate_record(rec: SegmentationRecord) -> list[str]:
    """Return a list of violations (empty list = valid)."""
    errors = []
    for name in ("batch", "condition", "well", "qc_reason"):
        if not getattr(rec, name):
            errors.append(f"missing {name}")
    if not isinstance(rec.day, int):
        errors.append("day must be int")
    if rec.modality not in MODALITIES:
        errors.append(f"bad modality {rec.modality!r}")
    if rec.segmentation_qc_status not in QC_STATUSES:
        errors.append(f"bad segmentation_qc_status {rec.segmentation_qc_status!r}")
    if rec.mask_provenance not in PROVENANCES:
        errors.append(f"bad mask_provenance {rec.mask_provenance!r}")
    if rec.selected_candidate is not None and rec.selected_candidate not in (0, 1, 2):
        errors.append(f"bad selected_candidate {rec.selected_candidate!r}")
    is_none = rec.segmentation_qc_status == QC_NONE
    if is_none != (rec.selected_candidate is None):
        errors.append("selected_candidate must be None iff status is NO_ACCEPTABLE_CANDIDATE")
    if is_none and rec.mask_path is not None:
        errors.append("NO_ACCEPTABLE_CANDIDATE must not carry a mask_path")
    if rec.segmentation_qc_status == QC_ACCEPT and rec.mask_provenance == PROV_AUTOMATIC:
        errors.append("automatic ACCEPT is not allowed in v1 (no validated acceptability gate)")
    if rec.segmentation_qc_status == QC_ACCEPT and rec.mask_path is None:
        errors.append("ACCEPT requires a persisted mask_path")
    return errors


def morphology_eligible(rec: SegmentationRecord) -> bool:
    """Only ACCEPT records enter automatic morphology."""
    return rec.segmentation_qc_status == QC_ACCEPT


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, mask=np.asarray(mask, dtype=bool))


def load_mask(path: Path) -> np.ndarray:
    with np.load(path) as z:
        return z["mask"].astype(bool)


def load_selected_mask(rec: SegmentationRecord, repo_root: Path) -> np.ndarray | None:
    """The only sanctioned way for downstream code to obtain a mask."""
    if rec.segmentation_qc_status == QC_NONE or rec.mask_path is None:
        return None
    return load_mask(repo_root / rec.mask_path)


# ---------------------------------------------------------------- fail-closed
class InterfaceFailClosed(RuntimeError):
    """Raised when downstream analysis is attempted without an eligible
    segmentation-interface record, or when a legacy fixed-candidate-index
    selection is reached while fail-closed mode is active."""


_LEGACY_FIXED_INDEX_BLOCKED = False


class fail_closed_mode:
    """Context manager: inside it, every legacy fixed-candidate-index
    selection (call sites use assert_legacy_fixed_index_allowed) raises
    instead of silently using configs candidate_index."""

    def __enter__(self):
        global _LEGACY_FIXED_INDEX_BLOCKED
        self._prev = _LEGACY_FIXED_INDEX_BLOCKED
        _LEGACY_FIXED_INDEX_BLOCKED = True
        return self

    def __exit__(self, *exc):
        global _LEGACY_FIXED_INDEX_BLOCKED
        _LEGACY_FIXED_INDEX_BLOCKED = self._prev
        return False


def assert_legacy_fixed_index_allowed(where: str) -> None:
    if _LEGACY_FIXED_INDEX_BLOCKED:
        raise InterfaceFailClosed(
            f"legacy fixed candidate_index selection reached in {where} while fail-closed mode is active; "
            "downstream code must receive the segmentation-interface mask")


def require_downstream_mask(rec: SegmentationRecord | None, repo_root: Path, purpose: str) -> np.ndarray:
    """Gate for every downstream analysis in the new batch path. Returns the
    interface mask only for a valid ACCEPT record; raises otherwise."""
    if rec is None:
        raise InterfaceFailClosed(f"{purpose}: no segmentation record")
    errs = validate_record(rec)
    if errs:
        raise InterfaceFailClosed(f"{purpose}: invalid record {rec.well} D{rec.day}: {errs}")
    if rec.segmentation_qc_status != QC_ACCEPT:
        raise InterfaceFailClosed(f"{purpose}: {rec.well} D{rec.day} is {rec.segmentation_qc_status} ({rec.qc_reason})")
    path = repo_root / rec.mask_path
    if not path.exists():
        raise InterfaceFailClosed(f"{purpose}: mask file missing for {rec.well} D{rec.day}: {rec.mask_path}")
    return load_mask(path)


# ---------------------------------------------------------------- reconciliation
def reconcile_status(owner_label: str | None, owner_category: str | None, selector_prediction: int,
                     historical_idx1_score: int | None, historical_idx1_same_mask: bool | None,
                     candidates_match_labeling_run: bool | None) -> tuple:
    """Owner-approved rules (2026-09-28). Returns
    (selected_candidate, segmentation_qc_status, qc_reason, mask_provenance).
    Never uses a confidence threshold."""
    if owner_label is not None:
        if candidates_match_labeling_run is False:
            return selector_prediction, QC_REVIEW, "candidate_regeneration_mismatch_with_labeling_run", PROV_AUTOMATIC
        if owner_label == "NONE":
            return None, QC_NONE, "owner_label_none", PROV_HUMAN_CONFIRMED
        k = int(owner_label)
        prov = PROV_HUMAN_CONFIRMED if k == selector_prediction else PROV_HUMAN_CORRECTED
        if owner_category == "CLEAN_CANDIDATE_EXISTS":
            return k, QC_ACCEPT, "owner_candidate_label_clean", prov
        return k, QC_REVIEW, f"owner_label_{(owner_category or 'unknown').lower()}_not_accepted", prov
    if historical_idx1_score is not None and historical_idx1_score >= 4:
        if not historical_idx1_same_mask:
            return selector_prediction, QC_REVIEW, "historical_idx1_score_refers_to_a_different_mask", PROV_AUTOMATIC
        if selector_prediction == 1:
            return 1, QC_ACCEPT, "historical_idx1_score_ge4_same_mask_and_selector_selects_idx1", PROV_HUMAN_CONFIRMED
        return selector_prediction, QC_REVIEW, "historical_idx1_score_ge4_but_selector_selects_other_candidate", PROV_AUTOMATIC
    if historical_idx1_score is not None:
        return selector_prediction, QC_REVIEW, "historical_idx1_score_le3_without_candidate_label", PROV_AUTOMATIC
    return selector_prediction, QC_REVIEW, "no_human_evidence", PROV_AUTOMATIC
