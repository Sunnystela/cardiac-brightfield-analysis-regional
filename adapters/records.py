"""Segmentation-interface records (engine format v1.0) and mask access.

A mask enters analysis ONLY through the engine's own gate
`cardio_organoid.pipeline.segmentation_interface.require_downstream_mask`, which accepts
nothing but a valid ACCEPT record with human provenance (human_confirmed / human_corrected)
and an existing mask file. Nothing here can promote a mask to ACCEPT on its own.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ._engine import engine


def _si():
    return engine("cardio_organoid.pipeline.segmentation_interface")


def load_records(path: Path) -> list:
    """Read records from `segmentation_interface_records.json` ({"records": [...]}), a list, or one record."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data["records"] if isinstance(data, dict) and "records" in data else (data if isinstance(data, list) else [data])
    return [_si().SegmentationRecord.from_dict(d) for d in items]


def find_record(records: list, well: str, day: int, modality: str | None = None):
    hits = [r for r in records if r.well == well and int(r.day) == int(day) and (modality is None or r.modality == modality)]
    if len(hits) != 1:
        raise LookupError(f"expected exactly one record for well={well} day={day} modality={modality}, found {len(hits)}")
    return hits[0]


def observation_id(rec) -> str:
    """Same construction as the original first-pass screening (`obs_id`)."""
    return f"{rec.batch}_{rec.condition}_{rec.well}_D{rec.day}_{rec.modality}"


def base_fields(rec, source_file, meta: dict | None = None) -> dict:
    """Batch -> Condition -> Plate -> Well -> Organoid -> Day -> Source traceability (CLAUDE.md #6)."""
    meta = meta or {}
    return {"observation_id": observation_id(rec), "batch": rec.batch, "condition": rec.condition,
            "plate": meta.get("plate"), "well": rec.well,
            "organoid_id": meta.get("organoid_id") or f"{rec.batch}_{rec.condition}_{rec.well}",
            "day": rec.day, "modality": rec.modality, "source_file": str(source_file) if source_file else None,
            "segmentation_qc_status": rec.segmentation_qc_status, "qc_reason": rec.qc_reason,
            "mask_provenance": rec.mask_provenance, "selected_candidate": rec.selected_candidate,
            "mask_path": rec.mask_path}


def require_mask(rec, mask_root: Path, purpose: str) -> np.ndarray:
    """Engine gate. Relative `mask_path` resolves against `mask_root`; absolute paths are used as-is."""
    si = _si()
    with si.fail_closed_mode():
        return si.require_downstream_mask(rec, Path(mask_root), purpose)


def resolved_mask_path(rec, mask_root: Path) -> Path:
    return (Path(mask_root) / rec.mask_path).resolve()


def make_record(*, batch: str, condition: str, well: str, day: int, modality: str, mask_path: str | None,
                segmentation_qc_status: str, mask_provenance: str, qc_reason: str,
                selected_candidate: int | None) -> object:
    """Build a record and validate it with the engine's own rules (raises on any violation).

    There are deliberately no defaults for status / provenance / reason: the reviewer states them.
    """
    si = _si()
    rec = si.SegmentationRecord(batch=batch, condition=condition, well=well, day=int(day), modality=modality,
                                selected_candidate=selected_candidate, selector_top1_probability=None, selector_margin=None,
                                segmentation_qc_status=segmentation_qc_status, qc_reason=qc_reason,
                                mask_provenance=mask_provenance, mask_path=mask_path)
    errs = si.validate_record(rec)
    if errs:
        raise ValueError(f"invalid segmentation record: {errs}")
    return rec


def write_records(records: list, path: Path) -> Path:
    si = _si()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"interface_version": si.INTERFACE_VERSION, "records": [r.to_dict() for r in records]},
                               indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def mask_png_to_npz(png_path: Path, npz_path: Path) -> Path:
    """Convert a binary PNG mask (non-zero = organoid) to the engine's npz format (key 'mask', bool)."""
    import cv2
    img = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(png_path)
    if img.ndim == 3:
        img = img[..., :3].max(axis=2)
    _si().save_mask(img > 0, Path(npz_path))
    return Path(npz_path)
