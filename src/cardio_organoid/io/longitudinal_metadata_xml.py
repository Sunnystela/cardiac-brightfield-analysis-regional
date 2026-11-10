"""Read-only parsing of this project's `*.tiff_metadata.xml` acquisition
sidecars (Zeiss ZEN "ImageMetadata" export) for longitudinal-identity
verification.

Inspected directly (round -- longitudinal identity verification) across a
random sample of sidecar files: there is NO populated well/plate/scene/
specimen/sample-name field, and NO populated stage X/Y position field --
every `<X>`/`<Y>`-family element found is either a template default (0)
or `NaN` (the acquisition software's "positioned regions" stage-move
feature was evidently not used/recorded for this dataset). `HolderName`
and `FileName` exist as tags but are always empty. This module therefore
does NOT fabricate those fields -- it extracts only what is ACTUALLY
present (acquisition timestamp, the original acquisition-software snap
filename, operator username, camera identity) and explicitly reports
`identity_fields_available` so an absence is visible in the data, never
silently assumed by a caller reading only prose documentation.

Every function here only reads a file's own XML text -- never writes,
moves, or renames the sidecar or its paired image.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from pathlib import Path, PureWindowsPath


def _find_text(root: ET.Element, tag_name: str) -> str | None:
    """First element anywhere in the tree whose LOCAL tag name (namespace
    stripped) matches `tag_name` and has non-empty text. None if no such
    element exists or every match is empty (e.g. `<FileName />`)."""
    for el in root.iter():
        local = el.tag.rsplit("}", 1)[-1]
        if local == tag_name and el.text and el.text.strip():
            return el.text.strip()
    return None


def parse_tiff_metadata_xml(path: Path) -> dict:
    """Parses one sidecar. Returns a dict with the fields actually found
    (`creation_date`, `acquisition_datetime`, `image_name_snap` -- the
    acquisition-software's own temp filename, e.g. "Snap-742.czi",
    `user_name`, `camera_serial`) PLUS the fields this project would want
    for longitudinal identity but this metadata format does not carry
    (`well_field_from_metadata`, `stage_x_from_metadata`,
    `stage_y_from_metadata`, `sample_name_from_metadata` -- always None
    here, see module docstring) and `identity_fields_available`: the
    subset of that wishlist actually found non-empty in THIS file (empty
    list if none, which is the observed case for every file inspected so
    far -- kept as a live check, not a hardcoded assumption, so a
    differently-configured acquisition session would still be picked up
    if it ever populated one of these).
    `xml_parse_error` is set (all other fields None) if the file is not
    valid XML."""
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        return {
            "creation_date": None, "acquisition_datetime": None, "image_name_snap": None,
            "user_name": None, "camera_serial": None,
            "well_field_from_metadata": None, "stage_x_from_metadata": None,
            "stage_y_from_metadata": None, "sample_name_from_metadata": None,
            "identity_fields_available": [], "xml_parse_error": str(exc),
        }
    root = tree.getroot()

    image_name_raw = _find_text(root, "ImageName")
    # The acquisition software records its own temp path in Windows form
    # (e.g. "C:\Users\...\Snap-742.czi") regardless of what OS this
    # inventory script runs on -- PureWindowsPath, not Path, so backslash
    # splitting works correctly on a POSIX host too.
    image_name_snap = PureWindowsPath(image_name_raw).name if image_name_raw else None

    camera_serial = None
    for el in root.iter():
        if el.tag.rsplit("}", 1)[-1] == "Device" and "axiocam" in (el.get("Model") or "").lower():
            camera_serial = el.get("SerialNumber") or el.get("UniqueName") or None
            break

    wishlist = {
        "well_field_from_metadata": _find_text(root, "Well") or _find_text(root, "WellName"),
        "stage_x_from_metadata": _find_text(root, "StageXPosition"),
        "stage_y_from_metadata": _find_text(root, "StageYPosition"),
        "sample_name_from_metadata": _find_text(root, "SampleName") or _find_text(root, "SpecimenName"),
    }
    identity_fields_available = [k for k, v in wishlist.items() if v]

    return {
        "creation_date": _find_text(root, "CreationDate"),
        "acquisition_datetime": _find_text(root, "AcquisitionDateAndTime"),
        "image_name_snap": image_name_snap,
        "user_name": _find_text(root, "UserName"),
        "camera_serial": camera_serial,
        **wishlist,
        "identity_fields_available": identity_fields_available,
        "xml_parse_error": None,
    }


def _find_child_text_local(el: ET.Element, tag_name: str) -> str | None:
    """Direct or nested child of `el` whose local tag matches, first
    match, text only (no namespace matching -- same convention as
    `_find_text`, scoped to one element's subtree instead of the whole
    document)."""
    for child in el.iter():
        if child.tag.rsplit("}", 1)[-1] == tag_name and child.text and child.text.strip():
            return child.text.strip()
    return None


def parse_pixel_calibration_um(path: Path) -> dict:
    """Reads this sidecar's own `<Scaling><Items><Distance Id="X"/"Y">`
    block (Zeiss ZEN "ImageMetadata" export) for a REAL, per-file
    recorded pixel calibration -- never a value estimated from image
    resolution or guessed from another file. `<Distance>/<Value>` is
    recorded in METERS regardless of the sibling `<DefaultUnitFormat>`
    display tag (cross-checked against this file's own
    `CameraPixelDistance` / `TotalMagnification`, e.g. 2.2 um / 10x =
    0.22 um/px, which matches the raw Value*1e6 conversion used here).
    Returns `um_per_pixel_x`/`um_per_pixel_y` = None (not estimated) if
    the file has no parseable `<Distance Id="X"/"Y">` block -- e.g. this
    project's `.avi` videos have no such sidecar at all, so a caller must
    check for the sidecar's existence itself before calling this."""
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        return {
            "um_per_pixel_x": None, "um_per_pixel_y": None,
            "objective_magnification": None, "camera_pixel_distance_um": None,
            "calibration_source": None, "xml_parse_error": str(exc),
        }
    root = tree.getroot()

    def _distance_um(axis_id: str) -> float | None:
        for el in root.iter():
            if el.tag.rsplit("}", 1)[-1] == "Distance" and el.get("Id") == axis_id:
                raw = _find_child_text_local(el, "Value")
                if raw is None:
                    continue
                try:
                    return float(raw) * 1e6  # recorded in meters -> micrometers
                except ValueError:
                    continue
        return None

    magnification = _find_text(root, "TotalMagnification")
    pixel_distance_raw = _find_text(root, "CameraPixelDistance")  # e.g. "2.2,2.2"
    camera_pixel_distance_um = None
    if pixel_distance_raw:
        try:
            camera_pixel_distance_um = float(pixel_distance_raw.split(",")[0])
        except ValueError:
            camera_pixel_distance_um = None

    um_x, um_y = _distance_um("X"), _distance_um("Y")
    return {
        "um_per_pixel_x": um_x, "um_per_pixel_y": um_y,
        "objective_magnification": float(magnification) if magnification else None,
        "camera_pixel_distance_um": camera_pixel_distance_um,
        "calibration_source": "tiff_metadata_xml_scaling_block" if (um_x or um_y) else None,
        "xml_parse_error": None,
    }


def expected_acquisition_date(batch_date_iso: str | None, day_number: int | None) -> date | None:
    """The date a `Dn` folder's files SHOULD have been acquired on, given
    the batch's own start date (`batch_date_iso`, "YYYY-MM-DD", the batch
    folder's own D0 date) plus `day_number` days -- e.g. batch_date
    "2026-08-26" + day_number 8 -> 2026-09-03. None if either input is
    missing/unparseable (never guessed)."""
    if not batch_date_iso or day_number is None:
        return None
    try:
        start = datetime.strptime(batch_date_iso, "%Y-%m-%d").date()
    except ValueError:
        return None
    return start + timedelta(days=day_number)


def check_date_consistency(expected: date | None, acquisition_datetime_str: str | None) -> dict:
    """Compares `expected` (see `expected_acquisition_date`) against the
    DATE portion of an XML sidecar's own `acquisition_datetime` (ISO
    8601, e.g. "2026-09-03T11:41:32.28778Z"). Returns {"actual_date",
    "expected_date", "diff_days", "matches"} -- `matches` is None (not
    True/False) if either side is unavailable to compare, never silently
    treated as a pass."""
    if expected is None or not acquisition_datetime_str:
        return {"actual_date": None, "expected_date": expected, "diff_days": None, "matches": None}
    try:
        actual = datetime.fromisoformat(acquisition_datetime_str.replace("Z", "+00:00")).date()
    except ValueError:
        return {"actual_date": None, "expected_date": expected, "diff_days": None, "matches": None}
    diff = (actual - expected).days
    return {"actual_date": actual, "expected_date": expected, "diff_days": diff, "matches": diff == 0}
