"""GUI-facing public API of the Cardiac bright-field standalone analysis.

Intended GUI flow (each step is one call):

    1. choose input        records = load_records(record_json); rec = find_record(records, well, day)
    2. confirm ROI / mask  render_mask_preview(video, rec.modality, require_mask(rec, mask_root, "preview"), png)
    3. analyse             analyze_morphology / analyze_legacy_rhythm / analyze_pixel_first / analyze_integrated
    4. save                save_results(rows, output_dir, name)

Import this package BEFORE anything from `cardio_organoid`, so the verbatim engine copy in this
project is the one that gets imported (an install of the original repo would otherwise shadow it).
"""
from ._engine import (EngineResolutionError, STANDALONE_VERSION, VALIDATED_CONFIG_SHA256, bootstrap, provenance,
                      verify_engine)

bootstrap()

from .integrated import analyze_integrated  # noqa: E402
from .legacy_rhythm import analyze_legacy_rhythm, reaggregate_legacy_rhythm  # noqa: E402
from .media import reference_frame, render_mask_preview  # noqa: E402
from .morphology import analyze_morphology, calibration_for_image  # noqa: E402
from .pixel_first import analyze_pixel_first, rederive_from_stored_primitives, row_from_summary  # noqa: E402
from .records import (find_record, load_records, make_record, mask_png_to_npz, observation_id, require_mask,  # noqa: E402
                      write_records)
from .results_io import save_results  # noqa: E402
from .regional_rhythm import RegionalRhythm, analyze_regional_rhythm  # noqa: E402  [added: multi-rhythm BPM]

__all__ = [
    "EngineResolutionError", "STANDALONE_VERSION", "VALIDATED_CONFIG_SHA256", "bootstrap", "provenance", "verify_engine",
    "load_records", "find_record", "make_record", "write_records", "mask_png_to_npz", "observation_id", "require_mask",
    "reference_frame", "render_mask_preview",
    "analyze_morphology", "calibration_for_image", "analyze_legacy_rhythm", "reaggregate_legacy_rhythm",
    "analyze_pixel_first", "rederive_from_stored_primitives", "row_from_summary", "analyze_integrated",
    "save_results", "RegionalRhythm", "analyze_regional_rhythm",
]
