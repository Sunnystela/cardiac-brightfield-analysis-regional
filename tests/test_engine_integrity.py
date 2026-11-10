"""The engine must be the unmodified copy recorded in configs/ENGINE_MANIFEST.json, and the one imported."""
import json
import sys
from pathlib import Path

import adapters
from adapters._engine import CONFIG_PATH, ENGINE_MANIFEST_PATH, PROJECT_ROOT, engine, sha256_file


def test_engine_files_unmodified():
    v = adapters.verify_engine()
    assert v["engine_unmodified"], v


def test_config_is_the_validated_run_config():
    assert sha256_file(CONFIG_PATH) == adapters.VALIDATED_CONFIG_SHA256


def test_manifest_records_identical_copies_and_intact_snapshots():
    man = json.loads(ENGINE_MANIFEST_PATH.read_text())
    assert man["summary"]["all_copies_identical"]
    assert all(v["code_snapshot_intact"] for v in man["run_manifests"].values())
    # only the three scripts documented in docs/SOURCE_CODE_MAP.md differ from some run snapshot
    assert set(man["summary"]["files_differing_from_any_snapshot"]) == {
        "scripts/p36_conv_first_pass_outputs.py", "scripts/pixel_first_motion_aggregate.py", "scripts/pixel_first_motion_batch.py"}


def test_engine_modules_resolve_inside_project():
    for name in ("p36_conv_first_pass_screening", "longitudinal_modality_aware_dry_run", "pixel_first_motion_batch",
                 "cardio_organoid.morphology.features", "cardio_organoid.functional.period_evidence_confidence"):
        assert Path(engine(name).__file__).resolve().is_relative_to(PROJECT_ROOT), name
    leaked = [n for n, m in sys.modules.items() if n.startswith("cardio_organoid")
              and getattr(m, "__file__", None) and not Path(m.__file__).resolve().is_relative_to(PROJECT_ROOT)]
    assert not leaked


def test_functional_bpm_module_not_on_the_engine_path():
    # Older inventories name functional/bpm.py as the BPM implementation; in the original repository it is an
    # empty (0-byte) file that no run imports, so it is not part of this engine and nothing may wire it in.
    man = json.loads(ENGINE_MANIFEST_PATH.read_text())
    assert "src/cardio_organoid/functional/bpm.py" not in {e["original_path"] for e in man["files"]}
    assert not (PROJECT_ROOT / "src/cardio_organoid/functional/bpm.py").exists()
    adapter_src = "".join(p.read_text() for p in (PROJECT_ROOT / "adapters").glob("*.py"))
    assert "functional.bpm" not in adapter_src
