"""Reproduction of stored P36/Conv results (needs read access to the original repository).

Opt-in: set CARDIAC_ORIGINAL_REPO to the original repository root. Runs the light checks only
(morphology, legacy re-aggregation, pixel-first column mapping); the derive re-run is in
`python launch.py validate`.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ORIG = os.environ.get("CARDIAC_ORIGINAL_REPO")
PROJECT = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(not ORIG, reason="set CARDIAC_ORIGINAL_REPO")
def test_light_reproduction_all_match(work_dir):
    subprocess.run([sys.executable, str(PROJECT / "validation" / "run_validation.py"), "--original-repo", ORIG,
                    "--skip-rederive", "--out", str(work_dir)], check=True, capture_output=True)
    s = json.loads((work_dir / "validation_summary.json").read_text())
    assert s["all_match"], s["mismatches"][:10]
    assert {"M_morphology", "L_legacy_rhythm", "PM_pixel_first_mapping", "LC_legacy_convention"} <= set(s["groups"])
