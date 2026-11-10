import os
import shutil
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tests"))
import adapters  # noqa: E402,F401  (engine bootstrap before any cardio_organoid import)

WORK = PROJECT / "tests" / "_work"


@pytest.fixture
def work_dir(request) -> Path:
    """Fresh per-test output folder inside the project (outputs are inspected, so not system temp)."""
    d = WORK / request.node.name
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    return d


def pytest_collection_modifyitems(config, items):
    if os.environ.get("CARDIAC_RUN_SLOW") == "1":
        return
    skip = pytest.mark.skip(reason="slow (decodes a synthetic video, Farneback on CPU); set CARDIAC_RUN_SLOW=1")
    for it in items:
        if "slow" in it.keywords:
            it.add_marker(skip)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: decodes a synthetic video and runs Farneback on CPU (~2-3 min)")
