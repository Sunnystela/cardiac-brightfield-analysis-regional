"""Engine bootstrap, isolation guard and provenance.

The calculation engine is a byte-identical copy of the original repository's code
(`src/cardio_organoid/**`, `engine_scripts/*.py`, `configs/default.yaml`; see
configs/ENGINE_MANIFEST.json). Adapters never edit it. They only:

  * put this project's `engine_scripts/` and `src/` first on sys.path (the same order the
    original scripts use for themselves),
  * refuse to run if any engine module resolves outside this project (an editable install
    of the original repo would otherwise shadow the copy silently),
  * temporarily rebind module-level PATH constants (never numeric constants) so the
    original batch functions read/write the user's chosen locations.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
ENGINE_SCRIPTS_DIR = PROJECT_ROOT / "engine_scripts"
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
ENGINE_MANIFEST_PATH = PROJECT_ROOT / "configs" / "ENGINE_MANIFEST.json"

STANDALONE_VERSION = "0.1.0"
# configs/default.yaml sha256 recorded in every P36/Conv first-pass and pixel-first run manifest.
VALIDATED_CONFIG_SHA256 = "aab9e85b92ff709b0a37be0620f41676208552b424ada2f74d5e306804cd67fb"
PACKAGES = ("numpy", "opencv-python-headless", "opencv-python", "scipy", "scikit-learn", "torch",
            "matplotlib", "PyYAML", "joblib")


class EngineResolutionError(RuntimeError):
    """An engine module was imported from outside this standalone project."""


def bootstrap() -> None:
    for p in (str(SRC_DIR), str(ENGINE_SCRIPTS_DIR)):  # final order: engine_scripts, src, ...
        while p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    assert_engine_local()


def _engine_module_names() -> set[str]:
    return {p.stem for p in ENGINE_SCRIPTS_DIR.glob("*.py")}


def assert_engine_local() -> None:
    scripts = _engine_module_names()
    for name, mod in list(sys.modules.items()):
        if name == "cardio_organoid" or name.startswith("cardio_organoid.") or name in scripts:
            f = getattr(mod, "__file__", None)
            if f and not Path(f).resolve().is_relative_to(PROJECT_ROOT):
                raise EngineResolutionError(
                    f"engine module {name!r} was imported from {f}, not from this project. "
                    "Start a fresh Python process and import `adapters` before any `cardio_organoid` import.")


def engine(module_name: str):
    """Import an engine module (library dotted name or engine_scripts module name)."""
    bootstrap()
    mod = importlib.import_module(module_name)
    assert_engine_local()
    return mod


@contextlib.contextmanager
def rebind_paths(module, **paths):
    """Temporarily set module-level path constants (e.g. REPO_ROOT, MANIFEST). Not thread-safe."""
    for k, v in paths.items():
        if not isinstance(v, Path):
            raise TypeError(f"rebind_paths only rebinds Path constants; {k}={v!r}")
        if not hasattr(module, k):
            raise AttributeError(f"{module.__name__} has no attribute {k}")
    old = {k: getattr(module, k) for k in paths}
    try:
        for k, v in paths.items():
            setattr(module, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(module, k, v)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def verify_engine() -> dict:
    """Re-hash every engine file listed in ENGINE_MANIFEST.json."""
    man = json.loads(ENGINE_MANIFEST_PATH.read_text(encoding="utf-8"))
    modified, missing = [], []
    for e in man["files"]:
        p = PROJECT_ROOT / e["standalone_path"]
        if not p.exists():
            missing.append(e["standalone_path"])
        elif sha256_file(p) != e["sha256_standalone"]:
            modified.append(e["standalone_path"])
    return {"n_files": len(man["files"]), "modified": modified, "missing": missing,
            "engine_unmodified": not modified and not missing}


def package_versions() -> dict:
    out = {}
    for n in PACKAGES:
        try:
            out[n] = importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:
            out[n] = None
    return out


def provenance(extra: dict | None = None) -> dict:
    man = json.loads(ENGINE_MANIFEST_PATH.read_text(encoding="utf-8"))
    cfg_sha = sha256_file(CONFIG_PATH)
    return {
        "standalone_version": STANDALONE_VERSION,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "engine_manifest_sha256": sha256_file(ENGINE_MANIFEST_PATH),
        "engine_source_git_head": man["original_git"]["head"],
        "engine_source_git_dirty": bool(man["original_git"]["dirty_paths"]),
        "engine_verification": verify_engine(),
        "config_path": "configs/default.yaml",
        "config_sha256": cfg_sha,
        "config_matches_validated_runs": cfg_sha == VALIDATED_CONFIG_SHA256,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        **(extra or {}),
    }
