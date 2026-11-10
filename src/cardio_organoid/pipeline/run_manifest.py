"""run_manifest.json for every new-pipeline analysis run (CLAUDE.md #13).

Records: git commit + dirty state (+ changed code paths), config path/hash,
segmentation-interface version, selector artifact path/hash, SAM2
checkpoint/model-config identifier, timestamp, input scope and output
namespace. When the working tree is dirty, a code snapshot (src/, scripts/,
configs/, tests/, pyproject.toml; no data, no results) is archived next to
the manifest so the run stays reproducible.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

CODE_PATHS = ("src", "scripts", "configs", "tests", "pyproject.toml")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(["git", "-c", f"safe.directory={repo_root}", *args], cwd=repo_root,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_state(repo_root: Path) -> dict:
    commit = _git(repo_root, "rev-parse", "HEAD")
    porcelain = _git(repo_root, "status", "--porcelain", "--", *CODE_PATHS) or ""
    changed = [line[3:] for line in porcelain.splitlines() if line.strip()]
    return {"git_commit": commit, "git_dirty_code": bool(changed), "git_changed_code_paths": changed}


def snapshot_code(repo_root: Path, out_path: Path) -> str:
    def _filter(ti: tarfile.TarInfo):
        return None if "__pycache__" in ti.name or ti.name.endswith(".pyc") else ti
    with tarfile.open(out_path, "w:gz") as tar:
        for p in CODE_PATHS:
            if (repo_root / p).exists():
                tar.add(repo_root / p, arcname=p, filter=_filter)
    return sha256_file(out_path)


def write_run_manifest(out_dir: Path, repo_root: Path, *, config_path: Path, interface_version: str,
                       selector_artifact: Path | None, sam2_checkpoint: Path | None, sam2_model_cfg: str | None,
                       input_scope: dict, extra: dict | None = None, hash_sam2_checkpoint: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    state = git_state(repo_root)
    manifest = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        **state,
        "config_path": str(config_path.relative_to(repo_root)),
        "config_sha256": sha256_file(config_path),
        "segmentation_interface_version": interface_version,
        "selector_artifact": str(selector_artifact.relative_to(repo_root)) if selector_artifact else None,
        "selector_artifact_sha256": sha256_file(selector_artifact) if selector_artifact else None,
        "sam2_checkpoint": str(sam2_checkpoint.relative_to(repo_root)) if sam2_checkpoint else None,
        "sam2_checkpoint_sha256": sha256_file(sam2_checkpoint) if (sam2_checkpoint and hash_sam2_checkpoint) else None,
        "sam2_model_cfg": sam2_model_cfg,
        "input_scope": input_scope,
        "output_namespace": str(out_dir.resolve().relative_to(repo_root)),
        **(extra or {}),
    }
    if state["git_dirty_code"]:
        manifest["code_snapshot"] = "code_snapshot.tar.gz"
        manifest["code_snapshot_sha256"] = snapshot_code(repo_root, out_dir / "code_snapshot.tar.gz")
    with open(out_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest
