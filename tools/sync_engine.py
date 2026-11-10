"""Copy the calculation engine VERBATIM from the original repository and record provenance.

Read-only with respect to the original repository. Writes only inside this standalone project:
  src/cardio_organoid/**   (library modules, byte-identical)
  engine_scripts/*.py      (original scripts that hold calculation code, byte-identical)
  configs/default.yaml     (byte-identical)
  configs/ENGINE_MANIFEST.json

For every copied file the manifest records:
  * sha256 of the original-repo file and of the copy (must be equal)
  * for each run manifest: whether the file inside that run's code_snapshot.tar.gz is byte-identical
    (the tarball itself is first checked against the code_snapshot_sha256 recorded in run_manifest.json)

Usage:
    python tools/sync_engine.py <original_repo_root> [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))
from import_closure import closure  # noqa: E402

# Entry points of the calculations reproduced by this project (see docs/SOURCE_CODE_MAP.md).
ENTRY_SCRIPTS = [
    "scripts/p36_conv_first_pass_screening.py",   # morphology + legacy BPM/IBI orchestration
    "scripts/p36_conv_first_pass_outputs.py",     # first-pass master table writer (integrate stage)
    "scripts/pixel_first_motion_batch.py",        # pixel-first primitives + derive
    "scripts/pixel_first_motion_aggregate.py",    # pixel-first master table writer
]
# Listed in the project's Notion code inventory but NOT on the import path of the runs above.
EXTRA_MODULES = ["src/cardio_organoid/qc/__init__.py", "src/cardio_organoid/qc/data_curation.py"]
CONFIG = "configs/default.yaml"

RUN_MANIFESTS = {
    "seg_v2": "results/_scratch_runs/p36_conv_segmentation_manifest_v2_human_reviewed_20260928T155748Z/manifest",
    "fp_morphology": "results/_scratch_runs/p36_conv_first_pass_screening_20260928T155956Z/manifests/morphology",
    "fp_functional_plan": "results/_scratch_runs/p36_conv_first_pass_screening_20260928T155956Z/manifests/functional_plan",
    "fp_functional_run": "results/_scratch_runs/p36_conv_first_pass_screening_20260928T155956Z/manifests/functional_run",
    "fp_functional_aggregate": "results/_scratch_runs/p36_conv_first_pass_screening_20260928T155956Z/manifests/functional_aggregate",
    "pf_plan": "results/_scratch_runs/pixel_first_screening_P36_Conv_9well_20260929T062953Z/manifests/plan",
    "pf_primitives": "results/_scratch_runs/pixel_first_screening_P36_Conv_9well_20260929T062953Z/manifests/primitives",
    "pf_derive": "results/_scratch_runs/pixel_first_screening_P36_Conv_9well_20260929T062953Z/manifests/derive",
}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(p.read_bytes())


def dest_for(rel: str) -> str:
    if rel.startswith("scripts/"):
        return "engine_scripts/" + rel[len("scripts/"):]
    return rel  # src/... and configs/... keep their relative location


def snapshot_hashes(repo: Path) -> dict:
    out = {}
    for name, d in RUN_MANIFESTS.items():
        man_dir = repo / d
        man = json.loads((man_dir / "run_manifest.json").read_text())
        tar_path = man_dir / man.get("code_snapshot", "code_snapshot.tar.gz")
        tar_sha = sha256_file(tar_path)
        members = {}
        with tarfile.open(tar_path, "r:gz") as tf:
            for m in tf.getmembers():
                if m.isfile() and m.name.endswith((".py", ".yaml", ".toml")):
                    name_ = m.name[2:] if m.name.startswith("./") else m.name
                    members[name_] = sha256_bytes(tf.extractfile(m).read())
        out[name] = {"run_manifest": f"{d}/run_manifest.json", "timestamp_utc": man.get("timestamp_utc"),
                     "git_commit": man.get("git_commit"), "git_dirty_code": man.get("git_dirty_code"),
                     "config_sha256": man.get("config_sha256"),
                     "code_snapshot_sha256_recorded": man.get("code_snapshot_sha256"),
                     "code_snapshot_sha256_actual": tar_sha,
                     "code_snapshot_intact": tar_sha == man.get("code_snapshot_sha256"),
                     "_members": members}
    return out


def git_state(repo: Path) -> dict:
    def g(*a):
        try:
            return subprocess.run(["git", "-c", f"safe.directory={repo}", "-C", str(repo), *a],
                                  capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as exc:  # recorded, not fatal
            return f"UNAVAILABLE: {exc!r}"
    return {"head": g("rev-parse", "HEAD"), "dirty_paths": g("status", "--porcelain").splitlines()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    repo = a.repo.resolve()

    files, third_party = closure(repo, [repo / e for e in ENTRY_SCRIPTS])
    rels = sorted({str(f.relative_to(repo)) for f in files} | set(EXTRA_MODULES) | {CONFIG})
    snaps = snapshot_hashes(repo)

    entries = []
    for rel in rels:
        src = repo / rel
        dst = PROJECT / dest_for(rel)
        if not a.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        s_sha = sha256_file(src)
        d_sha = sha256_file(dst) if dst.exists() else None
        per_snap = {}
        for name, s in snaps.items():
            h = s["_members"].get(rel)
            per_snap[name] = "absent" if h is None else ("identical" if h == s_sha else f"DIFFERS(snapshot {h[:12]})")
        entries.append({"original_path": rel, "standalone_path": dest_for(rel), "sha256_original": s_sha,
                        "sha256_standalone": d_sha, "copy_identical": s_sha == d_sha,
                        "in_import_closure": rel not in EXTRA_MODULES and rel != CONFIG, "run_snapshots": per_snap})

    manifest = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "original_repo": repo.name,  # name only; no absolute developer path
        "original_git": git_state(repo),
        "entry_scripts": ENTRY_SCRIPTS, "extra_modules": EXTRA_MODULES,
        "third_party_imports_static": sorted(third_party),
        "run_manifests": {k: {kk: vv for kk, vv in v.items() if kk != "_members"} for k, v in snaps.items()},
        "files": entries,
        "summary": {
            "n_files": len(entries),
            "all_copies_identical": all(e["copy_identical"] for e in entries),
            "files_differing_from_any_snapshot": sorted(
                e["original_path"] for e in entries if any(v.startswith("DIFFERS") for v in e["run_snapshots"].values())),
        },
    }
    if not a.dry_run:
        (PROJECT / "configs").mkdir(exist_ok=True)
        (PROJECT / "configs" / "ENGINE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(manifest["summary"], indent=2))
    print("snapshots intact:", {k: v["code_snapshot_intact"] for k, v in snaps.items()})


if __name__ == "__main__":
    main()
