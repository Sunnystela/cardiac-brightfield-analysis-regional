"""Compare the standalone adapters against stored P36/Conv results of the original repository.

CPU-only and light: no SAM2, no optical flow, no video decoding. Reads the original repository
(read-only) and writes only under this project's validation/ folder.

Checks
  M  morphology   adapters.analyze_morphology on stored manifest-v2 masks
                  vs first-pass morphology/morphology_results.json
  L  legacy       adapters.reaggregate_legacy_rhythm (post_motion_functional on the stored motion
                  caches) vs first-pass functional/functional_results.json
  PM pf mapping   adapters.row_from_summary on every stored pixel-first summary.json
                  vs aggregate/MASTER_9WELL_130.csv
  LC legacy conv  integrated.legacy_columns_pixel_first_convention on first-pass rows
                  vs the legacy_* columns of MASTER_9WELL_130.csv
  PD pf derive    adapters.rederive_from_stored_primitives (engine `derive` on stored int16 flow)
                  vs stored summary.json and MASTER_9WELL_130.csv   [--skip-rederive to omit]

Usage:
    python validation/run_validation.py --original-repo <path> [--skip-rederive]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
import adapters  # noqa: E402  (must precede any cardio_organoid import)
from adapters.integrated import legacy_columns_pixel_first_convention  # noqa: E402

SEG_MANIFEST = "results/_scratch_runs/p36_conv_segmentation_manifest_v2_human_reviewed_20260928T155748Z"
FIRST_PASS = "results/_scratch_runs/p36_conv_first_pass_screening_20260928T155956Z"
PIXEL_FIRST = "results/_scratch_runs/pixel_first_screening_P36_Conv_9well_20260929T062953Z"

MORPH_CASES = [("a1", 1), ("a1", 20), ("f3", 20), ("b2", 21), ("e1", 17)]  # worked examples + multi-component + holes
MORPH_COLS = ["morphology_status", "calibration_status", "um_per_pixel", "area_px", "perimeter_px", "equivalent_diameter_px",
              "circularity", "solidity", "eccentricity", "aspect_ratio", "radial_cv", "perimeter_over_sqrt_area",
              "area_um2", "equivalent_diameter_um"]
LEGACY_FIXED = ["P36_Conv_a1_D20_VIDEO", "P36_Conv_f3_D20_VIDEO"]
LEGACY_COLS = ["functional_qc_status", "candidate_bpm", "bpm", "fundamental_bpm", "median_ibi_s", "ibi_cv",
               "consensus_beat_count", "reliable_active_fraction", "observable_fraction", "n_total_patches",
               "n_reliable_patches", "n_period_families", "period_family_bpms_descriptive", "actual_fps", "frame_step",
               "triggered_failure_modes"]
REDERIVE = ["P36_Conv_a1_D20_VIDEO", "P36_Conv_h3_D20_VIDEO"]  # REUSED_PILOT primitives + COMPUTED primitives
LEGACY_CONV_COLS = ["legacy_bpm_reported", "legacy_bpm", "legacy_ibi_cv", "legacy_median_ibi_s"]


def norm(v):
    if isinstance(v, str):
        if v in ("", "None"):
            return None
        if v in ("True", "False"):
            return v == "True"
        try:
            return float(v)
        except ValueError:
            return v
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return float(v)
    return v


def compare(orig, new):
    a, b = norm(orig), norm(new)
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True, 0.0
        d = abs(a - b)
        return a == b, d
    return a == b, None


class Log:
    def __init__(self):
        self.rows = []

    def add(self, group, obs, metric, orig, new, note=""):
        ok, d = compare(orig, new)
        self.rows.append({"check": group, "observation_id": obs, "metric": metric,
                          "original_value": json.dumps(orig) if isinstance(orig, (list, dict)) else orig,
                          "standalone_value": json.dumps(new) if isinstance(new, (list, dict)) else new,
                          "abs_diff": d, "match": ok, "note": note})


def walk_numeric(a, b, path=""):
    """Yield (path, a, b) for every leaf of two JSON structures."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            yield from walk_numeric(a.get(k), b.get(k), f"{path}.{k}" if path else k)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            yield from walk_numeric(x, y, f"{path}[{i}]")
    else:
        yield path, a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original-repo", type=Path, required=True)
    ap.add_argument("--skip-rederive", action="store_true")
    ap.add_argument("--out", type=Path, default=PROJECT / "validation" / "results")
    a = ap.parse_args()
    repo = a.original_repo.resolve()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = Log()
    timing = {}

    records = adapters.load_records(repo / SEG_MANIFEST / "segmentation_interface_records.json")
    rev = {(r["well"], int(r["day"])): r for r in csv.DictReader(open(repo / SEG_MANIFEST / "SEGMENTATION_REVISION_TABLE.csv", newline=""))}

    # ---------------------------------------------------------------- M: morphology
    t = time.time()
    stored_m = {r["observation_id"]: r for r in json.loads((repo / FIRST_PASS / "morphology/morphology_results.json").read_text())}
    cases = [adapters.find_record(records, w, d) for w, d in MORPH_CASES]
    cases += [next(r for r in records if r.segmentation_qc_status == "NO_ACCEPTABLE_CANDIDATE")]  # gating case
    for rec in cases:
        src = repo / rev[(rec.well, rec.day)]["source_file"]
        new = adapters.analyze_morphology(rec, repo, src)
        old = stored_m[adapters.observation_id(rec)]
        for c in MORPH_COLS:
            log.add("M_morphology", new["observation_id"], c, old.get(c), new.get(c),
                    "sidecar read from original data folder" if c == "um_per_pixel" else "")
    timing["morphology_s"] = time.time() - t

    # ---------------------------------------------------------------- L: legacy re-aggregation
    t = time.time()
    fres = {r["observation_id"]: r for r in json.loads((repo / FIRST_PASS / "functional/functional_results.json").read_text())}
    plan = {p["observation_id"]: p for p in json.loads((repo / FIRST_PASS / "functional/functional_plan.json").read_text())}
    vids = [r for r in fres.values() if r["modality"] == "VIDEO" and r["execution_status"] in ("RERUN", "REUSED")]
    picks = list(LEGACY_FIXED)
    for status, qc in (("RERUN", "LOW_CONFIDENCE"), ("RERUN", "NOT_ESTIMABLE"), ("REUSED", "ESTIMABLE"),
                       ("REUSED", "NOT_ESTIMABLE"), ("REUSED", "LOW_CONFIDENCE")):
        picks.append(next(r["observation_id"] for r in sorted(vids, key=lambda x: x["observation_id"])
                          if r["execution_status"] == status and r["functional_qc_status"] == qc
                          and r["observation_id"] not in picks and (r.get("n_reliable_patches") or 0) > 0))
    for obs in picks:
        old = fres[obs]
        rec = next(r for r in records if adapters.observation_id(r) == obs)
        if old["execution_status"] == "RERUN":
            d = repo / FIRST_PASS / "functional/rerun" / obs
            motion_dir, r36 = d / "motion_pipeline" / obs, d / "round36" / f"{obs}_round36_raw_per_patch.json"
        else:
            motion_dir, r36 = repo / plan[obs]["motion_dir"], repo / plan[obs]["round36"]
        new = adapters.reaggregate_legacy_rhythm(repo / plan[obs]["source_file"], rec, motion_dir, r36)
        for c in LEGACY_COLS:
            log.add("L_legacy_rhythm", obs, c, old.get(c), new.get(c), f"stored cache: {old['execution_status']}")
    timing["legacy_s"] = time.time() - t

    # ---------------------------------------------------------------- PM / LC: pixel-first master mapping
    t = time.time()
    master = list(csv.DictReader(open(repo / PIXEL_FIRST / "aggregate/MASTER_9WELL_130.csv", newline="")))
    fp_master = {(r["well"], int(r["day"])): r for r in
                 csv.DictReader(open(repo / FIRST_PASS / "population/P36_CONV_MASTER_OBSERVATION_TABLE.csv", newline=""))}
    n_pm = 0
    for m in master:
        od = repo / PIXEL_FIRST / "observations" / m["observation_id"]
        fpm = fp_master[(m["well"], int(m["day"]))]
        conv = legacy_columns_pixel_first_convention({"functional_qc_status": fpm["functional_qc_status"], "bpm": norm(fpm["bpm"]),
                                                      "ibi_cv": norm(fpm["ibi_cv"]), "median_ibi_s": norm(fpm["median_ibi_s"])})
        for c in LEGACY_CONV_COLS:
            log.add("LC_legacy_convention", m["observation_id"], c, m[c], conv[c])
        if (od / "summary.json").exists():
            row = adapters.row_from_summary(json.loads((od / "summary.json").read_text()), json.loads((od / "primitives.json").read_text()))
            for c, v in row.items():
                log.add("PM_pixel_first_mapping", m["observation_id"], c, m.get(c), v)
            n_pm += 1
    timing["pf_mapping_s"] = time.time() - t

    # ---------------------------------------------------------------- PD: pixel-first derive re-run
    if not a.skip_rederive:
        for obs in REDERIVE:
            t = time.time()
            od = repo / PIXEL_FIRST / "observations" / obs
            stored = json.loads((od / "summary.json").read_text())
            meta = {k: stored[k] for k in ("observation_id", "well", "day", "role")}
            res = adapters.rederive_from_stored_primitives(od, repo, out / "pixel_first_rederive" / obs, meta, render_figures=False)
            new = json.loads(json.dumps(res["summary"], default=str))
            n_leaf = n_bad = 0
            for path, x, y in walk_numeric(stored, new):
                ok, d = compare(x, y)
                n_leaf += 1
                if not ok:
                    n_bad += 1
                    log.add("PD_pixel_first_derive_summary", obs, path, x, y)
            log.add("PD_pixel_first_derive_summary", obs, "ALL_SUMMARY_LEAVES_IDENTICAL", True, n_bad == 0,
                    f"{n_leaf} leaves compared, {n_bad} differ")
            import numpy as np
            for npz in ("derived_timeseries.npz", "derived_maps.npz"):
                za, zb = np.load(od / npz), np.load(out / "pixel_first_rederive" / obs / npz)
                same_keys = sorted(za.files) == sorted(zb.files)
                diff = [k for k in za.files if k in zb.files and not np.array_equal(za[k], zb[k], equal_nan=True)]
                log.add("PD_pixel_first_derive_arrays", obs, f"{npz}:ALL_ARRAYS_IDENTICAL", True, same_keys and not diff,
                        f"{len(za.files)} arrays; differing: {diff}; same key set: {same_keys}")
            m = next(r for r in master if r["observation_id"] == obs)
            for c, v in res["row"].items():
                log.add("PD_pixel_first_derive_row", obs, c, m.get(c), v)
            timing[f"rederive_{obs}_s"] = time.time() - t

    # ---------------------------------------------------------------- write
    res_csv = out / "VALIDATION_RESULTS.csv"
    with open(res_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(log.rows[0]))
        w.writeheader()
        w.writerows(log.rows)
    by = {}
    for r in log.rows:
        g = by.setdefault(r["check"], {"n": 0, "match": 0, "mismatch": 0, "observations": set()})
        g["n"] += 1
        g["match" if r["match"] else "mismatch"] += 1
        g["observations"].add(r["observation_id"])
    summary = {"original_repo": repo.name, "pixel_first_mapping_rows": n_pm, "rederive_run": not a.skip_rederive,
               "groups": {k: {**v, "observations": len(v["observations"])} for k, v in by.items()},
               "all_match": all(r["match"] for r in log.rows), "timing_s": timing,
               "mismatches": [r for r in log.rows if not r["match"]][:200]}
    (out / "validation_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "validation_provenance.json").write_text(json.dumps(adapters.provenance(), indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "mismatches"}, indent=2, default=str))
    print("mismatches:", len([r for r in log.rows if not r["match"]]))


if __name__ == "__main__":
    main()
