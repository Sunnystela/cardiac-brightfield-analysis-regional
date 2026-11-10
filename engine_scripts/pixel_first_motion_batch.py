"""Pixel-first motion screening -- P36/Conv 9-well allowlist (exploratory).

Method fixed to the owner-approved a1 D20 pilot
(scripts/pixel_first_motion_diagnostic.py; constants imported from it).
Stages (resumable, one checkpoint per observation):
  plan        inventory + input verification + frozen method config
  primitives  one pass per VIDEO: Farneback on the unmasked ROI, drift correction,
              validity maps, int16 flow chunks, background block series
              (pilot pass logic; a1 D20 reuses the pilot primitives after an
              equivalence check)
  derive      everything else from the stored primitives: strict-inside vs
              outer-boundary-support vs background summaries, motion-pattern
              regions (pilot rule), periods, timing, figures, movie
  aggregate   per-well and population outputs, review ZIP
No SAM2/selector run, no legacy fixed-idx1 fallback, no threshold fitting, no
canonical write. Allowlist only; P38 not accessed.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt_
import hashlib
import json
import multiprocessing as mp
import os
import sys
import traceback
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pixel_first_motion_diagnostic as PILOT  # noqa: E402  (fixed constants)
from cardio_organoid.motion.raft_flow import FARNEBACK_DEFAULT_PARAMS  # noqa: E402
from cardio_organoid.pipeline.run_manifest import sha256_file, write_run_manifest  # noqa: E402
from cardio_organoid.pipeline.segmentation_interface import (  # noqa: E402
    INTERFACE_VERSION, SegmentationRecord, fail_closed_mode, require_downstream_mask,
)

ALLOW = ["a1", "b2", "c1", "e1", "e3", "e4", "f1", "h2", "h3"]
PRIMARY = ["a1", "b2", "c1", "e1", "e3", "h2", "h3"]
MANIFEST = PILOT.MANIFEST
FIRST_PASS = REPO_ROOT / "results/_scratch_runs/p36_conv_first_pass_screening_20260928T155956Z"
PILOT_OUT = REPO_ROOT / "results/_scratch_runs/pixel_first_motion_diagnostic_P36_Conv_a1_D20_20260928T195044Z"
CONFIG_PATH = REPO_ROOT / "configs/default.yaml"
MAX_LAG_PAIRS = 5
DISPERSION_SAMPLE = 300
DISPERSION_EXCLUDE_BLOCKS = 10


def rel(p):
    return str(Path(p).resolve().relative_to(REPO_ROOT))


def jdump(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str, ensure_ascii=False)


def fixed_method():
    from global_boundary_motion_diagnostic import _load_config
    cfg = _load_config()
    return {
        "reference_implementation": "scripts/pixel_first_motion_diagnostic.py",
        "reference_pilot": rel(PILOT_OUT), "segmentation_manifest": rel(MANIFEST),
        "segmentation_manifest_sha256": sha256_file(MANIFEST / "segmentation_interface_records.json"),
        "frame_range": "all frames of the file; pairs (f, f+frame_step) for f = 0, step, 2*step, ...",
        "fps": "file-recorded (cv2.CAP_PROP_FPS)", "frame_step": cfg["motion"]["spatial_patches"]["frame_step"],
        "roi": f"mask bbox + motion.roi.expansion_px={cfg['motion']['roi']['expansion_px']} clipped to frame (existing rule)",
        "mask_reference_frame": "video midpoint frame_count//2 (mask defined there)",
        "farneback": FARNEBACK_DEFAULT_PARAMS, "drift": "per-pair whole-ROI median (remove_global_drift)",
        "primary_signal": "|u| = sqrt(ux^2+uy^2) per pixel of drift-corrected flow (norm first)",
        "comparison_signals": "signed radial and tangential projections from the mask centroid (centroid-based, NOT local boundary normal)",
        "validity": {"saturated": f"<= {PILOT.SAT_LO} or >= {PILOT.SAT_HI} in >= {PILOT.SAT_FRAC} of every-10th frames",
                     "low_texture": f"Sobel gradient (Gaussian sigma 3) < {PILOT.LOW_TEXTURE_GRAD} on the representative frame",
                     "fb_inconsistent": f"median forward-backward error > {PILOT.FB_MAX} px/pair over every {PILOT.FB_EVERY}th pair",
                     "time_varying": False},
        "regions": {"INSIDE_REFERENCE_MASK": "static manifest-v2 mask",
                    "OUTER_BOUNDARY_SUPPORT": f"mask dilated {PILOT.ADJ_BAND} px minus mask (reference only; never counted as tissue)",
                    "BACKGROUND_REFERENCE": f"> {PILOT.BG_MIN_DIST} px from the mask, within the functional ROI"},
        "storage_crop_pad_px": PILOT.CROP_PAD, "block_px": PILOT.BLOCK, "min_region_blocks": PILOT.MIN_REGION_BLOCKS,
        "clustering": "active blocks (block |u| temporal std > p99 of same-size fully valid background block stds over the "
                      "whole ROI); spatial 8-connectivity; AgglomerativeClustering(average, cosine on z-scored |u|, "
                      "distance_threshold 0.5) within each connected component only",
        "peaks": "simple_peak_indices (>= 0.3 s) above median + 5*MAD (pilot rule)",
        "periods": "pilot rule (first local autocorrelation max) + separately labelled first positive-autocorrelation max",
        "flow_storage": f"int16 = round(u*{PILOT.FLOW_Q}) (0.01 px/pair), clip +-32767, invalid code -32768 (unused: Farneback emits no NaN), "
                        f"{PILOT.CHUNK} pairs/chunk",
        "units": "displacement px/pair; speed px/s = px/pair / dt, dt = frame_step / fps; no um conversion (VIDEO uncalibrated)",
        "timing": f"best lag within +-{MAX_LAG_PAIRS} pairs, evaluated separately in first and second halves (continuous time blocks)",
    }


# ------------------------------------------------------------------ plan
def stage_plan(out):
    method = fixed_method()
    jdump(out / "config" / "FROZEN_METHOD.json", method)
    recs = [r for r in json.loads((MANIFEST / "segmentation_interface_records.json").read_text())["records"] if r["well"] in ALLOW]
    master = {(r["well"], int(r["day"])): r for r in csv.DictReader(open(FIRST_PASS / "population/P36_CONV_MASTER_OBSERVATION_TABLE.csv"))}
    inv = []
    with fail_closed_mode():
        for r in sorted(recs, key=lambda x: (x["well"], x["day"])):
            rec = SegmentationRecord.from_dict(r)
            m = master[(rec.well, rec.day)]
            row = {"observation_id": m["observation_id"], "well": rec.well, "day": rec.day, "modality": rec.modality,
                   "role": "primary" if rec.well in PRIMARY else "supplementary", "source_file": m["source_file"],
                   "reference_frame": m["reference_frame"], "segmentation_qc_status": rec.segmentation_qc_status,
                   "selected_candidate": rec.selected_candidate, "mask_path": rec.mask_path}
            if rec.modality != "VIDEO":
                row["pixel_first_status"] = "NOT_APPLICABLE_IMAGE"
                inv.append(row)
                continue
            try:
                mask = require_downstream_mask(rec, REPO_ROOT, "pixel_first_plan")
                cap = cv2.VideoCapture(str(REPO_ROOT / m["source_file"]))
                fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, fc // 2)
                ok, fr = cap.read()
                cap.release()
                if not ok or fr.shape[:2] != mask.shape or f"frame_index={fc // 2}/frame_count={fc}" not in m["reference_frame"]:
                    raise ValueError(f"frame/mask mismatch: ok={ok} frame={None if not ok else fr.shape[:2]} mask={mask.shape} ref={m['reference_frame']}")
                row.update(frame_count=fc, frame_shape=list(fr.shape[:2]), mask_sha256=sha256_file(REPO_ROOT / rec.mask_path),
                           pixel_first_status="PLANNED")
                if (rec.well, rec.day) == (PILOT.WELL, PILOT.DAY):
                    row["pixel_first_status"] = "REUSE_CANDIDATE_PILOT"
            except Exception as exc:
                row.update(pixel_first_status="INPUT_REVIEW_REQUIRED", input_problem=repr(exc))
            inv.append(row)
    jdump(out / "inventory.json", inv)
    write_run_manifest(out / "manifests" / "plan", REPO_ROOT, config_path=CONFIG_PATH, interface_version=INTERFACE_VERSION,
                       selector_artifact=None, sam2_checkpoint=None, sam2_model_cfg=None,
                       input_scope={"allowlist": ALLOW, "primary": PRIMARY, "n_observations": len(inv), "p38": "not accessed"},
                       extra={"frozen_method_sha256": sha256_file(out / "config" / "FROZEN_METHOD.json")}, hash_sam2_checkpoint=False)
    from collections import Counter
    print(len(inv), Counter(r["modality"] for r in inv), Counter(r["pixel_first_status"] for r in inv))


# ------------------------------------------------------------------ primitives (pilot pass logic)
def compute_primitives(obs, out_dir: Path):
    from cardio_organoid.motion.radial_flow import remove_global_drift
    from cardio_organoid.motion.raft_flow import compute_dense_optical_flow, compute_motion_roi
    from global_boundary_motion_diagnostic import _load_config
    cfg = _load_config()
    step = cfg["motion"]["spatial_patches"]["frame_step"]
    with fail_closed_mode():
        recs = {(r["well"], r["day"]): r for r in json.loads((MANIFEST / "segmentation_interface_records.json").read_text())["records"]}
        rec = SegmentationRecord.from_dict(recs[(obs["well"], obs["day"])])
        mask = require_downstream_mask(rec, REPO_ROOT, "pixel_first_primitives")
    video = REPO_ROOT / obs["source_file"]
    cap = cv2.VideoCapture(str(video))
    fc, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    dt = step / fps
    cap.set(cv2.CAP_PROP_POS_FRAMES, fc // 2)
    _, rep = cap.read()
    roi = compute_motion_roi(mask, expansion_px=cfg["motion"]["roi"]["expansion_px"], frame_shape=rep.shape[:2])
    rx0, ry0, rx1, ry1 = roi.roi_bbox_xyxy
    centroid = roi.centroid_xy_in_roi()
    local_roi = replace(roi, roi_bbox_xyxy=(0, 0, rx1 - rx0, ry1 - ry0))
    mroi = mask[ry0:ry1, rx0:rx1]
    x, y, w, h = cv2.boundingRect(mroi.astype(np.uint8))
    cx0, cy0 = max(0, x - PILOT.CROP_PAD), max(0, y - PILOT.CROP_PAD)
    cx1, cy1 = min(mroi.shape[1], x + w + PILOT.CROP_PAD), min(mroi.shape[0], y + h + PILOT.CROP_PAD)
    crop = (slice(cy0, cy1), slice(cx0, cx1))
    m8 = mroi.astype(np.uint8)
    dil = lambda k: cv2.dilate(m8, np.ones((3, 3), np.uint8), iterations=k) > 0
    tissue, adj = mroi[crop], (dil(PILOT.ADJ_BAND) & ~mroi)[crop]
    bg_full = ~dil(PILOT.BG_MIN_DIST)
    sat_count, n_s = np.zeros(mroi.shape, np.int32), 0
    for f in range(0, fc, 10):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)[ry0:ry1, rx0:rx1]
        sat_count += ((g <= PILOT.SAT_LO) | (g >= PILOT.SAT_HI))
        n_s += 1
    saturated = sat_count / n_s >= PILOT.SAT_FRAC
    rg = cv2.GaussianBlur(cv2.cvtColor(rep, cv2.COLOR_BGR2GRAY)[ry0:ry1, rx0:rx1].astype(np.float32), (0, 0), 3)
    low_texture = np.hypot(cv2.Sobel(rg, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(rg, cv2.CV_32F, 0, 1, ksize=3)) < PILOT.LOW_TEXTURE_GRAD
    fh, fw = mroi.shape
    B = PILOT.BLOCK
    fbg = (bg_full & ~saturated & ~low_texture)[: fh // B * B, : fw // B * B].reshape(fh // B, B, fw // B, B)
    bg_full_blocks = fbg.all(axis=(1, 3))
    fdir = out_dir / "flow_chunks"
    fdir.mkdir(parents=True, exist_ok=True)
    buf_ux, buf_uy, drifts, fb_err, bg_series, bg_med, bg_p95, pairs = [], [], [], [], [], [], [], []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    prev, fi, k, cid = None, 0, 0, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fi % step == 0:
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)[ry0:ry1, rx0:rx1]
            if prev is not None:
                flow = compute_dense_optical_flow(prev, g, local_roi, method="opencv_farneback")
                corr, drift = remove_global_drift(flow)
                drifts.append(drift)
                if k % PILOT.FB_EVERY == 0:
                    back = compute_dense_optical_flow(g, prev, local_roi, method="opencv_farneback")
                    gx, gy = np.meshgrid(np.arange(fw, dtype=np.float32), np.arange(fh, dtype=np.float32))
                    bx = cv2.remap(back[..., 0], gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR)
                    by = cv2.remap(back[..., 1], gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR)
                    fb_err.append(np.hypot(flow[..., 0] + bx, flow[..., 1] + by)[crop].astype(np.float32))
                ux, uy = corr[..., 0][crop], corr[..., 1][crop]
                buf_ux.append(np.clip(np.round(ux * PILOT.FLOW_Q), -32767, 32767).astype(np.int16))
                buf_uy.append(np.clip(np.round(uy * PILOT.FLOW_Q), -32767, 32767).astype(np.int16))
                fm = np.hypot(corr[..., 0], corr[..., 1])
                bmag = fm[bg_full & ~saturated]
                bg_med.append(float(np.median(bmag)))
                bg_p95.append(float(np.percentile(bmag, 95)))
                bg_series.append(fm[: fh // B * B, : fw // B * B].reshape(fh // B, B, fw // B, B).mean(axis=(1, 3))[bg_full_blocks].astype(np.float32))
                pairs.append((fi - step, fi))
                k += 1
                if len(buf_ux) == PILOT.CHUNK:
                    np.savez_compressed(fdir / f"chunk_{cid:03d}.npz", ux=np.stack(buf_ux), uy=np.stack(buf_uy), pair_index_start=k - PILOT.CHUNK)
                    buf_ux, buf_uy, cid = [], [], cid + 1
            prev = g
        fi += 1
    cap.release()
    if buf_ux:
        np.savez_compressed(fdir / f"chunk_{cid:03d}.npz", ux=np.stack(buf_ux), uy=np.stack(buf_uy), pair_index_start=k - len(buf_ux))
    fb_med = np.median(np.stack(fb_err), axis=0)
    np.savez_compressed(out_dir / "validity.npz", tissue_static_mask=tissue, adjacent_band=adj, background_reference=bg_full[crop],
                        saturated=saturated[crop], low_texture=low_texture[crop], fb_median_error=fb_med.astype(np.float32),
                        fb_inconsistent=fb_med > PILOT.FB_MAX)
    np.savez_compressed(out_dir / "series.npz", background_block_magnitude_fullroi=np.stack(bg_series), background_median_magnitude=np.array(bg_med),
                        background_p95_magnitude=np.array(bg_p95), drift_xy=np.array(drifts), pair_frames=np.array(pairs))
    meta = {"observation_id": obs["observation_id"], "source": "COMPUTED", "video": obs["source_file"], "frame_count": fc, "fps": fps,
            "frame_step": step, "dt_s": dt, "n_pairs": k, "functional_roi_xyxy_frame": [rx0, ry0, rx1, ry1],
            "storage_crop_xyxy_in_roi": [cx0, cy0, cx1, cy1], "centroid_xy_in_roi": list(centroid),
            "mask_path": rec.mask_path, "mask_sha256": sha256_file(REPO_ROOT / rec.mask_path),
            "flow_dir": rel(fdir), "validity": rel(out_dir / "validity.npz"), "series": rel(out_dir / "series.npz"),
            "n_background_blocks_fullroi": int(bg_full_blocks.sum())}
    jdump(out_dir / "primitives.json", meta)
    return meta


def reuse_pilot_primitives(obs, out_dir: Path):
    """a1 D20: link the pilot's primitives after verifying inputs/method/files."""
    s = json.loads((PILOT_OUT / "diagnostic_summary.json").read_text())
    pm = np.load(PILOT_OUT / "pixel_maps.npz")
    bw = np.load(PILOT_OUT / "block_waveforms.npz")
    checks = {"video": s["video"] == obs["source_file"], "mask": s["mask"] == obs["mask_path"],
              "mask_sha": sha256_file(REPO_ROOT / obs["mask_path"]) == obs["mask_sha256"],
              "farneback": s["farneback_params"] == FARNEBACK_DEFAULT_PARAMS, "n_pairs": s["n_pairs"] == 309,
              "chunks_cover_pairs": sum(np.load(f)["ux"].shape[0] for f in sorted((PILOT_OUT / "flow_chunks").glob("chunk_*.npz"))) == s["n_pairs"],
              "int16_storage": np.load(sorted((PILOT_OUT / "flow_chunks").glob("chunk_*.npz"))[0])["ux"].dtype == np.int16,
              "bg_fullroi_series": "background_block_magnitude_fullroi" in bw.files}
    if not all(checks.values()):
        raise RuntimeError(f"pilot reuse checks failed: {checks}")
    np.savez_compressed(out_dir / "validity.npz", tissue_static_mask=pm["tissue_static_mask"], adjacent_band=pm["adjacent_band"],
                        background_reference=pm["background_reference"], saturated=pm["saturated"], low_texture=pm["low_texture"],
                        fb_median_error=pm["fb_median_error"], fb_inconsistent=pm["fb_inconsistent"])
    np.savez_compressed(out_dir / "series.npz", background_block_magnitude_fullroi=bw["background_block_magnitude_fullroi"],
                        background_median_magnitude=bw["background_median_magnitude"], background_p95_magnitude=bw["background_p95_magnitude"],
                        drift_xy=bw["drift_xy"], pair_frames=bw["pair_frames"])
    from cardio_organoid.motion.raft_flow import compute_motion_roi
    from global_boundary_motion_diagnostic import _load_config
    mask = np.load(REPO_ROOT / obs["mask_path"])["mask"]
    roi = compute_motion_roi(mask, expansion_px=_load_config()["motion"]["roi"]["expansion_px"], frame_shape=mask.shape)
    meta = {"observation_id": obs["observation_id"], "source": "REUSED_PILOT", "reuse_checks": checks, "video": s["video"],
            "frame_count": s["frame_count"], "fps": s["fps_recorded"], "frame_step": s["frame_step"], "dt_s": s["dt_s"],
            "n_pairs": s["n_pairs"], "functional_roi_xyxy_frame": s["functional_roi_xyxy_frame"],
            "storage_crop_xyxy_in_roi": s["storage_crop_xyxy_in_roi"], "centroid_xy_in_roi": list(roi.centroid_xy_in_roi()),
            "mask_path": obs["mask_path"], "mask_sha256": obs["mask_sha256"], "flow_dir": rel(PILOT_OUT / "flow_chunks"),
            "validity": rel(out_dir / "validity.npz"), "series": rel(out_dir / "series.npz"),
            "n_background_blocks_fullroi": int(bw["background_block_magnitude_fullroi"].shape[1])}
    jdump(out_dir / "primitives.json", meta)
    return meta


def _prim_job(args):
    obs, out_dir = args
    out_dir = Path(out_dir)
    cv2.setNumThreads(3)
    if (out_dir / "primitives.json").exists():
        return obs["observation_id"], "SKIPPED_DONE"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (reuse_pilot_primitives if obs["pixel_first_status"] == "REUSE_CANDIDATE_PILOT" else compute_primitives)(obs, out_dir)
        return obs["observation_id"], "OK"
    except Exception as exc:
        jdump(out_dir / "primitives_failed.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        return obs["observation_id"], "FAILED"


def pick_workers(requested):
    try:
        avail_gb = int(open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0]) / 1e6
    except Exception:
        avail_gb = 8
    w = max(1, min(requested, (os.cpu_count() or 2) // 3, int(avail_gb // 3)))
    return w, avail_gb


def run_pool(fn, jobs, requested, label, out):
    w, avail = pick_workers(requested)
    print(f"[{label}] {len(jobs)} jobs, workers={w} (cpu={os.cpu_count()}, avail_RAM_GB={avail:.1f})", flush=True)
    with mp.get_context("spawn").Pool(w) as pool:
        for i, (o, st) in enumerate(pool.imap_unordered(fn, jobs), 1):
            print(f"[{label} {i}/{len(jobs)}] {o}: {st}", flush=True)
    return w


# ------------------------------------------------------------------ derive
def derive(obs, odir: Path):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from sklearn.cluster import AgglomerativeClustering
    from cardio_organoid.motion.pixel_first import autocorr_period_s, best_lag, exclusion_counts, prominent_peaks
    from cardio_organoid.motion.radial_flow import radial_motion_from_centroid, tangential_motion_from_centroid
    P = json.loads((odir / "primitives.json").read_text())
    V = np.load(REPO_ROOT / P["validity"])
    S = np.load(REPO_ROOT / P["series"])
    dt, n = P["dt_s"], P["n_pairs"]
    cx0, cy0, cx1, cy1 = P["storage_crop_xyxy_in_roi"]
    rx0, ry0 = P["functional_roi_xyxy_frame"][:2]
    ox, oy = rx0 + cx0, ry0 + cy0  # crop offset in frame coords
    ccx, ccy = P["centroid_xy_in_roi"][0] - cx0, P["centroid_xy_in_roi"][1] - cy0
    tissue, adj, sat, lowtex, fb_bad = (V[k] for k in ("tissue_static_mask", "adjacent_band", "saturated", "low_texture", "fb_inconsistent"))
    lowconf = sat | lowtex | fb_bad
    v_in, v_adj = tissue & ~lowconf, adj & ~lowconf
    reasons = {"saturated": sat, "low_texture": lowtex, "fb_inconsistent": fb_bad}
    excl = {"INSIDE_REFERENCE_MASK": exclusion_counts(tissue, reasons), "OUTER_BOUNDARY_SUPPORT": exclusion_counts(adj, reasons)}
    ch, cw = tissue.shape
    B = PILOT.BLOCK
    nby, nbx = ch // B, cw // B
    blk = lambda a: a[:nby * B, :nbx * B].reshape(nby, B, nbx, B)
    cnt_in, cnt_adj = blk(v_in).sum(axis=(1, 3)), blk(v_adj).sum(axis=(1, 3))
    cand = (cnt_in + cnt_adj) / (B * B) >= 0.5
    cy_, cx_ = np.nonzero(cand)
    ncand = len(cy_)
    arr = {k: np.zeros((n, ncand), np.float32) for k in ("in_mag", "adj_mag", "in_rad", "adj_rad", "abs_rad", "abs_tan")}
    wf = {k: np.zeros(n) for k in ("strict_mag", "strict_rad", "support_mag", "support_rad", "combined_mag", "combined_rad")}
    s_mag, s_mag2, s_rad = (np.zeros((ch, cw)) for _ in range(3))
    vv = v_in | v_adj
    flows = sorted((REPO_ROOT / P["flow_dir"]).glob("chunk_*.npz"))
    t = 0
    for f in flows:
        z = np.load(f)
        for qx, qy in zip(z["ux"], z["uy"]):
            ux, uy = qx.astype(np.float32) / PILOT.FLOW_Q, qy.astype(np.float32) / PILOT.FLOW_Q
            fl = np.stack([ux, uy], -1)
            mag = np.hypot(ux, uy)
            rad = np.nan_to_num(radial_motion_from_centroid(fl, (ccx, ccy)))
            tan = np.nan_to_num(tangential_motion_from_centroid(fl, (ccx, ccy)))
            s_mag += mag
            s_mag2 += mag * mag
            s_rad += rad
            for key, a, m in (("in_mag", mag, v_in), ("adj_mag", mag, v_adj), ("in_rad", rad, v_in), ("adj_rad", rad, v_adj),
                              ("abs_rad", np.abs(rad), vv), ("abs_tan", np.abs(tan), vv)):
                arr[key][t] = blk(a * m).sum(axis=(1, 3))[cy_, cx_]
            wf["strict_mag"][t], wf["strict_rad"][t] = mag[v_in].mean() if v_in.any() else np.nan, rad[v_in].mean() if v_in.any() else np.nan
            wf["support_mag"][t], wf["support_rad"][t] = mag[v_adj].mean() if v_adj.any() else np.nan, rad[v_adj].mean() if v_adj.any() else np.nan
            wf["combined_mag"][t], wf["combined_rad"][t] = mag[vv].mean() if vv.any() else np.nan, rad[vv].mean() if vv.any() else np.nan
            t += 1
    assert t == n
    ci, ca = cnt_in[cy_, cx_].astype(np.float32), cnt_adj[cy_, cx_].astype(np.float32)
    blk_mag = (arr["in_mag"] + arr["adj_mag"]) / np.maximum(ci + ca, 1)
    bg = S["background_block_magnitude_fullroi"]
    thr = float(np.percentile(bg.std(axis=0), 99))
    bg_std_med = float(np.median(bg.std(axis=0)))
    active = blk_mag.std(axis=0) > thr
    aidx = np.nonzero(active)[0]
    pos = {(cy_[i], cx_[i]): j for j, i in enumerate(aidx)}
    rows, cols = [], []
    for j, i in enumerate(aidx):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = pos.get((cy_[i] + dy, cx_[i] + dx))
                if q is not None and q != j:
                    rows.append(j)
                    cols.append(q)
    labels_active = np.full(len(aidx), -1)
    n_comp = 0
    if len(aidx):
        conn = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(aidx), len(aidx))).tocsr()
        X = blk_mag[:, aidx].T
        X = (X - X.mean(axis=1, keepdims=True)) / np.maximum(X.std(axis=1, keepdims=True), 1e-12)
        n_comp, comp = connected_components(conn, directed=False)
        nxt = 0
        for c in range(n_comp):
            idx = np.nonzero(comp == c)[0]
            if len(idx) < PILOT.MIN_REGION_BLOCKS:
                labels_active[idx] = nxt
                nxt += 1
                continue
            sub = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5, metric="cosine", linkage="average",
                                          connectivity=conn[idx][:, idx]).fit_predict(X[idx])
            labels_active[idx] = sub + nxt
            nxt += sub.max() + 1
    sizes = np.bincount(labels_active) if len(aidx) else np.array([])
    keep = [l for l in np.argsort(-sizes) if sizes[l] >= PILOT.MIN_REGION_BLOCKS] if len(sizes) else []
    remap = {l: r + 1 for r, l in enumerate(keep)}
    lab_c = np.zeros(ncand, int)
    for j, i in enumerate(aidx):
        lab_c[i] = remap.get(labels_active[j], -1)
    lab_map = np.zeros((nby, nbx), int)
    lab_map[cy_, cx_] = lab_c
    half = n // 2
    bgmed = S["background_median_magnitude"]
    regions = []
    for rid in range(1, len(keep) + 1):
        sel = lab_c == rid
        cin, cad = ci[sel].sum(), ca[sel].sum()
        comb_m = (arr["in_mag"][:, sel].sum(1) + arr["adj_mag"][:, sel].sum(1)) / (cin + cad)
        comb_r = (arr["in_rad"][:, sel].sum(1) + arr["adj_rad"][:, sel].sum(1)) / (cin + cad)
        str_m = arr["in_mag"][:, sel].sum(1) / cin if cin else np.full(n, np.nan)
        sup_m = arr["adj_mag"][:, sel].sum(1) / cad if cad else np.full(n, np.nan)
        str_r = arr["in_rad"][:, sel].sum(1) / cin if cin else np.full(n, np.nan)
        act_q = comb_m > np.percentile(comb_m, 75)
        tot_m = (arr["in_mag"][:, sel].sum(1) + arr["adj_mag"][:, sel].sum(1))
        radf = np.median(arr["abs_rad"][act_q][:, sel].sum(1) / tot_m[act_q])
        tanf = np.median(arr["abs_tan"][act_q][:, sel].sum(1) / tot_m[act_q])
        pk = prominent_peaks(comb_m, dt)
        acm, acr = autocorr_period_s(comb_m, dt), autocorr_period_s(comb_r, dt)
        # within-region timing dispersion (exploratory): sampled block vs region mean excluding a local window, per half
        disp = None
        if sel.sum() >= 100:
            rng = np.random.default_rng(12345)
            sidx = np.nonzero(sel)[0]
            samp = rng.choice(sidx, min(DISPERSION_SAMPLE, len(sidx)), replace=False)
            tot = arr["in_mag"][:, sidx].sum(1) + arr["adj_mag"][:, sidx].sum(1)
            totc = (ci[sidx] + ca[sidx]).sum()
            yy, xx = cy_[sidx], cx_[sidx]
            lags = {0: [], 1: []}
            for i in samp:
                near = (np.abs(yy - cy_[i]) <= DISPERSION_EXCLUDE_BLOCKS) & (np.abs(xx - cx_[i]) <= DISPERSION_EXCLUDE_BLOCKS)
                loc = arr["in_mag"][:, sidx[near]].sum(1) + arr["adj_mag"][:, sidx[near]].sum(1)
                refw = (tot - loc) / max(totc - (ci[sidx[near]] + ca[sidx[near]]).sum(), 1)
                for hi, (a0, a1) in enumerate(((0, half), (half, n))):
                    lags[hi].append(best_lag(refw[a0:a1], blk_mag[a0:a1, i], MAX_LAG_PAIRS)[0])
            l0, l1 = np.array(lags[0]), np.array(lags[1])
            disp = {"n_sampled_blocks": int(len(samp)), "lag_resolution_s": dt,
                    "first_half": {"frac_lag0": float((l0 == 0).mean()), "lag_iqr_pairs": float(np.subtract(*np.percentile(l0, [75, 25])))},
                    "second_half": {"frac_lag0": float((l1 == 0).mean()), "lag_iqr_pairs": float(np.subtract(*np.percentile(l1, [75, 25])))},
                    "frac_same_lag_both_halves": float((l0 == l1).mean()),
                    "note": "reference excludes a +-10-block window around each sampled block; exploratory; 0 lag = within dt"}
        by_, bx_ = cy_[sel], cx_[sel]
        regions.append({
            "region": rid, "n_blocks": int(sel.sum()), "n_valid_px": int(cin + cad), "n_valid_px_inside_mask": int(cin),
            "n_valid_px_outer_support": int(cad), "fraction_px_outer_support": float(cad / (cin + cad)),
            "centroid_xy_frame": [float(bx_.mean() * B + ox + B / 2), float(by_.mean() * B + oy + B / 2)],
            "bbox_xyxy_frame": [int(bx_.min() * B + ox), int(by_.min() * B + oy), int((bx_.max() + 1) * B + ox), int((by_.max() + 1) * B + oy)],
            "coverage_over_time": "constant (validity is static per pixel)",
            "combined_mag_mean_px_per_pair": float(comb_m.mean()), "combined_mag_p95_px_per_pair": float(np.percentile(comb_m, 95)),
            "combined_speed_p95_px_per_s": float(np.percentile(comb_m, 95) / dt),
            "strict_inside_mag_mean_px_per_pair": float(np.nanmean(str_m)), "support_only_mag_mean_px_per_pair": float(np.nanmean(sup_m)),
            "corr_strict_vs_support_mag": float(np.corrcoef(str_m, sup_m)[0, 1]) if cin and cad else None,
            "combined_signed_radial_std_px_per_pair": float(comb_r.std()),
            "snr_vs_background_block_std": float(comb_m.std() / bg_std_med) if bg_std_med > 0 else None,
            "corr_with_background_median_mag": float(np.corrcoef(comb_m, bgmed)[0, 1]),
            "n_mag_peaks": len(pk), "mag_period": acm, "signed_period": acr,
            "mag_peaks_per_positive_period": (len(pk) * acm["positive_peak_period_s"] / (n * dt)) if acm["positive_peak_period_s"] else None,
            "centroid_based_radial_fraction_active": float(radf), "centroid_based_tangential_fraction_active": float(tanf),
            "within_region_timing_dispersion": disp,
            "_w": {"combined_mag": comb_m.tolist(), "combined_rad": comb_r.tolist(), "strict_mag": str_m.tolist(),
                   "support_mag": sup_m.tolist(), "strict_rad": str_r.tolist()}})
    # inter-region timing (only when >= 2 regions)
    inter = []
    if len(regions) >= 2:
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                a, b = np.array(regions[i]["_w"]["combined_mag"]), np.array(regions[j]["_w"]["combined_mag"])
                row = {"regions": [i + 1, j + 1]}
                for name, (s0, s1) in (("first_half", (0, half)), ("second_half", (half, n))):
                    lag, r = best_lag(a[s0:s1], b[s0:s1], MAX_LAG_PAIRS)
                    row[name] = {"zero_lag_r": float(np.corrcoef(a[s0:s1], b[s0:s1])[0, 1]), "best_lag_pairs": lag,
                                 "best_lag_s": lag * dt, "best_lag_r": r}
                inter.append(row)
    mean_mag = s_mag / n
    std_mag = np.sqrt(np.maximum(s_mag2 / n - mean_mag ** 2, 0))
    pm_in = mean_mag[v_in]
    pix_dist = {"inside_mask_valid_px": int(v_in.sum()), "coverage_inside": excl["INSIDE_REFERENCE_MASK"]["measurable_fraction"],
                "median": float(np.median(pm_in)) if pm_in.size else None,
                "iqr": float(np.subtract(*np.percentile(pm_in, [75, 25]))) if pm_in.size else None,
                "p95": float(np.percentile(pm_in, 95)) if pm_in.size else None,
                "note": "distribution of per-pixel time-mean |u| over VALID inside-mask pixels only (conditional on coverage)"}
    act_s = wf["strict_mag"] > np.nanpercentile(wf["strict_mag"], 75) if v_in.any() else np.zeros(n, bool)
    summary = {
        "observation_id": obs["observation_id"], "well": obs["well"], "day": obs["day"], "role": obs["role"],
        "primitives_source": P["source"], "fps": P["fps"], "dt_s": dt, "n_pairs": n, "lag_resolution_s": dt,
        "exclusions": excl, "measurable_fraction_inside_mask": excl["INSIDE_REFERENCE_MASK"]["measurable_fraction"],
        "strict_inside": {"mag_mean_px_per_pair": float(np.nanmean(wf["strict_mag"])), "mag_p95_px_per_pair": float(np.nanpercentile(wf["strict_mag"], 95)),
                          "speed_p95_px_per_s": float(np.nanpercentile(wf["strict_mag"], 95) / dt),
                          "period": autocorr_period_s(np.nan_to_num(wf["strict_mag"]), dt), "n_mag_peaks": len(prominent_peaks(np.nan_to_num(wf["strict_mag"]), dt)),
                          "signed_radial_std_px_per_pair": float(np.nanstd(wf["strict_rad"]))},
        "support_inclusive": {"mag_mean_px_per_pair": float(np.nanmean(wf["combined_mag"])), "mag_p95_px_per_pair": float(np.nanpercentile(wf["combined_mag"], 95)),
                              "speed_p95_px_per_s": float(np.nanpercentile(wf["combined_mag"], 95) / dt)},
        "outer_support_only": {"mag_mean_px_per_pair": float(np.nanmean(wf["support_mag"])) if v_adj.any() else None},
        "corr_strict_vs_support_waveform": float(np.corrcoef(wf["strict_mag"], wf["support_mag"])[0, 1]) if v_in.any() and v_adj.any() else None,
        "support_to_strict_p95_ratio": float(np.nanpercentile(wf["support_mag"], 95) / np.nanpercentile(wf["strict_mag"], 95)) if v_in.any() and v_adj.any() else None,
        "corr_strict_with_background_median_mag": float(np.corrcoef(np.nan_to_num(wf["strict_mag"]), bgmed)[0, 1]) if v_in.any() else None,
        "block_activity_threshold_px_per_pair": thr, "n_candidate_blocks": int(ncand), "n_active_blocks": int(active.sum()),
        "n_spatial_components": int(n_comp), "n_regions": len(regions),
        "n_active_blocks_unassigned_small": int((lab_c == -1).sum()),
        "inter_region_synchrony": inter if inter else "NOT_AVAILABLE (fewer than 2 motion-pattern regions)",
        "pixel_distribution_inside": pix_dist,
        "regions": [{k: v for k, v in r.items() if k != "_w"} for r in regions],
    }
    jdump(odir / "summary.json", summary)
    np.savez_compressed(odir / "derived_timeseries.npz", **{k: v for k, v in wf.items()}, background_median_mag=bgmed,
                        pair_frames=S["pair_frames"], dt_s=dt,
                        **{f"region{r['region']}_{k}": np.array(v) for r in regions for k, v in r["_w"].items()})
    np.savez_compressed(odir / "derived_maps.npz", mean_mag=mean_mag.astype(np.float32), std_mag=std_mag.astype(np.float32),
                        mean_signed_radial=(s_rad / n).astype(np.float32), region_label_blocks=lab_map, block_px=B,
                        crop_offset_frame_xy=np.array([ox, oy]))
    figures(obs, odir, P, tissue, adj, sat, lowtex, fb_bad, V["background_reference"] & ~lowconf, mean_mag, std_mag, s_rad / n, lab_map,
            regions, wf, bgmed, dt, S["pair_frames"], (ox, oy), pm_in)
    return summary


def figures(obs, odir, P, tissue, adj, sat, lowtex, fb_bad, bgref, mean_mag, std_mag, mean_rad, lab_map, regions, wf, bgmed, dt,
            pair_frames, off, pm_in):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ox, oy = off
    ch, cw = tissue.shape
    video = REPO_ROOT / P["video"]
    lowconf = sat | lowtex | fb_bad
    cnt, _ = cv2.findContours(tissue.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    tag = f"P36/Conv/{obs['well']} D{obs['day']}"
    # alignment overlays: start / middle / end / strict signed-radial min (inward) and max (outward)
    sr = np.nan_to_num(np.array(wf["strict_rad"]))
    kf = {"start": 0, "middle": P["frame_count"] // 2, "end": P["frame_count"] - 1,
          "most_inward_pair": int(pair_frames[int(np.argmin(sr))][1]), "most_outward_pair": int(pair_frames[int(np.argmax(sr))][1])}
    cap = cv2.VideoCapture(str(video))
    tiles = []
    for name, f in kf.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        c = fr[oy:oy + ch, ox:ox + cw].copy()
        cv2.drawContours(c, cnt, -1, (0, 0, 0), 4)
        cv2.drawContours(c, cnt, -1, (0, 255, 255), 2)
        c = cv2.resize(c, None, fx=0.4, fy=0.4)
        band = np.full((26, c.shape[1], 3), 255, np.uint8)
        cv2.putText(band, f"{name} frame {f}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        tiles.append(np.vstack([band, c]))
    cv2.imwrite(str(odir / "01_alignment_overlays.png"), np.hstack(tiles))
    # maps
    fig, ax = plt.subplots(2, 3, figsize=(17, 10))
    cat = np.zeros(tissue.shape + (3,))
    cat[bgref] = (0.85, 0.85, 1.0)
    cat[adj & ~lowconf] = (1.0, 0.8, 0.35)
    cat[tissue & ~lowconf] = (0.3, 0.75, 0.3)
    cat[fb_bad] = (0.8, 0.2, 0.2)
    cat[lowtex] = (0.55, 0.55, 0.55)
    cat[sat] = (0.12, 0.12, 0.12)
    ax[0, 0].imshow(cat)
    ax[0, 0].set_title("measurable map: green INSIDE valid, orange OUTER SUPPORT valid, blue BACKGROUND ref;\n"
                       "black saturated, grey low texture, red FB-inconsistent (excluded, not zero)", fontsize=7)
    grey = np.ma.masked_where(~lowconf, np.ones_like(mean_mag))
    for a, img, t in ((ax[0, 1], mean_mag, "mean |u| (px/pair)"), (ax[0, 2], std_mag, "temporal std of |u| (px/pair)")):
        vm = float(np.percentile(img[~lowconf], 99.5)) if (~lowconf).any() else 1.0
        im = a.imshow(np.ma.masked_where(lowconf, img), cmap="inferno", vmin=0, vmax=vm)
        a.imshow(grey, cmap="gray", vmin=0, vmax=2, alpha=0.6)
        a.set_title(t + "; grey = low-confidence", fontsize=8)
        fig.colorbar(im, ax=a, shrink=0.7)
    vr = float(np.percentile(np.abs(mean_rad[~lowconf]), 99.5)) if (~lowconf).any() else 1.0
    im = ax[1, 0].imshow(np.ma.masked_where(lowconf, mean_rad), cmap="coolwarm", vmin=-vr, vmax=vr)
    ax[1, 0].set_title("mean signed radial (centroid-based; - inward)", fontsize=8)
    fig.colorbar(im, ax=ax[1, 0], shrink=0.7)
    lp = np.zeros((ch, cw))
    up = np.repeat(np.repeat(lab_map, PILOT.BLOCK, 0), PILOT.BLOCK, 1)
    lp[:up.shape[0], :up.shape[1]] = up
    ax[1, 1].imshow(np.ma.masked_where(lp <= 0, lp), cmap="tab10", vmin=0.5, vmax=max(1.5, lab_map.max() + 0.5))
    ax[1, 1].imshow(np.ma.masked_where(lp != -1, np.ones_like(lp)), cmap="gray", vmin=0, vmax=2, alpha=0.8)
    ax[1, 1].set_title(f"motion-pattern regions: {len(regions)} (grey = active, groups < {PILOT.MIN_REGION_BLOCKS} blocks)", fontsize=8)
    ax[1, 2].hist(pm_in, bins=80, color="tab:green") if pm_in.size else None
    ax[1, 2].set_title(f"per-pixel mean |u|, VALID inside-mask pixels only (n={pm_in.size}; conditional on coverage)", fontsize=8)
    for a in list(ax.flat)[:5]:
        for c in cnt:
            a.plot(c[:, 0, 0], c[:, 0, 1], color="cyan", lw=0.6)
        a.axis("off")
    fig.suptitle(f"{tag} pixel-first maps (exploratory)", fontsize=10)
    fig.tight_layout()
    fig.savefig(odir / "02_maps.png", dpi=80)
    plt.close(fig)
    # waveforms: strict vs support vs background; regions
    tt = np.arange(len(wf["strict_mag"])) * dt
    fig, axes = plt.subplots(1 + len(regions), 1, figsize=(14, 2.4 * (1 + len(regions))), sharex=True)
    axes = np.atleast_1d(axes)
    axes[0].plot(tt, wf["strict_mag"], color="green", lw=0.8, label="INSIDE mask (strict) mean |u|")
    axes[0].plot(tt, wf["support_mag"], color="orange", lw=0.8, label="OUTER SUPPORT mean |u|")
    axes[0].plot(tt, bgmed, color="grey", lw=0.8, label="BACKGROUND median |u|")
    b = axes[0].twinx()
    b.plot(tt, wf["strict_rad"], color="tab:blue", lw=0.5, alpha=0.7)
    b.set_ylabel("strict signed radial", fontsize=7, color="tab:blue")
    axes[0].legend(fontsize=7, loc="upper right")
    axes[0].set_ylabel("px/pair", fontsize=7)
    axes[0].set_title("strict inside vs outer support vs background (valid pixels only)", fontsize=8)
    for a, r in zip(axes[1:], regions):
        a.plot(tt, r["_w"]["combined_mag"], color="black", lw=0.8, label="region |u|")
        a.plot(tt, bgmed, color="grey", lw=0.6, label="background")
        bb = a.twinx()
        bb.plot(tt, r["_w"]["combined_rad"], color="tab:blue", lw=0.5, alpha=0.7)
        a.legend(fontsize=6, loc="upper right")
        a.set_title(f"region {r['region']}: {r['n_valid_px']} px ({r['fraction_px_outer_support']:.0%} outer support); "
                    f"|u| peaks {r['n_mag_peaks']}; positive-autocorr period {r['mag_period']['positive_peak_period_s']}", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(odir / "03_waveforms.png", dpi=80)
    plt.close(fig)
    # movie (sequential frame read; fixed |u| scale over the whole video)
    vmm = float(np.percentile(mean_mag[~lowconf] + 3 * std_mag[~lowconf], 99.5)) if (~lowconf).any() else 1.0
    want = {int(p[1]): i for i, p in enumerate(pair_frames)}
    chunks = sorted((REPO_ROOT / P["flow_dir"]).glob("chunk_*.npz"))
    cache = {}
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    writer, fi = None, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fi in want:
            pi = want[fi]
            ci_ = pi // PILOT.CHUNK
            if ci_ not in cache:
                cache = {ci_: np.load(chunks[ci_])}
            z = cache[ci_]
            k = pi - int(z["pair_index_start"])
            m = np.hypot(z["ux"][k].astype(np.float32), z["uy"][k].astype(np.float32)) / PILOT.FLOW_Q
            c = fr[oy:oy + ch, ox:ox + cw].copy()
            cv2.drawContours(c, cnt, -1, (255, 255, 0), 2)
            mm = cv2.applyColorMap(np.clip(m / vmm * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            mm[lowconf] = (128, 128, 128)
            cv2.drawContours(mm, cnt, -1, (255, 255, 0), 1)
            tile = np.hstack([cv2.resize(c, None, fx=0.5, fy=0.5), cv2.resize(mm, None, fx=0.5, fy=0.5)])
            band = np.full((26, tile.shape[1], 3), 255, np.uint8)
            cv2.putText(band, f"{tag} frame {fi} t={fi / P['fps']:.2f}s | |u| fixed 0-{vmm:.2f} px/pair | grey=low-confidence",
                        (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            tile = np.vstack([band, tile])
            if tile.shape[0] % 2:
                tile = tile[:-1]
            if tile.shape[1] % 2:
                tile = tile[:, :-1]
            if writer is None:
                writer = cv2.VideoWriter(str(odir / "04_synchronized_original_mask_magnitude.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         max(1, round(1 / dt)), (tile.shape[1], tile.shape[0]))
            writer.write(tile)
        fi += 1
    cap.release()
    if writer:
        writer.release()


def _derive_job(args):
    obs, odir = args
    odir = Path(odir)
    cv2.setNumThreads(2)
    if (odir / "summary.json").exists():
        return obs["observation_id"], "SKIPPED_DONE"
    if not (odir / "primitives.json").exists():
        return obs["observation_id"], "SKIPPED_NO_PRIMITIVES"
    try:
        derive(obs, odir)
        return obs["observation_id"], "OK"
    except Exception as exc:
        jdump(odir / "derive_failed.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        return obs["observation_id"], "FAILED"


def video_jobs(out):
    inv = json.loads((out / "inventory.json").read_text())
    return [(o, str(out / "observations" / o["observation_id"])) for o in inv
            if o["modality"] == "VIDEO" and o["pixel_first_status"] in ("PLANNED", "REUSE_CANDIDATE_PILOT")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["plan", "primitives", "derive", "aggregate"])
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", nargs="*", default=None, help="observation_ids (for checks / retries)")
    a = ap.parse_args()
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if a.stage == "plan":
        stage_plan(out)
        return
    if a.stage == "aggregate":
        from pixel_first_motion_aggregate import aggregate
        aggregate(out)
        return
    jobs = video_jobs(out)
    if a.only:
        jobs = [j for j in jobs if j[0]["observation_id"] in a.only]
    fn = _prim_job if a.stage == "primitives" else _derive_job
    w = run_pool(fn, jobs, a.workers, a.stage, out)
    write_run_manifest(out / "manifests" / a.stage, REPO_ROOT, config_path=CONFIG_PATH, interface_version=INTERFACE_VERSION,
                       selector_artifact=None, sam2_checkpoint=None, sam2_model_cfg=None,
                       input_scope={"allowlist": ALLOW, "stage": a.stage, "n_jobs": len(jobs), "workers": w, "p38": "not accessed"},
                       extra={"frozen_method_sha256": sha256_file(out / "config" / "FROZEN_METHOD.json")}, hash_sam2_checkpoint=False)


if __name__ == "__main__":
    main()
