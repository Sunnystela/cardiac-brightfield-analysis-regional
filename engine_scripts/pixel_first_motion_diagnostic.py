"""ONE-VIDEO pixel-first motion diagnostic (P36/Conv/a1 D20 VIDEO).

Separate diagnostic; the P36/Conv first-pass screening, masks and thresholds
are not changed and nothing else is re-run.

Method
- Farneback flow (existing params) on the UNMASKED functional ROI (existing
  rule: mask bbox + motion.roi.expansion_px), recorded fps, existing
  frame_step, existing whole-ROI median drift subtraction (drift stored).
- Primary signal per pixel and pair: m = sqrt(ux^2 + uy^2) of the
  drift-corrected flow (norm first, never averaged vectors). Signed radial
  (existing projection from the mask centroid) kept as the comparison
  signal. Units: px/pair; px/s = px/pair / dt, dt = frame_step / fps.
- The tissue region is applied AFTER flow: only valid tissue pixels are in
  the denominator; saturated / low-texture / forward-backward-inconsistent
  pixels are flagged low-confidence and excluded, never treated as zero.
- Fixed-coordinate maps: pixels are NOT tracked as tissue points; the fixed
  mask is the reference-frame mask and the boundary moves across it.
- Motion-pattern regions: 4x4-px blocks (per-pixel magnitude averaged over
  valid pixels only -- norm computed per pixel first), active if the block's
  magnitude temporal std exceeds the 99th percentile of same-size background
  blocks; clustered by spatial 8-connectivity + waveform similarity
  (average-linkage, cosine on z-scored magnitude waveforms, threshold 0.5);
  regions < MIN_REGION_BLOCKS blocks are left unassigned. Parameters are
  declared diagnostic choices, not validated.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cardio_organoid.motion.diagnostics import first_secondary_autocorrelation_peak  # noqa: E402
from cardio_organoid.motion.radial_flow import radial_motion_from_centroid, remove_global_drift  # noqa: E402
from cardio_organoid.motion.raft_flow import FARNEBACK_DEFAULT_PARAMS, compute_dense_optical_flow, compute_motion_roi  # noqa: E402
from cardio_organoid.functional.regional_lag_diagnostics import simple_peak_indices  # noqa: E402
from cardio_organoid.pipeline.segmentation_interface import SegmentationRecord, fail_closed_mode, require_downstream_mask  # noqa: E402
from global_boundary_motion_diagnostic import _load_config  # noqa: E402

WELL, DAY = "a1", 20
MANIFEST = REPO_ROOT / "results/_scratch_runs/p36_conv_segmentation_manifest_v2_human_reviewed_20260928T155748Z"
SCREEN_MOTION = REPO_ROOT / ("results/_scratch_runs/p36_conv_first_pass_screening_20260928T155956Z/functional/rerun/"
                             "P36_Conv_a1_D20_VIDEO/motion_pipeline/P36_Conv_a1_D20_VIDEO")
CROP_PAD = 120          # storage/analysis crop around the mask bbox (px)
ADJ_BAND = 20           # mask-adjacent band outside the static mask (px)
BG_MIN_DIST = 200       # background reference pixels: > this distance from the mask (px)
BLOCK = 4               # graph node size (px)
MIN_REGION_BLOCKS = 25  # 400 px
SAT_LO, SAT_HI, SAT_FRAC = 2, 253, 0.5
LOW_TEXTURE_GRAD = 2.0  # intensity units / px (Sobel, Gaussian sigma 3) on the representative frame
FB_MAX = 1.0            # px/pair median forward-backward error
FB_EVERY = 5
CHUNK = 16
FLOW_Q = 100            # stored flow = round(u * FLOW_Q) as int16 -> 0.01 px/pair resolution


def prominent_peaks(x, dt, k=5.0, min_sep_s=0.3):
    """Declared diagnostic peak rule (not tuned): local maxima >= min_sep_s apart
    whose height exceeds median + k * MAD of the same waveform."""
    x = np.asarray(x, float)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    return [i for i in simple_peak_indices(x, max(1, int(round(min_sep_s / dt)))) if x[i] > med + k * mad]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    out = ap.parse_args().output_dir
    out.mkdir(parents=True, exist_ok=True)
    cfg = _load_config()
    step = cfg["motion"]["spatial_patches"]["frame_step"]
    with fail_closed_mode():
        recs = {(r["well"], r["day"]): r for r in json.loads((MANIFEST / "segmentation_interface_records.json").read_text())["records"]}
        rec = SegmentationRecord.from_dict(recs[(WELL, DAY)])
        mask = require_downstream_mask(rec, REPO_ROOT, "pixel_first_motion_diagnostic")
    qc_ref = json.loads((SCREEN_MOTION / "motion_pipeline_qc.json").read_text())
    video = Path(qc_ref["video"])
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
    cx0, cy0 = max(0, x - CROP_PAD), max(0, y - CROP_PAD)
    cx1, cy1 = min(mroi.shape[1], x + w + CROP_PAD), min(mroi.shape[0], y + h + CROP_PAD)
    crop = (slice(cy0, cy1), slice(cx0, cx1))
    ch, cw = cy1 - cy0, cx1 - cx0
    m8 = mroi.astype(np.uint8)
    dil = lambda k: cv2.dilate(m8, np.ones((3, 3), np.uint8), iterations=k) > 0
    tissue = mroi[crop]
    adj = (dil(ADJ_BAND) & ~mroi)[crop]
    bg_full = ~dil(BG_MIN_DIST)

    # ---- observability pre-pass (sampled frames) + texture on the representative frame
    sat_count = np.zeros(mroi.shape, np.int32)
    n_s = 0
    for f in range(0, fc, 10):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)[ry0:ry1, rx0:rx1]
        sat_count += ((g <= SAT_LO) | (g >= SAT_HI))
        n_s += 1
    saturated = sat_count / n_s >= SAT_FRAC
    rg = cv2.GaussianBlur(cv2.cvtColor(rep, cv2.COLOR_BGR2GRAY)[ry0:ry1, rx0:rx1].astype(np.float32), (0, 0), 3)
    grad = np.hypot(cv2.Sobel(rg, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(rg, cv2.CV_32F, 0, 1, ksize=3))
    low_texture = grad < LOW_TEXTURE_GRAD

    # ---- single forward pass (+ backward flow on every FB_EVERY-th pair)
    fdir = out / "flow_chunks"
    fdir.mkdir(exist_ok=True)
    nby, nbx = ch // BLOCK, cw // BLOCK
    buf_ux, buf_uy, drifts, fb_err = [], [], [], []
    mag_sum, mag_sq, rad_sum, rad_sq = (np.zeros((ch, cw)) for _ in range(4))
    bg_mag_median, bg_mag_p95 = [], []
    frames_idx = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    prev, fi, k, chunk_id = None, 0, 0, 0
    # block accumulators are filled after validity is known; keep per-pair block sums of mag/radial for ALL pixels
    # restricted to candidate pixels (tissue | adj) and a same-size background block set
    ones = np.ones((BLOCK, BLOCK))
    blk = lambda a: a[:nby * BLOCK, :nbx * BLOCK].reshape(nby, BLOCK, nbx, BLOCK)
    mag_blocks, rad_blocks, cnt_blocks = [], [], None
    fh, fw = mroi.shape
    fbg = (bg_full & ~saturated & ~low_texture)[: fh // BLOCK * BLOCK, : fw // BLOCK * BLOCK].reshape(fh // BLOCK, BLOCK, fw // BLOCK, BLOCK)
    bg_full_blocks = fbg.all(axis=(1, 3))
    bg_full_series = []
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
                if k % FB_EVERY == 0:
                    back = compute_dense_optical_flow(g, prev, local_roi, method="opencv_farneback")
                    hh, ww = flow.shape[:2]
                    gx, gy = np.meshgrid(np.arange(ww, dtype=np.float32), np.arange(hh, dtype=np.float32))
                    bx = cv2.remap(back[..., 0], gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR)
                    by = cv2.remap(back[..., 1], gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR)
                    fb_err.append(np.hypot(flow[..., 0] + bx, flow[..., 1] + by)[crop].astype(np.float32))
                ux, uy = corr[..., 0][crop], corr[..., 1][crop]
                mag = np.hypot(ux, uy)
                rad = radial_motion_from_centroid(corr, centroid)[crop]
                buf_ux.append(np.clip(np.round(ux * FLOW_Q), -32767, 32767).astype(np.int16))
                buf_uy.append(np.clip(np.round(uy * FLOW_Q), -32767, 32767).astype(np.int16))
                mag_sum += mag
                mag_sq += mag * mag
                r0 = np.nan_to_num(rad)
                rad_sum += r0
                rad_sq += r0 * r0
                bmag = np.hypot(corr[..., 0], corr[..., 1])[bg_full & ~saturated]
                bg_mag_median.append(float(np.median(bmag)))
                bg_mag_p95.append(float(np.percentile(bmag, 95)))
                mag_blocks.append(blk(mag).astype(np.float32))
                fm = np.hypot(corr[..., 0], corr[..., 1])[: fh // BLOCK * BLOCK, : fw // BLOCK * BLOCK]
                bg_full_series.append(fm.reshape(fh // BLOCK, BLOCK, fw // BLOCK, BLOCK).mean(axis=(1, 3))[bg_full_blocks].astype(np.float32))
                rad_blocks.append(blk(r0).astype(np.float32))
                frames_idx.append((fi - step, fi))
                k += 1
                if len(buf_ux) == CHUNK:
                    np.savez_compressed(fdir / f"chunk_{chunk_id:03d}.npz", ux=np.stack(buf_ux), uy=np.stack(buf_uy),
                                        pair_index_start=k - CHUNK)
                    buf_ux, buf_uy, chunk_id = [], [], chunk_id + 1
            prev = g
        fi += 1
    cap.release()
    if buf_ux:
        np.savez_compressed(fdir / f"chunk_{chunk_id:03d}.npz", ux=np.stack(buf_ux), uy=np.stack(buf_uy), pair_index_start=k - len(buf_ux))
    n = k

    # ---- validity
    fb_med = np.median(np.stack(fb_err), axis=0)
    sat_c, lowtex_c = saturated[crop], low_texture[crop]
    fb_bad = fb_med > FB_MAX
    lowconf = sat_c | lowtex_c | fb_bad
    valid_tissue = tissue & ~lowconf
    valid_adj = adj & ~lowconf
    bg_crop = bg_full[crop] & ~lowconf
    mean_mag = mag_sum / n
    std_mag = np.sqrt(np.maximum(mag_sq / n - mean_mag ** 2, 0))
    mean_rad = rad_sum / n
    std_rad = np.sqrt(np.maximum(rad_sq / n - mean_rad ** 2, 0))

    # ---- whole-tissue waveforms (valid tissue pixels only) vs static background reference
    MB = np.stack(mag_blocks)  # (n, nby, BLOCK, nbx, BLOCK)
    RB = np.stack(rad_blocks)
    vt_b = blk(valid_tissue.astype(np.float32))
    va_b = blk((valid_tissue | valid_adj).astype(np.float32))
    bg_b = blk(bg_crop.astype(np.float32))

    def region_wave(A, wmask_b):
        num = (A * wmask_b[None]).sum(axis=(2, 4))
        den = wmask_b.sum(axis=(1, 3))
        return num, den

    num_m, den_t = region_wave(MB, vt_b)
    tissue_mag_wf = num_m.sum(axis=(1, 2)) / max(den_t.sum(), 1)
    num_r, _ = region_wave(RB, vt_b)
    tissue_rad_wf = num_r.sum(axis=(1, 2)) / max(den_t.sum(), 1)

    # ---- blocks: candidate nodes = >=50% valid (tissue|adjacent) pixels; same-size background blocks for noise
    cand_frac = va_b.mean(axis=(1, 3))
    bgfrac = bg_b.mean(axis=(1, 3))
    num_all, den_all = region_wave(MB, va_b)
    blk_mag = np.where(den_all[None] > 0, num_all / np.maximum(den_all[None], 1), np.nan)  # (n, nby, nbx)
    num_rall, _ = region_wave(RB, va_b)
    blk_rad = np.where(den_all[None] > 0, num_rall / np.maximum(den_all[None], 1), np.nan)
    bg_series = np.stack(bg_full_series)  # (n, n_bg_blocks): fully valid background 4x4 blocks over the whole functional ROI
    bg_std = bg_series.std(axis=0)
    act_thr = float(np.nanpercentile(bg_std, 99))
    node_std = np.nanstd(blk_mag, axis=0)
    cand = cand_frac >= 0.5
    active = cand & (node_std > act_thr)

    # ---- clustering: spatial 8-connectivity + waveform similarity
    from scipy.sparse import coo_matrix
    from sklearn.cluster import AgglomerativeClustering
    ids = -np.ones((nby, nbx), int)
    ys, xs = np.nonzero(active)
    ids[ys, xs] = np.arange(len(ys))
    rows, cols = [], []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0:
                continue
            yy, xx = ys + dy, xs + dx
            ok = (yy >= 0) & (yy < nby) & (xx >= 0) & (xx < nbx)
            nb = np.full(len(ys), -1)
            nb[ok] = ids[yy[ok], xx[ok]]
            sel = nb >= 0
            rows += list(np.arange(len(ys))[sel])
            cols += list(nb[sel])
    conn = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ys), len(ys))).tocsr()
    X = blk_mag[:, ys, xs].T
    X = (X - X.mean(axis=1, keepdims=True)) / np.maximum(X.std(axis=1, keepdims=True), 1e-12)
    from scipy.sparse.csgraph import connected_components
    n_comp, comp = connected_components(conn, directed=False)
    labels = np.full(len(ys), -1)
    nxt = 0
    for c in range(n_comp):
        idx = np.nonzero(comp == c)[0]
        if len(idx) < MIN_REGION_BLOCKS:
            labels[idx] = nxt
            nxt += 1
            continue
        sub = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5, metric="cosine", linkage="average",
                                      connectivity=conn[idx][:, idx]).fit_predict(X[idx])
        labels[idx] = sub + nxt
        nxt += sub.max() + 1
    sizes = np.bincount(labels)
    keep = [l for l in np.argsort(-sizes) if sizes[l] >= MIN_REGION_BLOCKS]
    remap = {l: i + 1 for i, l in enumerate(keep)}
    lab_map = np.zeros((nby, nbx), int)
    for (yy, xx, l) in zip(ys, xs, labels):
        lab_map[yy, xx] = remap.get(l, -1)  # -1 = active but in a small (unassigned) group

    # ---- per-region outputs
    bg_block_std_median = float(np.nanmedian(bg_std))
    bg_block_wave = np.median(bg_series, axis=1)
    sc = np.load(SCREEN_MOTION / "patch_waveforms_radial.npz")
    screen_wf = {tuple(int(v) for v in kk.split("_")): sc[kk] for kk in sc.files}
    L = int(round(cfg["motion"]["spatial_patches"]["cell_size_ratio"] * qc_ref["equivalent_diameter_px"]))
    regions = []
    for rid in range(1, len(keep) + 1):
        sel = lab_map == rid
        wmask = va_b * sel[:, None, :, None]
        num, den = region_wave(MB, wmask)
        mw = num.sum(axis=(1, 2)) / den.sum()
        numr, _ = region_wave(RB, wmask)
        rw = numr.sum(axis=(1, 2)) / den.sum()
        by_, bx_ = np.nonzero(sel)
        pix = np.zeros((ch, cw), bool)
        pix[:nby * BLOCK, :nbx * BLOCK] = np.repeat(np.repeat(sel, BLOCK, 0), BLOCK, 1)
        pix &= (valid_tissue | valid_adj)
        n_t, n_a = int((pix & valid_tissue).sum()), int((pix & valid_adj).sum())
        mpk = prominent_peaks(mw, dt)
        neg = prominent_peaks(-rw, dt)
        pos = prominent_peaks(rw, dt)
        ac_m = first_secondary_autocorrelation_peak(mw, dt)
        ac_r = first_secondary_autocorrelation_peak(rw, dt)
        # existing L-patches whose box lies >=50% inside this region (same location comparison)
        ov = []
        for (pr, pc), wf in screen_wf.items():
            X0, Y0 = pc * L - cx0, pr * L - cy0
            xs0, ys0 = max(0, X0), max(0, Y0)
            xe, ye = min(cw, X0 + L), min(ch, Y0 + L)
            if xe <= xs0 or ye <= ys0:
                continue
            if pix[ys0:ye, xs0:xe].sum() >= 0.5 * L * L:
                ov.append(wf)
        old = np.mean(ov, axis=0) if ov else None
        regions.append({"region": rid, "n_blocks": int(sel.sum()), "n_valid_pixels": int(pix.sum()),
                        "n_pixels_in_static_mask": n_t, "n_pixels_in_adjacent_band": n_a,
                        "centroid_xy_frame": [float(bx_.mean() * BLOCK + cx0 + rx0 + BLOCK / 2), float(by_.mean() * BLOCK + cy0 + ry0 + BLOCK / 2)],
                        "bbox_xyxy_frame": [int(bx_.min() * BLOCK + cx0 + rx0), int(by_.min() * BLOCK + cy0 + ry0),
                                            int((bx_.max() + 1) * BLOCK + cx0 + rx0), int((by_.max() + 1) * BLOCK + cy0 + ry0)],
                        "magnitude_mean_px_per_pair": float(mw.mean()), "magnitude_std_px_per_pair": float(mw.std()),
                        "magnitude_std_px_per_s": float(mw.std() / dt),
                        "snr_vs_same_size_background_block_std": float(mw.std() / bg_block_std_median) if bg_block_std_median > 0 else None,
                        "signed_radial_std_px_per_pair": float(rw.std()),
                        "n_magnitude_peaks": len(mpk), "n_signed_inward_peaks": len(neg), "n_signed_outward_peaks": len(pos),
                        "magnitude_autocorr_period_s": ac_m.get("lag_s") if isinstance(ac_m, dict) else None,
                        "signed_autocorr_period_s": ac_r.get("lag_s") if isinstance(ac_r, dict) else None,
                        "n_existing_L_patches_overlapping": len(ov),
                        "corr_existing_signed_vs_new_signed": float(np.corrcoef(old, rw)[0, 1]) if old is not None else None,
                        "corr_existing_signed_vs_new_magnitude": float(np.corrcoef(old, mw)[0, 1]) if old is not None else None,
                        "_mw": mw.tolist(), "_rw": rw.tolist(), "_old": old.tolist() if old is not None else None})

    # ---- save compact reusable data
    np.savez_compressed(out / "pixel_maps.npz", mean_magnitude=mean_mag.astype(np.float32), std_magnitude=std_mag.astype(np.float32),
                        mean_signed_radial=mean_rad.astype(np.float32), std_signed_radial=std_rad.astype(np.float32),
                        saturated=sat_c, low_texture=lowtex_c, fb_median_error=fb_med.astype(np.float32), fb_inconsistent=fb_bad,
                        tissue_static_mask=tissue, adjacent_band=adj, background_reference=bg_crop,
                        valid_tissue=valid_tissue, valid_adjacent=valid_adj,
                        region_label_blocks=lab_map, block_active=active, block_size=BLOCK)
    np.savez_compressed(out / "block_waveforms.npz", magnitude=blk_mag.astype(np.float32), signed_radial=blk_rad.astype(np.float32),
                        background_block_magnitude_fullroi=bg_series, pair_frames=np.array(frames_idx),
                        drift_xy=np.array(drifts), tissue_magnitude=tissue_mag_wf, tissue_signed_radial=tissue_rad_wf,
                        background_median_magnitude=np.array(bg_mag_median), background_p95_magnitude=np.array(bg_mag_p95))
    summary = {
        "observation": "P36/Conv/a1 D20 VIDEO", "video": str(video.relative_to(REPO_ROOT)), "mask": rec.mask_path,
        "frame_count": fc, "fps_recorded": fps, "frame_step": step, "dt_s": dt, "n_pairs": n,
        "functional_roi_xyxy_frame": [rx0, ry0, rx1, ry1], "storage_crop_xyxy_in_roi": [cx0, cy0, cx1, cy1],
        "crop_offset_in_frame_xy": [cx0 + rx0, cy0 + ry0], "farneback_params": FARNEBACK_DEFAULT_PARAMS,
        "drift": "whole-ROI per-pair median (existing remove_global_drift); drift_xy stored per pair",
        "flow_storage": "flow_chunks/chunk_*.npz: drift-corrected ux, uy as int16 = round(u*100) (0.01 px/pair), storage crop, 16 pairs per chunk; u = stored/100",
        "units": "px/pair unless stated; px/s = px/pair / dt",
        "low_confidence_rules": {"saturated": f"intensity <= {SAT_LO} or >= {SAT_HI} in >= {SAT_FRAC:.0%} of every-10th frames",
                                 "low_texture": f"Sobel gradient (Gaussian sigma 3) < {LOW_TEXTURE_GRAD} on the representative frame",
                                 "fb_inconsistent": f"median forward-backward error > {FB_MAX} px/pair over every {FB_EVERY}th pair"},
        "pixel_counts_in_crop": {"static_mask": int(tissue.sum()), "valid_tissue": int(valid_tissue.sum()),
                                 "mask_saturated": int((tissue & sat_c).sum()), "mask_low_texture": int((tissue & lowtex_c).sum()),
                                 "mask_fb_inconsistent": int((tissue & fb_bad).sum()),
                                 "adjacent_band": int(adj.sum()), "valid_adjacent": int(valid_adj.sum()), "background_reference": int(bg_crop.sum())},
        "block_activity_threshold_px_per_pair": act_thr, "n_candidate_blocks": int(cand.sum()), "n_active_blocks": int(active.sum()),
        "n_regions": len(keep), "n_spatial_components_of_active_blocks": int(n_comp),
        "n_background_blocks_fullroi": int(bg_series.shape[1]), "n_active_blocks_unassigned_small": int((lab_map == -1).sum()),
        "existing_method_video_result": {"functional_qc": "ESTIMABLE", "bpm": 54.687272727272685, "L_px": L},
        "regions": [{k: v for k, v in r.items() if not k.startswith("_")} for r in regions],
    }
    with open(out / "diagnostic_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    figures(out, video, (rx0, ry0), (cx0, cy0, cx1, cy1), tissue, adj, sat_c, lowtex_c, fb_bad, bg_crop, mean_mag, std_mag, mean_rad,
            lab_map, regions, tissue_mag_wf, tissue_rad_wf, np.array(bg_mag_median), bg_block_wave, dt, frames_idx, fdir, step)
    print(json.dumps({k: summary[k] for k in ("n_pairs", "dt_s", "pixel_counts_in_crop", "block_activity_threshold_px_per_pair",
                                              "n_active_blocks", "n_regions")}, indent=1))


def figures(out, video, roi_off, crop_box, tissue, adj, sat, lowtex, fb_bad, bg, mean_mag, std_mag, mean_rad, lab_map, regions,
            tmag, trad, bgmed, bgblk, dt, frames_idx, fdir, step):
    cx0, cy0, cx1, cy1 = crop_box
    ox, oy = roi_off
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
    _, rep = cap.read()
    repc = cv2.cvtColor(rep[oy + cy0:oy + cy1, ox + cx0:ox + cx1], cv2.COLOR_BGR2RGB)
    cnt, _ = cv2.findContours(tissue.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    def contour(a, col="cyan"):
        for c in cnt:
            a.plot(c[:, 0, 0], c[:, 0, 1], color=col, lw=0.7)

    lowconf = sat | lowtex | fb_bad
    grey = np.ma.masked_where(~lowconf, np.ones_like(mean_mag))
    vmax_m = float(np.percentile(mean_mag[~lowconf], 99.5))
    vmax_s = float(np.percentile(std_mag[~lowconf], 99.5))
    fig, ax = plt.subplots(2, 3, figsize=(18, 11))
    ax[0, 0].imshow(repc); contour(ax[0, 0])
    ax[0, 0].set_title("B. representative frame + static mask (cyan)", fontsize=9)
    cat = np.zeros(tissue.shape + (3,))
    cat[bg] = (0.85, 0.85, 1.0)
    cat[adj & ~lowconf] = (1.0, 0.85, 0.4)
    cat[tissue & ~lowconf] = (0.3, 0.75, 0.3)
    cat[sat] = (0.15, 0.15, 0.15)
    cat[lowtex & ~sat] = (0.55, 0.55, 0.55)
    cat[fb_bad & ~sat & ~lowtex] = (0.8, 0.2, 0.2)
    ax[0, 1].imshow(cat); contour(ax[0, 1], "white")
    ax[0, 1].set_title("C. confidence: green valid tissue, orange valid +20 px band, blue background ref,\n"
                       "black saturated, grey low texture, red FB-inconsistent (all low-confidence = excluded)", fontsize=8)
    for a, img, vm, t in ((ax[0, 2], mean_mag, vmax_m, "D. mean |u| (px/pair), fixed scale; grey = low-confidence"),
                          (ax[1, 0], std_mag, vmax_s, "D. temporal std of |u| (px/pair); grey = low-confidence")):
        im = a.imshow(np.ma.masked_where(lowconf, img), cmap="inferno", vmin=0, vmax=vm)
        a.imshow(grey, cmap="gray", vmin=0, vmax=2, alpha=0.6)
        contour(a); a.set_title(t, fontsize=8); fig.colorbar(im, ax=a, shrink=0.7)
    vr = float(np.percentile(np.abs(mean_rad[~lowconf]), 99.5))
    im = ax[1, 1].imshow(np.ma.masked_where(lowconf, mean_rad), cmap="coolwarm", vmin=-vr, vmax=vr)
    ax[1, 1].imshow(grey, cmap="gray", vmin=0, vmax=2, alpha=0.6)
    contour(ax[1, 1]); ax[1, 1].set_title("mean signed radial (px/pair; - inward, + outward)", fontsize=8); fig.colorbar(im, ax=ax[1, 1], shrink=0.7)
    lab_px = np.repeat(np.repeat(lab_map, 4, 0), 4, 1)
    lp = np.zeros(tissue.shape, float)
    lp[:lab_px.shape[0], :lab_px.shape[1]] = lab_px
    cm = plt.get_cmap("tab20", max(2, lab_map.max() + 1))
    ax[1, 2].imshow(repc, alpha=0.45)
    ax[1, 2].imshow(np.ma.masked_where(lp <= 0, lp), cmap=cm, vmin=0.5, vmax=max(1.5, lab_map.max() + 0.5), alpha=0.85)
    ax[1, 2].imshow(np.ma.masked_where(lp != -1, np.ones_like(lp)), cmap="gray", vmin=0, vmax=2, alpha=0.8)
    for r in regions:
        cxr = (r["centroid_xy_frame"][0] - ox - cx0); cyr = (r["centroid_xy_frame"][1] - oy - cy0)
        ax[1, 2].text(cxr, cyr, str(r["region"]), color="black", fontsize=9, ha="center", weight="bold")
    contour(ax[1, 2]); ax[1, 2].set_title("motion-pattern regions (spatial 8-conn. + magnitude waveform similarity);\n"
                                            "grey = active but in groups < 25 blocks (unassigned)", fontsize=8)
    for a in ax.flat:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out / "02_maps_confidence_regions.png", dpi=90)
    plt.close(fig)

    # region waveforms
    t = np.arange(len(tmag)) * dt
    nr = len(regions)
    fig, axes = plt.subplots(nr + 1, 1, figsize=(14, 2.2 * (nr + 1)), sharex=True)
    axes = np.atleast_1d(axes)
    axes[0].plot(t, tmag, color="black", lw=0.8, label="valid-tissue mean |u|")
    axes[0].plot(t, bgmed, color="grey", lw=0.8, label="background median |u|")
    a2 = axes[0].twinx(); a2.plot(t, trad, color="tab:blue", lw=0.6, alpha=0.7); a2.set_ylabel("signed radial", color="tab:blue", fontsize=7)
    axes[0].legend(fontsize=7, loc="upper right"); axes[0].set_ylabel("px/pair", fontsize=7)
    axes[0].set_title("whole valid tissue (static mask, valid pixels only) vs background reference", fontsize=8)
    for a, r in zip(axes[1:], regions):
        a.plot(t, r["_mw"], color="black", lw=0.8, label="region |u| (px/pair)")
        a.plot(t, bgblk, color="grey", lw=0.6, label="background 4x4-block median |u|")
        b = a.twinx()
        b.plot(t, r["_rw"], color="tab:blue", lw=0.6, alpha=0.8, label="region signed radial")
        if r["_old"] is not None:
            b.plot(t, r["_old"], color="tab:orange", lw=0.6, alpha=0.8, label=f"existing L-patch signed radial (n={r['n_existing_L_patches_overlapping']})")
        b.legend(fontsize=6, loc="upper right"); a.legend(fontsize=6, loc="upper left")
        a.set_title(f"region {r['region']}: {r['n_valid_pixels']} valid px ({r['n_pixels_in_static_mask']} in mask, "
                    f"{r['n_pixels_in_adjacent_band']} in +20 px band); magnitude peaks {r['n_magnitude_peaks']} vs signed inward "
                    f"{r['n_signed_inward_peaks']} / outward {r['n_signed_outward_peaks']}", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out / "03_region_waveforms.png", dpi=90)
    plt.close(fig)

    # A. synchronized movie (original | |u| | signed radial), fixed colour scales over the whole video
    lowc = lowconf
    vmr = float(np.percentile(np.abs(mean_rad[~lowc]), 99.5)) * 4
    vmm = vmax_s * 3
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fr_list = {}
    for f in sorted({b for _, b in frames_idx}):
        pass
    writer = None
    sc = 0.5
    chunk_files = sorted(fdir.glob("chunk_*.npz"))
    pair = 0
    from cardio_organoid.motion.radial_flow import radial_motion_from_centroid  # noqa
    for cf in chunk_files:
        z = np.load(cf)
        for ux, uy in zip(z["ux"], z["uy"]):
            f1 = frames_idx[pair][1]
            cap.set(cv2.CAP_PROP_POS_FRAMES, f1)
            _, fr = cap.read()
            fr = fr[oy + cy0:oy + cy1, ox + cx0:ox + cx1].copy()
            cv2.drawContours(fr, cnt, -1, (255, 255, 0), 2)
            m = np.hypot(ux.astype(np.float32) / FLOW_Q, uy.astype(np.float32) / FLOW_Q)
            mm = cv2.applyColorMap(np.clip(m / vmm * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            mm[lowc] = (128, 128, 128)
            cv2.drawContours(mm, cnt, -1, (255, 255, 0), 1)
            panels = [fr, mm]
            tile = np.hstack([cv2.resize(p, None, fx=sc, fy=sc) for p in panels])
            band = np.full((28, tile.shape[1], 3), 255, np.uint8)
            cv2.putText(band, f"frame {f1}  t={f1 / (step / dt):.2f}s | original+static mask | |u| fixed 0-{vmm:.2f} px/pair, grey=low-confidence",
                        (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            tile = np.vstack([band, tile])
            if writer is None:
                writer = cv2.VideoWriter(str(out / "01_synchronized_original_mask_magnitude.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         round(1 / dt), (tile.shape[1], tile.shape[0]))
            writer.write(tile)
            pair += 1
    writer.release()
    cap.release()


if __name__ == "__main__":
    main()
