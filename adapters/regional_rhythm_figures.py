"""Figures that show HOW patches were grouped for the regional BPMs (adapters/regional_rhythm.py).

  regional_rhythm_map.png ........ where each group is on the organoid + per-patch own BPM
  regional_rhythm_traces.png ..... group mean waveforms and every patch's beat events, sorted by group
  regional_rhythm_clustering.png . distance matrix ordered by group, stage-1 merge heights/cut, group-pair table

The patch grid is rebuilt with the engine's own functions, exactly as run_motion_pipeline builds it
(analyze.py: compute_motion_roi -> crop_mask_to_roi -> dilate_mask -> generate_patch_grid). If no
frame/mask is given, or the rebuilt grid does not match the cached patch ids, the map falls back to
an abstract row/column grid and says so in its title.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ._engine import engine

GREY = (0.62, 0.62, 0.62)


def group_colors(labels: list[str]) -> dict:
    import matplotlib
    cmap = matplotlib.colormaps["tab10"]
    out, k = {}, 0
    for lab in labels:
        if lab.startswith("G"):
            out[lab] = cmap(k % 10)[:3]
            k += 1
        else:
            out[lab] = GREY
    return out


def rebuild_patch_grid(mask: np.ndarray, frame_shape, config: dict, expected_ids: set) -> dict:
    """Same grid the motion pipeline used. Returns {"cells": {pid: bbox_xyxy_roi}, "roi_bbox": ..., "ok": bool, "reason"}."""
    feats = engine("cardio_organoid.morphology.features")
    rf = engine("cardio_organoid.motion.raft_flow")
    sg = engine("cardio_organoid.motion.spatial_gate")
    sp = engine("cardio_organoid.motion.spatial_patches")
    mcfg = config["motion"]
    mask = np.asarray(mask).astype(bool)
    eqd = feats.equivalent_diameter(feats.area(mask))
    roi = rf.compute_motion_roi(mask, expansion_px=mcfg["roi"]["expansion_px"], frame_shape=tuple(frame_shape[:2]))
    region = sg.dilate_mask(sg.crop_mask_to_roi(mask, roi.roi_bbox_xyxy), mcfg["signal_extraction"]["gate_dilation_px"])
    cell_px = max(1, int(round(mcfg["spatial_patches"]["cell_size_ratio"] * eqd)))
    cells = sp.generate_patch_grid(region, cell_px, mcfg["spatial_patches"]["min_coverage_fraction"])
    got = {(c.row, c.col): c.bbox_xyxy for c in cells}
    ok = set(got) == set(expected_ids)
    return {"cells": got, "roi_bbox": roi.roi_bbox_xyxy, "cell_px": cell_px, "ok": ok,
            "reason": None if ok else f"rebuilt grid has {len(got)} cells, cache has {len(expected_ids)} (ids differ)"}


def _abstract_cells(patch_ids, cell=20):
    return {p: (p[1] * cell, p[0] * cell, (p[1] + 1) * cell, (p[0] + 1) * cell) for p in patch_ids}


def render_map(rr, summary: dict, out_path: Path, frame_bgr=None, mask=None) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    pp = summary["per_patch"]
    ids = rr.patch_ids
    labels = [g["label"] for g in summary["groups_final"]]
    colors = group_colors(labels)
    grid_note, bg = "", None
    if frame_bgr is not None and mask is not None:
        grid = rebuild_patch_grid(mask, frame_bgr.shape, rr.config, set(ids))
        if grid["ok"]:
            x0, y0, x1, y1 = grid["roi_bbox"]
            bg = frame_bgr[y0:y1, x0:x1, ::-1]
            cells = grid["cells"]
        else:
            grid_note = f" | abstract grid: {grid['reason']}"
    else:
        grid_note = " | abstract row/col grid (no frame/mask given)"
    if bg is None:
        cells = _abstract_cells(ids)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.6), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    own = {p: pp[f"{p[0]}_{p[1]}"]["own_bpm"] for p in ids}
    valid_bpm = [v for v in own.values() if v]
    vmin, vmax = (min(valid_bpm), max(valid_bpm)) if valid_bpm else (0, 1)
    cmap = matplotlib.colormaps["viridis"]
    for ax, mode in ((axes[0], "group"), (axes[1], "own_bpm")):
        if bg is not None:
            ax.imshow(bg)
        else:
            xs = [c[2] for c in cells.values()]
            ys = [c[3] for c in cells.values()]
            ax.set_xlim(0, max(xs) + 5)
            ax.set_ylim(max(ys) + 5, 0)
            ax.set_facecolor("white")
        for p in ids:
            bx0, by0, bx1, by1 = cells[p]
            info = pp[f"{p[0]}_{p[1]}"]
            w, h = bx1 - bx0, by1 - by0
            if info["membership"] == "not_reliable":
                ax.add_patch(Rectangle((bx0, by0), w, h, fill=False, ec="white" if bg is not None else "0.7", lw=0.6, ls=":"))
                continue
            if mode == "group":
                col = colors.get(info["group_final"], GREY)
                unassigned = not str(info["group_final"]).startswith("G")
                ax.add_patch(Rectangle((bx0, by0), w, h, fc=(*col, 0.55), ec=col, lw=1.2,
                                       hatch="///" if unassigned else None))
                if info["membership"] == "attached":
                    ax.plot(bx0 + w * 0.5, by0 + h * 0.5, marker="o", ms=3.5, mfc="white", mec="black", mew=0.6)
            else:
                v = own[p]
                col = cmap((v - vmin) / (vmax - vmin)) if v and vmax > vmin else (0.5, 0.5, 0.5, 1)
                ax.add_patch(Rectangle((bx0, by0), w, h, fc=(*col[:3], 0.7), ec="black", lw=0.4))
                if v:
                    ax.text(bx0 + w / 2, by0 + h / 2, f"{v:.0f}", ha="center", va="center", fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
    # group labels at each group's patch centroid
    for g in summary["groups_final"]:
        if not g["assigned"]:
            continue
        pts = [cells[tuple(int(x) for x in k.split("_"))] for k in g["patch_ids"]]
        cx = np.mean([(b[0] + b[2]) / 2 for b in pts])
        cy = np.mean([(b[1] + b[3]) / 2 for b in pts])
        tb = g.get("beat_train_bpm")
        tag = {"reference": "REF", "locked": "locked", "not_locked": "INDEPENDENT"}.get(g.get("relation"), "?")
        axes[0].text(cx, cy, f"{g['label']} {tb:.1f}\n{tag}" if tb else g["label"], ha="center", va="center",
                     fontsize=9, weight="bold", color="black",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colors[g["label"]], lw=2, alpha=0.85))
    axes[0].set_title("timing groups (unweighted period + beat-timing distance); label = group beat-train BPM\n"
                      "REF = most regular group, locked = same rhythm (fixed delay, may drop beats)\n"
                      "hatched = unassigned, o = attached in stage 2, dotted = not reliable", fontsize=8)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(vmin, vmax))
    fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.02, label="patch own BPM")
    axes[1].set_title("per-patch own BPM (median of IBI+/IBI-/spectral/autocorr candidates)", fontsize=9)

    ax = axes[2]
    ax.axis("off")
    rel_short = {"reference": "REF", "locked": "locked", "not_locked": "indep.", "insufficient_beats": "few beats"}
    rows = []
    for g in summary["groups_final"]:
        pl = g.get("phase_lock") or {}
        rows.append([g["label"], g["n_patches"], _f(g.get("beat_train_bpm")), _f(g.get("beat_train_ibi_cv"), 2),
                     _f(g["bpm"]), rel_short.get(g.get("relation"), "-"), _f(pl.get("R"), 2), _f(pl.get("delay_s"), 2),
                     _pct(pl.get("follow_fraction")), "+".join(g["merged_from"])])
    tab = ax.table(cellText=rows, colLabels=["group", "n", "train\nBPM", "train\nCV", "official-\nrule BPM", "vs REF",
                                             "lock R", "delay\n(s)", "follows\nREF", "from"],
                   loc="upper center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1, 1.35)
    for (r, c), cell in tab.get_celld().items():
        if r > 0 and c == 0:
            cell.set_facecolor((*colors.get(rows[r - 1][0], GREY), 0.55))
    ax.set_title(f"{summary['status']}\ntrain = beats on group mean waveform\n(official bpm column unchanged)", fontsize=9)
    fig.legend(handles=[Patch(fc=colors[l], label=l) for l in labels if l.startswith("G")],
               loc="lower left", ncol=10, fontsize=8, frameon=False)
    fig.suptitle(f"{summary['observation_id']} - regional rhythm groups "
                 f"({summary['n_reliable']}/{summary['n_patches']} reliable patches){grid_note}", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return {"path": str(out_path), "grid": "frame" if bg is not None else "abstract", "note": grid_note.strip(" |")}


def render_traces(rr, summary: dict, out_path: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = summary["groups_final"]
    colors = group_colors([g["label"] for g in groups])
    shown = [g for g in groups if g["assigned"]] or groups[:1]
    n = len(shown)
    fig, axes = plt.subplots(n + 1, 1, figsize=(15, 1.7 * n + 4.5), sharex=True,
                             gridspec_kw={"height_ratios": [1] * n + [max(2.5, 0.09 * len(rr.reliable))]})
    axes = np.atleast_1d(axes)
    t = np.arange(len(next(iter(rr.waveforms.values())))) * rr.dt
    rla = engine("cardio_organoid.functional.regional_lag_diagnostics")
    ref = summary.get("reference")
    for ax, g in zip(axes[:n], shown):
        pids = [tuple(int(x) for x in k.split("_")) for k in g["patch_ids"]]
        w = rla.region_mean_waveform(rr.waveforms, pids)
        if ref is not None:
            for tb in ref["beat_train"]:
                ax.axvline(tb, color="0.75", lw=0.6, ls="--", zorder=0)
        ax.plot(t, w, color=colors[g["label"]], lw=1.0)
        bt = np.asarray(g.get("beat_train_s") or [])
        if bt.size:
            ax.plot(bt, np.interp(bt, t, w), "v", color="black", ms=4)
        pl = g.get("phase_lock") or {}
        info = ("REF" if g.get("relation") == "reference" else
                f"{g.get('relation')}\nR={_f(pl.get('R'), 2)} d={_f(pl.get('delay_s'), 2)}s\nfollows {_pct(pl.get('follow_fraction'))}")
        ax.set_ylabel(f"{g['label']} {_f(g.get('beat_train_bpm'))}\n{info}", rotation=0, ha="right", va="center", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_title("group mean radial waveform (px/pair); v = group beats; grey dashed = reference group beats", fontsize=9)
    ax = axes[-1]
    y, yt, yl = 0, [], []
    for g in groups:
        for k in g["patch_ids"]:
            p = tuple(int(x) for x in k.split("_"))
            ev = rr.pooled.get(p, [])
            ax.vlines(ev, y - 0.4, y + 0.4, color=colors[g["label"]], lw=1.0)
            yt.append(y)
            yl.append(f"{g['label']} {k}")
            y += 1
        y += 0.6
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=5)
    ax.invert_yaxis()
    ax.set_xlabel("time (s)")
    ax.set_title("every reliable patch's detected beat events (pos+neg), sorted by group - aligned columns = beating together", fontsize=9)
    fig.suptitle(f"{summary['observation_id']} - regional rhythm traces", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return str(out_path)


def render_clustering(rr, summary: dict, out_path: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = summary["groups_final"]
    colors = group_colors([g["label"] for g in groups])
    fig = plt.figure(figsize=(17, 7.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1, 1.25], height_ratios=[1, 1])
    ax = fig.add_subplot(gs[:, 0])
    if rr.raw_distance is not None:
        idx = {p: i for i, p in enumerate(rr.reliable)}
        order = [idx[tuple(int(x) for x in k.split("_"))] for g in groups for k in g["patch_ids"]]
        im = ax.imshow(rr.raw_distance[np.ix_(order, order)], cmap="magma_r", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="unweighted robust timing distance")
        pos = 0
        for g in groups:
            n = len(g["patch_ids"])
            ax.add_patch(plt.Rectangle((pos - 0.5, pos - 0.5), n, n, fill=False, ec=colors[g["label"]], lw=2))
            if n >= 2:
                ax.text(pos + n / 2 - 0.5, pos + n / 2 - 0.5, g["label"], ha="center", va="center", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=colors[g["label"]], alpha=0.85))
            pos += n
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("patch x patch distance, ordered by group\n(1 - mean of event match, narrow-lag corr, period similarity)", fontsize=9)
    else:
        ax.text(0.5, 0.5, "fewer than 2 reliable patches", ha="center")
        ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    h = summary["cut_info"].get("merge_heights") or []
    if h:
        ax.plot(range(1, len(h) + 1), h, ".-", color="0.2")
        k = summary["cut_info"].get("n_merges_kept")
        ax.axvline(k + 0.5, color="crimson", ls="--", lw=1)
        ax.text(k + 0.7, min(h), f"cut after merge {k}", color="crimson", fontsize=8, va="bottom")
        ax.axvspan(0.5, len(h) // 2 + 0.5, color="0.9", zorder=0)
        ax.text(1, max(h), "not searched (first half)", fontsize=7, va="top", color="0.4")
    ax.set_xlabel("merge #", fontsize=8)
    ax.set_ylabel("average-linkage height", fontsize=8)
    ax.set_title("stage 1: unweighted average-linkage merges, cut at largest relative jump", fontsize=9)
    ax.tick_params(labelsize=7)

    ax = fig.add_subplot(gs[1, 1])
    att = [(k, v) for k, v in summary["per_patch"].items() if v["membership"] == "attached"]
    left = [v for v in summary["per_patch"].values() if v["membership"] == "core" and not str(v["group_final"]).startswith("G")]
    if att:
        xs = range(len(att))
        ax.bar(xs, [a["attach_detail"]["mean_distance"] for _, a in att],
               color=[colors.get(a["group_final"], GREY) for _, a in att], label="mean distance to joined group")
        ax.scatter(xs, [a["attach_detail"]["threshold"] for _, a in att], marker="_", s=200, color="crimson",
                   label="that group's own max (threshold)")
        ax.set_xticks(list(xs))
        ax.set_xticklabels([f"{k}\n->{a['group_final']}" for k, a in att], fontsize=6)
        ax.legend(fontsize=7, loc="lower right")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "no leftover patch was within an existing group's own distance range\n"
                "(they stay as unassigned U* groups; their relation to the reference is in the map table)",
                ha="center", va="center", fontsize=8, transform=ax.transAxes)
    ax.set_title(f"stage 2: {len(att)} leftover patches attached by timing distance; {len(left)} stay unassigned", fontsize=9)
    ax.set_ylabel("mean distance to group", fontsize=8)
    ax.tick_params(labelsize=7)

    ax = fig.add_subplot(gs[:, 2])
    ax.axis("off")
    assigned = {g["label"] for g in groups if g["assigned"]}
    pairs = [p for p in summary["pairs_final"] if p["a"] in assigned and p["b"] in assigned]
    pairs = sorted(pairs, key=lambda p: (not p["merge_suggested"], -(p["phase_lock_R"] or -9)))[:18]
    rows = [[f"{p['a']}-{p['b']}", f"{_f(p['beat_train_bpm_a'])}/{_f(p['beat_train_bpm_b'])}", p["phase_base"],
             _f(p["phase_lock_R"], 2), _f(p["phase_lock_R_crit"], 2), _f(p["delay_s"], 2), _pct(p["follow_fraction"]),
             _f(p["lag_aligned_shape_correlation"], 2), "YES" if p["merge_suggested"] else ""] for p in pairs]
    if rows:
        tab = ax.table(cellText=rows, colLabels=["pair", "train BPM", "phase\nbase", "lock R", "R crit\n(p<.01)",
                                                 "delay\n(s)", "follows\nbase", "shape\ncorr", "suggest\nmerge"],
                       loc="upper center", cellLoc="center")
        tab.auto_set_font_size(False)
        tab.set_fontsize(7.5)
        tab.scale(1, 1.3)
        for r, p in enumerate(pairs, start=1):
            if p["merge_suggested"]:
                for c in range(9):
                    tab[r, c].set_facecolor("#fff2b3")
    ax.set_title("group pairs: phase locking of beat trains (suggest merge = locked: same rhythm,\n"
                 "fixed delay, dropped beats allowed). Suggestion only - merging is a manual decision", fontsize=9)
    fig.suptitle(f"{summary['observation_id']} - how the regional groups were formed", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return str(out_path)


def render_all(rr, summary: dict, out_dir: Path, frame_bgr=None, mask=None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {"map": render_map(rr, summary, out_dir / "regional_rhythm_map.png", frame_bgr, mask),
            "traces": render_traces(rr, summary, out_dir / "regional_rhythm_traces.png"),
            "clustering": render_clustering(rr, summary, out_dir / "regional_rhythm_clustering.png")}


def _f(v, nd=1):
    return "-" if v is None else f"{v:.{nd}f}"


def _pct(v):
    return "-" if v is None else f"{100 * v:.0f}%"
