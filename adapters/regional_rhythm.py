"""Regional (multi-rhythm) BPM from an existing legacy-rhythm motion cache.

The official `bpm` assumes ONE rhythm per organoid (confidence-weighted median over every reliable
patch). This module groups reliable patches that beat together -- similar period AND aligned beat
timing -- and reports one BPM per group, so an organoid with several independently beating areas
shows several BPMs. Nothing here edits the hash-locked engine; it only re-combines engine
functions, in the same order `post_motion_functional` uses them:

  per-patch events/own period ...... patch_phase_consensus_diagnostic.per_patch_events_and_features,
                                     fundamental_period_consensus.patch_own_period_estimate
  pairwise period+timing graph ..... harmonic_aware_timing.build_robust_patch_timing_graph
  grouping ......................... adaptive_timing_clustering.cluster_patches_by_robust_timing_similarity_hierarchical
                                     (cross-check: cluster_patches_by_density on the same distances)
  per-group BPM (official rule) .... period_evidence_confidence.population_candidate_bpm (+ weighted_median)
  per-group IBI / CV ............... cycle_aware_consensus.cycle_aware_consensus_for_group
  group-pair comparison ............ regional_lag_diagnostics.region_mean_waveform / region_pair_lag_diagnostic

Groups are only ever merged by a person (`RegionalRhythm.groups(merges=...)`); pair comparisons
only *suggest* a merge. No video decoding and no optical flow: input is the legacy_rhythm folder.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ._engine import engine

MIN_GROUP_PATCHES = 3            # smaller clusters are reported as "unassigned" (still mergeable by hand)
MAX_LAG_S = 1.0                  # same default as patch_phase_consensus_diagnostic / region_pair_lag_diagnostic
COVERAGE_WINDOW_S = 5.0          # diagnostic-only edge field (temporal_coverage_overlap); not part of the distance
SUGGEST_MIN_PERIOD_SIMILARITY = 0.90   # merge suggestion: harmonic-aware period similarity of the two groups
SUGGEST_MIN_SHAPE_CORRELATION = 0.80   # merge suggestion: lag-aligned correlation of the two group mean waveforms
ATTACH_LEFTOVERS = True          # stage 2 of grouping, see RegionalRhythm._attach_leftovers


def _pid(key: str) -> tuple[int, int]:
    r, c = key.split("_")
    return int(r), int(c)


def _key(pid) -> str:
    return f"{pid[0]}_{pid[1]}"


def find_cache(legacy_dir: Path) -> dict:
    """Locate motion cache + round-36 json inside `<obs>/legacy_rhythm` (as written by analyze_legacy_rhythm)."""
    legacy_dir = Path(legacy_dir).resolve()
    if (legacy_dir / "legacy_rhythm").is_dir():
        legacy_dir = legacy_dir / "legacy_rhythm"
    motion_dirs = [p for p in (legacy_dir / "motion_pipeline").glob("*") if (p / "patch_waveforms_radial.npz").is_file()]
    r36 = sorted((legacy_dir / "round36").glob("*_round36_raw_per_patch.json"))
    if len(motion_dirs) != 1 or len(r36) != 1:
        raise FileNotFoundError(f"{legacy_dir}: expected one motion_pipeline/<obs>/patch_waveforms_radial.npz and one "
                                f"round36/*_round36_raw_per_patch.json, found {len(motion_dirs)} / {len(r36)}")
    return {"legacy_dir": legacy_dir, "motion_dir": motion_dirs[0], "round36_json": r36[0],
            "observation_id": motion_dirs[0].name}


def rank_weights(confidence_by_patch: dict, patch_ids: list) -> dict:
    """Confidence -> (0, 1] by rank among `patch_ids` (ties share their mean rank).

    The official confidence is a z-score mean and is clipped at 0 by every engine consumer, so a
    reliable patch below the population mean gets weight 0 and ends up as a singleton cluster.
    Ranks keep the ordering but let every reliable patch take part in the grouping."""
    ids = list(patch_ids)
    if not ids:
        return {}
    vals = np.array([confidence_by_patch.get(p, 0.0) for p in ids], float)
    order = vals.argsort(kind="stable")
    ranks = np.empty(len(ids))
    ranks[order] = np.arange(1, len(ids) + 1)
    for v in np.unique(vals):  # average ties
        sel = vals == v
        ranks[sel] = ranks[sel].mean()
    return {p: float(r / len(ids)) for p, r in zip(ids, ranks)}


@dataclass
class Group:
    label: str                      # "G1", "G2", ... (size order) or "U1", ... for unassigned clusters
    patch_ids: list
    assigned: bool
    merged_from: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    mean_waveform: np.ndarray | None = None


class RegionalRhythm:
    """Holds one observation's per-patch evidence so groups can be recomputed after manual merges."""

    def __init__(self, legacy_dir: Path, config: dict | None = None):
        self.cache = find_cache(legacy_dir)
        self.config = config or engine("global_boundary_motion_diagnostic")._load_config()
        self._load()
        self._cluster()

    # ---------------------------------------------------------------- evidence
    def _load(self) -> None:
        diag = engine("patch_phase_consensus_diagnostic")
        fpc = engine("cardio_organoid.functional.fundamental_period_consensus")
        rhc = engine("cardio_organoid.motion.reliable_hotspot_candidates")
        r36 = json.loads(Path(self.cache["round36_json"]).read_text(encoding="utf-8"))
        per36 = r36["per_patch"]
        self.states = {_pid(k): v["state"] for k, v in per36.items()}
        self.confidence = {_pid(k): v["final_provisional_confidence"] for k, v in per36.items()}
        mdir = Path(self.cache["motion_dir"])
        self.qc = json.loads((mdir / "motion_pipeline_qc.json").read_text(encoding="utf-8"))
        npz = np.load(mdir / "patch_waveforms_radial.npz")
        self.waveforms = {_pid(k): npz[k] for k in npz.files}
        self.fps, self.frame_step = self.qc["fps"], self.qc["frame_step"]
        self.dt = self.frame_step / self.fps
        n_samples = len(next(iter(self.waveforms.values()))) if self.waveforms else 0
        self.duration_s = float(self.qc.get("video_duration_s") or n_samples * self.dt)
        self.patch_ids = sorted(self.waveforms)
        self.per_patch = {p: diag.per_patch_events_and_features(p, self.waveforms[p], self.dt, self.frame_step,
                                                                self.fps, self.config) for p in self.patch_ids}
        self.reliable = sorted(p for p in self.patch_ids if self.states.get(p) != rhc.STATE_C_NOT_OBSERVABLE)
        pp = self.per_patch
        self.pos_events = {p: [e.event_time for e in pp[p]["positive_events"]] for p in self.reliable}
        self.neg_events = {p: [e.event_time for e in pp[p]["negative_events"]] for p in self.reliable}
        self.pooled = {p: pp[p]["pooled_event_times_s"] for p in self.reliable}
        self.freq = {p: pp[p]["dominant_frequency_hz"] for p in self.reliable}
        self.candidate_table = {
            p: fpc.patch_own_period_estimate(self.pos_events[p], self.neg_events[p], self.freq[p],
                                             self.waveforms[p], self.dt) for p in self.reliable}
        self.own_period = {p: self.candidate_table[p]["own_period_estimate_s"] for p in self.reliable}
        self.rank_w = rank_weights(self.confidence, self.reliable)

    # ---------------------------------------------------------------- grouping
    def _cluster(self) -> None:
        be = engine("cardio_organoid.functional.beat_episodes")
        hat = engine("cardio_organoid.functional.harmonic_aware_timing")
        atc = engine("cardio_organoid.functional.adaptive_timing_clustering")
        cfg_be = self.config["functional"]["beat_episodes"]
        pooled_period = be.pooled_dominant_period_s(list(self.freq.values()))
        self.coincidence_tolerance_s = (be.suggest_episode_tolerance_s(pooled_period, cfg_be["episode_tolerance_period_fraction"])
                                        or cfg_be["fallback_episode_tolerance_s"])
        ids = self.reliable
        self.distance_matrix = self.raw_distance = None
        self.cut_info, self.density, self.attached = {}, {}, {}
        if len(ids) <= 1:
            self.core_clusters = self.auto_clusters = [list(ids)] if ids else []
            return
        self.graph = hat.build_robust_patch_timing_graph(
            ids, self.waveforms, self.pooled, self.own_period, self.dt,
            self.coincidence_tolerance_s, MAX_LAG_S, self.duration_s, COVERAGE_WINDOW_S)
        clusters, info = atc.cluster_patches_by_robust_timing_similarity_hierarchical(ids, self.graph, self.rank_w)
        self.core_clusters = [sorted(c) for c in clusters]
        self.distance_matrix = info.pop("distance_matrix")
        self.cut_info = info
        n = len(ids)
        self.raw_distance = np.array([[0.0 if i == j else hat.robust_pairwise_timing_distance(hat.get_edge(self.graph, ids[i], ids[j]))
                                       for j in range(n)] for i in range(n)])
        self.auto_clusters, self.attached = self._attach_leftovers(self.core_clusters) if ATTACH_LEFTOVERS else (self.core_clusters, {})
        dens, dinfo = atc.cluster_patches_by_density(sorted(ids), self.raw_distance)
        self.density = {"distance": "unweighted robust timing distance", "clusters": [[_key(p) for p in c] for c in dens],
                        **dinfo, "rand_index_vs_final_auto": _rand_index(self.auto_clusters, dens, ids)}

    def _attach_leftovers(self, core: list) -> tuple[list, dict]:
        """Stage 2: patches left in clusters smaller than MIN_GROUP_PATCHES join the core group they
        are closest to by the UNWEIGHTED robust distance, but only if that distance is within the
        group's own spread (the largest of its members' mean distances to the rest of the group --
        i.e. the patch would not be the group's most distant member).

        Why: the engine's weighted distance is 1 - similarity * sqrt(w_a * w_b), so a low-confidence
        patch is pushed toward distance 1 from everything even when its timing matches a group.
        Stage 1 keeps the grouping anchored on confident patches; stage 2 decides membership of the
        rest on timing evidence alone. Attached patches are reported separately."""
        idx = {p: i for i, p in enumerate(self.reliable)}
        D = self.raw_distance
        groups = [list(c) for c in core if len(c) >= MIN_GROUP_PATCHES]
        small = [c for c in core if len(c) < MIN_GROUP_PATCHES]
        thresholds = []
        for g in groups:
            gi = [idx[p] for p in g]
            member_mean = [D[i, [j for j in gi if j != i]].mean() for i in gi]
            thresholds.append(float(np.max(member_mean)))
        attached, leftover = {}, []
        for c in small:
            keep = []
            for p in c:
                d = [D[idx[p], [idx[q] for q in g]].mean() for g in groups]
                k = int(np.argmin(d)) if d else None
                if k is not None and d[k] <= thresholds[k]:
                    attached[p] = {"core_index": k, "mean_distance": float(d[k]), "threshold": thresholds[k]}
                else:
                    keep.append(p)
            if keep:
                leftover.append(keep)
        for p, a in attached.items():  # applied after all decisions: no cascading
            groups[a["core_index"]].append(p)
        return [sorted(g) for g in groups] + [sorted(c) for c in leftover], attached

    # ---------------------------------------------------------------- per group
    def group_metrics(self, ids: list) -> dict:
        fpc = engine("cardio_organoid.functional.fundamental_period_consensus")
        pec = engine("cardio_organoid.functional.period_evidence_confidence")
        cac = engine("cardio_organoid.functional.cycle_aware_consensus")
        cpa = engine("cardio_organoid.functional.cycle_phase_assignment")
        sp = engine("cardio_organoid.motion.spatial_patches")
        ids = sorted(ids)
        ct = self.candidate_table

        def bpm_with(weights):
            pos = pec.population_candidate_bpm(ids, ct, weights, "median_ibi_positive_s")
            neg = pec.population_candidate_bpm(ids, ct, weights, "median_ibi_negative_s")
            # official rule (evidence_bundle): weighted_median of the two equal-weight values = the LOWER one
            return pos, neg, fpc.weighted_median([(v["bpm"], 1.0) for v in (pos, neg) if v["bpm"]])

        weights, fallback = self.confidence, False
        pos, neg, bpm = bpm_with(weights)
        if bpm is None and any(ct[p]["candidates"].get(k) for p in ids for k in ("median_ibi_positive_s", "median_ibi_negative_s")):
            weights, fallback = self.rank_w, True     # every official weight in this group is <= 0
            pos, neg, bpm = bpm_with(weights)
        harmonic = fpc.resolve_harmonic_scales(ids, self.own_period, weights)
        resolved = harmonic["resolved_period_by_patch"]
        P = fpc.weighted_median([(resolved[p], weights.get(p, 1.0)) for p in resolved])
        cyc = None
        if P:
            ev = {p: self.pooled[p] for p in ids}
            cyc = cac.cycle_aware_consensus_for_group(ids, ev, P, cpa.choose_anchor_time_s(ev),
                                                      self.config["functional"]["beat_episodes"]["global_min_participation_fraction"])
        auto = pec.population_candidate_bpm(ids, ct, weights, "autocorrelation_period_s")
        spec = pec.population_candidate_bpm(ids, ct, weights, "spectral_period_s")
        agree = pec.estimator_pairwise_agreement({"median_ibi": bpm, "autocorrelation": auto["bpm"], "spectral": spec["bpm"]})
        own_bpms = [60.0 / self.own_period[p] for p in ids if self.own_period.get(p)]
        rows, cols = [p[0] for p in ids], [p[1] for p in ids]
        m = (cyc or {}).get("metrics") or {}
        return {
            "n_patches": len(ids),
            "fraction_of_reliable": len(ids) / len(self.reliable) if self.reliable else None,
            "bpm": bpm, "bpm_positive": pos["bpm"], "bpm_negative": neg["bpm"], "weight_fallback_rank": fallback,
            "fundamental_bpm": (60.0 / P) if P else None,
            "median_own_bpm": float(np.median(own_bpms)) if own_bpms else None,
            "median_ibi_s": m.get("median_ibi_s"), "ibi_cv": m.get("cv_ibi"),
            "consensus_beat_count": m.get("consensus_beat_count"),
            "mean_participation_fraction": m.get("mean_participation_fraction"),
            "autocorrelation_bpm": auto["bpm"], "spectral_bpm": spec["bpm"],
            "estimator_agreement": agree["mean_agreement"],
            "mean_confidence": float(np.mean([self.confidence.get(p, 0.0) for p in ids])),
            "n_spatial_components": len(sp.cluster_active_patches_into_regions(set(ids))),
            "bbox_rowcol": {"row_min": min(rows), "row_max": max(rows), "col_min": min(cols), "col_max": max(cols)},
        }

    def groups(self, merges: list[list[str]] | None = None) -> list[Group]:
        """Auto clusters (optionally merged by hand), labelled by size.

        `merges` lists label sets from the AUTO labelling, e.g. [["G1", "G3"], ["G2", "U1"]]."""
        rla = engine("cardio_organoid.functional.regional_lag_diagnostics")
        auto = self.auto_groups_raw()
        by_label = {g.label: g for g in auto}
        used, merged = set(), []
        for labels in merges or []:
            labels = [str(l).strip().upper() for l in labels if str(l).strip()]
            unknown = [l for l in labels if l not in by_label]
            if unknown:
                raise ValueError(f"unknown group label(s) {unknown}; available: {sorted(by_label)}")
            dup = [l for l in labels if l in used]
            if dup:
                raise ValueError(f"group(s) {dup} appear in more than one merge")
            if len(labels) < 2:
                continue
            used.update(labels)
            merged.append(Group("", sorted(p for l in labels for p in by_label[l].patch_ids), True, merged_from=labels))
        final = merged + [Group("", g.patch_ids, g.assigned, merged_from=[g.label]) for g in auto if g.label not in used]
        for g in final:
            g.assigned = len(g.patch_ids) >= MIN_GROUP_PATCHES
        return self._label_and_measure(final, rla)

    def auto_groups_raw(self) -> list[Group]:
        """Auto clusters with their AUTO labels (no metrics) -- the label space merges refer to."""
        gs = [Group("", c, len(c) >= MIN_GROUP_PATCHES) for c in self.auto_clusters]
        return _assign_labels(gs)

    def _label_and_measure(self, gs: list[Group], rla) -> list[Group]:
        gs = _assign_labels(gs)
        for g in gs:
            g.metrics = self.group_metrics(g.patch_ids)
            g.mean_waveform = rla.region_mean_waveform(self.waveforms, g.patch_ids)
        return gs

    def pair_comparisons(self, gs: list[Group]) -> list[dict]:
        """Every pair of groups (assigned or not): rate relation, timing offset, lag-aligned shape match."""
        hat = engine("cardio_organoid.functional.harmonic_aware_timing")
        rla = engine("cardio_organoid.functional.regional_lag_diagnostics")
        out = []
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                a, b = gs[i], gs[j]
                pa = 60.0 / a.metrics["fundamental_bpm"] if a.metrics.get("fundamental_bpm") else None
                pb = 60.0 / b.metrics["fundamental_bpm"] if b.metrics.get("fundamental_bpm") else None
                ps = hat.harmonic_aware_period_similarity(pa, pb)
                # search one half-cycle either way: any phase offset can be aligned, a full-cycle alias cannot
                max_lag = 0.5 * max(pa or 0.0, pb or 0.0) or MAX_LAG_S
                lag = (rla.region_pair_lag_diagnostic(a.mean_waveform, b.mean_waveform, self.dt, max_lag_s=max_lag)
                       if a.mean_waveform is not None and b.mean_waveform is not None else None)
                cc = (lag or {}).get("cross_correlation") or {}
                shape = cc.get("best_correlation")
                suggest = bool(ps["similarity"] is not None and ps["similarity"] >= SUGGEST_MIN_PERIOD_SIMILARITY
                               and ps["best_ratio"] == 1.0 and shape is not None and shape >= SUGGEST_MIN_SHAPE_CORRELATION)
                out.append({"a": a.label, "b": b.label, "bpm_a": a.metrics.get("bpm"), "bpm_b": b.metrics.get("bpm"),
                            "period_similarity": ps["similarity"], "harmonic_ratio": ps["best_ratio"],
                            "zero_lag_correlation": (lag or {}).get("zero_lag_correlation"),
                            "best_lag_s": cc.get("best_lag_s"), "lag_aligned_shape_correlation": shape,
                            "merge_suggested": suggest})
        return out

    # ---------------------------------------------------------------- output
    def summary(self, merges: list[list[str]] | None = None) -> dict:
        auto = self.groups()
        final = self.groups(merges) if merges else auto
        assigned = [g for g in final if g.assigned]
        if not self.reliable:
            status = "NO_RELIABLE_PATCHES"
        elif not assigned:
            status = "NO_GROUP_WITH_MIN_PATCHES"
        else:
            status = "MULTIPLE_RHYTHMS" if len(assigned) > 1 else "SINGLE_RHYTHM"
        return {
            "observation_id": self.cache["observation_id"], "status": status,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "inputs": {k: str(v) for k, v in self.cache.items()},
            "parameters": {"min_group_patches": MIN_GROUP_PATCHES, "max_lag_s": MAX_LAG_S,
                           "coverage_window_s": COVERAGE_WINDOW_S, "coincidence_tolerance_s": self.coincidence_tolerance_s,
                           "grouping_weights": "stage 1: rank of final_provisional_confidence among reliable patches",
                           "attach_leftovers": ATTACH_LEFTOVERS,
                           "attach_rule": "stage 2: unweighted mean distance to group <= max of members' own mean distance",
                           "bpm_weights": "final_provisional_confidence (official); rank fallback if all <= 0",
                           "suggest_min_period_similarity": SUGGEST_MIN_PERIOD_SIMILARITY,
                           "suggest_min_shape_correlation": SUGGEST_MIN_SHAPE_CORRELATION},
            "n_patches": len(self.patch_ids), "n_reliable": len(self.reliable),
            "merges": merges or [],
            "groups_auto": [_group_json(g) for g in auto],
            "groups_final": [_group_json(g) for g in final],
            "pairs_final": self.pair_comparisons(final),
            "cut_info": _jsonable(self.cut_info), "density_cross_check": _jsonable(self.density),
            "per_patch": {_key(p): {"state": self.states.get(p), "confidence": self.confidence.get(p),
                                    "rank_weight": self.rank_w.get(p),
                                    "own_period_s": self.own_period.get(p),
                                    "own_bpm": (60.0 / self.own_period[p]) if self.own_period.get(p) else None,
                                    "candidates": self.candidate_table[p]["candidates"] if p in self.candidate_table else None,
                                    "membership": ("not_reliable" if p not in self.rank_w else
                                                   "attached" if p in self.attached else "core"),
                                    "attach_detail": self.attached.get(p),
                                    "group_final": next((g.label for g in final if p in g.patch_ids), None)}
                          for p in self.patch_ids},
            "core_clusters": [[_key(p) for p in c] for c in getattr(self, "core_clusters", [])],
            "distance_matrix": {"patch_ids": [_key(p) for p in self.reliable],
                                "weighted_used_for_stage1": self.distance_matrix.round(4).tolist() if self.distance_matrix is not None else None,
                                "unweighted_used_for_stage2": self.raw_distance.round(4).tolist() if self.raw_distance is not None else None},
        }


def _assign_labels(gs: list[Group]) -> list[Group]:
    gs = sorted(gs, key=lambda g: (-len(g.patch_ids), g.patch_ids))
    na = nu = 0
    for g in gs:
        if g.assigned:
            na += 1
            g.label = f"G{na}"
        else:
            nu += 1
            g.label = f"U{nu}"
    return gs


def _group_json(g: Group) -> dict:
    return {"label": g.label, "assigned": g.assigned, "merged_from": g.merged_from,
            "patch_ids": [_key(p) for p in g.patch_ids], **_jsonable(g.metrics)}


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k) if not isinstance(k, tuple) else _key(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def _rand_index(clusters_a, clusters_b, ids) -> float | None:
    la = {p: i for i, c in enumerate(clusters_a) for p in c}
    lb = {p: i for i, c in enumerate(clusters_b) for p in c}
    ids = list(ids)
    n = len(ids)
    if n < 2:
        return None
    agree = sum((la[ids[i]] == la[ids[j]]) == (lb[ids[i]] == lb[ids[j]]) for i in range(n) for j in range(i + 1, n))
    return agree / (n * (n - 1) / 2)


def parse_merges(text: str | None) -> list[list[str]]:
    """'G1+G3, G2+U1' -> [['G1','G3'], ['G2','U1']]."""
    if not text:
        return []
    return [[p.strip().upper() for p in part.split("+") if p.strip()] for part in text.split(",") if part.strip()]


def load_merge_file(path: Path) -> list[list[str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [list(m) for m in data.get("merges", [])]


def write_merge_file(path: Path, merges: list[list[str]], note: str = "") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"merges": merges, "note": note,
                                "saved_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                                "label_space": "auto group labels of regional_rhythm.json:groups_auto"},
                               indent=2, ensure_ascii=False), encoding="utf-8")
    return path


GROUP_CSV_FIELDS = ["observation_id", "label", "assigned", "merged_from", "n_patches", "fraction_of_reliable", "bpm",
                    "bpm_positive", "bpm_negative", "fundamental_bpm", "median_own_bpm", "median_ibi_s", "ibi_cv",
                    "consensus_beat_count", "estimator_agreement", "autocorrelation_bpm", "spectral_bpm",
                    "mean_confidence", "weight_fallback_rank", "n_spatial_components", "patch_ids"]


def write_outputs(summary: dict, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "regional_rhythm.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    with open(out_dir / "regional_groups.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=GROUP_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for g in summary["groups_final"]:
            w.writerow({**g, "observation_id": summary["observation_id"], "merged_from": "+".join(g["merged_from"]),
                        "patch_ids": " ".join(g["patch_ids"])})
    return {"json": str(out_dir / "regional_rhythm.json"), "csv": str(out_dir / "regional_groups.csv")}


def row_columns(summary: dict, out_dir: Path | None = None) -> dict:
    """Columns added to an analysis row (the official `bpm` is never touched)."""
    assigned = [g for g in summary["groups_final"] if g["assigned"]]
    return {"regional_status": summary["status"], "n_rhythm_groups": len(assigned),
            "regional_bpms": [g["bpm"] for g in assigned],
            "regional_group_n_patches": [g["n_patches"] for g in assigned],
            "regional_group_fractions": [g["fraction_of_reliable"] for g in assigned],
            "regional_n_unassigned": sum(g["n_patches"] for g in summary["groups_final"] if not g["assigned"]),
            "regional_merge_applied": bool(summary["merges"]),
            "regional_dir": str(out_dir) if out_dir else None}


def overlay_inputs(video_path: Path | None = None, rec=None, mask_root: Path | None = None) -> tuple:
    """(frame_bgr, mask) for the map overlay: the video's midpoint frame (the frame every engine mask and
    patch grid is defined on) and the record's validated mask. Missing pieces -> (None, None)."""
    if video_path is None or rec is None:
        return None, None
    from .media import check_mask_matches_frame, reference_frame
    from .records import require_mask
    frame, info = reference_frame(Path(video_path), "VIDEO")
    mask = require_mask(rec, Path(mask_root) if mask_root else Path("."), "regional_rhythm")
    check_mask_matches_frame(mask, info)
    return frame, mask


def analyze_regional_rhythm(legacy_dir: Path, out_dir: Path | None = None, merges: list[list[str]] | None = None,
                            frame_bgr: np.ndarray | None = None, mask: np.ndarray | None = None,
                            render_figures: bool = True) -> dict:
    """End to end: cache -> groups -> files (+ figures). Returns the row columns plus output paths."""
    rr = RegionalRhythm(legacy_dir)
    out_dir = Path(out_dir) if out_dir else rr.cache["legacy_dir"].parent / "regional_rhythm"
    summary = rr.summary(merges)
    paths = write_outputs(summary, out_dir)
    if merges:
        paths["merge_file"] = str(write_merge_file(out_dir / "regional_merge.json", merges))
    if render_figures:
        from .regional_rhythm_figures import render_all
        paths["figures"] = render_all(rr, summary, out_dir, frame_bgr=frame_bgr, mask=mask)
    return {**row_columns(summary, out_dir), "regional_outputs": paths}
