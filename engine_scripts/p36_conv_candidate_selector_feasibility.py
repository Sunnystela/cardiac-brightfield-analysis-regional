"""Candidate-selector FEASIBILITY diagnostic, using the 58 human-labeled
candidate-review observations (P36_CONV_CANDIDATE_HUMAN_LABEL_TEMPLATE.csv
+ candidate_features.csv + review_index.csv).

STRICT SCOPE, per explicit instruction: DIAGNOSTIC ONLY.
  - No production selector changed. configs/default.yaml NOT touched.
  - SAM2 is not re-trained/fine-tuned anywhere in this script.
  - No canonical morphology/functional result is modified.
  - No full-dataset production run.
  - scikit-learn was installed into this project's .venv (via `uv pip
    install`, NOT added to pyproject.toml) specifically for the 3 small
    models this round explicitly asked for (logistic regression, a small
    random forest, shallow gradient boosting) -- BSD-3 licensed, the
    standard tool for exactly this task; no existing project dependency
    already provides these. Disclosed here per project convention on
    introducing a new dependency; not a production dependency change.

DATA/FEATURE EXCLUSIONS, per explicit instruction:
  - human_comment, error_direction, error_extent, phenotype_note,
    original_human_score are NEVER used as model input (annotation/
    evaluation metadata only).
  - well, day, and any row index are NEVER used as model input (to
    prevent memorizing a specific well/day).
  - existing_selected_candidate_index / is_existing_selected / area_
    ratio_to_current_selected are candidate_features.csv columns marked
    "metadata only" from the prior labeling round and are EXCLUDED from
    model input here too -- including them would let a model trivially
    reconstruct "which index production already picks," which is
    exactly the fixed-index baseline this experiment compares against.
  - the HUMAN phenotype label ("translucent/balloon-like") is never used
    as input -- only image-derived candidate_features are, so that
    whether those features can explain the phenotype can be evaluated
    afterwards (section 6C), not assumed beforehand.
  - modality (IMAGE/VIDEO) is compared WITH and WITHOUT, since it is a
    legitimate capture-regime context feature, not a well/day identifier.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE = REPO_ROOT / "results/_REVIEW_LATEST/P36_CONV_SCREENING/CANDIDATE_SELECTOR_HUMAN_LABELING"
OUTPUT_DIR = REPO_ROOT / "results/_REVIEW_LATEST/P36_CONV_SCREENING/CANDIDATE_SELECTOR_FEASIBILITY"

TRANSLUCENT_LOBE_WELLS = {"d1", "g1"}

# candidate-level features allowed as model input (see module docstring for exclusions)
FEATURE_COLUMNS = [
    "sam2_confidence_score", "sam2_score_rank", "area_fraction", "perimeter_px",
    "connected_component_count", "largest_component_fraction", "solidity", "circularity",
    "eccentricity", "aspect_ratio", "border_touch_fraction", "mean_intensity_inside_mask",
    "sd_intensity_inside_mask", "mask_vs_background_contrast", "mean_gradient_near_boundary",
    "texture_local_variance", "edge_support_fraction", "area_rank", "area_ratio_to_largest_candidate",
    "overlap_with_candidate_other1", "overlap_with_candidate_other2",
    "centroid_displacement_other1_px", "centroid_displacement_other2_px",
    "extra_area_px_beyond_core_candidate", "extra_region_mean_intensity",
    "extra_region_contiguous_with_core_fraction", "extra_region_intermediate_brightness_diagnostic_only",
    "is_core_candidate_this_observation",
]
MODALITY_FEATURE = ["modality_is_video"]


def load_data():
    labels = list(csv.DictReader(open(BASE / "P36_CONV_CANDIDATE_HUMAN_LABEL_TEMPLATE.csv")))
    features = list(csv.DictReader(open(BASE / "candidate_features.csv")))
    label_by_obs = {(r["well"], r["day"]): r["human_best_candidate"].strip() for r in labels}

    obs_candidates: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for r in features:
        key = (r["well"], r["day"])
        idx = int(r["candidate_index"])
        obs_candidates[key][idx] = r
    return labels, label_by_obs, obs_candidates


def to_float(v):
    if v is None or v == "" or v == "None":
        return np.nan
    if v == "True":
        return 1.0
    if v == "False":
        return 0.0
    try:
        return float(v)
    except ValueError:
        return np.nan


def build_feature_vector(cand_row: dict, use_modality: bool) -> list[float]:
    vec = [to_float(cand_row.get(c)) for c in FEATURE_COLUMNS]
    if use_modality:
        vec.append(1.0 if cand_row["modality"] == "VIDEO" else 0.0)
    return vec


def build_dataset(obs_candidates, label_by_obs, wells_filter=None, use_modality=True, exclude_none=True):
    """Returns (X, y_binary, groups(well), obs_keys, cand_indices) at the
    CANDIDATE level -- one row per candidate, y_binary=1 iff this candidate
    is the human_best_candidate for its observation. NONE-labeled
    observations excluded when exclude_none=True."""
    X, y, groups, obs_keys, cand_idx = [], [], [], [], []
    for key, cands in obs_candidates.items():
        well, day = key
        if wells_filter is not None and well not in wells_filter:
            continue
        label = label_by_obs.get(key, "")
        if exclude_none and label not in ("0", "1", "2"):
            continue
        for idx in sorted(cands):
            cand_row = cands[idx]
            X.append(build_feature_vector(cand_row, use_modality))
            y.append(1 if label == str(idx) else 0)
            groups.append(well)
            obs_keys.append(key)
            cand_idx.append(idx)
    return np.array(X, dtype=float), np.array(y, dtype=int), groups, obs_keys, cand_idx


def impute_nan(X: np.ndarray, fill_values: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if fill_values is None:
        fill_values = np.nanmedian(X, axis=0)
        fill_values = np.where(np.isnan(fill_values), 0.0, fill_values)
    X = X.copy()
    inds = np.where(np.isnan(X))
    X[inds] = np.take(fill_values, inds[1])
    return X, fill_values


def predict_from_scores(scores_by_idx: dict[int, float]) -> int:
    return max(scores_by_idx, key=scores_by_idx.get)


# ---------------- baselines ----------------

def baseline_fixed_index(obs_candidates, index: int) -> dict:
    return {key: index for key in obs_candidates}


def baseline_highest_sam2_score(obs_candidates) -> dict:
    out = {}
    for key, cands in obs_candidates.items():
        out[key] = predict_from_scores({i: float(c["sam2_confidence_score"]) for i, c in cands.items()})
    return out


def baseline_largest_area(obs_candidates) -> dict:
    out = {}
    for key, cands in obs_candidates.items():
        out[key] = predict_from_scores({i: float(c["area_fraction"]) for i, c in cands.items()})
    return out


def baseline_structural_prototype(obs_candidates, min_dominant_fraction: float = 0.9) -> dict:
    """Reuses the SAME rule as prototype_candidate_selection.select_
    candidate_structural_prototype (gate: largest_component_fraction >=
    0.9, then largest area among qualifying), computed directly from
    candidate_features.csv columns instead of re-running SAM2."""
    out = {}
    for key, cands in obs_candidates.items():
        qualifying = {i: c for i, c in cands.items() if float(c["largest_component_fraction"]) >= min_dominant_fraction}
        if not qualifying:
            out[key] = None
            continue
        out[key] = predict_from_scores({i: float(c["area_fraction"]) for i, c in qualifying.items()})
    return out


def evaluate_predictions(predictions: dict, label_by_obs: dict, wells_filter=None) -> dict:
    y_true, y_pred, obs_keys = [], [], []
    for key, pred in predictions.items():
        if wells_filter is not None and key[0] not in wells_filter:
            continue
        label = label_by_obs.get(key, "")
        if label not in ("0", "1", "2"):
            continue
        y_true.append(int(label))
        y_pred.append(pred if pred is not None else -1)
        obs_keys.append(key)
    n = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / n if n else None
    per_class_recall = {}
    for c in (0, 1, 2):
        idxs = [i for i, t in enumerate(y_true) if t == c]
        per_class_recall[c] = (sum(1 for i in idxs if y_pred[i] == c) / len(idxs)) if idxs else None
    cm = np.zeros((3, 4), dtype=int)  # rows=true 0/1/2, cols=pred 0/1/2/abstain(-1 or other)
    for t, p in zip(y_true, y_pred):
        col = p if p in (0, 1, 2) else 3
        cm[t, col] += 1
    per_well_acc = {}
    wells = sorted(set(k[0] for k in obs_keys))
    for w in wells:
        idxs = [i for i, k in enumerate(obs_keys) if k[0] == w]
        per_well_acc[w] = sum(1 for i in idxs if y_true[i] == y_pred[i]) / len(idxs) if idxs else None
    macro_balanced_acc = np.mean([v for v in per_class_recall.values() if v is not None])
    return {"n": n, "accuracy": accuracy, "per_class_recall": per_class_recall, "confusion_matrix": cm.tolist(),
            "per_well_accuracy": per_well_acc, "macro_balanced_accuracy": float(macro_balanced_acc)}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figdir = OUTPUT_DIR
    labels, label_by_obs, obs_candidates = load_data()

    wells = sorted(set(k[0] for k in obs_candidates))
    label_counts = Counter(label_by_obs.values())
    print("label distribution:", label_counts)
    print("wells:", wells)

    # ================= TASK 2: baselines =================
    baselines = {
        "A_fixed_index_1": baseline_fixed_index(obs_candidates, 1),
        "B_fixed_index_2": baseline_fixed_index(obs_candidates, 2),
        "C_highest_sam2_score": baseline_highest_sam2_score(obs_candidates),
        "D_largest_area": baseline_largest_area(obs_candidates),
        "E_structural_prototype": baseline_structural_prototype(obs_candidates),
    }
    baseline_rows = []
    baseline_results = {}
    for name, preds in baselines.items():
        res = evaluate_predictions(preds, label_by_obs)
        baseline_results[name] = res
        baseline_rows.append({
            "baseline": name, "n": res["n"], "accuracy": round(res["accuracy"], 3) if res["accuracy"] is not None else None,
            "macro_balanced_accuracy": round(res["macro_balanced_accuracy"], 3),
            "recall_candidate_0": res["per_class_recall"][0], "recall_candidate_1": res["per_class_recall"][1],
            "recall_candidate_2": res["per_class_recall"][2],
            "confusion_matrix_rows_true0_1_2_cols_pred0_1_2_abstain": res["confusion_matrix"],
        })
    with open(OUTPUT_DIR / "baseline_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(baseline_rows[0].keys()))
        w.writeheader()
        w.writerows(baseline_rows)

    # ================= TASK 4: leave-one-well-out CV for learned models =================
    model_specs = {
        "logistic_regression": lambda: LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
        "random_forest_small": lambda: RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=3, class_weight="balanced", random_state=0),
        "gradient_boosting_shallow": lambda: GradientBoostingClassifier(n_estimators=60, max_depth=2, learning_rate=0.1, random_state=0),
    }

    lowo_predictions_rows = []
    model_comparison_rows = []
    per_model_confusion = {}
    per_model_predictions_by_key = {}  # for error contact sheet (best model only)

    for use_modality in (True, False):
        modality_tag = "with_modality" if use_modality else "no_modality"
        for model_name, model_ctor in model_specs.items():
            full_tag = f"{model_name}__{modality_tag}"
            all_probs_by_key = {}
            for held_out_well in wells:
                train_wells = [w for w in wells if w != held_out_well]
                X_train, y_train, _, _, _ = build_dataset(obs_candidates, label_by_obs, wells_filter=train_wells, use_modality=use_modality)
                X_train_imp, fill_values = impute_nan(X_train)
                scaler = StandardScaler().fit(X_train_imp)
                X_train_s = scaler.transform(X_train_imp)

                model = model_ctor()
                model.fit(X_train_s, y_train)

                # test on the held-out well's non-NONE observations
                test_keys = [k for k in obs_candidates if k[0] == held_out_well and label_by_obs.get(k) in ("0", "1", "2")]
                for key in test_keys:
                    cands = obs_candidates[key]
                    idxs = sorted(cands)
                    Xc = np.array([build_feature_vector(cands[i], use_modality) for i in idxs], dtype=float)
                    Xc_imp, _ = impute_nan(Xc, fill_values)
                    Xc_s = scaler.transform(Xc_imp)
                    probs = model.predict_proba(Xc_s)[:, 1] if len(model.classes_) > 1 else np.zeros(len(idxs))
                    all_probs_by_key[key] = dict(zip(idxs, probs))

            predictions = {key: predict_from_scores(probs) for key, probs in all_probs_by_key.items()}
            res = evaluate_predictions(predictions, label_by_obs)
            model_comparison_rows.append({
                "model": full_tag, "n": res["n"], "pooled_accuracy": round(res["accuracy"], 3) if res["accuracy"] is not None else None,
                "macro_balanced_accuracy": round(res["macro_balanced_accuracy"], 3),
                "recall_candidate_0": res["per_class_recall"][0], "recall_candidate_1": res["per_class_recall"][1],
                "recall_candidate_2": res["per_class_recall"][2],
            })
            per_model_confusion[full_tag] = res["confusion_matrix"]

            if use_modality and model_name == "random_forest_small":
                per_model_predictions_by_key["headline_model"] = (predictions, all_probs_by_key)

            for key, probs in all_probs_by_key.items():
                sorted_probs = sorted(probs.values(), reverse=True)
                margin = (sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else None
                lowo_predictions_rows.append({
                    "model": full_tag, "well": key[0], "day": key[1],
                    "human_best_candidate": label_by_obs.get(key), "predicted_best_candidate": predictions.get(key),
                    "correct": label_by_obs.get(key) == str(predictions.get(key)),
                    "top1_prob": round(sorted_probs[0], 3) if sorted_probs else None,
                    "top1_top2_margin": round(margin, 3) if margin is not None else None,
                    "candidate_probs": json.dumps({str(k): round(v, 3) for k, v in probs.items()}),
                })

    with open(OUTPUT_DIR / "leave_one_well_out_predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lowo_predictions_rows[0].keys()))
        w.writeheader()
        w.writerows(lowo_predictions_rows)
    with open(OUTPUT_DIR / "model_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(model_comparison_rows[0].keys()))
        w.writeheader()
        w.writerows(model_comparison_rows)

    print("\nBaselines:")
    for r in baseline_rows:
        print(" ", r["baseline"], "acc=", r["accuracy"], "macro_bal_acc=", r["macro_balanced_accuracy"])
    print("\nModels (LOWO-CV):")
    for r in model_comparison_rows:
        print(" ", r["model"], "acc=", r["pooled_accuracy"], "macro_bal_acc=", r["macro_balanced_accuracy"])

    # ================= confusion matrices figure =================
    all_tags = list(baseline_results.keys()) + list(per_model_confusion.keys())
    n_plots = len(all_tags)
    cols = 4
    rows_n = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 3.6 * rows_n))
    axes = np.array(axes).reshape(-1)
    for ax, tag in zip(axes, all_tags):
        cm = np.array(baseline_results[tag]["confusion_matrix"]) if tag in baseline_results else np.array(per_model_confusion[tag])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(4)); ax.set_xticklabels(["pred0", "pred1", "pred2", "abstain"], fontsize=7)
        ax.set_yticks(range(3)); ax.set_yticklabels(["true0", "true1", "true2"], fontsize=7)
        for i in range(3):
            for j in range(4):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
        ax.set_title(tag, fontsize=8)
    for ax in axes[len(all_tags):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(figdir / "confusion_matrices.png", dpi=110)
    plt.close(fig)

    # ================= accuracy by well figure =================
    fig, ax = plt.subplots(figsize=(12, 5))
    compare_tags = ["A_fixed_index_1", "B_fixed_index_2", "C_highest_sam2_score", "E_structural_prototype"]
    width = 0.15
    x = np.arange(len(wells))
    for i, tag in enumerate(compare_tags):
        accs = [baseline_results[tag]["per_well_accuracy"].get(w) for w in wells]
        ax.bar(x + i * width, [a if a is not None else 0 for a in accs], width, label=tag)
    headline_res = evaluate_predictions(per_model_predictions_by_key["headline_model"][0], label_by_obs)
    accs = [headline_res["per_well_accuracy"].get(w) for w in wells]
    ax.bar(x + len(compare_tags) * width, [a if a is not None else 0 for a in accs], width, label="learned_RF_with_modality (LOWO)")
    ax.set_xticks(x + width * len(compare_tags) / 2)
    ax.set_xticklabels(wells)
    ax.set_ylabel("per-well accuracy")
    ax.legend(fontsize=7)
    ax.set_title("Per-well accuracy: baselines vs learned selector (leave-one-well-out)")
    fig.tight_layout()
    fig.savefig(figdir / "accuracy_by_well.png", dpi=115)
    plt.close(fig)

    # ================= feature importance (random forest, refit on ALL non-NONE data) =================
    X_all, y_all, _, _, _ = build_dataset(obs_candidates, label_by_obs, use_modality=True)
    X_all_imp, _ = impute_nan(X_all)
    rf_full = RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=3, class_weight="balanced", random_state=0)
    rf_full.fit(StandardScaler().fit_transform(X_all_imp), y_all)
    feat_names = FEATURE_COLUMNS + MODALITY_FEATURE
    importances = rf_full.feature_importances_
    order = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh([feat_names[i] for i in order][::-1], importances[order][::-1])
    ax.set_title("Random forest feature importance (all 54 non-NONE observations, with modality)")
    fig.tight_layout()
    fig.savefig(figdir / "candidate_feature_importance.png", dpi=115)
    plt.close(fig)

    # ================= TASK 6A/B: candidate 2 vs 1 vs 0 feature distributions =================
    key_features_for_plot = ["area_fraction", "sam2_confidence_score", "largest_component_fraction", "solidity",
                              "circularity", "mask_vs_background_contrast", "extra_area_px_beyond_core_candidate"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for ax, feat in zip(axes.flat, key_features_for_plot):
        data_by_label = {lbl: [] for lbl in ("0", "1", "2")}
        for key, cands in obs_candidates.items():
            lbl = label_by_obs.get(key)
            if lbl not in ("0", "1", "2"):
                continue
            idx = int(lbl)
            if idx in cands:
                v = to_float(cands[idx].get(feat))
                if not np.isnan(v):
                    data_by_label[lbl].append(v)
        ax.boxplot([data_by_label["0"], data_by_label["1"], data_by_label["2"]], tick_labels=["best=0\n(n=4)", "best=1\n(n=11)", "best=2\n(n=39)"])
        ax.set_title(feat, fontsize=9)
    axes.flat[-1].axis("off")
    fig.suptitle("Feature value of the human-chosen best candidate, grouped by which index was chosen")
    fig.tight_layout()
    fig.savefig(figdir / "candidate_label_feature_distributions.png", dpi=115)
    plt.close(fig)

    # is candidate 2 simply the largest-area / highest-score candidate on those 39 observations?
    n_cand2_is_largest = 0
    n_cand2_is_highest_score = 0
    n_cand2_both = 0
    n_cand2_total = 0
    for key, cands in obs_candidates.items():
        if label_by_obs.get(key) != "2":
            continue
        n_cand2_total += 1
        areas = {i: float(c["area_fraction"]) for i, c in cands.items()}
        scores = {i: float(c["sam2_confidence_score"]) for i, c in cands.items()}
        is_largest = max(areas, key=areas.get) == 2
        is_highest = max(scores, key=scores.get) == 2
        n_cand2_is_largest += int(is_largest)
        n_cand2_is_highest_score += int(is_highest)
        n_cand2_both += int(is_largest and is_highest)
    cand2_explanation = {
        "n_best_is_2": n_cand2_total,
        "fraction_where_2_is_largest_area": round(n_cand2_is_largest / n_cand2_total, 2) if n_cand2_total else None,
        "fraction_where_2_is_highest_sam2_score": round(n_cand2_is_highest_score / n_cand2_total, 2) if n_cand2_total else None,
        "fraction_where_2_is_both": round(n_cand2_both / n_cand2_total, 2) if n_cand2_total else None,
    }
    with open(OUTPUT_DIR / "_candidate2_explanation.json", "w") as f:
        json.dump(cand2_explanation, f, indent=2)
    print("\ncandidate=2 explanation:", cand2_explanation)

    # ================= TASK 6C: translucent lobe subgroup =================
    lobe_rows = []
    lobe_keys = [k for k in obs_candidates if k[0] in TRANSLUCENT_LOBE_WELLS]
    predictions_headline, probs_headline = per_model_predictions_by_key["headline_model"]
    for key in lobe_keys:
        well, day = key
        label = label_by_obs.get(key, "")
        pred = predictions_headline.get(key)
        probs = probs_headline.get(key, {})
        lobe_rows.append({
            "well": well, "day": day, "human_best_candidate": label,
            "learned_selector_predicted": pred, "correct": (label == str(pred)) if label in ("0", "1", "2") else None,
            "candidate_probs": json.dumps({str(k): round(v, 3) for k, v in probs.items()}) if probs else None,
        })
    with open(OUTPUT_DIR / "translucent_lobe_subgroup.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lobe_rows[0].keys()))
        w.writeheader()
        w.writerows(lobe_rows)

    # ================= TASK 5: NONE / abstain diagnostic =================
    none_keys = [k for k, lbl in label_by_obs.items() if lbl == "NONE"]
    # refit headline model on ALL non-NONE observations (no held-out well needed here -- just inspect NONE cases)
    X_all2, y_all2, _, _, _ = build_dataset(obs_candidates, label_by_obs, use_modality=True)
    X_all2_imp, fill_values2 = impute_nan(X_all2)
    scaler2 = StandardScaler().fit(X_all2_imp)
    rf_for_none = RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=3, class_weight="balanced", random_state=0)
    rf_for_none.fit(scaler2.transform(X_all2_imp), y_all2)

    none_diag_rows = []
    for key in none_keys:
        cands = obs_candidates[key]
        idxs = sorted(cands)
        Xc = np.array([build_feature_vector(cands[i], True) for i in idxs], dtype=float)
        Xc_imp, _ = impute_nan(Xc, fill_values2)
        probs = rf_for_none.predict_proba(scaler2.transform(Xc_imp))[:, 1]
        sorted_probs = sorted(probs, reverse=True)
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else None
        none_diag_rows.append({
            "well": key[0], "day": key[1], "top1_prob": float(sorted_probs[0]), "margin": float(margin) if margin is not None else None,
            "candidate_probs": dict(zip(idxs, [float(p) for p in probs])),
            "candidate_sam2_scores": {i: float(cands[i]["sam2_confidence_score"]) for i in idxs},
            "candidate_largest_component_fraction": {i: float(cands[i]["largest_component_fraction"]) for i in idxs},
        })

    # compare against non-NONE observations' top1_prob/margin distribution for the SAME model
    non_none_top1, non_none_margin = [], []
    for key, cands in obs_candidates.items():
        if label_by_obs.get(key) not in ("0", "1", "2"):
            continue
        idxs = sorted(cands)
        Xc = np.array([build_feature_vector(cands[i], True) for i in idxs], dtype=float)
        Xc_imp, _ = impute_nan(Xc, fill_values2)
        probs = rf_for_none.predict_proba(scaler2.transform(Xc_imp))[:, 1]
        sp = sorted(probs, reverse=True)
        non_none_top1.append(sp[0])
        non_none_margin.append(sp[0] - sp[1] if len(sp) > 1 else 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].boxplot([non_none_top1, [r["top1_prob"] for r in none_diag_rows]], tick_labels=[f"non-NONE (n={len(non_none_top1)})", f"NONE (n={len(none_diag_rows)})"])
    axes[0].set_title("top-1 predicted probability")
    axes[1].boxplot([non_none_margin, [r["margin"] for r in none_diag_rows]], tick_labels=[f"non-NONE (n={len(non_none_margin)})", f"NONE (n={len(none_diag_rows)})"])
    axes[1].set_title("top1-top2 probability margin")
    fig.suptitle("Does human_best_candidate=NONE separate as low-confidence / high-disagreement? (diagnostic only, no threshold fixed)")
    fig.tight_layout()
    fig.savefig(figdir / "none_abstain_diagnostic.png", dpi=115)
    plt.close(fig)

    with open(OUTPUT_DIR / "_none_diagnostic.json", "w") as f:
        json.dump({"none_cases": none_diag_rows,
                   "non_none_top1_prob_median": float(np.median(non_none_top1)),
                   "non_none_margin_median": float(np.median(non_none_margin))}, f, indent=2, default=str)

    # ================= error contact sheet =================
    build_error_contact_sheet(predictions_headline, label_by_obs, OUTPUT_DIR / "selector_error_contact_sheet.png")

    print(f"\nWrote outputs to {OUTPUT_DIR}")


def build_error_contact_sheet(predictions: dict, label_by_obs: dict, out_path: Path) -> None:
    from p36_conv_candidate_selector_human_labeling import IMAGE_DAYS, load_source_frame, overlay_mask, label_band
    from cardio_organoid.segmentation.sam2_segmenter import get_device, load_predictor, segment_frame
    from global_boundary_motion_diagnostic import _load_config

    error_keys = [k for k, p in predictions.items() if label_by_obs.get(k) in ("0", "1", "2") and label_by_obs.get(k) != str(p)]
    if not error_keys:
        print("No selector errors found -- skipping error contact sheet.")
        return

    config = _load_config()
    sam2_cfg = config["segmentation"]["sam2"]
    device = get_device(sam2_cfg["device"])
    predictor = load_predictor(checkpoint_path=REPO_ROOT / sam2_cfg["checkpoint_path"], model_cfg=sam2_cfg["model_cfg"], device=device)

    rows_img = []
    target_w = 280
    for well, day_str in sorted(error_keys):
        day = int(day_str)
        modality = "IMAGE" if day in IMAGE_DAYS else "VIDEO"
        frame_rgb = load_source_frame(well, day)
        scale = target_w / frame_rgb.shape[1]
        target_h = int(frame_rgb.shape[0] * scale)
        frame_small = cv2.resize(frame_rgb, (target_w, target_h))
        candidates = segment_frame(predictor, frame_rgb)
        tiles = [label_band(frame_small, f"{well} D{day} {modality}\nORIGINAL", band_h=40)]
        human_choice = label_by_obs.get((well, day_str))
        predicted_choice = predictions.get((well, day_str))
        for c in candidates[:3]:
            mask_small = cv2.resize(c.mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            ov = overlay_mask(frame_small, mask_small)
            tag = f"candidate={c.index}"
            if str(c.index) == human_choice:
                tag += " [HUMAN CHOICE]"
            if c.index == predicted_choice:
                tag += " [PREDICTED]"
            tiles.append(label_band(ov, tag, band_h=40))
        rows_img.append(np.hstack(tiles))
    max_w = max(r.shape[1] for r in rows_img)
    def pad(r):
        return np.hstack([r, np.full((r.shape[0], max_w - r.shape[1], 3), 255, dtype=np.uint8)]) if r.shape[1] < max_w else r
    sheet = np.vstack([pad(r) for r in rows_img])
    header = np.full((36, sheet.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(header, f"Selector errors ({len(error_keys)} of 54 non-NONE observations) -- learned RF (with modality), LOWO-CV", (6, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    sheet = np.vstack([header, sheet])
    cv2.imwrite(str(out_path), sheet)
    print(f"Wrote error contact sheet ({len(error_keys)} errors) to {out_path}")


if __name__ == "__main__":
    main()
