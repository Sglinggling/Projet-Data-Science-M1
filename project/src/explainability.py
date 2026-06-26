"""
Explainability analysis for the Random Forest Nutri-Score classifier.

Only the Random Forest pipeline (models/random_forest.joblib) and the
standalone preprocessor (models/preprocessor.joblib) are loaded — keeping
peak RAM low on memory-constrained machines (MacBook Air, ~8 GB).

Outputs
-------
notebooks/figures/feature_importance_rf.png   — native RF importances
notebooks/figures/permutation_importance.png   — permutation importances
notebooks/figures/shap_summary.png             — SHAP beeswarm (2 000 rows)
notebooks/figures/shap_bar.png                 — SHAP global bar chart
"""

import gc

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from src.config import (
    MODELS_DIR,
    NUM_FEATURES,
    RAW_DIR,
    ROOT,
    RANDOM_STATE,
    SAMPLE_SIZE,
)
from src.preprocessing import get_train_test, load_and_clean

# ── Constants ──────────────────────────────────────────────────────────────────
FIGURES_DIR = ROOT / "notebooks" / "figures"
RAW_PATH = RAW_DIR / "en.openfoodfacts.org.products.csv"
SHAP_SAMPLE = 2_000

# Short display names for plots
FEATURE_LABELS = {
    "energy_100g":        "Energy",
    "fat_100g":           "Fat",
    "saturated-fat_100g": "Saturated fat",
    "carbohydrates_100g": "Carbohydrates",
    "sugars_100g":        "Sugars",
    "proteins_100g":      "Proteins",
    "salt_100g":          "Salt",
    "fiber_100g":         "Fiber",
}
LABELS = [FEATURE_LABELS[f] for f in NUM_FEATURES]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _savefig(name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → notebooks/figures/{name}", flush=True)


# ── 1. Native feature importance ───────────────────────────────────────────────

def plot_feature_importance(rf_clf, feature_names: list[str]) -> np.ndarray:
    """Bar chart of RF mean impurity decrease (Gini importance)."""
    print("\n[1/3] Native feature importance …", flush=True)

    importances = rf_clf.feature_importances_
    order = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(importances)))
    ax.barh(
        [feature_names[i] for i in order],
        importances[order],
        color=colors,
    )
    ax.set_xlabel("Mean decrease in impurity (Gini)")
    ax.set_title("Random Forest — native feature importance")
    ax.tick_params(axis="y", labelsize=10)
    plt.tight_layout()
    _savefig("feature_importance_rf.png")

    top3 = [feature_names[i] for i in np.argsort(importances)[::-1][:3]]
    print(f"  Top 3 (native): {top3}", flush=True)
    return importances


# ── 2. Permutation importance ──────────────────────────────────────────────────

def plot_permutation_importance(pipe, X_test: pd.DataFrame, y_test: pd.Series) -> np.ndarray:
    """Permutation importance with error bars (n_repeats=10, f1_macro)."""
    print("\n[2/3] Permutation importance (n_repeats=10, f1_macro) …", flush=True)
    print("  This may take ~1–2 min …", flush=True)

    result = permutation_importance(
        pipe,
        X_test,
        y_test,
        scoring="f1_macro",
        n_repeats=10,
        n_jobs=1,          # n_jobs=-1 spawns subprocesses → SIGURG on macOS sandbox
        random_state=RANDOM_STATE,
    )

    means = result.importances_mean
    stds = result.importances_std
    order = np.argsort(means)
    feat_labels = [LABELS[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(means)))
    ax.barh(
        feat_labels,
        means[order],
        xerr=stds[order],
        color=colors,
        capsize=4,
        error_kw={"elinewidth": 1.2},
    )
    ax.set_xlabel("Mean F1-macro decrease (±1 std over 10 repeats)")
    ax.set_title("Random Forest — permutation importance")
    ax.tick_params(axis="y", labelsize=10)
    plt.tight_layout()
    _savefig("permutation_importance.png")

    top3 = [LABELS[i] for i in np.argsort(means)[::-1][:3]]
    print(f"  Top 3 (permutation): {top3}", flush=True)
    return means


# ── 3. SHAP ────────────────────────────────────────────────────────────────────

def plot_shap(rf_clf, X_test_scaled: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """
    SHAP TreeExplainer on a 2 000-row subsample of the (already scaled) test set.

    SHAP 0.52 TreeExplainer for multiclass RF returns shap_values of shape
    (n_samples, n_features, n_classes).  We aggregate |values| over classes
    for a single global importance ranking.
    """
    print(f"\n[3/3] SHAP TreeExplainer (sample={SHAP_SAMPLE}) …", flush=True)

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_test_scaled), size=min(SHAP_SAMPLE, len(X_test_scaled)), replace=False)
    X_sample = X_test_scaled[idx]

    explainer = shap.TreeExplainer(rf_clf)
    print("  Computing SHAP values …", flush=True)
    shap_values = explainer.shap_values(X_sample)

    # ── normalise shape ────────────────────────────────────────────────────────
    # Depending on SHAP version/RF type, shap_values can be:
    #   list of (n, p) arrays of length n_classes   → stack to (n, p, C)
    #   ndarray (n, p, C)
    #   ndarray (n, p)  (binary or already aggregated)
    if isinstance(shap_values, list):
        sv = np.stack(shap_values, axis=-1)          # (n, p, C)
    else:
        sv = shap_values
        if sv.ndim == 2:
            sv = sv[:, :, np.newaxis]                # (n, p, 1)

    # mean |SHAP| over classes → (n, p)
    sv_agg = np.abs(sv).mean(axis=-1)

    # ── beeswarm (summary plot) ───────────────────────────────────────────────
    print("  Plotting beeswarm …", flush=True)
    shap.summary_plot(
        sv_agg,
        features=X_sample,
        feature_names=feature_names,
        show=False,
        max_display=len(feature_names),
    )
    plt.title("SHAP — beeswarm (mean |SHAP| over classes, RF)", pad=12)
    plt.tight_layout()
    _savefig("shap_summary.png")

    # ── global bar chart ──────────────────────────────────────────────────────
    print("  Plotting SHAP bar chart …", flush=True)
    mean_abs = sv_agg.mean(axis=0)          # (p,)
    order = np.argsort(mean_abs)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(mean_abs)))
    ax.barh(
        [feature_names[i] for i in order],
        mean_abs[order],
        color=colors,
    )
    ax.set_xlabel("Mean |SHAP value| (averaged over 2 000 samples × 5 classes)")
    ax.set_title("SHAP global importance — Random Forest")
    ax.tick_params(axis="y", labelsize=10)
    plt.tight_layout()
    _savefig("shap_bar.png")

    top3 = [feature_names[i] for i in np.argsort(mean_abs)[::-1][:3]]
    print(f"  Top 3 (SHAP): {top3}", flush=True)
    return mean_abs


# ── Synthesis ──────────────────────────────────────────────────────────────────

def print_synthesis(
    native_imp: np.ndarray,
    perm_imp: np.ndarray,
    shap_imp: np.ndarray,
    feature_names: list[str],
) -> None:
    def top3(arr):
        return [feature_names[i] for i in np.argsort(arr)[::-1][:3]]

    t_native = top3(native_imp)
    t_perm   = top3(perm_imp)
    t_shap   = top3(shap_imp)
    all_top  = t_native + t_perm + t_shap
    consensus = sorted(set(all_top), key=lambda f: -all_top.count(f))

    print(f"\n{'=' * 60}", flush=True)
    print("SYNTHESIS — top 3 influential nutrients per method", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  Native (Gini)  : {t_native}", flush=True)
    print(f"  Permutation    : {t_perm}", flush=True)
    print(f"  SHAP           : {t_shap}", flush=True)
    print(f"\n  Overall consensus (most cited first): {consensus}", flush=True)

    if set(t_native[:2]) == set(t_perm[:2]) == set(t_shap[:2]):
        print(
            "\n  ✓ Strong agreement across all 3 methods: the top-2 nutrients "
            "are consistent, confirming robustness of the ranking.",
            flush=True,
        )
    elif len(set(consensus[:3]) & set(t_native) & set(t_perm) & set(t_shap)) >= 2:
        print(
            "\n  ~ Partial agreement: at least 2 of the top-3 nutrients appear "
            "across all 3 methods.",
            flush=True,
        )
    else:
        print(
            "\n  ! Methods diverge — check for correlated features or "
            "data-specific artefacts.",
            flush=True,
        )
    print(f"{'=' * 60}", flush=True)


# ── Orchestrator ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60, flush=True)
    print("Loading data …", flush=True)
    X, y = load_and_clean(RAW_PATH, nrows=SAMPLE_SIZE)
    _, X_test, _, y_test = get_train_test(X, y)
    print(f"Test set : {X_test.shape}", flush=True)
    print("=" * 60, flush=True)

    # ── load only RF pipeline + standalone preprocessor ───────────────────────
    print("\nLoading Random Forest pipeline …", flush=True)
    pipe = joblib.load(MODELS_DIR / "random_forest.joblib")
    print("  RF pipeline loaded.", flush=True)

    # Isolate the fitted classifier and preprocessor from the pipeline
    rf_clf = pipe.named_steps["clf"]
    pre    = pipe.named_steps["pre"]

    # Scale test data once (needed for permutation importance + SHAP)
    X_test_scaled = pre.transform(X_test)
    print(f"  X_test scaled : {X_test_scaled.shape}", flush=True)

    # ── 1. Native importance ──────────────────────────────────────────────────
    native_imp = plot_feature_importance(rf_clf, LABELS)

    # ── 2. Permutation importance (on the full pipeline) ──────────────────────
    perm_imp = plot_permutation_importance(pipe, X_test, y_test)

    # ── 3. SHAP ───────────────────────────────────────────────────────────────
    shap_imp = plot_shap(rf_clf, X_test_scaled, LABELS)

    # ── 4. Synthesis ──────────────────────────────────────────────────────────
    print_synthesis(native_imp, perm_imp, shap_imp, LABELS)

    # Free RF from memory
    del pipe, rf_clf, pre, X_test_scaled
    gc.collect()
    print("\nDone. All figures saved to notebooks/figures/.", flush=True)


if __name__ == "__main__":
    main()
