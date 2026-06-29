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

FIGURES_DIR = ROOT / "notebooks" / "figures"
RAW_PATH = RAW_DIR / "en.openfoodfacts.org.products.csv"
SHAP_SAMPLE = 2_000  # calculer SHAP sur tout le test set serait trop long

# Noms courts pour les graphiques
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


def _savefig(name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → notebooks/figures/{name}", flush=True)


def plot_feature_importance(rf_clf, feature_names: list[str]) -> np.ndarray:
    """Barplot de l'importance native du RF (diminution moyenne de l'impureté Gini)."""
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


def plot_permutation_importance(pipe, X_test: pd.DataFrame, y_test: pd.Series) -> np.ndarray:
    """Importance par permutation avec barres d'erreur (10 répétitions, score f1_macro)."""
    print("\n[2/3] Permutation importance (n_repeats=10, f1_macro) …", flush=True)
    print("  This may take ~1–2 min …", flush=True)

    result = permutation_importance(
        pipe,
        X_test,
        y_test,
        scoring="f1_macro",
        n_repeats=10,
        n_jobs=1,          # n_jobs=-1 crée des sous-processus → SIGURG sur le sandbox macOS
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


def plot_shap(rf_clf, X_test_scaled: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """SHAP TreeExplainer sur 2 000 lignes du test set (déjà normalisées).

    Le TreeExplainer renvoie des valeurs de forme (n, features, classes) pour un RF multiclasse.
    On agrège |SHAP| sur les classes pour obtenir un classement global unique.
    """
    print(f"\n[3/3] SHAP TreeExplainer (sample={SHAP_SAMPLE}) …", flush=True)

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_test_scaled), size=min(SHAP_SAMPLE, len(X_test_scaled)), replace=False)
    X_sample = X_test_scaled[idx]

    explainer = shap.TreeExplainer(rf_clf)
    print("  Computing SHAP values …", flush=True)
    shap_values = explainer.shap_values(X_sample)

    # Selon la version de SHAP/type de RF, shap_values peut être :
    #   liste de tableaux (n, p) de longueur n_classes → on stack en (n, p, C)
    #   ndarray (n, p, C) ou (n, p) si déjà agrégé
    if isinstance(shap_values, list):
        sv = np.stack(shap_values, axis=-1)          # (n, p, C)
    else:
        sv = shap_values
        if sv.ndim == 2:
            sv = sv[:, :, np.newaxis]                # (n, p, 1)

    # Moyenne de |SHAP| sur les classes → (n, p)
    sv_agg = np.abs(sv).mean(axis=-1)

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


def main() -> None:
    print("=" * 60, flush=True)
    print("Loading data …", flush=True)
    X, y = load_and_clean(RAW_PATH, nrows=SAMPLE_SIZE)
    _, X_test, _, y_test = get_train_test(X, y)
    print(f"Test set : {X_test.shape}", flush=True)
    print("=" * 60, flush=True)

    print("\nLoading Random Forest pipeline …", flush=True)
    pipe = joblib.load(MODELS_DIR / "random_forest.joblib")
    print("  RF pipeline loaded.", flush=True)

    # On extrait le classifieur et le préprocesseur du pipeline pour les utiliser séparément
    rf_clf = pipe.named_steps["clf"]
    pre    = pipe.named_steps["pre"]

    # Test set normalisé une seule fois — réutilisé par la permutation et SHAP
    X_test_scaled = pre.transform(X_test)
    print(f"  X_test scaled : {X_test_scaled.shape}", flush=True)

    native_imp = plot_feature_importance(rf_clf, LABELS)
    perm_imp = plot_permutation_importance(pipe, X_test, y_test)
    shap_imp = plot_shap(rf_clf, X_test_scaled, LABELS)
    print_synthesis(native_imp, perm_imp, shap_imp, LABELS)

    del pipe, rf_clf, pre, X_test_scaled
    gc.collect()
    print("\nDone. All figures saved to notebooks/figures/.", flush=True)


if __name__ == "__main__":
    main()
