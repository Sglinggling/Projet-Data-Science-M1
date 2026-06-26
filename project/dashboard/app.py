"""
Streamlit decisional dashboard — Nutri-Score classifier.

Run from the project/ directory:
    streamlit run dashboard/app.py

Models and figures must already exist (train_models.py + evaluate_models.py
+ explainability.py must have been run first).  Nothing is retrained here.
"""

import sys
from pathlib import Path

# ── path bootstrap: make `src` importable from any working directory ──────────
_HERE = Path(__file__).resolve().parent          # project/dashboard/
_ROOT = _HERE.parent                             # project/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import gc
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import MODELS_DIR, NUM_FEATURES

# ── Paths ─────────────────────────────────────────────────────────────────────
FIGURES_DIR = _ROOT / "notebooks" / "figures"

GRADE_COLORS = {
    "a": "#038141",
    "b": "#85BB2F",
    "c": "#FECB02",
    "d": "#EE8100",
    "e": "#E63E11",
}

SKLEARN_MODELS = {
    "Random Forest":       "random_forest",
    "Gradient Boosting":   "gradient_boosting",
    "SVM RBF":             "svm",
    "Logistic Regression": "logreg",
}

FEATURE_LABELS = {
    "energy_100g":        "Énergie (kcal/100g)",
    "fat_100g":           "Matières grasses (g/100g)",
    "saturated-fat_100g": "Graisses saturées (g/100g)",
    "carbohydrates_100g": "Glucides (g/100g)",
    "sugars_100g":        "Sucres (g/100g)",
    "proteins_100g":      "Protéines (g/100g)",
    "salt_100g":          "Sel (g/100g)",
    "fiber_100g":         "Fibres (g/100g)",
}

FEATURE_DEFAULTS = {
    "energy_100g":        1100.0,
    "fat_100g":           8.0,
    "saturated-fat_100g": 3.0,
    "carbohydrates_100g": 30.0,
    "sugars_100g":        5.0,
    "proteins_100g":      5.0,
    "salt_100g":          0.5,
    "fiber_100g":         2.0,
}

FEATURE_BOUNDS = {
    "energy_100g":        (0.0, 3700.0),
    "fat_100g":           (0.0, 100.0),
    "saturated-fat_100g": (0.0, 100.0),
    "carbohydrates_100g": (0.0, 100.0),
    "sugars_100g":        (0.0, 100.0),
    "proteins_100g":      (0.0, 100.0),
    "salt_100g":          (0.0, 100.0),
    "fiber_100g":         (0.0, 100.0),
}

# Total products in the cleaned OFF sample (deterministic: 80k rows → valid
# Nutri-Score grades only; split 80/20 → 49 525 train + 12 382 test)
N_PRODUCTS = 61_907
N_CLASSES = 5


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fig_path(name: str) -> Path:
    return FIGURES_DIR / name


def _show_fig(name: str, caption: str = "", use_column_width: bool = True) -> None:
    p = _fig_path(name)
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=use_column_width)
    else:
        st.warning(f"Figure manquante : `{p.relative_to(_ROOT)}`")


def _load_comparison() -> pd.DataFrame | None:
    p = MODELS_DIR / "comparison_results.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


# ── Lazy model loading (cached per model key) ──────────────────────────────────

@st.cache_resource(show_spinner="Chargement du modèle …")
def _load_sklearn_pipeline(key: str):
    path = MODELS_DIR / f"{key}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_resource(show_spinner="Chargement du label mapping …")
def _load_label_map() -> dict:
    p = MODELS_DIR / "label_mapping.joblib"
    if not p.exists():
        return {"int_to_label": {i: c for i, c in enumerate("abcde")},
                "label_to_int": {c: i for i, c in enumerate("abcde")}}
    return joblib.load(p)


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Nutri-Score Classifier",
    page_icon="🥗",
    layout="wide",
)

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Vue générale",
    "📊 Analyse des données",
    "🏆 Comparaison des modèles",
    "🔮 Simulation",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Vue générale
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.title("🥗 Nutri-Score Classifier — Open Food Facts")
    st.markdown(
        """
        > **Pitch métier** — Un fabricant qui conçoit un nouveau produit alimentaire
        > doit anticiper son Nutri-Score **avant** la mise en marché.
        > Ce tableau de bord prédit le Nutri-Score (A → E) à partir des **8 valeurs
        > nutritionnelles pour 100 g** déclarées sur l'étiquette, en s'appuyant sur
        > des modèles entraînés sur **61 907 produits Open Food Facts**.
        > Aucun réentraînement : tout est pré-calculé, la prédiction est instantanée.
        """
    )

    st.divider()

    # ── KPIs ─────────────────────────────────────────────────────────────────
    df_cmp = _load_comparison()
    best_model = "Random Forest"
    best_f1 = 0.9587
    if df_cmp is not None and not df_cmp.empty:
        best_row = df_cmp.sort_values("f1_macro", ascending=False).iloc[0]
        best_model = best_row["model"]
        best_f1 = best_row["f1_macro"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Produits (dataset nettoyé)", f"{N_PRODUCTS:,}".replace(",", " "))
    k2.metric("Classes Nutri-Score", str(N_CLASSES), "A → E")
    k3.metric("Meilleur modèle", best_model)
    k4.metric("F1-macro (test set)", f"{best_f1:.4f}")

    st.divider()

    # ── Distribution des Nutri-Scores ────────────────────────────────────────
    st.subheader("Distribution des Nutri-Scores dans le dataset")
    _show_fig("target_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Analyse des données
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Analyse exploratoire des données")

    st.subheader("Distributions nutritionnelles par Nutri-Score")
    st.markdown(
        "Chaque feature (nutriment/100g) montre un gradient clair A → E : "
        "les produits de grade A ont peu de sucres, graisses saturées et sel, "
        "davantage de fibres et protéines."
    )
    _show_fig("nutritional_distributions.png")

    st.divider()

    st.subheader("Carte de corrélation des nutriments")
    st.markdown(
        "Sucres et glucides sont fortement corrélés (r > 0.7). "
        "Sel et graisses saturées sont des prédicteurs indépendants, "
        "ce qui justifie de conserver les 8 features."
    )
    _show_fig("correlation_heatmap.png")

    st.divider()

    st.subheader("Boîtes à moustaches — nutriments par grade")
    st.markdown(
        "Les valeurs médianes séparent nettement les classes A/B des classes D/E, "
        "confirmant que les features numériques suffisent à distinguer les Nutri-Scores."
    )
    _show_fig("boxplots_by_grade.png")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Comparaison des modèles
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("Comparaison des modèles entraînés")

    # ── Tableau comparatif ────────────────────────────────────────────────────
    st.subheader("Métriques sur le test set (trié par F1-macro ↓)")
    if df_cmp is not None:
        df_styled = (
            df_cmp
            .sort_values("f1_macro", ascending=False)
            .reset_index(drop=True)
            .rename(columns={
                "model":           "Modèle",
                "accuracy":        "Accuracy",
                "precision_macro": "Précision macro",
                "recall_macro":    "Recall macro",
                "f1_macro":        "F1-macro",
            })
        )
        st.dataframe(
            df_styled.style.format({
                "Accuracy":        "{:.4f}",
                "Précision macro": "{:.4f}",
                "Recall macro":    "{:.4f}",
                "F1-macro":        "{:.4f}",
            }).highlight_max(
                subset=["F1-macro"],
                color="#d4edda",
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("Fichier `models/comparison_results.csv` introuvable.")

    st.divider()

    # ── Barplot comparatif ────────────────────────────────────────────────────
    st.subheader("Barplot comparatif — 4 métriques × 5 modèles")
    _show_fig("model_comparison.png")

    st.divider()

    # ── Matrice de confusion meilleur modèle ──────────────────────────────────
    st.subheader("Matrice de confusion — Random Forest (meilleur modèle)")
    st.markdown(
        "Normalisée par ligne (% des vrais positifs par classe). "
        "Les erreurs se concentrent sur les grades adjacents (b↔c, c↔d), "
        "ce qui est attendu : leurs profils nutritionnels se chevauchent."
    )
    _show_fig("confmat_random_forest.png")

    st.divider()

    # ── Interprétabilité ──────────────────────────────────────────────────────
    st.subheader("Interprétabilité — quels nutriments comptent le plus ?")
    col_shap, col_native = st.columns(2)
    with col_shap:
        st.markdown("**SHAP beeswarm** (2 000 échantillons, agrégé sur 5 classes)")
        _show_fig("shap_summary.png")
    with col_native:
        st.markdown("**Importance native RF** (diminution moyenne d'impureté)")
        _show_fig("feature_importance_rf.png")

    st.info(
        "**Consensus des 3 méthodes** (Gini, permutation, SHAP) : "
        "**Sel · Graisses saturées · Sucres** sont les 3 nutriments les plus "
        "influents pour prédire le Nutri-Score — résultat cohérent avec "
        "l'algorithme officiel Santé Publique France."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Simulation
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("🔮 Simulation — prédire le Nutri-Score d'un produit")
    st.markdown(
        "Saisissez les valeurs nutritionnelles **pour 100 g** du produit, "
        "sélectionnez un modèle, puis cliquez sur **Prédire**."
    )

    # ── Saisie des nutriments ─────────────────────────────────────────────────
    st.subheader("Composition nutritionnelle (pour 100 g)")

    col_a, col_b = st.columns(2)
    feature_values: dict[str, float] = {}

    features_left  = NUM_FEATURES[:4]
    features_right = NUM_FEATURES[4:]

    with col_a:
        for feat in features_left:
            lo, hi = FEATURE_BOUNDS[feat]
            feature_values[feat] = st.number_input(
                label=FEATURE_LABELS[feat],
                min_value=float(lo),
                max_value=float(hi),
                value=float(FEATURE_DEFAULTS[feat]),
                step=0.1 if feat == "salt_100g" else 1.0,
                key=f"input_{feat}",
            )

    with col_b:
        for feat in features_right:
            lo, hi = FEATURE_BOUNDS[feat]
            feature_values[feat] = st.number_input(
                label=FEATURE_LABELS[feat],
                min_value=float(lo),
                max_value=float(hi),
                value=float(FEATURE_DEFAULTS[feat]),
                step=0.1 if feat == "salt_100g" else 1.0,
                key=f"input_{feat}",
            )

    st.divider()

    # ── Sélection du modèle ───────────────────────────────────────────────────
    st.subheader("Modèle de prédiction")
    model_choice = st.selectbox(
        "Choisir un modèle",
        options=list(SKLEARN_MODELS.keys()),
        index=0,   # Random Forest by default
    )
    model_key = SKLEARN_MODELS[model_choice]

    # ── Bouton de prédiction ──────────────────────────────────────────────────
    if st.button("🔍 Prédire le Nutri-Score", type="primary", use_container_width=True):
        with st.spinner("Prédiction en cours …"):
            pipe = _load_sklearn_pipeline(model_key)

            if pipe is None:
                st.error(
                    f"Modèle introuvable : `models/{model_key}.joblib`.\n\n"
                    "Lancez d'abord `python -m src.train_models` depuis le dossier `project/`."
                )
            else:
                # Build input DataFrame with exact column names expected by the pipeline
                df_input = pd.DataFrame([feature_values], columns=NUM_FEATURES)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pred_int = int(pipe.predict(df_input)[0])
                    proba = pipe.predict_proba(df_input)[0]   # shape (5,)

                label_map = _load_label_map()
                int_to_label = label_map["int_to_label"]
                pred_letter = int_to_label.get(pred_int, str(pred_int))
                pred_color  = GRADE_COLORS.get(pred_letter, "#333")

                # ── Résultat ─────────────────────────────────────────────────
                st.markdown("---")
                st.subheader("Résultat de la prédiction")

                # Big colored grade badge
                st.markdown(
                    f"""
                    <div style="
                        display:inline-block;
                        background:{pred_color};
                        color:white;
                        font-size:5rem;
                        font-weight:bold;
                        padding:0.2em 0.6em;
                        border-radius:12px;
                        letter-spacing:0.05em;
                        text-align:center;
                        min-width:120px;
                    ">
                        {pred_letter.upper()}
                    </div>
                    <p style="margin-top:0.5em; font-size:1.1rem; color:#555;">
                        Nutri-Score prédit par <strong>{model_choice}</strong>
                    </p>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("")

                # ── Probabilités ─────────────────────────────────────────────
                grades = ["a", "b", "c", "d", "e"]
                colors_list = [GRADE_COLORS[g] for g in grades]

                df_proba = pd.DataFrame({
                    "Nutri-Score": [g.upper() for g in grades],
                    "Probabilité": proba,
                    "_color": colors_list,
                })

                fig = px.bar(
                    df_proba,
                    x="Nutri-Score",
                    y="Probabilité",
                    color="Nutri-Score",
                    color_discrete_map={g.upper(): c for g, c in GRADE_COLORS.items()},
                    text=df_proba["Probabilité"].map("{:.1%}".format),
                    title=f"Probabilités par classe — {model_choice}",
                    range_y=[0, 1],
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(
                    showlegend=False,
                    plot_bgcolor="white",
                    yaxis_tickformat=".0%",
                    height=380,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Confidence note
                conf = proba[pred_int]
                if conf >= 0.80:
                    st.success(f"Confiance élevée : **{conf:.1%}** pour le grade **{pred_letter.upper()}**.")
                elif conf >= 0.50:
                    st.warning(f"Confiance modérée : **{conf:.1%}** — le produit est proche d'un autre grade.")
                else:
                    st.error(f"Confiance faible : **{conf:.1%}** — revoir la formulation du produit.")
