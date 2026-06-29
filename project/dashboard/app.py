
import sys
from pathlib import Path

# Ajout de project/ au path pour que `src` soit importable peu importe le répertoire de lancement
_HERE = Path(__file__).resolve().parent          # project/dashboard/
_ROOT = _HERE.parent                             # project/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.metrics import confusion_matrix

from src.config import GRADE_ORDER, INT_TO_LABEL, LABEL_TO_INT, MODELS_DIR, NUM_FEATURES, RAW_DIR

# Paths
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
    "energy_100g":        "Énergie (kJ/100g)",
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

N_PRODUCTS = 61_907
N_CLASSES   = 5

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Paramètres Plotly communs pour le thème sombre
_PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e8eaf0"),
)


def _load_comparison() -> pd.DataFrame | None:
    p = MODELS_DIR / "comparison_results.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


@st.cache_data(show_spinner="Chargement de l'échantillon de données …")
def _load_sample(nrows: int = 15_000) -> pd.DataFrame:
    """Charge ~10 k lignes nettoyées pour les graphiques interactifs.

    On demande 15 k lignes brutes pour compenser celles supprimées lors du nettoyage.
    """
    from src.preprocessing import load_and_clean
    raw_path = RAW_DIR / "en.openfoodfacts.org.products.csv"
    X, y = load_and_clean(raw_path, nrows=nrows)
    df = X.copy()
    df["grade"] = y.map(INT_TO_LABEL).str.upper()
    df = df.dropna(subset=["grade"])
    return df.reset_index(drop=True)


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
        return {"int_to_label": INT_TO_LABEL, "label_to_int": LABEL_TO_INT}
    return joblib.load(p)


@st.cache_data(ttl=15, show_spinner=False)
def _api_health() -> bool:
    """Ping /health ; renvoie True uniquement si l'API répond et que le modèle est chargé."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.ok and r.json().get("model_loaded", False)
    except Exception:
        return False


def _render_prediction_result(
    pred_letter: str,
    confidence: float,
    probabilities: dict,
    model_label: str,
) -> None:
    """Affiche la carte de résultat (badge Nutri-Score + label de confiance) et le barplot des probabilités."""
    pred_color = GRADE_COLORS.get(pred_letter, "#333")

    if confidence >= 0.80:
        conf_label = f"Confiance élevée — {confidence:.1%}"
        conf_color = "#66bb6a"
    elif confidence >= 0.50:
        conf_label = f"Confiance modérée — {confidence:.1%}"
        conf_color = "#ffa726"
    else:
        conf_label = f"Confiance faible — {confidence:.1%}"
        conf_color = "#ef5350"

    st.markdown("---")
    st.subheader("Résultat de la prédiction")

    _res_col, _spacer = st.columns([1, 2])
    with _res_col:
        st.markdown(
            f"""
            <div class="result-card">
                <div style="
                    display:inline-block;
                    background:{pred_color};
                    color:white;
                    font-size:5rem;
                    font-weight:800;
                    padding:0.15em 0.55em;
                    border-radius:10px;
                    letter-spacing:0.04em;
                    text-align:center;
                    min-width:120px;
                    line-height:1.15;
                ">
                    {pred_letter.upper()}
                </div>
                <p style="margin:0.8rem 0 0.2rem; font-size:0.85rem;
                          color:#9e9e9e; text-transform:uppercase;
                          letter-spacing:0.08em;">
                    Prédit par {model_label}
                </p>
                <p style="margin:0; font-size:1rem; font-weight:600;
                          color:{conf_color};">
                    {conf_label}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    grades = ["a", "b", "c", "d", "e"]
    df_proba = pd.DataFrame({
        "Nutri-Score": [g.upper() for g in grades],
        "Probabilité": [float(probabilities.get(g, 0.0)) for g in grades],
    })
    fig_proba = px.bar(
        df_proba,
        x="Nutri-Score", y="Probabilité",
        color="Nutri-Score",
        color_discrete_map={g.upper(): c for g, c in GRADE_COLORS.items()},
        text=df_proba["Probabilité"].map("{:.1%}".format),
        title=f"Probabilités par classe — {model_label}",
        range_y=[0, 1],
    )
    fig_proba.update_traces(textposition="outside")
    fig_proba.update_layout(
        showlegend=False,
        yaxis_tickformat=".0%",
        height=380,
        **_PLOT_LAYOUT,
    )
    st.plotly_chart(fig_proba, use_container_width=True)


st.set_page_config(
    page_title="Nutri-Score Classifier",
    page_icon="🥗",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stTabs"] button {
        font-weight: 500;
        letter-spacing: 0.02em;
        padding: 0.5rem 1.2rem;
    }
    .block-container { padding-top: 2rem; }

    .kpi-card {
        background: #1a1f2e;
        border: 1px solid #2E7D32;
        border-radius: 10px;
        padding: 1.2rem 1rem;
        text-align: center;
        height: 100%;
    }
    .kpi-label {
        margin: 0;
        font-size: 0.72rem;
        color: #9e9e9e;
        text-transform: uppercase;
        letter-spacing: 0.09em;
    }
    .kpi-value {
        margin: 0.35rem 0 0;
        font-size: 1.75rem;
        font-weight: 700;
        color: #e8eaf0;
        line-height: 1.1;
    }
    .kpi-delta {
        margin: 0.2rem 0 0;
        font-size: 0.82rem;
        color: #66bb6a;
    }

    .result-card {
        background: #1a1f2e;
        border: 1px solid #333;
        border-radius: 14px;
        padding: 2rem 2.5rem;
        display: inline-block;
        min-width: 280px;
    }

    .app-footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #2a2a3a;
        text-align: center;
        font-size: 0.78rem;
        color: #6b7280;
        letter-spacing: 0.04em;
    }

    /* Centre les images SHAP à largeur fixe */
    .shap-img { display: block; margin: 0 auto; max-width: 700px; }
    </style>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Vue générale",
    "Analyse des données",
    "Comparaison des modèles",
    "Simulation",
])



# TAB 1 — Vue générale

with tab1:
    st.title("Nutri-Score Classifier — Open Food Facts")
    st.markdown(
        """
        > Un fabricant qui conçoit un nouveau produit alimentaire
        > doit anticiper son Nutri-Score **avant** la mise en marché.
        > Ce tableau de bord prédit le Nutri-Score (A → E) à partir des **8 valeurs
        > nutritionnelles pour 100 g** déclarées sur l'étiquette, en s'appuyant sur
        > des modèles entraînés sur **61 907 produits Open Food Facts**.
        > Aucun réentraînement : tout est pré-calculé, la prédiction est instantanée.
        """
    )

    st.divider()

    df_cmp = _load_comparison()
    best_model = "Random Forest"
    best_f1 = 0.9587
    if df_cmp is not None and not df_cmp.empty:
        best_row = df_cmp.sort_values("f1_macro", ascending=False).iloc[0]
        best_model = best_row["model"]
        best_f1 = best_row["f1_macro"]

    _n_fmt = f"{N_PRODUCTS:,}".replace(",", " ")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(
        f'<div class="kpi-card">'
        f'<p class="kpi-label">Produits (dataset nettoyé)</p>'
        f'<p class="kpi-value">{_n_fmt}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )
    k2.markdown(
        '<div class="kpi-card">'
        '<p class="kpi-label">Classes Nutri-Score</p>'
        '<p class="kpi-value">5</p>'
        '<p class="kpi-delta">A → E</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    k3.markdown(
        f'<div class="kpi-card">'
        f'<p class="kpi-label">Meilleur modèle</p>'
        f'<p class="kpi-value" style="font-size:1.2rem;">{best_model}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )
    k4.markdown(
        f'<div class="kpi-card">'
        f'<p class="kpi-label">F1-macro (test set)</p>'
        f'<p class="kpi-value">{best_f1:.4f}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    st.subheader("Distribution des Nutri-Scores dans le dataset")
    df_s1 = _load_sample()
    grade_counts = (
        df_s1["grade"]
        .value_counts()
        .reindex([g.upper() for g in GRADE_ORDER])
        .reset_index()
    )
    grade_counts.columns = ["Nutri-Score", "Produits"]
    fig_dist = px.bar(
        grade_counts,
        x="Nutri-Score", y="Produits",
        color="Nutri-Score",
        color_discrete_map={g.upper(): c for g, c in GRADE_COLORS.items()},
        text="Produits",
        title="Répartition des grades (échantillon ~10 k produits)",
    )
    fig_dist.update_traces(textposition="outside")
    fig_dist.update_layout(showlegend=False, height=380, **_PLOT_LAYOUT)
    st.plotly_chart(fig_dist, use_container_width=True)


# TAB 2 — Analyse des données (graphiques Plotly interactifs)

with tab2:
    st.header("Analyse exploratoire des données")
    df_eda = _load_sample()

    st.subheader("Distribution d'un nutriment par grade Nutri-Score")
    st.markdown(
        "Chaque nutriment montre un gradient clair A → E. "
        "Sélectionnez un nutriment pour explorer la distribution par grade."
    )
    sel_feat = st.selectbox(
        "Nutriment à afficher",
        options=NUM_FEATURES,
        format_func=lambda f: FEATURE_LABELS[f],
        key="sel_feat",
    )
    fig_box1 = px.box(
        df_eda.dropna(subset=[sel_feat]),
        x="grade", y=sel_feat,
        color="grade",
        color_discrete_map={g.upper(): c for g, c in GRADE_COLORS.items()},
        category_orders={"grade": [g.upper() for g in GRADE_ORDER]},
        points=False,
        labels={sel_feat: FEATURE_LABELS[sel_feat], "grade": "Nutri-Score"},
        title=f"Distribution de {FEATURE_LABELS[sel_feat]} par grade Nutri-Score",
    )
    fig_box1.update_layout(showlegend=False, height=420, **_PLOT_LAYOUT)
    st.plotly_chart(fig_box1, use_container_width=True)

    st.divider()

    st.subheader("Carte de corrélation des nutriments")
    st.markdown(
        "Sucres et glucides sont fortement corrélés (r > 0.7). "
        "Sel et graisses saturées sont des prédicteurs indépendants."
    )
    corr = df_eda[NUM_FEATURES].corr().round(2)
    short_labels = [FEATURE_LABELS[f].split(" (")[0] for f in NUM_FEATURES]
    fig_corr = px.imshow(
        corr,
        x=short_labels, y=short_labels,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=True,
        title="Corrélation de Pearson — nutriments (100 g)",
        aspect="auto",
    )
    fig_corr.update_layout(height=500, **_PLOT_LAYOUT)
    st.plotly_chart(fig_corr, use_container_width=True)



# TAB 3 — Comparaison des modèles

with tab3:
    st.header("Comparaison des modèles entraînés")

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
            }).highlight_max(subset=["F1-macro"], color="#1a3a1a"),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("Fichier `models/comparison_results.csv` introuvable.")

    st.divider()

    st.subheader("Comparaison des performances — 4 métriques × modèles")
    if df_cmp is not None:
        metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
        metric_labels = {
            "accuracy":        "Accuracy",
            "precision_macro": "Précision macro",
            "recall_macro":    "Recall macro",
            "f1_macro":        "F1-macro",
        }
        df_melt = df_cmp.melt(
            id_vars="model",
            value_vars=metrics,
            var_name="Métrique",
            value_name="Score",
        )
        df_melt["Métrique"] = df_melt["Métrique"].map(metric_labels)

        fig_cmp = px.bar(
            df_melt,
            x="model", y="Score",
            color="Métrique",
            barmode="group",
            text=df_melt["Score"].map("{:.3f}".format),
            title="Métriques par modèle (test set)",
            labels={"model": "Modèle"},
            range_y=[0.85, 1.0],
        )
        fig_cmp.update_traces(textposition="outside", textfont_size=10)
        fig_cmp.update_layout(height=460, legend_title_text="Métrique", **_PLOT_LAYOUT)
        st.plotly_chart(fig_cmp, use_container_width=True)
    else:
        st.info("CSV de comparaison introuvable — tableau ci-dessus uniquement.")

    st.divider()

    st.subheader("Matrice de confusion — Random Forest")
    st.markdown(
        "Normalisée par ligne (rappel par classe). "
        "Calculée sur l'échantillon dashboard (~10 k produits)."
    )
    rf_pipe = _load_sklearn_pipeline("random_forest")
    df_cm_src = _load_sample()
    if rf_pipe is not None and not df_cm_src.empty:
        X_cm = df_cm_src[NUM_FEATURES]
        y_true_cm = df_cm_src["grade"].str.lower().map(LABEL_TO_INT)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred_cm = rf_pipe.predict(X_cm)
        cm_norm = confusion_matrix(y_true_cm, y_pred_cm, normalize="true")
        labels_up = [g.upper() for g in GRADE_ORDER]
        fig_cm = px.imshow(
            np.round(cm_norm, 3),
            x=labels_up, y=labels_up,
            color_continuous_scale="Greens",
            zmin=0, zmax=1,
            text_auto=".2f",
            labels={"x": "Prédit", "y": "Réel"},
            title="Confusion matrix normalisée — Random Forest",
            aspect="auto",
        )
        fig_cm.update_layout(height=440, **_PLOT_LAYOUT)
        st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.warning("Modèle Random Forest ou données introuvables.")

    st.divider()

    st.subheader("Interprétabilité — importance des nutriments")

    col_fi, col_shap = st.columns([1, 1])

    with col_fi:
        st.markdown("**Importance native RF** (Gini, barplot interactif)")
        if rf_pipe is not None:
            importances = rf_pipe.named_steps["clf"].feature_importances_
            df_fi = pd.DataFrame({
                "Nutriment": [FEATURE_LABELS[f].split(" (")[0] for f in NUM_FEATURES],
                "Importance": importances,
            }).sort_values("Importance")
            fig_fi = px.bar(
                df_fi,
                x="Importance", y="Nutriment",
                orientation="h",
                text=df_fi["Importance"].map("{:.3f}".format),
                color="Importance",
                color_continuous_scale="Greens",
                title="Feature importance (Random Forest)",
            )
            fig_fi.update_traces(textposition="outside")
            fig_fi.update_layout(
                showlegend=False,
                coloraxis_showscale=False,
                height=400,
                **_PLOT_LAYOUT,
            )
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.warning("Modèle Random Forest introuvable.")

    with col_shap:
        st.markdown("**SHAP beeswarm** (2 000 échantillons, toutes classes)")
        shap_path = FIGURES_DIR / "shap_summary.png"
        if shap_path.exists():
            # Encodage base64 pour afficher l'image sans serveur de fichiers statiques
            st.markdown(
                f'<div style="text-align:center;">'
                f'<img src="data:image/png;base64,{__import__("base64").b64encode(shap_path.read_bytes()).decode()}"'
                f' style="max-width:680px; width:100%; border-radius:8px;">'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning("Image SHAP manquante.")

    st.info(
        "**Consensus Gini + SHAP** : "
        "**Sel · Graisses saturées · Sucres** sont les nutriments les plus influents — "
        "cohérent avec l'algorithme officiel Santé Publique France."
    )

# TAB 4 — Simulation  (logique inchangée)

with tab4:
    st.header("Simulation — prédire le Nutri-Score d'un produit")

    api_ok = _api_health()
    if api_ok:
        st.markdown(
            '<span style="color:#66bb6a; font-size:0.9rem;">&#9679; API connectée</span> '
            f'<span style="color:#6b7280; font-size:0.85rem;">({API_URL})</span>',
            unsafe_allow_html=True,
        )
    else:
        st.warning(
            "**API non disponible** — démarrez-la avec :\n"
            "```\nuvicorn src.api:app --port 8000\n```\n"
            "La simulation tourne en **mode local de secours**."
        )

    st.markdown(
        "Saisissez les valeurs nutritionnelles **pour 100 g** du produit"
        + (" et cliquez sur **Prédire**." if api_ok
           else " *(mode local — choisissez un modèle ci-dessous)* et cliquez sur **Prédire**.")
    )

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

    st.subheader("Modèle de prédiction")
    if api_ok:
        st.markdown(
            "_Mode API — le modèle **Random Forest** est utilisé "
            "(seul modèle exposé par l'API)._"
        )
        model_choice  = "Random Forest"
        model_key     = SKLEARN_MODELS[model_choice]
    else:
        model_choice = st.selectbox(
            "Choisir un modèle (mode local)",
            options=list(SKLEARN_MODELS.keys()),
            index=0,
        )
        model_key = SKLEARN_MODELS[model_choice]

    if st.button("Prédire le Nutri-Score", type="primary", use_container_width=True):
        with st.spinner("Prédiction en cours …"):

            if api_ok:
                payload = {
                    "energy_100g":        feature_values["energy_100g"],
                    "fat_100g":           feature_values["fat_100g"],
                    "saturated_fat_100g": feature_values["saturated-fat_100g"],
                    "carbohydrates_100g": feature_values["carbohydrates_100g"],
                    "sugars_100g":        feature_values["sugars_100g"],
                    "proteins_100g":      feature_values["proteins_100g"],
                    "salt_100g":          feature_values["salt_100g"],
                    "fiber_100g":         feature_values["fiber_100g"],
                }
                try:
                    resp = requests.post(
                        f"{API_URL}/predict",
                        json=payload,
                        timeout=5,
                    )
                    resp.raise_for_status()
                    data          = resp.json()
                    pred_letter   = data["nutriscore"]
                    confidence    = float(data["confidence"])
                    probabilities = data["probabilities"]
                    _render_prediction_result(
                        pred_letter, confidence, probabilities,
                        model_label="Random Forest (via API)",
                    )
                except requests.exceptions.RequestException as exc:
                    st.error(
                        f"Erreur lors de l'appel API : `{exc}`\n\n"
                        "Vérifiez que l'API est démarrée sur `{API_URL}`."
                    )

            else:
                # Mode local de secours : on charge le pipeline directement sans passer par l'API
                pipe = _load_sklearn_pipeline(model_key)
                if pipe is None:
                    st.error(
                        f"Modèle introuvable : `models/{model_key}.joblib`.\n\n"
                        "Lancez d'abord `python -m src.train_models` depuis `project/`."
                    )
                else:
                    df_input = pd.DataFrame([feature_values], columns=NUM_FEATURES)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pred_int = int(pipe.predict(df_input)[0])
                        proba    = pipe.predict_proba(df_input)[0]

                    label_map    = _load_label_map()
                    pred_letter  = label_map["int_to_label"].get(pred_int, str(pred_int))
                    confidence   = float(proba[pred_int])
                    probabilities = {
                        label_map["int_to_label"].get(i, str(i)): float(p)
                        for i, p in enumerate(proba)
                    }
                    _render_prediction_result(
                        pred_letter, confidence, probabilities,
                        model_label=f"{model_choice} (mode local)",
                    )

st.markdown(
    '<div class="app-footer">'
    'Projet Data Science M1 &nbsp;&middot;&nbsp; Samy HALIT &amp; Ananda CASSINI'
    '</div>',
    unsafe_allow_html=True,
)
