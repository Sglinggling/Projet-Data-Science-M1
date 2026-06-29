# 🥗 NutriPredict — Classification du Nutri-Score

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Interpretability-blue.svg)](https://github.com/slundberg/shap)
[![Jupyter](https://img.shields.io/badge/Notebooks-Jupyter-orange.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)

NutriPredict est une application d'apprentissage automatique de classification multiclasse pour prédire automatiquement la note Nutri-Score (A à E) d'un aliment selon ses apports nutritionnels pour 100g.

---

## 🏗️ Pipeline de Machine Learning

```mermaid
graph TD
    Raw["📊 Données Open Food Facts"]
    EDA["🔍 Analyse Exploratoire (Jupyter)"]
    Prep["⚙️ Nettoyage & Feature Engineering (Scikit-Learn)"]
    Train["🧠 Entraînement Modèles (Random Forest, SVM, MLP)"]
    SHAP["💡 Explicabilité (SHAP Values)"]
    App["💻 Streamlit Dashboard Interactif"]

    Raw --> EDA --> Prep --> Train
    Train --> SHAP
    Train --> App
```

## 📂 Fichiers du Projet
- `notebooks/01_eda.ipynb` : Analyse descriptive des variables et corrélations nutritionnelles.
- `src/config.py` : Fichiers de configuration globale du projet.
- `dashboard/app.py` : Code Streamlit permettant de simuler l'impact des ingrédients sur le score final.

---

## 👥 Binôme & Candidats
- **Samy HALIT** ([@Sglinggling](https://github.com/Sglinggling))
- **Ananda CASSINI** ([@ananda3cassini](https://github.com/ananda3cassini))
