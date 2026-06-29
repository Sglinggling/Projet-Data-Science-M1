import sys
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Ajout du répertoire project/ au path pour que `src` soit importable peu importe le cwd
_HERE = Path(__file__).resolve().parent   # src/
_ROOT = _HERE.parent                      # project/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import GRADE_ORDER, INT_TO_LABEL, MODELS_DIR, NUM_FEATURES

# État global chargé au démarrage et partagé entre les requêtes
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le pipeline RF et le label mapping au démarrage ; libère la mémoire à l'arrêt."""
    rf_path  = MODELS_DIR / "random_forest.joblib"
    lm_path  = MODELS_DIR / "label_mapping.joblib"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _state["pipeline"] = joblib.load(rf_path) if rf_path.exists() else None
        _state["label_map"] = (
            joblib.load(lm_path) if lm_path.exists()
            else {"int_to_label": INT_TO_LABEL}
        )
    _state["loaded"] = _state["pipeline"] is not None
    yield
    _state.clear()


app = FastAPI(
    title="Nutri-Score Classifier API",
    description="Predicts the Nutri-Score (A–E) of a food product from 8 nutritional values.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


class NutrientInput(BaseModel):
    """Valeurs nutritionnelles pour 100 g.

    Les noms JSON utilisent des underscores (contrainte Pydantic) ;
    to_dataframe() les reconvertit avec le tiret attendu par le pipeline (saturated-fat_100g).
    """
    energy_100g:        float = Field(..., ge=0, le=3700, description="Énergie (kJ/100g)")
    fat_100g:           float = Field(..., ge=0, le=100,  description="Matières grasses (g/100g)")
    saturated_fat_100g: float = Field(..., ge=0, le=100,  description="Graisses saturées (g/100g)")
    carbohydrates_100g: float = Field(..., ge=0, le=100,  description="Glucides (g/100g)")
    sugars_100g:        float = Field(..., ge=0, le=100,  description="Sucres (g/100g)")
    proteins_100g:      float = Field(..., ge=0, le=100,  description="Protéines (g/100g)")
    salt_100g:          float = Field(..., ge=0, le=100,  description="Sel (g/100g)")
    fiber_100g:         float = Field(..., ge=0, le=100,  description="Fibres (g/100g)")

    def to_dataframe(self) -> pd.DataFrame:
        """Construit un DataFrame d'une ligne avec les noms de colonnes exacts attendus par le pipeline."""
        row = {
            "energy_100g":        self.energy_100g,
            "fat_100g":           self.fat_100g,
            "saturated-fat_100g": self.saturated_fat_100g,  # tiret requis par le pipeline
            "carbohydrates_100g": self.carbohydrates_100g,
            "sugars_100g":        self.sugars_100g,
            "proteins_100g":      self.proteins_100g,
            "salt_100g":          self.salt_100g,
            "fiber_100g":         self.fiber_100g,
        }
        return pd.DataFrame([row], columns=NUM_FEATURES)


class PredictionResponse(BaseModel):
    nutriscore:    str
    confidence:    float
    probabilities: dict[str, float]


@app.get("/health", tags=["System"])
def health():
    """Vérifie que l'API est en vie et que le modèle est bien chargé."""
    return {
        "status": "ok",
        "model":  "random_forest",
        "model_loaded": _state.get("loaded", False),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(payload: NutrientInput):
    """Prédit le Nutri-Score (a–e), la confiance associée et les probabilités pour les 5 classes."""
    if not _state.get("loaded"):
        return JSONResponse(
            status_code=503,
            content={"error": "Model not loaded", "detail": "random_forest.joblib not found."},
        )

    df_input = payload.to_dataframe()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred_int = int(_state["pipeline"].predict(df_input)[0])
        proba    = _state["pipeline"].predict_proba(df_input)[0]

    int_to_label = _state["label_map"]["int_to_label"]
    pred_letter  = int_to_label.get(pred_int, str(pred_int))
    confidence   = float(np.round(proba[pred_int], 4))
    probabilities = {
        int_to_label.get(i, str(i)): float(np.round(p, 4))
        for i, p in enumerate(proba)
    }

    return PredictionResponse(
        nutriscore=pred_letter,
        confidence=confidence,
        probabilities=probabilities,
    )


@app.get("/model-info", tags=["System"])
def model_info():
    """Renvoie les métadonnées du modèle chargé (F1-macro, features, classes)."""
    f1_macro = None
    cmp_path = MODELS_DIR / "comparison_results.csv"
    if cmp_path.exists():
        import csv
        with open(cmp_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("model") == "Random Forest":
                    f1_macro = round(float(row["f1_macro"]), 4)
                    break

    return {
        "model":    "Random Forest",
        "f1_macro": f1_macro,
        "features": NUM_FEATURES,
        "n_classes": 5,
        "grades":   GRADE_ORDER,
    }
