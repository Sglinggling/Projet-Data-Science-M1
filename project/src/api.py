"""
FastAPI REST service — Nutri-Score classifier.

Run from the project/ directory:
    uvicorn src.api:app --reload --port 8000

Swagger UI: http://localhost:8000/docs
"""

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

# ── path bootstrap (works regardless of cwd) ─────────────────────────────────
_HERE = Path(__file__).resolve().parent   # src/
_ROOT = _HERE.parent                      # project/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import GRADE_ORDER, INT_TO_LABEL, MODELS_DIR, NUM_FEATURES

# ── Module-level state (populated at startup) ─────────────────────────────────
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load RF pipeline + label mapping once at startup; release on shutdown."""
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


# ── Global exception handler ───────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


# ── Schemas ───────────────────────────────────────────────────────────────────

class NutrientInput(BaseModel):
    """
    Nutritional values per 100 g.  All fields ≥ 0.
    Note: JSON uses Python-safe names (underscores); the model receives the
    exact column names from config.NUM_FEATURES (including saturated-fat_100g).
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
        """Build a 1-row DataFrame with the exact column names expected by the pipeline."""
        row = {
            "energy_100g":        self.energy_100g,
            "fat_100g":           self.fat_100g,
            "saturated-fat_100g": self.saturated_fat_100g,  # hyphen — model expects this
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


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """Liveness check — returns model load status."""
    return {
        "status": "ok",
        "model":  "random_forest",
        "model_loaded": _state.get("loaded", False),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(payload: NutrientInput):
    """
    Predict the Nutri-Score for a food product.

    Returns the predicted grade (a–e), the model's confidence for that grade,
    and the full probability vector across all 5 classes.
    """
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
    """Return metadata about the loaded model and available features."""
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
