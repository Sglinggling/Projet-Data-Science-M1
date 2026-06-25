"""
Data loading, cleaning, encoding, and train/test splitting.

Pipeline design
---------------
- build_preprocessor() returns a ColumnTransformer that is NOT fitted here.
  It must be placed INSIDE each model pipeline so that median imputation is
  computed on training data only (no leakage into the test fold).
- load_and_clean() handles everything upstream of sklearn: raw I/O, physical
  bound enforcement, and target encoding.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    GRADE_ORDER,
    INT_TO_LABEL,
    LABEL_TO_INT,
    MODELS_DIR,
    NUM_FEATURES,
    PHYSICAL_BOUNDS,
    RANDOM_STATE,
    TARGET,
    USECOLS,
)
from src.utils import load_off


# ── 1. Load & clean ────────────────────────────────────────────────────────────

def load_and_clean(
    path,
    nrows: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Read the raw OFF dump, apply quality filters, and return (X, y).

    Steps
    -----
    1. Load only USECOLS from the TSV.
    2. Keep rows where the target is a valid Nutri-Score grade {a,b,c,d,e}.
    3. Enforce physical bounds per column: values outside [lo, hi] → NaN.
       Rows are NOT dropped — downstream imputation handles the NaNs.
    4. Encode the target as integers 0–4 (a=0 … e=4).

    Returns
    -------
    X : pd.DataFrame  shape (n, 8)  — raw numerics, may contain NaN
    y : pd.Series     shape (n,)    — integer labels 0–4
    """
    df = load_off(path, usecols=USECOLS, nrows=nrows)

    # ── target cleaning ────────────────────────────────────────────────────────
    df[TARGET] = df[TARGET].str.strip().str.lower()
    df = df[df[TARGET].isin(GRADE_ORDER)].copy()

    # ── physical bounds: out-of-range values become NaN ───────────────────────
    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].where(df[col].between(lo, hi, inclusive="both"))

    # ── target encoding ────────────────────────────────────────────────────────
    y = df[TARGET].map(LABEL_TO_INT).astype(np.int8)
    X = df[NUM_FEATURES].copy()

    return X, y


# ── 2. Preprocessor (unfitted) ─────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    Return an unfitted ColumnTransformer.

    Numeric pipeline
    ----------------
    SimpleImputer(strategy="median") → StandardScaler

    The median is computed per training fold only — call this inside a
    sklearn Pipeline to avoid leakage.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUM_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ── 3. Train / test split ──────────────────────────────────────────────────────

def get_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified split.  The preprocessor is intentionally NOT applied here;
    fit it inside each model's Pipeline on X_train only.
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )


# ── 4. Persist label mapping ───────────────────────────────────────────────────

def save_label_mapping() -> None:
    """Dump INT_TO_LABEL and LABEL_TO_INT to models/ for the dashboard."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"int_to_label": INT_TO_LABEL, "label_to_int": LABEL_TO_INT},
        MODELS_DIR / "label_mapping.joblib",
    )


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.config import RAW_DIR, SAMPLE_SIZE

    RAW_PATH = RAW_DIR / "en.openfoodfacts.org.products.csv"

    print("Loading and cleaning data …")
    X, y = load_and_clean(RAW_PATH, nrows=SAMPLE_SIZE)

    print(f"\nX shape : {X.shape}")
    print(f"y shape : {y.shape}")

    print("\n% NaN per feature (before imputation):")
    nan_pct = X.isna().mean().mul(100).round(2).rename("NaN %")
    print(nan_pct.to_string())

    print(f"\nClass distribution (counts):")
    class_counts = y.value_counts().sort_index()
    for int_label, count in class_counts.items():
        grade = INT_TO_LABEL[int_label]
        print(f"  {grade} ({int_label}) : {count:>6}  ({count/len(y)*100:.1f}%)")

    X_train, X_test, y_train, y_test = get_train_test(X, y)
    print(f"\nTrain : {X_train.shape}  |  Test : {X_test.shape}")

    save_label_mapping()
    print(f"\nLabel mapping saved to {MODELS_DIR / 'label_mapping.joblib'}")
