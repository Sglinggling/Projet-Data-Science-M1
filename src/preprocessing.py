
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


def load_and_clean(
    path,
    nrows: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Charge le dump OFF, filtre les grades invalides et encode la cible en entier (0–4).

    Les valeurs hors bornes physiques deviennent NaN — elles ne sont pas supprimées,
    l'imputation s'en occupe en aval.
    """
    df = load_off(path, usecols=USECOLS, nrows=nrows)

    df[TARGET] = df[TARGET].str.strip().str.lower()
    df = df[df[TARGET].isin(GRADE_ORDER)].copy()

    # Valeurs hors plage physique → NaN (les lignes sont conservées)
    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].where(df[col].between(lo, hi, inclusive="both"))

    y = df[TARGET].map(LABEL_TO_INT).astype(np.int8)
    X = df[NUM_FEATURES].copy()

    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Renvoie un ColumnTransformer non entraîné : imputation médiane puis standardisation.

    À instancier à l'intérieur d'un Pipeline sklearn pour éviter toute fuite de données
    (la médiane est calculée uniquement sur le fold d'entraînement).
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


def get_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Découpe stratifiée train/test. Le préprocesseur n'est pas appliqué ici."""
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def save_label_mapping() -> None:
    """Sauvegarde la correspondance entier ↔ grade dans models/ pour le dashboard."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"int_to_label": INT_TO_LABEL, "label_to_int": LABEL_TO_INT},
        MODELS_DIR / "label_mapping.joblib",
    )



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
