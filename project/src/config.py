from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

RANDOM_STATE = 42
SAMPLE_SIZE = 80_000

TARGET = "nutrition_grade_fr"

# Les 8 valeurs nutritionnelles pour 100 g utilisées comme features
NUM_FEATURES = [
    "energy_100g",
    "fat_100g",
    "saturated-fat_100g",
    "carbohydrates_100g",
    "sugars_100g",
    "proteins_100g",
    "salt_100g",
    "fiber_100g",
]

NUTRITIONAL_FEATURES: list[str] = NUM_FEATURES
CATEGORICAL_FEATURES: list[str] = []

USECOLS = [TARGET] + NUM_FEATURES

# Valeurs physiquement impossibles → remplacées par NaN avant imputation
PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "energy_100g":        (0.0, 3700.0),
    "fat_100g":           (0.0,  100.0),
    "saturated-fat_100g": (0.0,  100.0),
    "carbohydrates_100g": (0.0,  100.0),
    "sugars_100g":        (0.0,  100.0),
    "proteins_100g":      (0.0,  100.0),
    "salt_100g":          (0.0,  100.0),
    "fiber_100g":         (0.0,  100.0),
}

GRADE_ORDER = ["a", "b", "c", "d", "e"]
LABEL_TO_INT = {g: i for i, g in enumerate(GRADE_ORDER)}   # a→0 … e→4
INT_TO_LABEL = {i: g for g, i in LABEL_TO_INT.items()}

# Colonnes à exclure pour éviter la fuite de données (elles encodent déjà le grade)
BLACKLIST_PATTERNS = [
    "nutriscore",
    "nutrition-score",
]
