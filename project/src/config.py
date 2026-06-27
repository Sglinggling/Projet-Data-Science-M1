from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

# ── Experiment ─────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
SAMPLE_SIZE = 80_000
# Real column name in the OFF dump (nutrition_grade_fr = Nutri-Score A–E)
TARGET = "nutrition_grade_fr"

# ── Numeric features (100 g basis) ────────────────────────────────────────────
# sodium_100g dropped: Spearman r=1.00 with salt_100g (salt = sodium × 2.54).
# pnns_groups_1 dropped: 95.9 % NaN in the sample — unusable as a feature.
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

# ── Notebook-facing aliases (01_eda.ipynb imports these names) ────────────────
# NUTRITIONAL_FEATURES mirrors NUM_FEATURES (post-selection; sodium already dropped).
# CATEGORICAL_FEATURES is empty — pnns_groups_1 was excluded (95.9 % NaN, see above).
NUTRITIONAL_FEATURES: list[str] = NUM_FEATURES
CATEGORICAL_FEATURES: list[str] = []

# ── Columns to load when reading the raw dump ──────────────────────────────────
USECOLS = [TARGET] + NUM_FEATURES

# ── Physical bounds for outlier clamping ──────────────────────────────────────
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

# ── Label encoding ─────────────────────────────────────────────────────────────
GRADE_ORDER = ["a", "b", "c", "d", "e"]
LABEL_TO_INT = {g: i for i, g in enumerate(GRADE_ORDER)}   # a→0 … e→4
INT_TO_LABEL = {i: g for g, i in LABEL_TO_INT.items()}

# ── Data-leakage blacklist ─────────────────────────────────────────────────────
# Any column whose name contains these substrings must NEVER be used as a feature.
BLACKLIST_PATTERNS = [
    "nutriscore",
    "nutrition-score",
]
