# Nutri-Score Classifier — Open Food Facts

Multiclass classification of Nutri-Score grades (A–E) using the Open Food Facts dataset.
Built as a modular M1 Data Science project.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Download the full dump (~9 GB) into `data/raw/`:

```bash
curl -L "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz" \
  -o data/raw/en.openfoodfacts.org.products.csv.gz
# then decompress
gunzip data/raw/en.openfoodfacts.org.products.csv.gz
```

> The `data/` directory is git-ignored. Only the `.gitkeep` placeholders are tracked.

## Project layout

```
project/
├── data/
│   ├── raw/          # original OFF dump (git-ignored)
│   └── processed/    # cleaned / encoded datasets (git-ignored)
├── notebooks/        # exploratory notebooks
├── src/
│   ├── config.py         # central paths, constants, feature lists
│   ├── utils.py          # I/O helpers (load_off)
│   ├── preprocessing.py  # cleaning & feature engineering
│   ├── train_models.py   # model training
│   ├── evaluate_models.py# metrics & comparison
│   └── explainability.py # SHAP analysis
├── models/           # serialised models (git-ignored binaries)
├── dashboard/
│   └── app.py        # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Run the dashboard

```bash
streamlit run dashboard/app.py
```

## Key design choices

- **No data leakage**: any column containing `nutriscore` or `nutrition-score` is
  blacklisted in `src/config.py` and must never appear as a feature.
- **RAM-efficient loading**: `load_off()` in `utils.py` passes `usecols` to the CSV
  parser so only the needed columns are decoded from the 9 GB file.
- **Reproducibility**: `RANDOM_STATE = 42` and `SAMPLE_SIZE = 80_000` are defined
  centrally in `config.py`.
