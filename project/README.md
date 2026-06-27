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

## Lancer l'application complète (architecture Front / API / Modèle)

L'onglet **Simulation** du dashboard appelle l'API FastAPI pour les prédictions.
Il faut deux terminaux lancés depuis `project/` :

**Terminal 1 — API (modèle)**
```bash
uvicorn src.api:app --port 8000
```

**Terminal 2 — Dashboard (front)**
```bash
streamlit run dashboard/app.py
```

Le dashboard détecte automatiquement si l'API répond (indicateur vert).
Si l'API est absente, il bascule en **mode local de secours** et charge le modèle
directement — la simulation fonctionne dans les deux cas.

L'URL de l'API est configurable via la variable d'environnement `API_URL`
(défaut : `http://localhost:8000`).

## Run the dashboard seul (sans API)

```bash
streamlit run dashboard/app.py
```

## API REST

L'API expose les prédictions du meilleur modèle (Random Forest) via HTTP.

### Lancement

```bash
# depuis project/
uvicorn src.api:app --reload --port 8000
```

Documentation Swagger interactive : **http://localhost:8000/docs**

### Endpoints

| Méthode | Route | Description |
|---------|-------|-------------|
| `GET`   | `/health`     | Liveness check + statut du modèle |
| `POST`  | `/predict`    | Prédiction Nutri-Score + probabilités |
| `GET`   | `/model-info` | Métadonnées du modèle chargé |

### Exemple — POST /predict

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "energy_100g": 2000,
    "fat_100g": 20,
    "saturated_fat_100g": 15,
    "carbohydrates_100g": 50,
    "sugars_100g": 40,
    "proteins_100g": 5,
    "salt_100g": 1.2,
    "fiber_100g": 2
  }' | python -m json.tool
```

Réponse attendue :

```json
{
  "nutriscore": "e",
  "confidence": 0.97,
  "probabilities": {"a": 0.0, "b": 0.0, "c": 0.01, "d": 0.02, "e": 0.97}
}
```

> **Note** : le champ JSON `saturated_fat_100g` (underscore) est remappé en interne
> vers la colonne `saturated-fat_100g` (tiret) attendue par le pipeline sklearn.

## Key design choices

- **No data leakage**: any column containing `nutriscore` or `nutrition-score` is
  blacklisted in `src/config.py` and must never appear as a feature.
- **RAM-efficient loading**: `load_off()` in `utils.py` passes `usecols` to the CSV
  parser so only the needed columns are decoded from the 9 GB file.
- **Reproducibility**: `RANDOM_STATE = 42` and `SAMPLE_SIZE = 80_000` are defined
  centrally in `config.py`.
