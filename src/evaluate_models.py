import gc
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

from src.config import (
    MODELS_DIR,
    RANDOM_STATE,
    RAW_DIR,
    ROOT,
    SAMPLE_SIZE,
)
from src.preprocessing import (
    build_preprocessor,
    get_train_test,
    load_and_clean,
)

FIGURES_DIR = ROOT / "notebooks" / "figures"
RAW_PATH = RAW_DIR / "en.openfoodfacts.org.products.csv"
CLASS_LABELS = ["a", "b", "c", "d", "e"]
SVM_SUBSAMPLE = 15_000
CV_FOLDS = 5

DISPLAY = {
    "logreg":            "Logistic Regression",
    "random_forest":     "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "svm":               "SVM RBF",
    "mlp":               "MLP (Keras)",
}


def _plot_confmat(y_true, y_pred, model_key: str) -> None:
    """Génère et sauvegarde la matrice de confusion normalisée par ligne (% par vraie classe)."""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_pct,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
        ax=ax,
        vmin=0,
        vmax=100,
        cbar_kws={"label": "% of true class"},
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(
        f"Confusion matrix — {DISPLAY.get(model_key, model_key)}\n"
        "(row-normalised, %)"
    )
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"confmat_{model_key}.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  → confmat saved: notebooks/figures/confmat_{model_key}.png", flush=True)


def _evaluate_one(key: str, y_true, y_pred) -> dict:
    """Affiche le rapport de classification, trace la confusion matrix et renvoie les métriques scalaires."""
    print(f"\n{'─' * 60}", flush=True)
    print(f"  {DISPLAY[key]}", flush=True)
    print(f"{'─' * 60}", flush=True)
    print(
        classification_report(y_true, y_pred, target_names=CLASS_LABELS, digits=4),
        flush=True,
    )
    _plot_confmat(y_true, y_pred, key)

    return {
        "model":           DISPLAY[key],
        "model_key":       key,
        "accuracy":        accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro":        f1_score(y_true, y_pred, average="macro"),
    }


def _plot_comparison(df: pd.DataFrame) -> None:
    """Barplot groupé comparant accuracy, précision, recall et F1-macro pour tous les modèles."""
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    labels = {
        "accuracy":        "Accuracy",
        "precision_macro": "Precision (macro)",
        "recall_macro":    "Recall (macro)",
        "f1_macro":        "F1 (macro)",
    }
    df_plot = df[["model"] + metrics].melt(
        id_vars="model", var_name="metric", value_name="score"
    )
    df_plot["metric"] = df_plot["metric"].map(labels)

    fig, ax = plt.subplots(figsize=(13, 5))
    sns.barplot(
        data=df_plot,
        x="model",
        y="score",
        hue="metric",
        palette=sns.color_palette("Set2", len(metrics)),
        ax=ax,
        width=0.7,
    )
    ax.set_ylim(0.6, 1.02)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_title("Model comparison — test set (sorted by F1-macro ↓)")
    ax.tick_params(axis="x", labelsize=9)
    ax.legend(title="Metric", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "model_comparison.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"\nComparison barplot saved → notebooks/figures/model_comparison.png", flush=True)


def _cv_rf_gb(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Validation croisée stratifiée (5 folds) pour Random Forest et Gradient Boosting."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    configs = [
        (
            "Random Forest (100 trees)",
            Pipeline([
                ("pre", build_preprocessor()),
                ("clf", RandomForestClassifier(
                    n_estimators=100, class_weight="balanced",
                    n_jobs=1, random_state=RANDOM_STATE,
                )),
            ]),
        ),
        (
            "Gradient Boosting",
            Pipeline([
                ("pre", build_preprocessor()),
                ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
            ]),
        ),
    ]
    for label, pipe in configs:
        print(f"  CV: {label} …", flush=True)
        scores = cross_val_score(
            pipe, X_train, y_train,
            cv=skf, scoring="f1_macro", n_jobs=1,
        )
        print(f"  {label:<42}  {scores.mean():.4f} ± {scores.std():.4f}", flush=True)


def _cv_svm_subsampled(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """CV manuelle pour le SVM : chaque fold est sous-échantillonné à 15 k lignes pour rester raisonnable."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    X_r = X_train.reset_index(drop=True)
    y_r = y_train.reset_index(drop=True)
    scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_r, y_r), 1):
        X_tr, y_tr = X_r.iloc[tr_idx], y_r.iloc[tr_idx]
        X_val, y_val = X_r.iloc[val_idx], y_r.iloc[val_idx]
        if len(X_tr) > SVM_SUBSAMPLE:
            X_tr, _, y_tr, _ = train_test_split(
                X_tr, y_tr,
                train_size=SVM_SUBSAMPLE,
                stratify=y_tr,
                random_state=RANDOM_STATE,
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe = Pipeline([
                ("pre", build_preprocessor()),
                ("clf", SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE)),
            ])
        pipe.fit(X_tr, y_tr)
        s = f1_score(y_val, pipe.predict(X_val), average="macro")
        scores.append(s)
        print(f"    SVM fold {fold}/{CV_FOLDS}  f1={s:.4f}", flush=True)
        del pipe; gc.collect()

    arr = np.array(scores)
    print(f"  {'SVM RBF (subsample=15k)':<42}  {arr.mean():.4f} ± {arr.std():.4f}", flush=True)


def _error_analysis_rf(y_true, y_pred_rf: np.ndarray) -> None:
    """Identifie les deux paires de grades les plus souvent confondues par le Random Forest."""
    cm = confusion_matrix(y_true, y_pred_rf)
    np.fill_diagonal(cm, 0)

    seen: list[tuple] = []
    for idx in np.argsort(cm.ravel())[::-1]:
        r, c = divmod(idx, len(CLASS_LABELS))
        if r == c:
            continue
        seen.append((r, c, cm[r, c]))
        if len(seen) == 2:
            break

    print(f"\n{'─' * 60}", flush=True)
    print("  Error analysis — Random Forest (most confused pairs)", flush=True)
    print(f"{'─' * 60}", flush=True)
    for true_i, pred_i, count in seen:
        tl, pl = CLASS_LABELS[true_i], CLASS_LABELS[pred_i]
        pct = count / np.sum(y_true == true_i) * 100
        print(f"  True={tl} → Predicted={pl} : {count} errors ({pct:.1f}% of true-{tl})", flush=True)
    print(
        "\n  Interpretation: adjacent Nutri-Score grades share similar nutritional\n"
        "  profiles — their separation is a continuous threshold, so borderline\n"
        "  products flip across it with small feature variations.",
        flush=True,
    )


def main() -> pd.DataFrame:
    print("=" * 60, flush=True)
    print("Loading data …", flush=True)
    X, y = load_and_clean(RAW_PATH, nrows=SAMPLE_SIZE)
    X_train, X_test, y_train, y_test = get_train_test(X, y)
    print(f"Train : {X_train.shape}  |  Test : {X_test.shape}", flush=True)
    print("=" * 60, flush=True)

    rows: list[dict] = []
    y_pred_rf: np.ndarray | None = None

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Modèles sklearn évalués un par un pour libérer la RAM entre chaque
    for key in ["logreg", "random_forest", "gradient_boosting", "svm"]:
        print(f"\n[sklearn] Loading {DISPLAY[key]} …", flush=True)
        try:
            pipe = joblib.load(MODELS_DIR / f"{key}.joblib")
            y_pred = pipe.predict(X_test)
            if key == "random_forest":
                y_pred_rf = y_pred.copy()
            rows.append(_evaluate_one(key, y_test, y_pred))
        except Exception as exc:
            print(f"  ERROR evaluating {key}: {exc}", flush=True)
        finally:
            del pipe, y_pred
            gc.collect()
            print(f"  [freed {key} from RAM]", flush=True)

    # MLP : on extrait les poids Dense puis on détruit le modèle Keras avant l'inférence.
    # mlp.predict() déclenche le moteur TF qui est tué par le sandbox macOS (SIGURG / exit 144)
    # quand de la mémoire sklearn est encore mappée. Solution : forward-pass en numpy pur.
    # Le Dropout est inactif à l'inférence donc le calcul est simplement relu(X@W+b) par couche.
    print(f"\n[Keras] Loading MLP …", flush=True)
    try:
        pre = joblib.load(MODELS_DIR / "preprocessor.joblib")
        print("  Preprocessor loaded.", flush=True)

        if not _TF_AVAILABLE:
            raise RuntimeError("TensorFlow not available.")
        mlp = tf.keras.models.load_model(MODELS_DIR / "mlp.keras")
        print("  MLP model loaded.", flush=True)

        # On ne garde que les couches Dense (Input et Dropout sont ignorés)
        dense_weights = [
            layer.get_weights()
            for layer in mlp.layers
            if isinstance(layer, tf.keras.layers.Dense)
        ]
        print(f"  Extracted weights from {len(dense_weights)} Dense layers.", flush=True)
        del mlp
        gc.collect()
        print("  TF model deleted — running numpy forward pass.", flush=True)

        X_te = pre.transform(X_test)
        print("  X_test transformed.", flush=True)
        del pre
        gc.collect()

        # Forward pass numpy — le Dropout est désactivé à l'inférence
        def _relu(z: np.ndarray) -> np.ndarray:
            return np.maximum(0.0, z)

        def _softmax(z: np.ndarray) -> np.ndarray:
            e = np.exp(z - z.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)

        h = X_te.astype(np.float32)
        for i, (W, b) in enumerate(dense_weights):
            h = h @ W + b
            if i < len(dense_weights) - 1:
                h = _relu(h)
        probs = _softmax(h)
        y_pred_mlp = probs.argmax(axis=1)
        print("  Numpy inference done.", flush=True)

        rows.append(_evaluate_one("mlp", y_test, y_pred_mlp))
        del X_te, dense_weights, h, probs, y_pred_mlp
        gc.collect()
        print("  [freed MLP from RAM]", flush=True)
    except Exception as exc:
        print(f"  ERROR evaluating MLP: {exc}", flush=True)
        import traceback; traceback.print_exc()

    df = (
        pd.DataFrame(rows)
        .sort_values("f1_macro", ascending=False)
        .reset_index(drop=True)
    )
    df_out = df.drop(columns="model_key")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(MODELS_DIR / "comparison_results.csv", index=False)
    print(f"\nComparison CSV saved → models/comparison_results.csv", flush=True)

    _plot_comparison(df)

    print(f"\n{'=' * 60}", flush=True)
    print("FINAL COMPARISON — test set (sorted by F1-macro ↓)", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(df_out.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"CROSS-VALIDATION ({CV_FOLDS} folds, StratifiedKFold) — f1_macro", flush=True)
    print(f"  (RF uses 100 trees for speed; SVM subsampled to 15k/fold)", flush=True)
    print(f"{'=' * 60}", flush=True)
    _cv_rf_gb(X_train, y_train)
    _cv_svm_subsampled(X_train, y_train)

    if y_pred_rf is not None:
        _error_analysis_rf(y_test, y_pred_rf)
    else:
        print("\n  [RF y_pred not available — skipping error analysis]", flush=True)

    return df


if __name__ == "__main__":
    main()
