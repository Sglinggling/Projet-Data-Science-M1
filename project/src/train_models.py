import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

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
    save_label_mapping,
)

SVM_SUBSAMPLE = 15_000  # le SVM est trop lent sur l'intégralité du train set
FIGURES_DIR = ROOT / "notebooks" / "figures"
RAW_PATH = RAW_DIR / "en.openfoodfacts.org.products.csv"


def _recap(name: str, y_true, y_pred, elapsed: float) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    print(f"\n{'─' * 54}")
    print(f"  {name}")
    print(f"  Accuracy : {acc:.4f}  |  F1-macro : {f1:.4f}  |  {elapsed:.1f}s")
    print(f"{'─' * 54}")
    return {"model": name, "accuracy": acc, "f1_macro": f1, "train_time_s": round(elapsed, 1)}


def train_sklearn(
    name: str,
    clf,
    X_train,
    y_train,
    X_test,
    y_test,
    subsample: int | None = None,
) -> dict:
    """Entraîne clf dans un Pipeline (préprocesseur + clf), sauvegarde le .joblib et renvoie les métriques.

    Si subsample est fourni, l'entraînement porte sur un sous-ensemble stratifié de cette taille ;
    l'évaluation utilise toujours le test set complet.
    """
    pipe = Pipeline([("pre", build_preprocessor()), ("clf", clf)])

    X_fit, y_fit = X_train, y_train
    if subsample and len(X_train) > subsample:
        print(f"  [SVM] Stratified subsample: {subsample} / {len(X_train)} rows")
        X_fit, _, y_fit, _ = train_test_split(
            X_train, y_train,
            train_size=subsample,
            stratify=y_train,
            random_state=RANDOM_STATE,
        )

    t0 = time.perf_counter()
    pipe.fit(X_fit, y_fit)
    elapsed = time.perf_counter() - t0

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODELS_DIR / f"{name}.joblib")
    print(f"  Saved → models/{name}.joblib")

    return _recap(name, y_test, pipe.predict(X_test), elapsed)


def train_mlp(X_train, y_train, X_test, y_test) -> dict:
    """Entraîne un MLP Keras et sauvegarde le modèle, le préprocesseur et les courbes d'entraînement.

    Nécessite TensorFlow (Python 3.11/3.12 uniquement — pas de wheel pour 3.13+).
    """
    if not _TF_AVAILABLE:
        raise RuntimeError(
            "TensorFlow is not installed or not compatible with this Python version "
            f"(you are running Python {'.'.join(str(v) for v in __import__('sys').version_info[:3])}).\n"
            "Create a Python 3.11/3.12 environment and install tensorflow there:\n"
            "  pyenv install 3.12 && pyenv local 3.12\n"
            "  pip install tensorflow"
        )

    # Fit du préprocesseur sur le train uniquement — évite toute fuite de données
    pre = build_preprocessor()
    X_tr = pre.fit_transform(X_train, y_train)
    X_te = pre.transform(X_test)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pre, MODELS_DIR / "preprocessor.joblib")
    print("  Preprocessor saved → models/preprocessor.joblib")

    # Poids de classe pour compenser le déséquilibre des grades
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=np.asarray(y_train))
    class_weight_dict = dict(zip(classes.tolist(), weights.tolist()))

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_tr.shape[1],)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(5, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    t0 = time.perf_counter()
    history = model.fit(
        X_tr, np.asarray(y_train),
        epochs=50,
        batch_size=256,
        validation_split=0.15,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=1,
    )
    elapsed = time.perf_counter() - t0

    model.save(MODELS_DIR / "mlp.keras")
    print("  Saved → models/mlp.keras")

    # Courbes d'entraînement (loss et accuracy par epoch)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["loss"],     label="train")
    ax1.plot(history.history["val_loss"], label="val")
    ax1.set_title("Loss (sparse categorical cross-entropy)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(history.history["accuracy"],     label="train")
    ax2.plot(history.history["val_accuracy"], label="val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    n_epochs = len(history.history["loss"])
    fig.suptitle(
        f"MLP training history — OFF Nutri-Score  ({n_epochs} epochs)",
        fontsize=12,
    )
    plt.tight_layout()
    out_path = FIGURES_DIR / "mlp_history.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Training curves saved → {out_path}")

    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    return _recap("MLP (Keras)", y_test, y_pred, elapsed)


def main() -> list[dict]:
    print("=" * 54)
    print("Loading and cleaning data …")
    X, y = load_and_clean(RAW_PATH, nrows=SAMPLE_SIZE)
    X_train, X_test, y_train, y_test = get_train_test(X, y)
    save_label_mapping()
    print(f"Train : {X_train.shape}  |  Test : {X_test.shape}")
    print("=" * 54)

    results: list[dict] = []

    print("\n[1/5] Logistic Regression (baseline) …")
    results.append(train_sklearn(
        "logreg",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        X_train, y_train, X_test, y_test,
    ))

    print("\n[2/5] Random Forest …")
    results.append(train_sklearn(
        "random_forest",
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        X_train, y_train, X_test, y_test,
    ))

    # GradientBoosting n'a pas de class_weight natif — acceptable pour ce dataset
    print("\n[3/5] Gradient Boosting …")
    results.append(train_sklearn(
        "gradient_boosting",
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        X_train, y_train, X_test, y_test,
    ))

    print(f"\n[4/5] SVM RBF (subsample ≤ {SVM_SUBSAMPLE}, full test set) …")
    results.append(train_sklearn(
        "svm",
        SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE,
        ),
        X_train, y_train, X_test, y_test,
        subsample=SVM_SUBSAMPLE,
    ))

    print("\n[5/5] MLP (Keras) …")
    results.append(train_mlp(X_train, y_train, X_test, y_test))

    print("\n" + "=" * 54)
    print("SUMMARY — quick eval on held-out test set")
    print("=" * 54)
    print(f"{'Model':<22} {'Accuracy':>9} {'F1-macro':>9} {'Time(s)':>8}")
    print("─" * 52)
    for r in results:
        print(
            f"{r['model']:<22} {r['accuracy']:>9.4f} "
            f"{r['f1_macro']:>9.4f} {r['train_time_s']:>8.1f}"
        )

    return results


if __name__ == "__main__":
    main()
