import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.config import MODELS_DIR, RANDOM_STATE, RAW_DIR, SAMPLE_SIZE
from src.preprocessing import build_preprocessor, get_train_test, load_and_clean

# Constants
RAW_PATH        = RAW_DIR / "en.openfoodfacts.org.products.csv"
TUNE_SUBSAMPLE  = 30_000   # les lignes de train utilisées pour le tuning (RandomizedSearchCV)
N_ITER          = 15
CV_FOLDS        = 3
SCORING         = "f1_macro"
DEFAULT_RF_F1   = 0.9587   # baseline de train_models.py

# Distributions des paramètres 

RF_PARAM_DIST = {
    "clf__n_estimators":     [100, 200, 300],
    "clf__max_depth":        [None, 10, 20, 30],
    "clf__min_samples_split":[2, 5, 10],
    "clf__max_features":     ["sqrt", "log2"],
}

GB_PARAM_DIST = {
    "clf__n_estimators":  [100, 200],
    "clf__learning_rate": [0.05, 0.1, 0.2],
    "clf__max_depth":     [3, 5, 7],
    "clf__subsample":     [0.8, 1.0],
}


# Helpers

def _header(text: str) -> None:
    bar = "═" * 60
    print(f"\n{bar}", flush=True)
    print(f"  {text}", flush=True)
    print(bar, flush=True)


def _subsample_train(X_train, y_train, n: int):
    """Stratified subsample of n rows from the train set."""
    X_sub, _, y_sub, _ = train_test_split(
        X_train, y_train,
        train_size=n,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )
    return X_sub, y_sub


def tune_model(
    label: str,
    base_clf,
    param_dist: dict,
    X_sub, y_sub,
    X_train, y_train,
    X_test, y_test,
    save_key: str,
) -> dict:
    """
    Run RandomizedSearchCV on X_sub/y_sub, then refit the best pipeline on
    the full X_train and evaluate on X_test.
    """
    pipe = Pipeline([("pre", build_preprocessor()), ("clf", base_clf)])
    cv   = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=N_ITER,
        scoring=SCORING,
        cv=cv,
        n_jobs=1,
        refit=False,
        verbose=1,
        random_state=RANDOM_STATE,
        error_score="raise",
    )

    print(f"\n→ Searching {N_ITER} candidates × {CV_FOLDS} folds on {len(X_sub):,} rows …", flush=True)
    t0 = time.perf_counter()
    search.fit(X_sub, y_sub)
    search_time = time.perf_counter() - t0

    best_params_raw = search.best_params_          # e.x {"clf__n_estimators": 200, …}
    best_cv_score   = round(search.best_score_, 4)


    best_clf_params = {k.replace("clf__", ""): v for k, v in best_params_raw.items()}

    print(f"\n  Best CV F1-macro : {best_cv_score}", flush=True)
    print(f"  Best params      : {best_clf_params}", flush=True)
    print(f"  Search time      : {search_time:.1f}s", flush=True)

    # Refit on full X_train with best params 
    print(f"\n→ Refitting {label} on full train set ({len(X_train):,} rows) …", flush=True)
    best_clf  = base_clf.__class__(**{**base_clf.get_params(), **best_clf_params})
    best_pipe = Pipeline([("pre", build_preprocessor()), ("clf", best_clf)])

    t1 = time.perf_counter()
    best_pipe.fit(X_train, y_train)
    fit_time = time.perf_counter() - t1

    # Evaluate on test set
    y_pred  = best_pipe.predict(X_test)
    f1_test = round(f1_score(y_test, y_pred, average="macro"), 4)

    print(f"  Test F1-macro    : {f1_test}", flush=True)
    print(f"  Refit time       : {fit_time:.1f}s", flush=True)

    # Save tuned pipeline
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / f"{save_key}_tuned.joblib"
    joblib.dump(best_pipe, out_path)
    print(f"  Saved → models/{save_key}_tuned.joblib", flush=True)

    return {
        "model":        label,
        "best_cv_f1":   best_cv_score,
        "test_f1":      f1_test,
        "best_params":  str(best_clf_params),
        "search_time_s": round(search_time, 1),
        "refit_time_s":  round(fit_time, 1),
    }


# Main 

def main() -> None:
    _header("Loading and cleaning data")
    X, y = load_and_clean(RAW_PATH, nrows=SAMPLE_SIZE)
    X_train, X_test, y_train, y_test = get_train_test(X, y)
    print(f"Train : {X_train.shape}  |  Test : {X_test.shape}", flush=True)

    X_sub, y_sub = _subsample_train(X_train, y_train, TUNE_SUBSAMPLE)
    print(f"Tuning subsample : {X_sub.shape}", flush=True)

    results: list[dict] = []

    # 1. Random Forest 
    _header("Tuning Random Forest")
    rf_base = RandomForestClassifier(
        class_weight="balanced",
        n_jobs=1,                  # n_jobs=1 to avoid macOS SIGURG
        random_state=RANDOM_STATE,
    )
    rf_result = tune_model(
        label="Random Forest (tuned)",
        base_clf=rf_base,
        param_dist=RF_PARAM_DIST,
        X_sub=X_sub, y_sub=y_sub,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        save_key="random_forest",
    )
    results.append(rf_result)

    # Comparison vs default RF 
    delta = round(rf_result["test_f1"] - DEFAULT_RF_F1, 4)
    sign  = "+" if delta >= 0 else ""
    print(f"\n  Default RF F1-macro : {DEFAULT_RF_F1}", flush=True)
    print(f"  Tuned  RF F1-macro  : {rf_result['test_f1']}", flush=True)
    verdict = "AMÉLIORE" if delta > 0 else ("ÉGAL" if delta == 0 else "DÉGRADE")
    print(f"  Δ = {sign}{delta}  → tuning {verdict} le Random Forest", flush=True)

    # 2. Gradient Boosting
    _header("Tuning Gradient Boosting")
    gb_base = GradientBoostingClassifier(random_state=RANDOM_STATE)
    gb_result = tune_model(
        label="Gradient Boosting (tuned)",
        base_clf=gb_base,
        param_dist=GB_PARAM_DIST,
        X_sub=X_sub, y_sub=y_sub,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        save_key="gradient_boosting",
    )
    results.append(gb_result)

    # Summary
    _header("Summary")
    print(f"{'Model':<32} {'CV F1':>8} {'Test F1':>9} {'Search(s)':>10} {'Refit(s)':>9}", flush=True)
    print("─" * 72, flush=True)
    for r in results:
        print(
            f"{r['model']:<32} {r['best_cv_f1']:>8.4f} {r['test_f1']:>9.4f} "
            f"{r['search_time_s']:>10.1f} {r['refit_time_s']:>9.1f}",
            flush=True,
        )

    # Save CSV
    out_csv = MODELS_DIR / "tuning_results.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\nTuning results saved → {out_csv}", flush=True)


if __name__ == "__main__":
    main()
