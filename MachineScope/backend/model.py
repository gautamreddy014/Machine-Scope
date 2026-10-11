"""
MachineScope machine-learning workflow.

Training / evaluation protocol (documented in README.md):

1. Inputs  : Type + five sensor readings (MODEL_INPUTS). The target is
             'Machine failure'. UDI / Product ID (identifiers) and the failure-mode
             flags TWF/HDF/PWF/OSF/RNF (which are *components of the label*) are
             excluded to prevent data leakage.
2. Split   : one stratified 80/20 train/test split, random_state=42. The test set
             is held out and used only once, for the final metrics.
3. Pipeline: PhysicsFeatures -> ColumnTransformer(impute, one-hot Type [, scale])
             -> classifier. Every preprocessing step is fitted on training rows only.
4. Models  : Random Forest (primary) and Logistic Regression (baseline), both with
             class_weight="balanced" because only ~3.4 % of records are failures.
5. Metrics : accuracy, precision, recall, F1, ROC-AUC, average precision and the
             confusion matrix, computed on the held-out test set at a 0.5 threshold.
             5-fold stratified CV on the training portion is reported for stability.
6. Scores for the maintenance planner are *out-of-fold* predictions
             (5-fold cross_val_predict over the whole dataset), so no record is
             scored by a model that saw its label.

Run ``python model.py`` from the backend folder to (re)train and save.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import FEATURE_META, PROJECT_ROOT, TARGET, Dataset, load_dataset  # noqa: E402
from ml_features import ENGINEERED, MODEL_INPUTS, NUMERIC_INPUTS, PhysicsFeatures  # noqa: E402

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
THRESHOLD = 0.5
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "machinescope_model.joblib")
ARTIFACT_VERSION = 3

# Hard physical-plausibility limits for prediction input (reject outside these).
HARD_LIMITS = {
    "Air_Temperature": (250.0, 350.0),
    "Process_Temperature": (250.0, 400.0),
    "Rotational_Speed": (0.0, 5000.0),
    "Torque": (0.0, 150.0),
    "Tool_Wear": (0.0, 500.0),
}


class ModelError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Pipelines
# --------------------------------------------------------------------------- #
def _preprocessor(scale: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), NUMERIC_INPUTS + ENGINEERED),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", ohe)]), ["Type"]),
        ]
    )


def build_random_forest() -> Pipeline:
    return Pipeline(
        [
            ("features", PhysicsFeatures()),
            ("prep", _preprocessor(scale=False)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_logistic_regression() -> Pipeline:
    return Pipeline(
        [
            ("features", PhysicsFeatures()),
            ("prep", _preprocessor(scale=True)),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def _downsample_curve(xs, ys, thr=None, max_points=200):
    xs, ys = np.asarray(xs), np.asarray(ys)
    if len(xs) > max_points:
        idx = np.unique(np.linspace(0, len(xs) - 1, max_points).astype(int))
        xs, ys = xs[idx], ys[idx]
    return {"x": np.round(xs, 5).tolist(), "y": np.round(ys, 5).tolist()}


def evaluate(y_true, proba, threshold=THRESHOLD) -> dict:
    y_true = np.asarray(y_true).astype(int)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y_true)) == 2
    out = {
        "n_test": int(len(y_true)),
        "n_test_failures": int(y_true.sum()),
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)) if both_classes else None,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc_auc": None,
        "average_precision": None,
        "roc_curve": None,
        "pr_curve": None,
    }
    if both_classes:
        out["roc_auc"] = float(roc_auc_score(y_true, proba))
        out["average_precision"] = float(average_precision_score(y_true, proba))
        fpr, tpr, _ = roc_curve(y_true, proba)
        out["roc_curve"] = _downsample_curve(fpr, tpr)
        prec, rec, _ = precision_recall_curve(y_true, proba)
        out["pr_curve"] = _downsample_curve(rec[::-1], prec[::-1])
    return out


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(ds: Dataset, save: bool = True, verbose: bool = True) -> dict:
    t0 = time.time()
    df = ds.df
    X = df[MODEL_INPUTS].copy()
    y = df[TARGET].astype(int).values
    classes, counts = np.unique(y, return_counts=True)

    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "sklearn_version": sklearn.__version__,
        "dataset_sha256": ds.sha256,
        "dataset_name": ds.name,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": MODEL_INPUTS,
        "excluded_columns": {
            "UDI": "record identifier (no predictive meaning)",
            "Product_ID": "identifier; first letter duplicates Type",
            "TWF/HDF/PWF/OSF/RNF": "failure-mode flags are components of the target label (leakage)",
        },
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "threshold": THRESHOLD,
        "valid": False,
        "message": None,
    }

    if len(classes) < 2 or counts.min() < CV_FOLDS:
        artifact["message"] = (
            "A valid evaluation cannot be performed: the target needs at least "
            f"{CV_FOLDS} examples of each class (found {dict(zip(classes.tolist(), counts.tolist()))})."
        )
        if save:
            _save(artifact)
        return artifact

    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # --- primary model ---
    rf = build_random_forest().fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_eval = evaluate(y_test, rf_proba)

    # --- baseline ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = build_logistic_regression().fit(X_train, y_train)
    lr_eval = evaluate(y_test, lr.predict_proba(X_test)[:, 1])

    # --- stability: CV on the TRAINING portion only ---
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv = cross_validate(
        build_random_forest(), X_train, y_train, cv=skf,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
    )
    cv_summary = {
        k.replace("test_", ""): {"mean": float(np.mean(v)), "std": float(np.std(v))}
        for k, v in cv.items() if k.startswith("test_")
    }

    # --- feature importance ---
    prep = rf.named_steps["prep"]
    names = [n.split("__", 1)[-1] for n in prep.get_feature_names_out()]
    impurity = sorted(
        (
            {"feature": n, "importance": float(v)}
            for n, v in zip(names, rf.named_steps["clf"].feature_importances_)
        ),
        key=lambda d: -d["importance"],
    )
    perm = permutation_importance(
        rf, X_test, y_test, scoring="average_precision", n_repeats=10, random_state=RANDOM_STATE, n_jobs=1
    )
    permutation = sorted(
        (
            {"feature": f, "mean": float(m), "std": float(s)}
            for f, m, s in zip(MODEL_INPUTS, perm.importances_mean, perm.importances_std)
        ),
        key=lambda d: -d["mean"],
    )

    # --- out-of-fold scores for every record (maintenance planner) ---
    skf_all = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(build_random_forest(), X, y, cv=skf_all, method="predict_proba")[:, 1]

    train_ranges = {
        c: {"min": float(X_train[c].min()), "max": float(X_train[c].max()),
            "p01": float(X_train[c].quantile(0.01)), "p99": float(X_train[c].quantile(0.99)),
            "median": float(X_train[c].median())}
        for c in NUMERIC_INPUTS
    }

    artifact.update(
        {
            "valid": True,
            "model": rf,
            "baseline": lr,
            "train_size": int(len(train_idx)),
            "test_size_n": int(len(test_idx)),
            "train_failures": int(y_train.sum()),
            "test_failures": int(y_test.sum()),
            "test_index": test_idx.tolist(),
            "test_proba": rf_proba.tolist(),
            "metrics": rf_eval,
            "baseline_metrics": lr_eval,
            "majority_baseline_accuracy": float(1 - y_test.mean()),
            "cv": cv_summary,
            "impurity_importance": impurity,
            "permutation_importance": permutation,
            "oof_proba": oof.tolist(),
            "train_ranges": train_ranges,
            "training_seconds": round(time.time() - t0, 2),
            "types_seen": sorted(X_train["Type"].dropna().unique().tolist()),
        }
    )
    if verbose:
        m = rf_eval
        print(
            f"[model] trained in {artifact['training_seconds']} s | test n={m['n_test']} "
            f"acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} "
            f"f1={m['f1']:.4f} auc={m['roc_auc']:.4f}"
        )
    if save:
        _save(artifact)
    return artifact


def _save(artifact: dict):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH, compress=3)


def load_or_train(ds: Dataset, force: bool = False, verbose: bool = True) -> dict:
    """Load the saved artifact if it matches this dataset + sklearn version, else retrain."""
    if not force and os.path.isfile(MODEL_PATH):
        try:
            art = joblib.load(MODEL_PATH)
            if (
                art.get("artifact_version") == ARTIFACT_VERSION
                and art.get("dataset_sha256") == ds.sha256
                and art.get("sklearn_version") == sklearn.__version__
            ):
                if verbose:
                    print(f"[model] loaded saved model from {MODEL_PATH}")
                return art
            if verbose:
                print("[model] saved model does not match dataset / scikit-learn version - retraining")
        except Exception as exc:  # corrupt or incompatible pickle
            if verbose:
                print(f"[model] could not load saved model ({exc}) - retraining")
    return train(ds, save=True, verbose=verbose)


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def validate_prediction_input(payload: dict, artifact: dict) -> tuple[dict, list, list]:
    """Return (clean_values, errors, warnings)."""
    errors, warns, clean = [], [], {}
    if not isinstance(payload, dict):
        return {}, ["Request body must be a JSON object."], []

    mtype = str(payload.get("Type", "M")).strip().upper()
    allowed_types = artifact.get("types_seen") or ["L", "M", "H"]
    if mtype not in allowed_types:
        errors.append(f"Type must be one of {', '.join(allowed_types)}.")
    clean["Type"] = mtype

    for col in NUMERIC_INPUTS:
        label = FEATURE_META[col]["label"]
        unit = FEATURE_META[col]["unit"]
        raw = payload.get(col)
        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            errors.append(f"{label} is required.")
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{label} must be a number.")
            continue
        if not np.isfinite(val):
            errors.append(f"{label} must be a finite number.")
            continue
        lo, hi = HARD_LIMITS[col]
        if not lo <= val <= hi:
            errors.append(f"{label} must be between {lo:g} and {hi:g} {unit}.")
            continue
        rng = artifact.get("train_ranges", {}).get(col)
        if rng and not rng["min"] <= val <= rng["max"]:
            warns.append(
                f"{label} {val:g} {unit} is outside the training range "
                f"({rng['min']:g}–{rng['max']:g} {unit}); the model is extrapolating."
            )
        clean[col] = val

    if not errors and clean["Process_Temperature"] < clean["Air_Temperature"]:
        warns.append("Process temperature is below air temperature, which never occurs in the training data.")
    return clean, errors, warns


def predict(artifact: dict, values: dict) -> dict:
    if not artifact.get("valid"):
        raise ModelError(artifact.get("message") or "Model is not available.")
    model = artifact["model"]
    X = pd.DataFrame([values], columns=MODEL_INPUTS)
    pred = int(model.predict(X)[0])
    has_proba = hasattr(model, "predict_proba")
    proba = float(model.predict_proba(X)[0, 1]) if has_proba else None
    base = artifact["baseline"]
    return {
        "predicted_class": pred,
        "predicted_label": "Failure predicted" if pred == 1 else "No failure predicted",
        "failure_score": proba,
        "probability_note": (
            "Random Forest vote share for the failure class (trained with balanced class "
            "weights). It ranks risk but is not a calibrated real-world probability."
        ),
        "baseline_prediction": int(base.predict(X)[0]),
        "baseline_score": float(base.predict_proba(X)[0, 1]),
        "threshold": artifact["threshold"],
        "derived": {
            "Temp_Difference": values["Process_Temperature"] - values["Air_Temperature"],
            "Power": values["Torque"] * values["Rotational_Speed"] * 2 * np.pi / 60.0,
        },
    }


if __name__ == "__main__":
    dataset = load_dataset()
    print(f"[data] {dataset.name}: {len(dataset.df)} rows")
    train(dataset, save=True, verbose=True)
    print(f"[model] saved to {MODEL_PATH}")
