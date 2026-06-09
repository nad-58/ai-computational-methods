"""Numerical examples for traditional machine-learning algorithms."""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def _round(values, digits: int = 4):
    return np.asarray(values).round(digits).tolist()


def make_small_datasets(seed: int = 7) -> dict:
    X_cls, y_cls = make_classification(
        n_samples=24,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        class_sep=1.3,
        random_state=seed,
    )
    X_reg, y_reg = make_regression(
        n_samples=20,
        n_features=3,
        noise=3.0,
        random_state=seed,
    )
    return {
        "classification_X": X_cls,
        "classification_y": y_cls,
        "regression_X": X_reg,
        "regression_y": y_reg,
    }


def classical_ml_examples(seed: int = 7) -> dict:
    data = make_small_datasets(seed)
    Xc, yc = data["classification_X"], data["classification_y"]
    Xr, yr = data["regression_X"], data["regression_y"]
    train_c, test_c = np.arange(16), np.arange(16, 24)
    train_r, test_r = np.arange(14), np.arange(14, 20)

    classifiers = {
        "decision_tree": DecisionTreeClassifier(max_depth=3, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=25, random_state=seed),
        "logistic_regression": LogisticRegression(max_iter=1000),
        "knn": KNeighborsClassifier(n_neighbors=3),
        "naive_bayes": GaussianNB(),
    }

    results: dict[str, dict] = {}
    for name, model in classifiers.items():
        model.fit(Xc[train_c], yc[train_c])
        predictions = model.predict(Xc[test_c])
        results[name] = {
            "train_samples": len(train_c),
            "test_samples": len(test_c),
            "first_test_vector": _round(Xc[test_c][0]),
            "true_labels": yc[test_c].tolist(),
            "predictions": predictions.tolist(),
            "accuracy": round(float(accuracy_score(yc[test_c], predictions)), 4),
        }

    linear_model = LinearRegression().fit(Xr[train_r], yr[train_r])
    regression_predictions = linear_model.predict(Xr[test_r])
    results["linear_regression"] = {
        "train_samples": len(train_r),
        "test_samples": len(test_r),
        "coefficients": _round(linear_model.coef_),
        "intercept": round(float(linear_model.intercept_), 4),
        "targets": _round(yr[test_r]),
        "predictions": _round(regression_predictions),
        "mae": round(float(mean_absolute_error(yr[test_r], regression_predictions)), 4),
        "r2": round(float(r2_score(yr[test_r], regression_predictions)), 4),
    }
    results["dataset_summary"] = {
        "classification_rows": int(len(Xc)),
        "classification_features": int(Xc.shape[1]),
        "regression_rows": int(len(Xr)),
        "regression_features": int(Xr.shape[1]),
    }
    results["flow"] = "synthetic rows -> fixed train/test split -> fit -> predict -> metric"
    return results
