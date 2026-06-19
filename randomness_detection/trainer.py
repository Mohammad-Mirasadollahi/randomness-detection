"""Logistic regression ensemble with CV tuning and probability calibration."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import CPU_FRACTION
from .parallel import extract_features_parallel, worker_count


class EnsembleModel:
    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    def predict_random_probability(self, features: list[float]) -> float:
        probability = self.pipeline.predict_proba([features])[0][1]
        return float(probability)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self.pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "EnsembleModel":
        with path.open("rb") as handle:
            pipeline = pickle.load(handle)
        return cls(pipeline)


def train_ensemble(
    texts: list[str],
    labels: list[int],
    freq_path: Path,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    cpu_fraction: float = CPU_FRACTION,
) -> tuple[EnsembleModel, dict[str, float | int | str]]:
    workers = worker_count(cpu_fraction)
    feature_matrix = np.array(
        extract_features_parallel(
            texts,
            str(freq_path),
            cpu_fraction=cpu_fraction,
        ),
        dtype=np.float64,
    )
    labels_array = np.array(labels)

    # Release any worker slots before sklearn spawns its own joblib pool.
    from .parallel import shutdown_joblib, stop_parallel

    stop_parallel()

    x_train, x_test, y_train, y_test = train_test_split(
        feature_matrix,
        labels_array,
        test_size=test_size,
        random_state=random_state,
        stratify=labels_array,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    base_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )

    param_grid = {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
    }
    search = GridSearchCV(
        base_pipeline,
        param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=workers,
        refit=True,
    )
    search.fit(x_train, y_train)
    best_c = search.best_params_["classifier__C"]

    calibrated = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                CalibratedClassifierCV(
                    LogisticRegression(
                        C=best_c,
                        max_iter=2000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=random_state,
                    ),
                    cv=cv,
                    method="sigmoid",
                    n_jobs=workers,
                ),
            ),
        ]
    )
    calibrated.fit(x_train, y_train)
    shutdown_joblib()

    test_probabilities = calibrated.predict_proba(x_test)[:, 1]
    test_predictions = calibrated.predict(x_test)

    metrics: dict[str, float | int | str] = {
        "accuracy": float(accuracy_score(y_test, test_predictions)),
        "auc": float(roc_auc_score(y_test, test_probabilities)),
        "brier_score": float(brier_score_loss(y_test, test_probabilities)),
        "best_C": float(best_c),
        "calibration": "sigmoid",
        "train_samples": float(len(x_train)),
        "test_samples": float(len(x_test)),
        "cpu_workers": workers,
        "cpu_fraction": cpu_fraction,
    }

    return EnsembleModel(calibrated), metrics
