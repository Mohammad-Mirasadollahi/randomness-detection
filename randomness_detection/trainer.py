"""Train and persist the gradient-boosted ensemble."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import CPU_FRACTION
from .ensemble_features import FEATURE_GROUPS, EnsembleFeatureVector
from .io_utils import atomic_pickle_dump
from .parallel import shutdown_joblib, worker_count


class EnsembleModel:
    """Calibrated gradient-boosting classifier over ensemble features."""

    def __init__(
        self,
        pipeline: Pipeline,
        *,
        active_groups: frozenset[str] | None = None,
        feature_names: tuple[str, ...] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.active_groups = active_groups or frozenset(FEATURE_GROUPS.keys())
        self.feature_names = feature_names

    def predict_random_probability(self, features: EnsembleFeatureVector) -> float:
        vector = features.as_list(active_groups=self.active_groups)
        probability = self.pipeline.predict_proba([vector])[0][1]
        return float(probability)

    def save(self, path: Path) -> None:
        payload = {
            "pipeline": self.pipeline,
            "active_groups": sorted(self.active_groups),
            "feature_names": self.feature_names,
        }
        atomic_pickle_dump(payload, path)

    @classmethod
    def load(cls, path: Path) -> "EnsembleModel":
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if isinstance(payload, Pipeline):
            return cls(payload)
        return cls(
            pipeline=payload["pipeline"],
            active_groups=frozenset(payload.get("active_groups", FEATURE_GROUPS.keys())),
            feature_names=payload.get("feature_names"),
        )


def train_ensemble(
    feature_rows: list[EnsembleFeatureVector],
    labels: list[int],
    *,
    active_groups: frozenset[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    cpu_fraction: float = CPU_FRACTION,
) -> tuple[EnsembleModel, dict[str, float | int | str]]:
    groups = active_groups or frozenset(FEATURE_GROUPS.keys())
    matrix = np.array(
        [row.as_list(active_groups=groups) for row in feature_rows],
        dtype=np.float64,
    )
    labels_array = np.array(labels)

    shutdown_joblib()
    workers = worker_count(cpu_fraction)

    x_train, x_test, y_train, y_test = train_test_split(
        matrix,
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
                HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=0.08,
                    max_depth=6,
                    min_samples_leaf=20,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )

    param_grid = {
        "classifier__max_depth": [4, 6, 8],
        "classifier__learning_rate": [0.05, 0.08, 0.12],
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
    best_params = search.best_params_

    calibrated = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                CalibratedClassifierCV(
                    HistGradientBoostingClassifier(
                        max_iter=300,
                        learning_rate=best_params["classifier__learning_rate"],
                        max_depth=best_params["classifier__max_depth"],
                        min_samples_leaf=20,
                        class_weight="balanced",
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
    test_predictions = (test_probabilities >= 0.5).astype(int)

    metrics: dict[str, float | int | str] = {
        "accuracy": float(accuracy_score(y_test, test_predictions)),
        "f1": float(f1_score(y_test, test_predictions)),
        "precision": float(precision_score(y_test, test_predictions, zero_division=0)),
        "recall": float(recall_score(y_test, test_predictions, zero_division=0)),
        "auc": float(roc_auc_score(y_test, test_probabilities)),
        "brier_score": float(brier_score_loss(y_test, test_probabilities)),
        "best_max_depth": int(best_params["classifier__max_depth"]),
        "best_learning_rate": float(best_params["classifier__learning_rate"]),
        "calibration": "sigmoid",
        "ensemble": "HistGradientBoosting",
        "train_samples": float(len(x_train)),
        "test_samples": float(len(x_test)),
        "feature_groups": ",".join(sorted(groups)),
        "cpu_workers": workers,
        "cpu_fraction": cpu_fraction,
    }

    return EnsembleModel(calibrated, active_groups=groups), metrics
