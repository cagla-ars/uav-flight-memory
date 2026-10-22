from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ExperimentResult:
    config: dict[str, Any]
    metrics: dict[str, float]
    classification_report: dict[str, Any]
    confusion_matrix: list[list[int]]
    labels: list[str]
    cv_macro_f1_mean: float | None
    cv_macro_f1_std: float | None
    cv_folds: int

class LeakageSafeKNN:
    def __init__(self, k: int, metric: str):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.knn = KNeighborsClassifier(n_neighbors=k, metric=metric, algorithm="brute")
        self.named_steps = {
            "imputer": self.imputer,
            "scaler": self.scaler,
            "knn": self.knn,
        }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LeakageSafeKNN":
        
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        y_encoded = self.label_encoder.fit_transform(y)
        self.knn.fit(X_scaled, y_encoded)
        return self

    def transform_features(self, X: pd.DataFrame) -> np.ndarray:
        X_imputed = self.imputer.transform(X)
        return self.scaler.transform(X_imputed)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        encoded = self.knn.predict(self.transform_features(X))
        return self.label_encoder.inverse_transform(encoded)

    def kneighbors(self, X: pd.DataFrame, n_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
        return self.knn.kneighbors(self.transform_features(X), n_neighbors=n_neighbors)


def make_knn_pipeline(k: int, metric: str) -> LeakageSafeKNN:
    return LeakageSafeKNN(k=k, metric=metric)

#modelin başarısını ölçme
def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, labels: list[str]) -> tuple[dict[str, float], dict, list[list[int]]]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist()
    return metrics, report, matrix


def cross_validate_macro_f1(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    k: int,
    metric: str,
    seed: int,
) -> tuple[float | None, float | None, int]:
    group_label_counts = y.groupby(groups).agg(lambda values: values.value_counts().index[0])
    min_group_class = group_label_counts.value_counts().min() if len(group_label_counts) else 0
    n_splits = int(min(5, min_group_class))
    if n_splits < 2 or len(set(y)) < 2:
        return None, None, 0

    scores: list[float] = []
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_index, validation_index in splitter.split(X, y, groups):
        pipeline = make_knn_pipeline(k, metric)
        pipeline.fit(X.iloc[train_index], y.iloc[train_index])
        predictions = pipeline.predict(X.iloc[validation_index])
        _, _, f1, _ = precision_recall_fscore_support(
            y.iloc[validation_index],
            predictions,
            average="macro",
            zero_division=0,
        )
        scores.append(float(f1))
    return float(np.mean(scores)), float(np.std(scores)), n_splits

def run_single_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    labels: list[str],
    train_ratio: float,
    k: int,
    metric: str,
    seed: int,
) -> ExperimentResult:
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Train and test windows must both be non-empty.")
    if k > len(train_df):
        raise ValueError(f"k={k} is larger than the available train window count ({len(train_df)}).")

    X_train = train_df[feature_columns]
    y_train = train_df["label"]
    X_test = test_df[feature_columns]
    y_test = test_df["label"]

    cv_mean, cv_std, cv_folds = cross_validate_macro_f1(
        X_train,
        y_train,
        train_df["flight_id"],
        k=k,
        metric=metric,
        seed=seed,
    )
    pipeline = make_knn_pipeline(k, metric)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metrics, report, matrix = evaluate_predictions(y_test, predictions, labels)
    return ExperimentResult(
        config={
            "train_ratio": train_ratio,
            "k": k,
            "metric": metric,
            "seed": seed,
            "scaler": "StandardScaler fit on train only",
            "imputer": "SimpleImputer(median) fit on train only",
        },
        metrics=metrics,
        classification_report=report,
        confusion_matrix=matrix,
        labels=labels,
        cv_macro_f1_mean=cv_mean,
        cv_macro_f1_std=cv_std,
        cv_folds=cv_folds,
    )

