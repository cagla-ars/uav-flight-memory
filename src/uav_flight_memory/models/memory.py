from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from uav_flight_memory.models.knn import make_knn_pipeline


@dataclass
class FlightMemoryModel:
    pipeline: object
    train_windows: pd.DataFrame
    feature_columns: list[str]

    def explain_window(self, window: pd.Series, neighbor_count: int) -> dict[str, Any]:
        query = window[self.feature_columns].to_frame().T
        prediction = str(self.pipeline.predict(query)[0])
        n_neighbors = min(neighbor_count, len(self.train_windows))
        distances, indices = self.pipeline.kneighbors(query, n_neighbors=n_neighbors)

        neighbors = []
        for rank, (distance, index) in enumerate(zip(distances[0], indices[0]), start=1):
            neighbor = self.train_windows.iloc[int(index)]
            neighbors.append(
                {
                    "rank": rank,
                    "flight_id": str(neighbor["flight_id"]),
                    "window_id": int(neighbor["window_id"]),
                    "label": str(neighbor["label"]),
                    "distance": float(distance),
                }
            )

        votes = {}
        for neighbor in neighbors:
            votes[neighbor["label"]] = votes.get(neighbor["label"], 0) + 1

        return {
            "current": {
                "flight_id": str(window["flight_id"]),
                "window_id": int(window["window_id"]),
                "true_label": str(window["label"]),
                "start_timestamp": float(window["start_timestamp"]),
                "end_timestamp": float(window["end_timestamp"]),
            },
            "predicted_label": prediction,
            "neighbors": neighbors,
            "votes": votes,
            "note": "Distances are kNN feature-space distances after train-fitted imputation and scaling, not calibrated probabilities.",
        }


def fit_flight_memory(
    train_windows: pd.DataFrame,
    feature_columns: list[str],
    k: int,
    metric: str,
) -> FlightMemoryModel:
    if k > len(train_windows):
        raise ValueError(f"k={k} is larger than the available train window count ({len(train_windows)}).")
    pipeline = make_knn_pipeline(k, metric)
    pipeline.fit(train_windows[feature_columns], train_windows["label"])
    return FlightMemoryModel(pipeline=pipeline, train_windows=train_windows.reset_index(drop=True), feature_columns=feature_columns)

