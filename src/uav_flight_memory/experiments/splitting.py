from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

def flight_level_split(
    windows: pd.DataFrame,
    train_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    usable = windows[windows["label_status"] == "usable"].copy()

    flight_labels = (
        usable.groupby("flight_id")["label"]
        .agg(lambda values: values.value_counts().index[0])
        .reset_index()
        .rename(columns={"label": "stratify_label"})
    )

    rng = np.random.default_rng(seed)
    test_ratio = 1.0 - train_ratio
    train_flights: list[str] = []
    test_flights: list[str] = []
    warnings: list[str] = []

    for label, group in flight_labels.groupby("stratify_label"):
        flights = sorted(group["flight_id"].tolist())
        rng.shuffle(flights)

        if len(flights) == 1:
            train_flights.extend(flights)
            warnings.append(f"Class {label!r} has one flight; kept in train to avoid single-sample test instability.")
            continue
        test_count = max(1, int(round(len(flights) * test_ratio)))

        if len(flights) - test_count < 1:
            test_count = len(flights) - 1

        test_flights.extend(flights[:test_count])
        train_flights.extend(flights[test_count:])

    train_set = set(train_flights)
    test_set = set(test_flights)
    assert train_set.isdisjoint(test_set), "Flight leakage detected: train/test flight IDs intersect."
    train_df = usable[usable["flight_id"].isin(train_set)].copy()
    test_df = usable[usable["flight_id"].isin(test_set)].copy()

    metadata = {
        "train_ratio": train_ratio,
        "seed": seed,
        "train_flight_count": len(train_set),
        "test_flight_count": len(test_set),
        "train_window_count": int(len(train_df)),
        "test_window_count": int(len(test_df)),
        "train_label_counts": dict(Counter(train_df["label"])),
        "test_label_counts": dict(Counter(test_df["label"])),
        "warnings": warnings,
    }
    return train_df, test_df, metadata

