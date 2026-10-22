from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from uav_flight_memory.config import FEATURE_STATS, LOCAL_POSITION_COLUMNS, TARGET_CLASSES
from uav_flight_memory.data.mapping import FlightRecord, intervals_by_class
from uav_flight_memory.labeling.intervals import overlap_duration


@dataclass(frozen=True)
class LabelDecision:
    label: str
    label_status: str
    coverage_by_class: dict[str, float]

def decide_window_label(
    record: FlightRecord,
    window_start: float,
    window_end: float,
    min_anomaly_coverage: float,
) -> LabelDecision:
    
    window_duration = max(1.0, float(window_end) - float(window_start))
    coverage_by_class: dict[str, float] = {}

    for klass, intervals in intervals_by_class(record).items():
        if klass == "Normal":
            continue
        overlap = overlap_duration(intervals, window_start, window_end)
        if overlap > 0:
            coverage_by_class[klass] = overlap / window_duration

    if not coverage_by_class:
        return LabelDecision("Normal", "usable", {})

    overlapping_classes = [klass for klass in coverage_by_class if klass in TARGET_CLASSES]
    if len(overlapping_classes) > 1:
        return LabelDecision("Uncertain", "uncertain_mixed", coverage_by_class)
    
    if not overlapping_classes:
        return LabelDecision("Uncertain", "uncertain_unknown_class", coverage_by_class)

    klass = overlapping_classes[0]
    if coverage_by_class[klass] >= min_anomaly_coverage:
        return LabelDecision(klass, "usable", coverage_by_class)
    return LabelDecision("Uncertain", "uncertain_partial", coverage_by_class)

def feature_names(
    columns: tuple[str, ...] = LOCAL_POSITION_COLUMNS,
    stats: tuple[str, ...] = FEATURE_STATS,
) -> list[str]:
    return [f"{column}__{stat}" for column in columns for stat in stats]

def extract_window_features(
    window: pd.DataFrame,
    columns: tuple[str, ...] = LOCAL_POSITION_COLUMNS,
    stats: tuple[str, ...] = FEATURE_STATS,
) -> dict[str, float]:
    features: dict[str, float] = {}

    for column in columns:
        values = pd.to_numeric(window[column], errors="coerce") if column in window.columns else pd.Series(dtype=float)
        for stat in stats:
            key = f"{column}__{stat}"
            if stat == "mean":
                value = values.mean()
            elif stat == "std":
                value = values.std(ddof=0)
            elif stat == "min":
                value = values.min()
            elif stat == "max":
                value = values.max()
            else:
                raise ValueError(f"Unsupported statistic: {stat}")
            features[key] = float(value) if pd.notna(value) else np.nan
    return features

def windows_for_flight(
    flight_id: str,
    csv_path: str | Path,
    record: FlightRecord,
    window_seconds: int,
    min_anomaly_coverage: float,
    columns: tuple[str, ...] = LOCAL_POSITION_COLUMNS,
) -> list[dict[str, Any]]:
    
    usecols = lambda column: column == "timestamp" or column in columns
    df = pd.read_csv(csv_path, usecols=usecols)

    if "timestamp" not in df.columns:
        return []
    
    for column in columns:
        if column not in df.columns:
            df[column] = np.nan

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp", kind="mergesort")

    if len(df) < 2:
        return []

    start = float(df["timestamp"].min())
    end = float(df["timestamp"].max())
    window_us = float(window_seconds * 1_000_000)
    rows: list[dict[str, Any]] = []
    window_id = 0
    current = start
    while current + window_us <= end:
        window_end = current + window_us
        mask = (df["timestamp"] >= current) & (df["timestamp"] < window_end)
        window = df.loc[mask]
        if len(window) >= 2:
            decision = decide_window_label(record, current, window_end, min_anomaly_coverage)
            row: dict[str, Any] = {
                "flight_id": flight_id,
                "window_id": window_id,
                "start_timestamp": current,
                "end_timestamp": window_end,
                "label": decision.label,
                "label_status": decision.label_status,
                "sample_count": int(len(window)),
                "coverage_by_class": decision.coverage_by_class,
            }
            row.update(extract_window_features(window, columns=columns))
            rows.append(row)
        window_id += 1
        current = window_end
    return rows

def coverage_bin(value: float) -> str:
    if value <= 0:
        return "0%"
    if value <= 0.10:
        return "0-10%"
    if value <= 0.25:
        return "10-25%"
    if value <= 0.50:
        return "25-50%"
    if value <= 0.75:
        return "50-75%"
    return "75-100%"

