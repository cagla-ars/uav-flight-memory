from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from uav_flight_memory.config import (
    CSV_ROOT,
    DEFAULT_EXTERNAL_LIMIT,
    DEFAULT_MIN_ANOMALY_COVERAGE,
    DEFAULT_NORMAL_LIMIT,
    DEFAULT_SEED,
    DEFAULT_WINDOW_SECONDS,
    DISTANCE_METRICS,
    EXPERIMENTS_DIR,
    FEATURE_STATS,
    K_VALUES,
    LOCAL_POSITION_COLUMNS,
    PROCESSED_DIR,
    RAW_DIR,
    SPLIT_TRAIN_RATIOS,
    TARGET_CLASSES,
)
from uav_flight_memory.data.download import ensure_csv, local_csv_path
from uav_flight_memory.data.mapping import (
    class_distribution,
    flight_selection_rows,
    load_mapping,
    parse_mapping,
)
from uav_flight_memory.data.quality import inspect_csv
from uav_flight_memory.experiments.splitting import flight_level_split
from uav_flight_memory.features.windowing import coverage_bin, feature_names, windows_for_flight
from uav_flight_memory.models.knn import run_single_experiment


def ensure_dirs() -> None:
    for path in (RAW_DIR, CSV_ROOT, PROCESSED_DIR, EXPERIMENTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def mapping_path() -> Path:
    return RAW_DIR / "mapping.json"


def load_records() -> dict:
    return parse_mapping(load_mapping(mapping_path()))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def analyze_mapping() -> dict[str, Any]:
    ensure_dirs()
    records = load_records()
    summary = class_distribution(records)
    write_json(EXPERIMENTS_DIR / "mapping_summary.json", summary)
    return summary


def select_flights(
    seed: int = DEFAULT_SEED,
    normal_limit: int = DEFAULT_NORMAL_LIMIT,
    external_position_limit: int = DEFAULT_EXTERNAL_LIMIT,
) -> pd.DataFrame:
    ensure_dirs()
    records = load_records()
    rows = flight_selection_rows(
        records,
        seed=seed,
        normal_limit=normal_limit,
        external_position_limit=external_position_limit,
    )
    df = pd.DataFrame(rows)
    df["seed"] = seed
    df["normal_limit"] = normal_limit
    df["external_position_limit"] = external_position_limit
    df.to_csv(PROCESSED_DIR / "selected_flights.csv", index=False)
    return df


def selected_flights_or_create() -> pd.DataFrame:
    path = PROCESSED_DIR / "selected_flights.csv"
    if path.exists():
        return pd.read_csv(path)
    return select_flights()


def download_selected(limit: int | None = None) -> pd.DataFrame:
    ensure_dirs()
    selected = selected_flights_or_create()
    if limit is not None:
        selected = selected.head(limit)

    manifest = []
    for row in selected.itertuples(index=False):
        path, status = ensure_csv(str(row.flight_id))
        manifest.append(
            {
                "flight_id": row.flight_id,
                "class_set": row.class_set,
                "multi_annotation": bool(row.multi_annotation),
                "status": status,
                "csv_path": str(path) if path else "",
            }
        )
    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(PROCESSED_DIR / "download_manifest.csv", index=False)
    return manifest_df


def build_windows(
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    min_anomaly_coverage: float = DEFAULT_MIN_ANOMALY_COVERAGE,
) -> pd.DataFrame:
    ensure_dirs()
    records = load_records()
    selected = selected_flights_or_create()
    all_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []

    for row in selected.itertuples(index=False):
        flight_id = str(row.flight_id)
        csv_path = local_csv_path(flight_id)
        if not csv_path.exists():
            quality_rows.append({"flight_id": flight_id, "usable": False, "reason": "missing_csv"})
            continue
        quality = inspect_csv(csv_path)
        quality["flight_id"] = flight_id
        quality_rows.append(quality)
        if not quality.get("usable"):
            continue
        rows = windows_for_flight(
            flight_id=flight_id,
            csv_path=csv_path,
            record=records[flight_id],
            window_seconds=window_seconds,
            min_anomaly_coverage=min_anomaly_coverage,
        )
        all_rows.extend(rows)

    windows = pd.DataFrame(all_rows)
    if not windows.empty:
        windows["coverage_by_class_json"] = windows["coverage_by_class"].apply(
            lambda value: json.dumps(value, ensure_ascii=False)
        )
        windows = windows.drop(columns=["coverage_by_class"])
    windows.to_csv(PROCESSED_DIR / "windows.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(PROCESSED_DIR / "data_quality.csv", index=False)

    summary = summarize_windows(windows, window_seconds, min_anomaly_coverage)
    write_json(EXPERIMENTS_DIR / "windowing_summary.json", summary)
    return windows


def summarize_windows(windows: pd.DataFrame, window_seconds: int, min_anomaly_coverage: float) -> dict[str, Any]:
    if windows.empty:
        return {
            "window_seconds": window_seconds,
            "min_anomaly_coverage": min_anomaly_coverage,
            "window_count": 0,
            "label_counts": {},
            "label_status_counts": {},
        }

    coverage_counter: Counter[str] = Counter()
    mixed_count = 0
    for value in windows.get("coverage_by_class_json", []):
        coverages = json.loads(value) if isinstance(value, str) and value else {}
        if len(coverages) > 1:
            mixed_count += 1
        if coverages:
            coverage_counter[coverage_bin(max(float(v) for v in coverages.values()))] += 1

    return {
        "window_seconds": window_seconds,
        "min_anomaly_coverage": min_anomaly_coverage,
        "window_count": int(len(windows)),
        "flight_count": int(windows["flight_id"].nunique()),
        "label_counts": dict(Counter(windows["label"])),
        "label_status_counts": dict(Counter(windows["label_status"])),
        "anomaly_coverage_bins": dict(coverage_counter),
        "mixed_overlap_window_count": mixed_count,
        "feature_set": feature_names(),
    }


def load_windows() -> pd.DataFrame:
    path = PROCESSED_DIR / "windows.csv"
    if not path.exists():
        raise FileNotFoundError(f"Window table not found: {path}. Run build-windows first.")
    return pd.read_csv(path)

#farklı k değerlerinin çalıştırılması
def run_experiment_grid(
    seed: int = DEFAULT_SEED,
    split_ratios: tuple[float, ...] = SPLIT_TRAIN_RATIOS,
    k_values: tuple[int, ...] = K_VALUES,
    metrics: tuple[str, ...] = DISTANCE_METRICS,
) -> dict[str, Any]:
    ensure_dirs()
    windows = load_windows()
    feature_columns = [column for column in feature_names() if column in windows.columns]
    labels = [label for label in TARGET_CLASSES if label in set(windows.loc[windows["label_status"] == "usable", "label"])]
    if len(labels) < 2:
        raise ValueError("At least two usable classes are required to run kNN experiments.")

    results = []
    split_metadata: dict[str, Any] = {}
    for train_ratio in split_ratios:
        train_df, test_df, metadata = flight_level_split(windows, train_ratio=train_ratio, seed=seed)
        split_key = str(train_ratio)
        split_metadata[split_key] = metadata
        for metric in metrics:
            for k in k_values:
                if k > len(train_df):
                    continue
                result = run_single_experiment(
                    train_df=train_df,
                    test_df=test_df,
                    feature_columns=feature_columns,
                    labels=labels,
                    train_ratio=train_ratio,
                    k=k,
                
                    metric=metric,
                    seed=seed,
                )
                results.append(
                    {
                        **result.config,
                        **result.metrics,
                        "cv_macro_f1_mean": result.cv_macro_f1_mean,
                        "cv_macro_f1_std": result.cv_macro_f1_std,
                        "cv_folds": result.cv_folds,
                        "classification_report": result.classification_report,
                        "confusion_matrix": result.confusion_matrix,
                        "labels": result.labels,
                    }
                )

    if not results:
        raise ValueError("No experiment could run with the available train window count.")

    summary = {
        "dataset": "UAV-SEAD",
        "seed": seed,
        "feature_columns": feature_columns,
        "feature_source_columns": list(LOCAL_POSITION_COLUMNS),
        "feature_statistics": list(FEATURE_STATS),
        "split_metadata": split_metadata,
        "experiments": results,
        "model_selection_rule": "Choose by train-only grouped CV macro F1; test metrics are reported for comparison, not for tuning.",
        "selected_by_cv": select_best_by_cv(results),
    }
    write_json(EXPERIMENTS_DIR / "latest_summary.json", summary)
    pd.DataFrame(
        [
            {key: value for key, value in row.items() if key not in {"classification_report", "confusion_matrix", "labels"}}
            for row in results
        ]
    ).to_csv(EXPERIMENTS_DIR / "experiment_metrics.csv", index=False)
    return summary


def select_best_by_cv(results: list[dict[str, Any]]) -> dict[str, Any]:
    with_cv = [row for row in results if row.get("cv_macro_f1_mean") is not None]
    pool = with_cv if with_cv else results
    key = "cv_macro_f1_mean" if with_cv else "macro_f1"
    best = max(pool, key=lambda row: (row.get(key) or -1.0, row.get("macro_f1") or -1.0))
    return {
        "train_ratio": best["train_ratio"],
        "k": best["k"],
        "metric": best["metric"],
        "selection_metric": key,
        "selection_value": best.get(key),
        "note": "No grouped CV was possible; fallback used test macro F1 only for smoke/demo." if not with_cv else "",
    }


def run_single_web_experiment(train_ratio: float, k: int, metric: str, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    windows = load_windows()
    feature_columns = [column for column in feature_names() if column in windows.columns]
    labels = [label for label in TARGET_CLASSES if label in set(windows.loc[windows["label_status"] == "usable", "label"])]
    train_df, test_df, split_info = flight_level_split(windows, train_ratio=train_ratio, seed=seed)
    result = run_single_experiment(
        train_df=train_df,
        test_df=test_df,
        feature_columns=feature_columns,
        labels=labels,
        train_ratio=train_ratio,
        k=k,
        metric=metric,
        seed=seed,
    )
    return {
        "split": split_info,
        "feature_columns": feature_columns,
        "result": {
            **result.config,
            **result.metrics,
            "classification_report": result.classification_report,
            "confusion_matrix": result.confusion_matrix,
            "labels": result.labels,
            "cv_macro_f1_mean": result.cv_macro_f1_mean,
            "cv_macro_f1_std": result.cv_macro_f1_std,
            "cv_folds": result.cv_folds,
        },
    }

