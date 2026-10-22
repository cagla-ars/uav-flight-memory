from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from uav_flight_memory.config import LOCAL_POSITION_COLUMNS


def inspect_csv(path: str | Path, required_columns: tuple[str, ...] = LOCAL_POSITION_COLUMNS) -> dict[str, Any]:
    df = pd.read_csv(path, usecols=lambda column: column == "timestamp" or column in required_columns)
    missing_required = [column for column in ("timestamp", *required_columns) if column not in df.columns]
    result: dict[str, Any] = {
        "path": str(path),
        "rows": int(len(df)),
        "missing_required_columns": missing_required,
        "usable": not missing_required and len(df) >= 2,
    }
    if "timestamp" not in df.columns or len(df) == 0:
        return result

    timestamps = pd.to_numeric(df["timestamp"], errors="coerce")
    diffs = timestamps.diff().dropna()
    result.update(
        {
            "timestamp_min": float(timestamps.min()),
            "timestamp_max": float(timestamps.max()),
            "duration_seconds": float((timestamps.max() - timestamps.min()) / 1_000_000),
            "timestamp_monotonic": bool(timestamps.is_monotonic_increasing),
            "duplicate_timestamps": int(timestamps.duplicated().sum()),
            "median_gap_seconds": float(diffs.median() / 1_000_000) if len(diffs) else None,
            "max_gap_seconds": float(diffs.max() / 1_000_000) if len(diffs) else None,
            "missing_fraction_by_required_column": {
                column: float(pd.to_numeric(df[column], errors="coerce").isna().mean())
                for column in required_columns
                if column in df.columns
            },
        }
    )
    return result

