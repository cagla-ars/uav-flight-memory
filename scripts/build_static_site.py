from __future__ import annotations

import json
import shutil
from pathlib import Path

from uav_flight_memory.experiments.runner import load_windows
from uav_flight_memory.experiments.splitting import flight_level_split
from uav_flight_memory.features.windowing import feature_names
from uav_flight_memory.models.memory import fit_flight_memory


ROOT = Path(__file__).resolve().parents[1]
STATIC_SRC = ROOT / "src" / "uav_flight_memory" / "web" / "static"
PUBLIC_DIR = ROOT / "dist"
PUBLIC_STATIC = PUBLIC_DIR / "static"
EXPERIMENTS_DIR = ROOT / "results" / "experiments"


def main() -> None:
    PUBLIC_STATIC.mkdir(parents=True, exist_ok=True)
    shutil.copy2(STATIC_SRC / "index.html", PUBLIC_DIR / "index.html")
    shutil.copy2(STATIC_SRC / "styles.css", PUBLIC_STATIC / "styles.css")
    shutil.copy2(STATIC_SRC / "app.js", PUBLIC_STATIC / "app.js")

    latest = json.loads((EXPERIMENTS_DIR / "latest_summary.json").read_text(encoding="utf-8"))
    windowing = json.loads((EXPERIMENTS_DIR / "windowing_summary.json").read_text(encoding="utf-8"))

    windows = load_windows()
    train_df, test_df, split_info = flight_level_split(windows, train_ratio=0.8, seed=42)
    selected = test_df[(test_df["flight_id"] == "2018-05-26/16_48_10") & (test_df["window_id"].astype(int) == 1)]
    memory = fit_flight_memory(
        train_df,
        feature_columns=[column for column in feature_names() if column in windows.columns],
        k=3,
        metric="euclidean",
    )
    explanation = memory.explain_window(selected.iloc[0], neighbor_count=3)
    explanation["split"] = split_info

    payload = {
        "latest_summary": latest,
        "windowing_summary": windowing,
        "memory": explanation,
    }
    (PUBLIC_STATIC / "static-data.js").write_text(
        "window.UAV_STATIC_DATA = " + json.dumps(payload, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
