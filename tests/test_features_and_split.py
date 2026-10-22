from __future__ import annotations

import unittest

import pandas as pd

from uav_flight_memory.experiments.splitting import flight_level_split
from uav_flight_memory.features.windowing import extract_window_features


class FeatureAndSplitTests(unittest.TestCase):
    def test_feature_extraction_uses_window_rows_only(self) -> None:
        window = pd.DataFrame(
            {
                "vehicle_local_position.x": [1.0, 3.0],
                "vehicle_local_position.y": [2.0, 4.0],
                "vehicle_local_position.z": [-1.0, -3.0],
                "vehicle_local_position.vx": [0.5, 1.5],
                "vehicle_local_position.vy": [1.5, 2.5],
                "vehicle_local_position.vz": [2.5, 3.5],
            }
        )
        features = extract_window_features(window)
        self.assertEqual(features["vehicle_local_position.x__mean"], 2.0)
        self.assertEqual(features["vehicle_local_position.x__min"], 1.0)
        self.assertEqual(features["vehicle_local_position.z__max"], -1.0)

    def test_flight_level_split_has_no_intersection(self) -> None:
        rows = []
        for klass in ["Normal", "Mechanical"]:
            for flight_idx in range(4):
                for window_idx in range(2):
                    rows.append(
                        {
                            "flight_id": f"{klass}-{flight_idx}",
                            "window_id": window_idx,
                            "label": klass,
                            "label_status": "usable",
                        }
                    )
        windows = pd.DataFrame(rows)
        train, test, _ = flight_level_split(windows, train_ratio=0.75, seed=7)
        self.assertTrue(set(train["flight_id"]).isdisjoint(set(test["flight_id"])))


if __name__ == "__main__":
    unittest.main()

