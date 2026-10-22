from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from uav_flight_memory.data.mapping import load_mapping, parse_mapping


class MappingAndTimestampTests(unittest.TestCase):
    def test_mapping_ranges_are_parsed_from_ranges_not_annotation_timestamp(self) -> None:
        path = Path("data/raw/mapping.json")
        if not path.exists():
            self.skipTest("mapping.json is not available")
        records = parse_mapping(load_mapping(path))
        record = records["2018-05-24/19_57_23"]
        annotation = record.annotations[0]
        self.assertEqual(annotation.anomaly_class, "External Position")
        self.assertIn("Position.Y", annotation.ranges_by_topic)
        self.assertTrue(annotation.annotation_timestamp.startswith("2025-"))

    def test_sample_range_matches_csv_timestamp_domain(self) -> None:
        mapping_path = Path("data/raw/mapping.json")
        csv_path = Path("data/raw/csv_files_annotated/2018-05-24/19_57_23.csv")
        if not mapping_path.exists() or not csv_path.exists():
            self.skipTest("sample mapping/CSV files are not available")
        records = parse_mapping(load_mapping(mapping_path))
        df = pd.read_csv(csv_path, usecols=["timestamp"])
        start = float(df["timestamp"].min())
        end = float(df["timestamp"].max())
        intervals = records["2018-05-24/19_57_23"].annotations[0].ranges_by_topic["Position.Y"]
        self.assertTrue(any(interval_start >= start and interval_end <= end for interval_start, interval_end in intervals))


if __name__ == "__main__":
    unittest.main()

