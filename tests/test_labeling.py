from __future__ import annotations

import unittest

from uav_flight_memory.data.mapping import Annotation, FlightRecord
from uav_flight_memory.features.windowing import decide_window_label


class LabelingTests(unittest.TestCase):
    def test_normal_when_no_range_overlap(self) -> None:
        record = FlightRecord(
            flight_id="f",
            annotations=(Annotation("External Position", "", "", {"Position.X": [(100.0, 120.0)]}),),
        )
        decision = decide_window_label(record, 0, 20, min_anomaly_coverage=0.10)
        self.assertEqual(decision.label, "Normal")
        self.assertEqual(decision.label_status, "usable")

    def test_single_class_overlap_above_threshold_is_labeled(self) -> None:
        record = FlightRecord(
            flight_id="f",
            annotations=(Annotation("Mechanical", "", "", {"Battery": [(5.0, 8.0), (7.0, 10.0)]}),),
        )
        decision = decide_window_label(record, 0, 20, min_anomaly_coverage=0.10)
        self.assertEqual(decision.label, "Mechanical")
        self.assertEqual(decision.label_status, "usable")
        self.assertAlmostEqual(decision.coverage_by_class["Mechanical"], 0.25)

    def test_low_coverage_is_uncertain(self) -> None:
        record = FlightRecord(
            flight_id="f",
            annotations=(Annotation("Altitude", "", "", {"Actuator": [(1.0, 2.0)]}),),
        )
        decision = decide_window_label(record, 0, 20, min_anomaly_coverage=0.10)
        self.assertEqual(decision.label, "Uncertain")
        self.assertEqual(decision.label_status, "uncertain_partial")

    def test_multiple_anomaly_classes_are_uncertain(self) -> None:
        record = FlightRecord(
            flight_id="f",
            annotations=(
                Annotation("Altitude", "", "", {"Actuator": [(1.0, 6.0)]}),
                Annotation("Mechanical", "", "", {"Battery": [(8.0, 12.0)]}),
            ),
        )
        decision = decide_window_label(record, 0, 20, min_anomaly_coverage=0.10)
        self.assertEqual(decision.label, "Uncertain")
        self.assertEqual(decision.label_status, "uncertain_mixed")


if __name__ == "__main__":
    unittest.main()

