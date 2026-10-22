from __future__ import annotations

import unittest

from uav_flight_memory.labeling.intervals import overlap_duration, total_duration, union_intervals


class IntervalTests(unittest.TestCase):
    def test_union_merges_overlapping_intervals_without_double_counting(self) -> None:
        merged = union_intervals([(0, 10), (5, 12), (20, 25), (25, 30)])
        self.assertEqual(merged, [(0.0, 12.0), (20.0, 30.0)])
        self.assertEqual(total_duration(merged), 22.0)

    def test_overlap_duration_is_clipped_to_window(self) -> None:
        intervals = union_intervals([(0, 10), (15, 30)])
        self.assertEqual(overlap_duration(intervals, 5, 20), 10.0)


if __name__ == "__main__":
    unittest.main()

