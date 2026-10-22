from __future__ import annotations

import unittest

import pandas as pd

from uav_flight_memory.models.knn import make_knn_pipeline


class KNNPipelineTests(unittest.TestCase):
    def test_scaler_is_fit_on_train_only_inside_pipeline(self) -> None:
        train = pd.DataFrame({"a": [0.0, 2.0], "b": [10.0, 12.0]})
        labels = pd.Series(["Normal", "Mechanical"])
        test = pd.DataFrame({"a": [1000.0], "b": [1000.0]})
        pipeline = make_knn_pipeline(k=1, metric="euclidean")
        pipeline.fit(train, labels)
        pipeline.predict(test)
        scaler = pipeline.named_steps["scaler"]
        self.assertEqual(scaler.mean_.tolist(), [1.0, 11.0])

    def test_knn_can_predict(self) -> None:
        train = pd.DataFrame({"a": [0.0, 10.0, 11.0], "b": [0.0, 10.0, 11.0]})
        labels = pd.Series(["Normal", "Mechanical", "Mechanical"])
        pipeline = make_knn_pipeline(k=1, metric="manhattan")
        pipeline.fit(train, labels)
        prediction = pipeline.predict(pd.DataFrame({"a": [10.5], "b": [10.5]}))
        self.assertEqual(prediction[0], "Mechanical")


if __name__ == "__main__":
    unittest.main()

