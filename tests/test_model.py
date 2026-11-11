import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from src import clustering_pipeline as pipeline


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        with patch.object(pipeline, "OUTPUT_DIR", Path(self.temp.name)):
            self.metadata = pipeline.data_preprocessing(
                pipeline.load_data("model_test")
            )

    def test_saved_pipeline_predictions(self):
        metadata = pipeline.build_save_model(self.metadata)
        summary = pipeline.load_model_predict(metadata)
        folder = Path(metadata["folder"])
        report = json.loads((folder / "training_report.json").read_text())
        model = joblib.load(folder / "clustering_pipeline.joblib")
        raw = pd.read_csv(folder / "predict.csv")
        saved = pd.read_csv(folder / "predictions.csv")
        self.assertEqual(
            model.named_steps["kmeans"].n_clusters, report["selected_k"]
        )
        np.testing.assert_array_equal(saved["cluster"], model.predict(raw))
        self.assertEqual(len(saved), len(raw))
        self.assertEqual(sum(summary["cluster_counts"].values()), len(raw))

    def test_missing_elbow_fails_without_saving_model(self):
        with patch("kneed.KneeLocator", return_value=SimpleNamespace(elbow=None)):
            with self.assertRaisesRegex(ValueError, "No elbow detected"):
                pipeline.build_save_model(self.metadata)
        folder = Path(self.metadata["folder"])
        self.assertFalse((folder / "clustering_pipeline.joblib").exists())


if __name__ == "__main__":
    unittest.main()
