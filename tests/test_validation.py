import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from src import clustering_pipeline as pipeline


class InputValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train = pd.DataFrame({"a": [1., 2., 3.], "b": [4., 5., 6.]})

    def run_loader(self, train, predict):
        train.to_csv(self.root / "breast_cancer_train.csv", index=False)
        predict.to_csv(self.root / "breast_cancer_predict.csv", index=False)
        with patch.object(pipeline, "DATA_DIR", self.root), patch.object(
            pipeline, "OUTPUT_DIR", self.root / "outputs"
        ):
            return pipeline.load_data("validation_test")

    def test_valid_input(self):
        result = self.run_loader(self.train, self.train.iloc[:1])
        self.assertTrue((Path(result["folder"]) / "train.csv").is_file())

    def test_missing_value_rejected(self):
        bad = self.train.copy()
        bad.loc[0, "a"] = float("nan")
        with self.assertRaisesRegex(ValueError, "missing or infinite"):
            self.run_loader(bad, self.train)

    def test_reordered_columns_rejected(self):
        with self.assertRaisesRegex(ValueError, "columns must match"):
            self.run_loader(self.train, self.train[["b", "a"]])

    def test_label_column_rejected(self):
        bad = self.train.assign(diagnosis=0)
        with self.assertRaisesRegex(ValueError, "Diagnosis labels"):
            self.run_loader(bad, bad)


if __name__ == "__main__":
    unittest.main()
