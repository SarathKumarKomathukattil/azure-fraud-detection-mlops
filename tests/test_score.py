import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


# Path to the real score.py
SCORE_PATH = Path("azureml/online_scoring/score.py")

# Load score.py as a module
spec = importlib.util.spec_from_file_location("score", SCORE_PATH)
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


class FakeModel:
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_proba(self, df):
        return np.array(
            [[1 - p, p] for p in self.probabilities]
        )


def test_prediction_below_threshold():
    score.model = FakeModel([0.20])
    score.threshold = 0.43953168

    raw_data = json.dumps({
        "input_data": [
            {"amount": 100.0}
        ]
    })

    result = score.run(raw_data)

    assert result["fraud_probability"] == pytest.approx([0.20])
    assert result["fraud_prediction"] == [0]


def test_prediction_above_threshold():
    score.model = FakeModel([0.80])
    score.threshold = 0.43953168

    raw_data = json.dumps({
        "input_data": [
            {"amount": 100.0}
        ]
    })

    result = score.run(raw_data)

    assert result["fraud_probability"] == pytest.approx([0.80])
    assert result["fraud_prediction"] == [1]


def test_prediction_exactly_at_threshold():
    score.model = FakeModel([0.43953168])
    score.threshold = 0.43953168

    raw_data = json.dumps({
        "input_data": [
            {"amount": 100.0}
        ]
    })

    result = score.run(raw_data)

    assert result["fraud_prediction"] == [1]


def test_multiple_transactions():
    score.model = FakeModel([0.20, 0.80])
    score.threshold = 0.43953168

    raw_data = json.dumps({
        "input_data": [
            {"amount": 100.0},
            {"amount": 5000.0}
        ]
    })

    result = score.run(raw_data)

    assert result["fraud_probability"] == pytest.approx([0.20, 0.80])
    assert result["fraud_prediction"] == [0, 1]


def test_init_loads_model_and_threshold(tmp_path, monkeypatch):
    model_root = tmp_path
    model_dir = model_root / "model"

    model_dir.mkdir()

    threshold_file = model_dir / "threshold.json"

    with open(threshold_file, "w") as file:
        json.dump(
            {"operating_threshold": 0.43953168},
            file
        )

    fake_model = object()

    monkeypatch.setenv(
        "AZUREML_MODEL_DIR",
        str(model_root)
    )

    monkeypatch.setattr(
        score.mlflow.xgboost,
        "load_model",
        lambda path: fake_model
    )

    score.init()

    assert score.model is fake_model
    assert score.threshold == 0.43953168