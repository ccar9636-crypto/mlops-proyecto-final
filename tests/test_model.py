import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

from app.model import IrisOnnxModel


MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.onnx"))
TEST_DATA_PATH = Path(os.getenv("TEST_DATA_PATH", "artifacts/test_data.csv"))


def download_test_data() -> None:
    url = os.getenv("TEST_DATA_URL", "")
    if TEST_DATA_PATH.exists():
        return
    if not url:
        pytest.fail("TEST_DATA_URL esta vacio y no existe artifacts/test_data.csv")

    TEST_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    TEST_DATA_PATH.write_bytes(response.content)


def test_model_responds_to_defined_input() -> None:
    model = IrisOnnxModel(str(MODEL_PATH))
    prediction = model.predict([5.1, 3.5, 1.4, 0.2])

    assert prediction["class_id"] in [0, 1, 2]
    assert prediction["class_name"] in ["setosa", "versicolor", "virginica"]
    assert len(prediction["probabilities"]) == 3


def test_model_accuracy_is_acceptable() -> None:
    download_test_data()
    data = pd.read_csv(TEST_DATA_PATH)
    model = IrisOnnxModel(str(MODEL_PATH))

    features = data[["sepal_length", "sepal_width", "petal_length", "petal_width"]].to_numpy()
    labels = data["label"].to_numpy()
    predictions = np.array([model.predict(row.tolist())["class_id"] for row in features])
    accuracy = float((predictions == labels).mean())

    assert accuracy >= 0.90

