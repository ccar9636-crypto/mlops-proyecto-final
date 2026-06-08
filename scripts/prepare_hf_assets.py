import os
from pathlib import Path

import numpy as np
import onnx
from huggingface_hub import upload_file
from onnx import TensorProto, helper, numpy_helper


MODEL_PATH = Path(".generated/model/iris_model.onnx")
TEST_DATA_PATH = Path(".generated/test/test_data.csv")


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def create_simple_iris_model() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    weights = np.array(
        [
            [0.3, 0.2, -0.1],
            [1.0, -0.2, -0.4],
            [-1.8, 1.0, 1.7],
            [-1.3, 0.3, 2.3],
        ],
        dtype=np.float32,
    )
    bias = np.array([2.5, -2.0, -6.0], dtype=np.float32)

    input_tensor = helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, 4])
    output_tensor = helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, 3])

    graph = helper.make_graph(
        nodes=[
            helper.make_node("MatMul", ["features", "weights"], ["logits_without_bias"]),
            helper.make_node("Add", ["logits_without_bias", "bias"], ["logits"]),
            helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1),
        ],
        name="simple_iris_classifier",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[
            numpy_helper.from_array(weights, name="weights"),
            numpy_helper.from_array(bias, name="bias"),
        ],
    )

    model = helper.make_model(
        graph,
        producer_name="mlops-proyecto-final",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, MODEL_PATH)


def create_test_data() -> None:
    TEST_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = np.array(
        [
            # setosa (label=0)
            [5.1, 3.5, 1.4, 0.2, 0],
            [4.9, 3.0, 1.4, 0.2, 0],
            [5.0, 3.4, 1.5, 0.2, 0],
            [4.7, 3.2, 1.3, 0.2, 0],
            [4.6, 3.1, 1.5, 0.2, 0],
            [5.4, 3.9, 1.7, 0.4, 0],
            [4.6, 3.4, 1.4, 0.3, 0],
            [4.4, 2.9, 1.4, 0.2, 0],
            [4.9, 3.1, 1.5, 0.1, 0],
            [5.8, 4.0, 1.2, 0.2, 0],
            # versicolor (label=1)
            [7.0, 3.2, 4.7, 1.4, 1],
            [6.4, 3.2, 4.5, 1.5, 1],
            [6.9, 3.1, 4.9, 1.5, 1],
            [5.5, 2.3, 4.0, 1.3, 1],
            [6.5, 2.8, 4.6, 1.5, 1],
            [5.7, 2.8, 4.5, 1.3, 1],
            [6.3, 3.3, 4.7, 1.6, 1],
            [4.9, 2.4, 3.3, 1.0, 1],
            [6.6, 2.9, 4.6, 1.3, 1],
            [5.2, 2.7, 3.9, 1.4, 1],
            # virginica (label=2)
            [6.3, 3.3, 6.0, 2.5, 2],
            [5.8, 2.7, 5.1, 1.9, 2],
            [7.2, 3.0, 5.8, 1.6, 2],
            [6.5, 3.0, 5.5, 1.8, 2],
            [7.7, 3.8, 6.7, 2.2, 2],
            [7.7, 2.6, 6.9, 2.3, 2],
            [7.7, 2.8, 6.7, 2.0, 2],
            [6.7, 3.3, 5.7, 2.1, 2],
            [6.4, 2.8, 5.6, 2.1, 2],
            [6.1, 2.6, 5.6, 1.4, 2],
        ],
        dtype=np.float32,
    )

    header = "sepal_length,sepal_width,petal_length,petal_width,label"
    np.savetxt(TEST_DATA_PATH, rows, delimiter=",", header=header, comments="", fmt="%.2f")


def upload_assets() -> None:
    repo_id = os.getenv("HF_ASSETS_REPO_ID")
    token = os.getenv("HF_TOKEN")

    if not repo_id or not token:
        raise ValueError("Configura HF_ASSETS_REPO_ID y HF_TOKEN antes de ejecutar este script.")

    upload_file(
        path_or_fileobj=str(MODEL_PATH),
        path_in_repo="model/iris_model.onnx",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message="Upload ONNX model",
    )
    upload_file(
        path_or_fileobj=str(TEST_DATA_PATH),
        path_in_repo="test/test_data.csv",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message="Upload test data",
    )

    for env in ["dev", "prod"]:
        log_file = Path(f".generated/logs/predicciones_{env}.txt")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.touch()
        upload_file(
            path_or_fileobj=str(log_file),
            path_in_repo=f"logs/predicciones_{env}.txt",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message=f"Create {env} prediction log",
        )


def main() -> None:
    create_simple_iris_model()
    create_test_data()
    upload_assets()
    print("Activos creados y subidos a Hugging Face Dataset.")


if __name__ == "__main__":
    main()
