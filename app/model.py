from pathlib import Path

import numpy as np
import onnxruntime as ort


CLASS_NAMES = {
    0: "setosa",
    1: "versicolor",
    2: "virginica",
}


class IrisOnnxModel:
    def __init__(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"No existe el modelo ONNX en {path}")

        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, features: list[float]) -> dict:
        array = np.array([features], dtype=np.float32)
        outputs = self.session.run([self.output_name], {self.input_name: array})
        probabilities = outputs[0][0]
        predicted_class = int(np.argmax(probabilities))

        return {
            "class_id": predicted_class,
            "class_name": CLASS_NAMES[predicted_class],
            "probabilities": {
                CLASS_NAMES[index]: float(value)
                for index, value in enumerate(probabilities)
            },
        }

