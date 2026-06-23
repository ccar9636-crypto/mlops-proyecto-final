import os

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.logging_hf import write_prediction_log
from app.model import IrisOnnxModel


MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.onnx")
APP_ENV = os.getenv("APP_ENV", "dev")

app = FastAPI(title="MLOps Iris ONNX", version="1.0.0")
model = IrisOnnxModel(MODEL_PATH)


class IrisRequest(BaseModel):
    sepal_length: float = Field(..., example=5.1)
    sepal_width: float = Field(..., example=3.5)
    petal_length: float = Field(..., example=1.4)
    petal_width: float = Field(..., example=0.2)


@app.get("/")
def root() -> dict:
    return {
        "message": "API de prediccion Iris con ONNX",
        "environment": APP_ENV,
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok!", "environment": APP_ENV}


@app.post("/predict")
def predict(payload: IrisRequest) -> dict:
    features = [
        payload.sepal_length,
        payload.sepal_width,
        payload.petal_length,
        payload.petal_width,
    ]
    prediction = model.predict(features)
    response = {"environment": APP_ENV, "prediction": prediction}
    write_prediction_log(APP_ENV, payload.model_dump(), prediction)
    return response

