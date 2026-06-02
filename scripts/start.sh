#!/bin/sh
set -e

if [ ! -f "${MODEL_PATH:-/app/artifacts/model.onnx}" ]; then
  python scripts/download_model.py --url "$MODEL_URL" --output "${MODEL_PATH:-/app/artifacts/model.onnx}"
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 7860
