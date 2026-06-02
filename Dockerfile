FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/artifacts/model.onnx

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY README.md .

ARG MODEL_URL=""
RUN if [ -n "$MODEL_URL" ]; then python scripts/download_model.py --url "$MODEL_URL" --output /app/artifacts/model.onnx; fi
RUN chmod +x scripts/start.sh

EXPOSE 7860

CMD ["scripts/start.sh"]

