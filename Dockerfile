# CPU-only image. This service only calls the Groq / OpenAI / Anthropic HTTP
# APIs and runs no local model inference, so no CUDA/GPU base is required.
# (requirements.txt was checked: no torch/tensorflow/cuda/nvidia/onnxruntime-gpu.)
FROM python:3.11-slim

# Predictable, container-friendly Python.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching (only re-runs when
# requirements.txt changes). All pins ship prebuilt manylinux/py3 wheels, so no
# system compiler is needed on slim. If a future pin needs building, add:
#   RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Application source (see .dockerignore — .env, data folders, venvs, caches are
# excluded so secrets are never baked into the image).
COPY . .

# Cloud Run injects PORT at runtime (commonly 8080). Provide a default so that a
# plain `docker run` locally still has a value to bind; Cloud Run overrides it.
ENV PORT=8058
EXPOSE 8058

# Shell form so $PORT expands. Binds Cloud Run's injected PORT (or the ENV
# default above for local runs). Never hardcode 8058 here, or Cloud Run's
# startup health check against its own PORT would fail.
CMD uvicorn agent.api:app --host 0.0.0.0 --port $PORT
