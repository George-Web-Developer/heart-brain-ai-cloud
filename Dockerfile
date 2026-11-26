FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements-cloud.txt ./requirements-cloud.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-cloud.txt

COPY --chown=user:user app.py ./app.py
COPY --chown=user:user src/__init__.py ./src/__init__.py
COPY --chown=user:user src/server/ ./src/server/
COPY --chown=user:user src/preprocessing/ ./src/preprocessing/
COPY --chown=user:user src/simulation/ ./src/simulation/
COPY --chown=user:user src/models/ ./src/models/
COPY --chown=user:user models/ ./models/

USER user

ENV PORT=10000 \
    PYTHONUNBUFFERED=1 \
    ENABLE_DEEP_LEARNING=1 \
    CLOUD_MINIMAL_MODELS=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    HOME=/home/user

EXPOSE 10000

CMD ["sh", "-c", "uvicorn src.server.app:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
