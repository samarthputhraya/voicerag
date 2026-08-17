# VoiceRAG API image.
#
# Two properties drive every choice here:
#
#   1. The stack is torch-free. model2vec static embeddings, bm25s and faiss all
#      run on numpy + onnxruntime, so the image is ~400MB instead of the ~4GB a
#      torch-based RAG service needs. That is not vanity: a smaller image means a
#      faster cold start, and a cold start on the first judged request would
#      destroy the latency numbers this project is measured on.
#
#   2. Models and the index are baked in at build time, never fetched at boot.
#      A container that downloads a model on startup has a multi-second first
#      request. We pay that cost once, in CI.

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

# OMP_NUM_THREADS=1 is deliberate. faiss and onnxruntime both spawn OpenMP pools
# sized to the host core count; on a small container that oversubscribes the CPU
# and makes single-query latency *worse* than single-threaded. Our queries are
# one-at-a-time and tiny, so thread pools cost more than they return.

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY eval/ ./eval/
COPY scripts/ ./scripts/
COPY pyproject.toml .

# --- model warm-bake ---------------------------------------------------------
# Pull the static embedding model into the image so the runtime never touches
# the network. ~120MB, and it is the difference between a 15ms first request and
# a 4000ms one.
ARG EMBED_MODEL=minishlab/potion-retrieval-32M
RUN python -c "\
from model2vec import StaticModel; \
m = StaticModel.from_pretrained('${EMBED_MODEL}'); \
m.save_pretrained('/app/models/embedder'); \
print('baked', '${EMBED_MODEL}')"

ENV VOICERAG_EMBEDDER=static:/app/models/embedder

# --- index ---------------------------------------------------------------
# The prebuilt index is expected at /app/data/index. Build it in CI with
#   python scripts/ingest.py --limit 20000 --strategy sentence_window --out data/index
# and COPY it in, or mount it as a volume. Building at runtime is not supported
# on purpose: a 200k-chunk index takes minutes and must never be on the boot path.
COPY data/ ./data/
ENV VOICERAG_INDEX_DIR=/app/data/index

RUN adduser --disabled-password --gecos "" --uid 10001 app \
 && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=4s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:8000/healthz || exit 1

# One worker. The index lives in process memory; a second worker would double
# RAM for zero throughput benefit at our concurrency, and would halve the
# effective CPU available to the request that is actually being timed.
CMD ["uvicorn", "voicerag.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--loop", "uvloop", "--http", "httptools", \
     "--no-access-log"]
