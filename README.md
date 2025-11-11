# Open-Source Retrieval Toolkit (RAG)

Production-minded, modular RAG toolkit with CLI and FastAPI demo, offline evaluation, MLflow tracking, and DVC pipeline for reproducible artifacts.

## Features
- End-to-end RAG: loaders → chunker → embeddings → FAISS index → retrieval → optional LLM
- Offline evaluation: nDCG@k and MRR; logs to MLflow
- Reproducible artifacts under `artifacts/` with deterministic seeds
- Config-driven via `config/settings.yaml`, override with `RAG_SETTINGS` (path to YAML)
- Data versioning with DVC (stages: chunk → embed → index)
- Experiment tracking with MLflow (default `MLFLOW_TRACKING_URI=./mlruns`)
- Sphinx docs and Dockerized demo server
- CLI (`rag`) and API (FastAPI on port 8002)

## Quickstart
1. Install dependencies and package:
   ```bash
   make setup
   ```

2. Initialize DVC (no SCM):
   ```bash
   make dvc-init
   ```

3. Build index from raw docs:
   ```bash
   rag index --data data/raw
   ```

4. Run a query (top-k contexts, optional LLM):
   ```bash
   rag query --q "what is in these docs?" --k 5
   ```

5. Evaluate with qrels and queries (logs to MLflow):
   ```bash
   rag eval --qrels data/qrels.tsv --queries data/queries.tsv --k 10
   ```

6. Run tests:
   ```bash
   make test
   ```

7. Build docs:
   ```bash
   make docs
   ```

8. Docker build & run:
   ```bash
   make docker-build
   make docker-run
   ```

## Configuration
- Default config: `config/settings.yaml`
- Override via `RAG_SETTINGS` env var pointing to a YAML file. Overrides are deep-merged.
- Key parameters:
  - `embedding.model_name`: sentence-transformers model (default lightweight)
  - `chunking.chunk_size`, `chunking.chunk_overlap`
  - `index.type`: `IndexFlatIP` (cosine via normalization)
  - `eval.k`: default cutoff for nDCG/MRR
  - `server.port`: default 8002

## Artifacts
- `artifacts/chunks.parquet`: chunk metadata and text
- `artifacts/embeddings.npy`: embedding vectors (normalized)
- `artifacts/index.faiss`: FAISS index
- `artifacts/index_meta.jsonl`: chunk → {doc_id, start, end, text}
- `artifacts/eval.json`: metrics summary
- `./mlruns`: MLflow tracking directory (default)

## API
- `POST /query` `{query, k, llm: bool}` → top-k contexts and optional answer
- `GET /health` → index/model status and config summary
- `GET /metrics` → Prometheus metrics

## LLM Integration
- If `OPENAI_API_KEY` is set, uses an OpenAI-compatible HTTP endpoint.
- Default fallback `NoLLM` returns concatenated contexts.

## Development
- Format: `make fmt`
- Lint: `make lint`
- Test: `make test`

## License
MIT