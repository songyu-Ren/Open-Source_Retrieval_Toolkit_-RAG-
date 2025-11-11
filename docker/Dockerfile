FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=./mlruns

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

# Build a small sample index (non-blocking if no data)
RUN python -c "import os; os.system('rag index --data data/raw || true')"

EXPOSE 8002
CMD ["uvicorn", "rag_toolkit.api:app", "--host", "0.0.0.0", "--port", "8002"]