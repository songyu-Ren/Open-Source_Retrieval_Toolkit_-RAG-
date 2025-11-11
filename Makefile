.PHONY: setup fmt lint test index query eval docs serve docker-build docker-run dvc-init

PORT ?= 8002

setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

fmt:
	black --line-length 100 .

lint:
	ruff check .
	black --check --line-length 100 .

test:
	pytest -q

index:
	rag index --data data/raw

query:
	rag query --q "what is in these docs?" --k 5

eval:
	rag eval --qrels data/qrels.tsv --queries data/queries.tsv --k 10

docs:
	$(MAKE) -C docs html

serve:
	uvicorn rag_toolkit.api:app --host 0.0.0.0 --port $(PORT)

docker-build:
	docker build -t rag-toolkit -f docker/Dockerfile .

docker-run:
	docker compose up --build

dvc-init:
	dvc init -f --no-scm