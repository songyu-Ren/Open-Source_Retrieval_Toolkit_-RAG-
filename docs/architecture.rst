Architecture
============

Overview
--------

The toolkit implements an end-to-end RAG pipeline:

- Loaders: read `.txt`, `.md`, `.pdf` with minimal metadata
- Chunker: character-based chunking with configurable overlap
- Embedder: sentence-transformers wrapper with normalized vectors
- Index: FAISS `IndexFlatIP` for cosine via normalization
- Retrieval: top-k search with scores and metadata
- LLM: optional OpenAI-compatible client or fallback stub
- Evaluation: nDCG@k and MRR, logs to MLflow
- Metrics: Prometheus counters/histograms for API endpoints

Modules
-------

.. automodule:: rag_toolkit.config
   :members:

.. automodule:: rag_toolkit.loaders
   :members:

.. automodule:: rag_toolkit.chunker
   :members:

.. automodule:: rag_toolkit.embedder
   :members:

.. automodule:: rag_toolkit.index_store
   :members:

.. automodule:: rag_toolkit.retrieval
   :members:

.. automodule:: rag_toolkit.llm
   :members:

.. automodule:: rag_toolkit.eval
   :members:

.. automodule:: rag_toolkit.metrics
   :members: