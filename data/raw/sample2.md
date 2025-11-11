# RAG Pipeline

The RAG pipeline includes:
- Document loaders for txt, md, pdf
- Character-based chunker
- Sentence-transformers embeddings
- FAISS IndexFlatIP (cosine via normalization)
- Retrieval and optional LLM answer