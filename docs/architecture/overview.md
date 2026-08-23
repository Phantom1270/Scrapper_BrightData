# Architecture Overview

## 1. System Goals
This system is designed to seamlessly ingest heterogeneous, schema-less JSON scraped by Bright Data tools and convert it into a highly precise RAG knowledge base. The primary goals are:
- **Decoupling**: Strictly separating scraping logic from the RAG pipeline (`UniversalNormalizer`).
- **Precision**: Removing near-duplicates that skew retrieval (`DocumentDeduplicator`).
- **Context Preservation**: Preserving document structure via `ParentChildBuilder`.
- **High Recall & Precision**: Achieving high recall through Hybrid Search and query expansion, and high precision via Cross-Encoder reranking.

## 2. High-Level Architecture Diagram

```mermaid
flowchart TD
    A[Raw Scraped JSON] --> B[UniversalNormalizer]
    B --> C[FieldClassifier]
    C --> B
    B --> D[DocumentDeduplicator]
    D --> E[ParentChildBuilder / Chunking]
    
    E --> F[ChromaVectorStore]
    E --> G[BM25Index]
    
    U[Client Request] --> API[FastAPI / POST api/v1/query]
    API --> RE[RetrievalEngine]
    
    RE --> QT[MultiQueryTransformer<br>qwen2.5:3b]
    QT --> H[SearchEngine]
    
    H --> F
    H --> G
    
    F --> RRF[ReciprocalRankFusion]
    G --> RRF
    
    RRF --> CE[CrossEncoderReranker<br>ms-marco-MiniLM-L-6-v2]
    CE --> CB[ContextBuilder]
    CB --> LLM[GenerationEngine]
    LLM --> API
```

## 3. Component Responsibilities
- **`UniversalNormalizer`**: Applies heuristic pattern matching (`FieldClassifier`) to map unknown JSON keys to known semantic blocks (e.g., `page_title` -> `title`).
- **`DocumentDeduplicator`**: Executes an O(n) pipeline (exact MD5 hash + MinHash/LSH) to purge duplicates.
- **`SearchEngine`**: Low-level interface interacting with `ChromaVectorStore` and `BM25Index`.
- **`RetrievalEngine`**: High-level orchestrator managing query transformation, routing queries to `SearchEngine`, applying `ReciprocalRankFusion`, and running `CrossEncoderReranker`.

## 4. Runtime Architecture
The FastAPI application (`rag/serving/app.py`) is initialized via a factory pattern. Crucially, the `SearchEngine` and `RetrievalEngine` are initialized at startup and injected as dependencies into the route handlers. This singleton behavior prevents reloading the embedding models (`SentenceTransformerEmbedder`), cross-encoders, and ChromaDB instances on every request, ensuring low latency.
