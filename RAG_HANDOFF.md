# RAG Pipeline: Hand-Off & Architecture Overview

## 1. Project Status: COMPLETED
The core Retrieval-Augmented Generation (RAG) architecture is now **fully complete** and highly optimized for the current hardware specifications. It operates as an end-to-end pipeline capable of ingesting raw scraped data, processing it intelligently, and serving high-accuracy LLM answers via a REST API.

---

## 2. What Has Been Built
The pipeline is composed of the following production-ready components:

1. **Universal Normalizer:** A schema-agnostic ingestion engine. It uses fuzzy-matching to automatically classify fields from the scraper's JSON output (e.g., matching `desc` or `summary` to `description`) and converts them into standardized `NormalizedDocument` objects. It completely decouples the scraper from the RAG engine.
2. **High-Speed Deduplicator:** Replaced an O(n²) string matcher with an **O(n) MinHash + Exact Hash LSH** approach. It processes hundreds of documents in seconds, effectively purging exact and near-duplicates before they pollute the Vector Store.
3. **Context-Aware Chunker:** Intelligently splits documents into LLM-friendly chunks while respecting the boundaries of code blocks, parameter lists, and prose.
4. **Hybrid Search Engine:** Combines dense semantic search (ChromaDB + `all-MiniLM-L6-v2`) with sparse keyword matching (BM25). Results are mathematically fused using **Reciprocal Rank Fusion (RRF)** for maximum relevance.
5. **Advanced Retrieval Techniques:** 
   - **Query Transformation:** Uses a local LLM (`qwen2.5:3b`) to generate multiple alternative semantic variations of the user's query to maximize retrieval surface area.
   - **Cross-Encoder Reranking:** Re-scores the retrieved candidates using `cross-encoder/ms-marco-MiniLM-L-6-v2` to ensure the absolute best chunks make it to the final LLM prompt.
6. **API Serving:** A robust FastAPI server (`uvicorn`) that injects a shared `SearchEngine` singleton to prevent memory leaks and handle concurrent requests.

---

## 3. Suggestions for Future Improvement
While the current system is optimal for local hardware, scaling to an Enterprise level could benefit from these future additions:

* **Semantic Caching:** Implement a Redis/SQLite cache for queries. If a user asks a question with >95% semantic similarity to a previously answered question, return the cached result instantly (~50ms) instead of running the LLM.
* **Semantic Chunking:** Upgrade from token-based chunking to NLP-based chunking (e.g., using `spaCy`) to ensure text is only split at the end of complete sentences or paragraphs, preserving absolute context.
* **LLM-Assisted Ingestion:** Run scraped chunks through a fast LLM *before* saving them to generate rich metadata tags (e.g., `difficulty: beginner`, `framework: python`). This allows for incredibly powerful front-end filtering.
* **Automated Evaluation:** Integrate **RAGAS** (RAG Assessment) or TrueLens to programmatically score the pipeline's *Context Precision* and *Hallucination Rate* over a massive test dataset.

---

## 4. Next Phase: Frontend Integration
The next phase involves connecting the user interface to the RAG backend. 

### Server Details
* **Base URL:** `http://127.0.0.1:8000` (or wherever deployed)
* **Endpoint:** `POST /api/v1/query`
* **Headers:** `Content-Type: application/json`

### Inputs (For the Frontend Developer)
The frontend should send a JSON payload. Here are the fields available to expose in the UI:

```json
{
  "question": "How do I use sklearn config_context?", // REQUIRED: The user's search query
  
  // --- OPTIONAL FIELDS ---
  "top_k": 3,                             // How many sources to return/use (Default: 5)
  "filter_content_type": "api_reference", // Dropdown: "api_reference", "tutorial", "notebook"
  "use_query_transform": true,            // Toggle: "Use AI to expand search query?"
  "use_reranking": true                   // Toggle: "Enable Deep Reranking (Slower but accurate)"
}
```

### Outputs (For the Frontend Developer)
The API will return the following JSON structure. The UI should parse this to display the AI's answer alongside clickable citations.

```json
{
  "answer": "The purpose of sklearn.config_context is...",
  
  "sources": [
    {
      "chunk_id": "9bf961cfdf1ea8d1",
      "heading": "sklearn.config_context > Description",
      "url": "https://scikit-learn.org/stable/modules/...html",
      "score": 7.338,
      "content_type": "prose",
      "source": "hybrid"
    }
  ],
  
  "confidence": "high",              // Can be used to show a green/yellow/red trust indicator
  "retrieval_time_ms": 1619.3,       // Display as "Retrieved in 1.6s"
  "generation_time_ms": 7555.6,      // Display as "Generated in 7.5s"
  
  "transform_used": "MultiQueryTransformer",
  "reranker_used": "cross-encoder/ms-marco-MiniLM-L-6-v2"
}
```

### UI/UX Recommendations
1. **Streaming:** If you eventually implement WebSockets or Server-Sent Events (SSE), the `answer` field can be streamed to the UI so the user doesn't have to wait 7 seconds to see the first word.
2. **Citations:** Make the `sources[].url` clickable. You can also display the `heading` text as the hyperlink title.
3. **Advanced Toggles:** Hide `use_query_transform` and `use_reranking` behind an "Advanced Settings" gear icon to keep the main search bar clean.
