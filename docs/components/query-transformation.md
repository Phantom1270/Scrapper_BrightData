# Query Transformation

The **doc//rag** system uses an LLM-powered Query Transformation step to maximize retrieval surface area before executing the search. This is handled by the `MultiQueryTransformer` class.

## 1. The Problem
Users often input short, poorly phrased, or ambiguous search queries. A direct vector embedding of a bad query will fail to retrieve highly relevant documents that use different terminology (e.g., querying "how to save model" when the documentation uses "serialization and persistence").

## 2. Implementation

Before querying ChromaDB or BM25, the `RetrievalEngine` intercepts the user's question and passes it to the `MultiQueryTransformer`.

1. **Prompting the LLM:** The transformer prompts a local LLM (default: `qwen2.5:3b` via Ollama) to generate alternative versions of the question from different perspectives.
2. **Execution:** 
   - The user query: "How do I save a model?"
   - LLM Variant 1: "What are the methods for model serialization in the framework?"
   - LLM Variant 2: "Saving, exporting, and persisting trained estimator objects."
3. **Multi-Search:** The `RetrievalEngine` executes a search for *all three* variations (the original + variants) concurrently across both the Dense and Sparse indices.

## 3. De-duplication
Because multiple queries are executed against the same indices, they often return the same highly relevant documents. The `RetrievalEngine` aggregates all results and deduplicates them by `chunk_id` before passing the unique candidate list to the Reciprocal Rank Fusion step.

## 4. Trade-offs
- **Advantages:** Drastically improves recall, especially for novice users who don't know the exact terminology used in the documentation.
- **Disadvantages:** Adds latency (calling the LLM before search) and increases search compute (executing 3x searches). This can be toggled off via `use_query_transform: false` in the API payload.
