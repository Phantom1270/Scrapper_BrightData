# Evaluation & Performance

This repository implements a local evaluation framework within the `rag.evaluation` module to measure retrieval and generation accuracy.

## 1. Implemented Evaluation Functionality

The `rag.evaluation.evaluator.EvaluationRunner` orchestrates the evaluation pipeline. It currently tests two distinct phases:

1. **Retrieval Metrics (`RetrievalMetrics`)**: Measures if the system retrieved the correct chunks.
2. **Generation Metrics (`GenerationMetrics`)**: Measures if the generated answer is faithful to the retrieved context and answers the question.

### Note on RAGAS
While external RAG evaluation frameworks like RAGAS exist, this repository implements its own internal scoring mechanisms to avoid external dependencies. *It does not currently use RAGAS.*

## 2. Evaluation Structure

The `EvaluationRunner.evaluate_question()` method traces the following path:
1. Calls `RetrievalEngine.retrieve()` for the given question.
2. Evaluates the returned chunk IDs against the `expected_chunk_ids` using `RetrievalMetrics`.
3. Calls `GenerationEngine.generate()` for the full answer.
4. Evaluates the output answer against `expected_answer_keywords` and the retrieved context using `GenerationMetrics`.

Results are aggregated into an `EvaluationReport` tracking `total_questions`, `aggregate_metrics`, and segmented metrics by `category` and `difficulty`.

## 3. Complexity & Performance Theory

While hardcoded latency benchmarks are not present in the repository, the theoretical complexity of the system is tightly controlled:

- **Normalization**: `UniversalNormalizer` maps keys in `O(n)` time per document, with `O(1)` dict lookups for fields.
- **Deduplication**: `DocumentDeduplicator` avoids the `O(n²)` pairwise comparison trap by utilizing exact MD5 hashes and LSH MinHash, bringing near-duplicate detection down to roughly `O(n)` time.
- **Retrieval**: `ChromaVectorStore` utilizes HNSW (Hierarchical Navigable Small World) graphs internally for sub-linear `O(log n)` dense search. `BM25Index` is implemented for fast sparse retrieval.
- **Reranking**: `CrossEncoderReranker` is `O(K)` where `K` is the number of candidates passed by the hybrid search. Because cross-encoders compute self-attention across the query and document jointly, they are extremely computationally heavy. Limiting `candidate_k` (default 20) is crucial for performance.
