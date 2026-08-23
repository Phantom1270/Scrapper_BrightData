# Hybrid Search & RRF Fusion

The **doc//rag** system combines Dense and Sparse retrieval, fused mathematically using Reciprocal Rank Fusion, orchestrated by `rag.search.search_engine.SearchEngine` and `rag.retrieval.fusion.ReciprocalRankFusion`.

## 1. Dense Retrieval (ChromaVectorStore)
- **Model:** `SentenceTransformerEmbedder` using `all-MiniLM-L6-v2`.
- **Mechanism:** Projects queries and chunks into a continuous vector space and calculates cosine similarity.
- **Strengths:** Excellent at semantic matching (e.g., matching "how to initialize" with "setup guidelines").

## 2. Sparse Retrieval (BM25Index)
- **Mechanism:** Lexical keyword matching using the BM25 algorithm. 
- **Strengths:** Excellent for exact identifier matches, error codes, and specific API parameter names which dense models often blur together.

## 3. Reciprocal Rank Fusion (RRF)
The `RetrievalEngine` queries both systems in parallel and passes the `RetrievalResult` lists to `ReciprocalRankFusion.fuse_vector_and_bm25()`.

### Mathematical Basis
RRF operates on the *rank* of the document, not the absolute score, circumventing the issue of normalizing fundamentally different score distributions (cosine distance vs BM25 unbounded scores).

The implementation uses the exact formula:

$RRF_{score} = weight \times \frac{1}{k + rank + 1}$

Where:
- $k$ is a smoothing constant, strictly configured to `60` (`settings.retrieval.rrf_k`).
- $rank$ is the 0-indexed position of the document in the result list.
- $weight$ is the configured system weight: `vector_weight = 0.6`, `bm25_weight = 0.4`.

### Worked Example
Suppose a document `chunk_A` is retrieved by both systems:
- Dense Rank: 2nd place ($rank = 1$)
- BM25 Rank: 5th place ($rank = 4$)

**Calculation:**
- Dense Contribution: $0.6 \times \frac{1}{60 + 1 + 1} = 0.6 \times \frac{1}{62} \approx 0.00967$
- BM25 Contribution: $0.4 \times \frac{1}{60 + 4 + 1} = 0.4 \times \frac{1}{65} \approx 0.00615$
- **Total Fused Score:** $0.00967 + 0.00615 = 0.01582$

The candidates are then sorted by this fused score in descending order, ensuring that documents ranking highly in *both* systems float to the absolute top.
