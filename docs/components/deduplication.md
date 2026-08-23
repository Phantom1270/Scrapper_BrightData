# Deduplication Engine

In the **doc//rag** system, the `DocumentDeduplicator` (`rag/pipeline/deduplicator.py`) is responsible for removing exact and near-duplicate documents prior to vector indexing. 

## 1. Algorithm Intuition
Web scraping frequently yields identical pages (exact duplicates) or pages with only slight dynamic variations like timestamps (near-duplicates). Indexing these pollutes the vector space, causing retrieval to return 5 identical chunks instead of 5 diverse sources. 

To solve this efficiently, we use a two-pass approach: Exact Hashing followed by MinHash + LSH (Locality Sensitive Hashing).

## 2. Implementation & Complexity

Naïve near-duplicate detection requires comparing every document against every other document: `O(n²)`. For 100,000 documents, this is computationally unfeasible. The implemented pipeline operates in roughly `O(n)` time by utilizing probabilistic hashing.

### Pass 1: Exact Hash Lookup
- **Implementation:** The `_full_text()` of a `NormalizedDocument` is encoded and hashed using `hashlib.md5()`.
- **Complexity:** String concatenation and MD5 hashing is `O(L)` where `L` is text length. For `n` documents, the pass is `O(n)`. Dictionary lookup is `O(1)`.

### Pass 2: MinHash + LSH
Implemented using the `datasketch` library.

1. **Shingle Generation:** The document text is tokenized into 3-word overlapping sequences (shingles).
2. **MinHash Signature Generation:** `NUM_PERM = 128` hash functions are applied to the shingles to create a dense signature. 
   - **Mathematical Basis:** The probability that the MinHash of set A equals the MinHash of set B is exactly their Jaccard Similarity.
   - $P[h_{min}(A) = h_{min}(B)] = J(A,B) = \frac{|A \cap B|}{|A \cup B|}$
3. **LSH Indexing:** Signatures are divided into bands. If two documents share identical signatures in at least one band, they become candidates.
   - **Configured Threshold:** The LSH threshold is set to `0.85` Jaccard similarity.
4. **Candidate Generation & Verification:** Instead of `O(n²)`, we only query the LSH index in `O(1)` time per document. If candidates are returned, the document is flagged as a duplicate.
5. **Final Selection:** The first occurrence of a document is kept, and subsequent near-duplicates are discarded.

- **Theoretical Complexity:** Shingle generation and MinHash updating takes `O(L * P)` where `P=128`. Querying the LSH index is `O(1)` on average. Overall complexity is `O(n)`.

## 3. Trade-offs
- **Advantages:** Scales linearly to millions of documents.
- **Disadvantages:** LSH is probabilistic, meaning a near-duplicate *might* theoretically evade detection (false negative), though the probability is extremely low with 128 permutations.
