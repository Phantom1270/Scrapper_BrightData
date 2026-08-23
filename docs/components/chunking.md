# Parent-Child Chunking

The repository implements a hierarchical chunking strategy via the `ParentChildBuilder` (`rag/chunking/parent_child.py`).

## 1. Intuition
Traditional fixed-size chunking (e.g., breaking text every 500 tokens) causes severe context fragmentation. An LLM might retrieve a chunk containing an answer, but lack the surrounding document context needed to make sense of it.

Parent-Child chunking solves this by decoupling the *retrieval unit* (the child) from the *generation unit* (the parent). We embed and search against granular child chunks for high precision, but pass the broader parent chunk to the LLM for high context.

## 2. Implementation

The `ParentChildBuilder.build()` method executes the following sequence:

1. **Section Grouping:** Chunks are grouped by their structural boundaries using `chunk.heading_path[:2]`. This ensures that a parent chunk logically bounds a specific document section.
2. **Recombination:** The child chunks within a group are concatenated back into a single string.
3. **Parent Splitting:** If the combined section exceeds `parent_max_tokens` (default: 1500 tokens), it is split using `GenericChunkingStrategy._split_text_by_tokens`.
4. **Metadata Linking:** 
   - Parent chunks are generated with metadata: `{"is_parent": True, "child_chunk_ids": [...]}`
   - Child chunks are updated to hold a reference: `child.parent_chunk_id = group_parents[0].chunk_id`

## 3. Trade-offs
- **Advantages:** Retrieves highly specific hits while feeding the LLM massive, unbroken contextual blocks. Greatly reduces hallucination.
- **Disadvantages:** Consumes significantly more tokens during the generation phase, increasing API costs and latency. Requires maintaining two interconnected sets of chunks in storage.
