"""
RAG Ingest Script
=================
Runs the full pipeline on a Phase 3 scraped JSON file:
  1. Normalize   — JSON → NormalizedDocument objects
  2. Deduplicate — remove exact / near-duplicate documents
  3. Store       — save documents to SQLite
  4. Chunk       — split documents into chunks and save
  5. Index       — embed chunks into vector store + BM25

Usage:
    python ingest.py --input "phase3_output (2).json"
    python ingest.py --input "phase3_output (2).json" --force-rebuild
"""

import argparse
import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest Phase 3 JSON into RAG pipeline")
    parser.add_argument("--input", required=True, help="Path to phase3_output JSON file")
    parser.add_argument("--force-rebuild", action="store_true", default=True,
                        help="Force rebuild of vector + BM25 indexes (default: True)")
    parser.add_argument("--skip-dedup", action="store_true", default=False,
                        help="Skip deduplication step")
    args = parser.parse_args()

    t_start = time.time()

    # ──────────────────────────────────────────────
    # 0. Bootstrap pipeline
    # ──────────────────────────────────────────────
    logger.info("Bootstrapping pipeline...")
    from rag.bootstrap import create_pipeline
    pipeline = create_pipeline()
    store   = pipeline["store"]
    chunker = pipeline["chunker"]
    indexer = pipeline["indexer"]

    # ──────────────────────────────────────────────
    # 1. Normalize
    # ──────────────────────────────────────────────
    logger.info("Step 1/5: Normalizing %s ...", args.input)
    from rag.pipeline.normalizer import UniversalNormalizer
    normalizer = UniversalNormalizer()
    docs = normalizer.normalize_file(args.input)
    stats = normalizer.get_stats()
    logger.info(
        "  Normalized: %d entries → %d documents (errors: %d)",
        stats["total_entries"], stats["total_docs"], stats["total_errors"],
    )

    # Filter out error / empty docs
    valid_docs = [d for d in docs if not d.error and d.content_blocks]
    logger.info("  Valid documents (with content): %d", len(valid_docs))

    if not valid_docs:
        logger.error("No valid documents found. Check the input file.")
        sys.exit(1)

    # ──────────────────────────────────────────────
    # 2. Deduplicate
    # ──────────────────────────────────────────────
    if not args.skip_dedup:
        logger.info("Step 2/5: Deduplicating %d documents ...", len(valid_docs))
        from rag.pipeline.deduplicator import DocumentDeduplicator
        deduplicator = DocumentDeduplicator()
        valid_docs = deduplicator.deduplicate(valid_docs)
        d_stats = deduplicator.get_stats()
        logger.info(
            "  After dedup: %d documents (exact removed: %d, near removed: %d)",
            d_stats["output_count"], d_stats["exact_removed"], d_stats["near_removed"],
        )
    else:
        logger.info("Step 2/5: Deduplication skipped.")

    # ──────────────────────────────────────────────
    # 3. Store documents
    # ──────────────────────────────────────────────
    logger.info("Step 3/5: Saving %d documents to store ...", len(valid_docs))
    store.save_documents(valid_docs)
    logger.info("  Store now has %d documents.", store.count_documents())

    # ──────────────────────────────────────────────
    # 4. Chunk
    # ──────────────────────────────────────────────
    logger.info("Step 4/5: Chunking all documents ...")
    chunk_result = chunker.chunk_all_documents(save=True)
    logger.info(
        "  Chunks created: %d total (%d parents, %d children) across %d documents",
        chunk_result.total_chunks,
        chunk_result.total_parents,
        chunk_result.total_children,
        chunk_result.total_documents,
    )
    logger.info("  Store now has %d chunks.", store.count_chunks())

    # ──────────────────────────────────────────────
    # 5. Index
    # ──────────────────────────────────────────────
    logger.info("Step 5/5: Building vector + BM25 indexes (force_rebuild=%s) ...", args.force_rebuild)
    index_result = indexer.build_all(force_rebuild=args.force_rebuild)
    if index_result.skipped:
        logger.info("  Indexes already exist and were not rebuilt. Use --force-rebuild to rebuild.")
    else:
        logger.info(
            "  Indexed: %d chunks embedded | Vector store: %d | BM25: %d | Time: %.1fs",
            index_result.chunks_embedded,
            index_result.vector_store_count,
            index_result.bm25_count,
            index_result.processing_time_seconds,
        )

    # ──────────────────────────────────────────────
    # Done
    # ──────────────────────────────────────────────
    elapsed = time.time() - t_start
    logger.info("")
    logger.info("✓ Ingest complete in %.1f seconds.", elapsed)
    logger.info("  Documents : %d", store.count_documents())
    logger.info("  Chunks    : %d", store.count_chunks())
    logger.info("  Vectors   : %d", index_result.vector_store_count)
    logger.info("  BM25      : %d", index_result.bm25_count)
    logger.info("")
    logger.info("You can now query the RAG pipeline:")
    logger.info("  python -m uvicorn rag.serving.app:create_app --factory --host 127.0.0.1 --port 8000")
    logger.info("  curl -X POST http://127.0.0.1:8000/api/v1/query -H 'Content-Type: application/json'")
    logger.info("       -d '{\"question\": \"What is config_context?\"}'")


if __name__ == "__main__":
    main()
