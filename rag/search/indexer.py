"""
Orchestrates building all indexes from stored chunks.
"""

import os
import json
import logging
import time
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class IndexBuildResult:
    total_chunks: int
    chunks_embedded: int
    embedding_model: str
    embedding_dimension: int
    vector_store_count: int
    bm25_count: int
    skipped: bool = False
    incremental: bool = False
    processing_time_seconds: float = 0.0
    manifest_path: str = ""


class IndexBuilder:
    """Reads all chunks from storage and populates vector store and BM25 index."""

    def __init__(self, settings=None, store=None, embedder=None, vector_store=None, bm25_index=None, search_engine=None):
        if settings is None:
            from rag.config.settings import get_settings
            settings = get_settings()
        self.settings = settings
        self.search_engine = search_engine

        if store is None:
            if search_engine and search_engine.store:
                store = search_engine.store
            else:
                from rag.storage.sqlite_store import SQLiteStore
                import os
                db_path = os.path.join(settings.general.data_dir, "rag.db")
                store = SQLiteStore(db_path)
        self.store = store

        if embedder is None:
            if search_engine and search_engine.embedder:
                embedder = search_engine.embedder
            else:
                from rag.search.embeddings.sentence_transformer import SentenceTransformerEmbedder
                embedder = SentenceTransformerEmbedder(settings)
        self.embedder = embedder

        if vector_store is None:
            if search_engine and search_engine.vector_store:
                vector_store = search_engine.vector_store
            else:
                from rag.search.vector_store.chroma_store import ChromaVectorStore
                vector_store = ChromaVectorStore(settings)
        self.vector_store = vector_store

        if bm25_index is None:
            if search_engine and search_engine.bm25_index:
                bm25_index = search_engine.bm25_index
            else:
                from rag.search.bm25.bm25_index import BM25Index
                bm25_index = BM25Index(settings)
        self.bm25_index = bm25_index

    def load_existing(self) -> bool:
        """Try to load existing BM25 index from disk."""
        return self.bm25_index.load(self.settings.bm25.index_path)

    def _get_manifest_path(self) -> str:
        return os.path.join(self.settings.general.data_dir, "indexes", "manifest.json")

    def _save_manifest(self, force_rebuild: bool, chunks_count: int) -> str:
        manifest_path = self._get_manifest_path()
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        manifest = {
            "built_at": datetime.utcnow().isoformat() + "Z",
            "total_chunks": chunks_count,
            "embedding_model": self.embedder.get_model_name(),
            "embedding_dimension": self.embedder.get_dimension(),
            "vector_store_provider": type(self.vector_store).__name__,
            "bm25_chunk_count": self.bm25_index.count(),
            "force_rebuild": force_rebuild,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return manifest_path

    def _create_metadata(self, chunk) -> dict:
        """Create flattened metadata for vector store."""
        meta = {
            "doc_id": chunk.doc_id,
            "url": chunk.url,
            "content_type": chunk.content_type,
            "heading_path": " > ".join(chunk.heading_path),
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
            "block_type": chunk.block_type,
            "language": chunk.language,
        }
        
        # Merge additional metadata, converting complex types to string
        if chunk.metadata:
            for k, v in chunk.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                else:
                    meta[k] = str(v)
                    
        return meta

    def build_all(self, force_rebuild: bool = False) -> IndexBuildResult:
        """Full index build."""
        start_time = time.time()
        
        chunks = self.store.get_all_chunks()
        if not chunks:
            return IndexBuildResult(0, 0, self.embedder.get_model_name(), self.embedder.get_dimension(), 0, 0)
            
        if not force_rebuild:
            self.load_existing()
            if self.vector_store.count() > 0 and self.bm25_index.count() > 0:
                logger.info("Indexes already exist. Use force_rebuild=True to rebuild.")
                return IndexBuildResult(
                    total_chunks=len(chunks),
                    chunks_embedded=0,
                    embedding_model=self.embedder.get_model_name(),
                    embedding_dimension=self.embedder.get_dimension(),
                    vector_store_count=self.vector_store.count(),
                    bm25_count=self.bm25_index.count(),
                    skipped=True,
                    incremental=False
                )

        # Clear existing
        self.vector_store.delete_all()
        self.bm25_index.clear()

        # Prepare for embedding
        texts = [chunk.content_with_heading for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [self._create_metadata(chunk) for chunk in chunks]

        # Embed all
        embeddings = self.embedder.embed_documents_batched(texts, show_progress=True)

        # Add to vector store
        self.vector_store.add_chunks(ids, embeddings, texts, metadatas)

        # Build BM25
        self.bm25_index.build(ids, texts)
        self.bm25_index.save(self.settings.bm25.index_path)

        # Save manifest
        manifest_path = self._save_manifest(force_rebuild=force_rebuild, chunks_count=len(chunks))

        processing_time = time.time() - start_time
        return IndexBuildResult(
            total_chunks=len(chunks),
            chunks_embedded=len(embeddings),
            embedding_model=self.embedder.get_model_name(),
            embedding_dimension=self.embedder.get_dimension(),
            vector_store_count=self.vector_store.count(),
            bm25_count=self.bm25_index.count(),
            skipped=False,
            incremental=False,
            processing_time_seconds=processing_time,
            manifest_path=manifest_path
        )

    def build_incremental(self, new_chunk_ids: List[str] = None) -> IndexBuildResult:
        """Add only new/changed chunks to existing indexes."""
        start_time = time.time()
        
        self.load_existing()
        
        if new_chunk_ids is None:
            # Detect new chunks
            all_chunks = self.store.get_all_chunks()
            store_ids = {c.chunk_id for c in all_chunks}
            indexed_ids = set(self.bm25_index.chunk_ids)
            new_chunk_ids = list(store_ids - indexed_ids)
            
        if not new_chunk_ids:
            return IndexBuildResult(
                total_chunks=self.vector_store.count(),
                chunks_embedded=0,
                embedding_model=self.embedder.get_model_name(),
                embedding_dimension=self.embedder.get_dimension(),
                vector_store_count=self.vector_store.count(),
                bm25_count=self.bm25_index.count(),
                skipped=True,
                incremental=True
            )
            
        new_chunks = [self.store.get_chunk(cid) for cid in new_chunk_ids if self.store.get_chunk(cid) is not None]
        
        if not new_chunks:
            return IndexBuildResult(
                total_chunks=self.vector_store.count(),
                chunks_embedded=0,
                embedding_model=self.embedder.get_model_name(),
                embedding_dimension=self.embedder.get_dimension(),
                vector_store_count=self.vector_store.count(),
                bm25_count=self.bm25_index.count(),
                skipped=True,
                incremental=True
            )

        texts = [c.content_with_heading for c in new_chunks]
        ids = [c.chunk_id for c in new_chunks]
        metadatas = [self._create_metadata(c) for c in new_chunks]

        # Embed new chunks
        embeddings = self.embedder.embed_documents_batched(texts, show_progress=True)

        # Add to vector store
        self.vector_store.add_chunks(ids, embeddings, texts, metadatas)

        # Add to BM25
        self.bm25_index.add_chunks(ids, texts)
        self.bm25_index.save(self.settings.bm25.index_path)

        # Update manifest
        manifest_path = self._save_manifest(force_rebuild=False, chunks_count=self.vector_store.count())

        processing_time = time.time() - start_time
        return IndexBuildResult(
            total_chunks=self.vector_store.count(),
            chunks_embedded=len(embeddings),
            embedding_model=self.embedder.get_model_name(),
            embedding_dimension=self.embedder.get_dimension(),
            vector_store_count=self.vector_store.count(),
            bm25_count=self.bm25_index.count(),
            skipped=False,
            incremental=True,
            processing_time_seconds=processing_time,
            manifest_path=manifest_path
        )
