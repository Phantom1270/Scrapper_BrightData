"""
Raw search engine primitives for Vector and BM25 search.
"""

from typing import List, Tuple, Optional, Dict
from rag.models.retrieval import RetrievalResult


class SearchEngine:
    """Low-level search primitives."""

    def __init__(self, settings=None, store=None, embedder=None, vector_store=None, bm25_index=None):
        if settings is None:
            from rag.config.settings import get_settings
            settings = get_settings()
        self.settings = settings

        if store is None:
            from rag.storage.sqlite_store import SQLiteStore
            import os
            db_path = os.path.join(settings.general.data_dir, "rag.db")
            store = SQLiteStore(db_path)
        self.store = store

        if embedder is None:
            from rag.search.embeddings.sentence_transformer import SentenceTransformerEmbedder
            embedder = SentenceTransformerEmbedder(settings)
        self.embedder = embedder

        if vector_store is None:
            from rag.search.vector_store.chroma_store import ChromaVectorStore
            vector_store = ChromaVectorStore(settings)
        self.vector_store = vector_store

        if bm25_index is None:
            from rag.search.bm25.bm25_index import BM25Index
            bm25_index = BM25Index()
            bm25_index.load(settings.bm25.index_path)
        self.bm25_index = bm25_index

    def get_chunk_content(self, chunk_id: str) -> Optional[Dict]:
        """Helper to look up a chunk's full data."""
        # Try vector store first
        result = self.vector_store.get_chunk_by_id(chunk_id)
        if result and result.get("document"):
            return result
            
        # Fall back to sqlite store
        chunk = self.store.get_chunk(chunk_id)
        if chunk:
            return {
                "document": chunk.content,
                "metadata": {
                    "url": chunk.url,
                    "heading_path": " > ".join(chunk.heading_path) if chunk.heading_path else "",
                    "content_type": chunk.content_type,
                    "block_type": chunk.block_type
                }
            }
            
        return None

    def vector_search(self, query: str, top_k: int = None, content_type_filter: str = None, doc_id_filter: str = None) -> List[RetrievalResult]:
        """Vector search against the store."""
        if top_k is None:
            top_k = self.settings.retrieval.top_k
            
        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            content_type_filter=content_type_filter,
            doc_id_filter=doc_id_filter
        )
        
        retrieval_results = []
        for r in results:
            meta = r.get("metadata", {})
            heading_str = meta.get("heading_path", "")
            heading_path = [h.strip() for h in heading_str.split(">")] if heading_str else []
            
            rr = RetrievalResult(
                chunk_id=r["id"],
                content=r["document"],
                url=meta.get("url", ""),
                heading_path=heading_path,
                content_type=meta.get("content_type", ""),
                score=r["score"],
                source="vector",
                metadata=meta
            )
            retrieval_results.append(rr)
            
        return retrieval_results

    def bm25_search(self, query: str, top_k: int = None) -> List[RetrievalResult]:
        """Keyword search using BM25."""
        if top_k is None:
            top_k = self.settings.retrieval.candidate_k
            
        raw_results = self.bm25_index.search(query, top_k=top_k)
        
        retrieval_results = []
        for chunk_id, score in raw_results:
            data = self.get_chunk_content(chunk_id)
            if not data:
                continue
                
            meta = data.get("metadata", {})
            heading_str = meta.get("heading_path", "")
            heading_path = [h.strip() for h in heading_str.split(">")] if heading_str else []
            
            rr = RetrievalResult(
                chunk_id=chunk_id,
                content=data.get("document", ""),
                url=meta.get("url", ""),
                heading_path=heading_path,
                content_type=meta.get("content_type", ""),
                score=score,
                source="bm25",
                metadata=meta
            )
            retrieval_results.append(rr)
            
        return retrieval_results

    def search_both(self, query: str, vector_top_k: int = None, bm25_top_k: int = None) -> Tuple[List[RetrievalResult], List[RetrievalResult]]:
        """Run both search methods simultaneously."""
        v_results = self.vector_search(query, top_k=vector_top_k)
        b_results = self.bm25_search(query, top_k=bm25_top_k)
        return v_results, b_results
