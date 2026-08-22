"""
Retrieval Orchestrator.
"""

import logging
from typing import List, Optional, Dict, Any

from rag.models.retrieval import RetrievalResult
from rag.models.query import QueryRequest
from rag.search.search_engine import SearchEngine
from rag.retrieval.fusion import ReciprocalRankFusion
from rag.retrieval.filter_builder import MetadataFilterBuilder
from rag.retrieval.reranker.base import BaseReranker
from rag.retrieval.query_transform.base import BaseQueryTransformer

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Main orchestrator for RAG retrieval."""

    def __init__(self, settings=None, search_engine=None,
                 reranker=None, query_transformer=None,
                 fusion=None, filter_builder=None):
                 
        if settings is None:
            from rag.config.settings import get_settings
            settings = get_settings()

        # Dependencies
        if search_engine is None:
            self.search_engine = SearchEngine(settings)
        else:
            self.search_engine = search_engine

        if reranker is None:
            if getattr(settings.reranker, 'enabled', False):
                from rag.retrieval.reranker.cross_encoder import CrossEncoderReranker
                self.reranker = CrossEncoderReranker(settings)
            else:
                self.reranker = None
        else:
            self.reranker = reranker

        self.query_transformer = query_transformer
        
        if fusion is None:
            self.fusion = ReciprocalRankFusion(k=getattr(settings.retrieval, 'rrf_k', 60))
        else:
            self.fusion = fusion
            
        if filter_builder is None:
            self.filter_builder = MetadataFilterBuilder(settings)
        else:
            self.filter_builder = filter_builder

        # Settings
        self.top_k = getattr(settings.retrieval, 'top_k', 5)
        self.candidate_k = getattr(settings.retrieval, 'candidate_k', 20)
        self.vector_weight = getattr(settings.retrieval, 'vector_weight', 0.6)
        self.bm25_weight = getattr(settings.retrieval, 'bm25_weight', 0.4)
        
        self.use_query_transform = getattr(settings.retrieval, 'use_query_transform', False)

    def set_query_transformer(self, transformer: BaseQueryTransformer) -> None:
        """Set or replace the query transformer."""
        self.query_transformer = transformer

    def set_reranker(self, reranker: BaseReranker) -> None:
        """Set or replace the reranker."""
        self.reranker = reranker

    def retrieve(self, query: str = None,
                 request: QueryRequest = None) -> List[RetrievalResult]:
        """Main retrieval method."""
        # 1. Extract parameters
        if request is not None:
            question = request.question
            top_k = request.top_k
            content_type_filter = request.filter_content_type
            doc_id_filter = request.filter_doc_id
            use_reranking = request.use_reranking
        else:
            question = query
            top_k = self.top_k
            content_type_filter = None
            doc_id_filter = None
            use_reranking = True

        if not question:
            raise ValueError("Query string cannot be empty")

        # 2. Build metadata filters
        if content_type_filter is None:
            auto_filters = self.filter_builder.build_filters(question)
            content_type_filter = auto_filters.get("content_type")

        # 3. Query transformation
        queries = [question]
        if self.use_query_transform and self.query_transformer is not None:
            try:
                queries = self.query_transformer.transform(question)
            except Exception as e:
                logger.warning(f"Query transformation failed: {e}. Using original query.")
                queries = [question]

        # 4. Hybrid search across all queries
        all_vector_results = []
        all_bm25_results = []
        
        for q in queries:
            vec, bm25 = self.search_engine.search_both(
                q,
                vector_top_k=self.candidate_k,
                bm25_top_k=self.candidate_k,
            )
            all_vector_results.extend(vec)
            all_bm25_results.extend(bm25)

        # Deduplicate by chunk_id keeping highest score
        def deduplicate(results: List[RetrievalResult]) -> List[RetrievalResult]:
            best = {}
            for r in results:
                if r.chunk_id not in best or r.score > best[r.chunk_id].score:
                    best[r.chunk_id] = r
            return list(best.values())

        unique_vector = deduplicate(all_vector_results)
        unique_bm25 = deduplicate(all_bm25_results)

        # 5. RRF Fusion
        fused = self.fusion.fuse_vector_and_bm25(
            unique_vector, unique_bm25,
            vector_weight=self.vector_weight,
            bm25_weight=self.bm25_weight
        )

        # 6. Build candidate set
        # Build dictionary for O(1) lookup
        original_results = {r.chunk_id: r for r in unique_vector + unique_bm25}
        
        candidates = []
        for chunk_id, fused_score in fused[:self.candidate_k]:
            if chunk_id in original_results:
                original_result = original_results[chunk_id]
                candidates.append(
                    RetrievalResult(
                        chunk_id=chunk_id,
                        content=original_result.content,
                        url=original_result.url,
                        heading_path=original_result.heading_path,
                        content_type=original_result.content_type,
                        score=fused_score,
                        source="hybrid",
                        metadata=original_result.metadata
                    )
                )
            else:
                # Fallback to fetching directly
                chunk_data = self.search_engine.get_chunk_content(chunk_id)
                if chunk_data:
                    candidates.append(
                        RetrievalResult(
                            chunk_id=chunk_id,
                            content=chunk_data.get("content", ""),
                            url=chunk_data.get("metadata", {}).get("url", ""),
                            heading_path=chunk_data.get("metadata", {}).get("heading_path", []),
                            content_type=chunk_data.get("metadata", {}).get("content_type", "unknown"),
                            score=fused_score,
                            source="hybrid",
                            metadata=chunk_data.get("metadata", {})
                        )
                    )

        # 7. Re-ranking
        if use_reranking and self.reranker is not None:
            try:
                candidates = self.reranker.rerank(question, candidates, top_k=top_k)
            except Exception as e:
                logger.warning(f"Re-ranking failed: {e}. Using hybrid results.")
                candidates = candidates[:top_k]
        else:
            candidates = candidates[:top_k]

        # 8. Post-retrieval filtering
        if content_type_filter is not None:
            filtered = []
            for c in candidates:
                if c.source == "reranked" or c.content_type == content_type_filter:
                    filtered.append(c)
            candidates = filtered

        if doc_id_filter is not None:
            candidates = [c for c in candidates if c.metadata.get("doc_id") == doc_id_filter]

        return candidates[:top_k]

    def retrieve_and_format(self, query: str = None,
                            request: QueryRequest = None) -> Dict[str, Any]:
        """Retrieve and format results for Generation Engine."""
        question = request.question if request else query
        
        results = self.retrieve(query=query, request=request)
        
        content_type_filter = request.filter_content_type if request else None
        doc_id_filter = request.filter_doc_id if request else None
        
        if content_type_filter is None:
            auto_filters = self.filter_builder.build_filters(question)
            content_type_filter = auto_filters.get("content_type")

        transform_used = None
        if self.query_transformer and self.use_query_transform:
            transform_used = self.query_transformer.get_transform_name()
            
        use_reranking = request.use_reranking if request else True
        reranker_used = None
        if self.reranker and use_reranking:
            reranker_used = self.reranker.get_model_name()

        chunks = []
        for r in results:
            chunks.append({
                "chunk_id": r.chunk_id,
                "content": r.content,
                "heading": " > ".join(r.heading_path) if isinstance(r.heading_path, list) else "",
                "url": r.url,
                "score": round(r.score, 4),
                "source": r.source,
                "content_type": r.content_type,
            })
            
        return {
            "query": question,
            "chunks": chunks,
            "filters_applied": {
                "content_type": content_type_filter,
                "doc_id": doc_id_filter,
            },
            "transform_used": transform_used,
            "reranker_used": reranker_used,
        }
