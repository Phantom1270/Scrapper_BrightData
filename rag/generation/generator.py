"""
Main Orchestrator for the Generation Engine.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import time
import logging

from rag.models.query import QueryRequest, QueryResponse
from rag.models.retrieval import RetrievalResult
from rag.generation.prompts import PromptBuilder
from rag.generation.context_builder import ContextBuilder
from rag.generation.citation import CitationExtractor

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    answer: str
    sources: List[dict]
    citations: List[dict]
    confidence: str
    retrieval_time_ms: float
    generation_time_ms: float
    chunks_retrieved: int
    chunks_used_in_context: int
    llm_model: str
    transform_used: Optional[str] = None
    reranker_used: Optional[str] = None


class GenerationEngine:
    """Takes retrieved chunks and generates a grounded answer."""

    def __init__(self, settings=None, retrieval_engine=None,
                 llm_client=None, prompt_builder=None,
                 context_builder=None, citation_extractor=None):
                 
        if settings is None:
            from rag.config.settings import get_settings
            settings = get_settings()

        if retrieval_engine is None:
            from rag.retrieval.retrieval_engine import RetrievalEngine
            self.retrieval_engine = RetrievalEngine(settings)
        else:
            self.retrieval_engine = retrieval_engine
            
        if llm_client is None:
            from rag.llm import create_llm_client
            self.llm_client = create_llm_client(settings)
        else:
            self.llm_client = llm_client
            
        if prompt_builder is None:
            self.prompt_builder = PromptBuilder(settings)
        else:
            self.prompt_builder = prompt_builder
            
        if context_builder is None:
            self.context_builder = ContextBuilder(settings)
        else:
            self.context_builder = context_builder
            
        if citation_extractor is None:
            self.citation_extractor = CitationExtractor()
        else:
            self.citation_extractor = citation_extractor

        self.temperature = settings.generation.temperature
        self.max_tokens = settings.generation.max_tokens

    def generate(self, query: str = None,
                 request: QueryRequest = None,
                 stream: bool = False) -> GenerationResult:
        """Full generation pipeline."""
        
        # 1. Extract question
        if request is not None:
            question = request.question
        else:
            question = query
            
        if not question:
            raise ValueError("Query string cannot be empty")

        # 2. Retrieve
        t0 = time.time()
        results = self.retrieval_engine.retrieve(query=query, request=request)
        retrieval_time_ms = (time.time() - t0) * 1000
        
        # Get metadata about retrieval
        transform_used = None
        reranker_used = None
        if hasattr(self.retrieval_engine, 'query_transformer') and getattr(self.retrieval_engine, 'use_query_transform', False):
            if self.retrieval_engine.query_transformer:
                transform_used = self.retrieval_engine.query_transformer.get_transform_name()
                
        use_reranking = request.use_reranking if request else True
        if hasattr(self.retrieval_engine, 'reranker') and self.retrieval_engine.reranker and use_reranking:
            reranker_used = self.retrieval_engine.reranker.get_model_name()

        # 3. Assess confidence
        confidence = self.prompt_builder.assess_confidence(results)
        
        # 4. Build context
        if confidence != "none":
            context = self.context_builder.build_context(results)
            usage = self.context_builder.estimate_token_usage(results)
            chunks_used = usage["chunks_included"] + usage["chunks_truncated"]
        else:
            context = ""
            chunks_used = 0
            
        # 5. Build prompt
        messages = self.prompt_builder.build_messages(
            query=question,
            context=context,
            has_context=bool(context),
            confidence=confidence,
        )

        # 6. Call LLM
        t1 = time.time()
        
        if stream:
            answer = self._generate_stream(messages)
        else:
            answer = self.llm_client.chat(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
        generation_time_ms = (time.time() - t1) * 1000

        # 7. Extract citations
        citations = self.citation_extractor.extract_citations(answer, results)
        
        # 8. Format sources
        sources = [
            {
                "chunk_id": r.chunk_id,
                "heading": " > ".join(r.heading_path) if isinstance(r.heading_path, list) else "",
                "url": r.url,
                "score": round(r.score, 4) if isinstance(r.score, (int, float)) else r.score,
                "content_type": r.content_type,
                "source": r.source,
            }
            for r in results[:5]
        ]
        
        return GenerationResult(
            answer=answer,
            sources=sources,
            citations=citations,
            confidence=confidence,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms,
            chunks_retrieved=len(results),
            chunks_used_in_context=chunks_used,
            llm_model=self.llm_client.get_model_name(),
            transform_used=transform_used,
            reranker_used=reranker_used
        )

    def _generate_stream(self, messages: List[dict]) -> str:
        """Streaming generation with fallback."""
        try:
            answer = ""
            for chunk in self.llm_client.chat_stream(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            ):
                print(chunk, end="", flush=True)
                answer += chunk
            print()
            return answer
        except AttributeError:
            # Fall back to non-streaming
            answer = self.llm_client.chat(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            print(answer)
            return answer

    def generate_and_format(self, query: str = None,
                            request: QueryRequest = None) -> QueryResponse:
        """Generate and format as QueryResponse."""
        result = self.generate(query=query, request=request)
        
        return QueryResponse(
            answer=result.answer,
            sources=result.sources,
            retrieval_time_ms=result.retrieval_time_ms,
            generation_time_ms=result.generation_time_ms,
            total_time_ms=result.retrieval_time_ms + result.generation_time_ms,
            cached=False,
        )
