"""
The main chunking engine.
"""

from typing import List, Optional
import time
import json
from dataclasses import dataclass, asdict

from rag.models.chunk import Chunk
from rag.models.document import NormalizedDocument
from rag.chunking.strategies.api_reference import ApiReferenceChunkingStrategy
from rag.chunking.strategies.tutorial import TutorialChunkingStrategy
from rag.chunking.strategies.notebook import NotebookChunkingStrategy
from rag.chunking.strategies.generic import GenericChunkingStrategy
from rag.chunking.heading_builder import HeadingBuilder
from rag.chunking.chunk_enricher import ChunkEnricher
from rag.chunking.parent_child import ParentChildBuilder
from rag.chunking.quality_auditor import ChunkQualityAuditor, ChunkAuditReport
from rag.storage.sqlite_store import SQLiteStore


@dataclass
class ChunkingResult:
    all_chunks: List[Chunk]
    total_documents: int
    total_chunks: int
    total_parents: int
    total_children: int
    by_strategy: dict
    audit: ChunkAuditReport
    processing_time_seconds: float


class ChunkingEngine:
    """Orchestrates strategy selection and the chunking pipeline."""

    def __init__(self, settings=None, store=None, config=None):
        if config is not None:
            settings = config
            
        if isinstance(settings, dict):
            self.config = dict(settings)
            resolved_settings = None
        else:
            if settings is None:
                from rag.config.settings import get_settings
                settings = get_settings()
            resolved_settings = settings
            self.config = {
                "max_tokens": settings.chunking.max_tokens,
                "min_tokens": settings.chunking.min_tokens,
                "overlap_tokens": settings.chunking.overlap_tokens,
                "encoding_name": settings.chunking.encoding_name,
                "parent_max_tokens": getattr(settings.chunking, "parent_max_tokens", 1500)
            }
            
        if store is None:
            if resolved_settings is None:
                from rag.config.settings import get_settings
                resolved_settings = get_settings()
            import os
            db_path = os.path.join(resolved_settings.general.data_dir, "rag.db")
            self.store = SQLiteStore(db_path=db_path)
        else:
            self.store = store

        self.strategies = {
            "api_reference": ApiReferenceChunkingStrategy(self.config),
            "tutorial": TutorialChunkingStrategy(self.config),
            "notebook": NotebookChunkingStrategy(self.config),
            "example": TutorialChunkingStrategy(self.config),
            "unknown": GenericChunkingStrategy(self.config),
        }
        
        self.heading_builder = HeadingBuilder()
        self.enricher = ChunkEnricher(self.config)
        self.parent_child_builder = ParentChildBuilder(self.config)
        self.auditor = ChunkQualityAuditor(self.config)

    def chunk_document(self, document: NormalizedDocument, use_parent_child: bool = False) -> List[Chunk]:
        """Chunk a single document."""
        strategy = self.strategies.get(document.content_type, self.strategies["unknown"])
        
        # Run strategy
        chunks = strategy.chunk(document)
        if not chunks:
            return []
            
        # Enrich
        chunks = self.enricher.enrich(chunks, document)
        
        # Parent-child
        if use_parent_child:
            parents, children = self.parent_child_builder.build(chunks, document)
            return parents + children
        else:
            return chunks

    def chunk_all_documents(self, use_parent_child: bool = False, save: bool = True) -> ChunkingResult:
        """Chunk all documents from the store."""
        start_time = time.time()
        
        if self.store is None:
            raise ValueError("Store is required to chunk all documents.")
            
        all_docs = self.store.get_all_documents()
        all_chunks = []
        by_strategy = {k: 0 for k in self.strategies.keys()}
        total_parents = 0
        total_children = 0
        
        for doc in all_docs:
            if doc.error:
                continue
                
            strategy_name = doc.content_type if doc.content_type in self.strategies else "unknown"
            by_strategy[strategy_name] += 1
            
            doc_chunks = self.chunk_document(doc, use_parent_child=use_parent_child)
            all_chunks.extend(doc_chunks)
            
            if use_parent_child:
                parents = sum(1 for c in doc_chunks if c.metadata.get("is_parent"))
                children = len(doc_chunks) - parents
                total_parents += parents
                total_children += children
                
        # Audit
        audit_report = self.auditor.audit(all_chunks)
        self.auditor.print_report(audit_report)
        
        # Save
        if save and all_chunks:
            self.store.save_chunks(all_chunks)
            
        processing_time = time.time() - start_time
        
        return ChunkingResult(
            all_chunks=all_chunks,
            total_documents=len(all_docs),
            total_chunks=len(all_chunks),
            total_parents=total_parents,
            total_children=total_children,
            by_strategy=by_strategy,
            audit=audit_report,
            processing_time_seconds=processing_time
        )

    def chunk_and_save_report(self, output_path: str, use_parent_child: bool = False) -> ChunkingResult:
        """Runs chunk_all_documents and saves audit report to output_path as JSON."""
        result = self.chunk_all_documents(use_parent_child=use_parent_child, save=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(result.audit), f, indent=2)
            
        return result
