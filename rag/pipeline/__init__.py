"""
RAG Data Pipeline — Phase 4.2

Orchestrates: schema discovery → normalization → cleaning → deduplication → storage.
"""

from rag.pipeline.data_pipeline import DataPipeline
from rag.pipeline.normalizer import UniversalNormalizer
from rag.pipeline.schema_discovery import SchemaDiscovery
from rag.pipeline.cleaner import ContentCleaner
from rag.pipeline.deduplicator import DocumentDeduplicator
from rag.pipeline.field_classifier import FieldClassifier

__all__ = [
    "DataPipeline",
    "UniversalNormalizer",
    "SchemaDiscovery",
    "ContentCleaner",
    "DocumentDeduplicator",
    "FieldClassifier",
]
