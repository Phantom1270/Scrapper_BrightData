"""
Generation Engine components for producing grounded answers from retrieved context.
"""

from rag.generation.prompts import PromptBuilder, SYSTEM_PROMPT, NO_CONTEXT_PROMPT, LOW_CONFIDENCE_PROMPT
from rag.generation.context_builder import ContextBuilder
from rag.generation.citation import CitationExtractor
from rag.generation.generator import GenerationEngine, GenerationResult

__all__ = [
    "PromptBuilder",
    "SYSTEM_PROMPT",
    "NO_CONTEXT_PROMPT",
    "LOW_CONFIDENCE_PROMPT",
    "ContextBuilder",
    "CitationExtractor",
    "GenerationEngine",
    "GenerationResult",
]
