"""
Generates multiple query variations for broader retrieval.
"""

from typing import List
import logging
from rag.retrieval.query_transform.base import BaseQueryTransformer
from rag.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


class MultiQueryTransformer(BaseQueryTransformer):
    """Generates alternative phrasings of the user's query using an LLM."""

    def __init__(self, settings=None, llm_client: BaseLLMClient = None,
                 n_queries: int = 3):
        # Backward compatibility for MultiQueryTransformer(llm_client)
        if hasattr(settings, "chat") or (hasattr(settings, "__class__") and "Client" in settings.__class__.__name__) or (hasattr(settings, "_mock_name")):
            llm_client = settings
            settings = None

        if llm_client is None:
            from rag.llm import create_llm_client
            llm_client = create_llm_client(settings)
            
        self.llm_client = llm_client
        self.n_queries = n_queries

    def transform(self, query: str) -> List[str]:
        """
        Generate multiple search queries based on the original query.
        Always returns the original query as the first element.
        """
        if not query or not query.strip():
            return [query]
            
        system_prompt = (
            f"Generate {self.n_queries} different search queries that "
            "would help find documentation to answer the user's question.\n"
            "Return one query per line. No numbering. No explanation."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        try:
            response = self.llm_client.chat(messages, temperature=0.7, max_tokens=200)
            
            lines = [line.strip() for line in response.split("\n")]
            variations = [line for line in lines if line]
            
            # Remove numbering if the LLM ignored instructions
            cleaned_variations = []
            for var in variations:
                # Basic cleanup for common list formats like "1. ", "- ", etc.
                if var[0].isdigit() and (len(var) > 1 and var[1] in [".", ")"]):
                    var = var[2:].strip()
                elif var.startswith("- "):
                    var = var[2:].strip()
                elif var.startswith("* "):
                    var = var[2:].strip()
                cleaned_variations.append(var)
                
            return [query] + cleaned_variations[:self.n_queries]
            
        except Exception as e:
            logger.warning("MultiQueryTransformer failed: %s. Using original query.", e)
            return [query]

    def get_transform_name(self) -> str:
        return "multi_query"
