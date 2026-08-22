"""
Hypothetical Document Embeddings (HyDE).
"""

from typing import List
import logging
from rag.retrieval.query_transform.base import BaseQueryTransformer
from rag.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


class HyDETransformer(BaseQueryTransformer):
    """Generates a hypothetical document passage to answer the query."""

    def __init__(self, settings=None, llm_client: BaseLLMClient = None):
        # Backward compatibility for HyDETransformer(llm_client)
        if hasattr(settings, "chat") or (hasattr(settings, "__class__") and "Client" in settings.__class__.__name__) or (hasattr(settings, "_mock_name")):
            llm_client = settings
            settings = None

        if llm_client is None:
            from rag.llm import create_llm_client
            llm_client = create_llm_client(settings)
            
        self.llm_client = llm_client

    def transform(self, query: str) -> List[str]:
        """
        Generate a hypothetical document passage.
        Returns just the hypothetical document as the query.
        """
        if not query or not query.strip():
            return [query]
            
        system_prompt = (
            "Write a short, factual documentation passage that would "
            "answer this question. Write as if it's from official documentation. "
            "Be specific and include technical details. Keep it under 200 words. "
            "Do not include any preamble like 'Here is...' — just the passage."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        try:
            response = self.llm_client.chat(messages, temperature=0.1, max_tokens=300)
            
            hypothetical_doc = response.strip()
            if not hypothetical_doc:
                return [query]
                
            return [hypothetical_doc]
            
        except Exception as e:
            logger.warning("HyDETransformer failed: %s. Using original query.", e)
            return [query]

    def get_transform_name(self) -> str:
        return "hyde"
