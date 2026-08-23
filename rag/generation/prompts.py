"""
Prompt templates and builder for the Generation Engine.
"""

from typing import List, Dict
from rag.models.retrieval import RetrievalResult

SYSTEM_PROMPT = """You are a technical documentation assistant.
Answer the user's question using ONLY the provided context.

Rules:
1. If the context contains the answer, provide it clearly with references.
2. If the context is insufficient or does not contain the answer, say:
   "I don't have enough information in the documentation to answer this
   fully." Do NOT make up information.
3. NEVER fabricate information not present in the context.
4. When referencing documentation, cite the source using [Source: Section Title].
5. If code examples are available in the context, include them in your answer.
6. If multiple approaches or options exist in the docs, mention all of them.

Format your answer:
- Direct answer first
- Supporting details and code examples if available
- Source references at the end
"""

NO_CONTEXT_PROMPT = """You are a technical documentation assistant.
The user asked a question but no relevant documentation was found.

Respond with: "I couldn't find relevant information in the documentation
for your question. Please check the official documentation or try
rephrasing your question."

Do NOT attempt to answer from your general knowledge.
"""

LOW_CONFIDENCE_PROMPT = """You are a technical documentation assistant.
The retrieved context may not fully answer the user's question.

Rules:
1. Answer what you can from the context.
2. Clearly state which parts of your answer come from the documentation
   and which parts you are uncertain about.
3. If the context is too vague or off-topic, say:
   "The available documentation doesn't directly address this question.
   Based on related information, here's what I found: ..."
4. Always cite your sources using [Source: Section Title].
"""


class PromptBuilder:
    """Builds prompt messages and assesses retrieval confidence."""

    def __init__(self, settings=None):
        if settings is None:
            from rag.config.settings import get_settings
            settings = get_settings()
            
        self.max_context_tokens = settings.generation.max_context_tokens
        self.encoding_name = settings.chunking.encoding_name

    def build_messages(self, query: str, context: str,
                       has_context: bool = True,
                       confidence: str = "high") -> List[Dict[str, str]]:
        """
        Build the messages list for the LLM.
        """
        if confidence == "none":
            system = NO_CONTEXT_PROMPT
            user = f"Question: {query}"
        else:
            system = SYSTEM_PROMPT if confidence == "high" else LOW_CONFIDENCE_PROMPT
            user = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]

    def assess_confidence(self, results: List[RetrievalResult]) -> str:
        """
        Assess retrieval confidence based on result scores.
        """
        if not results:
            return "none"

        # The first result should be the highest scoring one
        best_result = results[0]
        
        # Check if the source is reranked
        if best_result.source == "reranked":
            if best_result.score < 1.0:
                return "low"
            return "high"
            
        # Standard cosine similarity or hybrid scores
        if best_result.score < 0.3:
            return "low"
            
        if best_result.score < 0.5 and len(results) < 2:
            return "low"
            
        return "high"
