"""
OpenAI API backend stub.
"""

from typing import List, Dict, Optional
from rag.llm.base import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    """OpenAI API backend."""

    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI, OpenAIError
            self.OpenAIError = OpenAIError
        except ImportError:
            raise ImportError(
                "openai package required for OpenAI backend. "
                "Install with: pip install openai"
            )
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.1,
             max_tokens: int = 300) -> str:
        """Send a chat completion request to OpenAI."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            # More specific error handling depending on the error type
            error_msg = str(e).lower()
            if "authentication" in error_msg or "incorrect api key" in error_msg:
                raise RuntimeError("Invalid OpenAI API key") from e
            elif "rate limit" in error_msg:
                raise RuntimeError("OpenAI rate limit exceeded") from e
            elif "connection" in error_msg or "network" in error_msg:
                raise ConnectionError("Cannot reach OpenAI API") from e
            else:
                raise RuntimeError(f"OpenAI API error: {e}") from e

    def get_model_name(self) -> str:
        """Return the model identifier string."""
        return self.model_name

    def is_available(self) -> bool:
        """Check if the LLM backend is reachable."""
        try:
            # Minimal API call to check availability
            self.client.models.list()
            return True
        except Exception:
            return False
