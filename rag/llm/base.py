"""
Abstract base class for all LLM providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict


class BaseLLMClient(ABC):
    """Abstract interface for LLM backends."""

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.1,
             max_tokens: int = 300) -> str:
        """
        Send a chat completion request.
        messages format: [{"role": "system", "content": "..."},
                          {"role": "user", "content": "..."}]
        Returns: the assistant's response text (just the content string).
        Must raise ConnectionError if the LLM backend is unreachable.
        Must raise RuntimeError for API errors (bad response, rate limit, etc.).
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier string."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM backend is reachable.
        Returns True if the backend responds to a health check.
        Returns False if unreachable. Must NOT raise exceptions."""
        pass
