"""
LLM abstraction layer and factory.
"""

from rag.llm.base import BaseLLMClient


def create_llm_client(settings=None) -> BaseLLMClient:
    """
    Create the appropriate LLM client based on config.
    Reads settings.llm.provider to decide which backend to use.
    """
    if settings is None:
        from rag.config.settings import get_settings
        settings = get_settings()

    provider = getattr(settings.llm, 'provider', 'ollama')

    if provider == "ollama":
        from rag.llm.ollama_client import OllamaClient
        return OllamaClient(settings)
    elif provider == "openai":
        from rag.llm.openai_client import OpenAIClient
        import os
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable required "
                "when llm.provider is 'openai'"
            )
        return OpenAIClient(api_key=api_key, model=settings.llm.model)
    else:
        raise ValueError(f"Unknown LLM provider: '{provider}'. Use 'ollama' or 'openai'.")


__all__ = [
    "BaseLLMClient",
    "create_llm_client",
]
