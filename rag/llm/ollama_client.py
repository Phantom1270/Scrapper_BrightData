"""
Local LLM backend via Ollama.
"""

from typing import List, Dict, Optional
import urllib.request
import urllib.error
import json

from rag.llm.base import BaseLLMClient


class OllamaClient(BaseLLMClient):
    """Local LLM backend using Ollama."""

    def __init__(self, settings=None):
        if settings is None:
            from rag.config.settings import get_settings
            settings = get_settings()

        self.model_name = getattr(settings.llm, "model", "qwen2.5:3b")
        self.base_url = getattr(settings.llm, "base_url", "http://localhost:11434")
        self.default_temperature = getattr(settings.llm, "temperature", 0.1)
        self.default_max_tokens = getattr(settings.llm, "max_tokens", 300)

        try:
            import requests
            self._use_requests = True
        except ImportError:
            self._use_requests = False

    def _make_request(self, method: str, url: str, body: dict = None, timeout: int = 60) -> dict:
        """Internal helper for HTTP requests."""
        if self._use_requests:
            import requests
            try:
                response = requests.request(method, url, json=body, timeout=timeout)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.ConnectionError as e:
                raise ConnectionError(f"Cannot connect to Ollama at {self.base_url}. Is Ollama running? Start it with: ollama serve") from e
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"HTTP error from Ollama: {e}") from e
        else:
            req = urllib.request.Request(url, method=method)
            req.add_header('Content-Type', 'application/json')
            
            data = None
            if body is not None:
                data = json.dumps(body).encode('utf-8')
                
            try:
                with urllib.request.urlopen(req, data=data, timeout=timeout) as response:
                    return json.loads(response.read().decode('utf-8'))
            except urllib.error.URLError as e:
                if isinstance(e.reason, ConnectionRefusedError):
                    raise ConnectionError(f"Cannot connect to Ollama at {self.base_url}. Is Ollama running? Start it with: ollama serve") from e
                raise RuntimeError(f"Error calling Ollama: {e}") from e
            except Exception as e:
                raise RuntimeError(f"Unexpected error calling Ollama: {e}") from e

    def chat(self, messages: List[Dict[str, str]], temperature: Optional[float] = None,
             max_tokens: Optional[int] = None) -> str:
        """Send a chat completion request to Ollama."""
        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens

        body = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        url = f"{self.base_url.rstrip('/')}/api/chat"
        
        try:
            response_data = self._make_request("POST", url, body=body)
        except ConnectionError:
            raise
        except RuntimeError:
            raise
            
        try:
            return response_data["message"]["content"]
        except (KeyError, TypeError) as e:
            raise RuntimeError(f"Malformed response from Ollama: {response_data}") from e

    def get_model_name(self) -> str:
        """Return the model identifier string."""
        return self.model_name

    def is_available(self) -> bool:
        """Check if the LLM backend is reachable."""
        url = f"{self.base_url.rstrip('/')}/api/tags"
        try:
            self._make_request("GET", url, timeout=3)
            return True
        except Exception:
            return False
