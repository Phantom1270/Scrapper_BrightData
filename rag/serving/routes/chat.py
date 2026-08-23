"""
Direct LLM chat endpoint (no RAG).
"""
from typing import Optional, List
from pydantic import BaseModel, Field
from fastapi import APIRouter

router = APIRouter()


class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class DirectChatRequest(BaseModel):
    message: str = Field(..., description="The user's message")
    model: Optional[str] = Field(default=None, description="Override Ollama model name")
    temperature: Optional[float] = Field(default=0.7, ge=0, le=2)
    history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Previous messages in the conversation for context"
    )


@router.post("/chat")
def direct_chat(request: DirectChatRequest):
    """
    Send a message directly to the LLM without RAG retrieval.
    Supports conversation history for multi-turn dialogue.
    """
    from rag.config.settings import get_settings
    from rag.llm.ollama_client import OllamaClient

    settings = get_settings()

    client = OllamaClient(settings=settings)
    if request.model:
        client.model_name = request.model

    # Build messages array for the LLM
    messages = []

    if request.history:
        for msg in request.history:
            messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": request.message})

    try:
        answer = client.chat(
            messages=messages,
            temperature=request.temperature,
        )
        return {
            "answer": answer,
            "model": request.model or settings.llm.model,
        }
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail=f"LLM error: {str(e)}")
