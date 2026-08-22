import { useMemo, useState } from 'react'
import { queryRag } from '../services/api'
import type { QueryRequest } from '../types/api'
import type { AdvancedSettings, ChatMessage, ConfidenceLevel } from '../types/chat'

const initialAssistantMessage: ChatMessage = {
  id: 'assistant-welcome',
  role: 'assistant',
  content:
    "Hi! I can help you query your documentation corpus. Ask a question and I'll return an answer with cited sources.",
}

const defaultSettings: AdvancedSettings = {
  topK: 5,
  filterContentType: '',
  useQueryTransform: true,
  useReranking: true,
}

function createMessageId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }

  return `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function normalizeConfidence(value: string | undefined): ConfidenceLevel | undefined {
  if (!value) {
    return undefined
  }

  const normalized = value.toLowerCase()
  if (normalized === 'high' || normalized === 'medium' || normalized === 'low') {
    return normalized
  }

  return undefined
}

export function useRagChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([initialAssistantMessage])
  const [question, setQuestion] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [settings, setSettings] = useState<AdvancedSettings>(defaultSettings)

  const canSubmit = useMemo(() => !isLoading && question.trim().length > 0, [isLoading, question])

  async function submitQuestion(): Promise<void> {
    const trimmedQuestion = question.trim()
    if (!trimmedQuestion || isLoading) {
      return
    }

    setMessages((previous) => [
      ...previous,
      {
        id: createMessageId(),
        role: 'user',
        content: trimmedQuestion,
      },
    ])
    setQuestion('')
    setIsLoading(true)

    const payload: QueryRequest = {
      question: trimmedQuestion,
      top_k: settings.topK,
      use_query_transform: settings.useQueryTransform,
      use_reranking: settings.useReranking,
    }

    if (settings.filterContentType) {
      payload.filter_content_type = settings.filterContentType
    }

    try {
      const response = await queryRag(payload)
      setMessages((previous) => [
        ...previous,
        {
          id: createMessageId(),
          role: 'assistant',
          content: response.answer,
          sources: response.sources ?? [],
          confidence: normalizeConfidence(response.confidence),
          retrievalTimeMs: response.retrieval_time_ms,
          generationTimeMs: response.generation_time_ms,
          transformUsed: response.transform_used,
          rerankerUsed: response.reranker_used,
        },
      ])
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : 'Request failed due to an unknown error from the frontend runtime.'

      setMessages((previous) => [
        ...previous,
        {
          id: createMessageId(),
          role: 'assistant',
          content: `I could not complete that request.\n\n${message}`,
          isError: true,
        },
      ])
    } finally {
      setIsLoading(false)
    }
  }

  function resetChat(): void {
    if (isLoading) {
      return
    }

    setMessages([initialAssistantMessage])
    setQuestion('')
  }

  return {
    messages,
    question,
    isLoading,
    canSubmit,
    showSettings,
    settings,
    setQuestion,
    setShowSettings,
    setSettings,
    submitQuestion,
    resetChat,
  }
}
