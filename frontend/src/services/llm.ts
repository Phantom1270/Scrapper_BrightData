/**
 * Direct LLM chat API service.
 * For conversations that don't use RAG retrieval.
 */

import { getApiBaseUrl } from './api'

export type LLMChatRequest = {
  message: string
  model?: string
  temperature?: number
  history?: Array<{ role: string; content: string }>
}

export type LLMChatResponse = {
  answer: string
  model: string
}

export async function chatWithLLM(payload: LLMChatRequest): Promise<LLMChatResponse> {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 120_000)

  try {
    const response = await fetch(`${getApiBaseUrl()}/api/v1/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal,
    })
    clearTimeout(timeout)

    if (!response.ok) {
      const body = await response.json().catch(() => null)
      throw new Error(body?.detail ?? `LLM request failed: ${response.status}`)
    }

    return (await response.json()) as LLMChatResponse
  } catch (err) {
    clearTimeout(timeout)

    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new Error('Request timed out after 2 minutes. The model may still be loading.')
    }

    if (err instanceof TypeError && err.message.includes('fetch')) {
      throw new Error(`Cannot reach the backend at ${getApiBaseUrl()}. Is the server running?`)
    }

    throw err
  }
}
