import { useState, useCallback, useEffect, useRef } from 'react'
import { queryRag, type QueryResponse } from '../services/api'
import { chatWithLLM } from '../services/llm'
import { useSettings } from '../contexts/SettingsContext'
import { useAuth } from '../contexts/AuthContext'
import {
  saveConversation,
  loadConversation,
  deleteConversation,
  generateTitle,
  type ChatMode,
  type StoredConversation,
} from '../services/storage'

export type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: number
  // RAG-specific
  sources?: QueryResponse['sources']
  confidence?: string
  retrievalTimeMs?: number
  generationTimeMs?: number
  totalTimeMs?: number
  queryTransformUsed?: boolean
  rerankerUsed?: boolean
  // LLM-specific
  model?: string
  // Common
  isError?: boolean
}

export function useRagChat() {
  const { settings } = useSettings()
  const { user } = useAuth()

  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [question, setQuestion] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [mode, setMode] = useState<ChatMode>('llm')
  const [sourceUrl, setSourceUrl] = useState<string | null>(null)

  const hasSavedRef = useRef(false)

  // ── Auto-save ───────────────────────────────────────────────

  useEffect(() => {
    if (!user || messages.length === 0) {
      hasSavedRef.current = false
      return
    }

    const id = conversationId ?? crypto.randomUUID()
    if (!conversationId) setConversationId(id)

    const conversation: StoredConversation = {
      id,
      title: generateTitle(messages, mode, sourceUrl),
      mode,
      sourceUrl,
      scrapeJobId: null,
      createdAt: messages[0].timestamp,
      updatedAt: Date.now(),
      messages,
    }

    saveConversation(user.id, conversation)
    hasSavedRef.current = true
  }, [messages, conversationId, mode, sourceUrl, user])

  // ── Send message ────────────────────────────────────────────

  const sendMessage = useCallback(
    async (text?: string) => {
      const queryText = text ?? question
      if (!queryText.trim() || isLoading || !user) return

      const userMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'user',
        content: queryText.trim(),
        timestamp: Date.now(),
      }

      setMessages((prev) => [...prev, userMessage])
      setQuestion('')
      setIsLoading(true)

      try {
        let assistantMessage: ChatMessage

        if (mode === 'rag') {
          // ── RAG mode: use retrieval pipeline ──────────────
          const response = await queryRag({
            question: queryText.trim(),
            top_k: settings.topK || undefined,
            filter_content_type: settings.filterContentType || undefined,
            use_reranking: settings.useReranking,
            use_query_transform: settings.useQueryTransform,
          })

          assistantMessage = {
            id: crypto.randomUUID(),
            role: 'assistant',
            content: response.answer,
            timestamp: Date.now(),
            sources: response.sources,
            confidence: response.confidence,
            retrievalTimeMs: response.retrieval_time_ms,
            generationTimeMs: response.generation_time_ms,
            totalTimeMs: response.total_time_ms,
            queryTransformUsed: response.query_transform_used,
            rerankerUsed: response.reranker_used,
          }
        } else {
          // ── LLM mode: direct chat ─────────────────────────
          const history = messages
            .filter((m) => !m.isError)
            .map((m) => ({ role: m.role, content: m.content }))

          const response = await chatWithLLM({
            message: queryText.trim(),
            model: settings.llmModel || undefined,
            history,
          })

          assistantMessage = {
            id: crypto.randomUUID(),
            role: 'assistant',
            content: response.answer,
            timestamp: Date.now(),
            model: response.model,
          }
        }

        setMessages((prev) => [...prev, assistantMessage])
      } catch (err) {
        const errorMessage: ChatMessage = {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: err instanceof Error ? err.message : 'An unexpected error occurred.',
          timestamp: Date.now(),
          isError: true,
        }
        setMessages((prev) => [...prev, errorMessage])
      } finally {
        setIsLoading(false)
      }
    },
    [question, isLoading, settings, mode, messages, user]
  )

  // ── New chat ────────────────────────────────────────────────

  const newChat = useCallback(() => {
    setMessages([])
    setQuestion('')
    setConversationId(null)
    setMode('llm')
    setSourceUrl(null)
    hasSavedRef.current = false
  }, [])

  // ── Start LLM chat ─────────────────────────────────────────

  const startLLMChat = useCallback(() => {
    newChat()
    setMode('llm')
  }, [newChat])

  // ── Start RAG chat (after scrape completes) ────────────────

  const startRAGChat = useCallback((scrapedUrl: string) => {
    setMessages([])
    setQuestion('')
    setConversationId(null)
    setMode('rag')
    setSourceUrl(scrapedUrl)
    hasSavedRef.current = false
  }, [])

  // ── Load past conversation ─────────────────────────────────

  const loadChat = useCallback((id: string) => {
    if (!user) return
    const stored = loadConversation(user.id, id)
    if (!stored) return

    setMessages(stored.messages)
    setConversationId(stored.id)
    setMode(stored.mode)
    setSourceUrl(stored.sourceUrl ?? null)
    setQuestion('')
    hasSavedRef.current = true
  }, [user])

  // ── Delete conversation ────────────────────────────────────

  const removeChat = useCallback((id: string) => {
    if (!user) return
    deleteConversation(user.id, id)
    if (id === conversationId) newChat()
  }, [user, conversationId, newChat])

  return {
    messages,
    question,
    isLoading,
    showSettings,
    conversationId,
    mode,
    sourceUrl,

    setQuestion,
    setShowSettings,

    sendMessage,
    newChat,
    startLLMChat,
    startRAGChat,
    loadChat,
    removeChat,
  }
}
