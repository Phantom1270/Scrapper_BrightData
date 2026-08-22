import { useEffect, useRef, useState, type FormEvent } from 'react'
import './App.css'
import { queryRag } from './services/api'
import type { QueryRequest, QuerySource } from './types/api'

type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: QuerySource[]
  confidence?: 'high' | 'medium' | 'low'
  retrievalTimeMs?: number
  generationTimeMs?: number
  transformUsed?: string
  rerankerUsed?: string
  isError?: boolean
}

type AdvancedSettings = {
  topK: number
  filterContentType: string
  useQueryTransform: boolean
  useReranking: boolean
}

const initialAssistantMessage: ChatMessage = {
  id: 'assistant-welcome',
  role: 'assistant',
  content:
    "Hi! I can help you query your documentation corpus. Ask a question and I'll return an answer with cited sources.",
}

function createMessageId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }

  return `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function normalizeConfidence(value: string | undefined): 'high' | 'medium' | 'low' | undefined {
  if (!value) {
    return undefined
  }

  const normalized = value.toLowerCase()
  if (normalized === 'high' || normalized === 'medium' || normalized === 'low') {
    return normalized
  }

  return undefined
}

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([initialAssistantMessage])
  const [question, setQuestion] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [settings, setSettings] = useState<AdvancedSettings>({
    topK: 5,
    filterContentType: '',
    useQueryTransform: true,
    useReranking: true,
  })
  const messagesEndRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  async function handleSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault()
    const trimmedQuestion = question.trim()

    if (!trimmedQuestion || isLoading) {
      return
    }

    const userMessage: ChatMessage = {
      id: createMessageId(),
      role: 'user',
      content: trimmedQuestion,
    }
    setMessages((previous) => [...previous, userMessage])
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
      const assistantMessage: ChatMessage = {
        id: createMessageId(),
        role: 'assistant',
        content: response.answer,
        sources: response.sources,
        confidence: normalizeConfidence(response.confidence),
        retrievalTimeMs: response.retrieval_time_ms,
        generationTimeMs: response.generation_time_ms,
        transformUsed: response.transform_used,
        rerankerUsed: response.reranker_used,
      }
      setMessages((previous) => [...previous, assistantMessage])
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

  function handleNewChat(): void {
    if (isLoading) {
      return
    }

    setMessages([initialAssistantMessage])
    setQuestion('')
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <button type="button" className="new-chat-btn" onClick={handleNewChat} disabled={isLoading}>
          + New chat
        </button>
        <nav aria-label="Chat history" className="chat-history">
          <button type="button" className="history-item active" disabled>
            Current conversation
          </button>
        </nav>
      </aside>

      <main className="chat-pane">
        <header className="topbar">
          <h1>RAG Assistant</h1>
          <p>Connected to {import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'}</p>
        </header>

        <section className="messages" aria-label="Conversation">
          {messages.map((message) => (
            <article
              key={message.id}
              className={`message ${message.role === 'assistant' ? 'assistant' : 'user'} ${message.isError ? 'error' : ''}`}
            >
              <p>{message.content}</p>
              {message.role === 'assistant' && message.sources && message.sources.length > 0 && (
                <details className="sources">
                  <summary>Sources ({message.sources.length})</summary>
                  <ul>
                    {message.sources.map((source) => (
                      <li key={`${source.chunk_id}-${source.url}`}>
                        <a href={source.url} target="_blank" rel="noreferrer">
                          {source.heading}
                        </a>
                        <div className="source-meta">
                          <span>Score: {source.score.toFixed(3)}</span>
                          {source.content_type && <span>Type: {source.content_type}</span>}
                          {source.source && <span>From: {source.source}</span>}
                        </div>
                      </li>
                    ))}
                  </ul>
                </details>
              )}
              {message.role === 'assistant' &&
                (message.confidence ||
                  message.retrievalTimeMs !== undefined ||
                  message.generationTimeMs !== undefined) && (
                <div className="meta-row">
                  {message.confidence && (
                    <span className={`confidence ${message.confidence}`}>
                      Confidence: {message.confidence}
                    </span>
                  )}
                  {message.retrievalTimeMs !== undefined && (
                    <span>Retrieval: {message.retrievalTimeMs.toFixed(1)}ms</span>
                  )}
                  {message.generationTimeMs !== undefined && (
                    <span>Generation: {message.generationTimeMs.toFixed(1)}ms</span>
                  )}
                </div>
              )}
              {message.role === 'assistant' && (message.transformUsed || message.rerankerUsed) && (
                <div className="pipeline-row">
                  {message.transformUsed && <span>Transform: {message.transformUsed}</span>}
                  {message.rerankerUsed && <span>Reranker: {message.rerankerUsed}</span>}
                </div>
              )}
            </article>
          ))}
          {isLoading && (
            <article className="message assistant loading">
              <p>Thinking…</p>
            </article>
          )}
          <div ref={messagesEndRef} />
        </section>

        <footer className="composer-wrap">
          <form className="composer" onSubmit={handleSubmit}>
            {showSettings && (
              <fieldset className="advanced-settings">
                <legend>Advanced settings</legend>
                <label>
                  Top K
                  <input
                    type="number"
                    min={1}
                    max={20}
                    value={settings.topK}
                    onChange={(event) => {
                      const parsed = Number.parseInt(event.target.value, 10)
                      const safeValue = Number.isNaN(parsed) ? 1 : Math.min(Math.max(parsed, 1), 20)
                      setSettings((previous) => ({
                        ...previous,
                        topK: safeValue,
                      }))
                    }}
                  />
                </label>
                <label>
                  Content type filter
                  <select
                    value={settings.filterContentType}
                    onChange={(event) =>
                      setSettings((previous) => ({
                        ...previous,
                        filterContentType: event.target.value,
                      }))
                    }
                  >
                    <option value="">All</option>
                    <option value="api_reference">API Reference</option>
                    <option value="tutorial">Tutorial</option>
                    <option value="notebook">Notebook</option>
                  </select>
                </label>
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={settings.useQueryTransform}
                    onChange={(event) =>
                      setSettings((previous) => ({
                        ...previous,
                        useQueryTransform: event.target.checked,
                      }))
                    }
                  />
                  Use query transform
                </label>
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={settings.useReranking}
                    onChange={(event) =>
                      setSettings((previous) => ({
                        ...previous,
                        useReranking: event.target.checked,
                      }))
                    }
                  />
                  Use reranking
                </label>
              </fieldset>
            )}
            <textarea
              placeholder="Ask anything about your scraped docs..."
              rows={3}
              aria-label="Question input"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              disabled={isLoading}
            />
            <div className="composer-actions">
              <button
                type="button"
                className="settings-btn"
                aria-label="Advanced settings"
                onClick={() => setShowSettings((previous) => !previous)}
                disabled={isLoading}
              >
                ⚙
              </button>
              <button type="submit" className="send-btn" disabled={isLoading || !question.trim()}>
                {isLoading ? 'Sending…' : 'Send'}
              </button>
            </div>
          </form>
        </footer>
      </main>
    </div>
  )
}

export default App
