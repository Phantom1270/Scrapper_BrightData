import { useState } from 'react'
import Markdown from 'react-markdown'
import type { ChatMessage } from '../hooks/useRagChat'
import './MessageList.css'

type Props = {
  messages: ChatMessage[]
  isLoading: boolean
  onDeletePair: (messageId: string) => void
}

export function MessageList({ messages, isLoading, onDeletePair }: Props) {
  return (
    <section className="message-list">
      {messages.length === 0 && (
        <div className="empty-state">
          <div className="empty-icon">?</div>
          <p>Ask a question about your documents.</p>
        </div>
      )}

      {messages.map((msg) => (
        <MessageBubble key={msg.id} message={msg} onDeletePair={msg.role === 'user' ? onDeletePair : undefined} />
      ))}

      {isLoading && (
        <div className="message message--assistant">
          <div className="message-bubble">
            <div className="typing-indicator">
              <span /><span /><span />
            </div>
          </div>
        </div>
      )}
    </section>
  )
}

// ── Single message ────────────────────────────────────────────────

function MessageBubble({ message, onDeletePair }: { message: ChatMessage; onDeletePair?: (id: string) => void }) {
  const isUser = message.role === 'user'

  return (
    <div className={`message message--${message.role}`}>
      <div className={`message-bubble ${message.isError ? 'message--error' : ''}`}>
        {/* Content */}
        <div className="message-content">
          {isUser ? (
            <p>{message.content}</p>
          ) : (
            <Markdown>{message.content}</Markdown>
          )}
        </div>

        {/* Delete button — only on user messages, appears on hover */}
        {onDeletePair && (
          <button
            className="delete-pair-btn"
            onClick={() => onDeletePair(message.id)}
            title="Delete this Q&A pair"
            aria-label="Delete question and response"
          >
            🗑
          </button>
        )}

        {/* Copy button for assistant messages */}
        {!isUser && !message.isError && (
          <CopyButton text={message.content} />
        )}

        {/* Sources */}
        {message.sources && message.sources.length > 0 && (
          <SourceList sources={message.sources} />
        )}

        {/* Metadata row */}
        {!isUser && !message.isError && (
          <div className="message-meta">
            {message.confidence && (
              <span className={`confidence confidence--${message.confidence}`}>
                {message.confidence}
              </span>
            )}
            {message.totalTimeMs != null && (
              <span className="timing">{message.totalTimeMs}ms</span>
            )}
            {message.queryTransformUsed && (
              <span className="pipeline-badge">Transform</span>
            )}
            {message.rerankerUsed && (
              <span className="pipeline-badge">Reranked</span>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

// ── Copy button ───────────────────────────────────────────────────

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // Clipboard API may be blocked
    }
  }

  return (
    <button
      className="copy-btn"
      onClick={handleCopy}
      title="Copy answer"
      aria-label="Copy answer to clipboard"
    >
      {copied ? 'Copied' : 'Copy'}
    </button>
  )
}

// ── Source list ────────────────────────────────────────────────────

function SourceList({ sources }: { sources: ChatMessage['sources'] }) {
  const [expanded, setExpanded] = useState(false)

  if (!sources || sources.length === 0) return null

  return (
    <div className="sources-section">
      <button
        className="sources-toggle"
        onClick={() => setExpanded(!expanded)}
      >
        {expanded ? '▾' : '▸'} {sources.length} source{sources.length !== 1 ? 's' : ''}
      </button>

      {expanded && (
        <div className="sources-list">
          {sources.map((source, i) => (
            <SourceCard key={i} source={source} index={i} />
          ))}
        </div>
      )}
    </div>
  )
}

// ── Single source card ────────────────────────────────────────────

function SourceCard({
  source,
  index,
}: {
  source: NonNullable<ChatMessage['sources']>[number] & { [key: string]: any }
  index: number
}) {
  const meta = source.metadata ?? {}
  const heading = (source.heading as string) ?? (meta.heading as string) ?? (meta.title as string) ?? `Source ${index + 1}`
  const url = (source.url as string) ?? (meta.url as string | undefined)
  const contentType = (source.content_type as string) ?? (meta.content_type as string | undefined)
  
  const rawScore = source.score ?? meta.score ?? 0
  // Handle both probabilities (0.0-1.0) and cross-encoder logits (typically 1-10)
  const percentVal = rawScore > 1 ? rawScore * 10 : rawScore * 100
  const scoreFormatted = percentVal.toFixed(1)

  return (
    <div className="source-card">
      <div className="source-header">
        <span className="source-index">{index + 1}</span>
        <div className="source-info">
          {url ? (
            <a href={url} target="_blank" rel="noopener noreferrer" className="source-heading">
              {heading}
            </a>
          ) : (
            <span className="source-heading">{heading}</span>
          )}
          {contentType && (
            <span className={`content-type-badge badge-${contentType}`}>
              {contentType}
            </span>
          )}
        </div>
        <div className="source-score">
          <div className="score-bar">
            <div
              className="score-fill"
              style={{ width: `${Math.min(100, Math.max(0, percentVal))}%` }}
            />
          </div>
          <span className="score-label">{scoreFormatted}%</span>
        </div>
      </div>
      <p className="source-preview">{source.content?.slice(0, 200)}{(source.content?.length ?? 0) > 200 ? '...' : ''}</p>
    </div>
  )
}
