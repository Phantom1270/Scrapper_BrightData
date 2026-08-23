import { useState } from 'react'
import Markdown from 'react-markdown'
import type { ChatMessage } from '../hooks/useRagChat'
import './MessageList.css'

type Props = {
  messages: ChatMessage[]
  isLoading: boolean
}

export function MessageList({ messages, isLoading }: Props) {
  return (
    <section className="message-list">
      {messages.length === 0 && (
        <div className="empty-state">
          <div className="empty-icon">?</div>
          <p>Ask a question about your documents.</p>
        </div>
      )}

      {messages.map((msg) => (
        <MessageBubble key={msg.id} message={msg} />
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

function MessageBubble({ message }: { message: ChatMessage }) {
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
  source: NonNullable<ChatMessage['sources']>[number]
  index: number
}) {
  const meta = source.metadata ?? {}
  const heading = (meta.heading as string) ?? (meta.title as string) ?? `Source ${index + 1}`
  const url = meta.url as string | undefined
  const contentType = meta.content_type as string | undefined
  const scorePercent = Math.round((source.score ?? 0) * 100)

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
              style={{ width: `${scorePercent}%` }}
            />
          </div>
          <span className="score-label">{scorePercent}%</span>
        </div>
      </div>
      <p className="source-preview">{source.content?.slice(0, 200)}{(source.content?.length ?? 0) > 200 ? '...' : ''}</p>
    </div>
  )
}
