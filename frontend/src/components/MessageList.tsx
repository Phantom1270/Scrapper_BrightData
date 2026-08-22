import type { RefObject } from 'react'
import type { ChatMessage } from '../types/chat'

type MessageListProps = {
  messages: ChatMessage[]
  isLoading: boolean
  messagesEndRef: RefObject<HTMLDivElement | null>
}

export function MessageList({ messages, isLoading, messagesEndRef }: MessageListProps) {
  return (
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
  )
}
