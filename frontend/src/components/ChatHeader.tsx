import type { ChatMode } from '../services/storage'
import './ChatHeader.css'

type Props = {
  mode: ChatMode
  sourceUrl?: string | null
}

export function ChatHeader({ mode, sourceUrl }: Props) {
  const getLabel = () => {
    if (mode === 'rag' && sourceUrl) {
      try {
        return `RAG — ${new URL(sourceUrl).hostname}`
      } catch {
        return 'RAG'
      }
    }
    if (mode === 'rag') return 'RAG'
    return 'LLM Chat'
  }

  return (
    <header className="chat-header">
      <div className="header-left">
        <h2 className="header-title">doc//rag</h2>
        <span className="header-source">
          <span className={`header-mode-badge header-mode-${mode}`}>
            {mode === 'llm' ? '⚡ LLM' : '🔍 RAG'}
          </span>
          Connected to <strong>{getLabel()}</strong>
        </span>
      </div>
    </header>
  )
}
