import './ChatModeSelector.css'

type Props = {
  onSelectLLM: () => void
  onSelectRAG: () => void
}

export function ChatModeSelector({ onSelectLLM, onSelectRAG }: Props) {
  return (
    <div className="mode-selector">
      <div className="mode-content">
        <h1 className="mode-title">New Chat</h1>
        <p className="mode-subtitle">
          Choose how you want to chat.
        </p>

        <div className="mode-cards">
          <button className="mode-card" onClick={onSelectLLM}>
            <div className="mode-card-icon">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 2a7 7 0 0 1 7 7c0 2.4-1.2 4.5-3 5.7V17a1 1 0 0 1-1 1H9a1 1 0 0 1-1-1v-2.3C6.2 13.5 5 11.4 5 9a7 7 0 0 1 7-7z" />
                <line x1="9" y1="21" x2="15" y2="21" />
                <line x1="10" y1="24" x2="14" y2="24" />
              </svg>
            </div>
            <h2>Chat with LLM</h2>
            <p>
              Talk directly to your Ollama model. No documents, no retrieval — 
              just a conversation with the AI.
            </p>
            <span className="mode-action">Start chatting &rarr;</span>
          </button>

          <button className="mode-card mode-card-accent" onClick={onSelectRAG}>
            <div className="mode-card-icon">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10" />
                <line x1="2" y1="12" x2="22" y2="12" />
                <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
              </svg>
            </div>
            <h2>Scrape &amp; Chat with RAG</h2>
            <p>
              Provide a documentation URL. We'll crawl it, build a knowledge base, 
              and let you ask questions grounded in that data.
            </p>
            <span className="mode-action">Get started &rarr;</span>
          </button>
        </div>
      </div>
    </div>
  )
}
