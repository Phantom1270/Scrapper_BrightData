import './App.css'

type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: Array<{
    heading: string
    url: string
  }>
  confidence?: 'high' | 'medium' | 'low'
  retrievalTimeMs?: number
  generationTimeMs?: number
}

function App() {
  const messages: ChatMessage[] = [
    {
      id: '1',
      role: 'assistant',
      content:
        "Hi! I can help you query your documentation corpus. Ask a question and I'll return an answer with cited sources.",
    },
    {
      id: '2',
      role: 'user',
      content: 'How do I use sklearn config_context?',
    },
    {
      id: '3',
      role: 'assistant',
      content:
        'Use `sklearn.config_context(...)` as a temporary context manager to override global configuration values inside a code block.',
      confidence: 'high',
      retrievalTimeMs: 1619.3,
      generationTimeMs: 7555.6,
      sources: [
        {
          heading: 'sklearn.config_context > Description',
          url: 'https://scikit-learn.org/stable/modules/generated/sklearn.config_context.html',
        },
      ],
    },
  ]

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <button type="button" className="new-chat-btn">
          + New chat
        </button>
        <nav aria-label="Chat history" className="chat-history">
          <button type="button" className="history-item active">
            sklearn config_context
          </button>
          <button type="button" className="history-item">
            BrightData crawl summary
          </button>
        </nav>
      </aside>

      <main className="chat-pane">
        <header className="topbar">
          <h1>RAG Assistant</h1>
          <p>Connected to local retrieval API</p>
        </header>

        <section className="messages" aria-label="Conversation">
          {messages.map((message) => (
            <article
              key={message.id}
              className={`message ${message.role === 'assistant' ? 'assistant' : 'user'}`}
            >
              <p>{message.content}</p>
              {message.role === 'assistant' && message.sources && (
                <details className="sources">
                  <summary>Sources ({message.sources.length})</summary>
                  <ul>
                    {message.sources.map((source) => (
                      <li key={source.url}>
                        <a href={source.url} target="_blank" rel="noreferrer">
                          {source.heading}
                        </a>
                      </li>
                    ))}
                  </ul>
                </details>
              )}
              {message.role === 'assistant' && message.confidence && (
                <div className="meta-row">
                  <span className={`confidence ${message.confidence}`}>
                    Confidence: {message.confidence}
                  </span>
                  <span>Retrieval: {message.retrievalTimeMs?.toFixed(1)}ms</span>
                  <span>Generation: {message.generationTimeMs?.toFixed(1)}ms</span>
                </div>
              )}
            </article>
          ))}
        </section>

        <footer className="composer-wrap">
          <form className="composer" onSubmit={(event) => event.preventDefault()}>
            <textarea
              placeholder="Ask anything about your scraped docs..."
              rows={3}
              aria-label="Question input"
            />
            <div className="composer-actions">
              <button type="button" className="settings-btn" aria-label="Advanced settings">
                ⚙
              </button>
              <button type="submit" className="send-btn">
                Send
              </button>
            </div>
          </form>
        </footer>
      </main>
    </div>
  )
}

export default App
