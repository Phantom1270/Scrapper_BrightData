import './WelcomeScreen.css'

type Props = {
  onStartChat: () => void
  onStartScrape: () => void
}

export function WelcomeScreen({ onStartChat, onStartScrape }: Props) {
  return (
    <div className="welcome-screen">
      <div className="welcome-content">
        <div className="welcome-logo">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 2a7 7 0 0 1 7 7c0 2.4-1.2 4.5-3 5.7V17a1 1 0 0 1-1 1H9a1 1 0 0 1-1-1v-2.3C6.2 13.5 5 11.4 5 9a7 7 0 0 1 7-7z" />
            <line x1="9" y1="21" x2="15" y2="21" />
          </svg>
        </div>
        <h1 className="welcome-title">RAG Assistant</h1>
        <p className="welcome-subtitle">
          Chat directly with the AI or scrape a website to build a searchable knowledge base.
        </p>

        <div className="welcome-cards">
          {/* Card 1: Chat with LLM */}
          <button className="welcome-card" onClick={onStartChat}>
            <div className="welcome-card-icon">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
              </svg>
            </div>
            <h2>New Chat</h2>
            <p>
              Choose between direct AI chat or RAG-powered conversation with your documents.
            </p>
            <span className="welcome-action">Get started →</span>
          </button>

          {/* Card 2: Scrape & build */}
          <button className="welcome-card" onClick={onStartScrape}>
            <div className="welcome-card-icon">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10" />
                <line x1="2" y1="12" x2="22" y2="12" />
                <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
              </svg>
            </div>
            <h2>Scrape a Website</h2>
            <p>
              Provide a documentation URL. We'll crawl it, extract the content, and build a knowledge base.
            </p>
            <span className="welcome-action">Start scraping →</span>
          </button>
        </div>
      </div>
    </div>
  )
}
