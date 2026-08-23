import { useState } from 'react'
import { AuthProvider, useAuth } from './contexts/AuthContext'
import { SettingsProvider } from './contexts/SettingsContext'
import { useRagChat } from './hooks/useRagChat'
import { LoginScreen } from './components/LoginScreen'
import { WelcomeScreen } from './components/WelcomeScreen'
import { ChatModeSelector } from './components/ChatModeSelector'
import { ScrapeFlow } from './components/ScrapeFlow'
import { ChatSidebar } from './components/ChatSidebar'
import { ChatHeader } from './components/ChatHeader'
import { MessageList } from './components/MessageList'
import { Composer } from './components/Composer'
import { SettingsPanel } from './components/SettingsPanel'
import { StatusBar } from './components/StatusBar'
import type { ScrapeProgress } from './services/scraper'
import './App.css'

type AppView = 'welcome' | 'modeSelect' | 'scraping' | 'chat'

function AppContent() {
  const { isAuthenticated, login } = useAuth()
  const [view, setView] = useState<AppView>('welcome')

  const {
    messages,
    question,
    setQuestion,
    isLoading,
    showSettings,
    setShowSettings,
    conversationId,
    mode,
    sourceUrl,
    sendMessage,
    newChat,
    startLLMChat,
    startRAGChat,
    loadChat,
    removeChat,
  } = useRagChat()

  // ── Auth gate ───────────────────────────────────────────────

  if (!isAuthenticated) {
    return <LoginScreen onLogin={login} />
  }

  // ── Navigation handlers ─────────────────────────────────────

  const handleNewChat = () => {
    newChat()
    setView('modeSelect')
  }

  const handleSelectLLM = () => {
    startLLMChat()
    setView('chat')
  }

  const handleSelectRAG = () => {
    setView('scraping')
  }

  const handleScrapeComplete = (scrapedUrl: string, _progress: ScrapeProgress) => {
    startRAGChat(scrapedUrl)
    setView('chat')
  }

  const handleScrapeCancel = () => {
    setView('modeSelect')
  }

  const handleLoadChat = (id: string) => {
    loadChat(id)
    setView('chat')
  }

  const handleBackToWelcome = () => {
    setView('welcome')
  }

  // ── Render ──────────────────────────────────────────────────

  return (
    <div className="app-layout">
      <ChatSidebar
        activeConversationId={conversationId}
        onNewChat={handleNewChat}
        onLoadChat={handleLoadChat}
        onDeleteChat={removeChat}
        onScrapeNew={() => {
          newChat()
          setView('scraping')
        }}
      >
        <StatusBar />
      </ChatSidebar>

      <main className="chat-pane">
        {view === 'welcome' && (
          <WelcomeScreen
            onStartChat={handleNewChat}
            onStartScrape={() => {
              newChat()
              setView('scraping')
            }}
          />
        )}

        {view === 'modeSelect' && (
          <ChatModeSelector
            onSelectLLM={handleSelectLLM}
            onSelectRAG={handleSelectRAG}
          />
        )}

        {view === 'scraping' && (
          <ScrapeFlow
            onComplete={handleScrapeComplete}
            onCancel={handleScrapeCancel}
          />
        )}

        {view === 'chat' && (
          <>
            <ChatHeader mode={mode} sourceUrl={sourceUrl} />
            <MessageList messages={messages} isLoading={isLoading} />
            <Composer
              question={question}
              onQuestionChange={setQuestion}
              onSubmit={() => sendMessage()}
              isLoading={isLoading}
              onOpenSettings={() => setShowSettings(true)}
            />
          </>
        )}
      </main>

      <SettingsPanel
        open={showSettings}
        onClose={() => setShowSettings(false)}
      />
    </div>
  )
}

function App() {
  return (
    <AuthProvider>
      <SettingsProvider>
        <AppContent />
      </SettingsProvider>
    </AuthProvider>
  )
}

export default App
