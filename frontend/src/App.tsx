import { useEffect, useRef } from 'react'
import './App.css'
import { ChatHeader } from './components/ChatHeader'
import { ChatSidebar } from './components/ChatSidebar'
import { Composer } from './components/Composer'
import { MessageList } from './components/MessageList'
import { useRagChat } from './hooks/useRagChat'
import { API_BASE_URL } from './services/api'

function App() {
  const {
    messages,
    question,
    isLoading,
    canSubmit,
    showSettings,
    settings,
    setQuestion,
    setShowSettings,
    setSettings,
    submitQuestion,
    resetChat,
  } = useRagChat()

  const messagesEndRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  return (
    <div className="app-shell">
      <ChatSidebar isLoading={isLoading} onNewChat={resetChat} />
      <main className="chat-pane">
        <ChatHeader apiBaseUrl={API_BASE_URL} />
        <MessageList messages={messages} isLoading={isLoading} messagesEndRef={messagesEndRef} />
        <Composer
          question={question}
          isLoading={isLoading}
          canSubmit={canSubmit}
          showSettings={showSettings}
          settings={settings}
          onSubmit={submitQuestion}
          onQuestionChange={setQuestion}
          onShowSettingsToggle={() => setShowSettings((previous) => !previous)}
          onSettingsChange={setSettings}
        />
      </main>
    </div>
  )
}

export default App
