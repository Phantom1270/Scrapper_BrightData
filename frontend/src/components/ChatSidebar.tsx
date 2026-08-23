import { useState, useEffect } from 'react'
import { useAuth } from '../contexts/AuthContext'
import {
  listConversations,
  formatRelativeTime,
  type StoredConversation,
} from '../services/storage'
import './ChatSidebar.css'

type ConversationListItem = Omit<StoredConversation, 'messages'>

type Props = {
  activeConversationId: string | null
  onNewChat: () => void
  onLoadChat: (id: string) => void
  onDeleteChat: (id: string) => void
  onScrapeNew: () => void
  children?: React.ReactNode
}

export function ChatSidebar({
  activeConversationId,
  onNewChat,
  onLoadChat,
  onDeleteChat,
  onScrapeNew,
  children,
}: Props) {
  const { user, logout } = useAuth()
  const [conversations, setConversations] = useState<ConversationListItem[]>([])
  const [deletingId, setDeletingId] = useState<string | null>(null)

  useEffect(() => {
    if (user) setConversations(listConversations(user.id))
  }, [activeConversationId, user])

  useEffect(() => {
    if (user) setConversations(listConversations(user.id))
  }, [user])

  const handleDelete = (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    if (deletingId === id) {
      onDeleteChat(id)
      setDeletingId(null)
      if (user) setConversations(listConversations(user.id))
    } else {
      setDeletingId(id)
      setTimeout(() => setDeletingId(null), 3000)
    }
  }

  return (
    <aside className="chat-sidebar">
      {/* Header */}
      <div className="sidebar-header">
        <h1 className="sidebar-title">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2a10 10 0 100 20 10 10 0 000-20zm0 18a8 8 0 110-16 8 8 0 010 16z"/><circle cx="9" cy="10" r="1.5"/><circle cx="15" cy="10" r="1.5"/><path d="M12 15.5c-1.8 0-3.3-1.1-3.8-2.6h7.6c-.5 1.5-2 2.6-3.8 2.6z"/>
          </svg>
          RoboAi
        </h1>
        <div className="sidebar-actions">
          <button
            className="new-chat-btn"
            onClick={onNewChat}
            title="Start a new conversation"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="12" y1="5" x2="12" y2="19" />
              <line x1="5" y1="12" x2="19" y2="12" />
            </svg>
            New Chat
          </button>
          <button
            className="scrape-new-btn"
            onClick={onScrapeNew}
            title="Scrape a new website"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" />
              <line x1="2" y1="12" x2="22" y2="12" />
              <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
            </svg>
          </button>
        </div>
      </div>

      {/* Conversation list */}
      <nav className="conversation-list">
        {conversations.length === 0 ? (
          <div className="conversation-empty">
            <p>No conversations yet.</p>
            <p>Start chatting to see your history here.</p>
          </div>
        ) : (
          conversations.map((conv) => (
            <button
              key={conv.id}
              className={`conversation-item ${
                conv.id === activeConversationId ? 'conversation-active' : ''
              }`}
              onClick={() => onLoadChat(conv.id)}
              title={conv.title}
            >
              <div className="conversation-info">
                <div className="conversation-title-row">
                  <span className={`mode-dot mode-dot-${conv.mode}`}>
                    {conv.mode === 'llm' ? '⚡' : '🔍'}
                  </span>
                  <span className="conversation-title">{conv.title}</span>
                </div>
                {conv.sourceUrl && (
                  <span className="conversation-source">
                    {new URL(conv.sourceUrl).hostname}
                  </span>
                )}
                <span className="conversation-time">
                  {formatRelativeTime(conv.updatedAt)}
                </span>
              </div>
              <button
                className={`conversation-delete ${
                  deletingId === conv.id ? 'delete-confirm' : ''
                }`}
                onClick={(e) => handleDelete(conv.id, e)}
                title={deletingId === conv.id ? 'Click again to confirm' : 'Delete'}
              >
                {deletingId === conv.id ? '!' : '\u00D7'}
              </button>
            </button>
          ))
        )}
      </nav>

      {/* Bottom: status bar + user info */}
      <div className="sidebar-bottom">
        {children}
        <div className="sidebar-user">
          <div className="user-avatar">
            <img src="https://i.pravatar.cc/150?u=a042581f4e29026704d" alt="User avatar" />
          </div>
          <div className="user-info-text">
            <span className="user-name">{user?.displayName ?? 'Guest'}</span>
            <span className="user-plan">Free plan</span>
          </div>
          <button className="logout-btn" onClick={logout} title="Sign out">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path>
              <polyline points="16 17 21 12 16 7"></polyline>
              <line x1="21" y1="12" x2="9" y2="12"></line>
            </svg>
          </button>
        </div>
      </div>
    </aside>
  )
}
