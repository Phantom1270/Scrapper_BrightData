type ChatSidebarProps = {
  isLoading: boolean
  onNewChat: () => void
}

export function ChatSidebar({ isLoading, onNewChat }: ChatSidebarProps) {
  return (
    <aside className="sidebar">
      <button type="button" className="new-chat-btn" onClick={onNewChat} disabled={isLoading}>
        + New chat
      </button>
      <nav aria-label="Chat history" className="chat-history">
        <button type="button" className="history-item active" disabled>
          Current conversation
        </button>
      </nav>
    </aside>
  )
}
