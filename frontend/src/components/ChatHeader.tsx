type ChatHeaderProps = {
  apiBaseUrl: string
}

export function ChatHeader({ apiBaseUrl }: ChatHeaderProps) {
  return (
    <header className="topbar">
      <h1>RAG Assistant</h1>
      <p>Connected to {apiBaseUrl}</p>
    </header>
  )
}
