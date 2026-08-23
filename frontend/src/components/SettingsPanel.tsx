import { useState, useEffect, useCallback } from 'react'
import { useSettings } from '../contexts/SettingsContext'
import {
  checkHealth,
  getOllamaModels,
  getServerSettings,
  setApiBaseUrl,
  updateServerSettings,
  type ServerSettings,
} from '../services/api'
import './SettingsPanel.css'

type Props = {
  open: boolean
  onClose: () => void
}

export function SettingsPanel({ open, onClose }: Props) {
  const { settings, updateSettings, resetSettings } = useSettings()

  // Local state for the panel
  const [ollamaModels, setOllamaModels] = useState<string[]>([])
  const [ollamaError, setOllamaError] = useState<string | null>(null)
  const [serverSettings, setServerSettings] = useState<ServerSettings | null>(null)
  const [healthStatus, setHealthStatus] = useState<'idle' | 'checking' | 'ok' | 'error'>('idle')
  const [healthMessage, setHealthMessage] = useState('')
  const [reindexing, setReindexing] = useState(false)
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')
  const [saveMessage, setSaveMessage] = useState('')

  const handleSaveToServer = useCallback(async () => {
    setSaveStatus('saving')
    setSaveMessage('')

    try {
      const result = await updateServerSettings({
        llm_model: settings.llmModel || undefined,
        embedding_model: settings.embeddingModel || undefined,
        reranker_model: settings.rerankerModel || undefined,
        top_k: settings.topK,
        use_query_transform: settings.useQueryTransform,
        use_reranking: settings.useReranking,
      })

      setSaveStatus('saved')
      let msg = 'Settings saved to server.'
      if (result.warning) msg += ` ${result.warning}`
      if (result.yaml_status) msg += ` ${result.yaml_status}`
      setSaveMessage(msg)

      setTimeout(() => {
        setSaveStatus('idle')
        setSaveMessage('')
      }, 5000)
    } catch (err) {
      setSaveStatus('error')
      setSaveMessage(
        err instanceof Error ? err.message : 'Failed to save settings.'
      )
      setTimeout(() => {
        setSaveStatus('idle')
        setSaveMessage('')
      }, 5000)
    }
  }, [settings])

  // Fetch data when panel opens
  useEffect(() => {
    if (!open) return

    // Fetch Ollama models
    getOllamaModels().then((result) => {
      setOllamaModels(result.models)
      setOllamaError(result.error ?? null)
    })

    // Fetch server settings to populate defaults
    getServerSettings().then((s) => {
      if (s) {
        setServerSettings(s)
        // Only fill in empty fields from server defaults
        if (!settings.llmModel) updateSettings({ llmModel: s.llm.model })
        if (!settings.rerankerModel) updateSettings({ rerankerModel: s.reranker.model_name })
        if (!settings.embeddingModel) updateSettings({ embeddingModel: s.embedding.model_name })
      }
    })
  }, [open]) // eslint-disable-line react-hooks/exhaustive-deps

  // Test connection
  const testConnection = useCallback(async () => {
    setHealthStatus('checking')
    setHealthMessage('')

    // Persist the current URL first
    setApiBaseUrl(settings.apiBaseUrl)

    const result = await checkHealth(settings.apiBaseUrl)

    if (result.status === 'ok' || result.status === 'healthy') {
      setHealthStatus('ok')
      setHealthMessage('Connected successfully')
    } else if (result.status === 'unreachable') {
      setHealthStatus('error')
      setHealthMessage('Server unreachable — is the backend running?')
    } else {
      setHealthStatus('error')
      setHealthMessage(`Status: ${result.status}`)
    }
  }, [settings.apiBaseUrl])

  // Re-index handler
  const handleReindex = useCallback(async () => {
    if (!confirm('This will delete all indexed chunks and re-process every document. Continue?')) return
    setReindexing(true)
    try {
      const { getApiBaseUrl } = await import('../services/api')
      await fetch(`${getApiBaseUrl()}/api/v1/index`, { method: 'POST' })
      alert('Re-indexing started. Check server logs for progress.')
    } catch {
      alert('Failed to trigger re-indexing. Is the backend running?')
    } finally {
      setReindexing(false)
    }
  }, [])

  // Close on Escape
  useEffect(() => {
    if (!open) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [open, onClose])

  if (!open) return null

  return (
    <div className="settings-overlay" onClick={onClose}>
      <div className="settings-panel" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="settings-header">
          <h2>Settings</h2>
          <button className="settings-close" onClick={onClose} aria-label="Close settings">
            &times;
          </button>
        </div>

        <div className="settings-body">

          {/* ── Section A: Connection ─────────────────────────── */}
          <section className="settings-section">
            <h3>Connection</h3>

            <div className="settings-field">
              <label htmlFor="apiBaseUrl">API Base URL</label>
              <div className="settings-input-row">
                <input
                  id="apiBaseUrl"
                  type="text"
                  value={settings.apiBaseUrl}
                  onChange={(e) => updateSettings({ apiBaseUrl: e.target.value })}
                  placeholder="http://127.0.0.1:8000"
                />
                <button
                  className="btn-secondary"
                  onClick={testConnection}
                  disabled={healthStatus === 'checking'}
                >
                  {healthStatus === 'checking' ? 'Testing...' : 'Test'}
                </button>
              </div>
              {healthMessage && (
                <span className={`settings-hint ${healthStatus === 'ok' ? 'hint-ok' : 'hint-error'}`}>
                  {healthStatus === 'ok' ? '●' : '●'} {healthMessage}
                </span>
              )}
            </div>
          </section>

          {/* ── Section B: Language Model ─────────────────────── */}
          <section className="settings-section">
            <h3>Language Model</h3>

            <div className="settings-field">
              <label htmlFor="llmProvider">Provider</label>
              <select id="llmProvider" value="ollama" disabled>
                <option value="ollama">Ollama (local)</option>
              </select>
              <span className="settings-hint">OpenAI and Anthropic support coming soon.</span>
            </div>

            <div className="settings-field">
              <label htmlFor="llmModel">Model</label>
              {ollamaModels.length > 0 ? (
                <select
                  id="llmModel"
                  value={settings.llmModel}
                  onChange={(e) => updateSettings({ llmModel: e.target.value })}
                >
                  {ollamaModels.map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              ) : (
                <>
                  <input
                    id="llmModel"
                    type="text"
                    value={settings.llmModel}
                    onChange={(e) => updateSettings({ llmModel: e.target.value })}
                    placeholder="e.g. qwen2.5:3b"
                  />
                  {ollamaError && (
                    <span className="settings-hint hint-error">
                      Could not fetch models: {ollamaError}
                    </span>
                  )}
                </>
              )}
            </div>
          </section>

          {/* ── Section C: Retrieval ──────────────────────────── */}
          <section className="settings-section">
            <h3>Retrieval</h3>

            <div className="settings-field">
              <label htmlFor="topK">
                Top K: <strong>{settings.topK}</strong>
              </label>
              <input
                id="topK"
                type="range"
                min={1}
                max={20}
                step={1}
                value={settings.topK}
                onChange={(e) => updateSettings({ topK: Number(e.target.value) })}
              />
              <span className="settings-hint">Number of chunks to retrieve. Higher = more context, slower.</span>
            </div>

            <div className="settings-field">
              <label htmlFor="filterContentType">Content Type Filter</label>
              <select
                id="filterContentType"
                value={settings.filterContentType}
                onChange={(e) => updateSettings({ filterContentType: e.target.value })}
              >
                <option value="">All types</option>
                <option value="prose">Prose only</option>
                <option value="code">Code only</option>
                <option value="parameters">Parameters only</option>
              </select>
            </div>

            <div className="settings-field settings-toggle-row">
              <label htmlFor="useQueryTransform">
                Query Transform
                <span className="settings-tooltip" title="Uses the LLM to expand your query into multiple variations. Slower but finds more relevant results.">?</span>
              </label>
              <button
                id="useQueryTransform"
                className={`toggle-btn ${settings.useQueryTransform ? 'toggle-on' : 'toggle-off'}`}
                onClick={() => updateSettings({ useQueryTransform: !settings.useQueryTransform })}
                type="button"
                role="switch"
                aria-checked={settings.useQueryTransform}
              >
                {settings.useQueryTransform ? 'ON' : 'OFF'}
              </button>
            </div>

            <div className="settings-field settings-toggle-row">
              <label htmlFor="useReranking">
                Reranking
                <span className="settings-tooltip" title="Re-scores results with a neural cross-encoder model. More accurate but adds latency.">?</span>
              </label>
              <button
                id="useReranking"
                className={`toggle-btn ${settings.useReranking ? 'toggle-on' : 'toggle-off'}`}
                onClick={() => updateSettings({ useReranking: !settings.useReranking })}
                type="button"
                role="switch"
                aria-checked={settings.useReranking}
              >
                {settings.useReranking ? 'ON' : 'OFF'}
              </button>
            </div>
          </section>

          {/* ── Section D: Advanced Models ────────────────────── */}
          <section className="settings-section">
            <h3>Advanced</h3>

            <div className="settings-field">
              <label htmlFor="rerankerModel">Reranker Model</label>
              <input
                id="rerankerModel"
                type="text"
                value={settings.rerankerModel}
                onChange={(e) => updateSettings({ rerankerModel: e.target.value })}
                placeholder="cross-encoder/ms-marco-MiniLM-L-6-v2"
              />
            </div>

            <div className="settings-field">
              <label htmlFor="embeddingModel">Embedding Model</label>
              <input
                id="embeddingModel"
                type="text"
                value={settings.embeddingModel}
                onChange={(e) => updateSettings({ embeddingModel: e.target.value })}
                placeholder="all-MiniLM-L6-v2"
              />
              <span className="settings-hint hint-warning">
                Changing the embedding model requires re-indexing all documents.
              </span>
              <button
                className="btn-danger"
                onClick={handleReindex}
                disabled={reindexing}
              >
                {reindexing ? 'Re-indexing...' : 'Re-index Now'}
              </button>
            </div>
          </section>
        </div>

        {/* Footer */}
        <div className="settings-footer">
          <div className="settings-footer-left">
            <button className="btn-secondary" onClick={resetSettings}>
              Reset to Defaults
            </button>
          </div>
          <div className="settings-footer-right">
            {saveMessage && (
              <span className={`save-status ${saveStatus === 'saved' ? 'save-ok' : saveStatus === 'error' ? 'save-error' : ''}`}>
                {saveMessage}
              </span>
            )}
            <button
              className="btn-save"
              onClick={handleSaveToServer}
              disabled={saveStatus === 'saving'}
            >
              {saveStatus === 'saving' ? 'Saving...' : 'Save to Server'}
            </button>
            <button className="btn-primary" onClick={onClose}>
              Done
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
