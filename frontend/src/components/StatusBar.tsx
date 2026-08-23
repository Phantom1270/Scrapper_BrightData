import { useState, useEffect, useCallback } from 'react'
import { checkHealth, type HealthResponse } from '../services/api'
import { useSettings } from '../contexts/SettingsContext'
import './StatusBar.css'

export function StatusBar() {
  const { settings } = useSettings()
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [expanded, setExpanded] = useState(false)

  const fetchHealth = useCallback(async () => {
    const result = await checkHealth()
    setHealth(result)
  }, [])

  // Check on mount and every 30 seconds
  useEffect(() => {
    fetchHealth()
    const interval = setInterval(fetchHealth, 30_000)
    return () => clearInterval(interval)
  }, [fetchHealth])

  // Re-check when API URL changes
  useEffect(() => {
    fetchHealth()
  }, [settings.apiBaseUrl, fetchHealth])

  const isOnline = health?.status === 'ok' || health?.status === 'healthy'

  return (
    <div className="status-bar">
      <button
        className="status-indicator"
        onClick={() => setExpanded(!expanded)}
        title="Click for details"
      >
        <span className={`status-dot ${isOnline ? 'dot-ok' : 'dot-error'}`} />
        <span className="status-label">
          {isOnline ? 'Backend Online' : health === null ? 'Checking...' : 'Backend Offline'}
        </span>
      </button>

      {expanded && health && (
        <div className="status-details">
          <div className="status-detail-row">
            <span className="detail-key">Status</span>
            <span className="detail-value">{health.status}</span>
          </div>
          {Object.entries(health.components).map(([name, state]) => (
            <div key={name} className="status-detail-row">
              <span className="detail-key">{name}</span>
              <span className={`detail-value detail-${typeof state === 'string' ? state : 'unknown'}`}>
                {String(state)}
              </span>
            </div>
          ))}
          <div className="status-detail-row">
            <span className="detail-key">API URL</span>
            <span className="detail-value detail-url">{settings.apiBaseUrl}</span>
          </div>
        </div>
      )}
    </div>
  )
}
