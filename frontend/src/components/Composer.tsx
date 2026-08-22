import type { FormEvent } from 'react'
import type { AdvancedSettings } from '../types/chat'

type ComposerProps = {
  question: string
  isLoading: boolean
  canSubmit: boolean
  showSettings: boolean
  settings: AdvancedSettings
  onSubmit: () => Promise<void>
  onQuestionChange: (question: string) => void
  onShowSettingsToggle: () => void
  onSettingsChange: (settings: AdvancedSettings) => void
}

export function Composer({
  question,
  isLoading,
  canSubmit,
  showSettings,
  settings,
  onSubmit,
  onQuestionChange,
  onShowSettingsToggle,
  onSettingsChange,
}: ComposerProps) {
  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    await onSubmit()
  }

  return (
    <footer className="composer-wrap">
      <form className="composer" onSubmit={handleSubmit}>
        {showSettings && (
          <fieldset className="advanced-settings">
            <legend>Advanced settings</legend>
            <label>
              Top K
              <input
                type="number"
                min={1}
                max={20}
                value={settings.topK}
                onChange={(event) => {
                  const parsed = Number.parseInt(event.target.value, 10)
                  const safeValue = Number.isNaN(parsed) ? 1 : Math.min(Math.max(parsed, 1), 20)
                  onSettingsChange({
                    ...settings,
                    topK: safeValue,
                  })
                }}
              />
            </label>
            <label>
              Content type filter
              <select
                value={settings.filterContentType}
                onChange={(event) =>
                  onSettingsChange({
                    ...settings,
                    filterContentType: event.target.value,
                  })
                }
              >
                <option value="">All</option>
                <option value="api_reference">API Reference</option>
                <option value="tutorial">Tutorial</option>
                <option value="notebook">Notebook</option>
              </select>
            </label>
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={settings.useQueryTransform}
                onChange={(event) =>
                  onSettingsChange({
                    ...settings,
                    useQueryTransform: event.target.checked,
                  })
                }
              />
              Use query transform
            </label>
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={settings.useReranking}
                onChange={(event) =>
                  onSettingsChange({
                    ...settings,
                    useReranking: event.target.checked,
                  })
                }
              />
              Use reranking
            </label>
          </fieldset>
        )}
        <textarea
          placeholder="Ask anything about your scraped docs..."
          rows={3}
          aria-label="Question input"
          value={question}
          onChange={(event) => onQuestionChange(event.target.value)}
          disabled={isLoading}
        />
        <div className="composer-actions">
          <button
            type="button"
            className="settings-btn"
            aria-label="Advanced settings"
            onClick={onShowSettingsToggle}
            disabled={isLoading}
          >
            ⚙
          </button>
          <button type="submit" className="send-btn" disabled={!canSubmit}>
            {isLoading ? 'Sending…' : 'Send'}
          </button>
        </div>
      </form>
    </footer>
  )
}
