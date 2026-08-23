import { useState, useCallback, useRef, useEffect } from 'react'
import {
  startScrape,
  pollScrapeStatus,
  cancelScrape,
  DEFAULT_SCRAPE_CONFIG,
  type ScrapeConfig,
  type ScrapeProgress,
  type ScrapeStatus,
} from '../services/scraper'
import './ScrapeFlow.css'

// ── Component ─────────────────────────────────────────────────────

type Props = {
  onComplete: (sourceUrl: string, progress: ScrapeProgress) => void
  onCancel: () => void
}

type FlowStep = 'url' | 'running' | 'done'

const STEP_LABELS: Record<ScrapeStatus, string> = {
  crawling: 'Crawling URL frontier',
  scraping: 'Scraping pages',
  normalizing: 'Normalizing data',
  indexing: 'Indexing into RAG',
  done: 'Complete',
  error: 'Error',
  cancelled: 'Cancelled',
}

const STEP_ORDER: ScrapeStatus[] = ['crawling', 'scraping', 'normalizing', 'indexing', 'done']

export function ScrapeFlow({ onComplete, onCancel }: Props) {
  const [step, setStep] = useState<FlowStep>('url')
  const [config, setConfig] = useState<ScrapeConfig>(DEFAULT_SCRAPE_CONFIG)
  const [urlError, setUrlError] = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [progress, setProgress] = useState<ScrapeProgress | null>(null)
  const [errorMsg, setErrorMsg] = useState('')
  const cancelRef = useRef<(() => void) | null>(null)

  useEffect(() => {
    return () => {
      cancelRef.current?.()
    }
  }, [])

  // ── Update config field ─────────────────────────────────────

  const updateConfig = useCallback((patch: Partial<ScrapeConfig>) => {
    setConfig((prev) => ({ ...prev, ...patch }))
  }, [])

  // ── Validate URL ────────────────────────────────────────────

  const validateUrl = (input: string): boolean => {
    try {
      const parsed = new URL(input)
      if (!['http:', 'https:'].includes(parsed.protocol)) {
        setUrlError('URL must start with http:// or https://')
        return false
      }
      setUrlError('')
      return true
    } catch {
      setUrlError('Please enter a valid URL')
      return false
    }
  }

  // ── Start scraping ──────────────────────────────────────────

  const handleStart = useCallback(async () => {
    if (!validateUrl(config.startUrl)) return

    setStep('running')
    setErrorMsg('')

    try {
      const job = await startScrape(config.startUrl)

      const cancelFn = pollScrapeStatus(
        job.job_id,
        config.startUrl,
        (p) => setProgress(p),
        (p) => {
          setProgress(p)
          setStep('done')
        },
        (err) => {
          setErrorMsg(err)
          setStep('url')
        },
      )

      cancelRef.current = cancelFn
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Failed to start scraping')
      setStep('url')
    }
  }, [config])

  // ── Cancel ──────────────────────────────────────────────────

  const handleCancel = useCallback(() => {
    cancelRef.current?.()
    cancelRef.current = null
    onCancel()
  }, [onCancel])

  // ── Render ──────────────────────────────────────────────────

  return (
    <div className="scrape-flow">
      <div className="scrape-content">

        {/* ── Step 1: URL Input + Config ─────────────────────── */}
        {step === 'url' && (
          <div className="scrape-step step-url">
            <div className="step-icon">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10" />
                <line x1="2" y1="12" x2="22" y2="12" />
                <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
              </svg>
            </div>

            <h2>Scrape a documentation website</h2>
            <p className="step-description">
              Enter the URL of the documentation site you want to chat with.
              We'll crawl it, extract the content, and build a searchable knowledge base.
            </p>

            {/* ── Start URL ──────────────────────────────────── */}
            <div className="config-section">
              <label className="config-label" htmlFor="startUrl">
                Start URL <span className="required">*</span>
              </label>
              <div className="url-input-group">
                <input
                  id="startUrl"
                  type="text"
                  className={`url-input ${urlError ? 'input-error' : ''}`}
                  value={config.startUrl}
                  onChange={(e) => {
                    updateConfig({ startUrl: e.target.value })
                    if (urlError) setUrlError('')
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !showAdvanced) handleStart()
                  }}
                  placeholder="https://docs.example.com"
                  autoFocus
                />
              </div>
              {urlError && <span className="field-error">{urlError}</span>}
            </div>

            {/* ── Quick examples ─────────────────────────────── */}
            <div className="url-examples">
              <span className="examples-label">Quick start:</span>
              <button className="example-url" onClick={() => updateConfig({ startUrl: 'https://docs.python.org/3/' })}>
                Python Docs
              </button>
              <button className="example-url" onClick={() => updateConfig({ startUrl: 'https://react.dev/reference' })}>
                React Docs
              </button>
              <button className="example-url" onClick={() => updateConfig({ startUrl: 'https://docs.fastapi.tiangolo.com/' })}>
                FastAPI Docs
              </button>
            </div>

            {/* ── Advanced toggle ────────────────────────────── */}
            <button
              className="advanced-toggle"
              onClick={() => setShowAdvanced(!showAdvanced)}
              type="button"
            >
              {showAdvanced ? '▾' : '▸'} Advanced settings
            </button>

            {/* ── Advanced settings ──────────────────────────── */}
            {showAdvanced && (
              <div className="advanced-config">

                {/* ── Crawl Limits ───────────────────────────── */}
                <div className="config-group">
                  <h4 className="config-group-title">Crawl Limits</h4>

                  <div className="config-row">
                    <div className="config-field">
                      <label className="config-label" htmlFor="maxDepth">
                        Max Depth
                        <span className="config-hint">How many clicks deep from the start URL</span>
                      </label>
                      <div className="range-with-value">
                        <input
                          id="maxDepth"
                          type="range"
                          min={1}
                          max={10}
                          step={1}
                          value={config.maxDepth}
                          onChange={(e) => updateConfig({ maxDepth: Number(e.target.value) })}
                        />
                        <span className="range-value">{config.maxDepth}</span>
                      </div>
                    </div>

                    <div className="config-field">
                      <label className="config-label" htmlFor="maxPages">
                        Max Pages
                        <span className="config-hint">Hard stop to prevent infinite crawling</span>
                      </label>
                      <input
                        id="maxPages"
                        type="number"
                        className="config-input"
                        value={config.maxPages}
                        onChange={(e) => updateConfig({ maxPages: Math.max(1, Number(e.target.value)) })}
                        min={1}
                        max={10000}
                      />
                    </div>
                  </div>
                </div>

                {/* ── URL Filtering ──────────────────────────── */}
                <div className="config-group">
                  <h4 className="config-group-title">URL Filtering</h4>

                  <div className="config-field">
                    <label className="config-label" htmlFor="includePaths">
                      Include Paths
                      <span className="config-hint">Only scrape URLs containing these keywords. Comma-separated. Leave empty to allow all.</span>
                    </label>
                    <input
                      id="includePaths"
                      type="text"
                      className="config-input"
                      value={config.includePaths}
                      onChange={(e) => updateConfig({ includePaths: e.target.value })}
                      placeholder="/docs/,/api/,/guide/"
                    />
                  </div>

                  <div className="config-field">
                    <label className="config-label" htmlFor="excludePaths">
                      Exclude Paths
                      <span className="config-hint">Skip URLs matching these patterns. Comma-separated.</span>
                    </label>
                    <input
                      id="excludePaths"
                      type="text"
                      className="config-input"
                      value={config.excludePaths}
                      onChange={(e) => updateConfig({ excludePaths: e.target.value })}
                      placeholder="/blog/,/releases/,/archive/"
                    />
                  </div>
                </div>

                {/* ── Scraping Behavior ──────────────────────── */}
                <div className="config-group">
                  <h4 className="config-group-title">Scraping Behavior</h4>

                  <div className="config-field config-toggle-row">
                    <label className="config-label" htmlFor="jsRendering">
                      JavaScript Rendering
                      <span className="config-hint">Use a headless browser for dynamic sites (React, Next.js). Slower but handles JS-loaded content.</span>
                    </label>
                    <button
                      id="jsRendering"
                      className={`toggle-btn ${config.jsRendering ? 'toggle-on' : 'toggle-off'}`}
                      onClick={() => updateConfig({ jsRendering: !config.jsRendering })}
                      type="button"
                      role="switch"
                      aria-checked={config.jsRendering}
                    >
                      {config.jsRendering ? 'ON' : 'OFF'}
                    </button>
                  </div>

                  <div className="config-field">
                    <label className="config-label" htmlFor="contentSelector">
                      Content CSS Selector
                      <span className="config-hint">Which HTML element contains the actual text. Prevents ingesting headers, footers, and sidebars.</span>
                    </label>
                    <input
                      id="contentSelector"
                      type="text"
                      className="config-input config-input-mono"
                      value={config.contentSelector}
                      onChange={(e) => updateConfig({ contentSelector: e.target.value })}
                      placeholder="main, .markdown-body, #content"
                    />
                    <div className="selector-presets">
                      <span className="examples-label">Common:</span>
                      <button className="example-url" onClick={() => updateConfig({ contentSelector: 'main' })}>
                        main
                      </button>
                      <button className="example-url" onClick={() => updateConfig({ contentSelector: '.markdown-body' })}>
                        .markdown-body
                      </button>
                      <button className="example-url" onClick={() => updateConfig({ contentSelector: '#content' })}>
                        #content
                      </button>
                      <button className="example-url" onClick={() => updateConfig({ contentSelector: 'article' })}>
                        article
                      </button>
                    </div>
                  </div>

                  <div className="config-field">
                    <label className="config-label" htmlFor="rateLimit">
                      Rate Limit
                      <span className="config-hint">Delay between requests. Higher = safer but slower. Reduce risk of IP bans.</span>
                    </label>
                    <div className="range-with-value">
                      <input
                        id="rateLimit"
                        type="range"
                        min={0}
                        max={5000}
                        step={250}
                        value={config.rateLimitMs}
                        onChange={(e) => updateConfig({ rateLimitMs: Number(e.target.value) })}
                      />
                      <span className="range-value">{config.rateLimitMs}ms</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* ── Action buttons ─────────────────────────────── */}
            <div className="step-actions">
              <button className="back-link" onClick={onCancel}>
                &larr; Back to home
              </button>
              <button
                className="btn-start"
                onClick={handleStart}
                disabled={!config.startUrl.trim()}
              >
                Start Scraping
              </button>
            </div>

            {errorMsg && <span className="field-error">{errorMsg}</span>}
          </div>
        )}

        {/* ── Step 2: Progress ───────────────────────────────── */}
        {step === 'running' && progress && (
          <div className="scrape-step step-running">
            <h2>Scraping in progress</h2>
            <p className="step-description source-url-label">
              {progress.source_url}
            </p>

            <div className="progress-steps">
              {STEP_ORDER.slice(0, -1).map((status, i) => {
                const currentIdx = STEP_ORDER.indexOf(progress.status)
                const isComplete = i < currentIdx
                const isCurrent = i === currentIdx

                return (
                  <div
                    key={status}
                    className={`progress-step ${isComplete ? 'step-complete' : ''} ${isCurrent ? 'step-current' : ''} ${i > currentIdx ? 'step-pending' : ''}`}
                  >
                    <div className="step-indicator">
                      {isComplete ? '✓' : isCurrent ? (
                        <span className="step-spinner" />
                      ) : (
                        <span className="step-dot" />
                      )}
                    </div>
                    <div className="step-info">
                      <span className="step-name">{STEP_LABELS[status]}</span>
                      {status === 'crawling' && isComplete && (
                        <span className="step-detail">{progress.pages_found} pages found</span>
                      )}
                      {status === 'scraping' && (isComplete || isCurrent) && (
                        <span className="step-detail">
                          {progress.pages_scraped}/{progress.pages_found} scraped
                        </span>
                      )}
                      {status === 'indexing' && isComplete && (
                        <span className="step-detail">{progress.chunks_created} chunks created</span>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>

            <button className="btn-cancel" onClick={handleCancel}>
              Cancel
            </button>
          </div>
        )}

        {/* ── Step 3: Done ───────────────────────────────────── */}
        {step === 'done' && progress && (
          <div className="scrape-step step-done">
            <div className="done-icon">
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#4ade80" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                <polyline points="22 4 12 14.01 9 11.01" />
              </svg>
            </div>

            <h2>RAG pipeline ready</h2>

            <div className="done-stats">
              <div className="stat">
                <span className="stat-value">{progress.source_url}</span>
                <span className="stat-label">Source</span>
              </div>
              <div className="stat-row">
                <div className="stat">
                  <span className="stat-value">{progress.pages_scraped}</span>
                  <span className="stat-label">Pages indexed</span>
                </div>
                <div className="stat">
                  <span className="stat-value">{progress.chunks_created}</span>
                  <span className="stat-label">Chunks created</span>
                </div>
              </div>
            </div>

            <button
              className="btn-primary btn-start-chat"
              onClick={() => onComplete(progress.source_url, progress)}
            >
              Start Chatting &rarr;
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
