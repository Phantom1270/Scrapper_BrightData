/**
 * Scraper API service.
 *
 * Currently STUBBED — returns mock responses so the UI can be built
 * and tested without the scraper backend.
 *
 * When the scraper backend is ready, replace the mock functions
 * with real fetch calls. The contract is documented below.
 */

import { getApiBaseUrl } from './api'

// ── Types ─────────────────────────────────────────────────────────

export type ScrapeConfig = {
  startUrl: string
  maxDepth: number
  maxPages: number
  includePaths: string
  excludePaths: string
  jsRendering: boolean
  contentSelector: string
  rateLimitMs: number
}

export const DEFAULT_SCRAPE_CONFIG: ScrapeConfig = {
  startUrl: '',
  maxDepth: 3,
  maxPages: 500,
  includePaths: '',
  excludePaths: '/blog/,/releases/,/archive/,?lang=',
  jsRendering: false,
  contentSelector: '',
  rateLimitMs: 1000,
}

export type ScrapeJob = {
  job_id: string
  status: ScrapeStatus
}

export type ScrapeStatus =
  | 'crawling'
  | 'scraping'
  | 'normalizing'
  | 'indexing'
  | 'done'
  | 'error'
  | 'cancelled'

export type ScrapeProgress = {
  job_id: string
  status: ScrapeStatus
  pages_found: number
  pages_scraped: number
  chunks_created: number
  error: string | null
  source_url: string
}

// ── Real API calls (uncomment when backend is ready) ──────────────
//
// export async function startScrape(config: ScrapeConfig): Promise<ScrapeJob> {
//   const response = await fetch(`${getApiBaseUrl()}/api/v1/scrape`, {
//     method: 'POST',
//     headers: { 'Content-Type': 'application/json' },
//     body: JSON.stringify({
//       url: config.startUrl,
//       max_depth: config.maxDepth,
//       max_pages: config.maxPages,
//       include_paths: config.includePaths.split(',').map(s => s.trim()).filter(Boolean),
//       exclude_paths: config.excludePaths.split(',').map(s => s.trim()).filter(Boolean),
//       js_rendering: config.jsRendering,
//       content_selector: config.contentSelector || null,
//       rate_limit_ms: config.rateLimitMs,
//     }),
//   })
//   if (!response.ok) {
//     const body = await response.json().catch(() => null)
//     throw new Error(body?.detail ?? `Scrape request failed: ${response.status}`)
//   }
//   return response.json()
// }
//
// export async function getScrapeStatus(jobId: string): Promise<ScrapeProgress> {
//   const response = await fetch(`${getApiBaseUrl()}/api/v1/scrape/${jobId}/status`)
//   if (!response.ok) throw new Error('Failed to get scrape status')
//   return response.json()
// }
//
// export async function cancelScrape(jobId: string): Promise<void> {
//   await fetch(`${getApiBaseUrl()}/api/v1/scrape/${jobId}/cancel`, {
//     method: 'POST',
//   })
// }

// ── Mock implementation (delete when real API is wired) ────────────

const MOCK_STEPS: ScrapeStatus[] = ['crawling', 'scraping', 'normalizing', 'indexing', 'done']
const MOCK_TIMING_MS = 2000 // time per step in mock

const activeMocks = new Map<string, {
  stepIndex: number
  pagesFound: number
  interval: ReturnType<typeof setInterval>
  resolve: (value: ScrapeProgress) => void
}>()

export async function startScrape(config: ScrapeConfig): Promise<ScrapeJob> {
  const job_id = `mock-${Date.now()}`

  return { job_id, status: 'crawling' }
}

export function pollScrapeStatus(
  jobId: string,
  sourceUrl: string,
  onUpdate: (progress: ScrapeProgress) => void,
  onDone: (progress: ScrapeProgress) => void,
  onError: (error: string) => void,
): () => void {
  let stepIndex = 0
  let pagesFound = 0
  let pagesScraped = 0
  let chunksCreated = 0

  const interval = setInterval(() => {
    const status = MOCK_STEPS[stepIndex]

    if (status === 'crawling') {
      pagesFound = Math.floor(Math.random() * 30) + 15
    } else if (status === 'scraping') {
      pagesScraped = pagesFound
    } else if (status === 'normalizing') {
      // no-op
    } else if (status === 'indexing') {
      chunksCreated = pagesScraped * (Math.floor(Math.random() * 5) + 3)
    }

    const progress: ScrapeProgress = {
      job_id: jobId,
      status,
      pages_found: pagesFound,
      pages_scraped: pagesScraped,
      chunks_created: chunksCreated,
      error: null,
      source_url: sourceUrl,
    }

    if (status === 'done') {
      clearInterval(interval)
      onDone(progress)
    } else {
      onUpdate(progress)
      stepIndex++
    }
  }, MOCK_TIMING_MS)

  // Return a cancel function
  return () => {
    clearInterval(interval)
  }
}

export async function cancelScrape(_jobId: string): Promise<void> {
  // Mock: no-op, the interval is cancelled by the returned function
}
