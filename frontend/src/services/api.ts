/**
 * API service layer.
 * All backend communication goes through this file.
 */

// ── Dynamic base URL ──────────────────────────────────────────────
// Reads from localStorage first (set by SettingsPanel),
// falls back to env var, then to localhost default.

export function getApiBaseUrl(): string {
  return (
    localStorage.getItem('rag_apiBaseUrl') ??
    import.meta.env.VITE_API_BASE_URL ??
    'http://127.0.0.1:8000'
  )
}

export function setApiBaseUrl(url: string): void {
  localStorage.setItem('rag_apiBaseUrl', url.replace(/\/+$/, '')) // strip trailing slash
}

// ── Types ─────────────────────────────────────────────────────────

export type QueryRequest = {
  question: string
  top_k?: number
  filter_content_type?: string
  filter_doc_id?: string
  use_reranking?: boolean
  use_query_transform?: boolean
}

export type Source = {
  content: string
  metadata: Record<string, unknown>
  score: number
}

export type QueryResponse = {
  answer: string
  sources: Source[]
  confidence: string
  retrieval_time_ms: number
  generation_time_ms: number
  total_time_ms: number
  query_transform_used: boolean
  reranker_used: boolean
}

export type HealthResponse = {
  status: string
  components: Record<string, string>
}

export type ServerSettings = {
  llm: {
    provider: string
    model: string
    base_url: string
  }
  embedding: {
    model_name: string
  }
  reranker: {
    enabled: boolean
    model_name: string
  }
  retrieval: {
    top_k: number
    use_query_transform: boolean
    use_reranking: boolean
  }
}

// ── Core query endpoint ───────────────────────────────────────────

export async function queryRag(payload: QueryRequest): Promise<QueryResponse> {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 120_000) // 2 min

  try {
    const response = await fetch(`${getApiBaseUrl()}/api/v1/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal,
    })
    clearTimeout(timeout)

    if (!response.ok) {
      const body = await response.json().catch(() => null)
      const detail = body?.detail ?? body?.error ?? `HTTP ${response.status}`
      throw new Error(detail)
    }

    return (await response.json()) as QueryResponse
  } catch (err) {
    clearTimeout(timeout)

    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new Error(
        'Request timed out after 2 minutes. The model may still be loading — try again.'
      )
    }

    if (err instanceof TypeError && err.message.includes('fetch')) {
      throw new Error(
        `Cannot reach the backend at ${getApiBaseUrl()}. Is the server running?`
      )
    }

    throw err
  }
}

// ── Health check ──────────────────────────────────────────────────

export async function checkHealth(baseUrl?: string): Promise<HealthResponse> {
  const url = baseUrl ?? getApiBaseUrl()
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 5_000)

  try {
    const response = await fetch(`${url}/api/v1/health`, {
      signal: controller.signal,
    })
    clearTimeout(timeout)

    if (!response.ok) {
      return { status: 'error', components: {} }
    }

    return (await response.json()) as HealthResponse
  } catch {
    clearTimeout(timeout)
    return { status: 'unreachable', components: {} }
  }
}

// ── Ollama model list ─────────────────────────────────────────────

export async function getOllamaModels(): Promise<{
  models: string[]
  error?: string
}> {
  try {
    const response = await fetch(`${getApiBaseUrl()}/api/v1/ollama/models`)
    if (!response.ok) return { models: [] }
    return (await response.json()) as { models: string[]; error?: string }
  } catch {
    return { models: [], error: 'Cannot reach backend' }
  }
}

// ── Server settings ───────────────────────────────────────────────

export async function getServerSettings(): Promise<ServerSettings | null> {
  try {
    const response = await fetch(`${getApiBaseUrl()}/api/v1/settings`)
    if (!response.ok) return null
    return (await response.json()) as ServerSettings
  } catch {
    return null
  }
}

// ── Update server settings (write-back) ──────────────────────────

export type SettingsPatch = {
  llm_model?: string
  llm_base_url?: string
  embedding_model?: string
  reranker_model?: string
  reranker_enabled?: boolean
  top_k?: number
  use_query_transform?: boolean
  use_reranking?: boolean
}

export type SettingsUpdateResponse = {
  status: string
  changes?: Record<string, unknown>
  note?: string
  warning?: string
  yaml_status?: string
  message?: string
}

export async function updateServerSettings(
  patch: SettingsPatch
): Promise<SettingsUpdateResponse> {
  const response = await fetch(`${getApiBaseUrl()}/api/v1/settings`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  })

  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new Error(body?.detail ?? `Settings update failed: ${response.status}`)
  }

  return (await response.json()) as SettingsUpdateResponse
}
