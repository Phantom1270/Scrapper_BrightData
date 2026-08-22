import type { QuerySource } from './api'

export type ConfidenceLevel = 'high' | 'medium' | 'low'

export type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: QuerySource[]
  confidence?: ConfidenceLevel
  retrievalTimeMs?: number
  generationTimeMs?: number
  transformUsed?: string
  rerankerUsed?: string
  isError?: boolean
}

export type AdvancedSettings = {
  topK: number
  filterContentType: string
  useQueryTransform: boolean
  useReranking: boolean
}
