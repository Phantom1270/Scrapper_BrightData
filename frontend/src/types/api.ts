export type QueryRequest = {
  question: string
  top_k?: number
  filter_content_type?: string
  use_query_transform?: boolean
  use_reranking?: boolean
}

export type QuerySource = {
  chunk_id: string
  heading: string
  url: string
  score: number
  content_type?: string
  source?: string
}

export type QueryResponse = {
  answer: string
  sources: QuerySource[]
  confidence?: string
  retrieval_time_ms?: number
  generation_time_ms?: number
  transform_used?: string
  reranker_used?: string
}
