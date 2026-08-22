import type { QueryRequest, QueryResponse } from '../types/api'

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'
const QUERY_ENDPOINT = '/api/v1/query'

export async function queryRag(payload: QueryRequest): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE_URL}${QUERY_ENDPOINT}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })

  if (!response.ok) {
    const responseBody = await response.text()
    throw new Error(
      `Query request failed with status ${response.status}: ${responseBody || 'No response body'}`
    )
  }

  return (await response.json()) as QueryResponse
}
