/**
 * Conversation and scrape job persistence.
 * All data is namespaced by user ID.
 */

import type { ChatMessage } from '../hooks/useRagChat'
import {
  getUserConversations,
  setUserConversations,
  getUserScrapeJobs,
  setUserScrapeJobs,
} from '../contexts/AuthContext'

// ── Types ─────────────────────────────────────────────────────────

export type ChatMode = 'llm' | 'rag'

export type StoredConversation = {
  id: string
  title: string
  mode: ChatMode
  sourceUrl: string | null
  scrapeJobId: string | null
  createdAt: number
  updatedAt: number
  messages: ChatMessage[]
}

export type StoredScrapeJob = {
  jobId: string
  sourceUrl: string
  status: string
  pagesFound: number
  pagesScraped: number
  chunksCreated: number
  createdAt: number
}

// ── Constants ─────────────────────────────────────────────────────

const MAX_AGE_DAYS = 30
const MAX_CONVERSATIONS = 100
const MAX_SCRAPE_JOBS = 50

// ── Conversation CRUD ─────────────────────────────────────────────

function readConversations(userId: string): StoredConversation[] {
  try {
    const raw = getUserConversations(userId)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed as StoredConversation[]
  } catch {
    return []
  }
}

function writeConversations(userId: string, conversations: StoredConversation[]): void {
  try {
    setUserConversations(userId, JSON.stringify(conversations))
  } catch {
    const pruned = pruneConversations(conversations)
    try {
      setUserConversations(userId, JSON.stringify(pruned))
    } catch {
      // give up silently
    }
  }
}

function pruneConversations(conversations: StoredConversation[]): StoredConversation[] {
  const cutoff = Date.now() - MAX_AGE_DAYS * 24 * 60 * 60 * 1000
  return conversations
    .filter((c) => c.updatedAt > cutoff)
    .slice(0, MAX_CONVERSATIONS)
}

export function saveConversation(userId: string, conversation: StoredConversation): void {
  const all = readConversations(userId)
  const index = all.findIndex((c) => c.id === conversation.id)
  if (index >= 0) {
    all[index] = conversation
  } else {
    all.unshift(conversation)
  }
  writeConversations(userId, all)
}

export function loadConversation(userId: string, id: string): StoredConversation | null {
  return readConversations(userId).find((c) => c.id === id) ?? null
}

export function listConversations(userId: string): Omit<StoredConversation, 'messages'>[] {
  const all = readConversations(userId)
  const pruned = pruneConversations(all)
  if (pruned.length !== all.length) {
    writeConversations(userId, pruned)
  }
  return pruned.map(({ messages, ...rest }) => rest)
}

export function deleteConversation(userId: string, id: string): void {
  const filtered = readConversations(userId).filter((c) => c.id !== id)
  writeConversations(userId, filtered)
}

// ── Scrape Job CRUD ───────────────────────────────────────────────

function readScrapeJobs(userId: string): StoredScrapeJob[] {
  try {
    const raw = getUserScrapeJobs(userId)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed as StoredScrapeJob[]
  } catch {
    return []
  }
}

function writeScrapeJobs(userId: string, jobs: StoredScrapeJob[]): void {
  try {
    setUserScrapeJobs(userId, JSON.stringify(jobs.slice(0, MAX_SCRAPE_JOBS)))
  } catch {
    // silent fail
  }
}

export function saveScrapeJob(userId: string, job: StoredScrapeJob): void {
  const all = readScrapeJobs(userId)
  const index = all.findIndex((j) => j.jobId === job.jobId)
  if (index >= 0) {
    all[index] = job
  } else {
    all.unshift(job)
  }
  writeScrapeJobs(userId, all)
}

export function listScrapeJobs(userId: string): StoredScrapeJob[] {
  return readScrapeJobs(userId)
}

export function getCompletedScrapeJobs(userId: string): StoredScrapeJob[] {
  return readScrapeJobs(userId).filter((j) => j.status === 'done')
}

// ── Helpers ───────────────────────────────────────────────────────

export function generateTitle(messages: ChatMessage[], mode: ChatMode, sourceUrl?: string | null): string {
  const firstUser = messages.find((m) => m.role === 'user')
  const text = firstUser?.content.trim() || 'New conversation'

  if (mode === 'rag' && sourceUrl) {
    try {
      const hostname = new URL(sourceUrl).hostname
      const shortText = text.length > 35 ? text.slice(0, 35) + '...' : text
      return `${hostname} — ${shortText}`
    } catch {
      // fall through
    }
  }

  return text.length > 50 ? text.slice(0, 50) + '...' : text
}

export function formatRelativeTime(timestamp: number): string {
  const seconds = Math.floor((Date.now() - timestamp) / 1000)
  if (seconds < 60) return 'just now'
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`
  if (seconds < 604800) return `${Math.floor(seconds / 86400)}d ago`
  return new Date(timestamp).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}
