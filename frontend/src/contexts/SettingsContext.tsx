import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  type ReactNode,
} from 'react'
import { useAuth, getUserSettings, setUserSettings } from './AuthContext'

export type AppSettings = {
  apiBaseUrl: string
  topK: number
  filterContentType: string
  useQueryTransform: boolean
  useReranking: boolean
  llmModel: string
  rerankerModel: string
  embeddingModel: string
}

const DEFAULTS: AppSettings = {
  apiBaseUrl: 'http://127.0.0.1:8000',
  topK: 5,
  filterContentType: '',
  useQueryTransform: true,
  useReranking: true,
  llmModel: '',
  rerankerModel: '',
  embeddingModel: '',
}

function loadForUser(userId: string): AppSettings {
  try {
    const raw = getUserSettings(userId)
    if (!raw) return { ...DEFAULTS }
    return { ...DEFAULTS, ...JSON.parse(raw) }
  } catch {
    return { ...DEFAULTS }
  }
}

type SettingsContextValue = {
  settings: AppSettings
  updateSettings: (patch: Partial<AppSettings>) => void
  resetSettings: () => void
}

const SettingsContext = createContext<SettingsContextValue | null>(null)

export function SettingsProvider({ children }: { children: ReactNode }) {
  const { user } = useAuth()

  const [settings, setSettings] = useState<AppSettings>(
    user ? loadForUser(user.id) : { ...DEFAULTS }
  )

  // Reload settings when user changes
  useEffect(() => {
    if (user) {
      setSettings(loadForUser(user.id))
    } else {
      setSettings({ ...DEFAULTS })
    }
  }, [user])

  // Persist on change
  useEffect(() => {
    if (user) {
      try {
        setUserSettings(user.id, JSON.stringify(settings))
      } catch {
        // silent fail
      }
    }
  }, [settings, user])

  const updateSettings = useCallback((patch: Partial<AppSettings>) => {
    setSettings((prev) => ({ ...prev, ...patch }))
  }, [])

  const resetSettings = useCallback(() => {
    setSettings({ ...DEFAULTS })
  }, [])

  return (
    <SettingsContext.Provider value={{ settings, updateSettings, resetSettings }}>
      {children}
    </SettingsContext.Provider>
  )
}

export function useSettings(): SettingsContextValue {
  const ctx = useContext(SettingsContext)
  if (!ctx) throw new Error('useSettings must be used inside a <SettingsProvider>')
  return ctx
}
