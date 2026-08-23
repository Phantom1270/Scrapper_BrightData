import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  type ReactNode,
} from 'react'

// ── Hardcoded users ──────────────────────────────────────────────

const USERS: Record<string, { password: string; displayName: string }> = {
  admin: { password: 'admin123', displayName: 'Admin' },
  demo:  { password: 'demo123',  displayName: 'Demo User' },
}

// ── Types ────────────────────────────────────────────────────────

export type User = {
  id: string
  displayName: string
}

type AuthContextValue = {
  user: User | null
  login: (username: string, password: string) => boolean
  logout: () => void
  isAuthenticated: boolean
}

// ── Storage helpers ──────────────────────────────────────────────

const USER_KEY = 'rag_current_user'

function loadUser(): User | null {
  try {
    const raw = localStorage.getItem(USER_KEY)
    if (!raw) return null
    const userId = JSON.parse(raw)
    if (typeof userId === 'string' && USERS[userId]) {
      return { id: userId, displayName: USERS[userId].displayName }
    }
    return null
  } catch {
    return null
  }
}

function saveUser(userId: string): void {
  localStorage.setItem(USER_KEY, JSON.stringify(userId))
}

function clearUser(): void {
  localStorage.removeItem(USER_KEY)
}

// ── Context ──────────────────────────────────────────────────────

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(loadUser)

  const login = useCallback((username: string, password: string): boolean => {
    const account = USERS[username]
    if (!account || account.password !== password) {
      return false
    }
    const newUser: User = { id: username, displayName: account.displayName }
    saveUser(username)
    setUser(newUser)
    return true
  }, [])

  const logout = useCallback(() => {
    clearUser()
    setUser(null)
  }, [])

  return (
    <AuthContext.Provider
      value={{
        user,
        login,
        logout,
        isAuthenticated: user !== null,
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext)
  if (!ctx) {
    throw new Error('useAuth must be used inside an <AuthProvider>')
  }
  return ctx
}

// ── User-scoped storage helpers ──────────────────────────────────
// All localStorage keys are namespaced by user ID to isolate data.

export function userKey(userId: string, suffix: string): string {
  return `rag_user_${userId}_${suffix}`
}

export function getUserSettings(userId: string): string | null {
  return localStorage.getItem(userKey(userId, 'settings'))
}

export function setUserSettings(userId: string, value: string): void {
  localStorage.setItem(userKey(userId, 'settings'), value)
}

export function getUserConversations(userId: string): string | null {
  return localStorage.getItem(userKey(userId, 'conversations'))
}

export function setUserConversations(userId: string, value: string): void {
  localStorage.setItem(userKey(userId, 'conversations'), value)
}

export function getUserScrapeJobs(userId: string): string | null {
  return localStorage.getItem(userKey(userId, 'scrape_jobs'))
}

export function setUserScrapeJobs(userId: string, value: string): void {
  localStorage.setItem(userKey(userId, 'scrape_jobs'), value)
}
