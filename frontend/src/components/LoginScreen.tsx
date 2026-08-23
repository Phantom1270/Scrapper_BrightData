import { useState, type FormEvent } from 'react'
import './LoginScreen.css'

type Props = {
  onLogin: (username: string, password: string) => boolean
}

export function LoginScreen({ onLogin }: Props) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [shake, setShake] = useState(false)

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault()
    setError('')

    if (!username.trim() || !password.trim()) {
      setError('Please enter both username and password.')
      triggerShake()
      return
    }

    const success = onLogin(username.trim(), password)
    if (!success) {
      setError('Invalid username or password.')
      triggerShake()
    }
  }

  const triggerShake = () => {
    setShake(true)
    setTimeout(() => setShake(false), 400)
  }

  return (
    <div className="login-screen">
      <div className={`login-card ${shake ? 'login-shake' : ''}`}>
        <div className="login-header">
          <div className="login-logo">R</div>
          <h1>RAG Assistant</h1>
          <p>Sign in to your account</p>
        </div>

        <form className="login-form" onSubmit={handleSubmit}>
          <div className="login-field">
            <label htmlFor="username">Username</label>
            <input
              id="username"
              type="text"
              value={username}
              onChange={(e) => {
                setUsername(e.target.value)
                if (error) setError('')
              }}
              placeholder="admin"
              autoFocus
              autoComplete="username"
            />
          </div>

          <div className="login-field">
            <label htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => {
                setPassword(e.target.value)
                if (error) setError('')
              }}
              placeholder="••••••••"
              autoComplete="current-password"
            />
          </div>

          {error && <span className="login-error">{error}</span>}

          <button type="submit" className="login-btn">
            Sign In
          </button>
        </form>

        <div className="login-hint">
          <p>Demo credentials:</p>
          <code>admin / admin123</code> or <code>demo / demo123</code>
        </div>
      </div>
    </div>
  )
}
