# doc//rag — Documentation

Welcome to the internal documentation for the **doc//rag** system. This project is a full-stack Retrieval-Augmented Generation platform combining a production-grade Python RAG pipeline with a dark-themed React chat interface.

---

## Table of Contents

### Getting Started
- [Setup & Installation](../README.md#setup--installation)
- [Usage Guide](../README.md#usage-guide)

### Architecture
- [System Overview](architecture/overview.md)
- [Ingestion Pipeline](architecture/ingestion.md)
- [Retrieval Pipeline](architecture/retrieval.md)

### Frontend
- [Application Structure](#frontend-application)
- [Component Reference](#components)
- [Theming & CSS Variables](#theming)

### Backend Components
- [Normalizer & Field Classifier](components/normalizer.md)
- [Deduplication Engine](components/deduplication.md)
- [Context-Aware Chunking](components/chunking.md)
- [Hybrid Search & RRF](components/hybrid-search.md)
- [Query Transformation](components/query-transformation.md)
- [Cross-Encoder Reranking](components/reranking.md)

### API
- [API Reference](api/reference.md)

### Evaluation
- [Evaluation & Performance](evaluation/performance.md)

### Design Decisions
- [ADR 001: Hybrid Retrieval](decisions/001-hybrid-retrieval.md)
- [ADR 002: Deduplication Strategy](decisions/002-deduplication.md)

---

## Frontend Application

The frontend is a **React 18 + TypeScript** SPA built with Vite. It uses Vanilla CSS with a dark theme and CSS custom properties for theming.

### Application Views

The app is a single-page application that transitions between these views, managed in `App.tsx`:

| View | Component | Description |
|:---|:---|:---|
| `login` | `LoginScreen` | Authentication gate (shown when unauthenticated) |
| `welcome` | `WelcomeScreen` | Landing page with New Chat / Scrape actions |
| `modeSelect` | `ChatModeSelector` | Choose between ⚡ LLM Chat or 🔍 RAG Chat |
| `scraping` | `ScrapeFlow` | Multi-step scraping configuration & progress |
| `chat` | `ChatHeader` + `MessageList` + `Composer` | Full chat interface |

---

## Components

### `LoginScreen`
Authentication screen with username/password form. Credentials are validated against a hardcoded demo store in `AuthContext`.

**Demo credentials:**
- `admin` / `admin123`
- `demo` / `demo123`

---

### `WelcomeScreen`
Landing page shown after login. Displays the **doc//rag** branding and two action cards: **New Chat** and **Scrape a Website**.

---

### `ChatModeSelector`
Lets the user pick a chat mode before starting a conversation:
- **⚡ LLM Chat** — Direct LLM conversation without document context
- **🔍 RAG Chat** — Triggers the scrape flow, then enables document-grounded answers

---

### `ScrapeFlow`
Multi-step flow for configuring and running a website scrape:
1. Enter the documentation URL
2. Configure depth, max pages, CSS selectors, and advanced options
3. Monitor live scraping progress with step indicators
4. On completion, transitions to the chat view

---

### `ChatSidebar`
Left sidebar containing:
- **doc//rag** brand logo and title
- **New Chat** button
- **Scrape** button
- List of past conversations with timestamps, mode badges, and source URL
- Delete confirmation per conversation
- User avatar, display name, and logout button at the bottom
- Status bar (backend connectivity indicator)

---

### `ChatHeader`
Top bar of the chat pane. Shows:
- **doc//rag** title
- Mode badge (`⚡ LLM` or `🔍 RAG`)
- Connected source label (hostname of scraped URL for RAG mode)

---

### `MessageList`
Scrollable message feed. Features:
- User bubbles (purple accent, white text, right-aligned)
- Assistant bubbles (dark surface, left-aligned)
- Markdown rendering (headers, code blocks, lists, tables, blockquotes)
- Typing indicator during loading
- Per-response metadata: confidence badge, retrieval time, generation time, pipeline info
- Collapsible **sources** section with score bar, content type badge, heading link, and snippet preview
- Copy button on each bubble
- Delete message-pair button (removes the Q&A pair from history)

---

### `Composer`
Input area at the bottom of the chat pane:
- Auto-resizing textarea (up to 160px tall)
- **Send** button (disabled while loading)
- **⚙️ Settings** button to open the settings panel
- Focus ring border using `--accent` color

---

### `SettingsPanel`
Modal panel for configuring the RAG pipeline at runtime:
- Backend URL
- LLM model name
- Top K (number of chunks to retrieve)
- Query transformation toggle
- Reranking toggle
- Save settings to server
- Reset / wipe local data

---

### `StatusBar`
Small indicator inside the sidebar showing backend connectivity status (online / offline / checking).

---

## Theming

Global CSS variables are defined in `frontend/src/index.css`:

```css
:root {
  /* Backgrounds */
  --bg:             #212121;   /* Page background */
  --surface:        #171717;   /* Card / panel background */
  --surface-subtle: #2a2a2a;   /* Subtle surface */
  --surface-hover:  #333333;   /* Hover state */

  /* Borders */
  --border:         #333333;
  --border-strong:  #444444;

  /* Text */
  --text-primary:   #ffffff;
  --text-secondary: #cccccc;
  --text-muted:     #888888;

  /* Accent (brand color) */
  --accent:         #7c6af7;   /* Purple — used for buttons, badges, highlights */
  --accent-hover:   #6c5ae7;
  --accent-bg:      rgba(124, 106, 247, 0.12);
  --accent-border:  rgba(124, 106, 247, 0.3);

  /* Semantic */
  --link:           #a89ff8;
  --success:        #00b894;
  --warning:        #fdcb6e;
  --error:          #e17055;
}
```

> **Note:** The Login page overrides `--accent` locally — the logo box and Sign In button use `#111` (black) with white text, intentionally deviating from the global accent for a clean, branded entry point.

---

## Data Flow (Frontend → Backend)

```
User types question
        │
        ▼
Composer → useRagChat hook
        │
        ▼
POST /api/v1/query
{
  question: "...",
  top_k: 5,
  use_query_transform: true,
  use_reranking: true
}
        │
        ▼
FastAPI → RAG Pipeline → JSON response
{
  answer: "...",
  sources: [...],
  confidence: "high",
  retrieval_time_ms: ...,
  generation_time_ms: ...
}
        │
        ▼
MessageList renders answer + sources
```

---

## Conversation Storage

Conversations are persisted to **localStorage** via `frontend/src/services/storage.ts`.  
Each conversation stores:
- ID, title, mode (`llm` | `rag`), source URL
- Full message history (role, content, sources, metadata)
- Created/updated timestamps

No backend persistence is required for conversation history.
