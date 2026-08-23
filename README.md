# doc//rag

A full-stack, production-oriented Retrieval-Augmented Generation (RAG) system — featuring an intelligent document ingestion pipeline, hybrid search engine, cross-encoder reranking, and a polished React chat interface.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.0+-green.svg)
![React](https://img.shields.io/badge/React-18+-61DAFB.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange.svg)
![Vite](https://img.shields.io/badge/Vite-Frontend-646CFF.svg)

---

## Overview

**doc//rag** is a two-part system:

1. **RAG Backend** — A Python/FastAPI pipeline that ingests raw scraped JSON, normalises it, deduplicates it using LSH, embeds it, and serves answers via a hybrid dense + sparse retrieval engine with cross-encoder reranking.
2. **Chat Frontend** — A React + TypeScript SPA that lets users log in, scrape a documentation URL, and chat with the knowledge base in real-time, or use a direct LLM chat mode.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     FRONTEND (React)                     │
│  Login → Welcome → Mode Select → Scrape Flow → Chat     │
│  Components: ChatSidebar, MessageList, Composer,         │
│              ChatHeader, SettingsPanel, WelcomeScreen    │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP POST /api/v1/query
┌────────────────────▼────────────────────────────────────┐
│                  BACKEND (FastAPI)                        │
│                                                          │
│  User Query                                              │
│      │                                                   │
│      ▼                                                   │
│  Query Transformation (qwen2.5:3b via Ollama)           │
│      │                                                   │
│      ├──────────────────────┐                           │
│      ▼                      ▼                           │
│  Dense Retrieval        Sparse Retrieval                │
│  (ChromaDB +            (BM25 keyword                   │
│   all-MiniLM-L6-v2)      index)                         │
│      │                      │                           │
│      └──────────┬───────────┘                           │
│                 ▼                                        │
│         RRF Fusion                                       │
│                 │                                        │
│                 ▼                                        │
│    Cross-Encoder Reranking                               │
│    (ms-marco-MiniLM-L-6-v2)                             │
│                 │                                        │
│                 ▼                                        │
│    LLM Answer Generation (qwen2.5:3b)                   │
│                 │                                        │
│                 ▼                                        │
│           JSON Response                                  │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|:---|:---|:---|
| **Frontend** | React 18 + TypeScript | Chat UI |
| **Build Tool** | Vite | Frontend dev server & bundler |
| **Styling** | Vanilla CSS | Component styles |
| **API Framework** | FastAPI | REST API |
| **ASGI Server** | Uvicorn | API server |
| **Vector Store** | ChromaDB | Dense semantic retrieval |
| **Embeddings** | all-MiniLM-L6-v2 | Sentence embeddings |
| **Sparse Retrieval** | BM25 | Keyword retrieval |
| **Local LLM** | Ollama / qwen2.5:3b | Query expansion + generation |
| **Reranking** | ms-marco-MiniLM-L-6-v2 | Cross-encoder reranking |
| **Scraper** | Bright Data | Web scraping (external) |

---

## Project Structure

```
doc-rag/
├── frontend/                  # React + TypeScript SPA
│   ├── src/
│   │   ├── components/        # UI components
│   │   │   ├── ChatHeader.tsx/css
│   │   │   ├── ChatModeSelector.tsx/css
│   │   │   ├── ChatSidebar.tsx/css
│   │   │   ├── Composer.tsx/css
│   │   │   ├── LoginScreen.tsx/css
│   │   │   ├── MessageList.tsx/css
│   │   │   ├── ScrapeFlow.tsx/css
│   │   │   ├── SettingsPanel.tsx/css
│   │   │   ├── StatusBar.tsx/css
│   │   │   └── WelcomeScreen.tsx/css
│   │   ├── contexts/          # React contexts (Auth, Settings)
│   │   ├── hooks/             # Custom hooks (useRagChat)
│   │   ├── services/          # API client, storage, scraper
│   │   ├── types/             # TypeScript type definitions
│   │   ├── App.tsx            # Root app component
│   │   └── index.css          # Global CSS variables & theme
│   ├── index.html
│   └── package.json
│
├── rag/                       # Python RAG backend
│   ├── chunking/              # Context-aware text splitters
│   ├── config/                # Settings & Pydantic config
│   ├── generation/            # LLM answer generation
│   ├── llm/                   # Ollama LLM client
│   ├── models/                # Data models & schemas
│   ├── pipeline/              # Normalizer & deduplicator
│   ├── retrieval/             # Query transform, RRF, reranking
│   ├── search/                # Vector store, BM25, embeddings
│   ├── serving/               # FastAPI app & request schemas
│   ├── storage/               # SQLite metadata store
│   └── utils/                 # Helper utilities
│
├── docs/                      # Extended architecture docs
├── ingest.py                  # CLI: run the full ingestion pipeline
├── requirements.txt           # Python dependencies
└── README.md
```

---

## System Requirements

- **Python:** 3.9+
- **Node.js:** 18+ (for the frontend)
- **RAM:** 8 GB minimum, 16 GB recommended
- **[Ollama](https://ollama.com/):** Running locally for query transformation and answer generation

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/Phantom1270/Scrapper_BrightData.git
cd Scrapper_BrightData
```

### 2. Set up the Python backend

```bash
# Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Set up Ollama (local LLM)

Install [Ollama](https://ollama.com/) for your OS, then pull the required model:

```bash
ollama pull qwen2.5:3b
```

Verify it's running:

```bash
ollama list
```

### 4. Ingest your data

Before starting the API, run the ingestion pipeline to normalise, deduplicate, embed, and index your scraped JSON file:

```bash
python ingest.py --input "phase3_output (2).json"
```

To force a full rebuild of the vector and BM25 indexes:

```bash
python ingest.py --input "phase3_output (2).json" --force-rebuild
```

### 5. Start the backend API

```bash
python -m uvicorn rag.serving.app:create_app --factory --reload --host 127.0.0.1 --port 8000
```

The API will be available at `http://127.0.0.1:8000`.  
Interactive API docs: `http://127.0.0.1:8000/docs`

### 6. Set up and start the frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:5173`.

---

## Usage Guide

### Logging In

Open `http://localhost:5173` in your browser. You will be presented with the **doc//rag** login screen.

**Demo credentials:**

| Username | Password |
|:---|:---|
| `admin` | `admin123` |
| `demo` | `demo123` |

### Choosing a Chat Mode

After logging in, you will see the Welcome screen. Click **New Chat** to choose a mode:

| Mode | Description |
|:---|:---|
| ⚡ **LLM Chat** | Direct conversation with the LLM (no document context) |
| 🔍 **RAG Chat** | Scrape a documentation URL, then chat with it using the full RAG pipeline |

### Scraping a Documentation URL (RAG Mode)

1. Select **RAG Chat** and enter a documentation URL (e.g., `https://scikit-learn.org/stable/`).
2. Configure optional scraping parameters (depth, max pages, CSS selectors).
3. Click **Start Scraping** — the system will crawl the site and build a knowledge base.
4. Once complete, you will be taken directly to the chat interface.

### Chatting

- Type your question in the composer and press **Enter** or click the send button.
- The assistant will retrieve the most relevant document chunks, rerank them, and generate a grounded answer.
- Each response shows the **confidence level**, **retrieval time**, **generation time**, and **source documents**.
- Click **Show sources** on any response to see exactly which document chunks were used.

### Settings Panel

Click the ⚙️ gear icon in the composer to open the Settings panel, where you can configure:

- **Backend URL** — point to a remote API if needed
- **LLM Model** — change the Ollama model
- **Top K** — number of document chunks to retrieve
- **Query Transformation** — enable/disable AI query expansion
- **Reranking** — enable/disable cross-encoder reranking

### Conversation History

All conversations are saved locally in your browser. The left sidebar shows your history — click any conversation to reload it, or delete it with the `×` button.

---

## API Reference

### `POST /api/v1/query`

Runs an end-to-end RAG query against the ingested knowledge base.

**Request body:**

```json
{
  "question": "How do I use sklearn config_context?",
  "top_k": 5,
  "filter_content_type": "api_reference",
  "use_query_transform": true,
  "use_reranking": true
}
```

| Field | Type | Required | Default | Description |
|:---|:---|:---|:---|:---|
| `question` | string | ✅ | — | The user's question |
| `top_k` | integer | ❌ | 5 | Number of chunks to retrieve |
| `filter_content_type` | string | ❌ | null | Filter: `"prose"`, `"code"`, `"api_reference"` |
| `filter_doc_id` | string | ❌ | null | Filter by a specific document ID |
| `use_query_transform` | boolean | ❌ | true | Enable AI query expansion |
| `use_reranking` | boolean | ❌ | true | Enable cross-encoder reranking |

**Response:**

```json
{
  "answer": "The purpose of sklearn.config_context is...",
  "sources": [
    {
      "chunk_id": "9bf961cfdf1ea8d1",
      "heading": "sklearn.config_context > Description",
      "url": "https://scikit-learn.org/stable/modules/...",
      "score": 7.338,
      "content_type": "prose",
      "source": "hybrid"
    }
  ],
  "confidence": "high",
  "retrieval_time_ms": 1619.3,
  "generation_time_ms": 7555.6,
  "total_time_ms": 9174.9,
  "cached": false,
  "llm_model": "qwen2.5:3b",
  "transform_used": "MultiQueryTransformer",
  "reranker_used": "cross-encoder/ms-marco-MiniLM-L-6-v2"
}
```

---

## Key Features

- **Universal Normalizer** — Schema-agnostic ingestion; auto-classifies arbitrary JSON fields via fuzzy matching into standardised `NormalizedDocument` objects.
- **O(n) Deduplication** — MinHash + Exact Hash LSH removes exact and near-duplicate documents before they enter the vector store.
- **Context-Aware Chunking** — Respects code block, parameter list, and prose boundaries when splitting documents.
- **Hybrid Retrieval** — Dense (ChromaDB + `all-MiniLM-L6-v2`) and Sparse (BM25) retrieval fused via Reciprocal Rank Fusion (RRF).
- **Query Transformation** — Local LLM generates multiple semantic variations of the user's query to maximise retrieval coverage.
- **Cross-Encoder Reranking** — Deep pairwise scoring of retrieved candidates to surface the most relevant chunks.
- **Full Chat Frontend** — Dark-themed React SPA with login, conversation history, mode selector, scrape flow, settings panel, and source citation display.
- **LLM Chat Mode** — Fallback direct LLM conversation without document context.

---

## Limitations

- **Local LLM speed** — Query transformation and generation are bottlenecked by local CPU/GPU throughput.
- **Static index** — The knowledge base is rebuilt by re-running `ingest.py`; real-time incremental updates are not yet implemented.
- **No authentication on the API** — The FastAPI server runs without API keys or rate limiting; intended for local/internal use only.

---

## Roadmap

- [ ] **Streaming responses** — SSE or WebSocket token-by-token streaming to the frontend.
- [ ] **Semantic caching** — Redis layer to return cached results for semantically similar queries in ~50ms.
- [ ] **Incremental ingestion** — Add new documents to the index without a full rebuild.
- [ ] **RAGAS evaluation** — Automated scoring of context precision, recall, and hallucination rate.
- [ ] **API authentication** — API key validation and rate limiting for public deployment.
- [ ] **Distributed vector store** — Migrate from local ChromaDB to a managed/distributed solution.
