# RAG Pipeline Documentation

Welcome to the internal documentation for the RAG architecture pipeline. This repository is built as a highly decoupled, production-oriented system that ingests Bright Data scraped JSON, normalizes it, deduplicates it using LSH, and serves it via a hybrid retrieval engine with cross-encoder reranking.

## Table of Contents

### Architecture
- [Architecture Overview](architecture/overview.md)
- [Ingestion Pipeline](architecture/ingestion.md)
- [Retrieval Pipeline](architecture/retrieval.md)

### Components
- [Normalizer & Field Classifier](components/normalizer.md)
- [Deduplication Engine](components/deduplication.md)
- [Context-Aware & Parent-Child Chunking](components/chunking.md)
- [Hybrid Search & RRF](components/hybrid-search.md)
- [Query Transformation](components/query-transformation.md)
- [Cross-Encoder Reranking](components/reranking.md)

### API
- [API Reference](api/reference.md)

### Development
- [Setup & Ingestion](development/setup.md)
- [Project Structure](development/project-structure.md)

### Evaluation
- [Evaluation & Performance](evaluation/performance.md)

### Design Decisions
- [ADR 001: Hybrid Retrieval](decisions/001-hybrid-retrieval.md)
- [ADR 002: Deduplication Strategy](decisions/002-deduplication.md)
