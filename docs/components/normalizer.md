# Universal Normalizer

The `UniversalNormalizer` (`rag/pipeline/normalizer.py`) and its helper `FieldClassifier` (`rag/pipeline/field_classifier.py`) are the foundational components that decouple the Bright Data scraper from the RAG pipeline.

## 1. Intuition
Scraping arbitrary documentation sites produces heterogeneous JSON schemas. One site might use `page_title` and `summary`, while another uses `heading` and `description`. The **doc//rag** pipeline avoids breaking on unexpected schemas because the normalizer heuristically transforms these arbitrary JSON keys into a standardized `NormalizedDocument`.

## 2. Implementation: Field Classification

The `FieldClassifier.classify_fields()` method iterates over all keys in the raw scraped JSON and maps them to semantic roles (e.g., `title`, `description`, `code`, `parameter`).

### Matching Strategy
The system **does not** use heavy distance-based metrics like Levenshtein distance. Instead, it uses a lightweight, highly deterministic substring heuristic pattern-matcher via a hardcoded `_ROLE_REGISTRY`.

```python
_ROLE_REGISTRY = [
    ("title",        ["page_title", "notebook_title", "title", "name", "heading"]),
    ("description",  ["description", "summary", "abstract"]),
    # ...
]
```
The algorithm converts the incoming field name to lowercase and executes a substring check (`if pattern in lower:`). 

### Conflict Resolution
The `_ROLE_REGISTRY` is evaluated in a strict priority order (first match wins). For example, `NOTEBOOK` is checked before `NOTE` to prevent a field like `notebook_content` from falsely matching the substring `note`.

## 3. Normalized Document Generation

Once fields are bucketed into roles, the `UniversalNormalizer` maps the values into a `NormalizedDocument` object. This standardizes the data for the deduplication and chunking engines down the line, ensuring they only ever interact with known data structures regardless of the original scrape target.
