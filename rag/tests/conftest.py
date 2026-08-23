"""
Shared pytest fixtures for RAG tests.

All test modules import from here via conftest.py auto-discovery.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List

import pytest

# Ensure the project root (d:/projects/Scrapper) is on the path
# so that `from rag.xxx import ...` works when running from any directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rag.models.chunk import Chunk
from rag.models.document import ContentBlock, NormalizedDocument
from rag.storage.sqlite_store import SQLiteStore
from rag.utils.ids import generate_chunk_id, generate_doc_id


# ---------------------------------------------------------------------------
# Sample documents
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_document() -> NormalizedDocument:
    """
    A realistic API reference document resembling scikit-learn's config_context.
    Contains 5+ blocks of varied types.
    """
    doc_id = generate_doc_id("https://scikit-learn.org/stable/modules/generated/sklearn.config_context.html")
    return NormalizedDocument(
        doc_id=doc_id,
        url="https://scikit-learn.org/stable/modules/generated/sklearn.config_context.html",
        title="config_context — scikit-learn 1.4 documentation",
        description=(
            "Context manager for global scikit-learn configuration. "
            "New configuration will be effective only in the with block."
        ),
        content_blocks=[
            ContentBlock(
                block_type="function_signature",
                text="sklearn.config_context(*, assume_finite=False, working_memory=1024, print_changed_only=True, display='diagram', pairwise_dist_chunk_n_steps=10, enable_cython_pxd=True)",
                heading="sklearn.config_context",
            ),
            ContentBlock(
                block_type="prose",
                text=(
                    "Context manager for global scikit-learn configuration. "
                    "New configuration will be effective only in the ``with`` block. "
                    "Parameters set via config_context override global settings from "
                    "set_config but do not affect subsequent calls to set_config."
                ),
                heading="Description",
            ),
            ContentBlock(
                block_type="parameter_list",
                text="assume_finite : bool, default=False\n    If True, validation of array elements for finiteness is skipped.\n\nworking_memory : int, default=1024\n    If set, scikit-learn will attempt to limit the size of temporary arrays to this number of MiB.",
                heading="Parameters",
                structured_data={
                    "parameters": [
                        {"name": "assume_finite", "type_info": "bool, default=False", "description": "If True, validation is skipped."},
                        {"name": "working_memory", "type_info": "int, default=1024", "description": "Limit for temporary arrays in MiB."},
                    ]
                },
            ),
            ContentBlock(
                block_type="code",
                text='>>> import sklearn\n>>> from sklearn import set_config\n>>> set_config(display="diagram")\n>>> with sklearn.config_context(assume_finite=True):\n...     pass',
                heading="Examples",
                language="python",
            ),
            ContentBlock(
                block_type="note",
                text="All parameter constraints apply within the context block only. Changes revert after the ``with`` block exits.",
                heading="Notes",
            ),
            ContentBlock(
                block_type="prose",
                text="See also: set_config, get_config for persistent global configuration.",
                heading="See Also",
            ),
        ],
        metadata={
            "function_name": "sklearn.config_context",
            "module": "sklearn",
            "version": "1.4",
        },
        template_id="tpl_002",
        content_type="api_reference",
        source_link="https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/_config.py",
    )


@pytest.fixture
def sample_document_tutorial() -> NormalizedDocument:
    """
    A tutorial-style document with sections and mixed content.
    """
    doc_id = generate_doc_id("https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html")
    return NormalizedDocument(
        doc_id=doc_id,
        url="https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html",
        title="Linear Regression Example — scikit-learn",
        description="A simple linear regression example using scikit-learn.",
        content_blocks=[
            ContentBlock(
                block_type="prose",
                text="This example shows how to use LinearRegression to fit a simple linear model.",
                heading="Introduction",
            ),
            ContentBlock(
                block_type="code",
                text="import numpy as np\nfrom sklearn.linear_model import LinearRegression\n\nX = np.array([[1], [2], [3]])\ny = np.array([2, 4, 6])\nmodel = LinearRegression().fit(X, y)\nprint(model.coef_)",
                heading="Code",
                language="python",
            ),
            ContentBlock(
                block_type="prose",
                text="The model should output a coefficient of approximately 2.0.",
                heading="Results",
            ),
        ],
        metadata={"example_type": "plot", "complexity": "beginner"},
        template_id="tpl_005",
        content_type="tutorial",
    )


@pytest.fixture
def sample_document_error() -> NormalizedDocument:
    """
    A document representing a failed extraction (dead page or parse error).
    """
    doc_id = generate_doc_id("https://scikit-learn.org/stable/dead_page.html")
    return NormalizedDocument(
        doc_id=doc_id,
        url="https://scikit-learn.org/stable/dead_page.html",
        title="",
        description="",
        content_blocks=[],
        metadata={},
        template_id="tpl_002",
        content_type="unknown",
        error="HTTP 404: Page not found",
    )


# ---------------------------------------------------------------------------
# Sample chunks
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_chunks(sample_document) -> List[Chunk]:
    """
    5 chunks derived from sample_document, with varied types and heading paths.
    """
    doc_id = sample_document.doc_id
    url = sample_document.url

    return [
        Chunk(
            chunk_id=generate_chunk_id(doc_id, 0),
            doc_id=doc_id,
            url=url,
            content="sklearn.config_context(*, assume_finite=False, working_memory=1024)",
            content_type="function_signature",
            heading_path=["API Reference", "sklearn.config_context"],
            chunk_index=0,
            token_count=20,
            block_type="function_signature",
        ),
        Chunk(
            chunk_id=generate_chunk_id(doc_id, 1),
            doc_id=doc_id,
            url=url,
            content="Context manager for global scikit-learn configuration.",
            content_type="prose",
            heading_path=["API Reference", "sklearn.config_context", "Description"],
            chunk_index=1,
            token_count=10,
            block_type="prose",
        ),
        Chunk(
            chunk_id=generate_chunk_id(doc_id, 2),
            doc_id=doc_id,
            url=url,
            content="assume_finite : bool, default=False\n    If True, validation is skipped.",
            content_type="parameter_list",
            heading_path=["API Reference", "sklearn.config_context", "Parameters"],
            chunk_index=2,
            token_count=18,
            block_type="parameter_list",
        ),
        Chunk(
            chunk_id=generate_chunk_id(doc_id, 3),
            doc_id=doc_id,
            url=url,
            content=">>> with sklearn.config_context(assume_finite=True):\n...     pass",
            content_type="code",
            heading_path=["API Reference", "sklearn.config_context", "Examples"],
            chunk_index=3,
            token_count=22,
            block_type="code",
            language="python",
        ),
        Chunk(
            chunk_id=generate_chunk_id(doc_id, 4),
            doc_id=doc_id,
            url=url,
            content="Changes revert after the with block exits.",
            content_type="prose",
            heading_path=["API Reference", "sklearn.config_context", "Notes"],
            chunk_index=4,
            token_count=9,
            block_type="note",
        ),
    ]


# ---------------------------------------------------------------------------
# Sample scraped JSON (Phase 3 format)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_scraped_json(tmp_path) -> Path:
    """
    A minimal but complete Phase 3 scraped JSON written to a temp file.
    Returns the path to the file.
    """
    payload = {
        "domain": "scikit-learn.org",
        "total_processed": 6,
        "total_healed": 0,
        "failed": 1,
        "results": {
            "tpl_002": [
                {
                    "url": "https://scikit-learn.org/stable/modules/generated/sklearn.config_context.html",
                    "status": "extracted",
                    "data": [
                        {
                            "page_title": "sklearn.config_context",
                            "description": "Context manager for global scikit-learn configuration.",
                            "function_signature": "sklearn.config_context(*)",
                            "parameters": [
                                {"name": "assume_finite", "type_info": "bool", "description": "Skip validation."}
                            ],
                            "code_examples": [">>> with sklearn.config_context(assume_finite=True):\n...     pass"],
                            "see_also": [{"function_name": "set_config", "function_description": "Set global config."}],
                            "notes": "Changes revert after with block.",
                            "source_link": "https://github.com/scikit-learn/scikit-learn",
                            "input": {"url": "https://scikit-learn.org/stable/modules/generated/sklearn.config_context.html"},
                        }
                    ],
                },
                {
                    "url": "https://scikit-learn.org/stable/modules/generated/sklearn.set_config.html",
                    "status": "extracted",
                    "data": [
                        {
                            "page_title": "sklearn.set_config",
                            "description": "Set global scikit-learn configuration.",
                            "function_signature": "sklearn.set_config(**kwargs)",
                            "parameters": [],
                            "code_examples": [],
                            "see_also": [],
                            "notes": "",
                            "source_link": "",
                            "input": {"url": "https://scikit-learn.org/stable/modules/generated/sklearn.set_config.html"},
                        }
                    ],
                },
                {
                    "url": "https://scikit-learn.org/stable/modules/generated/sklearn.dead_page.html",
                    "status": "failed",
                    "data": [],
                    "error": "HTTP 404",
                },
            ],
            "tpl_005": [
                {
                    "url": "https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html",
                    "status": "extracted",
                    "data": [
                        {
                            "page_title": "Linear Regression Example",
                            "description": "A simple OLS example.",
                            "content": "This example shows how to use LinearRegression.",
                            "code_blocks": ["import sklearn"],
                            "input": {"url": "https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html"},
                        }
                    ],
                },
                {
                    "url": "https://scikit-learn.org/stable/auto_examples/linear_model/plot_ridge.html",
                    "status": "extracted",
                    "data": [
                        {
                            "page_title": "Ridge Regression Example",
                            "description": "Ridge regression with cross-validation.",
                            "content": "This example shows Ridge regression.",
                            "code_blocks": ["from sklearn.linear_model import Ridge"],
                            "input": {"url": "https://scikit-learn.org/stable/auto_examples/linear_model/plot_ridge.html"},
                        }
                    ],
                },
                {
                    "url": "https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso.html",
                    "status": "extracted",
                    "data": [
                        {
                            "page_title": "Lasso Example",
                            "description": "Lasso with coordinate descent.",
                            "content": "This example shows Lasso.",
                            "code_blocks": ["from sklearn.linear_model import Lasso"],
                            "input": {"url": "https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso.html"},
                        }
                    ],
                },
            ],
        },
    }

    output_file = tmp_path / "phase3_output.json"
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_file


# ---------------------------------------------------------------------------
# Temporary SQLite store
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_store(tmp_path) -> SQLiteStore:
    """A SQLiteStore initialized in a temporary directory."""
    db_path = str(tmp_path / "test_rag.db")
    return SQLiteStore(db_path=db_path)
