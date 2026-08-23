"""
Field role classifier.

Heuristically classifies any field name from a scraped JSON item into a
semantic role. This is the core intelligence that lets the normalizer
work schema-agnostically across any website or template.
"""

from __future__ import annotations

from typing import Dict, List

# ---------------------------------------------------------------------------
# Role pattern registries
# ---------------------------------------------------------------------------

TITLE_PATTERNS: list[str] = [
    "page_title", "notebook_title", "title", "name", "heading",
]
DESCRIPTION_PATTERNS: list[str] = [
    "description", "summary", "abstract",
]
SIGNATURE_PATTERNS: list[str] = [
    "function_signature", "method_signature", "signature",
]
PARAMETER_PATTERNS: list[str] = [
    "parameters", "parameter", "params", "arguments", "argument", "args", "properties", "param",
]
CODE_PATTERNS: list[str] = [
    "code_examples", "code_snippets", "code_blocks", "code_example",
    "code_snippet", "code", "example", "snippet", "sample",
]
NOTE_PATTERNS: list[str] = [
    "notes", "note", "warning", "caution", "tip",
]
SOURCE_PATTERNS: list[str] = [
    "source_link", "source_url", "github", "repository", "source",
]
SECTION_PATTERNS: list[str] = [
    "sections", "section", "chapter",
]
RELATION_PATTERNS: list[str] = [
    "see_also", "related_functions", "related_examples", "related",
    "cross_references", "references",
]
INTRODUCTION_PATTERNS: list[str] = [
    "introduction", "intro", "overview",
]
INSTALL_PATTERNS: list[str] = [
    "installation_pip", "installation_conda", "installation", "install",
]
ERROR_PATTERNS: list[str] = [
    "error_code", "error_message", "error",
]
NOTEBOOK_PATTERNS: list[str] = [
    "notebook_content", "notebook_title", "notebook", "ipynb",
]

# Roles in detection priority order (first match wins).
# NOTEBOOK must come before NOTE to prevent 'notebook_content' from
# matching the 'note' substring before it can match 'notebook'.
_ROLE_REGISTRY: list[tuple[str, list[str]]] = [
    ("title",        TITLE_PATTERNS),
    ("description",  DESCRIPTION_PATTERNS),
    ("signature",    SIGNATURE_PATTERNS),
    ("parameter",    PARAMETER_PATTERNS),
    ("notebook",     NOTEBOOK_PATTERNS),   # before NOTE
    ("code",         CODE_PATTERNS),
    ("note",         NOTE_PATTERNS),
    ("source",       SOURCE_PATTERNS),
    ("section",      SECTION_PATTERNS),
    ("relation",     RELATION_PATTERNS),
    ("introduction", INTRODUCTION_PATTERNS),
    ("install",      INSTALL_PATTERNS),
    ("error",        ERROR_PATTERNS),
]

# Content block priority order (determines ordering in NormalizedDocument)
_PRIORITY_ORDER: list[str] = [
    "title",
    "introduction",
    "description",
    "signature",
    "parameter",
    "note",
    "section",
    "code",
    "relation",
    "install",
    "notebook",
    "error",
    "other",
]

# Fields that are always excluded from classification
_EXCLUDED_FIELDS: frozenset[str] = frozenset({"input"})


class FieldClassifier:
    """
    Classify field names from a scraped JSON data item into semantic roles.

    Usage::

        classifier = FieldClassifier()
        roles = classifier.classify_fields(item)
        # roles == {"title": ["page_title"], "parameter": ["parameters"], ...}
    """

    def classify_fields(self, item: dict) -> Dict[str, List[str]]:
        """
        Map each field in *item* to a semantic role.

        Args:
            item: One data item dict from a scraped JSON "data" array.

        Returns:
            Dict mapping role name → list of field names with that role.
            Every field in *item* appears in exactly one role bucket.
            Fields in ``_EXCLUDED_FIELDS`` are silently dropped.
        """
        result: Dict[str, List[str]] = {role: [] for role, _ in _ROLE_REGISTRY}
        result["other"] = []

        for field_name in item:
            if field_name in _EXCLUDED_FIELDS:
                continue
            role = self._classify_one(field_name)
            result[role].append(field_name)

        # Remove empty buckets so callers can use `if roles.get("title"):`
        return {role: fields for role, fields in result.items() if fields}

    def _classify_one(self, field_name: str) -> str:
        """Return the role for a single field name."""
        lower = field_name.lower()
        for role, patterns in _ROLE_REGISTRY:
            for pattern in patterns:
                if pattern in lower:
                    return role
        return "other"

    def get_priority_ordered_roles(self) -> List[str]:
        """
        Return the ordered list of roles as they should appear in output
        content blocks. Callers iterate this list to build blocks in order.
        """
        return list(_PRIORITY_ORDER)
