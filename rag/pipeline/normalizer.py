"""
Universal normalizer.

Converts any Phase 3 template output into NormalizedDocument objects,
regardless of the field schema. Uses FieldClassifier to discover roles
dynamically and ContentCleaner is applied downstream.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag.models.document import ContentBlock, NormalizedDocument
from rag.pipeline.field_classifier import FieldClassifier
from rag.utils.ids import generate_doc_id
from rag.utils.text import extract_code_language

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Content-type detection mapping
# ---------------------------------------------------------------------------
# Maps detected role combinations → NormalizedDocument.content_type
# Must only use values valid in Phase 4.1's NormalizedDocument model:
#   api_reference, tutorial, notebook, example, unknown
_CONTENT_TYPE_RULES: list[tuple[set[str], str]] = [
    ({"signature", "parameter"}, "api_reference"),
    ({"notebook"},               "notebook"),
    ({"section"},                "tutorial"),
    ({"install"},                "tutorial"),
]


class UniversalNormalizer:
    """
    Convert any Phase 3 scraped JSON entry into NormalizedDocument objects.

    The normalizer is intentionally schema-agnostic: it uses FieldClassifier
    to discover what each field means and builds ContentBlocks accordingly.
    """

    def __init__(self, settings=None) -> None:
        self._classifier = FieldClassifier()
        # stats for the last normalize_file() call
        self._stats: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize_entry(self, entry: dict) -> List[NormalizedDocument]:
        """
        Normalize one URL entry (has "url", "status", "data" keys).

        Returns a list — one NormalizedDocument per item in entry["data"].
        Empty list if the entry has no data items.
        """
        url = entry.get("url", "")
        data_items = entry.get("data") or []
        docs: List[NormalizedDocument] = []

        for idx, item in enumerate(data_items):
            doc = self._normalize_item(item, url=url, index=idx, entry=entry)
            if doc is not None:
                docs.append(doc)

        return docs

    def normalize_file(self, json_path: str) -> List[NormalizedDocument]:
        """
        Load a Phase 3 scraped JSON file and normalize every entry.

        Returns a flat list of NormalizedDocuments from all templates.
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Scraped JSON not found: {path}")

        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)

        results = raw.get("results", {}) or {}
        all_docs: List[NormalizedDocument] = []
        total_entries = 0
        total_errors = 0
        total_empties = 0
        by_template: dict = {}

        for tpl_id, entries in results.items():
            entries = entries or []
            tpl_docs: List[NormalizedDocument] = []
            tpl_errors = 0

            for entry in entries:
                total_entries += 1
                status = entry.get("status", "")
                data_items = entry.get("data") or []

                if status == "failed" or (not data_items):
                    # Create a single error/empty doc
                    err_doc = self._make_error_doc(entry)
                    if err_doc is not None:
                        tpl_docs.append(err_doc)
                        tpl_errors += 1
                    else:
                        total_empties += 1
                    continue

                entry_docs = self.normalize_entry(entry)
                # Stamp each doc with template_id
                stamped = []
                for doc in entry_docs:
                    stamped.append(NormalizedDocument(
                        doc_id=doc.doc_id,
                        url=doc.url,
                        title=doc.title,
                        description=doc.description,
                        content_blocks=doc.content_blocks,
                        metadata=doc.metadata,
                        template_id=tpl_id,
                        content_type=doc.content_type,
                        source_link=doc.source_link,
                        error=doc.error,
                    ))
                tpl_docs.extend(stamped)

            all_docs.extend(tpl_docs)
            by_template[tpl_id] = {
                "input": len(entries),
                "output": len(tpl_docs),
                "errors": tpl_errors,
            }

        total_errors = sum(v["errors"] for v in by_template.values())
        self._stats = {
            "total_entries": total_entries,
            "total_docs": len(all_docs),
            "total_errors": total_errors,
            "total_empties": total_empties,
            "by_template": by_template,
        }
        logger.info(
            "normalize_file: %d entries → %d docs (%d errors, %d empties)",
            total_entries, len(all_docs), total_errors, total_empties,
        )
        return all_docs

    def get_stats(self) -> dict:
        return dict(self._stats)

    # ------------------------------------------------------------------
    # Private: item-level normalization
    # ------------------------------------------------------------------

    def _normalize_item(
        self,
        item: dict,
        *,
        url: str,
        index: int,
        entry: dict,
    ) -> Optional[NormalizedDocument]:
        """Normalize a single data item dict."""
        # Error items
        if "error" in item:
            return NormalizedDocument(
                doc_id=generate_doc_id(url, index),
                url=url,
                title="",
                description="",
                content_blocks=[],
                metadata={},
                template_id="",
                content_type="unknown",
                error=str(item["error"]),
            )

        roles = self._classifier.classify_fields(item)
        priority_roles = self._classifier.get_priority_ordered_roles()

        # --- Scalar extractions ---
        title = self._extract_scalar(item, roles, "title", "")
        description = self._extract_scalar(item, roles, "description", "")
        source_link = self._extract_scalar(item, roles, "source", None)

        # --- Build content blocks in priority order ---
        blocks: List[ContentBlock] = []
        used_fields: set[str] = {"input"}  # always exclude

        for role in priority_roles:
            if role in ("title", "source"):
                # These become scalar fields, not blocks (except when multi-valued)
                for fname in roles.get(role, []):
                    used_fields.add(fname)
                continue

            for fname in roles.get(role, []):
                used_fields.add(fname)
                new_blocks = self._build_blocks_for_role(
                    role=role,
                    fname=fname,
                    value=item[fname],
                    heading=self._field_to_heading(fname),
                )
                blocks.extend(new_blocks)

        # --- "other" role fields ---
        # All 'other' fields go to metadata.
        # Long text-like values ALSO produce a prose block.
        for fname in roles.get("other", []):
            if fname in used_fields:
                continue
            value = item[fname]
            # Only include as a block if it's a meaningfully long string
            if isinstance(value, str) and len(value.split()) >= 5:
                blocks.append(ContentBlock(
                    block_type="prose",
                    text=value,
                    heading=self._field_to_heading(fname),
                ))
            # Always add to used_fields so it goes to metadata below
            used_fields.add(fname)

        # --- Metadata: everything not consumed as a block ---
        # For 'other' fields, include them in metadata even if they made a block.
        metadata: dict = {}
        for fname, fvalue in item.items():
            if fname == "input":
                continue
            # Fields classified as 'other' always go to metadata
            if fname in (roles.get("other") or []):
                metadata[fname] = fvalue
            elif fname not in used_fields:
                metadata[fname] = fvalue

        # --- content_type detection ---
        content_type = self._detect_content_type(roles)

        return NormalizedDocument(
            doc_id=generate_doc_id(url, index),
            url=url,
            title=title,
            description=description,
            content_blocks=blocks,
            metadata=metadata,
            template_id="",   # stamped by normalize_file
            content_type=content_type,
            source_link=source_link,
            error=None,
        )

    # ------------------------------------------------------------------
    # Private: role → ContentBlock builders
    # ------------------------------------------------------------------

    def _build_blocks_for_role(
        self,
        *,
        role: str,
        fname: str,
        value: Any,
        heading: str,
    ) -> List[ContentBlock]:
        """Dispatch to the correct builder for *role*."""
        if role == "description":
            return self._build_prose_blocks(value, heading)
        if role == "signature":
            return self._build_signature_blocks(value, heading)
        if role == "introduction":
            return self._build_prose_blocks(value, heading)
        if role == "note":
            return self._build_note_blocks(value, heading)
        if role == "parameter":
            return self._build_parameter_blocks(value)
        if role == "section":
            return self._build_section_blocks(value)
        if role == "code":
            return self._build_code_blocks(value, heading)
        if role == "relation":
            return self._build_relation_blocks(value, heading)
        if role == "install":
            return self._build_install_blocks(value, heading)
        if role == "notebook":
            return self._build_notebook_blocks(value)
        if role == "error":
            return self._build_prose_blocks(str(value), heading) if value else []
        # fallback
        return self._build_prose_blocks(value, heading)

    # --- Prose / Description / Introduction / Note ---

    def _build_prose_blocks(self, value: Any, heading: str) -> List[ContentBlock]:
        text = self._to_text(value)
        if not text:
            return []
        return [ContentBlock(block_type="prose", text=text, heading=heading)]

    def _build_note_blocks(self, value: Any, heading: str) -> List[ContentBlock]:
        text = self._to_text(value)
        if not text:
            return []
        return [ContentBlock(block_type="note", text=text, heading=heading)]

    # --- Signature ---

    def _build_signature_blocks(self, value: Any, heading: str) -> List[ContentBlock]:
        text = self._to_text(value)
        if not text:
            return []
        return [ContentBlock(block_type="function_signature", text=text, heading=heading)]

    # --- Parameters ---

    def _build_parameter_blocks(self, params: Any) -> List[ContentBlock]:
        if not isinstance(params, list):
            # Could be a string description of params
            text = self._to_text(params)
            if text:
                return [ContentBlock(block_type="parameter_list", text=text, heading="Parameters")]
            return []

        blocks: List[ContentBlock] = []
        for param in params:
            if not isinstance(param, dict):
                continue
            name = (
                param.get("name")
                or param.get("param")
                or param.get("arg")
                or ""
            )
            type_info = (
                param.get("type_info")
                or param.get("type")
                or param.get("dtype")
                or ""
            )
            desc = (
                param.get("description")
                or param.get("desc")
                or param.get("summary")
                or ""
            )
            parts = []
            if name:
                parts.append(f"Parameter: {name}")
            if type_info:
                parts.append(f"Type: {type_info}")
            if desc:
                parts.append(desc)
            text = "\n".join(parts)
            if not text.strip():
                continue
            blocks.append(ContentBlock(
                block_type="parameter_list",
                text=text,
                heading=f"Parameter: {name}" if name else "Parameter",
                structured_data=param,
            ))
        return blocks

    # --- Sections ---

    def _build_section_blocks(self, sections: Any) -> List[ContentBlock]:
        if not isinstance(sections, list):
            text = self._to_text(sections)
            if text:
                return [ContentBlock(block_type="prose", text=text, heading="Section")]
            return []

        blocks: List[ContentBlock] = []
        for sec in sections:
            if isinstance(sec, dict):
                sec_title = (
                    sec.get("section_title")
                    or sec.get("title")
                    or sec.get("heading")
                    or ""
                )
                sec_content = (
                    sec.get("section_content")
                    or sec.get("content")
                    or sec.get("body")
                    or ""
                )
                # Clean trailing "#" from heading
                sec_title = sec_title.rstrip("#").strip()
                if sec_content:
                    blocks.append(ContentBlock(
                        block_type="prose",
                        text=self._to_text(sec_content),
                        heading=sec_title,
                    ))
                # Also check for nested code blocks within a section
                code_val = sec.get("code") or sec.get("code_block")
                if code_val:
                    blocks.extend(self._build_code_blocks(code_val, sec_title or "Code"))
            elif isinstance(sec, str) and sec.strip():
                blocks.append(ContentBlock(block_type="prose", text=sec, heading="Section"))
        return blocks

    # --- Code ---

    def _build_code_blocks(self, value: Any, heading: str) -> List[ContentBlock]:
        if not value:
            return []
        if isinstance(value, str):
            lang = extract_code_language(value)
            return [ContentBlock(block_type="code", text=value, heading=heading, language=lang)]

        if isinstance(value, list):
            blocks: List[ContentBlock] = []
            for item in value:
                if isinstance(item, str) and item.strip():
                    lang = extract_code_language(item)
                    blocks.append(ContentBlock(
                        block_type="code", text=item, heading=heading, language=lang,
                    ))
                elif isinstance(item, dict):
                    code_text = (
                        item.get("content")
                        or item.get("code")
                        or item.get("text")
                        or ""
                    )
                    lang = item.get("language") or extract_code_language(code_text)
                    if code_text:
                        blocks.append(ContentBlock(
                            block_type="code", text=code_text, heading=heading, language=lang,
                        ))
            return blocks

        text = self._to_text(value)
        if text:
            return [ContentBlock(block_type="code", text=text, heading=heading,
                                 language=extract_code_language(text))]
        return []

    # --- Relations (see_also, related, etc.) ---

    def _build_relation_blocks(self, relations: Any, default_heading: str) -> List[ContentBlock]:
        heading = default_heading or "See Also"
        if not relations:
            return []

        if isinstance(relations, list):
            lines: List[str] = []
            for rel in relations:
                if isinstance(rel, dict):
                    name = (
                        rel.get("function_name")
                        or rel.get("name")
                        or rel.get("title")
                        or ""
                    )
                    desc = (
                        rel.get("function_description")
                        or rel.get("description")
                        or rel.get("link")
                        or ""
                    )
                    if name and desc:
                        lines.append(f"- {name}: {desc}")
                    elif name:
                        lines.append(f"- {name}")
                    elif desc:
                        lines.append(f"- {desc}")
                elif isinstance(rel, str) and rel.strip():
                    lines.append(f"- {rel}")
            text = "\n".join(lines)
            if text:
                return [ContentBlock(block_type="prose", text=text, heading=heading)]
            return []

        text = self._to_text(relations)
        if text:
            return [ContentBlock(block_type="prose", text=text, heading=heading)]
        return []

    # --- Install ---

    def _build_install_blocks(self, value: Any, heading: str) -> List[ContentBlock]:
        text = self._to_text(value)
        if not text:
            return []
        return [ContentBlock(block_type="code", text=text, heading=heading, language="bash")]

    # --- Notebooks ---

    def _build_notebook_blocks(self, value: Any) -> List[ContentBlock]:
        if not value:
            return []
        raw_str = self._to_text(value)
        return self._parse_notebook_content(raw_str)

    def _parse_notebook_content(self, raw: str) -> List[ContentBlock]:
        """
        Try to parse a raw Jupyter notebook JSON string.
        Falls back to a single prose block on failure.
        """
        if not raw:
            return []
        try:
            nb = json.loads(raw)
            cells = nb.get("cells") or nb.get("worksheets", [{}])[0].get("cells", [])
            blocks: List[ContentBlock] = []
            for cell in cells:
                ctype = cell.get("cell_type", "")
                source = cell.get("source", "")
                if isinstance(source, list):
                    source = "".join(source)
                source = source.strip()
                if not source:
                    continue
                if ctype == "markdown":
                    blocks.append(ContentBlock(block_type="prose", text=source, heading=""))
                elif ctype == "code":
                    lang = extract_code_language(source) or "python"
                    blocks.append(ContentBlock(block_type="code", text=source, language=lang))
                else:
                    blocks.append(ContentBlock(block_type="prose", text=source, heading=""))
            return blocks
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            return [ContentBlock(block_type="prose", text=raw, heading="Notebook Content")]

    # ------------------------------------------------------------------
    # Private: helpers
    # ------------------------------------------------------------------

    def _extract_scalar(
        self,
        item: dict,
        roles: Dict[str, List[str]],
        role: str,
        default: Any,
    ) -> Any:
        """Extract the first value for a role as a scalar string."""
        fields = roles.get(role, [])
        for fname in fields:
            val = item.get(fname)
            if val and isinstance(val, str):
                return val.strip()
            if val and not isinstance(val, (list, dict)):
                return str(val).strip()
        return default

    def _detect_content_type(self, roles: Dict[str, List[str]]) -> str:
        for required_roles, ctype in _CONTENT_TYPE_RULES:
            if required_roles.issubset(roles.keys()):
                return ctype
        return "unknown"

    def _make_error_doc(self, entry: dict) -> Optional[NormalizedDocument]:
        url = entry.get("url", "")
        if not url:
            return None
        error_msg = entry.get("error") or entry.get("status") or "unknown error"
        return NormalizedDocument(
            doc_id=generate_doc_id(url, 0),
            url=url,
            title="",
            description="",
            content_blocks=[],
            metadata={},
            template_id="",
            content_type="unknown",
            error=str(error_msg),
        )

    @staticmethod
    def _to_text(value: Any) -> str:
        """Convert any scalar/list/dict to a text string."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    # Try common text keys
                    for key in ("text", "content", "body", "description"):
                        if key in item:
                            parts.append(str(item[key]))
                            break
                    else:
                        parts.append(str(item))
                elif item is not None:
                    parts.append(str(item))
            return "\n".join(p for p in parts if p)
        if isinstance(value, dict):
            return str(value)
        return ""

    @staticmethod
    def _field_to_heading(field_name: str) -> str:
        """Convert snake_case/lower_case field name to Title Case heading."""
        return field_name.replace("_", " ").replace("-", " ").title()
