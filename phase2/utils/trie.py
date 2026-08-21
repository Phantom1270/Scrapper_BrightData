"""
Trie (prefix tree) data structure for URL path analysis.
Used to discover common path prefixes and group URLs by structure.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from models import ParsedURL
from config import MIN_GROUP_SIZE
from utils.url_parser import split_path_segments
import re


def _is_literal_like(segment: str) -> bool:
    """
    Heuristic: is a segment 'literal-like'?
    - All lowercase
    - Only letters, numbers, underscores, hyphens
    - No file extension (.html, .rst, etc.)
    - Not camelCase
    - Not a dotted path
    """
    if not segment:
        return False
    # Check for file extension
    known_ext = {'html', 'htm', 'rst', 'md', 'php', 'aspx', 'txt',
                 'json', 'xml', 'csv', 'css', 'js', 'mjs', 'png',
                 'jpg', 'jpeg', 'gif', 'svg', 'ico', 'webp'}
    if '.' in segment:
        last_part = segment.rsplit('.', 1)[-1].lower()
        if last_part in known_ext:
            return False
        # dotted path
        return False
    # Not all lowercase? Then camelCase or something else
    if segment != segment.lower():
        return False
    # Must only have letters, digits, underscores, hyphens
    if not re.match(r'^[a-z][a-z0-9_-]*$', segment):
        return False
    return True


@dataclass
class TrieNode:
    """
    A single node in the URL trie.

    The trie stores path segments as keys. Leaf nodes hold the actual URLs.
    Intermediate nodes hold aggregate statistics about their children.
    """
    segment: str = ""                              # The path segment: "modules", "generated", etc.
    depth: int = 0                                 # 0 = root
    path_from_root: str = ""                       # e.g. "auto_examples/cluster"
    children: dict[str, "TrieNode"] = field(default_factory=dict)
    urls: list[str] = field(default_factory=list)  # Only populated at leaf nodes
    is_leaf: bool = False

    # Computed metrics — filled in by compute_metrics()
    child_count: int = 0
    descendant_count: int = 0                      # Total leaf URLs under this node
    unique_segment_count: int = 0                  # How many unique child segment values
    child_literal_ratio: float = 0.0               # Fraction of children with literal-like names

    def add_url(self, segments: list[str], url: str, path_so_far: str = "") -> None:
        """
        Insert a URL into the trie by following its path segments.

        Example:
            trie.add_url(["modules", "generated", "sklearn.linear_model.LogisticRegression.html"],
                         "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html")

        Creates nodes: root → modules → generated → sklearn.linear_model.LogisticRegression.html (leaf)
        """
        if not segments:
            # This node is the leaf
            self.is_leaf = True
            self.urls.append(url)
            return

        first, rest = segments[0], segments[1:]
        child_path = f"{path_so_far}/{first}" if path_so_far else first

        if first not in self.children:
            self.children[first] = TrieNode(
                segment=first,
                depth=self.depth + 1,
                path_from_root=child_path
            )

        self.children[first].add_url(rest, url, child_path)

    def compute_metrics(self) -> None:
        """
        Walk the entire tree and compute metrics for every node.
        Must be called AFTER all URLs have been added.
        """
        # Recurse into children first (post-order)
        for child in self.children.values():
            child.compute_metrics()

        self.child_count = len(self.children)
        self.unique_segment_count = len(self.children)

        # descendant_count: count all leaf urls in this subtree
        self.descendant_count = len(self.urls)  # urls at this node (if leaf)
        for child in self.children.values():
            self.descendant_count += child.descendant_count

        # child_literal_ratio
        if self.child_count > 0:
            literal_count = sum(
                1 for seg in self.children if _is_literal_like(seg)
            )
            self.child_literal_ratio = literal_count / self.child_count
        else:
            self.child_literal_ratio = 0.0

    def find_pattern_anchors(self, literal_ratio_threshold: float = 0.3) -> list["TrieNode"]:
        """
        Find nodes where the child literal ratio drops below the threshold.
        These are "pattern anchors" — points where URL content becomes variable.

        Returns the DEEPEST qualifying nodes (don't return a parent if a child also qualifies).
        """
        results = []

        def _walk(node: TrieNode):
            child_qualifies = False
            for child in node.children.values():
                _walk(child)
                # After walking child, check if it was added to results
                if child in results:
                    child_qualifies = True
            
            # This node qualifies if its ratio is below threshold
            # AND it has enough descendants
            if (node.child_literal_ratio < literal_ratio_threshold 
                and node.descendant_count >= MIN_GROUP_SIZE
                and node.child_count > 0
                and not child_qualifies):  # Only if no child already qualified
                results.append(node)

        _walk(self)
        return results

    def find_structural_junctions(self, literal_ratio_threshold: float = 0.7) -> list["TrieNode"]:
        """
        Find nodes where most children are literal values.
        These are "structural junctions" — points where the site branches
        into distinct sections (user_guide/, modules/, auto_examples/).

        Returns nodes where child_literal_ratio >= threshold AND child_count >= 2.
        """
        results = []

        def _walk(node: TrieNode) -> None:
            if (node.child_literal_ratio >= literal_ratio_threshold
                    and node.child_count >= 2):
                results.append(node)
            for child in node.children.values():
                _walk(child)

        _walk(self)
        return results

    def collect_leaves(self) -> list[str]:
        """Collect all URLs stored in leaves of this subtree."""
        collected = list(self.urls)
        for child in self.children.values():
            collected.extend(child.collect_leaves())
        return collected

    def to_string(self, indent: int = 0) -> str:
        """
        Pretty-print the trie for debugging.
        """
        lines = []
        prefix = "  " * indent
        label = self.segment or "root"
        stats = f"[descendants={self.descendant_count}, child_ratio={self.child_literal_ratio:.1f}]"
        lines.append(f"{prefix}{label} {stats}")

        child_items = list(self.children.items())
        for i, (seg, child) in enumerate(child_items):
            if i < 5:  # Limit display for large tries
                lines.append(child.to_string(indent + 1))
            elif i == 5:
                remaining = len(child_items) - 5
                lines.append(f"{'  ' * (indent + 1)}... ({remaining} more children)")
                break

        # Show leaf URLs (limit)
        if self.is_leaf and self.urls:
            for url in self.urls[:3]:
                lines.append(f"{'  ' * (indent + 1)}→ {url}")
            if len(self.urls) > 3:
                lines.append(f"{'  ' * (indent + 1)}→ ... ({len(self.urls) - 3} more)")

        return "\n".join(lines)


def build_trie(urls: list[ParsedURL], use_version_free_path: bool = True) -> TrieNode:
    root = TrieNode()
    for url in urls:
        path = url.version_free_path if use_version_free_path and url.version_free_path else url.path
        segments = split_path_segments(path)
        root.add_url(segments, url.canonical_url, "")
    root.compute_metrics()
    return root
