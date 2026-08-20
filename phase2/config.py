"""
Central configuration for Phase 2.
Every constant used across the project lives here.
No magic numbers anywhere else.
"""

# ─── Asset filtering ─────────────────────────────────────────
# URLs ending with these extensions are assets, not pages.
# Phase 2 does NOT build templates for assets.
ASSET_EXTENSIONS: set[str] = {
    ".css", ".js", ".mjs",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".woff", ".woff2", ".ttf", ".eot",
    ".mp4", ".mp3", ".wav",
    ".zip", ".tar", ".gz",
    ".pdf",
    ".xml",  # except sitemap.xml — handled separately
    ".map",  # source maps
}

# ─── Segment classification ──────────────────────────────────
# Thresholds for determining if a path segment is static or variable.
# If a segment value appears in >80% of URLs at the same position, treat as static.
STATIC_THRESHOLD: float = 0.80

# If a segment value appears in <10% of URLs, it is likely unique per page (variable).
VARIABLE_THRESHOLD: float = 0.10

# ─── Pattern grouping ────────────────────────────────────────
# Minimum number of URLs to form a group.
# Groups smaller than this get merged into a catch-all or sent to uncovered.
MIN_GROUP_SIZE: int = 3

# Two templates are considered mergeable if their fingerprints differ
# by at most this many positions.
TEMPLATE_MERGE_DISTANCE: int = 1

# ─── Coverage ────────────────────────────────────────────────
# Target coverage percentage. If below this, the pipeline logs a warning.
TARGET_COVERAGE: float = 0.85

# ─── Version detection ───────────────────────────────────────
# Regex pattern for version prefixes in URL paths.
# Matches: /stable/, /dev/, /latest/, /0.24/, /3.1.2/, /v2/, /en/stable/
VERSION_PATTERN: str = (
    r"^/"
    r"(?:en/|fr/|de/|es/|ja/|zh/|ko/)?"  # optional language prefix
    r"(?:stable|dev|latest|master|main"    # named versions
    r"|v?\d+(?:\.\d+)*(?:\.\d+)?)"        # numeric versions like 0.24, v2, 3.1.2
    r"(/|$)"
)

# ─── Generator detection ─────────────────────────────────────
# Known path indicators for documentation generators
GENERATOR_SIGNALS: dict[str, dict[str, str]] = {
    "sphinx": {
        "static_dir": "_static",
        "sources_dir": "_sources",
        "objects_inv": "objects.inv",
        "search_js": "_static/searchtools.js",
    },
    "mkdocs": {
        "search_json": "search/search_index.json",
        "mkdocs_theme": "mkdocs",
    },
    "docusaurus": {
        "assets_dir": "assets/js",
        "docusaurus_js": "docusaurus",
    },
    "vuepress": {
        "assets_dir": ".vuepress",
    },
    "hugo": {
        "index_json": "index.json",
    },
}

# ─── Output ──────────────────────────────────────────────────
# Maximum number of example URLs to include per template in the output
MAX_EXAMPLES_PER_TEMPLATE: int = 5

# Maximum number of uncovered URLs to include in output (before truncating)
MAX_UNCOVERED_IN_OUTPUT: int = 50
