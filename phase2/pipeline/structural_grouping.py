"""
Stage 2.5 + 2.6 — Structural Grouping and Template Derivation
Takes parsed URLs, builds trie, discovers groups, derives templates.
This is the core intelligence of Phase 2.
"""

from models import (
    ParsedURL, TemplatePattern, SegmentType, ParsedSegment
)
from utils.trie import TrieNode, build_trie
from utils.segment_classifier import (
    compute_position_frequencies, classify_positions, generate_fingerprint
)
from config import MIN_GROUP_SIZE, TEMPLATE_MERGE_DISTANCE, MAX_EXAMPLES_PER_TEMPLATE
import re


class URLGroup:
    """
    A group of URLs that likely share the same scraper template.
    Intermediate representation before becoming a TemplatePattern.
    """
    def __init__(self, group_id: str, urls: list[str], path_prefix: str):
        self.group_id = group_id
        self.urls = urls              # List of canonical URLs
        self.path_prefix = path_prefix  # Common prefix, e.g. "/modules/generated"
        self.template_pattern = None   # Filled in by derive_template()
        self.fingerprint = None        # Filled in by derive_template()


def discover_groups(
    urls: list[ParsedURL],
    generator_hint: str | None = None,
) -> list[URLGroup]:
    """
    Main grouping function. Discovers URL groups from a list of parsed URLs.

    Steps:
    1. Build a trie from the URLs (using version_free_path)
    2. Find pattern anchors (nodes where content becomes variable)
    3. Find structural junctions (nodes where site branches)
    4. For each pattern anchor:
       a. Collect all leaf URLs under it
       b. If count >= MIN_GROUP_SIZE → create a URLGroup
       c. If count < MIN_GROUP_SIZE → mark as candidate for catch-all group
    5. For structural junctions with only 1-2 leaf children (small sections):
       a. Group them into a "static_pages" catch-all
    6. Return all groups
    """
    if not urls:
        return []

    trie = build_trie(urls, use_version_free_path=True)

    groups: list[URLGroup] = []
    small_url_pool: list[str] = []
    covered_urls: set[str] = set()

    # ── Step 2: Find pattern anchors ──
    pattern_anchors = trie.find_pattern_anchors(literal_ratio_threshold=0.3)

    group_counter = [1]

    def make_group_id() -> str:
        gid = f"grp_{group_counter[0]:03d}"
        group_counter[0] += 1
        return gid

    for anchor in pattern_anchors:
        leaf_urls = anchor.collect_leaves()
        if len(leaf_urls) >= MIN_GROUP_SIZE:
            # Reconstruct the path prefix from the anchor's ancestors
            # We use the anchor's segment as the leaf prefix
            # Build prefix by walking the segment chain
            prefix = _build_prefix(anchor)
            grp = URLGroup(make_group_id(), leaf_urls, prefix)
            groups.append(grp)
            covered_urls.update(leaf_urls)
        else:
            small_url_pool.extend(leaf_urls)

    # ── Step 3: Find structural junctions ──
    junctions = trie.find_structural_junctions(literal_ratio_threshold=0.7)

    for junction in junctions:
        for seg, child_node in junction.children.items():
            leaf_urls = child_node.collect_leaves()
            # Skip already covered
            new_urls = [u for u in leaf_urls if u not in covered_urls]
            if not new_urls:
                continue
            if len(new_urls) >= MIN_GROUP_SIZE:
                prefix = _build_prefix(child_node)
                grp = URLGroup(make_group_id(), new_urls, prefix)
                groups.append(grp)
                covered_urls.update(new_urls)
            else:
                small_url_pool.extend(new_urls)

    # ── Step 4: Check root-level pages not yet covered ──
    for purl in urls:
        if purl.canonical_url not in covered_urls:
            small_url_pool.append(purl.canonical_url)

    # Deduplicate small pool
    small_url_pool = list(dict.fromkeys(u for u in small_url_pool if u not in covered_urls))

    # ── Step 5: Catch-all group for small/leftover URLs ──
    if small_url_pool:
        grp = URLGroup(make_group_id(), small_url_pool, "/")
        grp.group_id = grp.group_id  # keep as-is; name it catch_all later
        groups.append(grp)

    return groups


def _build_prefix(node: TrieNode) -> str:
    """
    Reconstruct the path prefix up to and including this node's segment.
    Since TrieNode doesn't store parent references, we store the segment
    path during traversal instead. We approximate by using segment value.
    """
    # The node's segment is the last part of the prefix.
    # We can't easily walk up, so we return the segment itself as a hint.
    return f"/{node.segment}" if node.segment else "/"


def derive_template(group: URLGroup, all_urls: list[ParsedURL]) -> TemplatePattern:
    """
    Given a URLGroup, derive a TemplatePattern.

    Steps:
    1. Get all ParsedURL objects that belong to this group
    2. All URLs in a group should have the same number of path segments.
       If not, split into depth-based subgroups.
    3. For each segment position:
       a. Collect all values at that position across the group
       b. If all identical → static segment, use the literal value
       c. If highly varied → variable segment, use <TYPE> placeholder
    4. Build the pattern string
    5. Build the fingerprint
    6. Compute confidence
    7. Return TemplatePattern
    """
    # Find ParsedURL objects for the group's URLs
    url_set = set(group.urls)
    group_parsed = [u for u in all_urls if u.canonical_url in url_set]

    # Fall back: if no parsed matches found, use raw URL strings for building
    if not group_parsed:
        # Build minimal pattern from URLs directly
        return TemplatePattern(
            template_id="",
            pattern="/<unknown>",
            fingerprint="UNKNOWN",
            member_count=len(group.urls),
            example_urls=group.urls[:MAX_EXAMPLES_PER_TEMPLATE],
            confidence=0.1,
        )

    # Use version_free_path for grouping logic
    def get_path(u: ParsedURL) -> str:
        return u.version_free_path if u.version_free_path else u.path

    # Group by segment depth
    depth_groups: dict[int, list[ParsedURL]] = {}
    for u in group_parsed:
        segs = [s for s in get_path(u).split("/") if s]
        d = len(segs)
        depth_groups.setdefault(d, []).append(u)

    # Pick the largest depth group
    best_depth = max(depth_groups, key=lambda d: len(depth_groups[d]))
    main_group = depth_groups[best_depth]

    # Compute position frequencies for the main group
    pos_freqs = compute_position_frequencies(main_group)
    pos_types = classify_positions(
        pos_freqs,
        static_threshold=0.80,
        variable_threshold=0.10,
    )

    # Build pattern and fingerprint
    pattern_parts = []
    fingerprint_parts = []

    # Get segment count from a representative URL
    rep_segs = [s for s in get_path(main_group[0]).split("/") if s]

    # Collect dominant types per position
    for pos, seg_val in enumerate(rep_segs):
        pos_type = pos_types.get(pos, "variable")

        if pos_type == "static":
            # Use the most frequent value
            freq_at_pos = pos_freqs.get(pos, {})
            dominant_val = max(freq_at_pos, key=lambda v: freq_at_pos[v], default=seg_val)
            pattern_parts.append(dominant_val)
            fingerprint_parts.append(dominant_val.upper())
        else:
            # Determine the most common segment type at this position
            types_at_pos: dict[str, int] = {}
            for u in main_group:
                segs = [s for s in get_path(u).split("/") if s]
                if pos < len(segs):
                    from utils.url_parser import classify_segment
                    seg_type = classify_segment(segs[pos])
                    types_at_pos[seg_type.value] = types_at_pos.get(seg_type.value, 0) + 1

            if types_at_pos:
                dominant_type = max(types_at_pos, key=lambda t: types_at_pos[t])
            else:
                dominant_type = SegmentType.UNKNOWN.value

            pattern_parts.append(f"<{dominant_type}>")
            fingerprint_parts.append(dominant_type.upper())

    pattern = "/" + "/".join(pattern_parts)
    fingerprint = "/".join(fingerprint_parts)

    # Compute confidence: based on group size and type uniformity
    size_factor = min(1.0, len(group.urls) / 20)
    type_uniformity = sum(1 for p in pos_types.values() if p == "static") / max(len(pos_types), 1)
    confidence = round(0.5 + 0.3 * size_factor + 0.2 * type_uniformity, 4)
    confidence = min(confidence, 1.0)

    # Collect versions covered
    versions_covered = list({
        u.version for u in group_parsed if u.version
    })

    return TemplatePattern(
        template_id="",  # assigned later
        pattern=pattern,
        fingerprint=fingerprint,
        member_count=len(group.urls),
        versions_covered=versions_covered,
        example_urls=group.urls[:MAX_EXAMPLES_PER_TEMPLATE],
        confidence=confidence,
    )


def merge_similar_templates(
    templates: list[TemplatePattern],
) -> list[TemplatePattern]:
    """
    After initial template derivation, check if any templates should be merged.

    Merge criteria:
    Two templates can be merged if:
    1. They have the same number of segments
    2. Their fingerprints differ in at most TEMPLATE_MERGE_DISTANCE positions
    3. The differing positions are all variable (both are type names, not literals)
    """
    if len(templates) <= 1:
        return templates

    merged = list(templates)
    changed = True

    while changed:
        changed = False
        result = []
        skip = set()

        for i, tpl_a in enumerate(merged):
            if i in skip:
                continue
            merged_with = None

            parts_a = tpl_a.fingerprint.split("/")

            for j, tpl_b in enumerate(merged):
                if j <= i or j in skip:
                    continue
                parts_b = tpl_b.fingerprint.split("/")

                if len(parts_a) != len(parts_b):
                    continue

                # Count differing positions
                diffs = [
                    k for k, (pa, pb) in enumerate(zip(parts_a, parts_b))
                    if pa != pb
                ]

                if len(diffs) > TEMPLATE_MERGE_DISTANCE:
                    continue

                # All diffs must be variable (uppercase type names, not literal values)
                all_variable = all(
                    parts_a[k] == parts_a[k].upper() and parts_a[k].replace("_", "").isalpha()
                    and parts_b[k] == parts_b[k].upper() and parts_b[k].replace("_", "").isalpha()
                    for k in diffs
                )
                if not all_variable:
                    continue

                # Merge: keep the bigger template, absorb the smaller
                if tpl_a.member_count >= tpl_b.member_count:
                    winner = tpl_a
                    loser = tpl_b
                else:
                    winner = tpl_b
                    loser = tpl_a

                new_examples = list(dict.fromkeys(winner.example_urls + loser.example_urls))[:MAX_EXAMPLES_PER_TEMPLATE]
                new_versions = list(set(winner.versions_covered + loser.versions_covered))
                merged_tpl = winner.model_copy(update={
                    "member_count": winner.member_count + loser.member_count,
                    "example_urls": new_examples,
                    "versions_covered": new_versions,
                })
                merged_with = merged_tpl
                skip.add(j)
                changed = True
                break

            if merged_with:
                result.append(merged_with)
            else:
                result.append(tpl_a)

        merged = result

    return merged


def finalize_templates(
    groups: list[URLGroup],
    all_urls: list[ParsedURL],
) -> list[TemplatePattern]:
    """
    Convert URLGroups → TemplatePatterns, then merge similar ones.

    Steps:
    1. For each group, call derive_template(group, all_urls)
    2. Assign template_ids: tpl_001, tpl_002, tpl_003, ...
    3. Call merge_similar_templates(templates)
    4. Re-assign template_ids after merge (no gaps)
    5. Sort by member_count descending (biggest template first)
    6. Return final list
    """
    templates = []
    for i, group in enumerate(groups, start=1):
        tpl = derive_template(group, all_urls)
        tpl = tpl.model_copy(update={"template_id": f"tpl_{i:03d}"})
        templates.append(tpl)

    # Merge similar
    templates = merge_similar_templates(templates)

    # Sort by member_count descending
    templates.sort(key=lambda t: t.member_count, reverse=True)

    # Re-assign IDs
    final = []
    for i, tpl in enumerate(templates, start=1):
        name = _infer_template_name(tpl.pattern)
        final.append(tpl.model_copy(update={
            "template_id": f"tpl_{i:03d}",
            "name": name,
        }))

    return final


def _infer_template_name(pattern: str) -> str:
    """
    Heuristically infer a human-readable name from the pattern string.
    """
    lower = pattern.lower()
    if "generated" in lower or "api" in lower:
        return "api_reference"
    if "user_guide" in lower or "guide" in lower:
        return "user_guide"
    if "auto_example" in lower or "example" in lower:
        return "examples"
    if "tutorial" in lower:
        return "tutorials"
    if "changelog" in lower or "release" in lower:
        return "changelog"
    return "catch_all"
