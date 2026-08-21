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
from utils.url_parser import split_path_segments, classify_segment
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


def discover_groups(urls, generator_hint=None):
    """
    CORE RULE: Structural junctions are section boundaries.
    Never merge URLs from different sections into the same group.

    EXCEPTION: If a junction's children all have identical subtree shapes
    (same depth, same child types), they are "category variants" like
    auto_examples/cluster, auto_examples/linear_model. These CAN merge.
    """
    from collections import defaultdict

    trie = build_trie(urls, use_version_free_path=True)
    url_lookup = {u.canonical_url: u for u in urls}
    groups = []
    assigned_urls = set()

    def _is_section_name(segment, node):
        seg_type = classify_segment(segment)
        if seg_type not in (SegmentType.LITERAL, SegmentType.SLUG):
            return False
        if node.descendant_count < 3:
            return False
        return True

    def _are_category_variants(children):
        """Check if children are same content type under different names."""
        if len(children) < 2:
            return False
        profiles = []
        for seg, child in children.items():
            leaves = child.collect_leaves()
            if not leaves:
                continue
            profiles.append({
                "child_literal_ratio": child.child_literal_ratio,
                "descendant_count": child.descendant_count,
                "has_children": len(child.children) > 0,
            })
        if len(profiles) < 2:
            return False
        all_variable = all(p["child_literal_ratio"] < 0.5 for p in profiles)
        all_have_content = all(p["descendant_count"] > 0 for p in profiles)
        all_same_depth = len(set(p["has_children"] for p in profiles)) == 1
        return all_variable and all_have_content and all_same_depth

    def _collect_group(node):
        leaves = node.collect_leaves()
        uncovered = [u for u in leaves if u not in assigned_urls]
        if len(uncovered) < MIN_GROUP_SIZE:
            return
        prefix = "/" + node.path_from_root if node.path_from_root else "/"
        group = URLGroup(
            group_id=f"grp_{len(groups) + 1:03d}",
            urls=uncovered,
            path_prefix=prefix,
        )
        groups.append(group)
        assigned_urls.update(uncovered)

    def process_node(node):
        if not node.children:
            return
        literal_children = {}
        for seg, child in node.children.items():
            if _is_section_name(seg, child):
                literal_children[seg] = child
        is_junction = len(literal_children) >= 2

        # Special case: exactly 1 literal child with a large subtree alongside non-literal children
        is_mixed_with_one_large_literal = False
        if len(literal_children) == 1 and len(node.children) > 1:
            literal_child = list(literal_children.values())[0]
            if literal_child.descendant_count >= MIN_GROUP_SIZE:
                is_mixed_with_one_large_literal = True

        if is_junction or is_mixed_with_one_large_literal:
            if is_junction and _are_category_variants(literal_children):
                _collect_group(node)
            else:
                for seg, child in node.children.items():
                    process_node(child)
        else:
            _collect_group(node)

    process_node(trie)

    # Collect remaining uncovered URLs by depth
    all_leaves = trie.collect_leaves()
    remaining = [u for u in all_leaves if u not in assigned_urls]
    if remaining:
        by_depth = defaultdict(list)
        for url_str in remaining:
            u = url_lookup.get(url_str)
            if u:
                path = u.version_free_path or u.path
                depth = len(split_path_segments(path))
                by_depth[depth].append(url_str)
        for depth, depth_urls in sorted(by_depth.items()):
            if len(depth_urls) >= MIN_GROUP_SIZE:
                group = URLGroup(
                    group_id=f"grp_{len(groups) + 1:03d}",
                    urls=depth_urls,
                    path_prefix=f"depth_{depth}",
                )
                groups.append(group)
                assigned_urls.update(depth_urls)

    return groups

def derive_template(group, all_urls):
    """
    RULES:
    - If ALL URLs have same value at position → use literal value
    - If values differ → use <type> placeholder
    - NEVER replace a known literal with <literal>
    """
    from collections import Counter

    url_set = set(group.urls)
    group_urls = [u for u in all_urls if u.canonical_url in url_set]
    if not group_urls:
        return TemplatePattern(
            template_id="", pattern=group.path_prefix,
            fingerprint="", member_count=0, confidence=0.0,
        )

    all_seg_lists = []
    for u in group_urls:
        p = u.version_free_path or u.path
        all_seg_lists.append(split_path_segments(p))

    depths = [len(s) for s in all_seg_lists]
    min_depth = min(depths)
    max_depth = max(depths)

    pattern_parts = []
    fingerprint_parts = []

    for pos in range(min_depth):
        values = [sl[pos] for sl in all_seg_lists if len(sl) > pos]
        unique_values = set(values)
        total = len(values)

        if len(unique_values) == 1:
            value = values[0]
            pattern_parts.append(value)
            fingerprint_parts.append(value.upper())
        elif len(unique_values) <= 3:
            most_common_val, most_common_count = Counter(values).most_common(1)[0]
            freq = most_common_count / total
            if freq >= 0.75:
                pattern_parts.append(most_common_val)
                fingerprint_parts.append(most_common_val.upper())
            else:
                types = [classify_segment(v) for v in values]
                most_common_type = Counter(types).most_common(1)[0][0]
                pattern_parts.append(f"<{most_common_type.value}>")
                fingerprint_parts.append(most_common_type.value)
        else:
            types = [classify_segment(v) for v in values]
            type_counts = Counter(types)
            most_common_type = type_counts.most_common(1)[0][0]
            pattern_parts.append(f"<{most_common_type.value}>")
            fingerprint_parts.append(most_common_type.value)

    if max_depth > min_depth:
        deeper_patterns = []
        for sl in all_seg_lists:
            if len(sl) > min_depth:
                deeper_patterns.append("/".join(sl[min_depth:]))
        unique_deeper = set(deeper_patterns)
        if len(unique_deeper) == 1:
            pattern_parts.append(unique_deeper.pop())
            fingerprint_parts.append("STATIC_DEEP")
        else:
            pattern_parts.append("...")
            fingerprint_parts.append("DEEPER")

    pattern = "/" + "/".join(pattern_parts)
    fingerprint = "/".join(fingerprint_parts)

    count_score = min(len(group_urls) / 50.0, 1.0)
    static_positions = sum(
        1 for pos in range(min_depth)
        if len(set(sl[pos] for sl in all_seg_lists if len(sl) > pos)) == 1
    )
    uniformity = static_positions / max(min_depth, 1)
    has_prefix = bool(group.path_prefix and group.path_prefix != "/" and not group.path_prefix.startswith("/<"))
    prefix_bonus = 0.15 if has_prefix else 0.0
    confidence = min(count_score * 0.25 + uniformity * 0.5 + prefix_bonus + 0.15, 1.0)
    confidence = max(confidence, 0.1)

    versions = sorted(set(u.version for u in group_urls if u.version))
    name = _infer_template_name(group, pattern)
    examples = [u.canonical_url for u in group_urls[:MAX_EXAMPLES_PER_TEMPLATE]]

    return TemplatePattern(
        template_id="", name=name, pattern=pattern, fingerprint=fingerprint,
        member_count=len(group_urls), versions_covered=versions,
        scope="internal", example_urls=examples, confidence=round(confidence, 2),
    )

def _infer_template_name(group, pattern):
    prefix = group.path_prefix.lower()
    pattern_lower = pattern.lower()
    name_map = {
        "modules/generated": "api_reference",
        "user_guide": "user_guide",
        "auto_examples": "examples",
        "whats_new": "changelog",
        "developers": "developer_docs",
        "api": "api_reference",
        "_downloads": "downloads",
        "lite/lab": "interactive_lite",
        "modules": "module_overview",
    }
    for key, name in name_map.items():
        if key in prefix:
            return name
    if "generated" in pattern_lower:
        return "api_reference"
    segments = prefix.strip("/").split("/")
    for seg in segments:
        if seg and not seg.startswith("<") and not seg.startswith("depth_"):
            return seg
    return "catch_all"

def merge_similar_templates(templates):
    """Merge ONLY if truly same content type. If ANY literal differs, DO NOT MERGE."""
    if len(templates) <= 1:
        return templates

    merged = list(templates)
    changed = True
    while changed:
        changed = False
        result = []
        used = set()
        for i, t1 in enumerate(merged):
            if i in used:
                continue
            best_match_idx = None
            best_distance = float('inf')
            for j, t2 in enumerate(merged):
                if j <= i or j in used:
                    continue
                distance = _compute_merge_distance(t1, t2)
                if distance is not None and distance <= TEMPLATE_MERGE_DISTANCE and distance < best_distance:
                    best_match_idx = j
                    best_distance = distance
            if best_match_idx is not None:
                t2 = merged[best_match_idx]
                merged_template = _do_merge(t1, t2)
                result.append(merged_template)
                used.add(i)
                used.add(best_match_idx)
                changed = True
            else:
                result.append(t1)
                used.add(i)
        merged = result
    return merged

def _compute_merge_distance(t1, t2):
    """
    Returns None if cannot merge. Returns int if mergeable.
    Fingerprint format: literal values UPPERCASE, type names lowercase.
    """
    parts1 = t1.fingerprint.split("/")
    parts2 = t2.fingerprint.split("/")
    if len(parts1) != len(parts2):
        return None
    distance = 0
    for p1, p2 in zip(parts1, parts2):
        if p1 == p2:
            continue
        p1_is_literal = p1.isalpha() and p1 == p1.upper()
        p2_is_literal = p2.isalpha() and p2 == p2.upper()
        if p1_is_literal or p2_is_literal:
            return None  # CANNOT merge if any literal differs
        distance += 1
    return distance

def _do_merge(t1, t2):
    if t1.member_count >= t2.member_count:
        primary, secondary = t1, t2
    else:
        primary, secondary = t2, t1
    merged_examples = list(set(primary.example_urls + secondary.example_urls))[:MAX_EXAMPLES_PER_TEMPLATE]
    merged_versions = sorted(set(primary.versions_covered + secondary.versions_covered))
    return TemplatePattern(
        template_id=primary.template_id, name=primary.name,
        pattern=primary.pattern, fingerprint=primary.fingerprint,
        member_count=primary.member_count + secondary.member_count,
        versions_covered=merged_versions, scope=primary.scope,
        example_urls=merged_examples,
        confidence=min(primary.confidence, secondary.confidence),
    )

def finalize_templates(groups, all_urls):
    templates = []
    for group in groups:
        template = derive_template(group, all_urls)
        if template.member_count > 0:
            templates.append(template)
    templates = merge_similar_templates(templates)
    def sort_key(t):
        parts = t.pattern.strip("/").split("/")
        literal_count = sum(1 for p in parts if not p.startswith("<") and p != "...")
        return (-literal_count, -len(parts), -t.member_count)
    templates.sort(key=sort_key)
    for i, template in enumerate(templates):
        template.template_id = f"tpl_{i + 1:03d}"
    return templates
