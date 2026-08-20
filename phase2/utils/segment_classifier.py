"""
Higher-level segment analysis utilities.
Works on collections of ParsedURL objects.
Used by the structural grouping stage.
"""

from models import ParsedURL, ParsedSegment, SegmentType
from typing import Optional


def compute_position_frequencies(
    urls: list[ParsedURL],
) -> dict[int, dict[str, float]]:
    """
    For each path position (0, 1, 2, ...), compute how frequently
    each unique segment value appears.

    Input: 143 URLs, all with path like /modules/generated/X.html
    Output: {
        0: {"modules": 1.0},          # position 0 is always "modules"
        1: {"generated": 1.0},        # position 1 is always "generated"
        2: {"sklearn.linear_model.LogisticRegression.html": 0.007,
            "sklearn.ensemble.RandomForestClassifier.html": 0.007,
            ...}                       # position 2 is highly variable
    }

    IMPORTANT:
    - Only process URLs that have version_free_path set (if available),
      otherwise use path.
    - All URLs in the input list should have the same depth.
      If they don't, only use positions up to the minimum depth.
    """
    if not urls:
        return {}

    # Collect segments for each URL
    def get_segments(url: ParsedURL) -> list[str]:
        path = url.version_free_path if url.version_free_path else url.path
        return [s for s in path.split("/") if s]

    all_segments = [get_segments(u) for u in urls]

    # Find minimum depth to stay safe
    if all_segments:
        min_depth = min(len(s) for s in all_segments)
    else:
        return {}

    total = len(urls)
    position_counts: dict[int, dict[str, int]] = {}

    for segments in all_segments:
        for pos in range(min(len(segments), min_depth)):
            val = segments[pos]
            if pos not in position_counts:
                position_counts[pos] = {}
            position_counts[pos][val] = position_counts[pos].get(val, 0) + 1

    # Convert counts to frequencies
    position_frequencies: dict[int, dict[str, float]] = {}
    for pos, counts in position_counts.items():
        position_frequencies[pos] = {val: cnt / total for val, cnt in counts.items()}

    return position_frequencies


def classify_positions(
    position_frequencies: dict[int, dict[str, float]],
    static_threshold: float = 0.80,
    variable_threshold: float = 0.10,
) -> dict[int, str]:
    """
    Based on frequency analysis, determine if each position is
    "static" or "variable".

    Algorithm:
    - If ANY single value at a position has frequency >= static_threshold → "static"
    - If NO single value has frequency >= variable_threshold → "variable"
    - Otherwise → "semi_variable" (a few repeated values, like enum)

    Input: position_frequencies from compute_position_frequencies
    Output: {0: "static", 1: "static", 2: "variable"}
    """
    result: dict[int, str] = {}
    for pos, freq_dict in position_frequencies.items():
        max_freq = max(freq_dict.values()) if freq_dict else 0.0
        if max_freq >= static_threshold:
            result[pos] = "static"
        elif max_freq < variable_threshold:
            result[pos] = "variable"
        else:
            result[pos] = "semi_variable"
    return result


def generate_fingerprint(
    segments: list[ParsedSegment],
    position_types: dict[int, str],
) -> str:
    """
    Generate a structural fingerprint string for a URL's path.

    For static positions, use the actual value.
    For variable positions, use the SegmentType name.

    Input segments: ["modules", "generated", "sklearn.linear_model.LogisticRegression.html"]
    Input position_types: {0: "static", 1: "static", 2: "variable"}
    Segment types: [LITERAL, LITERAL, FILENAME]

    Output: "modules/generated/FILENAME"

    For a different URL: ["user_guide", "linear_model.html"]
    Position types: {0: "static", 1: "variable"}
    Segment types: [LITERAL, FILENAME]

    Output: "user_guide/FILENAME"
    """
    parts = []
    for seg in segments:
        pos_type = position_types.get(seg.position, "variable")
        if pos_type == "static":
            parts.append(seg.raw)
        else:
            parts.append(seg.lexical_type.value.upper())
    return "/".join(parts)
