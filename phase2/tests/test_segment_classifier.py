"""
Unit tests for utils/segment_classifier.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.segment_classifier import (
    compute_position_frequencies, classify_positions, generate_fingerprint
)
from models import ParsedURL, ParsedSegment, URLClassification, SegmentType


def make_url(path: str, vfp: str = None) -> ParsedURL:
    """Minimal ParsedURL helper."""
    segs = [s for s in path.split("/") if s]
    parsed_segs = [
        ParsedSegment(raw=s, position=i, lexical_type=SegmentType.LITERAL)
        for i, s in enumerate(segs)
    ]
    return ParsedURL(
        original_url=f"https://example.com{path}",
        canonical_url=f"https://example.com{path}",
        domain="example.com",
        scheme="https",
        path=path,
        segments=parsed_segs,
        classification=URLClassification.INTERNAL,
        is_asset=False,
        version_free_path=vfp or path,
    )


class TestComputePositionFrequencies:
    def test_all_same_prefix(self):
        urls = [
            make_url("/modules/generated/a.html"),
            make_url("/modules/generated/b.html"),
            make_url("/modules/generated/c.html"),
        ]
        freqs = compute_position_frequencies(urls)
        assert freqs[0]["modules"] == 1.0
        assert freqs[1]["generated"] == 1.0
        # position 2 should have 3 different values
        assert len(freqs[2]) == 3

    def test_empty_input(self):
        assert compute_position_frequencies([]) == {}

    def test_mixed_depth(self):
        """Should use min depth."""
        urls = [
            make_url("/a/b/c.html"),
            make_url("/a/d.html"),
        ]
        freqs = compute_position_frequencies(urls)
        # Min depth = 2, so only positions 0 and 1 used
        assert 0 in freqs
        assert 1 in freqs


class TestClassifyPositions:
    def test_static_position(self):
        freqs = {0: {"modules": 1.0}}
        result = classify_positions(freqs)
        assert result[0] == "static"

    def test_variable_position(self):
        # Many unique values, none above threshold
        freqs = {
            0: {f"item_{i}": 0.01 for i in range(100)}
        }
        result = classify_positions(freqs)
        assert result[0] == "variable"

    def test_semi_variable(self):
        freqs = {0: {"a": 0.5, "b": 0.3, "c": 0.2}}
        result = classify_positions(freqs)
        assert result[0] == "semi_variable"


class TestGenerateFingerprint:
    def test_all_static(self):
        segs = [
            ParsedSegment(raw="modules", position=0, lexical_type=SegmentType.LITERAL),
            ParsedSegment(raw="generated", position=1, lexical_type=SegmentType.LITERAL),
        ]
        pos_types = {0: "static", 1: "static"}
        result = generate_fingerprint(segs, pos_types)
        assert result == "modules/generated"

    def test_mixed(self):
        segs = [
            ParsedSegment(raw="modules", position=0, lexical_type=SegmentType.LITERAL),
            ParsedSegment(raw="generated", position=1, lexical_type=SegmentType.LITERAL),
            ParsedSegment(raw="LogisticRegression.html", position=2, lexical_type=SegmentType.FILENAME),
        ]
        pos_types = {0: "static", 1: "static", 2: "variable"}
        result = generate_fingerprint(segs, pos_types)
        assert result == "modules/generated/FILENAME"
