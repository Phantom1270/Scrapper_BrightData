"""
Unit tests for pipeline/version_detection.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.version_detection import extract_version


class TestExtractVersion:
    def test_stable_prefix(self):
        version, vfp = extract_version("/stable/modules/generated/file.html")
        assert version == "stable"
        assert vfp == "/modules/generated/file.html"

    def test_numeric_version(self):
        version, vfp = extract_version("/0.24/user_guide/linear_model.html")
        assert version == "0.24"
        assert vfp == "/user_guide/linear_model.html"

    def test_language_and_version(self):
        version, vfp = extract_version("/en/stable/install.html")
        assert version == "stable"
        assert vfp == "/install.html"

    def test_no_version(self):
        version, vfp = extract_version("/install.html")
        assert version is None
        assert vfp == "/install.html"

    def test_dev_version(self):
        version, vfp = extract_version("/dev/modules/generated/file.html")
        assert version == "dev"
        assert vfp == "/modules/generated/file.html"

    def test_v_prefix_version(self):
        version, vfp = extract_version("/v2/api/reference.html")
        assert version == "v2"
        assert vfp == "/api/reference.html"

    def test_root_path(self):
        version, vfp = extract_version("/")
        assert version is None
