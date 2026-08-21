import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.generator_detection import detect_generator
from models import Phase1Output, ParsedURL, URLClassification

def make_phase1(signals=None, internal_urls=None):
    return Phase1Output(
        crawl_id="test",
        root_domain="example.com",
        root_urls=["https://example.com/"],
        signals=signals or {},
        internal_urls=internal_urls or [],
    )

def make_parsed_url(path):
    return ParsedURL(
        original_url=f"https://example.com{path}",
        canonical_url=f"https://example.com{path}",
        domain="example.com",
        scheme="https",
        path=path,
        segments=[],
        classification=URLClassification.INTERNAL,
        is_asset=False,
    )

class TestGeneratorDetection:
    def test_objects_inv_gives_sphinx_high_confidence(self):
        phase1 = make_phase1(signals={"has_objects_inv": True})
        gen, conf = detect_generator(phase1, [])
        assert gen == "sphinx"
        assert conf >= 0.90

    def test_static_dir_gives_sphinx_medium_confidence(self):
        urls = [make_parsed_url("/_static/css/theme.css")]
        phase1 = make_phase1()
        gen, conf = detect_generator(phase1, urls)
        assert gen == "sphinx"
        assert conf >= 0.60

    def test_no_signals_returns_none(self):
        phase1 = make_phase1()
        gen, conf = detect_generator(phase1, [])
        assert gen is None
        assert conf == 0.0

    def test_search_index_json_gives_mkdocs(self):
        urls = [make_parsed_url("/search/search_index.json")]
        phase1 = make_phase1()
        gen, conf = detect_generator(phase1, urls)
        assert gen == "mkdocs"
        assert conf >= 0.80
