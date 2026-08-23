import pytest
from rag.retrieval.filter_builder import MetadataFilterBuilder


class TestMetadataFilterBuilder:
    def test_detects_tutorial_content_type(self):
        builder = MetadataFilterBuilder()
        assert builder.build_filters("how to install sklearn") == {"content_type": "tutorial"}
        assert builder.build_filters("give me an example") == {"content_type": "tutorial"}
        assert builder.build_filters("show a demo of this") == {"content_type": "tutorial"}

    def test_detects_api_reference_content_type(self):
        builder = MetadataFilterBuilder()
        assert builder.build_filters("what parameters does config_context accept") == {"content_type": "api_reference"}
        assert builder.build_filters("show the class for SVM") == {"content_type": "api_reference"}
        assert builder.build_filters("what is the default config") == {"content_type": "api_reference"}

    def test_detects_api_reference_function_query(self):
        builder = MetadataFilterBuilder()
        assert builder.build_filters("sklearn.set_config function signature") == {"content_type": "api_reference"}

    def test_detects_notebook_content_type(self):
        builder = MetadataFilterBuilder()
        assert builder.build_filters("show me the jupyter notebook") == {"content_type": "notebook"}
        assert builder.build_filters("where is the ipynb") == {"content_type": "notebook"}

    def test_no_filter_for_general_query(self):
        builder = MetadataFilterBuilder()
        assert builder.build_filters("what is machine learning") == {"content_type": None}
        assert builder.build_filters("explain standard scaler") == {"content_type": None}

    def test_should_filter_returns_true_for_constrained_query(self):
        builder = MetadataFilterBuilder()
        assert builder.should_filter("config_context parameters") is True
        assert builder.should_filter("how to do something") is True

    def test_should_filter_returns_false_for_general_query(self):
        builder = MetadataFilterBuilder()
        assert builder.should_filter("what is sklearn") is False

    def test_filter_is_case_insensitive(self):
        builder = MetadataFilterBuilder()
        f1 = builder.build_filters("API Reference")
        f2 = builder.build_filters("api reference")
        assert f1 == f2
        assert f1["content_type"] == "api_reference"

    def test_multiple_signals_takes_strongest(self):
        builder = MetadataFilterBuilder()
        # "example" is tutorial, "function" is api_reference
        # Tutorial is checked first in the implementation, so it should win
        filters = builder.build_filters("example of the function")
        assert filters["content_type"] == "tutorial"
