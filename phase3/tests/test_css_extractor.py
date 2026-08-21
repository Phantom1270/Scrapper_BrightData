from utils.css_extractor import extract_from_html
from models import (
    ValidationSchema, FieldSchema, FieldImportance, FieldType
)


def make_schema(fields):
    return ValidationSchema(
        template_id="test",
        template_pattern="/test/<filename>",
        fields=fields,
    )


class TestCSSExtractor:
    def test_extract_text(self):
        html = "<html><body><h1 class='title'>Hello World</h1></body></html>"
        schema = make_schema([
            FieldSchema(
                name="title", field_type=FieldType.TEXT,
                importance=FieldImportance.REQUIRED, css_selector="h1.title",
            ),
        ])
        result = extract_from_html(html, schema)
        assert result["title"] == "Hello World"

    def test_missing_field_returns_none(self):
        html = "<html><body><p>Content</p></body></html>"
        schema = make_schema([
            FieldSchema(
                name="title", field_type=FieldType.TEXT,
                importance=FieldImportance.REQUIRED, css_selector="h1",
            ),
        ])
        result = extract_from_html(html, schema)
        assert result["title"] is None

    def test_fallback_selector(self):
        html = "<html><body><span class='page-title'>My Title</span></body></html>"
        schema = make_schema([
            FieldSchema(
                name="title", field_type=FieldType.TEXT,
                importance=FieldImportance.REQUIRED, css_selector="h1",
                fallback_selectors=[".page-title"],
            ),
        ])
        result = extract_from_html(html, schema)
        assert result["title"] == "My Title"

    def test_extract_list(self):
        html = """
        <html><body>
            <ul>
                <li>Alpha</li>
                <li>Beta</li>
                <li>Gamma</li>
            </ul>
        </body></html>
        """
        schema = make_schema([
            FieldSchema(
                name="items", field_type=FieldType.LIST,
                importance=FieldImportance.REQUIRED, css_selector="ul",
            ),
        ])
        result = extract_from_html(html, schema)
        assert len(result["items"]) == 3
        assert "Alpha" in result["items"]

    def test_extract_table(self):
        html = """
        <html><body>
            <table>
                <tr><th>Name</th><th>Type</th></tr>
                <tr><td>X</td><td>int</td></tr>
                <tr><td>Y</td><td>str</td></tr>
            </table>
        </body></html>
        """
        schema = make_schema([
            FieldSchema(
                name="params", field_type=FieldType.TABLE,
                importance=FieldImportance.REQUIRED, css_selector="table",
            ),
        ])
        result = extract_from_html(html, schema)
        assert len(result["params"]) == 2
        assert result["params"][0]["Name"] == "X"

    def test_extract_code(self):
        html = """
        <html><body>
            <div class="highlight">
                <pre>print("hello")</pre>
            </div>
        </body></html>
        """
        schema = make_schema([
            FieldSchema(
                name="code", field_type=FieldType.CODE,
                importance=FieldImportance.OPTIONAL, css_selector=".highlight",
            ),
        ])
        result = extract_from_html(html, schema)
        assert "hello" in result["code"][0]

    def test_extract_urls(self):
        html = """
        <html><body>
            <nav>
                <a href="/docs">Docs</a>
                <a href="/api">API</a>
            </nav>
        </body></html>
        """
        schema = make_schema([
            FieldSchema(
                name="links", field_type=FieldType.URL,
                importance=FieldImportance.OPTIONAL, css_selector="nav a",
            ),
        ])
        result = extract_from_html(html, schema)
        assert "/docs" in result["links"]
        assert "/api" in result["links"]
