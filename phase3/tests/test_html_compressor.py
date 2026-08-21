from utils.html_compressor import compress_html


class TestCompressHTML:
    def test_removes_scripts(self):
        html = "<html><body><script>alert('x')</script><p>Hello</p></body></html>"
        result = compress_html(html)
        assert "alert" not in result
        assert "Hello" in result or "p" in result

    def test_removes_styles(self):
        html = "<html><body><style>.x{color:red}</style><p>Text</p></body></html>"
        result = compress_html(html)
        assert "color:red" not in result

    def test_keeps_structure(self):
        html = """
        <html><body>
            <div class="document">
                <div class="body">
                    <h1>Title</h1>
                    <p>Content here that is long enough</p>
                </div>
            </div>
        </body></html>
        """
        result = compress_html(html)
        assert "document" in result
        assert "h1" in result

    def test_replaces_long_text(self):
        long_text = "word " * 100
        html = f"<html><body><p>{long_text}</p></body></html>"
        result = compress_html(html)
        assert "LONG_TEXT" in result

    def test_respects_max_chars(self):
        html = "<html><body>" + "<div><p>text content</p></div>" * 500 + "</body></html>"
        result = compress_html(html, max_chars=500)
        assert len(result) <= 600

    def test_empty_html(self):
        result = compress_html("")
        assert isinstance(result, str)

    def test_malformed_html(self):
        html = "<div><p>unclosed<p>tags<div>"
        result = compress_html(html)
        assert isinstance(result, str)
        assert len(result) > 0
