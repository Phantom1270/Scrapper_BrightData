"""
Compress HTML to a structural skeleton.
Strip all content, keep only tag names, class names, nesting.
Used for sending to LLM without wasting tokens on content.
"""

from bs4 import BeautifulSoup, Tag


# CSS classes that indicate structural elements (not content)
STRUCTURAL_CLASSES = {
    'document', 'body', 'header', 'footer', 'sidebar', 'content',
    'navbar', 'nav', 'main', 'section', 'article', 'aside',
    # Sphinx specific
    'sig', 'sig-object', 'sig-name', 'sig-prename', 'descclassname',
    'descname', 'field-list', 'field-body', 'highlight', 'docutils',
    'admonition', 'note', 'warning', 'deprecated', 'versionadded',
    'versionchanged', 'rubric', 'toctree', 'reference', 'internal',
    'external', 'code', 'literal', 'pre', 'desc', 'class', 'method',
    'function', 'attribute', 'property', 'module', 'py method',
    'py function', 'py class', 'py module',
    # MkDocs specific
    'md-content', 'md-sidebar', 'md-header', 'md-footer', 'md-main',
    'admonition', 'note', 'warning', 'tip', 'info',
    # Docusaurus specific
    'markdown', 'toc', 'breadcrumbs', 'pagination',
    # Sphinx-gallery (examples)
    'sphx-glr', 'sphx-glr-thumbcontainer', 'sphx-glr-script-out',
    'sphx-glr-single-img',
}


def compress_html(html: str, max_chars: int = 6000) -> str:
    """
    Compress HTML to structural skeleton.

    Input:  Full HTML page (could be 100KB+)
    Output: Structural skeleton (~2-6KB)

    What we KEEP:
    - Tag names (h1, h2, div, section, table, etc.)
    - Structural CSS classes (document, body, sig, field-list, etc.)
    - Nesting structure
    - Text length indicators ([SHORT_TEXT], [LONG_TEXT])

    What we REMOVE:
    - All script and style tags
    - All SVG and image data
    - All non-structural CSS classes
    - All inline styles
    - All data attributes
    - Actual text content (replace with length indicator)
    """
    soup = BeautifulSoup(html, 'lxml')

    # Remove noise elements entirely
    for tag_name in ['script', 'style', 'svg', 'noscript', 'iframe',
                     'link', 'meta', 'path', 'circle', 'rect']:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Get the body (or root if no body)
    body = soup.find('body') or soup

    # Walk the tree and build compressed representation
    lines = []
    _walk_element(body, lines, depth=0)

    result = '\n'.join(lines)

    # Truncate if too long
    if len(result) > max_chars:
        result = result[:max_chars] + "\n... [TRUNCATED]"

    return result


def _walk_element(element, lines: list[str], depth: int) -> None:
    """Recursively walk HTML tree, building compressed lines."""
    if not isinstance(element, Tag):
        return

    # Skip hidden elements
    style = element.get('style', '')
    if 'display:none' in style or 'display: none' in style:
        return

    tag_name = element.name

    # Get structural classes only
    classes = element.get('class', [])
    structural = [c for c in classes if c.lower() in STRUCTURAL_CLASSES]

    # Build the line
    class_str = '.'.join(structural[:3])  # Max 3 classes to keep it short
    if class_str:
        entry = f"{tag_name}.{class_str}"
    else:
        entry = tag_name

    # Add ID if it looks structural (not random)
    elem_id = element.get('id', '')
    if elem_id and len(elem_id) < 40 and not elem_id.startswith('id-'):
        entry += f"#{elem_id}"

    # Check for direct text content
    direct_text = element.get_text(strip=True)
    children_tags = [c for c in element.children if isinstance(c, Tag)]

    if not children_tags and direct_text:
        # Leaf node with text
        if len(direct_text) > 80:
            entry += " [LONG_TEXT]"
        else:
            entry += f' "{direct_text[:50]}"'

    indent = '  ' * depth
    lines.append(f"{indent}{entry}")

    # Recurse into children (limit depth to avoid noise)
    if depth < 12:
        for child in element.children:
            if isinstance(child, Tag):
                _walk_element(child, lines, depth + 1)
