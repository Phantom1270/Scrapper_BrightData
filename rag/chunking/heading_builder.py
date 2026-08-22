"""
Utility for constructing consistent heading_path breadcrumbs.
"""

from typing import List


class HeadingBuilder:
    """Constructs and formats heading paths for chunk context."""

    def build_path(
        self,
        document_title: str,
        block_heading: str = "",
        extra_levels: List[str] = None
    ) -> List[str]:
        """
        Construct a heading_path list.
        - Always starts with document_title.
        - Appends extra_levels if provided.
        - Appends block_heading if non-empty.
        - Deduplicates consecutive identical headings.
        - Strips trailing "#" from all headings.
        """
        raw_path = [document_title]
        if extra_levels:
            raw_path.extend(extra_levels)
        if block_heading:
            raw_path.append(block_heading)

        # Clean and deduplicate
        final_path: List[str] = []
        for level in raw_path:
            if not level:
                continue
            
            # Clean trailing spaces and '#' 
            cleaned = level.strip().rstrip("#").strip()
            if not cleaned:
                continue
                
            # Deduplicate consecutive identical headings
            if not final_path or final_path[-1] != cleaned:
                final_path.append(cleaned)

        return final_path

    def path_to_string(self, heading_path: List[str]) -> str:
        """Join heading_path with ' > ' separator."""
        return " > ".join(heading_path)

    def path_to_prefixed_text(self, heading_path: List[str], content: str) -> str:
        """
        Prepend heading path as a formatted header to content text.
        Format: "## {path_to_string}\n\n{content}"
        """
        if not heading_path:
            return content
        path_str = self.path_to_string(heading_path)
        return f"## {path_str}\n\n{content}"
