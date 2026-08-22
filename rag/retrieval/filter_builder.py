"""
Infers metadata filters from the user's query.
"""

import re
from typing import Dict, Optional


class MetadataFilterBuilder:
    """Analyze a user query and determine implied metadata constraints."""

    def __init__(self, settings=None):
        pass

    def build_filters(self, query: str) -> Dict[str, Optional[str]]:
        """
        Analyze the query text and return a dict of filters.
        """
        filter_content_type = None
        query_lower = query.lower()

        # Tutorial signals
        tutorial_signals = [
            "example", "sample", "demo", "tutorial", "how to", "how do i",
            "walkthrough", "getting started"
        ]
        
        # API Reference signals
        api_signals = [
            "parameter", "argument", "option", "config", "setting",
            "function", "method", "class", "api", "signature",
            "constructor", "import", "module"
        ]
        
        # Notebook signals
        notebook_signals = [
            "notebook", "jupyter", "ipynb"
        ]

        # Check for tutorial signals
        if any(signal in query_lower for signal in tutorial_signals):
            filter_content_type = "tutorial"
            
        # Check for API signals (wins over tutorial if both are present in the code order)
        # But wait, the requirements say "Filter rules (applied in order): 1... 2... 3..."
        # If we check tutorial first, and then api, api will overwrite tutorial.
        # Let's check them strictly in order and break early.
        
        # The prompt says: "Query with both tutorial and API signals -> the first match wins based on priority order in the implementation."
        # The priority order from prompt: 1. tutorial, 2. API, 3. notebook.
        
        for signal in tutorial_signals:
            if re.search(r'\b' + re.escape(signal) + r'\b', query_lower):
                return {"content_type": "tutorial"}
                
        for signal in api_signals:
            if re.search(r'\b' + re.escape(signal) + r'\b', query_lower):
                return {"content_type": "api_reference"}
                
        for signal in notebook_signals:
            if re.search(r'\b' + re.escape(signal) + r'\b', query_lower):
                return {"content_type": "notebook"}
                
        # If no regex boundary match, fallback to simple substring if needed, 
        # but regex word boundary is safer so "example" doesn't match "examples" unless we want it to.
        # Actually, let's just do simple substring match to be more lenient, as "examples" should trigger "example" logic.
        
        if any(s in query_lower for s in tutorial_signals):
            return {"content_type": "tutorial"}
            
        if any(s in query_lower for s in api_signals):
            return {"content_type": "api_reference"}
            
        if any(s in query_lower for s in notebook_signals):
            return {"content_type": "notebook"}

        return {"content_type": None}

    def should_filter(self, query: str) -> bool:
        """Returns True if build_filters would produce any non-None filter."""
        filters = self.build_filters(query)
        return filters.get("content_type") is not None
