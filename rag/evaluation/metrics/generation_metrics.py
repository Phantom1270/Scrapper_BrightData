"""
Metrics for measuring generation quality.
"""

from typing import List
import re


class GenerationMetrics:
    """Calculates heuristic generation metrics like faithfulness, relevance."""

    def __init__(self):
        self.stop_words = {
            "the", "is", "at", "which", "on", "a", "an", "and", "or", "but",
            "in", "with", "to", "for", "of", "how", "what", "does", "do", "are",
            "can", "could", "should", "would", "will", "this", "that", "these",
            "those", "it", "its"
        }

    def keyword_coverage(self, answer: str, expected_keywords: List[str]) -> float:
        """What fraction of expected keywords appear in the answer?"""
        if not expected_keywords:
            return 1.0
            
        answer_lower = answer.lower()
        found_count = 0
        
        for kw in expected_keywords:
            if kw.lower() in answer_lower:
                found_count += 1
                
        return found_count / len(expected_keywords)

    def faithfulness(self, answer: str, context: str) -> float:
        """Estimate how faithful the answer is to the provided context."""
        if not answer:
            return 1.0
            
        # Split answer into sentences
        # Simple heuristic split on sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 1.0
            
        context_lower = context.lower()
        sentences_with_support = 0
        
        for sentence in sentences:
            # Extract significant words
            words = [w.lower() for w in re.findall(r'\b\w+\b', sentence) if len(w) > 3]
            
            # Check if at least one significant word appears in context
            supported = False
            for word in words:
                if word in context_lower:
                    supported = True
                    break
                    
            if supported or not words:
                # If no significant words, assume supported (e.g. "Yes.")
                sentences_with_support += 1
                
        return sentences_with_support / len(sentences)

    def answer_relevance(self, answer: str, question: str) -> float:
        """Estimate how relevant the answer is to the question."""
        if not question or not answer:
            return 0.0
            
        q_words = set([
            w.lower() for w in re.findall(r'\b\w+\b', question)
            if w.lower() not in self.stop_words
        ])
        
        if not q_words:
            return 1.0
            
        a_words = set([
            w.lower() for w in re.findall(r'\b\w+\b', answer)
        ])
        
        overlap = len(q_words.intersection(a_words))
        relevance = overlap / len(q_words)
        
        return min(1.0, relevance)

    def evaluate_generation(self, answer: str, question: str, expected_keywords: List[str], context: str = "") -> dict:
        """Compute all generation metrics."""
        
        has_citations = bool(re.search(r'\[Source:\s*(.+?)\]', answer))
        
        answer_lower = answer.lower()
        refused_answer = any(phrase in answer_lower for phrase in [
            "i don't have enough information",
            "i couldn't find",
            "i do not have enough information",
            "i could not find"
        ])
        
        metrics = {
            "keyword_coverage": self.keyword_coverage(answer, expected_keywords),
            "answer_relevance": self.answer_relevance(answer, question),
            "answer_length_chars": len(answer),
            "answer_length_words": len(answer.split()),
            "has_citations": has_citations,
            "refused_answer": refused_answer,
        }
        
        if context:
            metrics["faithfulness"] = self.faithfulness(answer, context)
        else:
            metrics["faithfulness"] = None
            
        return metrics
