"""
Evaluation dataset format and loader.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import json
import os
from pathlib import Path


@dataclass
class EvalQuestion:
    """A single evaluation question with expected answers."""
    question: str
    expected_answer_keywords: List[str]
    expected_chunk_ids: List[str]
    expected_content_types: List[str]
    category: str = "general"
    difficulty: str = "medium"
    notes: str = ""


class TestSet:
    """A collection of evaluation questions loaded from a JSON file."""

    def __init__(self, questions: List[EvalQuestion], metadata: Dict[str, Any] = None):
        self.questions = questions
        self.metadata = metadata or {}

    def __len__(self) -> int:
        return len(self.questions)

    def get_by_category(self, category: str) -> List[EvalQuestion]:
        """Return questions matching the given category."""
        return [q for q in self.questions if q.category == category]

    def get_by_difficulty(self, difficulty: str) -> List[EvalQuestion]:
        """Return questions matching the given difficulty."""
        return [q for q in self.questions if q.difficulty == difficulty]

    @classmethod
    def load(cls, path: str) -> 'TestSet':
        """Load a TestSet from a JSON file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Test set file not found: {path}")
            
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        metadata = data.get("metadata", {})
        questions_data = data.get("questions", [])
        
        questions = []
        for q_data in questions_data:
            questions.append(EvalQuestion(
                question=q_data.get("question", ""),
                expected_answer_keywords=q_data.get("expected_answer_keywords", []),
                expected_chunk_ids=q_data.get("expected_chunk_ids", []),
                expected_content_types=q_data.get("expected_content_types", []),
                category=q_data.get("category", "general"),
                difficulty=q_data.get("difficulty", "medium"),
                notes=q_data.get("notes", "")
            ))
            
        return cls(questions=questions, metadata=metadata)

    def save(self, path: str) -> None:
        """Save the test set to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "metadata": self.metadata,
            "questions": [asdict(q) for q in self.questions]
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    @classmethod
    def create_sample(cls) -> 'TestSet':
        """Create a sample test set with generic questions."""
        metadata = {
            "name": "Sample RAG Evaluation Set",
            "version": "1.0",
            "created": "2024-01-15",
            "description": "Placeholder evaluation questions. Replace with real questions after indexing your documentation."
        }
        
        questions = [
            EvalQuestion(
                question="What is the purpose of this library?",
                expected_answer_keywords=[],
                expected_chunk_ids=[],
                expected_content_types=[],
                category="conceptual",
                difficulty="easy",
                notes="Replace with a real question about your indexed documentation"
            ),
            EvalQuestion(
                question="How do I get started with the basic usage?",
                expected_answer_keywords=[],
                expected_chunk_ids=[],
                expected_content_types=["tutorial"],
                category="tutorial",
                difficulty="easy",
                notes="Replace with a real getting-started question"
            ),
            EvalQuestion(
                question="What configuration options are available?",
                expected_answer_keywords=[],
                expected_chunk_ids=[],
                expected_content_types=["api_reference"],
                category="api",
                difficulty="medium",
                notes="Replace with a real configuration question"
            )
        ]
        
        return cls(questions=questions, metadata=metadata)
