import pytest
import os
import json

from rag.evaluation.test_set import EvalQuestion, TestSet


class TestTestSet:
    def test_eval_question_creation(self):
        q = EvalQuestion(
            question="Q?",
            expected_answer_keywords=["a", "b"],
            expected_chunk_ids=["c1"],
            expected_content_types=["api"],
            category="api",
            difficulty="hard",
            notes="note"
        )
        assert q.question == "Q?"
        assert q.expected_answer_keywords == ["a", "b"]
        assert q.expected_chunk_ids == ["c1"]
        assert q.expected_content_types == ["api"]
        assert q.category == "api"
        assert q.difficulty == "hard"
        assert q.notes == "note"

    def test_test_set_creation(self):
        q1 = EvalQuestion("1", [], [], [])
        q2 = EvalQuestion("2", [], [], [])
        q3 = EvalQuestion("3", [], [], [])
        
        ts = TestSet([q1, q2, q3])
        assert len(ts) == 3

    def test_test_set_save_and_load(self, tmp_path):
        q1 = EvalQuestion("1", ["k1"], ["c1"], ["api"], "api", "easy")
        ts1 = TestSet([q1], metadata={"name": "test"})
        
        file_path = tmp_path / "test_set.json"
        ts1.save(str(file_path))
        
        ts2 = TestSet.load(str(file_path))
        
        assert len(ts2) == 1
        assert ts2.metadata["name"] == "test"
        assert ts2.questions[0].question == "1"
        assert ts2.questions[0].expected_answer_keywords == ["k1"]

    def test_test_set_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            TestSet.load("does_not_exist_at_all.json")

    def test_test_set_get_by_category(self):
        q1 = EvalQuestion("1", [], [], [], category="api")
        q2 = EvalQuestion("2", [], [], [], category="tutorial")
        q3 = EvalQuestion("3", [], [], [], category="api")
        ts = TestSet([q1, q2, q3])
        
        api_qs = ts.get_by_category("api")
        assert len(api_qs) == 2
        assert api_qs[0].question == "1"
        assert api_qs[1].question == "3"

    def test_test_set_get_by_difficulty(self):
        q1 = EvalQuestion("1", [], [], [], difficulty="easy")
        q2 = EvalQuestion("2", [], [], [], difficulty="hard")
        q3 = EvalQuestion("3", [], [], [], difficulty="easy")
        ts = TestSet([q1, q2, q3])
        
        easy_qs = ts.get_by_difficulty("easy")
        assert len(easy_qs) == 2

    def test_create_sample(self):
        ts = TestSet.create_sample()
        assert len(ts) >= 3
        assert ts.metadata["name"] == "Sample RAG Evaluation Set"
        assert ts.questions[0].difficulty == "easy"

    def test_test_set_json_format(self, tmp_path):
        ts = TestSet.create_sample()
        file_path = tmp_path / "test_set.json"
        ts.save(str(file_path))
        
        with open(file_path, "r") as f:
            data = json.load(f)
            
        assert "metadata" in data
        assert "questions" in data
        assert isinstance(data["questions"], list)
