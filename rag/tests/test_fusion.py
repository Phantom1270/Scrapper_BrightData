import pytest
from rag.retrieval.fusion import ReciprocalRankFusion
from rag.models.retrieval import RetrievalResult

@pytest.fixture
def vector_results():
    return [("c1", 0.95), ("c2", 0.80), ("c3", 0.60), ("c4", 0.40)]

@pytest.fixture
def bm25_results():
    return [("c2", 3.5), ("c4", 2.1), ("c1", 1.8), ("c5", 1.0)]

class TestReciprocalRankFusion:
    def test_fuse_two_lists_basic(self, vector_results, bm25_results):
        fusion = ReciprocalRankFusion(k=60)
        fused = fusion.fuse([vector_results, bm25_results])
        
        assert isinstance(fused, list)
        assert len(fused) > 0
        assert isinstance(fused[0], tuple)
        assert isinstance(fused[0][0], str)
        assert isinstance(fused[0][1], float)
        
        # Verify sorted descending
        scores = [f[1] for f in fused]
        assert scores == sorted(scores, reverse=True)

    def test_fuse_item_in_both_lists_gets_higher_score(self, vector_results, bm25_results):
        fusion = ReciprocalRankFusion(k=60)
        fused = fusion.fuse([vector_results, bm25_results])
        
        fused_dict = dict(fused)
        
        # "c2" is 2nd in vector, 1st in bm25
        # "c1" is 1st in vector, 3rd in bm25
        # They should both be very high
        assert fused_dict["c2"] > fused_dict["c3"]
        assert fused_dict["c2"] > fused_dict["c5"]
        assert fused_dict["c1"] > fused_dict["c3"]

    def test_fuse_item_only_in_one_list(self, vector_results, bm25_results):
        fusion = ReciprocalRankFusion(k=60)
        fused = fusion.fuse([vector_results, bm25_results])
        
        fused_dict = dict(fused)
        assert "c3" in fused_dict
        assert "c5" in fused_dict

    def test_fuse_with_weights(self, vector_results, bm25_results):
        fusion = ReciprocalRankFusion(k=60)
        fused_eq = dict(fusion.fuse([vector_results, bm25_results], weights=[1.0, 1.0]))
        fused_vec_heavy = dict(fusion.fuse([vector_results, bm25_results], weights=[2.0, 0.5]))
        
        # "c3" is only in vector, so with 2.0 vector weight, its score should be higher
        assert fused_vec_heavy["c3"] > fused_eq["c3"]
        # "c5" is only in bm25, so with 0.5 bm25 weight, its score should be lower
        assert fused_vec_heavy["c5"] < fused_eq["c5"]

    def test_fuse_empty_lists(self):
        fusion = ReciprocalRankFusion()
        assert fusion.fuse([]) == []
        assert fusion.fuse([[], []]) == []

    def test_fuse_one_empty_one_populated(self, vector_results):
        fusion = ReciprocalRankFusion(k=60)
        fused = fusion.fuse([vector_results, []])
        
        assert len(fused) == len(vector_results)
        
        fused_ids = [f[0] for f in fused]
        vec_ids = [v[0] for v in vector_results]
        assert fused_ids == vec_ids

    def test_fuse_weight_length_mismatch_raises(self, vector_results):
        fusion = ReciprocalRankFusion()
        with pytest.raises(ValueError, match="Length of weights must match length of ranked_lists"):
            fusion.fuse([vector_results, vector_results], weights=[1.0])

    def test_fuse_single_list(self, vector_results):
        fusion = ReciprocalRankFusion(k=60)
        fused = fusion.fuse([vector_results])
        assert len(fused) == len(vector_results)

    def test_fuse_preserves_all_unique_items(self, vector_results, bm25_results):
        fusion = ReciprocalRankFusion()
        fused = fusion.fuse([vector_results, bm25_results])
        
        fused_ids = {f[0] for f in fused}
        all_ids = {v[0] for v in vector_results}.union({b[0] for b in bm25_results})
        assert fused_ids == all_ids

    def test_fuse_vector_and_bm25_convenience(self):
        v1 = RetrievalResult(chunk_id="c1", content="", url="", heading_path=[], content_type="", score=0.9, source="vector")
        v2 = RetrievalResult(chunk_id="c2", content="", url="", heading_path=[], content_type="", score=0.8, source="vector")
        
        b1 = RetrievalResult(chunk_id="c2", content="", url="", heading_path=[], content_type="", score=2.5, source="bm25")
        b2 = RetrievalResult(chunk_id="c3", content="", url="", heading_path=[], content_type="", score=1.5, source="bm25")
        
        fusion = ReciprocalRankFusion(k=60)
        fused = fusion.fuse_vector_and_bm25([v1, v2], [b1, b2])
        
        assert len(fused) == 3
        fused_ids = {f[0] for f in fused}
        assert fused_ids == {"c1", "c2", "c3"}

    def test_fuse_deterministic(self, vector_results, bm25_results):
        fusion = ReciprocalRankFusion(k=60)
        fused1 = fusion.fuse([vector_results, bm25_results])
        fused2 = fusion.fuse([vector_results, bm25_results])
        assert fused1 == fused2
