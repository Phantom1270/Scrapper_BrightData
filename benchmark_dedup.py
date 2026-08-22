import time
from rag.pipeline.normalizer import UniversalNormalizer
from rag.pipeline.deduplicator import DocumentDeduplicator

normalizer = UniversalNormalizer()
docs = normalizer.normalize_file('phase3_output (2).json')
valid = [d for d in docs if not d.error and d.content_blocks]
print(f'Input: {len(valid)} documents')

t = time.time()
dedup = DocumentDeduplicator()
result = dedup.deduplicate(valid)
elapsed = time.time() - t

stats = dedup.get_stats()
print(f"Output: {stats['output_count']} documents")
print(f"Exact removed: {stats['exact_removed']}")
print(f"Near removed: {stats['near_removed']}")
print(f"Time: {elapsed:.2f}s")
