import sys
from collections import Counter
from urllib.parse import urlparse

sys.path.append('.')
from pipeline.ingestion import ingest
from pipeline.canonicalization import canonicalize_batch
from pipeline.version_detection import apply_version_detection
from pipeline.generator_detection import detect_generator
from pipeline.structural_grouping import discover_groups, finalize_templates
from pipeline.coverage import compute_coverage

ingested = ingest('../phase1/phase1_output.json')
internal = canonicalize_batch(ingested['parsed_internal'])
internal = apply_version_detection(internal)
generator, _ = detect_generator(ingested['phase1'], internal)
groups = discover_groups(internal, generator_hint=generator)
templates = finalize_templates(groups, internal)
coverage, uncovered, _ = compute_coverage(internal, templates)

paths = [urlparse(u.url).path for u in uncovered]

dirs = Counter()
for p in paths:
    d = '/'.join(p.split('/')[:-1])
    if not d:
        d = '/'
    dirs[d] += 1

print('Total uncovered:', coverage.uncovered_urls)
print('\n--- Top Uncovered Directories ---')
for d, count in dirs.most_common(30):
    print(f'{d}: {count}')

prefix2 = Counter()
for p in paths:
    parts = [x for x in p.split('/') if x]
    if len(parts) >= 2:
        prefix2['/' + '/'.join(parts[:2])] += 1
    elif len(parts) == 1:
        prefix2['/' + parts[0]] += 1
    else:
        prefix2['ROOT'] += 1

print('\n--- Top Uncovered Prefixes (2 segments) ---')
for p, count in prefix2.most_common(30):
    print(f'{p}: {count}')
