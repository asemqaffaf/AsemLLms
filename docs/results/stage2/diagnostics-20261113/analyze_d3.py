#!/usr/bin/env python3
"""D3: R7 fit on its own training distribution, by task kind. Reads d3-training-fit.r7.json; prints tables; writes nothing.
Metric: teacher-forced exact answer (every answer token is the full-vocabulary argmax). K = 8192 per token, chance ~0%."""
import json, os, re, collections, sys
from bpe_port import ntok
D = os.path.dirname(os.path.abspath(__file__))
name = sys.argv[1] if len(sys.argv) > 1 else 'd3-training-fit.r7.json'
its = json.load(open(os.path.join(D, name)))['items']
STOP = set('''a an the is in of to for by as at on and it was has have who whom whose which what where how many did does do person
lives lived live house town grew up still there years works work job earns living trade pet owns own takes care keeps home animal
favorite color likes like best loves most always picks prefers prefer colors plays play practices practice every day learning
instrument performs old age just turned reached scored score points finished game earned earn tall stands height cm taught teach
gave lessons hired hire called call got from phoned thanked thank visited visit went see helped help give given oldest youngest
tallest shortest fewest highest lowest four first then next after that finally started ended stopped go place last right before
now also today really already usually these days moment happens all accounts seems we hear once again usual according note far
anyone knows everyone agrees rumor says record report tell me please goes with had been be'''.split())
def sents(p): return [s.strip() for s in re.split(r'(?<=\.)\s+', p) if s.strip()]
def word_in(w, s): return re.search(r'(?<![A-Za-z0-9])%s(?![A-Za-z0-9])' % re.escape(w), s) is not None
def classify(x):
    k = x['kind']; q = x['question']; S = sents(x['passage']); x['nsent'] = len(S)
    if k == 'attr': x['task'] = 'reverse' if re.match(r'(Who|Whose|Which person)\b', q) else 'extract'
    else: x['task'] = {'num': 'numeric extract', 'rel': 'relation', 'cmp': 'compare', 'tmp': 'temporal'}[k]
    x['dir'] = 'n/a'
    if k in ('attr', 'num', 'rel'):
        gs = [s for s in S if word_in(x['answer'], s)]
        cands = []
        for s in gs:
            for w in re.findall(r'[A-Za-z0-9]+', q):
                if w.lower() not in STOP and w != x['answer'] and word_in(w, s): cands.append((s, w))
        if cands:
            s, w = cands[0]
            x['dir'] = 'answer AFTER queried entity' if s.find(x['answer']) > s.find(w) else 'answer BEFORE queried entity'
        else: x['dir'] = 'unresolved'
    if k == 'cmp':
        nums = re.findall(r'\b\d+\b', x['passage']); shapes = set(ntok(' ' + n) for n in nums)
        x['shape'] = 'same token count' if len(shapes) == 1 else 'mixed token counts'
        x['npeople'] = len(nums)
    if k == 'tmp':
        ql = q.lower()
        x['sub'] = 'first' if 'first' in ql else 'last' if 'last' in ql else 'after' if 'after' in ql else 'before' if 'before' in ql else '?'
    return x
its = [classify(x) for x in its]
def row(label, xs):
    n = len(xs)
    if not n: return
    print('  %-52s exact %3d/%3d = %5.1f%%   first token %3d   mean first-token NLL %.2f' % (
        label, sum(x['prefix_exact'] for x in xs), n, 100 * sum(x['prefix_exact'] for x in xs) / n,
        sum(x['first_token_correct'] for x in xs), sum(x['first_nll'] for x in xs) / n))
print('All %d questions of the 256 traced training sequences (dense context, as trained)' % len(its))
row('ALL', its); row('first question of each sequence', [x for x in its if x['ordinal'] == 0]); row('later questions', [x for x in its if x['ordinal'] > 0])
print('By task')
for t in ('extract', 'reverse', 'numeric extract', 'relation', 'compare', 'temporal'): row(t, [x for x in its if x['task'] == t])
print('By task and position of the answer relative to the queried entity (same sentence)')
for t in ('extract', 'reverse', 'numeric extract', 'relation'):
    for d in ('answer AFTER queried entity', 'answer BEFORE queried entity', 'unresolved'):
        row('%s | %s' % (t, d), [x for x in its if x['task'] == t and x['dir'] == d])
print('Answer BEFORE queried entity, by number of fact sentences in the passage')
for t in ('reverse', 'relation', 'extract'):
    for n in sorted(set(x['nsent'] for x in its)):
        row('%s | before | %d sentences' % (t, n), [x for x in its if x['task'] == t and x['dir'].startswith('answer BEFORE') and x['nsent'] == n])
print('Compare, by number of people and by number token shape')
for n in (2, 3, 4): row('compare | %d people' % n, [x for x in its if x['task'] == 'compare' and x['npeople'] == n])
for s in ('same token count', 'mixed token counts'): row('compare | %s' % s, [x for x in its if x['task'] == 'compare' and x['shape'] == s])
for s in ('same token count', 'mixed token counts'): row('compare | 4 people | %s' % s, [x for x in its if x['task'] == 'compare' and x['shape'] == s and x['npeople'] == 4])
print('Temporal, by question type and number of stops')
for s in ('first', 'last', 'after', 'before', '?'): row('temporal | %s' % s, [x for x in its if x['task'] == 'temporal' and x['sub'] == s])
for n in sorted(set(x['nsent'] for x in its if x['task'] == 'temporal')): row('temporal | %d sentences' % n, [x for x in its if x['task'] == 'temporal' and x['nsent'] == n])
