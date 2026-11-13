#!/usr/bin/env python3
"""Pilot 2 analysis from saved predictions only. Prints tables; writes nothing.
Ranking tables: K = 4 options, chance 25%, summed option log-probability (not length-normalised).
Exact-answer tables: teacher-forced full-vocabulary argmax on every answer token, K = 8192 per token, chance ~0%."""
import json, os, collections
from math import comb
P = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(P, '..', '..', 'diagnostics-20261113'); M = os.path.join(P, '..')
TAG = {'R7': 'r7', 'Control': 'p2con', 'Treatment': 'p2tre'}
def cell(name, tag):
    return {x['id']: x['summed_option_prediction'] == x['key'] for x in json.load(open(os.path.join(D, '%s.%s.json' % (name, tag))))['conditions']['real']['items']}
def mcn(a, b):
    g = sum(a[i] and not b[i] for i in a); l = sum(b[i] and not a[i] for i in a); n = g + l; k = min(g, l)
    return g, l, (min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0)
print('GATE CELLS (frozen D1 items, n = 200 each, K = 4, chance 25%)')
GATE = {'d1-rev-pf': ('person-first reverse', 120), 'd1-rev-vf': ('value-first reverse', 142), 'd1-ext-pf': ('person-first extraction', 169)}
ok = True
for name in ('d1-rev-pf', 'd1-rev-vf', 'd1-ext-pf', 'd1-ext-vf', 'd4-rev-pf-far', 'd4-rev-vf-far', 'd4-ext-pf-far', 'd4-ext-vf-far'):
    c = {k: cell(name, t) for k, t in TAG.items()}
    g1, l1, p1 = mcn(c['Treatment'], c['R7']); g2, l2, p2 = mcn(c['Treatment'], c['Control']); g3, l3, p3 = mcn(c['Control'], c['R7'])
    gate = ''
    if name in GATE:
        met = sum(c['Treatment'].values()) >= GATE[name][1]; ok = ok and met
        gate = '  GATE >= %d: %s' % (GATE[name][1], 'met' if met else 'MISSED')
    print('%-14s R7 %3d  Control %3d  Treatment %3d | T vs R7 +%d/-%d p=%.3g | T vs C +%d/-%d p=%.3g | C vs R7 +%d/-%d p=%.3g%s' % (
        name, sum(c['R7'].values()), sum(c['Control'].values()), sum(c['Treatment'].values()), g1, l1, p1, g2, l2, p2, g3, l3, p3, gate))
print('PRIMARY GATE:', 'PASS' if ok else 'FAIL')
print()
print('SELECTION SET (frozen mission selection set, n = 500, 100 per category, K = 4, chance 25%; +/-4.4 points at n = 500)')
sel = {}
for k, f in (('R7', os.path.join(M, 'r7-selection.json')), ('Control', os.path.join(P, 'control-selection.json')), ('Treatment', os.path.join(P, 'treatment-selection.json'))):
    its = json.load(open(f))['conditions']['real']['items']
    sel[k] = {x['id']: (x['category'], x['summed_option_prediction'] == x['key'], x['gold_length_greedy_prefix_exact']) for x in its}
for cat in ('extract', 'relation', 'reverse', 'compare', 'temporal', 'ALL'):
    ids = [i for i in sel['R7'] if cat == 'ALL' or sel['R7'][i][0] == cat]
    a = {k: {i: sel[k][i][1] for i in ids} for k in sel}
    g1, l1, p1 = mcn(a['Treatment'], a['R7']); g2, l2, p2 = mcn(a['Control'], a['R7'])
    print('%-9s n=%3d  R7 %3d  Control %3d (+%d/-%d p=%.3g)  Treatment %3d (+%d/-%d p=%.3g) | exact prefix R7 %d C %d T %d' % (
        cat, len(ids), sum(a['R7'].values()), sum(a['Control'].values()), g2, l2, p2, sum(a['Treatment'].values()), g1, l1, p1,
        sum(sel['R7'][i][2] for i in ids), sum(sel['Control'][i][2] for i in ids), sum(sel['Treatment'][i][2] for i in ids)))
print()
print('FIT ON FOUR-PERSON ANSWER-BEFORE QUESTIONS, training wording, training-pool entities (exact answer, K = 8192/token)')
for k in ('r7', 'control', 'treatment'):
    its = json.load(open(os.path.join(P, 'answer-before-fit.%s.json' % k)))['items']
    t = collections.defaultdict(lambda: [0, 0])
    for x in its:
        t[x['set'] + ' all'][0] += x['prefix_exact']; t[x['set'] + ' all'][1] += 1
        if x['set'] == 'fresh': t['fresh ' + x['task']][0] += x['prefix_exact']; t['fresh ' + x['task']][1] += 1
    print('%-10s' % k, '  '.join('%s %d/%d (%.1f%%)' % (n, v[0], v[1], 100 * v[0] / v[1]) for n, v in sorted(t.items())))
print()
print('TRAINING LOSS (log.jsonl, mean answer-token loss per 20 steps)')
for arm in ('control', 'treatment'):
    rows = [json.loads(l) for l in open(os.path.join(P, '..', '..', '..', '..', '..', 'data/runs/retrieval/p2-answer-before', arm, 'log.jsonl'))]
    tr = [r for r in rows if r.get('event') == 'train']
    print('%-10s' % arm, ' '.join('%d:%.2f' % (r['step'], r['train_loss']) for r in tr if r['step'] in (20, 100, 200, 300, 400, 500)),
          '| mean of last 5 logs %.3f' % (sum(r['train_loss'] for r in tr[-5:]) / 5))
