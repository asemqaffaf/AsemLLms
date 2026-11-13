#!/usr/bin/env python3
"""Step 0 analysis of saved predictions (D4 adjacency vs order, D5 temporal place strings). Prints tables; writes nothing.
All tables: K = 4 options, chance 25%, option ranking by summed log P(option tokens | prompt), not length-normalised."""
import json, os, collections, sys
from math import comb
D = os.path.dirname(os.path.abspath(__file__)); TAG = sys.argv[1] if len(sys.argv) > 1 else 'r7'
def load(name):
    d = json.load(open(os.path.join(D, '%s.%s.json' % (name, TAG))))
    src = {json.loads(l)['id']: json.loads(l) for l in open(os.path.join(D, name + '.jsonl'))}
    out = {}
    for x in d['conditions']['real']['items']:
        s = x['summed_option_logprob']; k = x['key']
        x['ok'] = x['summed_option_prediction'] == k; x['gold_lp'] = s[k]
        x['margin'] = s[k] - max(v for j, v in enumerate(s) if j != k)
        x['opts'] = src[x['id']]['options']; x['pred_name'] = x['opts'][x['summed_option_prediction']]; x['sub'] = src[x['id']]['subtype']
        out[x['id']] = x
    return out
def mcn(a, b, ids):
    g = sum(a[i]['ok'] and not b[i]['ok'] for i in ids); l = sum(b[i]['ok'] and not a[i]['ok'] for i in ids); n = g + l; k = min(g, l)
    return g, l, (min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0)
def acc(x, ids): return sum(x[i]['ok'] for i in ids)
def med(v): v = sorted(v); return v[len(v) // 2]

print('=' * 10, 'D4 adjacency vs order (200 paired fact sets, training wording, dev-pool entities; K=4, chance 25%)')
for q, qn in (('rev', 'REVERSE value->person'), ('ext', 'EXTRACT person->value')):
    cell = {('pf', 'near'): load('d1-%s-pf' % q), ('vf', 'near'): load('d1-%s-vf' % q),
            ('pf', 'far'): load('d4-%s-pf-far' % q), ('vf', 'far'): load('d4-%s-vf-far' % q)}
    ids = list(cell[('pf', 'near')])
    for o in ('pf', 'vf'):
        a, b = cell[(o, 'near')], cell[(o, 'far')]; g, l, p = mcn(b, a, ids)
        print('%-22s %s (%s)  near %3d/200 (%4.1f%%)  far %3d/200 (%4.1f%%)  far-only %3d  near-only %3d  McNemar p=%.3g' % (
            qn, o, 'answer BEFORE queried entity' if (q == 'rev') == (o == 'pf') else 'answer AFTER queried entity',
            acc(a, ids), acc(a, ids) / 2, acc(b, ids), acc(b, ids) / 2, g, l, p))
    g, l, p = mcn(cell[('vf', 'far')], cell[('pf', 'near')], ids)
    print('   %s: vf-far vs pf-near (order differs, the far cell has MORE distance): %d vs %d, gains %d losses %d, p=%.3g' % (
        qn, acc(cell[('vf', 'far')], ids), acc(cell[('pf', 'near')], ids), g, l, p))
    for sub in ('town', 'instrument'):
        s = [i for i in ids if cell[('pf', 'near')][i]['sub'].startswith(sub)]
        print('   %-10s n=%d  pf near %d far %d | vf near %d far %d' % (sub, len(s), acc(cell[('pf', 'near')], s), acc(cell[('pf', 'far')], s),
                                                                        acc(cell[('vf', 'near')], s), acc(cell[('vf', 'far')], s)))

print(); print('=' * 10, 'D5 temporal place strings (100 temporal public dev items, dev wording; K=4, chance 25%)')
rot = [load('d5-rot%d' % k) for k in range(4)]; ids = list(rot[0])
for k in range(4):
    print('rotation %d%-16s %2d/100  median margin %+6.2f  mean gold lp %6.2f' % (k, ' (original item)' if k == 0 else '', acc(rot[k], ids),
          med([rot[k][i]['margin'] for i in ids]), sum(rot[k][i]['gold_lp'] for i in ids) / 100))
print('pooled %d/400 = %.1f%%' % (sum(acc(r, ids) for r in rot), sum(acc(r, ids) for r in rot) / 4))
nc = collections.Counter(sum(r[i]['ok'] for r in rot) for i in ids)
print('items by rotations correct (0..4):', [nc[k] for k in range(5)], '| pure string-following puts every item at exactly 1')
nn = collections.Counter(len(set(r[i]['pred_name'] for r in rot)) for i in ids)
print('items by distinct places chosen over 4 rotations (1 = same place every time):', [nn[k] for k in range(1, 5)])
modal = sum(max(collections.Counter(r[i]['pred_name'] for r in rot).values()) for i in ids)
print('predictions equal to the item\'s most-chosen place: %d/400 = %.1f%%' % (modal, modal / 4))
pc = collections.Counter(r[i]['pred_name'] for r in rot for i in ids); gc = collections.Counter(r[i]['opts'][r[i]['key']] for r in rot for i in ids)
print('place: chosen / gold over 400:', {p: (pc[p], gc[p]) for p in sorted(gc)})
for cond in ('train', 'hard', 'onetok'):
    x = load('d5-' + cond); g, l, p = mcn(x, rot[0], ids)
    tl = collections.Counter(n for i in ids for n in x[i]['option_token_lengths'])
    print('all four places %-7s %2d/100 vs original %2d: gains %d losses %d p=%.3g | median margin %+6.2f  mean gold lp %6.2f | token lengths %s' % (
        cond, acc(x, ids), acc(rot[0], ids), g, l, p, med([x[i]['margin'] for i in ids]), sum(x[i]['gold_lp'] for i in ids) / 100, dict(sorted(tl.items()))))
    by = collections.defaultdict(lambda: [0, 0])
    for i in ids: by[x[i]['subtype']][0] += x[i]['ok']; by[x[i]['subtype']][1] += 1
    b0 = collections.defaultdict(lambda: [0, 0])
    for i in ids: b0[rot[0][i]['subtype']][0] += rot[0][i]['ok']; b0[rot[0][i]['subtype']][1] += 1
    print('     by question type (%s | original):' % cond, {k: ('%d/%d' % tuple(by[k]), '%d/%d' % tuple(b0[k])) for k in sorted(by)})
