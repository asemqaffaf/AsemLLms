#!/usr/bin/env python3
"""Model-free analysis of the saved diagnostic predictions (*.r7.json). Prints tables; writes nothing."""
import json, os, collections
from math import comb
D = os.path.dirname(os.path.abspath(__file__))

def load(name):
    d = json.load(open(os.path.join(D, name + '.r7.json')))
    assert d['checkpoint_sha256'] == '33a5c24521dfb1b28b456b8344bfd0869b84fe56aa622191b8f6bfa71cc42aa1'
    src = {json.loads(l)['id']: json.loads(l) for l in open(os.path.join(D, name + '.jsonl'))}
    out = {}
    for x in d['conditions']['real']['items']:
        s = x['summed_option_logprob']; k = x['key']
        x['ok'] = x['summed_option_prediction'] == k
        x['margin'] = s[k] - max(v for j, v in enumerate(s) if j != k)
        x['gold_lp'] = s[k]
        x['opts'] = src[x['id']]['options']; x['sub'] = src[x['id']]['subtype']
        x['pred_name'] = x['opts'][x['summed_option_prediction']]
        out[x['id']] = x
    return out

def mcnemar(a, b, ids):
    g = sum(a[i]['ok'] and not b[i]['ok'] for i in ids); l = sum(b[i]['ok'] and not a[i]['ok'] for i in ids)
    n = g + l; k = min(g, l)
    p = min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0
    return g, l, p

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** .5
    return 100 * (c - h) / d, 100 * (c + h) / d

def acc(x, ids): return sum(x[i]['ok'] for i in ids)
def med(v): v = sorted(v); return v[len(v) // 2]
def line(label, x, ids):
    k, n = acc(x, ids), len(ids); lo, hi = wilson(k, n)
    return '%-34s %3d/%3d = %5.1f%% [%4.1f, %4.1f]  prefix %3d  median margin %+6.2f  mean gold lp %6.2f' % (
        label, k, n, 100 * k / n, lo, hi, sum(x[i]['gold_length_greedy_prefix_exact'] for i in ids),
        med([x[i]['margin'] for i in ids]), sum(x[i]['gold_lp'] for i in ids) / n)

print('=' * 20, 'D1 reverse binding: person-first (pf) vs value-first (vf), identical facts/question/options')
for tag, what in (('d1', 'dev-pool real names (primary)'), ('d1t', 'training-pool names (secondary replicate)')):
    print('---', what)
    for q, qn in (('rev', 'REVERSE value->person'), ('ext', 'EXTRACT person->value (mirror)')):
        pf, vf = load('%s-%s-pf' % (tag, q)), load('%s-%s-vf' % (tag, q))
        ids = list(pf)
        groups = [('all', ids), ('town', [i for i in ids if vf[i]['sub'].startswith('town')]),
                  ('  town vf-A "In V lives P"', [i for i in ids if vf[i]['sub'] == 'town-vf-A']),
                  ('  town vf-B "V is where P lives"', [i for i in ids if vf[i]['sub'] == 'town-vf-B']),
                  ('instrument', [i for i in ids if vf[i]['sub'].startswith('instrument')])]
        for gname, g in groups:
            both = sum(pf[i]['ok'] and vf[i]['ok'] for i in g); neither = sum(not pf[i]['ok'] and not vf[i]['ok'] for i in g)
            a, b, p = mcnemar(vf, pf, g)
            print('%s | %-32s n=%3d  pf %3d (%4.1f%%)  vf %3d (%4.1f%%)  both %3d  vf-only %3d  pf-only %3d  neither %3d  McNemar p=%.4g' % (
                qn[:7], gname, len(g), acc(pf, g), 100 * acc(pf, g) / len(g), acc(vf, g), 100 * acc(vf, g) / len(g), both, a, b, neither, p))
        print('   pf:', line('', pf, ids)); print('   vf:', line('', vf, ids))
    # crossover: per item (rev_vf - rev_pf) - (ext_vf - ext_pf)
    R = {c: load('%s-rev-%s' % (tag, c)) for c in ('pf', 'vf')}; E = {c: load('%s-ext-%s' % (tag, c)) for c in ('pf', 'vf')}
    dd = [(R['vf'][i]['ok'] - R['pf'][i]['ok']) - (E['vf'][i]['ok'] - E['pf'][i]['ok']) for i in R['pf']]
    m = sum(dd) / len(dd); sd = (sum((v - m) ** 2 for v in dd) / (len(dd) - 1)) ** .5
    print('   interaction (rev vf-pf) - (ext vf-pf): %+.1f points, 95%% CI +/- %.1f (normal approx, n=%d)' % (100 * m, 196 * sd / len(dd) ** .5, len(dd)))

print()
print('=' * 20, 'D2 answer-string bias on the 100 reverse + 100 compare public dev items (dev wording)')
rot = [load('d2-rot%d' % k) for k in range(4)]
for cat in ('reverse', 'compare'):
    ids = [i for i in rot[0] if rot[0][i]['category'] == cat]
    print('---', cat, 'n =', len(ids))
    for k in range(4): print('  ', line('rotation %d%s' % (k, ' (original item)' if k == 0 else ''), rot[k], ids))
    tot = sum(acc(r, ids) for r in rot); lo, hi = wilson(tot, 4 * len(ids))
    print('   pooled over 4 rotations: %d/%d = %.1f%% (items are not independent across rotations)' % (tot, 4 * len(ids), 25 * tot / len(ids)))
    ncorr = collections.Counter(sum(r[i]['ok'] for r in rot) for i in ids)
    print('   items by number of rotations answered correctly (0..4):', [ncorr[k] for k in range(5)],
          '| expected if choice ignores the passage and follows the name: all items at exactly 1')
    names = collections.Counter(len(set(r[i]['pred_name'] for r in rot)) for i in ids)
    print('   items by number of distinct names chosen across the 4 rotations (1 = same name every time):', [names[k] for k in range(1, 5)])
    modal = sum(max(collections.Counter(r[i]['pred_name'] for r in rot).values()) for i in ids)
    print('   predictions equal to the item\'s most-chosen name: %d/%d = %.1f%% (25%% if names were irrelevant, 100%% if only the name mattered)' % (modal, 4 * len(ids), 25 * modal / len(ids)))
    # does the chosen name follow the role? among items with the same name in all 4 rotations -> exactly 1 correct
    same = [i for i in ids if len(set(r[i]['pred_name'] for r in rot)) == 1]
    track = [i for i in ids if sum(r[i]['ok'] for r in rot) == 4]
    print('   fixed-name items: %d (%.0f%%); fully role-tracking items (4/4 correct): %d (%.0f%%)' % (len(same), 100 * len(same) / len(ids), len(track), 100 * len(track) / len(ids)))
    for cond in ('easy', 'hard', 'train'):
        x = load('d2-' + cond); sub = [i for i in ids if i in x]
        a, b, p = mcnemar(x, rot[0], sub)
        tl = collections.Counter(n for i in sub for n in x[i]['option_token_lengths'])
        print('  ', line('all four names ' + cond, x, sub))
        print('       vs original on the same %d items: original %d, %s %d; gains %d, losses %d, McNemar p=%.4g; option token lengths %s' % (
            len(sub), acc(rot[0], sub), cond, acc(x, sub), a, b, p, dict(sorted(tl.items()))))
        spread = lambda y: sum(max(y[i]['summed_option_logprob']) - min(y[i]['summed_option_logprob']) for i in sub) / len(sub)
        print('       mean within-item score spread (max-min over 4 options): original %.2f, %s %.2f' % (spread(rot[0]), cond, spread(x)))
