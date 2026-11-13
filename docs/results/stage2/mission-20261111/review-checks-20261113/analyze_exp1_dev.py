#!/usr/bin/env python3
"""Model-free analysis of the reviewer-check B1 predictions (frozen R7). Prints tables; writes nothing.
K = 4 summed option log-probability, chance 25%. Paired exact two-sided McNemar. p-values are unadjusted."""
import json, os, collections
from math import comb
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', '..'))
DIAG = os.path.join(ROOT, 'docs/results/stage2/diagnostics-20261113')
R7 = '33a5c24521dfb1b28b456b8344bfd0869b84fe56aa622191b8f6bfa71cc42aa1'

def load(path_no_ext):
    d = json.load(open(path_no_ext + '.r7.json'))
    assert d['checkpoint_sha256'] == R7
    src = {json.loads(l)['id']: json.loads(l) for l in open(path_no_ext + '.jsonl')}
    out = {}
    for x in d['conditions']['real']['items']:
        s, k = x['summed_option_logprob'], x['key']
        out[x['id']] = {'ok': x['summed_option_prediction'] == k, 'pred': x['summed_option_prediction'], 'key': k, 's': s,
                        'ex': x['gold_length_greedy_prefix_exact'], 'opts': src[x['id']]['options'],
                        'passage': src[x['id']]['passage']}
    return out

def mcnemar(a, b, ids, f='ok'):
    g = sum((not a[i][f]) and b[i][f] for i in ids); l = sum(a[i][f] and (not b[i][f]) for i in ids)
    n, k = g + l, min(g, l)
    return g, l, (min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0)

def row(label, a, b, ids):
    n = len(ids); ka, kb = sum(a[i]['ok'] for i in ids), sum(b[i]['ok'] for i in ids)
    both = sum(a[i]['ok'] and b[i]['ok'] for i in ids); nei = sum(not a[i]['ok'] and not b[i]['ok'] for i in ids)
    g, l, p = mcnemar(a, b, ids)
    eg, el, ep = mcnemar(a, b, ids, 'ex')
    print('%-34s n=%3d  K4 %3d -> %3d (%5.1f%% -> %5.1f%%, %+5.1f pts)  both %3d  gain %3d  loss %3d  neither %3d  p=%.3g | Exact %3d -> %3d (+%d/-%d p=%.3g)' % (
        label, n, ka, kb, 100 * ka / n, 100 * kb / n, 100 * (kb - ka) / n, both, g, l, nei, p,
        sum(a[i]['ex'] for i in ids), sum(b[i]['ex'] for i in ids), eg, el, ep))

meta = json.load(open(os.path.join(HERE, 'pairs-meta.json')))
by = collections.defaultdict(list)
for m in meta: by[m['set']].append(m)

print('=' * 10, 'Gold-only exposure swap on frozen R7. K = 4, chance 25%. Exact = greedy prefix, K = 8,192, chance ~0%.')
for tag, mset, title in (('e1-train', 'e1', 'Experiment 1 pairs, TRAINING wording (pipeline check; saved: 15->55/90, 34->10/67)'),
                         ('e1-dev', 'e1', 'Experiment 1 pairs, DEV wording, person-first (PRIMARY)'),
                         ('devrev', 'devrev', 'Real dev REVERSE items, dev wording (secondary)'),
                         ('devcmp', 'devcmp', 'Real dev COMPARE items, dev wording (secondary)')):
    a, b = load(os.path.join(HERE, tag + '-orig')), load(os.path.join(HERE, tag + '-swap'))
    print('---', title)
    for d in ('zero_to_hi100', 'hi100_to_zero'):
        row(d, a, b, [m['id'] for m in by[mset] if m['direction'] == d])
    # pooled in the Experiment 1 convention: "low-exposure gold" vs "high-exposure gold" on the same item
    lo, hi = {}, {}
    for m in by[mset]:
        i = m['id']; z2h = m['direction'] == 'zero_to_hi100'
        lo[i], hi[i] = (a[i], b[i]) if z2h else (b[i], a[i])
    row('pooled zero-gold -> hi100-gold', lo, hi, [m['id'] for m in by[mset]])
    if mset == 'e1':
        for attr in ('town', 'instrument'):
            row('  pooled, %s' % attr, lo, hi, [m['id'] for m in by[mset] if m['attr'] == attr])
        if tag == 'e1-dev':
            for pf in (True, False):
                row('  pooled, prefix=%s' % pf, lo, hi, [m['id'] for m in by[mset] if m['prefix'] == pf])
    # mean change of the gold option's summed log-probability and of the best distractor's
    dg = [hi[i]['s'][hi[i]['key']] - lo[i]['s'][lo[i]['key']] for i in lo]
    dd = [max(v for j, v in enumerate(hi[i]['s']) if j != hi[i]['key']) - max(v for j, v in enumerate(lo[i]['s']) if j != lo[i]['key']) for i in lo]
    print('   mean summed log-prob change, zero-gold -> hi100-gold: gold option %+.2f, best distractor %+.2f' % (sum(dg) / len(dg), sum(dd) / len(dd)))

print()
print('=' * 10, 'Wording effect with names held fixed (the 157 Experiment 1 items, original names): training vs dev wording')
a, b = load(os.path.join(HERE, 'e1-train-orig')), load(os.path.join(HERE, 'e1-dev-orig'))
for d in ('zero_to_hi100', 'hi100_to_zero'):
    row('gold stratum %s' % d.split('_to_')[0], a, b, [m['id'] for m in by['e1'] if m['direction'] == d])

print()
print('=' * 10, 'All-four rename (no new inference; saved diagnostics-20261113 predictions)')
d1, d1t = load(os.path.join(DIAG, 'd1-rev-pf')), load(os.path.join(DIAG, 'd1t-rev-pf'))
ids = sorted(d1); t = {i: d1t[i.replace('d1-', 'd1t-')] for i in ids}
row('D1 rev-pf: dev-pool -> train-pool', d1, t, ids)
rot0, tr = load(os.path.join(DIAG, 'd2-rot0')), load(os.path.join(DIAG, 'd2-train'))
for cat in ('reverse', 'compare'):
    ids = [i for i in tr if ('-%s-' % cat) in i]
    row('D2 %s: dev-pool -> train-pool' % cat, rot0, tr, ids)

print()
print('=' * 10, 'Which option does R7 pick, by exposure stratum of the option name? (saved D1 rev-pf and D2 rot0 predictions)')
import sys
sys.path.insert(0, DIAG)
from bpe_port import encode
tm = {t['id']: t for t in json.load(open(os.path.join(ROOT, 'docs/results/stage2/mission-20261111/pilot4/audit_raw.json')))['vocab_tokens']}
def stratum(name):
    ids = encode(' ' + name)
    p3, r7 = min(tm[i]['p3_all'] for i in ids), min(tm[i]['r7_ret_all'] for i in ids)
    return 'hi100' if p3 >= 100 and r7 >= 100 else 'zero' if p3 == 0 and r7 == 0 else 'mid'
for label, x, ids in (('D1 rev-pf (training wording, n=200)', d1, sorted(d1)),
                      ('dev reverse (dev wording, n=100)', rot0, [i for i in rot0 if '-reverse-' in i]),
                      ('dev compare (dev wording, n=100)', rot0, [i for i in rot0 if '-compare-' in i])):
    c = collections.Counter()
    for i in ids:
        for j, o in enumerate(x[i]['opts']):
            c[(stratum(o), j == x[i]['key'], j == x[i]['pred'])] += 1
    print('---', label, '(each cell: picked / option slots)')
    for st in ('zero', 'mid', 'hi100'):
        gp, gn = c[(st, True, True)], c[(st, True, True)] + c[(st, True, False)]
        dp, dn = c[(st, False, True)], c[(st, False, True)] + c[(st, False, False)]
        print('   %-6s gold slots %3d/%3d = %5.1f%%   distractor slots %3d/%3d = %5.1f%%   all slots %3d/%3d = %5.1f%%' % (
            st, gp, gn, 100 * gp / max(1, gn), dp, dn, 100 * dp / max(1, dn), gp + dp, gn + dn, 100 * (gp + dp) / max(1, gn + dn)))
