#!/usr/bin/env python3
"""Analyze preflight separate-sentence filler test alongside D1 near and D4 in-sentence filler.

All tables use frozen R7 (SHA-256 33a5c245...), K = 4 options, chance 25%, summed option log-prob.
"""
import json, os, sys
from math import comb

P3 = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(P3, '..', '..', '..', '..', '..'))
DIAG = os.path.join(ROOT, 'docs', 'results', 'stage2', 'diagnostics-20261113')

def load(path):
    d = json.load(open(path))
    out = {}
    for x in d['conditions']['real']['items']:
        out[x['id']] = (x['summed_option_prediction'] == x['key'], x['gold_length_greedy_prefix_exact'])
    return out

def mcn(a, b):
    ids = list(a)
    g = sum(a[i][0] and not b[i][0] for i in ids)
    l = sum(b[i][0] and not a[i][0] for i in ids)
    n = g + l
    k = min(g, l)
    p = min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / (2 ** n)) if n else 1.0
    return g, l, p

def acc(x):
    return sum(v[0] for v in x.values())

def pfx(x):
    return sum(v[1] for v in x.values())

def main():
    print('R7 frozen (SHA 33a5c245...), n = 200 paired per cell, K = 4 (chance 25%), summed option log-prob')
    print()
    for q, qlabel in (('rev', 'REVERSE (value -> person)'), ('ext', 'EXTRACT (person -> value)')):
        print(f'=== {qlabel} ===')
        near_pf = load(os.path.join(DIAG, f'd1-{q}-pf.r7.json'))
        near_vf = load(os.path.join(DIAG, f'd1-{q}-vf.r7.json'))
        ins_pf = load(os.path.join(DIAG, f'd4-{q}-pf-far.r7.json'))
        ins_vf = load(os.path.join(DIAG, f'd4-{q}-vf-far.r7.json'))
        s1_pf = load(os.path.join(P3, f'preflight-s1-pf-{q}.r7.json'))
        s1_vf = load(os.path.join(P3, f'preflight-s1-vf-{q}.r7.json'))
        s2_pf = load(os.path.join(P3, f'preflight-s2-pf-{q}.r7.json'))
        s2_vf = load(os.path.join(P3, f'preflight-s2-vf-{q}.r7.json'))
        s3_pf = load(os.path.join(P3, f'preflight-s3-pf-{q}.r7.json'))
        s3_vf = load(os.path.join(P3, f'preflight-s3-vf-{q}.r7.json'))

        rows = [
            ('D1 baseline (no filler)', 'pf', near_pf, 'vf', near_vf, near_pf, near_vf),
            ('D4 in-sentence filler', 'pf', ins_pf, 'vf', ins_vf, near_pf, near_vf),
            ('Preflight s3 (D1 + post-fact filler sent)', 'pf', s3_pf, 'vf', s3_vf, near_pf, near_vf),
            ('Preflight s1 (2-sent split, no filler)', 'pf', s1_pf, 'vf', s1_vf, near_pf, near_vf),
            ('Preflight s2 (2-sent split + filler sent)', 'pf', s2_pf, 'vf', s2_vf, s1_pf, s1_vf),
        ]
        for label, _, a_pf, _, a_vf, b_pf, b_vf in rows:
            pf_dir = 'ans-first' if q == 'rev' else 'ans-after'
            vf_dir = 'ans-after' if q == 'rev' else 'ans-first'
            g_pf, l_pf, p_pf = mcn(a_pf, b_pf)
            g_vf, l_vf, p_vf = mcn(a_vf, b_vf)
            g_ord, l_ord, p_ord = mcn(a_vf, a_pf)
            print(f'  {label:42s} | pf({pf_dir}) {acc(a_pf):3d}/200 ({acc(a_pf)/2:4.1f}%, vs base +{g_pf}/-{l_pf} p={p_pf:.3g})'
                  f' | vf({vf_dir}) {acc(a_vf):3d}/200 ({acc(a_vf)/2:4.1f}%, vs base +{g_vf}/-{l_vf} p={p_vf:.3g})'
                  f' | vf-pf +{g_ord}/-{l_ord} p={p_ord:.3g}')
        print()

if __name__ == '__main__':
    main()
