#!/usr/bin/env python3
"""Preflight inference-only item builder: separate-sentence filler on frozen D1 facts (n = 200 paired).

Writes new JSONL files inside pilot3/ only; never modifies D1, R7, or any locked/eval file.
Three paired conditions (all on the same 200 D1 fact sets, pf = P before V, vf = V before P):
  s1 : two-sentence entity/answer split, no filler (wording baseline for cross-sentence split)
  s2 : same two-sentence split with one full unrelated filler sentence between the two halves
  s3 : exact D1 single-sentence wording intact, followed immediately by one full unrelated sentence
"""
import json, os, re, sys

P3 = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(P3, '..', '..', '..', '..', '..'))
DIAG = os.path.join(ROOT, 'docs', 'results', 'stage2', 'diagnostics-20261113')
sys.path.insert(0, DIAG)
from bpe_port import ntok

CTX = 128
FILLERS = ['Water ran.', 'Day began.', 'It is cold.', 'The sun set.']

def prompt(p, q, o):
    return 'Text: %s\nQ: %s\nA. %s\nB. %s\nC. %s\nD. %s\nThe answer is' % (p, q, *o)

def fits(p, q, o):
    return 1 + ntok(prompt(p, q, o)) + max(ntok(' ' + x) for x in o) <= CTX

def write_jsonl(name, items):
    path = os.path.join(P3, name)
    if os.path.exists(path):
        sys.exit('refusing to overwrite ' + path)
    with open(path, 'w') as f:
        for it in items:
            f.write(json.dumps(it) + '\n')
    print('%-34s %3d items' % (name, len(items)))

def main():
    d1_entities = set(l.strip().lower() for l in open(os.path.join(P3, '..', 'pilot2', 'd1-entities.txt')) if l.strip())
    for fl in FILLERS:
        for w in re.findall(r'[A-Za-z]+', fl):
            assert w.lower() not in d1_entities, f'filler word {w} overlaps D1 entities'

    for qtype in ('rev', 'ext'):
        rows_pf = [json.loads(l) for l in open(os.path.join(DIAG, f'd1-{qtype}-pf.jsonl'))]
        rows_vf = [json.loads(l) for l in open(os.path.join(DIAG, f'd1-{qtype}-vf.jsonl'))]
        assert len(rows_pf) == 200 and len(rows_vf) == 200
        out = {f'{s}-{o}': [] for s in ('s1', 's2', 's3') for o in ('pf', 'vf')}
        max_len = 0
        for i, (rpf, rvf) in enumerate(zip(rows_pf, rows_vf)):
            assert rpf['id'] == rvf['id']
            sents_pf = [s.rstrip('.') for s in rpf['passage'].split('. ')]
            sents_vf = [s.rstrip('.') for s in rvf['passage'].split('. ')]
            s1_pf, s1_vf, s2_pf, s2_vf, s3_pf, s3_vf = [], [], [], [], [], []
            for k, (sp, sv) in enumerate(zip(sents_pf, sents_vf)):
                ws = sp.split()
                P, V = ws[0], ws[-1]
                fl = FILLERS[(i + k) % len(FILLERS)]
                if 'lives in' in sp:
                    s1_pf.append(f'{P} lives somewhere. It is {V}.')
                    s1_vf.append(f'Someone lives in {V}. It is {P}.')
                    s2_pf.append(f'{P} lives somewhere. {fl} It is {V}.')
                    s2_vf.append(f'Someone lives in {V}. {fl} It is {P}.')
                else:
                    s1_pf.append(f'{P} plays music. It is the {V}.')
                    s1_vf.append(f'Someone plays the {V}. It is {P}.')
                    s2_pf.append(f'{P} plays music. {fl} It is the {V}.')
                    s2_vf.append(f'Someone plays the {V}. {fl} It is {P}.')
                s3_pf.append(f'{sp}. {fl}')
                s3_vf.append(f'{sv}. {fl}')

            variants = {
                's1-pf': (rpf, ' '.join(s1_pf)),
                's1-vf': (rvf, ' '.join(s1_vf)),
                's2-pf': (rpf, ' '.join(s2_pf)),
                's2-vf': (rvf, ' '.join(s2_vf)),
                's3-pf': (rpf, ' '.join(s3_pf)),
                's3-vf': (rvf, ' '.join(s3_vf)),
            }
            for tag, (base, passage) in variants.items():
                tl = 1 + ntok(prompt(passage, base['question'], base['options'])) + max(ntok(' ' + o) for o in base['options'])
                max_len = max(max_len, tl)
                assert tl <= CTX, f'{qtype} {tag} {base["id"]} len={tl} > {CTX}'
                it = dict(base)
                it['passage'] = passage
                it['subtype'] = base['subtype'] + '-' + tag
                out[tag].append(it)
        print(f'{qtype}: max_tok={max_len} <= {CTX} (0 dropped)')
        for tag, items in out.items():
            write_jsonl(f'preflight-{tag}-{qtype}.jsonl', items)

if __name__ == '__main__':
    main()
