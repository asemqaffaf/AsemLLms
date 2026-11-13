#!/usr/bin/env python3
"""Reviewer check B1 (2026-11-13): Experiment 1 minimal pairs in dev wording. Model-free; writes new JSONL only, never overwrites.

Scored afterwards by the unmodified lisp/bin/qa-analysis.lisp (controls=0 paired=0) on frozen R7 only. No training.
Inputs: pilot4/causal_probe_items.json (the 157 Experiment 1 pairs), lisp/eval/qa/dev.jsonl (public dev, already scored on R7),
pilot4/audit_raw.json (token exposure counts). confirmation.sexp, test.jsonl and test.lock are not read.

Design, fixed before any inference
----------------------------------
A pair = one item rendered twice. Only the gold person's name differs (passage + option list); facts, sentence order,
frames, question and option order are byte-identical otherwise. Exposure strata are the Experiment 1 strata:
  zero  = some subword of " "+name has 0 occurrences in R7 retrieval data and in Pilot 3 data
  hi100 = every subword has >= 100 occurrences in both
  mid   = anything else (excluded, as in Experiment 1)
Swap names come from the same pools and the same cycling rule as pilot4/prepare_causal_probes.py.

e1-train  The 157 Experiment 1 pairs exactly as saved (training wording, person-first). Pipeline check: must reproduce
          the saved canonical counts 15 -> 55/90 and 34 -> 10/67.
e1-dev    PRIMARY. The same 157 pairs (same ids, facts, names, swap names, option order) with passage and question
          re-rendered in dev wording, person-first order kept so only the wording changes:
            town        "{P} resides in {V}."
            instrument  "In the band, {P} is on the {V}." or "{P} has mastered the {V}." (seeded choice per sentence)
            prefix      "A record of four members follows." on a seeded half of the items (dev rate 49/100)
            questions   "Who lives in" -> "Who resides in {V}?";  "Which person lives in" -> "Whose place of residence is {V}?"
                        "Who plays the" -> "Who is associated with the {V}?";  "Which person plays the" -> "Whose instrument is the {V}?"
devrev    SECONDARY. The real public dev reverse items (n = 100, the D2 items), unchanged vs gold-only swap.
devcmp    SECONDARY. The real public dev compare items (n = 100, the D2 items), unchanged vs gold-only swap.
Outcome: K = 4 summed option log-probability (chance 25%), paired exact McNemar, per direction.
"""
import json, os, random, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'docs/results/stage2/diagnostics-20261113'))
from bpe_port import ntok, encode  # noqa: E402

SEED = 2026111301
CTX = 128
P4 = os.path.join(ROOT, 'docs/results/stage2/mission-20261111/pilot4')

tm = {t['id']: t for t in json.load(open(os.path.join(P4, 'audit_raw.json')))['vocab_tokens']}

def stratum(name):
    ids = encode(' ' + name)
    p3, r7 = min(tm[i]['p3_all'] for i in ids), min(tm[i]['r7_ret_all'] for i in ids)
    if p3 >= 100 and r7 >= 100: return 'hi100'
    if p3 == 0 and r7 == 0: return 'zero'
    return 'mid'

HI2 = ['Lena', 'Mia', 'Neil', 'Rosa', 'Sara', 'Kai', 'Gus', 'Dev', 'Jean']
HI3 = ['Kira', 'Otto', 'Pedro']
Z0 = ['Edith', 'Adam', 'Alba', 'Eva', 'Andre', 'Emma', 'Una']
ZT = ['Finn', 'Hugo', 'Tess', 'Wade', 'Jack', 'Mae', 'Noah', 'Raj']
assert all(stratum(n) == 'hi100' for n in HI2 + HI3) and all(stratum(n) == 'zero' for n in Z0 + ZT)

def prompt(p, q, o): return 'Text: %s\nQ: %s\nA. %s\nB. %s\nC. %s\nD. %s\nThe answer is' % (p, q, *o)
def fits(p, q, o): return 1 + ntok(prompt(p, q, o)) + max(ntok(' ' + x) for x in o) <= CTX

def item(id_, cat, sub, passage, question, options, answer):
    assert len(set(options)) == 4
    assert all(re.search(r'\b%s\b' % re.escape(o), passage) for o in options)
    return {'id': id_, 'split': 'dev', 'category': cat, 'subtype': sub, 'passage': passage, 'question': question,
            'options': options, 'answer': answer, 'answer_letter': 'ABCD'[answer], 'cf_question': '', 'cf_answer': 0,
            'qframe': '', 'query': '()', 'facts': '()'}

def write(name, items):
    path = os.path.join(HERE, name)
    if os.path.exists(path): sys.exit('refusing to overwrite ' + path)
    with open(path, 'w') as f:
        for it in items: f.write(json.dumps(it) + '\n')
    print('%-24s %4d items' % (name, len(items)))

def swap_name(text, old, new):
    out = re.sub(r'\b%s\b' % re.escape(old), new, text)
    assert out != text
    return out

# ---------------------------------------------------------------- e1-train and e1-dev (the 157 saved pairs)
pairs = json.load(open(os.path.join(P4, 'causal_probe_items.json')))['swap_pairs']
assert len(pairs) == 157
rng = random.Random(SEED)
QMAP = {'Who lives in': 'Who resides in %s?', 'Which person lives in': 'Whose place of residence is %s?',
        'Who plays the': 'Who is associated with the %s?', 'Which person plays the': 'Whose instrument is the %s?'}
meta = []
out = {'e1-train-orig': [], 'e1-train-swap': [], 'e1-dev-orig': [], 'e1-dev-swap': []}
for p in pairs:
    gold, new = p['orig_gold'], p['swap_gold']
    assert p['options'][p['answer']] == gold and stratum(gold) in ('zero', 'hi100') and stratum(new) != stratum(gold)
    sents = [s.strip() for s in p['passage'].split('.') if s.strip()]
    assert len(sents) == 4
    town = ' lives in ' in sents[0]
    facts = [re.fullmatch(r'(\w+) lives in (\w+)' if town else r'(\w+) plays the (\w+)', s).groups() for s in sents]
    m = re.fullmatch(r'(Who lives in|Which person lives in|Who plays the|Which person plays the) (\w+)\?', p['question'])
    q = QMAP[m.group(1)] % m.group(2)
    prefix = 'A record of four members follows. ' if rng.random() < 0.5 else ''
    frames = ['%s resides in %s.'] * 4 if town else [rng.choice(['In the band, %s is on the %s.', '%s has mastered the %s.']) for _ in range(4)]
    dev_passage = prefix + ' '.join(f % pv for f, pv in zip(frames, facts))
    sub = ('town' if town else 'instrument') + '-pf'
    sw_opts = [new if o == gold else o for o in p['options']]
    assert sw_opts == p['swap_options'] and swap_name(p['passage'], gold, new) == p['swap_passage']
    rows = {'e1-train-orig': (p['passage'], p['question'], p['options']),
            'e1-train-swap': (p['swap_passage'], p['question'], sw_opts),
            'e1-dev-orig': (dev_passage, q, p['options']),
            'e1-dev-swap': (swap_name(dev_passage, gold, new), q, sw_opts)}
    for k, (pa, qu, op) in rows.items():
        assert fits(pa, qu, op), (k, p['id'])
        out[k].append(item(p['id'], 'reverse', sub, pa, qu, op, p['answer']))
    meta.append({'id': p['id'], 'set': 'e1', 'direction': p['direction'], 'orig_gold': gold, 'swap_gold': new,
                 'attr': 'town' if town else 'instrument', 'prefix': bool(prefix)})
for k, v in out.items(): write(k + '.jsonl', v)

# ---------------------------------------------------------------- devrev / devcmp (real dev items, gold-only swap)
dev = [json.loads(l) for l in open(os.path.join(ROOT, 'lisp/eval/qa/dev.jsonl'))]
for tag, cat in (('devrev', 'reverse'), ('devcmp', 'compare')):
    src = [r for r in dev if r['category'] == cat]
    assert len(src) == 100
    hi_idx = z0_idx = zt_idx = n_h2z = 0
    orig, swap, dropped = [], [], []
    for r in src:
        gold = r['options'][r['answer']]
        st = stratum(gold)
        if st == 'mid': continue
        used = set(r['options']) | set(re.findall(r'[A-Za-z]+', r['passage'] + ' ' + r['question']))
        if st == 'zero':
            cand = (HI3 + HI2) if ntok(' ' + gold) >= 3 else (HI2 + HI3)
            k = next(k for k in range(len(cand)) if cand[(hi_idx + k) % len(cand)] not in used)
            new = cand[(hi_idx + k) % len(cand)]; hi_idx = (hi_idx + k + 1) % len(cand)
            direction = 'zero_to_hi100'
        else:
            prefer_z0 = n_h2z % 2 == 0; n_h2z += 1
            cand = (Z0 + ZT) if prefer_z0 else (ZT + Z0)
            ptr = z0_idx if prefer_z0 else zt_idx
            k = next(k for k in range(len(cand)) if cand[(ptr + k) % len(cand)] not in used)
            new = cand[(ptr + k) % len(cand)]
            if prefer_z0: z0_idx = (z0_idx + k + 1) % len(cand)
            else: zt_idx = (zt_idx + k + 1) % len(cand)
            direction = 'hi100_to_zero'
        assert not re.search(r'\b%s\b' % re.escape(gold), r['question'])
        sp = swap_name(r['passage'], gold, new)
        so = [new if o == gold else o for o in r['options']]
        if not (fits(r['passage'], r['question'], r['options']) and fits(sp, r['question'], so)):
            dropped.append(r['id']); continue
        orig.append(item(r['id'], cat, r['subtype'], r['passage'], r['question'], r['options'], r['answer']))
        swap.append(item(r['id'], cat, r['subtype'], sp, r['question'], so, r['answer']))
        meta.append({'id': r['id'], 'set': tag, 'direction': direction, 'orig_gold': gold, 'swap_gold': new,
                     'attr': r['subtype'], 'prefix': r['passage'].startswith('A record of four members follows.')})
    write(tag + '-orig.jsonl', orig); write(tag + '-swap.jsonl', swap)
    print('  dropped for context length:', dropped)

mp = os.path.join(HERE, 'pairs-meta.json')
if os.path.exists(mp): sys.exit('refusing to overwrite ' + mp)
json.dump(meta, open(mp, 'w'), indent=1)
from collections import Counter
print(Counter((m['set'], m['direction']) for m in meta))
