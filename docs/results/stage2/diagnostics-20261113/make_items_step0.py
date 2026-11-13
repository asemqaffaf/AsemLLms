#!/usr/bin/env python3
"""Step 0 item builder (2026-11-13). Model-free; new JSONL files only, never overwrites. No locked set is read.

D4 adjacency vs order. The 200 D1 fact sets, with a training-style modifier inserted BETWEEN the queried entity and the answer
   ("far"), paired with the unchanged D1 items ("near"). Order is unchanged by the filler; only the distance grows.
     town pf   "{P} lives in {V}."                      -> "{P} lives {M} in {V}."
     town vf-A "In {V} lives {P}."                      -> "In {V} lives {M} {P}."
     town vf-B "{V} is where {P} lives."                -> "{V} is where {M} {P} lives."
     instr pf  "{P} plays the {V}."                     -> "{P} plays {M} the {V}."
     instr vf  "The {V} is the instrument {P} plays."   -> "The {V} is {M} the instrument {P} plays."
   {M} is one of the multi-word modifiers the training augmentation inserts at random word gaps (retrieval.lisp *modifiers*),
   the same one in all four sentences of an item and in both orders.
   If adjacency drives D1, value-first reverse should fall toward person-first when the filler is added.
   If order drives D1, value-first reverse stays high with the filler and person-first reverse stays low.
D5 temporal place strings. The 100 temporal public dev items, dev wording unchanged.
   rot0..rot3: the item's own four places cyclically reassigned to the four stops (passage and question renamed together;
     the option block is byte-identical; each place is the gold answer exactly once). rot0 = original item.
   train / hard / onetok: all four places renamed to one homogeneous set:
     train  = single-word places from the training split of the generator's place pool
     hard   = theater, cinema, clinic, hotel (the four dev places with the lowest mean option log-probability under R7)
     onetok = single-token common place nouns outside the generator's place pool
"""
import json, os, random, re, sys
from bpe_port import ntok
D = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(D, '..', '..', '..', '..'))
SEED = 20261113; CTX = 128
def prompt(p, q, o): return 'Text: %s\nQ: %s\nA. %s\nB. %s\nC. %s\nD. %s\nThe answer is' % (p, q, *o)
def fits(p, q, o): return 1 + ntok(prompt(p, q, o)) + max(ntok(' ' + x) for x in o) <= CTX
def no_substrings(xs): return all(a == b or a.lower() not in b.lower() for a in xs for b in xs)
def rename(text, mapping):
    return re.sub(r'\b(%s)\b' % '|'.join(map(re.escape, sorted(mapping, key=len, reverse=True))), lambda m: mapping[m.group(1)], text)
def write(name, items):
    path = os.path.join(D, name)
    if os.path.exists(path): sys.exit('refusing to overwrite ' + path)
    with open(path, 'w') as f:
        for it in items: f.write(json.dumps(it) + '\n')
    print('%-26s %4d items' % (name, len(items)))
def rows(name): return [json.loads(l) for l in open(os.path.join(D, name))]

# ------------------------------------------------------------ D4
MODS = ['these days', 'for now', 'at the moment', 'as it happens', 'by all accounts', 'it seems', 'we hear', 'once again', 'as usual']
RULES = [(r'^(\w+) lives in (\w+)\.$', r'\1 lives {M} in \2.'), (r'^In (\w+) lives (\w+)\.$', r'In \1 lives {M} \2.'),
         (r'^(\w+) is where (\w+) lives\.$', r'\1 is where {M} \2 lives.'), (r'^(\w+) plays the (\w+)\.$', r'\1 plays {M} the \2.'),
         (r'^The (\w+) is the instrument (\w+) plays\.$', r'The \1 is {M} the instrument \2 plays.')]
def far(passage, m):
    out = []
    for s in re.split(r'(?<=\.)\s+', passage):
        for pat, rep in RULES:
            if re.match(pat, s): out.append(re.sub(pat, rep.replace('{M}', m), s)); break
        else: raise ValueError(s)
    return ' '.join(out)
for q in ('rev', 'ext'):
    for c in ('pf', 'vf'):
        new = []
        for i, it in enumerate(rows('d1-%s-%s.jsonl' % (q, c))):
            it = dict(it); it['passage'] = far(it['passage'], MODS[i % len(MODS)]); it['subtype'] += '-far'
            assert fits(it['passage'], it['question'], it['options'])
            new.append(it)
        write('d4-%s-%s-far.jsonl' % (q, c), new)

# ------------------------------------------------------------ D5
dev = [json.loads(l) for l in open(os.path.join(ROOT, 'lisp/eval/qa/dev.jsonl'))]
src = [r for r in dev if r['category'] == 'temporal']
assert len(src) == 100
TRAINP = ['bakery', 'library', 'market', 'gym', 'bank', 'station', 'school', 'garden', 'barber', 'zoo', 'chapel', 'farm', 'factory',
          'stadium', 'laundry', 'airport', 'beach']
HARDP = ['theater', 'cinema', 'clinic', 'hotel']
ONETOK = [w for w in ['river', 'bridge', 'church', 'tower', 'village', 'valley', 'island', 'office', 'shop', 'store', 'house', 'hall',
                      'field', 'road', 'street', 'city', 'mountain', 'lake', 'court', 'college', 'hospital', 'kitchen', 'camp', 'temple']
          if ntok(' ' + w) == 1]
assert len(ONETOK) >= 8, ONETOK
rng = random.Random(SEED + 5)
sets = {k: [] for k in ('rot0', 'rot1', 'rot2', 'rot3', 'train', 'hard', 'onetok')}; dropped = {}
for r in src:
    o, g = r['options'], r['answer']
    assert all(re.search(r'\b%s\b' % re.escape(x), r['passage']) for x in o)
    var = {}
    for k in range(4):
        m = {o[j]: o[(j + k) % 4] for j in range(4)}
        var['rot%d' % k] = (rename(r['passage'], m), rename(r['question'], m), list(o), (g + k) % 4)
    for cond, pool in (('train', TRAINP), ('hard', HARDP), ('onetok', ONETOK)):
        while True:
            new = rng.sample(pool, 4)
            if no_substrings(new) and not any(re.search(r'\b%s\b' % x, r['passage'] + ' ' + r['question']) and x not in o for x in new): break
        m = dict(zip(o, new))
        var[cond] = (rename(r['passage'], m), rename(r['question'], m), new, g)
    assert var['rot0'][0] == r['passage'] and var['rot0'][1] == r['question']
    for cond, (p, q, opts, ans) in var.items():
        if fits(p, q, opts):
            it = dict(r); it.update(passage=p, question=q, options=opts, answer=ans, answer_letter='ABCD'[ans], cf_question='', cf_answer=0, query='()', facts='()')
            sets[cond].append(it)
        else: dropped.setdefault(cond, []).append(r['id'])
for k, v in sets.items(): write('d5-%s.jsonl' % k, v)
print('dropped for context length:', dropped); print('onetok pool:', ONETOK)
