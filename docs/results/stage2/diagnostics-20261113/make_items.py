#!/usr/bin/env python3
"""Inference-only diagnostic item builder (2026-11-13). Model-free; writes new JSONL files only, never overwrites.

Scored afterwards by the unmodified lisp/bin/qa-analysis.lisp (controls=0 paired=0) with frozen R7 weights.
Inputs: lisp/eval/qa/dev.jsonl (public dev) and the saved training trace audit-training.sexp. No locked set is read.

Design, fixed before any inference
----------------------------------
D1 reverse binding. 200 fact sets (100 town, 100 instrument), four people each, dev-pool entities, TRAINING wording.
   Each set is rendered twice with identical facts, sentence order, question and options:
     pf  person-first : "{P} lives in {V}."            / "{P} plays the {V}."
     vf  value-first  : "In {V} lives {P}." (50) or "{V} is where {P} lives." (50) / "The {V} is the instrument {P} plays."
   Reverse question (value -> person, options = the four people) is the primary outcome.
   Mirror control: an extraction question (person -> value, options = the four values) about the same fact on the same
   two passages. If retrieval is directional, reverse should be better in vf and extraction better in pf (crossover).
   If vf frames are merely less familiar, both should fall in vf.
   Secondary replicate d1t: the same 200 sets with the four people renamed to training-pool names.
D2 answer-string bias. The 100 reverse + 100 compare public dev items, dev wording unchanged.
   rot0..rot3: the item's own four names are cyclically reassigned to the four person slots in the passage. Question and
     option block are byte-identical across rotations; each name is the gold answer exactly once. rot0 = original item.
     A prediction that follows the name is right in exactly one rotation; one that follows the role is right in all four.
   easy / hard / train: all four people renamed to names from one homogeneous set (rot0 structure):
     easy  = 12 dev names with the highest mean option log-probability in the saved R7 dev predictions
     hard  = 12 dev names with the lowest
     train = names taken from options in the saved R7 training trace (training-pool names)
"""
import json, os, random, re, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
OUT = os.path.dirname(os.path.abspath(__file__))
SEED = 20261113
CTX = 128

# ---------------------------------------------------------------- tokenizer port (length pre-check only)
L = open(os.path.join(ROOT, 'data/fineweb/stage1/tokenizer.bpe')).read().split('\n')
MERGES = [tuple(map(int, l.split())) for l in L[2:] if l.strip()]
RANK = {m: i for i, m in enumerate(MERGES)}

def cls(c):
    if c.isalpha(): return 'l'
    if c.isdigit(): return 'd'
    if c in ' \t\n\r\x0c\x0b': return 's'
    return 'o'

def pretok(t):
    n, i, out = len(t), 0, []
    def run(a, k):
        while a < n and cls(t[a]) == k: a += 1
        return a
    while i < n:
        c = t[i]
        if c == ' ' and i + 1 < n and cls(t[i + 1]) != 's':
            e = run(i + 1, cls(t[i + 1]))
        elif cls(c) == 's':
            e = run(i, 's')
            if e < n and e - i > 1 and t[e - 1] == ' ': e -= 1
        else:
            e = run(i, cls(c))
        out.append(t[i:e]); i = e
    return out

_cache = {}
def enc_piece(p):
    if p in _cache: return _cache[p]
    s = [b + 1 for b in p.encode()]
    while True:
        best = None
        for i in range(len(s) - 1):
            r = RANK.get((s[i], s[i + 1]))
            if r is not None and (best is None or r < best[0]): best = (r, i)
        if not best: break
        r, i = best; a, b = s[i], s[i + 1]; o = []; j = 0
        while j < len(s):
            if j + 1 < len(s) and s[j] == a and s[j + 1] == b: o.append(257 + r); j += 2
            else: o.append(s[j]); j += 1
        s = o
    _cache[p] = s
    return s

def ntok(t): return sum(len(enc_piece(p)) for p in pretok(t))

def prompt(passage, question, options):
    return 'Text: %s\nQ: %s\nA. %s\nB. %s\nC. %s\nD. %s\nThe answer is' % (passage, question, *options)

def fits(passage, question, options):
    return 1 + ntok(prompt(passage, question, options)) + max(ntok(' ' + o) for o in options) <= CTX

# ---------------------------------------------------------------- pools
dev = [json.loads(l) for l in open(os.path.join(ROOT, 'lisp/eval/qa/dev.jsonl'))]
NAMES, TOWNS, INST = set(), set(), set()
for r in dev:
    if r['category'] in ('reverse', 'compare', 'relation'): NAMES |= set(r['options'])
    if r['category'] == 'extract' and r['subtype'] == 'town': TOWNS |= set(r['options'])
    if r['category'] == 'extract' and r['subtype'] == 'instrument': INST |= set(r['options'])
NAMES, TOWNS, INST = sorted(NAMES), sorted(TOWNS), sorted(INST)
assert len(NAMES) == 36 and len(INST) == 4

# Ranked by mean summed option log-probability over reverse+compare+relation dev items in
# docs/results/stage2/r7-analysis/qa-r7-last.json (computed before this script was written).
HARD = ['Ines', 'Edith', 'Andre', 'Alba', 'Una', 'Wade', 'Emma', 'Jack', 'Raj', 'Eva', 'Aster', 'Adam']
EASY = ['Otto', 'Kira', 'Sara', 'Kai', 'Finn', 'Jean', 'Mia', 'Neil', 'Pedro', 'Lena', 'Lou', 'Ben']
assert set(HARD) <= set(NAMES) and set(EASY) <= set(NAMES)

trace = open(os.path.join(ROOT, 'docs/results/stage2/mission-20261111/audit-training.sexp')).read()
TRAIN = []
for rec in re.split(r'(?=\(:KIND :\w+ :IDS)', trace)[1:]:
    if re.match(r'\(:KIND :(REL|CMP)', rec):
        for o in re.findall(r':OPTIONS \(([^)]*)\)', rec):
            for n in re.findall(r'"([^"]*)"', o):
                if n.isalpha() and n[0].isupper() and 4 <= len(n) <= 7 and 2 <= ntok(' ' + n) <= 3 and n not in NAMES and n not in TRAIN:
                    TRAIN.append(n)
TRAIN.sort()

def no_substrings(xs):
    return all(a == b or a.lower() not in b.lower() for a in xs for b in xs)

def pick4(pool, rng, avoid_text=''):
    while True:
        xs = rng.sample(pool, 4)
        if no_substrings(xs) and not any(re.search(r'\b%s\b' % re.escape(x), avoid_text) for x in xs): return xs

def rename(text, mapping):
    """Simultaneous whole-word replacement."""
    keys = sorted(mapping, key=len, reverse=True)
    return re.sub(r'\b(%s)\b' % '|'.join(map(re.escape, keys)), lambda m: mapping[m.group(1)], text)

def item(id_, cat, sub, passage, question, options, answer):
    assert len(set(options)) == 4 and no_substrings(options)
    assert all(re.search(r'\b%s\b' % re.escape(o), passage) for o in options)
    return {'id': id_, 'split': 'dev', 'category': cat, 'subtype': sub, 'passage': passage, 'question': question,
            'options': options, 'answer': answer, 'answer_letter': 'ABCD'[answer], 'cf_question': '', 'cf_answer': 0,
            'qframe': '', 'query': '()', 'facts': '()'}

def write(name, items):
    path = os.path.join(OUT, name)
    if os.path.exists(path): sys.exit('refusing to overwrite ' + path)
    with open(path, 'w') as f:
        for it in items: f.write(json.dumps(it) + '\n')
    print('%-28s %4d items' % (name, len(items)))

# ---------------------------------------------------------------- D1
FR = {'town': {'pf': ['{P} lives in {V}.'], 'vf': ['In {V} lives {P}.', '{V} is where {P} lives.'],
               'qrev': ['Who lives in {V}?', 'Which person lives in {V}?'],
               'qext': ['Where does {P} live?', 'Which town does {P} live in?']},
      'instrument': {'pf': ['{P} plays the {V}.'], 'vf': ['The {V} is the instrument {P} plays.'],
                     'qrev': ['Who plays the {V}?', 'Which person plays the {V}?'],
                     'qext': ['What instrument does {P} play?', 'What does {P} play?']}}

def fill(f, p, v): return f.replace('{P}', p).replace('{V}', v)

def build_d1(tag, name_pool):
    rng = random.Random(SEED + (0 if tag == 'd1' else 1))
    base = random.Random(SEED)          # facts structure identical for d1 and d1t apart from the names
    out = {k: [] for k in ('rev-pf', 'rev-vf', 'ext-pf', 'ext-vf')}
    for i in range(200):
        attr = 'town' if i < 100 else 'instrument'
        j = i % 100
        gold_letter, gold_sent, vfi, qi = j % 4, (j // 4) % 4, (j // 16) % 2, (j // 32) % 2
        vals = base.sample(TOWNS, 4) if attr == 'town' else base.sample(INST, 4)
        while not no_substrings(vals): vals = base.sample(TOWNS, 4)
        people = pick4(name_pool, rng)
        operm = base.sample(range(4), 4)                     # distractor order for the option lists
        others = [k for k in operm if k != gold_sent]
        order = others[:gold_letter] + [gold_sent] + others[gold_letter:]
        f = FR[attr]
        vf_frame = f['vf'][vfi % len(f['vf'])]
        passages = {'pf': ' '.join(fill(f['pf'][0], p, v) for p, v in zip(people, vals)),
                    'vf': ' '.join(fill(vf_frame, p, v) for p, v in zip(people, vals))}
        qrev = fill(f['qrev'][qi], '', vals[gold_sent]); qext = fill(f['qext'][qi], people[gold_sent], '')
        for c in ('pf', 'vf'):
            sub = '%s-%s%s' % (attr, c, ('-' + 'AB'[vfi % len(f['vf'])]) if (c == 'vf' and attr == 'town') else '')
            ro, eo = [people[k] for k in order], [vals[k] for k in order]
            assert fits(passages[c], qrev, ro) and fits(passages[c], qext, eo)
            out['rev-' + c].append(item('%s-%03d' % (tag, i), 'reverse', sub, passages[c], qrev, ro, gold_letter))
            out['ext-' + c].append(item('%s-%03d' % (tag, i), 'extract', sub, passages[c], qext, eo, gold_letter))
    for k, v in out.items(): write('%s-%s.jsonl' % (tag, k), v)

build_d1('d1', NAMES)
build_d1('d1t', TRAIN)

# ---------------------------------------------------------------- D2
src = [r for r in dev if r['category'] in ('reverse', 'compare')]
assert len(src) == 200
rng = random.Random(SEED + 2)
sets = {'rot0': [], 'rot1': [], 'rot2': [], 'rot3': [], 'easy': [], 'hard': [], 'train': []}
dropped = {}
for r in src:
    o, g = r['options'], r['answer']
    assert not any(re.search(r'\b%s\b' % re.escape(n), r['question']) for n in o)
    variants = {}
    for k in range(4):                                       # slot j (original name o[j]) now carries name o[(j+k)%4]
        variants['rot%d' % k] = (rename(r['passage'], {o[j]: o[(j + k) % 4] for j in range(4)}), list(o), (g + k) % 4)
    for cond, pool in (('easy', EASY), ('hard', HARD), ('train', TRAIN)):
        new = pick4(pool, rng, avoid_text='' if cond != 'train' else r['passage'] + ' ' + r['question'])
        variants[cond] = (rename(r['passage'], dict(zip(o, new))), new, g)
    assert variants['rot0'][0] == r['passage']
    for cond, (p, opts, ans) in variants.items():
        if fits(p, r['question'], opts):
            sets[cond].append(item(r['id'], r['category'], r['subtype'], p, r['question'], opts, ans))
        else:
            dropped.setdefault(cond, []).append(r['id'])
for k, v in sets.items(): write('d2-%s.jsonl' % k, v)
print('dropped for context length:', {k: len(v) for k, v in dropped.items()}, dropped)
print('train-pool names available:', len(TRAIN))
