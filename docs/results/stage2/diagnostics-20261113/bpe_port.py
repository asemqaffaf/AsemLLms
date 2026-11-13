"""Python port of lisp/src/bpe.lisp (pretokenize + merge ranks). Used for length checks and token-shape labels only."""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
_L = open(os.path.join(ROOT, 'data/fineweb/stage1/tokenizer.bpe')).read().split('\n')
MERGES = [tuple(map(int, l.split())) for l in _L[2:] if l.strip()]
RANK = {m: i for i, m in enumerate(MERGES)}
BYTES = {0: b''}
for _i in range(1, 257): BYTES[_i] = bytes([_i - 1])
for _i, (_a, _b) in enumerate(MERGES): BYTES[257 + _i] = BYTES[_a] + BYTES[_b]

def _cls(c):
    if c.isalpha(): return 'l'
    if c.isdigit(): return 'd'
    if c in ' \t\n\r\x0c\x0b': return 's'
    return 'o'

def pretok(t):
    n, i, out = len(t), 0, []
    def run(a, k):
        while a < n and _cls(t[a]) == k: a += 1
        return a
    while i < n:
        c = t[i]
        if c == ' ' and i + 1 < n and _cls(t[i + 1]) != 's':
            e = run(i + 1, _cls(t[i + 1]))
        elif _cls(c) == 's':
            e = run(i, 's')
            if e < n and e - i > 1 and t[e - 1] == ' ': e -= 1
        else:
            e = run(i, _cls(c))
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

def encode(t): return [i for p in pretok(t) for i in enc_piece(p)]
def ntok(t): return len(encode(t))
def pieces(t): return [BYTES[i].decode('utf8', 'replace') for i in encode(t)]
