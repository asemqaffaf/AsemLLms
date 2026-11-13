#!/usr/bin/env python3
"""Reproducible detailed diagnostics and SHA-256 manifest builder for Pilot 3."""
import collections
import hashlib
import json
import os
import subprocess
from math import comb

P3 = os.path.dirname(os.path.abspath(__file__))
M = os.path.abspath(os.path.join(P3, '..'))
ROOT = os.path.abspath(os.path.join(P3, '..', '..', '..', '..', '..'))
DIAG = os.path.join(ROOT, 'docs', 'results', 'stage2', 'diagnostics-20261113')
RUN = os.path.join(ROOT, 'data', 'runs', 'retrieval', 'p3-learnability')
R7_CKPT = os.path.join(ROOT, 'data', 'runs', 'retrieval', 'r7-L123-aug-opts-rev-rsn', 'last.ckpt')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_eval(path):
    d = json.load(open(path))
    its = d['conditions']['real']['items']
    out = {}
    for x in its:
        out[x['id']] = {
            'cat': x['category'],
            'sub': x['subtype'],
            'k4': x['summed_option_prediction'] == x['key'],
            'exact': bool(x['gold_length_greedy_prefix_exact']),
            'bounded': bool(x['bounded_free_generation_exact']),
            'gen': x['generated_text'],
            'gold': x['gold_text'],
            'opt_lens': x['option_token_lengths'],
            'key': x['key'],
            'pred_k4': x['summed_option_prediction'],
        }
    return out


def mcn(a, b, field='k4', ids=None):
    if ids is None:
        ids = list(a)
    g = sum(bool(a[i][field]) and not bool(b[i][field]) for i in ids)
    l = sum(bool(b[i][field]) and not bool(a[i][field]) for i in ids)
    n = g + l
    k = min(g, l)
    p = min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / (2 ** n)) if n else 1.0
    return g, l, p


def get_d1_subword_min_counts():
    sbcl_code = r'''
(let* ((bpe (asem-bpe:load-bpe "data/fineweb/stage1/tokenizer.bpe"))
       (all-counts (make-array 8192 :initial-element 0))
       (data (with-open-file (in "data/runs/retrieval/p3-learnability/data-train.sexp") (read in))))
  (dolist (seq data)
    (dolist (id (first seq)) (incf (aref all-counts id))))
  (format t "JSON_START~%{")
  (let ((first t))
    (dolist (w '("Adam" "Alba" "Andre" "Aster" "Ben" "Carla" "Colin" "Dev" "Edith" "Emma" "Eva" "Finn"
                 "Gene" "Gus" "Hugo" "Ines" "Jack" "Jean" "Kai" "Kira" "Lena" "Lou" "Mae" "Mia"
                 "Neil" "Noah" "Otto" "Pedro" "Raj" "Rosa" "Sara" "Sofia" "Tess" "Una" "Wade" "Yusuf"))
      (let* ((sp (asem-bpe:bpe-encode bpe (concatenate 'string " " w)))
             (mc (apply #'min (mapcar (lambda (i) (aref all-counts i)) sp))))
        (unless first (write-string ","))
        (setf first nil)
        (format t "~s:~d" w mc))))
  (format t "}~%"))
'''
    out = subprocess.check_output(
        ['sbcl', '--noinform', '--non-interactive', '--load', os.path.join(ROOT, 'lisp/src/bpe.lisp'), '--eval', sbcl_code],
        cwd=ROOT, text=True
    )
    return json.loads(out.split('JSON_START\n')[1].strip())


def main():
    models = [
        ('R7 step 0', 'r7.step0'),
        ('arm-5e-4 step 250', 'arm-5e-4.step250'),
        ('arm-5e-4 step 500', 'arm-5e-4.step500'),
        ('arm-1e-3 step 250', 'arm-1e-3.step250'),
        ('arm-1e-3 step 500', 'arm-1e-3.step500'),
    ]

    print('=== 1. LOG TRAJECTORIES (50-STEP TRAIN LOSS & EVAL LOSS) ===')
    for arm in ('5e-4', '1e-3'):
        rows = [json.loads(l) for l in open(os.path.join(RUN, f'arm-{arm}', 'log.jsonl'))]
        tr = [r for r in rows if r['event'] == 'train']
        ev = {r['step']: r for r in rows if r['event'] == 'eval'}
        done = [r for r in rows if r['event'] == 'done'][-1]
        print(f'arm-{arm}: wall_seconds={done["seconds"]:.1f}')
        for r in tr:
            print(f'  step {r["step"]:3d}: train_loss={r["train_loss"]:.6f} train_acc={r["train_acc"]:.6f} lr={r["lr"]:.6f} grad_norm={r["grad_norm"]:.4f} tok/s={r["tokens_per_sec"]:.1f}')
        for st in sorted(ev):
            r = ev[st]
            print(f'  eval {st:3d}: dev_acc={r["dev_acc"]:.4f} ({r["dev_correct"]}/{r["dev_n"]}) dev_loss={r["dev_loss"]:.6f} train_probe_acc={r["train_probe_acc"]:.4f} train_probe_loss={r["train_probe_loss"]:.6f}')

    def load_tag(prefix, tag):
        if tag == 'r7.step0' and prefix in ('d1-rev-pf', 'd1-rev-vf', 'd1-ext-pf'):
            return load_eval(os.path.join(DIAG, f'{prefix}.r7.json'))
        if tag == 'r7.step0' and prefix == 'selection':
            return load_eval(os.path.join(M, 'r7-selection.json'))
        return load_eval(os.path.join(P3, f'{prefix}.{tag}.json'))

    print('\n=== 2. PAIRED MCNEMAR FOR EVERY K=4 AND EXACT CELL VS R7 ===')
    for ds_name, prefix, subgroups in [
        ('Fresh 500', 'probe-fresh-500', [('ALL', lambda x: True), ('reverse', lambda x: x['cat'] == 'reverse'), ('extract', lambda x: x['cat'] == 'extract')]),
        ('Seen 500', 'probe-seen-500', [('ALL', lambda x: True), ('reverse', lambda x: x['cat'] == 'reverse'), ('extract', lambda x: x['cat'] == 'extract')]),
        ('D1 rev-pf', 'd1-rev-pf', [('ALL', lambda x: True), ('town-pf', lambda x: x['sub'] == 'town-pf'), ('instrument-pf', lambda x: x['sub'] == 'instrument-pf')]),
        ('D1 rev-vf', 'd1-rev-vf', [('ALL', lambda x: True)]),
        ('D1 ext-pf', 'd1-ext-pf', [('ALL', lambda x: True)]),
        ('Selection', 'selection', [('ALL', lambda x: True), ('extract', lambda x: x['cat'] == 'extract'), ('relation', lambda x: x['cat'] == 'relation'), ('reverse', lambda x: x['cat'] == 'reverse'), ('compare', lambda x: x['cat'] == 'compare'), ('temporal', lambda x: x['cat'] == 'temporal')]),
    ]:
        print(f'-- {ds_name} --')
        loaded = {tag: load_tag(prefix, tag) for _, tag in models}
        r7m = loaded['r7.step0']
        for sub_name, pred in subgroups:
            ids = [i for i, v in r7m.items() if pred(v)]
            row_parts = []
            for label, tag in models:
                cur = loaded[tag]
                k4 = sum(cur[i]['k4'] for i in ids)
                ex = sum(cur[i]['exact'] for i in ids)
                gk, lk, pk = mcn(cur, r7m, 'k4', ids)
                ge, le, pe = mcn(cur, r7m, 'exact', ids)
                if tag == 'r7.step0':
                    row_parts.append(f'R7: K4={k4}/{len(ids)} ex={ex}/{len(ids)}')
                else:
                    row_parts.append(f'{tag}: K4={k4}/{len(ids)} (+{gk}/-{lk} p={pk:.3g}) ex={ex}/{len(ids)} (+{ge}/-{le} p={pe:.3g})')
            print(f'  {sub_name:14s}: ' + ' | '.join(row_parts))

    print('\n=== 3. D1 REV-PF COPY & BPE-SUBWORD SUPPORT MECHANISM (POST-HOC) ===')
    minc = get_d1_subword_min_counts()
    d1_in = {x['id']: x for x in (json.loads(l) for l in open(os.path.join(DIAG, 'd1-rev-pf.jsonl')))}
    for grp, pred in [
        ('ALL (n=200)', lambda n: True),
        ('All BPE subwords >=100 in P3 train', lambda n: minc[n] >= 100),
        ('Subword count 1..99 in P3 train', lambda n: 0 < minc[n] < 100),
        ('Has 0-count subword in P3 train', lambda n: minc[n] == 0),
    ]:
        ids = [i for i, x in d1_in.items() if pred(x['options'][x['answer']])]
        print(f'  {grp:36s} (n={len(ids):3d}):')
        r7d = load_tag('d1-rev-pf', 'r7.step0')
        for label, tag in models:
            m = load_tag('d1-rev-pf', tag)
            ex = sum(m[i]['exact'] for i in ids)
            k4 = sum(m[i]['k4'] for i in ids)
            in_opt = sum(any(m[i]['gen'].strip().startswith(o) for o in d1_in[i]['options']) for i in ids)
            gk, lk, pk = mcn(m, r7d, 'k4', ids)
            ge, le, pe = mcn(m, r7d, 'exact', ids)
            print(f'    {label:18s}: K4={k4:2d}/{len(ids)} ({100*k4/len(ids):5.1f}%, +{gk}/-{lk} p={pk:.3g}) | ex={ex:2d}/{len(ids)} ({100*ex/len(ids):5.1f}%, +{ge}/-{le} p={pe:.3g}) | gen_in_opts={in_opt:2d}/{len(ids)} ({100*in_opt/len(ids):5.1f}%)')

    print('\n=== 4. PROBE-SEEN-500 BY TRAINING STEP BUCKET ===')
    sn_meta = {x['id']: x for x in (json.loads(l) for l in open(os.path.join(P3, 'probe-seen-500.jsonl')))}
    sn_r7 = load_tag('probe-seen-500', 'r7.step0')
    sn_5_250 = load_tag('probe-seen-500', 'arm-5e-4.step250')
    sn_5_500 = load_tag('probe-seen-500', 'arm-5e-4.step500')
    sn_1_250 = load_tag('probe-seen-500', 'arm-1e-3.step250')
    sn_1_500 = load_tag('probe-seen-500', 'arm-1e-3.step500')
    for lo, hi in [(1, 100), (101, 200), (201, 250), (251, 300), (301, 400), (401, 500)]:
        ids = [i for i, m in sn_meta.items() if lo <= m['train_step'] <= hi]
        print(f'  steps {lo:3d}..{hi:3d} (n={len(ids):3d}): R7={sum(sn_r7[i]["exact"] for i in ids):3d} | 5e-4@250={sum(sn_5_250[i]["exact"] for i in ids):3d} | 5e-4@500={sum(sn_5_500[i]["exact"] for i in ids):3d} | 1e-3@250={sum(sn_1_250[i]["exact"] for i in ids):3d} | 1e-3@500={sum(sn_1_500[i]["exact"] for i in ids):3d}')

    # Build complete SHA-256 manifest
    manifest = {
        'schema': 'asem-pilot3-manifest-1',
        'frozen_inputs': {
            'data/runs/retrieval/r7-L123-aug-opts-rev-rsn/last.ckpt': sha256(R7_CKPT),
            'data/fineweb/stage1/tokenizer.bpe': sha256(os.path.join(ROOT, 'data/fineweb/stage1/tokenizer.bpe')),
            'data/runs/retrieval/p3-learnability/data-train.sexp': sha256(os.path.join(RUN, 'data-train.sexp')),
            'docs/results/stage2/mission-20261111/selection.sexp': sha256(os.path.join(M, 'selection.sexp')),
            'lisp/bin/qa-analysis.lisp': sha256(os.path.join(ROOT, 'lisp/bin/qa-analysis.lisp')),
            'lisp/bin/mission-eval.lisp': sha256(os.path.join(ROOT, 'lisp/bin/mission-eval.lisp')),
            'lisp/src/qagen.lisp': sha256(os.path.join(ROOT, 'lisp/src/qagen.lisp')),
            'lisp/src/retrieval.lisp': sha256(os.path.join(ROOT, 'lisp/src/retrieval.lisp')),
            'lisp/src/sft.lisp': sha256(os.path.join(ROOT, 'lisp/src/sft.lisp')),
        },
        'pilot3_checkpoints': {
            'data/runs/retrieval/p3-learnability/arm-5e-4/step-00250.ckpt': sha256(os.path.join(RUN, 'arm-5e-4/step-00250.ckpt')),
            'data/runs/retrieval/p3-learnability/arm-5e-4/step-00500.ckpt': sha256(os.path.join(RUN, 'arm-5e-4/step-00500.ckpt')),
            'data/runs/retrieval/p3-learnability/arm-5e-4/last.ckpt': sha256(os.path.join(RUN, 'arm-5e-4/last.ckpt')),
            'data/runs/retrieval/p3-learnability/arm-5e-4/log.jsonl': sha256(os.path.join(RUN, 'arm-5e-4/log.jsonl')),
            'data/runs/retrieval/p3-learnability/arm-1e-3/step-00250.ckpt': sha256(os.path.join(RUN, 'arm-1e-3/step-00250.ckpt')),
            'data/runs/retrieval/p3-learnability/arm-1e-3/step-00500.ckpt': sha256(os.path.join(RUN, 'arm-1e-3/step-00500.ckpt')),
            'data/runs/retrieval/p3-learnability/arm-1e-3/last.ckpt': sha256(os.path.join(RUN, 'arm-1e-3/last.ckpt')),
            'data/runs/retrieval/p3-learnability/arm-1e-3/log.jsonl': sha256(os.path.join(RUN, 'arm-1e-3/log.jsonl')),
        },
        'pilot3_bundle': {},
    }
    for fn in sorted(os.listdir(P3)):
        fp = os.path.join(P3, fn)
        if os.path.isfile(fp) and fn != 'manifest.json':
            manifest['pilot3_bundle'][f'docs/results/stage2/mission-20261111/pilot3/{fn}'] = sha256(fp)
    with open(os.path.join(P3, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'\nWrote {os.path.join(P3, "manifest.json")} ({len(manifest["pilot3_bundle"])} bundle files)')


if __name__ == '__main__':
    main()
