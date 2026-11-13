#!/usr/bin/env python3
"""Analyze Pilot 3 saved predictions and training logs only. Prints tables and evaluates Decision Table A-E.

Ranking tables: K = 4 options, chance 25%, summed option log-probability (not length-normalized).
Exact-answer tables: autoregressive greedy prefix exact match over full vocab (K = 8192 per token, chance ~0%).
"""
import argparse, collections, hashlib, json, os
from math import comb

P3 = os.path.dirname(os.path.abspath(__file__))
M = os.path.abspath(os.path.join(P3, '..'))
ROOT = os.path.abspath(os.path.join(P3, '..', '..', '..', '..', '..'))
DIAG = os.path.join(ROOT, 'docs', 'results', 'stage2', 'diagnostics-20261113')
RUN = os.path.join(ROOT, 'data', 'runs', 'retrieval', 'p3-learnability')
R7_CKPT = os.path.join(ROOT, 'data', 'runs', 'retrieval', 'r7-L123-aug-opts-rev-rsn', 'last.ckpt')
R7_SHA = '33a5c24521dfb1b28b456b8344bfd0869b84fe56aa622191b8f6bfa71cc42aa1'

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
        }
    return out, d['conditions']['real']['summary']

def mcn(a, b, ids=None):
    if ids is None:
        ids = list(a)
    g = sum(a[i]['k4'] and not b[i]['k4'] for i in ids)
    l = sum(b[i]['k4'] and not a[i]['k4'] for i in ids)
    n = g + l
    k = min(g, l)
    p = min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / (2 ** n)) if n else 1.0
    return g, l, p

def mcn_metric(a, b, field, ids=None):
    if ids is None:
        ids = list(a)
    g = sum(a[i][field] and not b[i][field] for i in ids)
    l = sum(b[i][field] and not a[i][field] for i in ids)
    n = g + l
    k = min(g, l)
    p = min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / (2 ** n)) if n else 1.0
    return g, l, p

def load_probe_meta(name):
    return {x['id']: x for x in (json.loads(l) for l in open(os.path.join(P3, name)))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write-json', action='store_true')
    args = ap.parse_args()

    r7_cur = sha256(R7_CKPT)
    assert r7_cur == R7_SHA, f'R7 changed: {r7_cur}'
    print(f'R7 verified unchanged: {r7_cur}')

    models = [
        ('R7 step 0', 'r7.step0', R7_CKPT),
        ('arm-5e-4 step 250', 'arm-5e-4.step250', os.path.join(RUN, 'arm-5e-4', 'step-00250.ckpt')),
        ('arm-5e-4 step 500', 'arm-5e-4.step500', os.path.join(RUN, 'arm-5e-4', 'step-00500.ckpt')),
        ('arm-1e-3 step 250', 'arm-1e-3.step250', os.path.join(RUN, 'arm-1e-3', 'step-00250.ckpt')),
        ('arm-1e-3 step 500', 'arm-1e-3.step500', os.path.join(RUN, 'arm-1e-3', 'step-00500.ckpt')),
    ]

    print('\n=== CHECKPOINTS ===')
    ckpts = {}
    for label, tag, path in models:
        h = sha256(path)
        ckpts[tag] = h
        print(f'  {label:20s}  sha256={h}')

    print('\n=== TRAINING LOSS TRAJECTORY (log.jsonl, mean answer-token loss per 50 steps) ===')
    loss_info = {}
    for arm in ('5e-4', '1e-3'):
        log_path = os.path.join(RUN, f'arm-{arm}', 'log.jsonl')
        rows = [json.loads(l) for l in open(log_path)]
        tr = [r for r in rows if r.get('event') == 'train']
        done = [r for r in rows if r.get('event') == 'done']
        secs = done[-1]['seconds'] if done else None
        pts = {r['step']: r['train_loss'] for r in tr}
        first_half = [r['train_loss'] for r in tr if r['step'] <= 250]
        second_half = [r['train_loss'] for r in tr if r['step'] > 250]
        early_250 = [r['train_loss'] for r in tr if 250 <= r['step'] <= 350]
        late_500 = [r['train_loss'] for r in tr if 400 <= r['step'] <= 500]
        loss_info[arm] = {
            'pts': pts,
            'mean_250_350': sum(early_250) / len(early_250) if early_250 else None,
            'mean_400_500': sum(late_500) / len(late_500) if late_500 else None,
            'step250': pts.get(250),
            'step500': pts.get(500),
            'seconds': secs,
        }
        line = ' '.join(f'{r["step"]}:{r["train_loss"]:.3f}' for r in tr)
        print(f'  arm-{arm:4s} ({secs:.1f}s): {line}')
        print(f'           mean(250..350)={loss_info[arm]["mean_250_350"]:.3f} -> mean(400..500)={loss_info[arm]["mean_400_500"]:.3f} | diff={loss_info[arm]["mean_400_500"] - loss_info[arm]["mean_250_350"]:+.3f}')

    seen_meta = load_probe_meta('probe-seen-500.jsonl')
    fresh_meta = load_probe_meta('probe-fresh-500.jsonl')
    seen_ids_le250 = [i for i, m in seen_meta.items() if m['train_step'] <= 250]
    seen_ids_gt250 = [i for i, m in seen_meta.items() if m['train_step'] > 250]

    def load_for_tag(prefix, tag):
        if tag == 'r7.step0' and prefix in ('d1-rev-pf', 'd1-rev-vf', 'd1-ext-pf'):
            return load_eval(os.path.join(DIAG, f'{prefix}.r7.json'))
        if tag == 'r7.step0' and prefix == 'selection':
            return load_eval(os.path.join(M, 'r7-selection.json'))
        return load_eval(os.path.join(P3, f'{prefix}.{tag}.json'))

    print('\n=== FRESH & SEEN PROBES (n = 500 each, K = 4 chance 25%, exact greedy K = 8192/token chance ~0%) ===')
    fresh_res = {}
    seen_res = {}
    for label, tag, _ in models:
        fr, _ = load_for_tag('probe-fresh-500', tag)
        sn, _ = load_for_tag('probe-seen-500', tag)
        fresh_res[tag] = fr
        seen_res[tag] = sn

    r7_fr = fresh_res['r7.step0']
    r7_sn = seen_res['r7.step0']
    summary_out = {'models': {}, 'loss': loss_info, 'ckpts': ckpts}

    for label, tag, _ in models:
        fr, sn = fresh_res[tag], seen_res[tag]
        fr_ex = sum(v['exact'] for v in fr.values())
        fr_bd = sum(v['bounded'] for v in fr.values())
        fr_k4 = sum(v['k4'] for v in fr.values())
        sn_ex = sum(v['exact'] for v in sn.values())
        sn_bd = sum(v['bounded'] for v in sn.values())
        sn_k4 = sum(v['k4'] for v in sn.values())
        g_fk4, l_fk4, p_fk4 = mcn(fr, r7_fr)
        g_sk4, l_sk4, p_sk4 = mcn(sn, r7_sn)
        sn_ex_le250 = sum(sn[i]['exact'] for i in seen_ids_le250)
        sn_ex_gt250 = sum(sn[i]['exact'] for i in seen_ids_gt250)
        sn_k4_le250 = sum(sn[i]['k4'] for i in seen_ids_le250)
        sn_k4_gt250 = sum(sn[i]['k4'] for i in seen_ids_gt250)
        gap_ex = (sn_ex - fr_ex) / 5.0
        gap_k4 = (sn_k4 - fr_k4) / 5.0
        print(f'  {label:18s} | Fresh: exact {fr_ex:3d}/500 ({fr_ex/5:5.1f}%) [bounded {fr_bd}] K4 {fr_k4:3d}/500 ({fr_k4/5:5.1f}%, vs R7 +{g_fk4}/-{l_fk4} p={p_fk4:.3g})'
              f' | Seen: exact {sn_ex:3d}/500 ({sn_ex/5:5.1f}%) [bounded {sn_bd}] K4 {sn_k4:3d}/500 ({sn_k4/5:5.1f}%, vs R7 +{g_sk4}/-{l_sk4} p={p_sk4:.3g})'
              f' | Seen-Fresh: exact {gap_ex:+5.1f}pp, K4 {gap_k4:+5.1f}pp')
        print(f'                     | Seen split (step<=250 n=250 | step>250 n=250): exact {sn_ex_le250}/{sn_ex_gt250}, K4 {sn_k4_le250}/{sn_k4_gt250}')
        for cat in ('reverse', 'extract'):
            fids = [i for i in fr if fr[i]['cat'] == cat]
            sids = [i for i in sn if sn[i]['cat'] == cat]
            fex = sum(fr[i]['exact'] for i in fids)
            fk4 = sum(fr[i]['k4'] for i in fids)
            sex = sum(sn[i]['exact'] for i in sids)
            sk4 = sum(sn[i]['k4'] for i in sids)
            gf, lf, pf = mcn(fr, r7_fr, fids)
            gs, ls, ps = mcn(sn, r7_sn, sids)
            print(f'                       {cat:7s}: Fresh exact {fex:3d}/{len(fids)} ({100*fex/len(fids):5.1f}%) K4 {fk4:3d}/{len(fids)} ({100*fk4/len(fids):5.1f}% +{gf}/-{lf} p={pf:.3g})'
                  f' | Seen exact {sex:3d}/{len(sids)} ({100*sex/len(sids):5.1f}%) K4 {sk4:3d}/{len(sids)} ({100*sk4/len(sids):5.1f}% +{gs}/-{ls} p={ps:.3g})')
        summary_out['models'][tag] = {
            'fresh_exact': fr_ex, 'fresh_exact_pct': fr_ex / 5.0, 'fresh_k4': fr_k4, 'fresh_k4_pct': fr_k4 / 5.0,
            'seen_exact': sn_ex, 'seen_exact_pct': sn_ex / 5.0, 'seen_k4': sn_k4, 'seen_k4_pct': sn_k4 / 5.0,
            'seen_minus_fresh_exact_pp': gap_ex,
        }

    print('\n=== FROZEN D1 ITEMS (n = 200 per cell, K = 4, chance 25%) ===')
    for cell, cname in (('d1-rev-pf', 'Person-first reverse'), ('d1-rev-vf', 'Value-first reverse'), ('d1-ext-pf', 'Person-first extract')):
        res = {tag: load_for_tag(cell, tag)[0] for _, tag, _ in models}
        r7c = res['r7.step0']
        print(f'  {cname} ({cell}):')
        for label, tag, _ in models:
            cur = res[tag]
            k4 = sum(v['k4'] for v in cur.values())
            ex = sum(v['exact'] for v in cur.values())
            g, l, p = mcn(cur, r7c)
            print(f'    {label:18s}: K4 {k4:3d}/200 ({k4/2:5.1f}%, vs R7 +{g}/-{l} p={p:.3g}) | exact prefix {ex:3d}/200 ({ex/2:5.1f}%)')
            summary_out['models'][tag][cell] = k4

    print('\n=== MISSION SELECTION RETENTION (info only, n = 500, 100 per category, K = 4, chance 25%, +/-4.4pp overall) ===')
    sel_res = {tag: load_for_tag('selection', tag)[0] for _, tag, _ in models}
    r7s = sel_res['r7.step0']
    for cat in ('extract', 'relation', 'reverse', 'compare', 'temporal', 'ALL'):
        ids = [i for i in r7s if cat == 'ALL' or r7s[i]['cat'] == cat]
        parts = []
        for label, tag, _ in models:
            cur = sel_res[tag]
            k4 = sum(cur[i]['k4'] for i in ids)
            g, l, p = mcn(cur, r7s, ids)
            if tag == 'r7.step0':
                parts.append(f'R7 {k4}/{len(ids)}')
            else:
                parts.append(f'{tag} {k4}/{len(ids)} (+{g}/-{l} p={p:.3g})')
        print(f'  {cat:8s}: ' + ' | '.join(parts))
        for _, tag, _ in models:
            if cat == 'ALL':
                summary_out['models'][tag]['selection_all'] = sum(sel_res[tag][i]['k4'] for i in ids)

    print('\n=== PRE-REGISTERED DECISION TABLE VERDICT (judged on best arm final step-500 checkpoint) ===')
    finals = ['arm-5e-4.step500', 'arm-1e-3.step500']
    best_tag = max(finals, key=lambda t: (summary_out['models'][t]['fresh_exact'], summary_out['models'][t]['d1-rev-pf'], summary_out['models'][t]['seen_exact']))
    bm = summary_out['models'][best_tag]
    s5 = summary_out['models']['arm-5e-4.step500']['seen_exact_pct']
    s1 = summary_out['models']['arm-1e-3.step500']['seen_exact_pct']
    fr_pct = bm['fresh_exact_pct']
    sn_pct = bm['seen_exact_pct']
    gap = bm['seen_minus_fresh_exact_pp']
    d1_pf = bm['d1-rev-pf']
    arm_key = '5e-4' if best_tag.startswith('arm-5e-4') else '1e-3'
    loss_diff = loss_info[arm_key]['mean_400_500'] - loss_info[arm_key]['mean_250_350']
    loss_step_diff = loss_info[arm_key]['step500'] - loss_info[arm_key]['step250']

    print(f'  Best arm final checkpoint: {best_tag}')
    print(f'    fresh exact = {fr_pct:.1f}% ({bm["fresh_exact"]}/500), seen exact = {sn_pct:.1f}% ({bm["seen_exact"]}/500), seen - fresh = {gap:+.1f} pp, D1 rev-pf = {d1_pf}/200')
    print(f'    both arms seen exact at step 500: arm-5e-4 = {s5:.1f}%, arm-1e-3 = {s1:.1f}%')
    print(f'    best arm loss step 250 -> 500: {loss_info[arm_key]["step250"]:.3f} -> {loss_info[arm_key]["step500"]:.3f} (diff {loss_step_diff:+.3f}); mean(250..350)->mean(400..500): diff {loss_diff:+.3f}')

    if fr_pct >= 75.0 and gap <= 10.0 and d1_pf >= 120:
        outcome = 'A. Learns and transfers'
    elif fr_pct >= 75.0 and d1_pf < 120:
        outcome = 'B. Learns in-dist, fails D1'
    elif sn_pct >= 70.0 and gap > 20.0:
        outcome = 'C. Memorises'
    elif s5 < 60.0 and s1 < 60.0 and abs(loss_diff) < 0.05 and abs(loss_step_diff) < 0.08:
        outcome = 'D. Capacity or optimisation limit (pending human review of loss trajectory flatness)'
    else:
        outcome = 'E. Anything else'
    print(f'  Outcome candidate: {outcome}')
    summary_out['outcome_candidate'] = outcome
    summary_out['best_arm_final'] = best_tag

    if args.write_json:
        out_p = os.path.join(P3, 'results-summary.json')
        with open(out_p, 'w') as f:
            json.dump(summary_out, f, indent=2)
        print(f'\nWrote {out_p}')

if __name__ == '__main__':
    main()
