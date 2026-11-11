#!/usr/bin/env python3
"""Recompute Phase 2 paired statistics without model inference or locked-test access.

Run from the repository root. Output goes to stdout; existing artifacts are read only.
This checks saved predictions, not the correctness of the model's forward pass.
"""
import hashlib
import json
import math
from pathlib import Path

BASE = Path('docs/results/stage2/phase2-paraphrase-pilot')
METRICS = {
    'summed_option_likelihood': 'summed_option_prediction',
    'mean_token_option_likelihood': 'mean_token_option_prediction',
    'letter_A_to_D': 'letter_prediction',
    'gold_length_greedy_prefix_exact': 'gold_length_greedy_prefix_exact',
    'bounded_free_generation_exact': 'bounded_free_generation_exact',
    'bounded_free_generation_terminated_exact': 'bounded_free_generation_terminated_exact',
}


def load(path):
    return json.loads(path.read_text())


def hit(row, field):
    return bool(row[field]) if field.endswith('_exact') else row[field] == row['key']


def paired(a, b, field):
    assert len(a) == len(b) and a
    for x, y in zip(a, b):
        for key in ('id', 'category', 'subtype', 'key', 'gold_text', 'question'):
            assert x[key] == y[key], (key, x['id'])
    ha, hb = [[hit(x, field) for x in rows] for rows in (a, b)]
    gained = sum(not x and y for x, y in zip(ha, hb))
    lost = sum(x and not y for x, y in zip(ha, hb))
    n, discordant = len(a), gained + lost
    delta = (gained - lost) / n
    p = min(1., 2 * sum(math.comb(discordant, k)
                        for k in range(min(gained, lost) + 1)) / 2**discordant)
    # Unbiased sample-variance SE; legacy uses population variance / n.
    se = math.sqrt((discordant - n * delta**2) / (n * (n - 1))) if n > 1 else None
    return dict(n=n, first_correct=sum(ha), second_correct=sum(hb),
                gained=gained, lost=lost, diff=delta,
                mcnemar_exact_two_sided_p=p, unbiased_se=se)


report = {'scope': 'saved Phase 2 predictions and explicit provenance paths only',
          'locked_test_accessed': False, 'comparisons_checked': 0, 'datasets': {}}
for dataset in ('qa-dev', 'l3-new', 'selection'):
    arms = [load(BASE / f'wording-2x2-{dataset}-{arm}-step300.json')
            for arm in ('control', 'treatment')]
    reference = load(BASE / f'compare-{dataset}-control-vs-treatment-step300.json')
    dataset_report = {}
    for cell_type in ('cells', 'counterfactual_cells'):
        for cell in arms[0]['policies']['seeded'][cell_type]:
            rows = [a['policies']['seeded'][cell_type][cell]['items'] for a in arms]
            cell_report = {}
            for metric, field in METRICS.items():
                ref = reference[cell_type][cell]['paired_second_minus_first'][metric]
                groups = [('all', rows[0], rows[1], ref)] + [
                    (cat, [r for r in rows[0] if r['category'] == cat],
                     [r for r in rows[1] if r['category'] == cat], values)
                    for cat, values in ref['by_category'].items()]
                metric_report = {}
                for group, a, b, target in groups:
                    result = paired(a, b, field)
                    for k in ('n', 'first_correct', 'second_correct', 'gained', 'lost'):
                        assert result[k] == target[k], (dataset, cell, metric, group, k)
                    for k in ('diff', 'mcnemar_exact_two_sided_p'):
                        assert abs(result[k] - target[k]) < 1e-10, (dataset, cell, metric, group, k)
                    report['comparisons_checked'] += 1
                    metric_report[group] = result
                cell_report[metric] = metric_report
            dataset_report[f'{cell_type}:{cell}'] = cell_report
    report['datasets'][dataset] = dataset_report

expected = {
    'lisp/src/retrieval.lisp': '8e47f3b1f2280f10f6e9416349f85f0e255f6d3a063571022521f3d6b717c0b6',
    'lisp/src/bpe.lisp': '25523805c53ef53d36ed71625a41140c3a5ee3246bbc96555a48997b0cfbb65f',
    'data/runs/retrieval/r7-L123-aug-opts-rev-rsn/last.ckpt': '33a5c24521dfb1b28b456b8344bfd0869b84fe56aa622191b8f6bfa71cc42aa1',
    'data/runs/retrieval/p2-paraphrase-pilot/control-data.sexp': '5a16eba9aacca54062490f5c9d81afc14124b8828b92af3d27f671a4a2f658d2',
    'data/runs/retrieval/p2-paraphrase-pilot/treatment-data.sexp': 'addd345a8a2d32195794c5d8adf27a7c28b4dad78c70d15ebd9708108bf6138a',
    'data/runs/retrieval/p2-paraphrase-pilot/selection-data.sexp': 'c2bd5691c3b52c2d1ccbb31b7d09abfc1f385064633b612d2b513044a8fefb8b',
    'data/runs/retrieval/p2-paraphrase-pilot/control/last.ckpt': '435b5d7c2eb1d8afecd67e0c9eb86709aecb966b60fbe76bbc40aee5ae959a84',
    'data/runs/retrieval/p2-paraphrase-pilot/treatment/last.ckpt': 'bce9f94765e4503bedf2674a40e066c13780709425441e9c7a7de2d7e9d8833c',
}
for path, digest in expected.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
report['verified_sha256'] = expected
print(json.dumps(report, indent=2, sort_keys=True))
