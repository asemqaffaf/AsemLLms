#!/usr/bin/env python3
"""Summarize saved Common Lisp R7 development evaluations; no model execution.

Run from any directory: python3 docs/results/stage2/r7-analysis/summarize.py
Inputs are qa-*.json (the qa-analysis.lisp schema) and retrieval-*.json in this
directory. No dataset answer file is opened. Outputs: summary.json and qa.svg.
All inference is exploratory: repeated development-set and checkpoint selection
invalidate a confirmatory interpretation of the unadjusted paired p-values.
"""

import argparse
from collections import defaultdict
from html import escape
import hashlib
import json
from math import comb, sqrt
from pathlib import Path


METRICS = {
    "summed_option_likelihood": "summed_option_prediction",
    "mean_token_option_likelihood": "mean_token_option_prediction",
    "letter_A_to_D": "letter_prediction",
    "gold_length_greedy_prefix_exact": "gold_length_greedy_prefix_exact",
    "bounded_free_generation_exact": "bounded_free_generation_exact",
    "bounded_free_generation_terminated_exact": "bounded_free_generation_terminated_exact",
}
CATEGORIES = ("extract", "relation", "reverse", "compare", "temporal")
KIND_METRICS = {
    "exact_prefix": "exact_prefix", "summed_option_likelihood": "summed_option_prediction",
    "mean_token_option_likelihood": "mean_token_option_prediction", "letter_A_to_D": "letter_prediction",
    "no_question_exact_prefix": "no_question_exact_prefix", "counterfactual_exact_prefix": "counterfactual_exact_prefix",
}
OLD_CONDITIONS = {
    "real": "real_prompt", "counterfactual": "counterfactual_question_same_passage",
    "no-question": "no_question", "no-passage": "no_passage",
    "mismatched": "mismatched_question_from_another_item",
    "reversed": "options_in_reverse_order",
}
CAUTIONS = [
    "Exploratory development analysis; the same 500 items have been used repeatedly.",
    "Exact two-sided McNemar p-values are unadjusted for multiple comparisons or checkpoint selection.",
    "Familiar-question conditions use the first training frame for the same structured query, retaining the original development passage/options/key; this tests one deterministic familiar wording per query type.",
    "The prefix metric checks a generated prefix of the gold token count, with no termination scoring; it is not full-answer exact match.",
    "Bounded free generation stops at EOT, a period/newline, the context boundary, or eight generated tokens. Terminated exact additionally requires an observed delimiter/EOT.",
    "Retrieval loss supervises answer tokens only, excluding the following delimiter; stopping failure alone does not establish a retrieval or name-copying failure.",
    "Retrieval seen/unseen cells contain different generated facts; their wording comparison is unpaired.",
    "No locked-test dataset is consumed by this script.",
]


def wilson(k, n, z=1.96):
    """Wilson interval; z=1.96 matches the existing Common Lisp evaluator."""
    if type(n) is not int or type(k) is not int or n < 0 or not 0 <= k <= n:
        raise ValueError("require integer counts 0 <= correct <= n")
    if not n:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    radius = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [0.0 if k == 0 else centre - radius,
            1.0 if k == n else centre + radius]


def accuracy(hits):
    hits = list(hits)
    if any(type(x) is not bool for x in hits):
        raise TypeError("correctness values must be booleans")
    n, k = len(hits), sum(hits)
    return {"n": n, "correct": k, "accuracy": k / n if n else None,
            "wilson95": wilson(k, n)}


def paired(a, b):
    """Exact two-sided equal-tail McNemar, positive change favouring B."""
    a, b = list(a), list(b)
    if len(a) != len(b) or any(type(x) is not bool for x in a + b):
        raise ValueError("paired correctness arrays must have equal lengths")
    gained = sum(not x and y for x, y in zip(a, b))
    lost = sum(x and not y for x, y in zip(a, b))
    n, discordant = len(a), gained + lost
    probability = min(1.0, 2 * sum(comb(discordant, k)
                      for k in range(min(gained, lost) + 1)) / 2 ** discordant)
    return {"n": n, "first_correct": sum(a), "second_correct": sum(b),
            "both_correct": sum(x and y for x, y in zip(a, b)),
            "both_wrong": sum(not x and not y for x, y in zip(a, b)),
            "gained": gained, "lost": lost, "discordant": discordant,
            "change": (gained - lost) / n if n else None,
            "mcnemar_exact_two_sided_p_unadjusted": probability}


def hit(row, metric):
    value = row[METRICS[metric]]
    if metric in ("gold_length_greedy_prefix_exact", "bounded_free_generation_exact", "bounded_free_generation_terminated_exact"):
        if type(value) is not bool:
            raise ValueError("generation exact metrics must be JSON booleans")
        return value
    return value == row["key"]


def describe_rows(rows):
    out = {}
    subtypes = sorted({row["category"] + "/" + row["subtype"] for row in rows})
    for metric in METRICS:
        out[metric] = accuracy(hit(row, metric) for row in rows)
        out[metric]["by_category"] = {
            cat: accuracy(hit(row, metric) for row in rows if row["category"] == cat)
            for cat in CATEGORIES
        }
        out[metric]["by_subtype"] = {
            subtype: accuracy(hit(row, metric) for row in rows
                              if row["category"] + "/" + row["subtype"] == subtype)
            for subtype in subtypes
        }
    return out


def index_rows(rows):
    out = {row["id"]: row for row in rows}
    if len(out) != len(rows):
        raise ValueError("duplicate item IDs")
    return out


def compare_rows(first, second, same_key=True):
    a, b = index_rows(first), index_rows(second)
    if a.keys() != b.keys():
        raise ValueError("paired runs do not contain the same item IDs")
    ids = sorted(a)
    for ident in ids:
        if (a[ident]["category"], a[ident]["gold_text"]) != (b[ident]["category"], b[ident]["gold_text"]):
            raise ValueError("paired runs changed category or gold text: " + ident)
        if same_key and a[ident]["key"] != b[ident]["key"]:
            raise ValueError("paired runs changed the gold option index: " + ident)
    def prediction(row, metric, remap=False):
        if metric == "gold_length_greedy_prefix_exact":
            return row["generated_token_ids"]
        if metric.startswith("bounded_free_generation"):
            return row["bounded_generated_token_ids"]
        value = row[METRICS[metric]]
        return 3 - value if remap else value

    return {
        metric: dict(
            paired((hit(a[i], metric) for i in ids), (hit(b[i], metric) for i in ids)),
            by_category={cat: paired(
                (hit(a[i], metric) for i in ids if a[i]["category"] == cat),
                (hit(b[i], metric) for i in ids if a[i]["category"] == cat)) for cat in CATEGORIES},
            identical_predictions=sum(prediction(a[i], metric) == prediction(b[i], metric, remap=not same_key) for i in ids),
            prediction_comparison=("same option after mapping reversed indices back" if not same_key else "same predicted output"),
        ) for metric in METRICS
    }


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_analysis(path):
    if path.name in ("test.jsonl", "test.json") or "locked-test" in path.name:
        raise ValueError("refusing locked-test input")
    obj = json.loads(path.read_text())
    if obj.get("locked_test") is True:
        raise ValueError("refusing locked-test analysis")
    if "conditions" in obj and obj.get("locked_test") is not False:
        raise ValueError("QA analysis must explicitly identify itself as development-only")
    if Path(str(obj.get("items", ""))).name == "test.jsonl":
        raise ValueError("refusing locked-test provenance")
    return obj


def model_label(obj):
    checkpoint = Path(obj["checkpoint"])
    run = checkpoint.parent.name.split("-")[0].upper()
    return run + " " + checkpoint.stem


def verify_saved_summaries(obj):
    for condition, cell in obj["conditions"].items():
        computed = describe_rows(cell["items"])
        for metric in METRICS:
            saved = cell["summary"][metric]
            current = computed[metric]
            if (saved["n"], saved["correct"]) != (current["n"], current["correct"]):
                raise ValueError(f"Saved summary disagrees with per-item records: {condition}/{metric}")
            if current["n"] and abs(saved["accuracy"] - current["accuracy"]) > 1e-12:
                raise ValueError(f"Saved accuracy disagrees with per-item records: {condition}/{metric}")
            if current["n"] and any(abs(saved[key] - current["wilson95"][i]) > 1e-10
                                    for i, key in enumerate(("ci95_low", "ci95_high"))):
                raise ValueError(f"Saved Wilson interval differs: {condition}/{metric}")
            for cat in CATEGORIES:
                a, b = saved["by_category"][cat], current["by_category"][cat]
                if (a["n"], a["correct"]) != (b["n"], b["correct"]):
                    raise ValueError(f"Saved category disagrees with per-item records: {condition}/{metric}/{cat}")


def copy_breakdown(rows):
    groups = defaultdict(list)
    for row in rows:
        length = row["option_token_lengths"][row["key"]]
        groups[str(length)].append(row)
    return {key: describe_rows(value) for key, value in sorted(groups.items(), key=lambda x: int(x[0]))}


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def copy_errors(rows):
    """Describe decoding errors without treating nearest strings as correct."""
    prefix_misses = [row for row in rows if not row["gold_length_greedy_prefix_exact"]]
    retrieved_misses = [row for row in prefix_misses if hit(row, "summed_option_likelihood")]
    examples = [
        {"id": row["id"], "category": row["category"], "gold": row["gold_text"],
         "generated_prefix": row["generated_text"],
         "trimmed_character_edit_distance": edit_distance(row["gold_text"].strip(), row["generated_text"].strip()),
         "gold_tokens": row["option_token_lengths"][row["key"]],
         "summed_option_correct": hit(row, "summed_option_likelihood")}
        for row in prefix_misses
    ]
    return {"prefix_miss_n": len(prefix_misses),
            "summed_option_correct_but_prefix_wrong_n": len(retrieved_misses),
            "summed_option_correct_but_prefix_wrong_by_category": {
                cat: sum(row["category"] == cat for row in retrieved_misses) for cat in CATEGORIES},
            "all_prefix_errors_in_input_order": examples,
            "note": "Edit distance is descriptive only; no accuracy metric gives credit for approximate spelling."}


def termination_breakdown(rows):
    reasons = defaultdict(list)
    for row in rows:
        reasons[row["generation_termination_reason"]].append(row)
    return {reason: {"n": len(group),
                     "exact": sum(row["bounded_free_generation_exact"] for row in group),
                     "terminated_exact": sum(row["bounded_free_generation_terminated_exact"] for row in group)}
            for reason, group in sorted(reasons.items())}


def counterfactual_consistency(real, counterfactual):
    a, b = index_rows(real), index_rows(counterfactual)
    if a.keys() != b.keys():
        raise ValueError("counterfactual IDs differ")
    result = {"n": len(a), "gold_answer_changes": sum(a[i]["key"] != b[i]["key"] for i in a), "metrics": {}}
    for metric, field in METRICS.items():
        records = []
        for ident in a:
            x, y = a[ident], b[ident]
            if metric == "gold_length_greedy_prefix_exact":
                output_x, output_y = x["generated_token_ids"], y["generated_token_ids"]
            elif metric.startswith("bounded_free_generation"):
                output_x, output_y = x["bounded_generated_token_ids"], y["bounded_generated_token_ids"]
            else:
                output_x, output_y = x[field], y[field]
            records.append((x["category"], hit(x, metric) and hit(y, metric), output_x != output_y))
        result["metrics"][metric] = {
            "both_correct": sum(row[1] for row in records), "prediction_changed": sum(row[2] for row in records),
            "by_category": {cat: {"n": sum(row[0] == cat for row in records),
                                    "both_correct": sum(row[1] for row in records if row[0] == cat),
                                    "prediction_changed": sum(row[2] for row in records if row[0] == cat)}
                            for cat in CATEGORIES}}
        if metric == "gold_length_greedy_prefix_exact":
            result["metrics"][metric]["note"] = "Prediction-change count can reflect different gold prefix lengths; use bounded outputs or option predictions for conditioning."
    return result


def reproduction(obj, root):
    """Compare saved legacy counts when an exact checkpoint SHA was evaluated."""
    out = {}
    paths = list((root / "docs/results/stage2/retrieval").glob("qa-dev-r[567]-*.json"))
    paths += list(Path(__file__).resolve().parent.glob("qa-dev-*-original-*.json"))
    for path in sorted(paths):
        old = read_analysis(path)
        if old["checkpoint_sha256"] != obj["checkpoint_sha256"]:
            continue
        if old["items_sha256"] != obj.get("items_sha256"):
            raise ValueError("historical reproduction uses a different dataset hash")
        metric = "summed_option_likelihood" if old["mode"] == "text" else "letter_A_to_D"
        for new_cond, old_cond in OLD_CONDITIONS.items():
            if new_cond not in obj["conditions"]:
                continue
            current = describe_rows(obj["conditions"][new_cond]["items"])[metric]
            reference = old[old_cond]
            out[path.name + ":" + new_cond] = {
                "reference_json_sha256": sha256(path),
                "metric": metric, "old_correct": reference["correct"],
                "new_correct": current["correct"], "old_n": reference["n"],
                "new_n": current["n"], "counts_identical":
                (current["correct"], current["n"]) == (reference["correct"], reference["n"]),
            }
    return out


def retrieval_summary(obj):
    results = {}
    for wording in ("seen", "unseen"):
        cells = [value for key, value in obj.items() if key.startswith("L") and key.endswith("/" + wording)]
        pooled = {}
        for metric in ("real", "option_likelihood", "option_likelihood_mean_per_token", "letter_A_to_D"):
            available = [cell[metric] for cell in cells if metric in cell]
            n = sum(cell["n"] for cell in available)
            k = sum(cell["correct"] for cell in available)
            pooled[metric] = {"n": n, "correct": k, "accuracy": k / n if n else None,
                              "wilson95": wilson(k, n)}
        results[wording] = pooled
    results["cells"] = {k: v for k, v in obj.items() if k.startswith("L")}
    return results


def kind_hit(row, metric):
    value = row[KIND_METRICS[metric]]
    if value is None:
        return None
    if metric.endswith("exact_prefix") or metric == "exact_prefix":
        if type(value) is not bool:
            raise ValueError("kind exact-prefix metrics must be boolean or null")
        return value
    return value == row["key"]


def describe_kind_rows(rows):
    return {metric: accuracy(value for row in rows if (value := kind_hit(row, metric)) is not None)
            for metric in KIND_METRICS}


def kind_summary(obj):
    result = {"checkpoint": obj["checkpoint"], "checkpoint_sha256": obj["checkpoint_sha256"],
              "dataset_sha256": obj["dataset_sha256"], "cells": {}}
    for wording, cell in obj["cells"].items():
        rows = cell["items"]
        computed = describe_kind_rows(rows)
        by_kind = {kind: describe_kind_rows([r for r in rows if r["kind"] == kind])
                   for kind in sorted({r["kind"] for r in rows})}
        for expected, actual in [(cell["summary"], computed)] + [(cell["by_kind"][kind], stats) for kind, stats in by_kind.items()]:
            for metric in KIND_METRICS:
                if (expected[metric]["n"], expected[metric]["correct"]) != (actual[metric]["n"], actual[metric]["correct"]):
                    raise ValueError("Kind evaluator summary disagrees with per-item records")
        result["cells"][wording] = {"summary": computed, "by_kind": by_kind}
    return result


def compare_kind_runs(a, b):
    if a["dataset_sha256"] != b["dataset_sha256"]:
        raise ValueError("kind runs do not use the same snapshot hash")
    result = {}
    for wording in a["cells"]:
        first, second = index_rows(a["cells"][wording]["items"]), index_rows(b["cells"][wording]["items"])
        if first.keys() != second.keys():
            raise ValueError("kind run IDs differ")
        for ident in first:
            if (first[ident]["kind"], first[ident]["key"]) != (second[ident]["kind"], second[ident]["key"]):
                raise ValueError("kind run targets differ")
        groups = {"all": sorted(first)}
        groups.update({kind: [i for i in sorted(first) if first[i]["kind"] == kind]
                       for kind in sorted({r["kind"] for r in first.values()})})
        result[wording] = {}
        for group, ids in groups.items():
            result[wording][group] = {}
            for metric in KIND_METRICS:
                eligible = [i for i in ids if kind_hit(first[i], metric) is not None and kind_hit(second[i], metric) is not None]
                result[wording][group][metric] = paired((kind_hit(first[i], metric) for i in eligible),
                                                       (kind_hit(second[i], metric) for i in eligible))
    return result


def chart(summary, path):
    labels = sorted(summary["qa"])
    width, height = 1000, 590
    plot_x, plot_y, plot_w, plot_h = 75, 95, 860, 355
    colors = ("#4263eb", "#099268", "#f08c00", "#ae3ec9", "#c92a2a")
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
           '<title>AsemLLM QA development accuracy and Wilson intervals</title>',
           '<rect width="100%" height="100%" fill="#fcfcfb"/>',
           '<g font-family="system-ui,sans-serif" fill="#202124">',
           '<text x="40" y="33" font-size="21">AsemLLM: identical QA development items</text>',
           '<text x="40" y="59" font-size="13">Summed option likelihood; n=100 per category; error bars: 95% Wilson intervals</text>']
    for tick in range(0, 101, 20):
        y = plot_y + plot_h * (1 - tick / 100)
        svg += [f'<line x1="{plot_x}" y1="{y}" x2="{plot_x + plot_w}" y2="{y}" stroke="#deded9"/>',
                f'<text x="{plot_x - 12}" y="{y + 4}" text-anchor="end" font-size="12">{tick}%</text>']
    chance_y = plot_y + plot_h * .75
    svg += [f'<line x1="{plot_x}" y1="{chance_y}" x2="{plot_x + plot_w}" y2="{chance_y}" stroke="#888" stroke-dasharray="5 4"/>']
    group_w = plot_w / len(CATEGORIES)
    bar_w = min(23, group_w * .7 / max(1, len(labels)))
    for cat_no, cat in enumerate(CATEGORIES):
        centre = plot_x + group_w * (cat_no + .5)
        for model_no, label in enumerate(labels):
            row = summary["qa"][label]["conditions"]["real"]["summed_option_likelihood"]["by_category"][cat]
            if not row["n"]:
                continue
            x = centre + (model_no - (len(labels) - 1) / 2) * (bar_w + 5)
            y = plot_y + plot_h * (1 - row["accuracy"])
            lo, hi = row["wilson95"]
            ci_top, ci_bottom = plot_y + plot_h * (1 - hi), plot_y + plot_h * (1 - lo)
            svg += [f'<rect x="{x - bar_w / 2}" y="{y}" width="{bar_w}" height="{plot_y + plot_h - y}" fill="{colors[model_no % len(colors)]}"/>',
                    f'<path d="M{x},{ci_top}V{ci_bottom} M{x-4},{ci_top}H{x+4} M{x-4},{ci_bottom}H{x+4}" stroke="#333" fill="none"/>']
        svg += [f'<text x="{centre}" y="{plot_y + plot_h + 24}" text-anchor="middle" font-size="13">{escape(cat)}</text>']
    for model_no, label in enumerate(labels):
        x = 65 + model_no * (880 / max(1, len(labels)))
        svg += [f'<rect x="{x}" y="502" width="13" height="13" fill="{colors[model_no % len(colors)]}"/>',
                f'<text x="{x+20}" y="513" font-size="13">{escape(label)}</text>']
    svg += ['<text x="40" y="554" font-size="12">Exploratory: repeated use of this development set and checkpoint selection are not covered by these intervals.</text>',
            '<text x="40" y="575" font-size="12">Dashed line: four-choice guessing accuracy (25%). Locked test remains unused.</text>', '</g></svg>']
    path.write_text("\n".join(svg) + "\n")


def wording_chart(summary, path):
    labels = sorted(summary["retrieval"])
    if not labels:
        return
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="495" viewBox="0 0 1100 495" role="img">',
           '<title>Mixed-task retrieval with familiar and unfamiliar wording</title>',
           '<rect width="100%" height="100%" fill="#fcfcfb"/>',
           '<g font-family="system-ui,sans-serif" fill="#202124">',
           '<text x="30" y="32" font-size="21">Mixed-task retrieval: familiar vs unfamiliar wording</text>',
           '<text x="30" y="57" font-size="13">Same reverse + reasoning task mixture for every checkpoint; pooled levels and entity pools; 95% Wilson intervals</text>']
    metrics = (("real", "Greedy gold-length prefix"), ("option_likelihood", "Option likelihood"), ("letter_A_to_D", "A–D letter logits"))
    top, bottom, plot_h = 108, 375, 267
    for panel, (metric, title) in enumerate(metrics):
        left, plot_w = 60 + panel * 365, 285
        svg += [f'<text x="{left + plot_w/2}" y="88" text-anchor="middle" font-size="15">{title}</text>']
        for tick in range(0, 101, 25):
            y = bottom - plot_h * tick / 100
            svg += [f'<line x1="{left}" x2="{left+plot_w}" y1="{y}" y2="{y}" stroke="#deded9"/>',
                    f'<text x="{left-8}" y="{y+4}" text-anchor="end" font-size="11">{tick}%</text>']
        group_w = plot_w / len(labels)
        for model_no, label in enumerate(labels):
            centre = left + group_w * (model_no+.5)
            for w_no, (wording, color) in enumerate((("seen", "#099268"), ("unseen", "#e67700"))):
                row = summary["retrieval"][label][wording][metric]
                if not row["n"]:
                    continue
                x, bw = centre + (w_no - .5) * 23, 18
                y = bottom - plot_h * row["accuracy"]
                lo, hi = row["wilson95"]
                y_hi, y_lo = bottom - plot_h * hi, bottom - plot_h * lo
                svg += [f'<rect x="{x-bw/2}" y="{y}" width="{bw}" height="{bottom-y}" fill="{color}"/>',
                        f'<path d="M{x},{y_hi}V{y_lo} M{x-3},{y_hi}H{x+3} M{x-3},{y_lo}H{x+3}" stroke="#333" fill="none"/>']
            run, checkpoint = label.split(" ", 1)
            svg += [f'<text x="{centre}" y="{bottom+20}" text-anchor="middle" font-size="11">{escape(run)}</text>',
                    f'<text x="{centre}" y="{bottom+34}" text-anchor="middle" font-size="11">{escape(checkpoint)}</text>']
    svg += ['<rect x="350" y="433" width="13" height="13" fill="#099268"/><text x="370" y="444" font-size="13">Familiar wording</text>',
            '<rect x="555" y="433" width="13" height="13" fill="#e67700"/><text x="575" y="444" font-size="13">Unfamiliar wording</text>',
            '<text x="30" y="475" font-size="12">Unpaired generated facts across wording cells; these results are development diagnostics, not a locked-test result.</text>', '</g></svg>']
    path.write_text("\n".join(svg) + "\n")


def reasoning_chart(summary, path):
    labels = sorted(summary["retrieval_by_kind"])
    if not labels:
        return
    first_cells = summary["retrieval_by_kind"][labels[0]]["cells"]
    counts = {(kind, wording): first_cells[wording]["by_kind"][kind]["exact_prefix"]["n"]
              for kind in ("cmp", "tmp") for wording in ("seen", "unseen")}
    colors = ("#099268", "#f08c00", "#ae3ec9", "#4263eb")
    groups = (("cmp", "seen", "Comparison", "familiar"), ("cmp", "unseen", "Comparison", "unfamiliar"),
              ("tmp", "seen", "Temporal", "familiar"), ("tmp", "unseen", "Temporal", "unfamiliar"))
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="510" viewBox="0 0 1100 510" role="img">',
           '<title>Comparison and temporal retrieval by wording</title>',
           '<rect width="100%" height="100%" fill="#fcfcfb"/>',
           '<g font-family="system-ui,sans-serif" fill="#202124">',
           '<text x="30" y="32" font-size="21">Comparison and temporal retrieval by wording</text>',
           '<text x="30" y="57" font-size="13">Fresh level-3 probes with unseen entities; identical probe snapshot for all checkpoints; 95% Wilson intervals</text>']
    bottom, plot_h = 365, 255
    for panel, (metric, title) in enumerate((("exact_prefix", "Greedy gold-length prefix"), ("summed_option_likelihood", "Option likelihood"))):
        left, plot_w = 65 + panel * 550, 450
        svg += [f'<text x="{left+plot_w/2}" y="88" text-anchor="middle" font-size="15">{title}</text>']
        for tick in range(0, 101, 25):
            y = bottom - plot_h * tick / 100
            svg += [f'<line x1="{left}" x2="{left+plot_w}" y1="{y}" y2="{y}" stroke="#deded9"/>',
                    f'<text x="{left-8}" y="{y+4}" text-anchor="end" font-size="11">{tick}%</text>']
        if panel == 1:
            y = bottom - plot_h * .25
            svg += [f'<line x1="{left}" x2="{left+plot_w}" y1="{y}" y2="{y}" stroke="#777" stroke-dasharray="5 4"/>']
        for group_no, (kind, wording, kind_label, wording_label) in enumerate(groups):
            centre = left + plot_w / 4 * (group_no + .5)
            for model_no, label in enumerate(labels):
                row = summary["retrieval_by_kind"][label]["cells"][wording]["by_kind"][kind][metric]
                x = centre + (model_no - (len(labels)-1)/2) * 22
                y = bottom - plot_h * row["accuracy"]
                lo, hi = row["wilson95"]
                y_hi, y_lo = bottom - plot_h * hi, bottom - plot_h * lo
                svg += [f'<rect x="{x-8}" y="{y}" width="16" height="{bottom-y}" fill="{colors[model_no % len(colors)]}"/>',
                        f'<path d="M{x},{y_hi}V{y_lo} M{x-3},{y_hi}H{x+3} M{x-3},{y_lo}H{x+3}" stroke="#333" fill="none"/>']
            svg += [f'<text x="{centre}" y="{bottom+20}" text-anchor="middle" font-size="12">{kind_label}</text>',
                    f'<text x="{centre}" y="{bottom+35}" text-anchor="middle" font-size="11">{wording_label}</text>']
    for i, label in enumerate(labels):
        x = 260 + i * 220
        svg += [f'<rect x="{x}" y="430" width="13" height="13" fill="{colors[i % len(colors)]}"/>',
                f'<text x="{x+20}" y="441" font-size="13">{escape(label)}</text>']
    count_note = (f'Comparison: n={counts["cmp", "seen"]}/{counts["cmp", "unseen"]} familiar/unfamiliar. '
                  f'Temporal: n={counts["tmp", "seen"]}/{counts["tmp", "unseen"]}. Wording cells contain different facts.')
    svg += [f'<text x="30" y="475" font-size="12">{count_note}</text>',
            '<text x="30" y="495" font-size="12">Dashed line: 25% option guessing. Prefix generation has no fixed guessing floor. Exploratory development evidence.</text>', '</g></svg>']
    path.write_text("\n".join(svg) + "\n")


def self_test():
    assert wilson(0, 0) is None
    assert wilson(0, 100)[0] == 0 and wilson(100, 100)[1] == 1
    assert abs(wilson(25, 100)[0] - 0.1754509400372426) < 1e-12
    assert abs(wilson(229, 500)[0] - 0.41481382806562594) < 1e-12
    assert paired([], [])["mcnemar_exact_two_sided_p_unadjusted"] == 1
    assert paired([False] * 6, [True] * 6)["mcnemar_exact_two_sided_p_unadjusted"] == 0.03125
    test = paired([False] * 7 + [True] * 2, [True] * 7 + [False] * 2)
    assert (test["gained"], test["lost"], test["change"]) == (7, 2, 5 / 9)
    assert test["mcnemar_exact_two_sided_p_unadjusted"] == 0.1796875
    assert paired([True, False], [False, True])["mcnemar_exact_two_sided_p_unadjusted"] == 1
    assert edit_distance("", "abc") == 3 and edit_distance("kitten", "sitting") == 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        print("Statistical checks passed.")
        return
    root = Path(__file__).resolve().parents[4]
    summary = {"cautions": CAUTIONS, "qa": {}, "retrieval": {}, "paired_qa": {}, "inputs": {},
               "retrieval_by_kind": {}, "paired_kind_runs": {},
               "summary_script_sha256": sha256(Path(__file__)),
               "statistics": {"accuracy_interval": "Wilson score, z=1.96", "paired_test": "Exact two-sided equal-tail McNemar",
                              "multiple_comparison_adjustment": "none; exploratory development evidence"}}
    raw = {}
    for path in sorted(args.input_dir.glob("qa-*.json")):
        obj = read_analysis(path)
        if "conditions" not in obj:
            continue
        verify_saved_summaries(obj)
        label = model_label(obj)
        if label in raw:
            raise ValueError("duplicate QA model label: " + label)
        raw[label] = obj
        summary["inputs"][path.name] = sha256(path)
        real = obj["conditions"]["real"]["items"]
        conditions = {name: describe_rows(cell["items"]) for name, cell in obj["conditions"].items()}
        mismatch = obj["conditions"].get("mismatched", {}).get("items", [])
        unchanged = [r for r in mismatch if r["mismatched_question_literally_unchanged"]]
        result = {"checkpoint": obj["checkpoint"], "checkpoint_sha256": obj["checkpoint_sha256"],
                  "items_sha256": obj.get("items_sha256"), "conditions": conditions,
                  "copy_by_gold_token_count": copy_breakdown(real),
                  "copy_errors": copy_errors(real),
                  "generation_termination": termination_breakdown(real),
                  "historical_reproduction": reproduction(obj, root),
                  "mismatched_question_literally_unchanged": {
                      "n": len(unchanged), "total": len(mismatch),
                      "by_category": {cat: sum(r["category"] == cat for r in unchanged) for cat in CATEGORIES},
                  }, "mismatched_literally_changed_only": describe_rows([
                      r for r in mismatch if not r["mismatched_question_literally_unchanged"]]), "paired_wording": {}}
        if "counterfactual" in obj["conditions"]:
            result["counterfactual_consistency"] = counterfactual_consistency(real, obj["conditions"]["counterfactual"]["items"])
        for familiar, original in (("familiar-question-real", "real"),
                                    ("familiar-question-counterfactual", "counterfactual")):
            if familiar in obj["conditions"] and original in obj["conditions"]:
                result["paired_wording"][original + " -> " + familiar] = compare_rows(
                    obj["conditions"][original]["items"], obj["conditions"][familiar]["items"])
        if "reversed" in obj["conditions"]:
            result["reversed_option_parity"] = compare_rows(real, obj["conditions"]["reversed"]["items"], same_key=False)
        summary["qa"][label] = result
    labels = sorted(raw)
    for i, first in enumerate(labels):
        for second in labels[i + 1:]:
            if raw[first].get("items_sha256") != raw[second].get("items_sha256"):
                raise ValueError("QA dataset hashes differ")
            summary["paired_qa"][first + " -> " + second] = compare_rows(
                raw[first]["conditions"]["real"]["items"], raw[second]["conditions"]["real"]["items"])
    for path in sorted(args.input_dir.glob("retrieval-*.json")):
        obj = read_analysis(path)
        if not any(key.startswith("L") and isinstance(value, dict) and "real" in value for key, value in obj.items()):
            continue
        label = model_label(obj)
        summary["inputs"][path.name] = sha256(path)
        summary["retrieval"][label] = {"checkpoint": obj["checkpoint"], "checkpoint_sha256": obj["checkpoint_sha256"],
                                            **retrieval_summary(obj)}
    pilot = args.input_dir / "pilot-summary.json"
    if pilot.exists():
        summary["pilot"] = read_analysis(pilot)
        summary["inputs"][pilot.name] = sha256(pilot)
    kinds = {}
    for path in sorted(args.input_dir.glob("kinds-*.json")):
        obj = read_analysis(path)
        if "cells" not in obj:
            continue
        label = model_label(obj)
        kinds[label] = obj
        summary["retrieval_by_kind"][label] = kind_summary(obj)
        summary["inputs"][path.name] = sha256(path)
    kind_labels = sorted(kinds)
    for i, first in enumerate(kind_labels):
        for second in kind_labels[i + 1:]:
            summary["paired_kind_runs"][first + " -> " + second] = compare_kind_runs(kinds[first], kinds[second])
    if not raw:
        raise ValueError("no per-item QA analysis inputs found")
    (args.input_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    chart(summary, args.input_dir / "qa.svg")
    wording_chart(summary, args.input_dir / "wording.svg")
    reasoning_chart(summary, args.input_dir / "reasoning.svg")
    print(f"Summarized {len(raw)} QA checkpoints and {len(summary['retrieval'])} retrieval evaluations.")


if __name__ == "__main__":
    main()
