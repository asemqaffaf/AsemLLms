#!/usr/bin/env python3
"""Summarize Pilot 4 audit_raw.json together with Pilot 1 and Pilot 2 audits."""
import json
from collections import defaultdict

BASE = "docs/results/stage2/mission-20261111"
d = json.load(open(f"{BASE}/pilot4/audit_raw.json"))
toks = d["vocab_tokens"]

print("=== PART A1: 8,192 BPE VOCAB COVERAGE BY SHAPE ===")
by_shape = defaultdict(list)
for t in toks:
    by_shape[t["shape"]].append(t)

print(
    f"{'Shape':20s} {'N':>5s} {'FW>0':>5s} {'R7all>0':>7s} {'R7ans>0':>7s} "
    f"{'R7ans>=100':>10s} {'P3all>0':>7s} {'P3ans>0':>7s} {'P3ans>=10':>9s} "
    f"{'P3ans>=100':>10s} {'BN>0':>5s} {'D1>0':>5s}"
)
for sh in sorted(by_shape, key=lambda k: -len(by_shape[k])):
    xs = by_shape[sh]
    print(
        f"{sh:20s} {len(xs):5d} "
        f"{sum(t['fw_train'] > 0 for t in xs):5d} "
        f"{sum(t['r7_ret_all'] > 0 for t in xs):7d} "
        f"{sum(t['r7_ret_ans'] > 0 for t in xs):7d} "
        f"{sum(t['r7_ret_ans'] >= 100 for t in xs):10d} "
        f"{sum(t['p3_all'] > 0 for t in xs):7d} "
        f"{sum(t['p3_ans'] > 0 for t in xs):7d} "
        f"{sum(t['p3_ans'] >= 10 for t in xs):9d} "
        f"{sum(t['p3_ans'] >= 100 for t in xs):10d} "
        f"{sum(t['broad_names_tok_types'] > 0 for t in xs):5d} "
        f"{sum(t['d1_names_tok_types'] > 0 for t in xs):5d}"
    )

print("\n=== PART A2: R7 COPY-MIX COVERAGE (15,000 seqs, lo=1..8191) ===")
r7_copy = [t["r7_copy"] for t in toks]
print(
    f"id=0 count={r7_copy[0]}; ids 1..8191: zero_count={sum(x == 0 for x in r7_copy[1:])}, "
    f"min={min(r7_copy[1:])}, max={max(r7_copy[1:])}, mean={sum(r7_copy[1:]) / 8191:.2f}"
)

print("\n=== PART A3: FINEWEB-PRESENT TOKENS MISSING / RARE IN RETRIEVAL ===")
for sh in ("space_title_alpha", "bare_lower_alpha", "space_lower_alpha", "bare_title_alpha"):
    xs = [t for t in by_shape[sh] if t["fw_train"] > 0]
    r7_all0 = [t for t in xs if t["r7_ret_all"] == 0]
    r7_ans0 = [t for t in xs if t["r7_ret_ans"] == 0]
    p3_all0 = [t for t in xs if t["p3_all"] == 0]
    p3_ans0 = [t for t in xs if t["p3_ans"] == 0]
    p3_lt10 = [t for t in xs if t["p3_ans"] < 10]
    both0 = [t for t in xs if t["r7_ret_all"] == 0 and t["p3_all"] == 0]
    print(
        f"{sh:20s}: fw>0={len(xs):4d} | r7_all==0:{len(r7_all0):4d} r7_ans==0:{len(r7_ans0):4d} | "
        f"p3_all==0:{len(p3_all0):4d} p3_ans==0:{len(p3_ans0):4d} p3_ans<10:{len(p3_lt10):4d} | both_all==0:{len(both0):4d}"
    )

print("\n=== PART A4: ENTITY POOL STATS ===")
for pname, ps in d["pools"].items():
    n = ps["n_strings"]
    ic = ps["initial_char_counts"]
    vowels = sum(ic.get(v, 0) for v in "AEIOUaeiou")
    print(
        f"{pname:18s}: n={n:5d}, distinct_sp_toks={ps['distinct_spaced_tokens']:4d}, "
        f"mean_sp_toks={ps['mean_spaced_tokens']:.3f}, vowel_init={vowels}/{n} ({100 * vowels / max(1, n):4.1f}%), "
        f"len_hist={ps['spaced_token_length_hist']}"
    )

print("\n=== PART B1: R7 TASK & WEIGHT DISTRIBUTION ===")
r7 = d["r7"]
tot_qs = r7["ret_questions"]
tot_ans_toks = r7["ret_answer_tokens"]
copy_seqs = r7["copy_sequences"]
tot_wt = tot_ans_toks + copy_seqs
print(
    f"ret_seqs={r7['ret_sequences']}, copy_seqs={copy_seqs}, total_seqs={r7['total_sequences']}, "
    f"aug_modified_q={r7['aug_modified_q_count']}/{tot_qs} ({100 * r7['aug_modified_q_count'] / tot_qs:.1f}%), "
    f"unresolved={r7['unresolved_op_count']}"
)
print(
    f"ret_qs={tot_qs}, ret_ans_toks={tot_ans_toks}, copy_tgt_toks={r7['copy_target_tokens']}, "
    f"tot_obj_wt={tot_wt}, copy_wt_share={100 * copy_seqs / tot_wt:.2f}%"
)

by_kind = defaultdict(lambda: [0, 0])
by_k_op_ord = defaultdict(lambda: [0, 0])
by_lvl_kind_ord = defaultdict(lambda: [0, 0])
for cell, qn in r7["questions_by_cell"].items():
    tn = r7["answer_tokens_by_cell"][cell]
    lvl, kind, attr, op, order = cell.split("|")
    by_kind[kind][0] += qn
    by_kind[kind][1] += tn
    by_k_op_ord[f"{kind}|{op}|{order}"][0] += qn
    by_k_op_ord[f"{kind}|{op}|{order}"][1] += tn
    by_lvl_kind_ord[f"{lvl}|{kind}|{order}"][0] += qn
    by_lvl_kind_ord[f"{lvl}|{kind}|{order}"][1] += tn

print("  By kind:")
for k, (q, t) in sorted(by_kind.items()):
    print(f"    {k:6s}: qs={q:6d} ({100 * q / tot_qs:5.2f}%)  wt/toks={t:6d} ({100 * t / tot_wt:5.2f}% total wt)")

print("  By kind|op|order:")
for k, (q, t) in sorted(by_k_op_ord.items()):
    print(f"    {k:28s}: qs={q:6d} ({100 * q / tot_qs:5.2f}%)  wt/toks={t:6d} ({100 * t / tot_wt:5.2f}% total wt)")

print("  L1/L2/L3 x kind x order (attr, rel, num):")
for k, (q, t) in sorted(by_lvl_kind_ord.items()):
    if any(x in k for x in ("|attr|", "|rel|", "|num|")):
        print(f"    {k:22s}: qs={q:6d} ({100 * q / tot_qs:5.2f}%)  wt/toks={t:6d} ({100 * t / tot_wt:5.2f}% total wt)")

print("\n=== PART B2: PILOT 3 TASK DISTRIBUTION ===")
p3 = d["pilot3"]
print(json.dumps(p3, indent=2))

print("\n=== PART B3: PILOT 1 & PILOT 2 DATA AUDITS ===")
p1 = json.load(open(f"{BASE}/pilot1/data-audit.json"))
p2 = json.load(open(f"{BASE}/pilot2/data-audit.json"))
print("Pilot 1 data-audit.json:", json.dumps(p1, indent=2))
print("Pilot 2 data-audit.json:", json.dumps(p2, indent=2))
