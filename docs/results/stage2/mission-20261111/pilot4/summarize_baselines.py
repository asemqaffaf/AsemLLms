#!/usr/bin/env python3
"""Summarize Pilot 4 7-probe baselines (R7, P3 5e-4, P3 1e-3) or compare additional checkpoints vs R7 with paired McNemar tests."""
import json
import math
import sys
from collections import defaultdict

IN_PATH = sys.argv[1] if len(sys.argv) > 1 else "docs/results/stage2/mission-20261111/pilot4/baseline_eval_results.json"
d = json.load(open(IN_PATH))["results"]


def mcnemar_exact(a, b):
    gains = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    losses = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = gains + losses
    if n == 0:
        return gains, losses, 1.0
    k = min(gains, losses)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return gains, losses, min(1.0, p)


ckpts = list(d.keys())
print("=== INDUCTION HEALTH CHECKS (n=50 seqs, half=40, seed=17) ===")
print(f"{'Ckpt':10s} | {'Std loss1->2 (acc2)':22s} | {'SeenHi100 loss1->2 (acc2)':25s} | {'EvalHeldout loss1->2 (acc2)':27s}")
for ck in ckpts:
    ind = d[ck]["induction"]
    cols = []
    for k in ("std_300_7300", "seen_hi100_alpha", "eval_heldout_alpha"):
        x = ind[k]
        cols.append(f"{x['loss_first_copy']:5.2f}->{x['loss_second_copy']:5.2f} ({100*x['acc_second_copy']:4.1f}%)")
    print(f"{ck:10s} | {cols[0]:22s} | {cols[1]:25s} | {cols[2]:27s}")

print("\n=== PRIMARY 7-PROBE SUMMARY TABLE (K=4 chance=25.0%, Greedy K=8192 chance~0%) ===")
r7 = d.get("r7")
probe_order = ["fresh_ans_before", "entity_copy", "extract_ret", "relation_ret", "reverse_ret", "compare_ret", "temporal_ret"]
for pn in probe_order:
    print(f"\n-- Probe: {pn} --")
    for ck in ckpts:
        rows = d[ck]["probes"][pn]
        n = len(rows)
        has_k4 = any(r["k4"] >= 0 for r in rows)
        has_ex = any(r["exact"] >= 0 for r in rows)
        parts = [f"  {ck:10s} (n={n:3d}):"]
        if has_k4:
            k4_rows = [r for r in rows if r["k4"] >= 0]
            nk = len(k4_rows)
            sk4 = sum(r["k4"] for r in k4_rows)
            s = f"K4={sk4:3d}/{nk} ({100*sk4/nk:5.1f}%)"
            if r7 and ck != "r7":
                r7_rows = [r for r in r7["probes"][pn] if r["k4"] >= 0]
                g, l, pv = mcnemar_exact([x["k4"] for x in r7_rows], [x["k4"] for x in k4_rows])
                dp = 100.0 * (sk4 - sum(x["k4"] for x in r7_rows)) / nk
                s += f" [vs R7 {dp:+5.1f}pp, +{g}/-{l}, p={pv:.3g}]"
            parts.append(s)
        if has_ex:
            sex = sum(r["exact"] for r in rows)
            sbd = sum(r["bounded"] for r in rows)
            nll = sum(r["nll"] for r in rows) / n
            s = f"Ex={sex:3d}/{n} ({100*sex/n:5.1f}%) Bd={sbd:3d}/{n} ({100*sbd/n:5.1f}%) NLL={nll:.2f}"
            if r7 and ck != "r7":
                r7_ex = [x["exact"] for x in r7["probes"][pn]]
                g, l, pv = mcnemar_exact(r7_ex, [r["exact"] for r in rows])
                dp = 100.0 * (sex - sum(r7_ex)) / n
                s += f" [Ex vs R7 {dp:+5.1f}pp, +{g}/-{l}, p={pv:.3g}]"
            parts.append(s)
        print(" | ".join(parts))

print("\n=== STRATIFIED BREAKDOWNS PER PROBE ===")
for pn in probe_order:
    rows0 = d[ckpts[0]]["probes"][pn]
    print(f"\n[{pn}]")
    # Stratify by tag and split
    groups = defaultdict(list)
    for idx, r in enumerate(rows0):
        w = r.get("wording", r["split"])
        if pn == "fresh_ans_before":
            kind = "pf_rev" if "-pf-" in r["id"] else "vf_ext"
            groups[f"{kind}|{r['tag']}|{w}"].append(idx)
            groups[f"ALL|{kind}|{r['tag']}"].append(idx)
        elif pn == "entity_copy":
            st = r["subtype"]
            # collapse pf/vf into s1 regime
            if st.startswith("s1-"):
                base = "_".join(st.split("-")[:-1])
                groups[f"{base}|{st.split('-')[-1]}"].append(idx)
                groups[f"ALL|{base}"].append(idx)
            else:
                groups[f"ALL|{st}"].append(idx)
        elif pn == "extract_ret":
            kind = "num" if "-num-" in r["id"] else "attr"
            groups[f"{kind}|{w}"].append(idx)
        elif pn == "reverse_ret":
            groups[f"{r['tag']}|{w}"].append(idx)
        elif pn == "relation_ret":
            sub_dir = "subj" if r["subtype"].endswith("-subj") else "obj"
            groups[f"{r['tag']}|{sub_dir}|{w}"].append(idx)
        else:
            groups[f"{r['tag']}|{w}"].append(idx)

    for gk in sorted(groups):
        idxs = groups[gk]
        line_parts = [f"  {gk:32s} (n={len(idxs):3d}):"]
        for ck in ckpts:
            rs = [d[ck]["probes"][pn][i] for i in idxs]
            if rs[0]["exact"] >= 0 and rs[0]["k4"] >= 0:
                k4 = sum(x["k4"] for x in rs)
                ex = sum(x["exact"] for x in rs)
                line_parts.append(f"{ck}=K4:{100*k4/len(idxs):4.1f}%/Ex:{100*ex/len(idxs):4.1f}%")
            elif rs[0]["exact"] >= 0:
                ex = sum(x["exact"] for x in rs)
                bd = sum(x["bounded"] for x in rs)
                line_parts.append(f"{ck}=Ex:{ex}/{len(idxs)}({100*ex/len(idxs):4.1f}%)/Bd:{bd}")
            else:
                k4 = sum(x["k4"] for x in rs)
                line_parts.append(f"{ck}=K4:{k4}/{len(idxs)}({100*k4/len(idxs):4.1f}%)")
        print(" | ".join(line_parts))
