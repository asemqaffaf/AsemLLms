#!/usr/bin/env python3
"""Analyze Pilot 4 causal disentanglement micro-probe results with paired McNemar p-values."""
import json, math
from collections import Counter, defaultdict

IN_PATH = "docs/results/stage2/mission-20261111/pilot4/causal_probe_results.json"
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


print("=== EXP 2: STRATIFIED RANDOM-TOKEN INDUCTION PROBE (n=50 seqs, half=40, seed=17) ===")
print(f"{'Ckpt':8s} | {'Std loss1->2 (acc2)':22s} | {'Hi100 loss1->2 (acc2)':22s} | {'ZeroFW loss1->2 (acc2)':22s}")
for ck in ("r7", "p3_5e4", "p3_1e3", "p2_trt"):
    ind = d[ck]["induction"]
    cols = []
    for bk in ("std_300_7300", "p3_r7_hi100_alpha", "zero_ret_high_fw_alpha"):
        x = ind[bk]
        cols.append(f"{x['loss_first_copy']:5.2f}->{x['loss_second_copy']:5.2f} ({100*x['acc_second_copy']:4.1f}%)")
    print(f"{ck:8s} | {cols[0]:22s} | {cols[1]:22s} | {cols[2]:22s}")

print("\n=== EXP 1: CAUSAL ENTITY-SWAP ON FROZEN D1 REV-PF (n=157 pairs) ===")
for direction in ("zero_to_hi100", "hi100_to_zero"):
    print(f"\n-- Direction: {direction} --")
    for ck in ("r7", "p3_5e4", "p3_1e3", "p2_trt"):
        ps = [p for p in d[ck]["d1_swap_pairs"] if p["direction"] == direction]
        n = len(ps)
        ok4 = [p["orig_k4"] for p in ps]
        sk4 = [p["swap_k4"] for p in ps]
        oex = [p["orig_ex"] for p in ps]
        sex = [p["swap_ex"] for p in ps]
        obd = [p["orig_bd"] for p in ps]
        sbd = [p["swap_bd"] for p in ps]
        ot0 = [p["orig_tok0"] for p in ps]
        st0 = [p["swap_tok0"] for p in ps]
        onll = sum(p["orig_nll"] for p in ps) / n
        snll = sum(p["swap_nll"] for p in ps) / n
        gk4, lk4, pk4 = mcnemar_exact(ok4, sk4)
        gex, lex, pex = mcnemar_exact(oex, sex)
        gbd, lbd, pbd = mcnemar_exact(obd, sbd)
        print(
            f"  {ck:7s} (n={n:2d}): "
            f"K4 {sum(ok4):2d}->{sum(sk4):2d} ({100*sum(ok4)/n:4.1f}%->{100*sum(sk4)/n:4.1f}%, +{gk4}/-{lk4} p={pk4:.3g}) | "
            f"Ex {sum(oex):2d}->{sum(sex):2d} ({100*sum(oex)/n:4.1f}%->{100*sum(sex)/n:4.1f}%, +{gex}/-{lex} p={pex:.3g}) | "
            f"Bd {sum(obd):2d}->{sum(sbd):2d} (+{gbd}/-{lbd} p={pbd:.3g}) | "
            f"Tok0 {sum(ot0):2d}->{sum(st0):2d} | NLL {onll:.2f}->{snll:.2f}"
        )
        # Sub-strata
        if direction == "zero_to_hi100":
            for st in ("zero_tok0", "zero_tail"):
                sub = [p for p in ps if p["stratum"] == st]
                print(
                    f"     [{st:9s} n={len(sub)}]: "
                    f"K4 {sum(p['orig_k4'] for p in sub)}->{sum(p['swap_k4'] for p in sub)} | "
                    f"Ex {sum(p['orig_ex'] for p in sub)}->{sum(p['swap_ex'] for p in sub)} | "
                    f"Bd {sum(p['orig_bd'] for p in sub)}->{sum(p['swap_bd'] for p in sub)} | "
                    f"Tok0 {sum(p['orig_tok0'] for p in sub)}->{sum(p['swap_tok0'] for p in sub)}"
                )
        else:
            for st in ("tok0_zero", "tail_zero"):
                sub = [p for p in ps if p["swap_sub_regime"] == st]
                print(
                    f"     [-> {st:9s} n={len(sub)}]: "
                    f"K4 {sum(p['orig_k4'] for p in sub)}->{sum(p['swap_k4'] for p in sub)} | "
                    f"Ex {sum(p['orig_ex'] for p in sub)}->{sum(p['swap_ex'] for p in sub)} | "
                    f"Bd {sum(p['orig_bd'] for p in sub)}->{sum(p['swap_bd'] for p in sub)} | "
                    f"Tok0 {sum(p['orig_tok0'] for p in sub)}->{sum(p['swap_tok0'] for p in sub)}"
                )

print("\n=== EXP 3: 1-SENT COPY CUE vs 4-SENT L3 RETRIEVAL CUE (n=60 per regime) ===")
for cue in ("s1_copy", "s4_ret"):
    print(f"\n-- Cue: {cue} --")
    for regime in ("seen_hi100", "d1_zero", "fw_zero"):
        print(f"  Regime={regime}:")
        for ck in ("r7", "p3_5e4", "p3_1e3", "p2_trt"):
            rs = [x for x in d[ck]["prose_probes"] if x["cue"] == cue and x["regime"] == regime]
            n = len(rs)
            ex = sum(x["exact"] for x in rs)
            bd = sum(x["bounded"] for x in rs)
            t0 = sum(x["tok0_ok"] for x in rs)
            nll = sum(x["nll"] for x in rs) / n
            k4_str = ""
            if cue == "s4_ret":
                k4 = sum(x["k4"] for x in rs)
                k4_str = f" | K4={k4:2d}/{n} ({100*k4/n:4.1f}%)"
            errs = [(x["gold"], x["gen"]) for x in rs if x["exact"] == 0][:4]
            print(f"    {ck:7s}: Exact={ex:2d}/{n} ({100*ex/n:5.1f}%) | Bd={bd:2d}/{n} ({100*bd/n:5.1f}%) | Tok0={t0:2d}/{n} ({100*t0/n:5.1f}%){k4_str} | NLL={nll:.2f} | ex_err={errs}")
