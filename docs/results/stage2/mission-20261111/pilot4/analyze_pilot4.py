#!/usr/bin/env python3
"""Merge Pilot 4 checkpoint evaluations with frozen baselines, test all binding gates,
compute 2x2 factorial contrasts, summarize frozen D1 200 cells, and evaluate the Stop Rule.
"""
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
DIAG = ROOT.parent.parent / "diagnostics-20261113"
P3 = ROOT.parent / "pilot3"
RUN_DIR = ROOT.parent.parent.parent.parent.parent / "data/runs/retrieval/p4-retention"


def mcnemar_exact(a, b):
    gains = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    losses = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = gains + losses
    if n == 0:
        return gains, losses, 1.0
    k = min(gains, losses)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return gains, losses, min(1.0, p)


def load_results():
    base = json.loads((ROOT / "baseline_eval_results.json").read_text())["results"]
    merged = {}
    for k in ("r7", "p3_5e4", "p3_1e3"):
        if k in base:
            merged[k] = base[k]
    arm_files = [
        ("arm_a", ROOT / "eval_arm_a.json"),
        ("arm_b", ROOT / "eval_arm_b.json"),
        ("arm_c", ROOT / "eval_arm_c.json"),
        ("arm_d", ROOT / "eval_arm_d.json"),
        ("arm_b_250", ROOT / "eval_arm_b_250.json"),
        ("arm_c_250", ROOT / "eval_arm_c_250.json"),
        ("arm_d_250", ROOT / "eval_arm_d_250.json"),
    ]
    for tag, fp in arm_files:
        if fp.exists():
            res = json.loads(fp.read_text())["results"]
            for k, v in res.items():
                merged[tag] = v
    out_path = ROOT / "p4_eval_results.json"
    out_path.write_text(json.dumps({"schema": "asem-pilot4-probe-eval-1", "results": merged}, indent=2) + "\n")
    return merged


def vec(ckpt_obj, probe, metric, pred_fn=None):
    rows = ckpt_obj["probes"][probe]
    if pred_fn is not None:
        rows = [r for r in rows if pred_fn(r)]
    return [r[metric] for r in rows]


def check_retention_gate(arm_obj, r7_obj, probe, floor_k, subgroup_name=None, subgroup_fn=None, subgroup_floor=None):
    a_v = vec(arm_obj, probe, "k4")
    r_v = vec(r7_obj, probe, "k4")
    n = len(a_v)
    ac = sum(a_v)
    rc = sum(r_v)
    g, l, p = mcnemar_exact(r_v, a_v)
    neg_ok = True if ac >= rc else (p > 0.05)
    floor_ok = ac >= floor_k
    sub_res = None
    sub_ok = True
    if subgroup_fn is not None:
        sa = vec(arm_obj, probe, "k4", subgroup_fn)
        sr = vec(r7_obj, probe, "k4", subgroup_fn)
        sg, sl, sp = mcnemar_exact(sr, sa)
        sac, src = sum(sa), sum(sr)
        sub_ok = sac >= subgroup_floor
        sub_res = {
            "name": subgroup_name,
            "k": sac,
            "n": len(sa),
            "r7_k": src,
            "dp": 100.0 * (sac - src) / len(sa),
            "g": sg,
            "l": sl,
            "p": sp,
            "floor": subgroup_floor,
            "pass": sub_ok,
        }
    return {
        "k": ac,
        "n": n,
        "pct": 100.0 * ac / n,
        "r7_k": rc,
        "dp": 100.0 * (ac - rc) / n,
        "g": g,
        "l": l,
        "p": p,
        "floor": floor_k,
        "floor_ok": floor_ok,
        "neg_p_ok": neg_ok,
        "subgroup": sub_res,
        "pass": floor_ok and neg_ok and sub_ok,
    }


def evaluate_gates(merged, ck):
    r7 = merged["r7"]
    m = merged[ck]

    # G1a: fresh_ans_before K4 >= 140/300 AND p <= 0.05 vs R7
    g1a_a = vec(m, "fresh_ans_before", "k4")
    g1a_r = vec(r7, "fresh_ans_before", "k4")
    g, l, p = mcnemar_exact(g1a_r, g1a_a)
    g1a = {
        "k": sum(g1a_a), "n": len(g1a_a), "pct": 100.0 * sum(g1a_a) / len(g1a_a),
        "dp": 100.0 * (sum(g1a_a) - sum(g1a_r)) / len(g1a_a),
        "g": g, "l": l, "p": p, "floor": 140,
        "pass": (sum(g1a_a) >= 140) and (p <= 0.05),
    }

    # G1b: fresh_ans_before Exact >= 48/300 AND p <= 0.05 vs R7
    g1b_a = vec(m, "fresh_ans_before", "exact")
    g1b_r = vec(r7, "fresh_ans_before", "exact")
    g, l, p = mcnemar_exact(g1b_r, g1b_a)
    g1b = {
        "k": sum(g1b_a), "n": len(g1b_a), "pct": 100.0 * sum(g1b_a) / len(g1b_a),
        "dp": 100.0 * (sum(g1b_a) - sum(g1b_r)) / len(g1b_a),
        "g": g, "l": l, "p": p, "floor": 48,
        "pass": (sum(g1b_a) >= 48) and (p <= 0.05),
    }

    # G1c: pf_rev|seen100 Exact >= 20/50 AND pf_rev|recomb_tr Exact >= 5/50
    s100_a = vec(m, "fresh_ans_before", "exact", lambda r: "-pf-" in r["id"] and r["tag"] == "seen100")
    s100_r = vec(r7, "fresh_ans_before", "exact", lambda r: "-pf-" in r["id"] and r["tag"] == "seen100")
    rec_a = vec(m, "fresh_ans_before", "exact", lambda r: "-pf-" in r["id"] and r["tag"] == "recomb_tr")
    rec_r = vec(r7, "fresh_ans_before", "exact", lambda r: "-pf-" in r["id"] and r["tag"] == "recomb_tr")
    hev_a = vec(m, "fresh_ans_before", "exact", lambda r: "-pf-" in r["id"] and r["tag"] == "heldout_ev")
    sg1, sl1, sp1 = mcnemar_exact(s100_r, s100_a)
    sg2, sl2, sp2 = mcnemar_exact(rec_r, rec_a)
    g1c = {
        "seen100_k": sum(s100_a), "seen100_n": len(s100_a), "seen100_p": sp1, "seen100_floor": 20,
        "recomb_tr_k": sum(rec_a), "recomb_tr_n": len(rec_a), "recomb_tr_p": sp2, "recomb_tr_floor": 5,
        "heldout_ev_k": sum(hev_a), "heldout_ev_n": len(hev_a),
        "pass": (sum(s100_a) >= 20) and (sum(rec_a) >= 5),
    }

    # G2a: s1_seen_synth >= 70/80 AND s1_seen_synth|vf >= 34/40
    ss_a = vec(m, "entity_copy", "exact", lambda r: r["subtype"].startswith("s1-seen_synth"))
    ss_vf = vec(m, "entity_copy", "exact", lambda r: r["subtype"] == "s1-seen_synth-vf")
    ss_r = vec(r7, "entity_copy", "exact", lambda r: r["subtype"].startswith("s1-seen_synth"))
    ss_vfr = vec(r7, "entity_copy", "exact", lambda r: r["subtype"] == "s1-seen_synth-vf")
    _, _, p_ss = mcnemar_exact(ss_r, ss_a)
    _, _, p_vf = mcnemar_exact(ss_vfr, ss_vf)
    g2a = {
        "all_k": sum(ss_a), "all_n": len(ss_a), "all_p": p_ss, "all_floor": 70,
        "vf_k": sum(ss_vf), "vf_n": len(ss_vf), "vf_p": p_vf, "vf_floor": 34,
        "pass": (sum(ss_a) >= 70) and (sum(ss_vf) >= 34),
    }

    # G2b: s1_recomb_tr Exact >= 44/80 AND p <= 0.01 vs R7
    rt_a = vec(m, "entity_copy", "exact", lambda r: r["subtype"].startswith("s1-recomb_tr"))
    rt_r = vec(r7, "entity_copy", "exact", lambda r: r["subtype"].startswith("s1-recomb_tr"))
    g, l, p = mcnemar_exact(rt_r, rt_a)
    g2b = {
        "k": sum(rt_a), "n": len(rt_a), "pct": 100.0 * sum(rt_a) / len(rt_a),
        "dp": 100.0 * (sum(rt_a) - sum(rt_r)) / len(rt_a),
        "g": g, "l": l, "p": p, "floor": 44,
        "pass": (sum(rt_a) >= 44) and (p <= 0.01),
    }

    # G2c: s1_heldout_ev Exact >= 22/80 (and if < R7, p > 0.05) AND eval_heldout_alpha >= 90% AND std_300_7300 >= 90%
    he_a = vec(m, "entity_copy", "exact", lambda r: r["subtype"].startswith("s1-heldout_ev"))
    he_r = vec(r7, "entity_copy", "exact", lambda r: r["subtype"].startswith("s1-heldout_ev"))
    g, l, p = mcnemar_exact(he_r, he_a)
    ind_ev = 100.0 * m["induction"]["eval_heldout_alpha"]["acc_second_copy"]
    ind_st = 100.0 * m["induction"]["std_300_7300"]["acc_second_copy"]
    he_neg_ok = True if sum(he_a) >= sum(he_r) else (p > 0.05)
    g2c = {
        "k": sum(he_a), "n": len(he_a), "pct": 100.0 * sum(he_a) / len(he_a),
        "dp": 100.0 * (sum(he_a) - sum(he_r)) / len(he_a),
        "g": g, "l": l, "p": p, "floor": 22,
        "ind_ev": ind_ev, "ind_std": ind_st,
        "pass": (sum(he_a) >= 22) and he_neg_ok and (ind_ev >= 90.0) and (ind_st >= 90.0),
    }

    # G2d: L3_recomb_tr Exact >= 6/30 AND p <= 0.05 AND NLL <= 3.50 AND L3_heldout_ev NLL <= 6.24
    l3r_ex = vec(m, "entity_copy", "exact", lambda r: r["subtype"] == "L3_recomb_tr")
    l3r_exr = vec(r7, "entity_copy", "exact", lambda r: r["subtype"] == "L3_recomb_tr")
    l3r_nll = sum(vec(m, "entity_copy", "nll", lambda r: r["subtype"] == "L3_recomb_tr")) / 30.0
    l3h_ex = vec(m, "entity_copy", "exact", lambda r: r["subtype"] == "L3_heldout_ev")
    l3h_nll = sum(vec(m, "entity_copy", "nll", lambda r: r["subtype"] == "L3_heldout_ev")) / 30.0
    l3r_k4 = vec(m, "entity_copy", "k4", lambda r: r["subtype"] == "L3_recomb_tr")
    l3h_k4 = vec(m, "entity_copy", "k4", lambda r: r["subtype"] == "L3_heldout_ev")
    g, l, p = mcnemar_exact(l3r_exr, l3r_ex)
    g2d = {
        "recomb_k": sum(l3r_ex), "recomb_n": 30, "p": p, "recomb_nll": l3r_nll,
        "heldout_k": sum(l3h_ex), "heldout_nll": l3h_nll,
        "recomb_k4": sum(l3r_k4), "heldout_k4": sum(l3h_k4),
        "pass": (sum(l3r_ex) >= 6) and (p <= 0.05) and (l3r_nll <= 3.50) and (l3h_nll <= 6.24),
    }

    # G3..G7
    g3 = check_retention_gate(m, r7, "extract_ret", 148, "num", lambda r: "-num-" in r["id"], 60)
    g4 = check_retention_gate(m, r7, "relation_ret", 123)
    g5 = check_retention_gate(m, r7, "reverse_ret", 112, "ans_after", lambda r: r["tag"] == "ans_after", 69)
    g6 = check_retention_gate(m, r7, "compare_ret", 64)
    g7 = check_retention_gate(m, r7, "temporal_ret", 77)

    # Aggregate Retention (1,000) >= 533/1000 AND if < R7 (553), p > 0.05
    ret_probes = ("extract_ret", "relation_ret", "reverse_ret", "compare_ret", "temporal_ret")
    agg_a = [x for p_name in ret_probes for x in vec(m, p_name, "k4")]
    agg_r = [x for p_name in ret_probes for x in vec(r7, p_name, "k4")]
    ag_g, ag_l, ag_p = mcnemar_exact(agg_r, agg_a)
    agg = {
        "k": sum(agg_a), "n": 1000, "pct": 100.0 * sum(agg_a) / 1000.0,
        "r7_k": sum(agg_r), "dp": 100.0 * (sum(agg_a) - sum(agg_r)) / 1000.0,
        "g": ag_g, "l": ag_l, "p": ag_p, "floor": 533,
        "floor_ok": sum(agg_a) >= 533,
        "neg_p_ok": True if sum(agg_a) >= sum(agg_r) else (ag_p > 0.05),
        "pass": (sum(agg_a) >= 533) and (True if sum(agg_a) >= sum(agg_r) else (ag_p > 0.05)),
    }

    g1_all = g1a["pass"] and g1b["pass"] and g1c["pass"]
    ret_all = g3["pass"] and g4["pass"] and g5["pass"] and g6["pass"] and g7["pass"] and agg["pass"]
    non_stop_pass = g1_all and g2c["pass"] and ret_all
    full_promo_pass = non_stop_pass and g2a["pass"] and g2b["pass"] and g2d["pass"]

    return {
        "G1a": g1a, "G1b": g1b, "G1c": g1c, "G1": g1_all,
        "G2a": g2a, "G2b": g2b, "G2c": g2c, "G2d": g2d,
        "G3": g3, "G4": g4, "G5": g5, "G6": g6, "G7": g7, "Agg": agg,
        "RetAll": ret_all,
        "NonStopPass": non_stop_pass,
        "FullPromoPass": full_promo_pass,
    }


def load_d1_file(path):
    d = json.loads(path.read_text())
    its = d["conditions"]["real"]["items"]
    return {
        x["id"]: {
            "sub": x["subtype"],
            "k4": int(x["summed_option_prediction"] == x["key"]),
            "exact": int(bool(x["gold_length_greedy_prefix_exact"])),
        }
        for x in its
    }


def analyze_d1():
    cells = ("d1-rev-pf", "d1-rev-vf", "d1-ext-pf")
    models = [
        ("r7", lambda c: DIAG / f"{c}.r7.json"),
        ("p3_5e4", lambda c: P3 / f"{c}.arm-5e-4.step500.json"),
        ("p3_1e3", lambda c: P3 / f"{c}.arm-1e-3.step500.json"),
        ("arm_a", lambda c: ROOT / f"{c}.arm-a.json"),
        ("arm_b", lambda c: ROOT / f"{c}.arm-b.json"),
        ("arm_c", lambda c: ROOT / f"{c}.arm-c.json"),
        ("arm_d", lambda c: ROOT / f"{c}.arm-d.json"),
    ]
    out = {}
    for cell in cells:
        r7_path = DIAG / f"{cell}.r7.json"
        if not r7_path.exists():
            continue
        r7_items = load_d1_file(r7_path)
        ids = list(r7_items.keys())
        out[cell] = {}
        for mname, fn in models:
            p = fn(cell)
            if not p.exists():
                continue
            cur = load_d1_file(p)
            k4_cur = [cur[i]["k4"] for i in ids]
            k4_r7 = [r7_items[i]["k4"] for i in ids]
            ex_cur = [cur[i]["exact"] for i in ids]
            ex_r7 = [r7_items[i]["exact"] for i in ids]
            gk, lk, pk = mcnemar_exact(k4_r7, k4_cur)
            ge, le, pe = mcnemar_exact(ex_r7, ex_cur)
            out[cell][mname] = {
                "k4": sum(k4_cur), "n": len(ids), "k4_pct": 100.0 * sum(k4_cur) / len(ids),
                "k4_g": gk, "k4_l": lk, "k4_p": pk,
                "exact": sum(ex_cur), "exact_pct": 100.0 * sum(ex_cur) / len(ids),
                "ex_g": ge, "ex_l": le, "ex_p": pe,
            }
    return out


def analyze_selection():
    r7_path = ROOT.parent / "r7-selection.json"
    if not r7_path.exists():
        return {}
    r7_items = json.loads(r7_path.read_text())["conditions"]["real"]["items"]
    r7_map = {x["id"]: {"cat": x["category"], "k4": int(x["summed_option_prediction"] == x["key"])} for x in r7_items}
    models = [
        ("r7", r7_path),
        ("p3_5e4", P3 / "selection.arm-5e-4.step500.json"),
        ("p3_1e3", P3 / "selection.arm-1e-3.step500.json"),
        ("arm_a", ROOT / "selection.arm-a.json"),
        ("arm_b", ROOT / "selection.arm-b.json"),
        ("arm_c", ROOT / "selection.arm-c.json"),
        ("arm_d", ROOT / "selection.arm-d.json"),
    ]
    loaded = {}
    for mn, mp in models:
        if mp.exists():
            its = json.loads(mp.read_text())["conditions"]["real"]["items"]
            loaded[mn] = {x["id"]: int(x["summed_option_prediction"] == x["key"]) for x in its}
    cats = ("extract", "relation", "reverse", "compare", "temporal", "ALL")
    out = {}
    for cat in cats:
        ids = [i for i, v in r7_map.items() if cat == "ALL" or v["cat"] == cat]
        out[cat] = {}
        r7_v = [r7_map[i]["k4"] for i in ids]
        for mn, cur in loaded.items():
            cv = [cur[i] for i in ids]
            g, l, p = mcnemar_exact(r7_v, cv)
            out[cat][mn] = {"k": sum(cv), "n": len(ids), "pct": 100.0 * sum(cv) / len(ids), "g": g, "l": l, "p": p}
    return out


def main():
    merged = load_results()
    ckpts = [c for c in ("r7", "p3_5e4", "p3_1e3", "arm_a", "arm_b_250", "arm_b", "arm_c_250", "arm_c", "arm_d_250", "arm_d") if c in merged]
    print("=== INDUCTION HEALTH CHECKS (K=8192, n=50 seqs, half=40, seed=17) ===")
    for c in ckpts:
        ind = merged[c]["induction"]
        s = ind["std_300_7300"]
        e = ind["eval_heldout_alpha"]
        h = ind["seen_hi100_alpha"]
        print(f"  {c:10s} | std={100*s['acc_second_copy']:5.1f}% (l2={s['loss_second_copy']:5.2f}) | eval_heldout={100*e['acc_second_copy']:5.1f}% (l2={e['loss_second_copy']:5.2f}) | seen_hi100={100*h['acc_second_copy']:5.1f}% (l2={h['loss_second_copy']:5.2f})")

    arms_final = [a for a in ("arm_a", "arm_b", "arm_c", "arm_d") if a in merged]
    gate_results = {}
    for ck in [c for c in ("p3_5e4", "arm_a", "arm_b", "arm_c", "arm_d") if c in merged]:
        gate_results[ck] = evaluate_gates(merged, ck)

    print("\n=== BINDING GATE MATRIX (Final Checkpoints vs Frozen R7) ===")
    for ck, gr in gate_results.items():
        print(f"\n[{ck}] NonStopPass={gr['NonStopPass']} FullPromoPass={gr['FullPromoPass']} (G1={gr['G1']}, G2c={gr['G2c']['pass']}, RetAll={gr['RetAll']})")
        print(f"  G1a fresh K4      : {gr['G1a']['k']:3d}/300 ({gr['G1a']['pct']:5.1f}%, {gr['G1a']['dp']:+5.1f}pp, +{gr['G1a']['g']}/-{gr['G1a']['l']} p={gr['G1a']['p']:.3g}) [floor >=140 & p<=0.05] -> {gr['G1a']['pass']}")
        print(f"  G1b fresh Exact   : {gr['G1b']['k']:3d}/300 ({gr['G1b']['pct']:5.1f}%, {gr['G1b']['dp']:+5.1f}pp, +{gr['G1b']['g']}/-{gr['G1b']['l']} p={gr['G1b']['p']:.3g}) [floor >=48 & p<=0.05] -> {gr['G1b']['pass']}")
        print(f"  G1c pf seen100/rec: seen100={gr['G1c']['seen100_k']:2d}/50 (>=20), recomb_tr={gr['G1c']['recomb_tr_k']:2d}/50 (>=5), heldout_ev={gr['G1c']['heldout_ev_k']:2d}/50 -> {gr['G1c']['pass']}")
        print(f"  G2a seen_synth    : all={gr['G2a']['all_k']:2d}/80 (>=70), vf={gr['G2a']['vf_k']:2d}/40 (>=34) -> {gr['G2a']['pass']}")
        print(f"  G2b recomb_tr     : {gr['G2b']['k']:2d}/80 ({gr['G2b']['pct']:5.1f}%, +{gr['G2b']['g']}/-{gr['G2b']['l']} p={gr['G2b']['p']:.3g}) [>=44 & p<=0.01] -> {gr['G2b']['pass']}")
        print(f"  G2c heldout_ev+ind: {gr['G2c']['k']:2d}/80 ({gr['G2c']['pct']:5.1f}%, {gr['G2c']['dp']:+5.1f}pp, p={gr['G2c']['p']:.3g}) [>=22] | ind_ev={gr['G2c']['ind_ev']:4.1f}% ind_std={gr['G2c']['ind_std']:4.1f}% [>=90%] -> {gr['G2c']['pass']}")
        print(f"  G2d L3_copy       : recomb Ex={gr['G2d']['recomb_k']:2d}/30 (>=6, p={gr['G2d']['p']:.3g}) NLL={gr['G2d']['recomb_nll']:.2f} (<=3.50) | heldout NLL={gr['G2d']['heldout_nll']:.2f} (<=6.24) [K4 rec={gr['G2d']['recomb_k4']}/30 hev={gr['G2d']['heldout_k4']}/30] -> {gr['G2d']['pass']}")
        for gn in ("G3", "G4", "G5", "G6", "G7", "Agg"):
            x = gr[gn]
            sub_str = ""
            if x.get("subgroup"):
                sb = x["subgroup"]
                sub_str = f" | {sb['name']}={sb['k']}/{sb['n']} (>={sb['floor']}, ok={sb['pass']})"
            print(f"  {gn:3s} retention     : {x['k']:3d}/{x['n']} ({x['pct']:5.1f}%, {x['dp']:+5.1f}pp, +{x['g']}/-{x['l']} p={x['p']:.3g}) [>={x['floor']}]{sub_str} -> {x['pass']}")

    if len(arms_final) == 4:
        print("\n=== 2x2 FACTORIAL CONTRASTS (Treat <6.0pp on n=300 as Single-Seed Noise) ===")
        contrasts = [
            ("Copy (+Copy, noReplay): B vs A", "arm_a", "arm_b"),
            ("Copy (+Copy, +Replay) : D vs C", "arm_c", "arm_d"),
            ("Replay (-Copy, +Replay): C vs A", "arm_a", "arm_c"),
            ("Replay (+Copy, +Replay): D vs B", "arm_b", "arm_d"),
        ]
        metrics = [
            ("fresh_ans_before K4 (n=300)", "fresh_ans_before", "k4", None, True),
            ("fresh_ans_before Ex (n=300)", "fresh_ans_before", "exact", None, True),
            ("s1_recomb_tr Ex (n=80)", "entity_copy", "exact", lambda r: r["subtype"].startswith("s1-recomb_tr"), False),
            ("s1_heldout_ev Ex (n=80)", "entity_copy", "exact", lambda r: r["subtype"].startswith("s1-heldout_ev"), False),
            ("extract_ret K4 (n=200)", "extract_ret", "k4", None, False),
            ("relation_ret K4 (n=200)", "relation_ret", "k4", None, False),
            ("reverse_ret K4 (n=200)", "reverse_ret", "k4", None, False),
        ]
        for label, a0, a1 in contrasts:
            print(f"\n[{label}]")
            for mlabel, pn, fld, fn, n300 in metrics:
                v0 = vec(merged[a0], pn, fld, fn)
                v1 = vec(merged[a1], pn, fld, fn)
                g, l, p = mcnemar_exact(v0, v1)
                dp = 100.0 * (sum(v1) - sum(v0)) / len(v0)
                noise = " [NOISE <6.0pp]" if (n300 and abs(dp) < 6.0) else ""
                print(f"  {mlabel:30s}: {sum(v0):3d} -> {sum(v1):3d}/{len(v0)} ({dp:+5.1f}pp, +{g}/-{l}, p={p:.3g}){noise}")
            agg0 = [x for pn in ("extract_ret", "relation_ret", "reverse_ret", "compare_ret", "temporal_ret") for x in vec(merged[a0], pn, "k4")]
            agg1 = [x for pn in ("extract_ret", "relation_ret", "reverse_ret", "compare_ret", "temporal_ret") for x in vec(merged[a1], pn, "k4")]
            g, l, p = mcnemar_exact(agg0, agg1)
            dp = 100.0 * (sum(agg1) - sum(agg0)) / 1000.0
            print(f"  {'Aggregate Ret K4 (n=1000)':30s}: {sum(agg0):3d} -> {sum(agg1):3d}/1000 ({dp:+5.1f}pp, +{g}/-{l}, p={p:.3g})")

    d1 = analyze_d1()
    if d1:
        print("\n=== FROZEN D1 200 CELLS (n=200 per cell, K=4 chance 25%, Exact K=8192 chance ~0%) ===")
        for cell, md in d1.items():
            print(f"  [{cell}]")
            for mn, s in md.items():
                print(f"    {mn:8s}: K4={s['k4']:3d}/200 ({s['k4_pct']:5.1f}%, +{s['k4_g']}/-{s['k4_l']} p={s['k4_p']:.3g}) | Exact={s['exact']:3d}/200 ({s['exact_pct']:5.1f}%, +{s['ex_g']}/-{s['ex_l']} p={s['ex_p']:.3g})")

    summary_out = {"gates": gate_results, "d1": d1}
    sel = analyze_selection()
    if sel:
        summary_out["selection"] = sel
        print("\n=== MISSION SELECTION DIAGNOSTIC (info only, n=500, K=4 chance 25%) ===")
        for cat, md in sel.items():
            parts = []
            for mn, s in md.items():
                parts.append(f"{mn}={s['k']}/{s['n']}({s['pct']:.1f}%,p={s['p']:.3g})" if mn != "r7" else f"r7={s['k']}/{s['n']}")
            print(f"  {cat:10s}: " + " | ".join(parts))
    if len(arms_final) == 4:
        any_non_stop = any(gate_results[a]["NonStopPass"] for a in arms_final)
        summary_out["stop_rule"] = {
            "any_non_stop_pass": any_non_stop,
            "passing_arms": [a for a in arms_final if gate_results[a]["NonStopPass"]],
            "full_promo_arms": [a for a in arms_final if gate_results[a]["FullPromoPass"]],
            "verdict": "PROCEED_PROPOSAL_ONLY" if any_non_stop else "STOP_STAGE2_RETRIEVAL_LINE",
        }
        print(f"\n=== STOP RULE VERDICT: {summary_out['stop_rule']} ===")

    (ROOT / "p4_summary.json").write_text(json.dumps(summary_out, indent=2) + "\n")


if __name__ == "__main__":
    main()
