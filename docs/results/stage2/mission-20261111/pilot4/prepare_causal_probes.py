#!/usr/bin/env python3
"""Prepare causal entity-swap items (n=157 pairs) and token/word lists for Pilot 4 causal micro-probes."""
import json, re, struct
from collections import Counter

BASE = "docs/results/stage2/mission-20261111"
AUDIT_PATH = f"{BASE}/pilot4/audit_raw.json"
D1_PATH = "docs/results/stage2/diagnostics-20261113/d1-rev-pf.jsonl"
OUT_PATH = f"{BASE}/pilot4/causal_probe_items.json"

audit = json.load(open(AUDIT_PATH))
toks = audit["vocab_tokens"]
tm = {t["id"]: t for t in toks}

# Load non-test exclusion strings to keep FineWeb candidates disjoint from dev/D1/P3-exclusions
d1_entities = {line.strip() for line in open(f"{BASE}/pilot2/d1-entities.txt") if line.strip()}
extra_excl = {line.strip() for line in open(f"{BASE}/pilot3/excluded-extra.txt") if line.strip()}
dev_entities = set()
for line in open("lisp/eval/qa/dev.jsonl"):
    obj = json.loads(line)
    for o in obj.get("options", []):
        dev_entities.add(o.strip())
    for w in re.findall(r"[A-Za-z]+", obj.get("passage", "")):
        if w[0].isupper():
            dev_entities.add(w)

# Token lists for D1 names
d1_toks = {
    "Adam": [1716, 337], "Alba": [1002, 7887], "Andre": [1475, 262], "Aster": [316, 2834],
    "Ben": [6130], "Carla": [1604, 357], "Colin": [1312, 260], "Dev": [1492, 119],
    "Edith": [2730, 349], "Emma": [3257, 2026], "Eva": [5448, 98], "Finn": [431, 6763],
    "Gene": [1707, 102], "Gus": [438, 384], "Hugo": [403, 6378], "Ines": [517, 272],
    "Jack": [468, 689], "Jean": [468, 2950], "Kai": [638, 2863], "Kira": [638, 326, 98],
    "Lena": [455, 3712], "Lou": [3007], "Mae": [364, 1953], "Mia": [364, 523],
    "Neil": [1939, 301], "Noah": [3176, 1372], "Otto": [491, 117, 1376], "Pedro": [374, 273, 305],
    "Raj": [413, 1047], "Rosa": [413, 7344], "Sara": [335, 3562], "Sofia": [1169, 103, 523],
    "Tess": [310, 409], "Una": [644, 98], "Wade": [389, 642], "Yusuf": [821, 384, 118, 103],
}

hi_pool_2 = ["Lena", "Mia", "Neil", "Rosa", "Sara", "Kai", "Gus", "Dev", "Jean"]
hi_pool_3 = ["Kira", "Otto", "Pedro"]
z0_pool = ["Edith", "Adam", "Alba", "Eva", "Andre", "Emma", "Una"]
zt_pool = ["Finn", "Hugo", "Tess", "Wade", "Jack", "Mae", "Noah", "Raj"]

d1_items = [json.loads(line) for line in open(D1_PATH)]
swap_pairs = []
hi_idx = 0
z0_idx = 0
zt_idx = 0

for idx, it in enumerate(d1_items):
    gold = it["options"][it["answer"]]
    ids = d1_toks[gold]
    p3m = min(tm[i]["p3_all"] for i in ids)
    r7m = min(tm[i]["r7_ret_all"] for i in ids)
    if p3m >= 100 and r7m >= 100:
        stratum = "all_ge_100"
    elif p3m == 0 and r7m == 0:
        stratum = "zero_tok0" if tm[ids[0]]["p3_all"] == 0 else "zero_tail"
    else:
        continue  # mid_1_99 excluded from crisp 0 vs >=100 causal pair

    used = set(it["options"]) | set(re.findall(r"[A-Za-z]+", it["passage"]))
    if stratum.startswith("zero_"):
        cand_list = (hi_pool_3 + hi_pool_2) if len(ids) >= 3 else (hi_pool_2 + hi_pool_3)
        chosen = None
        for k in range(len(cand_list)):
            c = cand_list[(hi_idx + k) % len(cand_list)]
            if c not in used:
                chosen = c
                hi_idx = (hi_idx + k + 1) % len(cand_list)
                break
        assert chosen is not None
        direction = "zero_to_hi100"
    else:
        # Alternate targeting Tok0==0 vs Tok1+==0 replacements
        prefer_z0 = (len([x for x in swap_pairs if x["direction"] == "hi100_to_zero"]) % 2 == 0)
        cand_list = (z0_pool + zt_pool) if prefer_z0 else (zt_pool + z0_pool)
        ptr = z0_idx if prefer_z0 else zt_idx
        chosen = None
        for k in range(len(cand_list)):
            c = cand_list[(ptr + k) % len(cand_list)]
            if c not in used:
                chosen = c
                if prefer_z0:
                    z0_idx = (z0_idx + k + 1) % len(cand_list)
                else:
                    zt_idx = (zt_idx + k + 1) % len(cand_list)
                break
        assert chosen is not None
        direction = "hi100_to_zero"

    pat = re.compile(r"\b" + re.escape(gold) + r"\b")
    new_passage = pat.sub(chosen, it["passage"])
    assert new_passage != it["passage"]
    new_opts = list(it["options"])
    new_opts[it["answer"]] = chosen
    swap_pairs.append({
        "id": it["id"],
        "stratum": stratum,
        "direction": direction,
        "orig_gold": gold,
        "swap_gold": chosen,
        "swap_sub_regime": "tok0_zero" if chosen in z0_pool else ("tail_zero" if chosen in zt_pool else "all_ge_100"),
        "passage": it["passage"],
        "question": it["question"],
        "options": it["options"],
        "answer": it["answer"],
        "swap_passage": new_passage,
        "swap_options": new_opts,
    })

# Stratified induction token sets
hi_alpha_ids = sorted(t["id"] for t in toks if t["shape"] in ("space_title_alpha", "bare_lower_alpha") and t["p3_all"] >= 100 and t["r7_ret_all"] >= 100)
zero_fw_alpha_ids = sorted(t["id"] for t in toks if t["shape"] in ("space_title_alpha", "bare_lower_alpha") and t["fw_train"] >= 500 and t["r7_ret_all"] == 0 and t["p3_all"] == 0)

# Mine 150 clean 2-token capitalized FineWeb words with BOTH tokens 0 in r7_ret_all & p3_all
zero_title = {t["id"] for t in toks if t["shape"] == "space_title_alpha" and t["r7_ret_all"] == 0 and t["p3_all"] == 0}
zero_tail = {t["id"] for t in toks if t["shape"] == "bare_lower_alpha" and t["r7_ret_all"] == 0 and t["p3_all"] == 0}
data = open("data/fineweb/stage1/train_00000.bin", "rb").read()
arr = struct.unpack(f"<{len(data)//2}H", data)
bigrams = Counter()
for i in range(len(arr) - 2):
    a, b, c = arr[i], arr[i + 1], arr[i + 2]
    if a in zero_title and b in zero_tail:
        sc = tm[c]["str"]
        if not sc or not sc[0].isalpha():
            bigrams[(a, b)] += 1

fw_words = []
seen_w = set()
exclude_ci = {s.lower() for s in (d1_entities | extra_excl | dev_entities)}
for (a, b), cnt in bigrams.most_common(400):
    w = (tm[a]["str"] + tm[b]["str"]).strip()
    if cnt >= 15 and 4 <= len(w) <= 12 and w.lower() not in exclude_ci and w not in seen_w:
        seen_w.add(w)
        fw_words.append(w)
    if len(fw_words) >= 150:
        break

out = {
    "swap_pairs": swap_pairs,
    "induction_buckets": {
        "p3_r7_hi100_alpha": hi_alpha_ids,
        "zero_ret_high_fw_alpha": zero_fw_alpha_ids,
    },
    "fw_zero_ret_words": fw_words,
    "d1_zero_names": sorted(z0_pool + zt_pool),
}
json.dump(out, open(OUT_PATH, "w"), indent=2)
print(
    f"Wrote {OUT_PATH}: swap_pairs={len(swap_pairs)} "
    f"(zero_to_hi100={sum(x['direction']=='zero_to_hi100' for x in swap_pairs)}, "
    f"hi100_to_zero={sum(x['direction']=='hi100_to_zero' for x in swap_pairs)}), "
    f"hi_alpha_ids={len(hi_alpha_ids)}, zero_fw_alpha_ids={len(zero_fw_alpha_ids)}, fw_words={len(fw_words)}"
)
print("Sample fw_words[:20]:", fw_words[:20])
