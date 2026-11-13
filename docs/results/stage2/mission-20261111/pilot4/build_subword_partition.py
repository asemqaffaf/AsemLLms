#!/usr/bin/env python3
"""Build deterministic 75/25 train_copy vs eval_heldout partition of zero-retrieval FineWeb BPE tokens and mine disjoint multi-token words across Stage-1 FineWeb shards."""
import glob
import json
import os
import re
import struct
from collections import Counter

BASE = "docs/results/stage2/mission-20261111"
AUDIT_PATH = f"{BASE}/pilot4/audit_raw.json"
OUT_PATH = f"{BASE}/pilot4/subword_partition.json"

audit = json.load(open(AUDIT_PATH))
toks = audit["vocab_tokens"]
tm = {t["id"]: t for t in toks}

# Build non-test exclusion set (D1, P3 excluded-extra, qa dev.jsonl, selection.sexp)
excl_ci = set()
for p in (f"{BASE}/pilot2/d1-entities.txt", f"{BASE}/pilot3/excluded-extra.txt"):
    with open(p) as f:
        for line in f:
            s = line.strip()
            if s:
                excl_ci.add(s.lower())

for p in (
    "lisp/eval/qa/dev.jsonl",
    "docs/results/stage2/diagnostics-20261113/d1-rev-pf.jsonl",
    "docs/results/stage2/diagnostics-20261113/d1-rev-vf.jsonl",
    "docs/results/stage2/diagnostics-20261113/d1-extract-pf.jsonl",
):
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                obj = json.loads(line)
                for o in obj.get("options", []):
                    excl_ci.add(o.strip().lower())
                for w in re.findall(r"[A-Za-z]+", obj.get("passage", "")):
                    excl_ci.add(w.lower())

# Exclude any capitalized entities / options present in selection.sexp
sel_txt = open(f"{BASE}/selection.sexp").read()
for m in re.findall(r'"([A-Za-z][A-Za-z\- ]{1,30})"', sel_txt):
    for w in m.split():
        if w.isalpha() and len(w) >= 2:
            excl_ci.add(w.lower())

target_shapes = ("space_title_alpha", "bare_lower_alpha", "space_lower_alpha", "bare_title_alpha")
partition = {"train_copy": {}, "eval_heldout": {}}
tr_sets = {}
ev_sets = {}

for sh in target_shapes:
    cands = [
        t for t in toks
        if t["shape"] == sh and t["fw_train"] > 0 and t["r7_ret_all"] == 0 and t["p3_all"] == 0
    ]
    # Sort descending by fw_train, tie-break by token id for exact determinism
    cands.sort(key=lambda t: (-t["fw_train"], t["id"]))
    tr_ids = []
    ev_ids = []
    for idx, t in enumerate(cands):
        if idx % 4 == 3:
            ev_ids.append(t["id"])
        else:
            tr_ids.append(t["id"])
    partition["train_copy"][sh] = tr_ids
    partition["eval_heldout"][sh] = ev_ids
    tr_sets[sh] = set(tr_ids)
    ev_sets[sh] = set(ev_ids)

# Also partition D1 names to verify accidental split (without cherry-picking)
d1_pf_items = [json.loads(line) for line in open("docs/results/stage2/diagnostics-20261113/d1-rev-pf.jsonl")]
d1_zero_ids = sorted({
    tid for t in toks
    if t["d1_names_tok_types"] > 0 and t["r7_ret_all"] == 0 and t["p3_all"] == 0 and t["shape"] in target_shapes
    for tid in [t["id"]]
})
all_ev_id_set = set().union(*ev_sets.values())
all_tr_id_set = set().union(*tr_sets.values())

# Scan all Stage-1 FineWeb shards for multi-token TitleCase words strictly within tr_only or ev_only
shard_paths = sorted(glob.glob("data/fineweb/stage1/train_*.bin"))
tr_bi, ev_bi = Counter(), Counter()
tr_tri, ev_tri = Counter(), Counter()
tr_all_title = tr_sets["space_title_alpha"]
tr_all_tail = tr_sets["bare_lower_alpha"]
ev_all_title = ev_sets["space_title_alpha"]
ev_all_tail = ev_sets["bare_lower_alpha"]

for sp in shard_paths:
    data = open(sp, "rb").read()
    arr = struct.unpack(f"<{len(data)//2}H", data)
    n = len(arr)
    for i in range(n - 3):
        a, b, c = arr[i], arr[i + 1], arr[i + 2]
        if a in tr_all_title and b in tr_all_tail:
            sc = tm[c]["str"]
            if not sc or not sc[0].isalpha():
                tr_bi[(a, b)] += 1
            elif c in tr_all_tail:
                sd = tm[arr[i + 3]]["str"]
                if not sd or not sd[0].isalpha():
                    tr_tri[(a, b, c)] += 1
        elif a in ev_all_title and b in ev_all_tail:
            sc = tm[c]["str"]
            if not sc or not sc[0].isalpha():
                ev_bi[(a, b)] += 1
            elif c in ev_all_tail:
                sd = tm[arr[i + 3]]["str"]
                if not sd or not sd[0].isalpha():
                    ev_tri[(a, b, c)] += 1


def extract_words(counter, min_cnt, max_items):
    out = []
    seen = set()
    for tup, cnt in counter.most_common():
        if cnt < min_cnt:
            break
        raw = "".join(tm[i]["str"] for i in tup)
        w = raw.strip()
        if (
            4 <= len(w) <= 13
            and re.fullmatch(r"[A-Z][a-z]+", w)
            and w.lower() not in excl_ci
            and w.lower() not in seen
        ):
            seen.add(w.lower())
            out.append({"word": w, "ids": list(tup), "fw_count": cnt, "vowel_init": w[0] in "AEIOU"})
            if len(out) >= max_items:
                break
    return out


fw_tr_2 = extract_words(tr_bi, 3, 1500)
fw_ev_2 = extract_words(ev_bi, 1, 500)
fw_tr_3 = extract_words(tr_tri, 3, 600)
fw_ev_3 = extract_words(ev_tri, 1, 200)

out = {
    "schema": "asem-pilot4-subword-partition-1",
    "rule": "For each shape in fw_train>0 & r7_ret_all==0 & p3_all==0 sorted by (-fw_train, id): idx%4==3 -> eval_heldout (25%), else -> train_copy (75%).",
    "shards_scanned": shard_paths,
    "train_copy": partition["train_copy"],
    "eval_heldout": partition["eval_heldout"],
    "d1_zero_ret_token_split": {
        "total": len(d1_zero_ids),
        "in_train_copy": sum(1 for tid in d1_zero_ids if tid in all_tr_id_set),
        "in_eval_heldout": sum(1 for tid in d1_zero_ids if tid in all_ev_id_set),
    },
    "fw_tr_only_2tok": fw_tr_2,
    "fw_ev_only_2tok": fw_ev_2,
    "fw_tr_only_3tok": fw_tr_3,
    "fw_ev_only_3tok": fw_ev_3,
}

with open(OUT_PATH, "w") as f:
    json.dump(out, f, indent=2)

for sh in target_shapes:
    print(f"{sh:20s}: train_copy={len(partition['train_copy'][sh]):4d} | eval_heldout={len(partition['eval_heldout'][sh]):4d}")
print(
    f"Mined words: tr_2tok={len(fw_tr_2)} (vowel={sum(x['vowel_init'] for x in fw_tr_2)}), "
    f"ev_2tok={len(fw_ev_2)} (vowel={sum(x['vowel_init'] for x in fw_ev_2)}), "
    f"tr_3tok={len(fw_tr_3)} (vowel={sum(x['vowel_init'] for x in fw_tr_3)}), "
    f"ev_3tok={len(fw_ev_3)} (vowel={sum(x['vowel_init'] for x in fw_ev_3)})"
)
print("D1 zero-ret tokens accidental split:", out["d1_zero_ret_token_split"])
print("Sample ev_2tok[:15]:", [x["word"] for x in fw_ev_2[:15]])
print("Sample ev_2tok vowel[:10]:", [x["word"] for x in fw_ev_2 if x["vowel_init"]][:10])
