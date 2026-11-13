#!/usr/bin/env python3
"""Prepare strict zero-leakage exclusion and subword assets for Pilot 4."""
import glob
import json
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
P4_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(ROOT, "data", "runs", "retrieval", "p4-retention")

TEMPLATE_STOP = {
    "the", "who", "where", "which", "whose", "what", "how", "why", "when", "in",
    "of", "by", "at", "then", "next", "after", "before", "for", "according", "as",
    "everyone", "rumor", "report", "from", "all", "both", "most", "least", "first",
    "last", "right", "is", "are", "was", "were", "does", "did", "has", "have",
    "he", "she", "it", "one", "two", "three", "four", "on", "to", "a", "an",
    "and", "lives", "lived", "years", "house", "grew", "up", "still", "there",
    "resides", "place", "works", "earns", "living", "trade", "job", "occupation",
    "employed", "owns", "pet", "takes", "care", "keeps", "home", "animal", "kind",
    "favorite", "color", "likes", "best", "loves", "always", "picks", "colors",
    "prefers", "adores", "fond", "plays", "practices", "day", "every", "learning",
    "instrument", "performs", "mastered", "band", "associated", "old", "age",
    "just", "turned", "reached", "aged", "born", "ago", "points", "scored",
    "finished", "game", "earned", "got", "total", "cm", "tall", "stands",
    "height", "measures", "started", "went", "stopped", "ended", "today", "now",
    "already", "also", "usually", "really", "once", "later", "finally", "taught",
    "gave", "lessons", "hired", "called", "thanked", "visited", "helped",
    "person", "town", "fewest", "smallest", "lowest", "youngest", "shortest",
    "oldest", "tallest", "largest", "highest", "biggest", "greatest", "many",
    "knows", "agrees", "says", "record", "seems", "accounts", "hear", "again",
    "usual", "moment", "days", "these", "happens", "far", "note", "arrived",
    "assist", "assisted", "done", "express", "expressed", "finish", "get",
    "gratitude", "had", "head", "headed", "help", "live", "measure", "morning",
    "own", "owner", "paid", "phoned", "play", "prefer", "prior", "rang",
    "recruit", "recruited", "residence", "score", "see", "start", "studied",
    "tutored", "visit", "whom", "with", "work", "s",
}

def extract_jsonl_entities(paths):
    exact = set()
    for p in sorted(glob.glob(paths)):
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                for o in r.get("options", []):
                    s = str(o).strip()
                    if s and not s.isdigit():
                        exact.add(s)
                at = r.get("answer_text")
                if at is not None:
                    s = str(at).strip()
                    if s and not s.isdigit():
                        exact.add(s)
                for txt in (r.get("passage", ""), r.get("question", "")):
                    for w in re.findall(r"[A-Za-z]+", txt):
                        if w.lower() not in TEMPLATE_STOP and len(w) >= 2:
                            exact.add(w)
    return exact

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sp = json.load(open(os.path.join(P4_DIR, "subword_partition.json"), encoding="utf-8"))
    man = json.load(open(os.path.join(P4_DIR, "probe_suite_manifest.json"), encoding="utf-8"))

    probe_ents = extract_jsonl_entities(os.path.join(P4_DIR, "probes", "*.jsonl"))
    d1_ents = extract_jsonl_entities(os.path.join(ROOT, "docs", "results", "stage2", "diagnostics-20261113", "d1*.jsonl"))
    with open(os.path.join(ROOT, "docs", "results", "stage2", "mission-20261111", "pilot2", "d1-entities.txt"), encoding="utf-8") as fh:
        d1_ents |= {line.strip() for line in fh if line.strip()}

    with open(os.path.join(ROOT, "docs", "results", "stage2", "mission-20261111", "pilot3", "excluded-extra.txt"), encoding="utf-8") as fh:
        ex_extra = {line.strip() for line in fh if line.strip()}

    recomb_tr = set(man["reserved_recomb_tr_words"])
    synth_towns = set(man["synth_towns_eval_pool"])
    fw_ev = {x["word"] for x in sp["fw_ev_only_2tok"]} | {x["word"] for x in sp["fw_ev_only_3tok"]}

    forbid = probe_ents | d1_ents | ex_extra | recomb_tr | synth_towns | fw_ev
    forbid_lower = sorted({x.lower() for x in forbid})
    forbid_exact = sorted(forbid)

    fw_tr_2tok = [x["word"] for x in sp["fw_tr_only_2tok"] if x["word"].lower() not in set(forbid_lower)]
    fw_tr_3tok = [x["word"] for x in sp["fw_tr_only_3tok"] if x["word"].lower() not in set(forbid_lower)]
    assert len(fw_tr_2tok) == len(sp["fw_tr_only_2tok"]), "fw_tr_2tok overlap!"
    assert len(fw_tr_3tok) == len(sp["fw_tr_only_3tok"]), "fw_tr_3tok overlap!"

    eval_heldout_ids = sorted({
        (tok["id"] if isinstance(tok, dict) else int(tok))
        for cat in sp["eval_heldout"].values()
        for tok in cat
    })
    assert len(eval_heldout_ids) == 845

    train_copy = {
        cat: [(tok["id"] if isinstance(tok, dict) else int(tok)) for tok in toks]
        for cat, toks in sp["train_copy"].items()
    }

    payload = {
        "schema": "asem-pilot4-assets-1",
        "counts": {
            "probe_entities": len(probe_ents),
            "d1_entities": len(d1_ents),
            "forbid_exact": len(forbid_exact),
            "forbid_lower": len(forbid_lower),
            "eval_heldout_ids": len(eval_heldout_ids),
            "fw_tr_2tok": len(fw_tr_2tok),
            "fw_tr_3tok": len(fw_tr_3tok),
            "train_copy": {k: len(v) for k, v in train_copy.items()},
        },
        "probe_and_d1_entities": sorted(probe_ents | d1_ents),
        "forbid_exact": forbid_exact,
        "forbid_lower": forbid_lower,
        "eval_heldout_ids": eval_heldout_ids,
        "fw_tr_2tok": fw_tr_2tok,
        "fw_tr_3tok": fw_tr_3tok,
        "train_copy": train_copy,
    }
    out_path = os.path.join(OUT_DIR, "pilot4-assets.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(json.dumps(payload["counts"], indent=2))
    print("wrote", out_path)

if __name__ == "__main__":
    main()
