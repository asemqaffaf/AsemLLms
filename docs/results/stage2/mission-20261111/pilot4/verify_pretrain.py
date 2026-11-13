#!/usr/bin/env python3
"""Independent pre-training verification for Pilot 4: exact dose, component identity, zero leakage."""
import glob
import hashlib
import json
import os
import re
import subprocess
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
P4_DIR = os.path.dirname(__file__)
RUN_DIR = os.path.join(ROOT, "data", "runs", "retrieval", "p4-retention")

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def main():
    assets = json.load(open(os.path.join(RUN_DIR, "pilot4-assets.json"), encoding="utf-8"))
    eval_heldout_ids = set(assets["eval_heldout_ids"])
    forbid_entities = set(assets["forbid_exact"]) | set(assets["forbid_lower"])
    forbid_lower = {w.lower() for w in forbid_entities}

    # Export decoded texts + id lists from SBCL for all 4 arms and heldout-strings
    with tempfile.NamedTemporaryFile(suffix=".jsonl", dir="/tmp", delete=False) as tf_d:
        dump_path = tf_d.name
    with tempfile.NamedTemporaryFile(suffix=".json", dir="/tmp", delete=False) as tf_hs:
        hs_path = tf_hs.name

    with tempfile.NamedTemporaryFile(suffix=".lisp", dir="/tmp", delete=False, mode="w", encoding="utf-8") as tf_l:
        lisp_path = tf_l.name
        tf_l.write(f'''
(dolist (f '("util" "sha256" "json" "bpe" "data" "qagen" "retrieval"))
  (load (format nil "lisp/src/~a.lisp" f)))
(let ((bpe (asem-bpe:load-bpe "data/fineweb/stage1/tokenizer.bpe"))
      (hs nil)
      (id-map (make-array 8192 :initial-element "")))
  (maphash (lambda (k v) (declare (ignore v)) (push k hs)) (asem-retrieval::heldout-strings))
  (dotimes (i (asem-bpe:bpe-vocab-size bpe))
    (setf (aref id-map i) (string-trim (list (code-char 32) (code-char 9) (code-char 10) (code-char 13))
                                       (asem-bpe:bpe-decode bpe (list i)))))
  (with-open-file (o {json.dumps(hs_path)} :direction :output :if-exists :supersede :external-format :utf-8)
    (asem-json:write-json (cons :obj (list (cons "heldout_strings" (coerce hs 'vector))
                                           (cons "vocab_trimmed" id-map)))
                          o))
  (let ((all-rows nil))
    (dolist (arm '("a" "b" "c" "d"))
      (let ((rows (with-open-file (in (format nil "{RUN_DIR}/data-train-~a.sexp" arm))
                    (let ((*read-eval* nil)) (read in)))))
        (loop for (kind ids rws tgts) in rows for i from 0 do
          (let ((txt (if (eq kind :copy) "" (asem-bpe:bpe-decode bpe ids))))
            (push (cons :obj (list (cons "arm" arm)
                                   (cons "idx" i)
                                   (cons "kind" (string-downcase (symbol-name kind)))
                                   (cons "ids" (coerce ids 'vector))
                                   (cons "tgts" (coerce tgts 'vector))
                                   (cons "text" txt)))
                  all-rows)))))
    (with-open-file (o {json.dumps(dump_path)} :direction :output :if-exists :supersede :external-format :utf-8)
      (asem-json:write-json (coerce (nreverse all-rows) 'vector) o))))
''')
    try:
        res = subprocess.run(
            ["sbcl", "--noinform", "--script", lisp_path],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if res.returncode != 0:
            raise RuntimeError(f"SBCL failed ({res.returncode}): {res.stderr[-800:]}")
        hs_obj = json.load(open(hs_path, encoding="utf-8"))
        hs_list = hs_obj["heldout_strings"]
        vocab_trimmed = hs_obj["vocab_trimmed"]
        forbid_lower |= {w.lower() for w in hs_list if w}
        forbid_single_tok_ids = {
            i for i, s in enumerate(vocab_trimmed)
            if s and s.isalpha() and s.lower() in forbid_lower
        }

        arm_recs = {"a": [], "b": [], "c": [], "d": []}
        eh_hits = {"a": 0, "b": 0, "c": 0, "d": 0}
        word_leaks = {"a": [], "b": [], "c": [], "d": []}
        guard_leaks = {"b": [], "d": []}
        with open(dump_path, encoding="utf-8") as fh:
            for r in json.load(fh):
                arm = r["arm"]
                arm_recs[arm].append(r)
                all_tids = set(r["ids"]) | set(r["tgts"])
                if all_tids & eval_heldout_ids:
                    eh_hits[arm] += 1
                if r["kind"] != "copy":
                    for w in re.findall(r"[A-Za-z]+", r["text"]):
                        if w.lower() in forbid_lower:
                            word_leaks[arm].append((r["idx"], w))
                else:
                    # random-token induction guard: zero eval_heldout AND zero single-token forbidden entity IDs
                    bad_ids = all_tids & (eval_heldout_ids | forbid_single_tok_ids)
                    if bad_ids:
                        guard_leaks[arm].append((r["idx"], list(bad_ids)))

        sigs = {}
        for arm, recs in arm_recs.items():
            sigs[arm] = [(tuple(r["ids"]), tuple(r["tgts"])) for r in recs]

        set_a = set(sigs["a"])
        assert len(sigs["a"]) == 6080 and len(set_a) == 6080, f"Arm A unique count {len(set_a)}"
        copy_in_b = [s for s in sigs["b"] if s not in set_a]
        replay_in_c = [s for s in sigs["c"] if s not in set_a]
        assert set_a.issubset(set(sigs["b"])) and len(copy_in_b) == 2560
        assert set_a.issubset(set(sigs["c"])) and len(replay_in_c) == 7360
        assert set_a.issubset(set(sigs["d"]))
        assert set(copy_in_b).issubset(set(sigs["d"]))
        assert set(replay_in_c).issubset(set(sigs["d"]))
        assert len(sigs["d"]) == 6080 + 2560 + 7360

        # Component-level leak report on Arm D
        set_copy = set(copy_in_b)
        set_replay = set(replay_in_c)
        comp_stats = {
            "ans_before": {"n": 0, "eh_id_leaks": 0, "word_leaks": 0},
            "copy_prose": {"n": 0, "eh_id_leaks": 0, "word_leaks": 0},
            "copy_guard": {"n": 0, "eh_id_leaks": 0, "word_leaks": 0},
            "replay":     {"n": 0, "eh_id_leaks": 0, "word_leaks": 0},
        }
        for r in arm_recs["d"]:
            sig = (tuple(r["ids"]), tuple(r["tgts"]))
            if sig in set_a:
                comp = "ans_before"
            elif sig in set_copy:
                comp = "copy_guard" if r["kind"] == "copy" else "copy_prose"
            else:
                comp = "replay"
            comp_stats[comp]["n"] += 1
            if (set(r["ids"]) | set(r["tgts"])) & eval_heldout_ids:
                comp_stats[comp]["eh_id_leaks"] += 1
            if r["kind"] == "copy" and ((set(r["ids"]) | set(r["tgts"])) & forbid_single_tok_ids):
                comp_stats[comp]["eh_id_leaks"] += 1
            if r["kind"] != "copy":
                for w in re.findall(r"[A-Za-z]+", r["text"]):
                    if w.lower() in forbid_lower:
                        comp_stats[comp]["word_leaks"] += 1

        report = {
            "arm_counts": {k: len(v) for k, v in arm_recs.items()},
            "forbid_single_tok_ids_checked": len(forbid_single_tok_ids),
            "exact_component_sharing": {
                "a_subset_of_bcd": True,
                "b_copy_identical_in_d": len(copy_in_b),
                "c_replay_identical_in_d": len(replay_in_c),
            },
            "component_leakage": comp_stats,
            "eval_heldout_id_hits": eh_hits,
            "guard_forbid_id_hits": {k: len(v) for k, v in guard_leaks.items()},
            "word_leaks": {k: len(v) for k, v in word_leaks.items()},
            "sha256": {
                f: sha256_file(os.path.join(RUN_DIR, f))
                for f in ["pilot4-assets.json", "data-train-a.sexp", "data-train-b.sexp",
                          "data-train-c.sexp", "data-train-d.sexp", "internal-dev-200.sexp", "data-audit.json"]
            },
            "report_files": {
                "pilot3_results": sha256_file(os.path.join(ROOT, "docs/results/stage2/mission-20261111/pilot3/RESULTS.md")),
                "r7_exp1_canon": sha256_file(os.path.join(P4_DIR, "r7_exp1_canon.json")),
                "data_audit_md": sha256_file(os.path.join(P4_DIR, "DATA_AUDIT.md")),
                "eval_spec_md": sha256_file(os.path.join(P4_DIR, "eval-spec.md")),
                "init_ckpt_r7": sha256_file(os.path.join(ROOT, "data/runs/retrieval/r7-L123-aug-opts-rev-rsn/last.ckpt")),
            }
        }
        print(json.dumps(report, indent=2))
        for k in eh_hits:
            assert eh_hits[k] == 0
            assert len(word_leaks[k]) == 0
        for c, st in comp_stats.items():
            assert st["eh_id_leaks"] == 0 and st["word_leaks"] == 0, f"Leak in {c}: {st}"
    finally:
        for p in (dump_path, hs_path, lisp_path):
            if os.path.exists(p):
                os.unlink(p)

if __name__ == "__main__":
    main()
