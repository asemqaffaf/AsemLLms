#!/usr/bin/env python3
"""Measure what the tuner's loss numbers cannot: answers and retained knowledge.

  python3 scripts/capability_eval.py MODEL_DIR [--adapter DIR] [--sets DIR] [--n N]

  reply_f1    for the first User/Assistant turn of N heldout dialogues, greedy-generate the reply and score token F1
              against the real one (the heldout dialogues share no user question with train)
  knowledge   accuracy on eval/knowledge_mc.jsonl (40 general-knowledge questions, 4 options): the option with the
              highest mean log-probability after "Question: ... Answer:" wins (chance 25%). A 4B model is near 100%, so
              accuracy saturates; knowledge_margin (mean logprob of the right option minus the best wrong one) is the
              continuous number that moves when knowledge degrades.
Run it once without --adapter (the base) and once with, and compare. Prints one JSON line.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

import mlx.core as mx
from mlx_lm import generate, load

HERE = os.path.dirname(os.path.abspath(__file__))
MC = os.path.join(os.path.dirname(HERE), "eval", "knowledge_mc.jsonl")


def toks(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def f1(pred, ref):
    p, r = toks(pred), toks(ref)
    common = sum((Counter(p) & Counter(r)).values())
    if not p or not r or common == 0:
        return 0.0
    pr, rc = common / len(p), common / len(r)
    return 2 * pr * rc / (pr + rc)


def first_turn(d):
    m = re.match(r"User: (.*?) Assistant: (.*?)(?= User:|$)", d)
    return (m.group(1), m.group(2)) if m else None


def reply_f1(model, tok, heldout, n):
    scores, shown = [], []
    for d in heldout:
        t = first_turn(d)
        if not t:
            continue
        q, ref = t
        out = generate(model, tok, prompt="User: %s Assistant:" % q, max_tokens=40, verbose=False)
        pred = re.split(r"\n| User:", out.strip())[0].strip()
        scores.append(f1(pred, ref))
        if len(shown) < 3:
            shown.append({"q": q, "ref": ref, "pred": pred})
        if len(scores) == n:
            break
    return sum(scores) / len(scores), len(scores), shown


def option_logprob(model, tok, prompt, option):
    p = tok.encode(prompt)
    o = tok.encode(" " + option)
    ids = mx.array([p + o])
    logits = model(ids[:, :-1])
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tgt = ids[:, 1:]
    picked = mx.take_along_axis(lp, tgt[..., None], axis=-1)[0, :, 0]
    return float(mx.mean(picked[len(p) - 1:]).item())


def knowledge(model, tok):
    items = [json.loads(l) for l in open(MC) if l.strip()]
    right, margins = 0, []
    for it in items:
        prompt = "Question: %s\nAnswer:" % it["q"]
        s = [option_logprob(model, tok, prompt, o) for o in it["options"]]
        right += int(max(range(len(s)), key=s.__getitem__) == it["answer"])
        margins.append(s[it["answer"]] - max(v for i, v in enumerate(s) if i != it["answer"]))
    return right / len(items), sum(margins) / len(margins), len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--adapter")
    ap.add_argument("--sets", default="")
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    model, tok = load(a.model, adapter_path=a.adapter) if a.adapter else load(a.model)
    out = {"adapter": a.adapter or "none"}
    out["knowledge_acc"], out["knowledge_margin"], out["knowledge_n"] = knowledge(model, tok)
    if a.sets:
        heldout = [l.rstrip("\n") for l in open(os.path.join(a.sets, "heldout.txt")) if l.strip()]
        out["reply_f1"], out["reply_n"], out["examples"] = reply_f1(model, tok, heldout, a.n)
    json.dump(out, sys.stdout, ensure_ascii=False)
    print()


if __name__ == "__main__":
    main()
