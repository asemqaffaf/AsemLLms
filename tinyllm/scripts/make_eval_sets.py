#!/usr/bin/env python3
"""Build disjoint train / heldout / garbage sets from the repo's chat data, one dialogue per line.

  python3 scripts/make_eval_sets.py OUT_DIR [--train N] [--heldout N] [--seed S]
    ->  OUT_DIR/train.txt     dialogues the tuner learns from
        OUT_DIR/heldout.txt   dialogues never trained on, sharing no user question with train
        OUT_DIR/garbage.txt   the train dialogues with every word replaced by a random vocabulary word, so format
                              and word statistics stay but the content is meaningless (a control that must NOT commit)
        OUT_DIR/probe.txt     copy of eval/probe_general.txt: varied general text, unlike the chat domain

Splitting by user question (not by dialogue) matters: dataset/chat_clean.txt repeats exchanges across dialogues,
so a plain dialogue split leaks the same question/answer into both sets and a held-out "gain" is just memorising.
"""
import argparse
import os
import random
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CHAT = os.path.join(ROOT, "..", "dataset", "chat_clean.txt")
PROBE = os.path.join(ROOT, "eval", "probe_general.txt")


def dialogues(path):
    with open(path, encoding="utf-8") as f:
        blocks = [b.strip() for b in f.read().split("\n\n") if b.strip()]
    return [" ".join(b.split("\n")) for b in blocks]


def user_turns(d):
    return set(re.findall(r"User: (.*?)(?= Assistant:|$)", d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--train", type=int, default=40)
    ap.add_argument("--heldout", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    ds = sorted(set(dialogues(CHAT)))
    rnd.shuffle(ds)
    train, train_q = [], set()
    for d in ds:
        if len(train) == a.train:
            break
        train.append(d)
        train_q |= user_turns(d)
    held = []
    for d in ds:
        if len(held) == a.heldout:
            break
        if d not in train and not (user_turns(d) & train_q):
            held.append(d)
    if len(held) < a.heldout:
        raise SystemExit("only %d dialogues disjoint from train; lower --train or --heldout" % len(held))
    vocab = sorted({w for d in train for w in d.split()})
    garbage = [" ".join(rnd.choice(vocab) for _ in d.split()) for d in train]
    os.makedirs(a.out, exist_ok=True)
    for name, lines in (("train", train), ("heldout", held), ("garbage", garbage)):
        with open(os.path.join(a.out, name + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    shutil.copy(PROBE, os.path.join(a.out, "probe.txt"))
    print("train %d  heldout %d  garbage %d  probe %d  (shared user questions train/heldout: %d)" % (
        len(train), len(held), len(garbage), sum(1 for _ in open(PROBE)), len(train_q & set().union(*map(user_turns, held)))))


if __name__ == "__main__":
    main()
