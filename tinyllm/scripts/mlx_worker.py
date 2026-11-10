#!/usr/bin/env python3
"""MLX LoRA worker for tinyllm_engine_port (protocol: see src/tinyllm_engine_port.erl).

UNTESTED against MLX. Only the Python syntax and the argument handling are checked; the `mlx_lm.lora` flags,
its output lines ("Train loss", "Test loss") and the adapter file names below come from memory of mlx-lm and
may differ in your installed version. Check each operation by hand on a tiny model before trusting a tuner run:

  python3 scripts/mlx_worker.py eval --base-model Qwen/Qwen2.5-0.5B-Instruct --adapter none --data heldout.txt

Needs: pip install mlx-lm. An adapter is a directory holding adapters.safetensors and adapter_config.json.
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile


def lines_of(path):
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def write_jsonl(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(json.dumps({"text": ln}, ensure_ascii=False) + "\n")


def run_lora(args):
    cmd = [sys.executable, "-m", "mlx_lm.lora"] + args
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = p.stdout + p.stderr
    sys.stdout.write(out[-4000:] + "\n")
    if p.returncode != 0:
        sys.exit(p.returncode)
    return out


def last_float(pattern, text):
    found = re.findall(pattern, text)
    if not found:
        sys.stderr.write("no match for %r in worker output\n" % pattern)
        sys.exit(4)
    return float(found[-1])


def op_train(a):
    os.makedirs(a.adapter_out, exist_ok=True)
    samples = lines_of(a.data)
    data_dir = os.path.join(a.adapter_out, "data")
    os.makedirs(data_dir, exist_ok=True)
    for name in ("train", "valid", "test"):
        write_jsonl(os.path.join(data_dir, name + ".jsonl"), samples)
    cfg = os.path.join(a.adapter_out, "lora.yaml")
    with open(cfg, "w") as f:
        f.write("lora_parameters:\n  rank: %d\n  scale: 20.0\n  dropout: 0.0\n" % a.rank)
    args = ["--model", a.base_model, "--train", "--data", data_dir, "--iters", str(a.steps),
            "--learning-rate", repr(a.lr), "--batch-size", str(min(4, len(samples))),
            "--steps-per-report", "1", "--steps-per-eval", str(a.steps + 1), "--adapter-path", a.adapter_out,
            "-c", cfg]
    if a.adapter_in != "none":
        for name in ("adapters.safetensors", "adapter_config.json"):
            shutil.copy(os.path.join(a.adapter_in, name), os.path.join(a.adapter_out, name))
        args += ["--resume-adapter-file", os.path.join(a.adapter_out, "adapters.safetensors")]
    out = run_lora(args)
    print("RESULT train_loss=%.8g" % last_float(r"Train loss ([0-9.eE+-]+)", out))


def op_eval(a):
    samples = lines_of(a.data)
    with tempfile.TemporaryDirectory() as d:
        for name in ("train", "valid", "test"):
            write_jsonl(os.path.join(d, name + ".jsonl"), samples)
        args = ["--model", a.base_model, "--data", d, "--test", "--batch-size", str(min(4, len(samples)))]
        if a.adapter != "none":
            args += ["--adapter-path", a.adapter]
        out = run_lora(args)
    print("RESULT loss=%.8g" % last_float(r"Test loss ([0-9.eE+-]+)", out))


def op_distance(a):
    import mlx.core as mx

    def load(p):
        return {} if p == "none" else mx.load(os.path.join(p, "adapters.safetensors"))

    x, y = load(a.a), load(a.b)
    total = 0.0
    for k in set(x) | set(y):
        zero = None
        u = x.get(k)
        v = y.get(k)
        ref = u if u is not None else v
        zero = mx.zeros_like(ref)
        d = (v if v is not None else zero) - (u if u is not None else zero)
        total += float(mx.sum(d * d).item())
    print("RESULT dist=%.8g" % math.sqrt(total))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("op", choices=["train", "eval", "distance"])
    ap.add_argument("--base-model", default="")
    ap.add_argument("--adapter-in", default="none")
    ap.add_argument("--adapter-out")
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--data")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--a", default="none")
    ap.add_argument("--b", default="none")
    a = ap.parse_args()
    {"train": op_train, "eval": op_eval, "distance": op_distance}[a.op](a)


if __name__ == "__main__":
    main()
