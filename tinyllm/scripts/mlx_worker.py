#!/usr/bin/env python3
"""MLX LoRA worker for tinyllm_engine_port (protocol: see src/tinyllm_engine_port.erl).

Checked against mlx-lm 0.32.2 / mlx 0.32.4 on Linux CPU with a tiny local model (scripts/tiny_mlx_model.py).
NOT yet run on Apple silicon or on a real downloaded model. Facts it relies on, read from the installed source:
  - `python -m mlx_lm lora` takes --model --train --test --data DIR(train/valid/test.jsonl) --iters
    --learning-rate --batch-size --adapter-path --resume-adapter-file --num-layers -c config.yaml;
  - training progress goes to a rich terminal UI, NOT to a parseable "Train loss" line, so this worker never
    scrapes it: train_loss is measured by evaluating the new adapter on the training samples (same code as eval);
  - `--test` without `--train` loads the adapter in --adapter-path unless it is "" (then it scores the base);
  - the final summary line "Test loss X, Test ppl Y." is plain text and is what eval parses;
  - an adapter is a directory with adapters.safetensors and adapter_config.json.
Env: TLM_MLX_NUM_LAYERS (LoRA layers; unset = mlx-lm default 16, which errors on models with fewer layers);
     TLM_MLX_BATCH, TLM_MLX_GRAD_CKPT=1, TLM_MLX_MAX_SEQ (training memory; see memory_args).
Needs: pip install mlx-lm (on Linux: "mlx[cpu]" too).
"""
import argparse
import json
import math
import os
import re
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
    cmd = [sys.executable, "-m", "mlx_lm", "lora"] + args
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = p.stdout + p.stderr
    sys.stdout.write(out[-3000:] + "\n")
    if p.returncode != 0:
        sys.exit(p.returncode)
    return out


def exact_batch(n, cap=8):
    """Largest batch size <= cap that divides n, so a test pass sees every sample (mlx-lm drops the remainder)."""
    return max(d for d in range(1, min(n, cap) + 1) if n % d == 0)


def layers_args():
    n = os.environ.get("TLM_MLX_NUM_LAYERS")
    return ["--num-layers", n] if n else []


def memory_args():
    """Training memory knobs for machines where the default does not fit (a 4B model on a busy 16 GB Mac ran out of
    Metal memory at batch 4): TLM_MLX_BATCH (default 4), TLM_MLX_GRAD_CKPT=1, TLM_MLX_MAX_SEQ (tokens per sample)."""
    out = []
    if os.environ.get("TLM_MLX_GRAD_CKPT") == "1":
        out.append("--grad-checkpoint")
    if os.environ.get("TLM_MLX_MAX_SEQ"):
        out += ["--max-seq-length", os.environ["TLM_MLX_MAX_SEQ"]]
    return out


def loss_on(model, adapter, samples):
    """Mean token loss of base+adapter on the samples (adapter 'none' = the base model)."""
    with tempfile.TemporaryDirectory() as d:
        for name in ("train", "valid", "test"):
            write_jsonl(os.path.join(d, name + ".jsonl"), samples)
        args = ["--model", model, "--data", d, "--test", "--test-batches", "-1",
                "--batch-size", str(exact_batch(len(samples))),
                "--adapter-path", "" if adapter == "none" else adapter]
        out = run_lora(args)
    found = re.findall(r"Test loss ([0-9.eE+-]+)", out)
    if not found:
        sys.stderr.write("no 'Test loss' line in mlx_lm output\n")
        sys.exit(4)
    return float(found[-1])


def op_train(a):
    os.makedirs(a.adapter_out, exist_ok=True)
    samples = lines_of(a.data)
    if not samples:
        sys.stderr.write("no training samples\n")
        sys.exit(5)
    data_dir = os.path.join(a.adapter_out, "data")
    os.makedirs(data_dir, exist_ok=True)
    for name in ("train", "valid", "test"):
        write_jsonl(os.path.join(data_dir, name + ".jsonl"), samples)
    cfg = os.path.join(a.adapter_out, "lora.yaml")
    with open(cfg, "w") as f:
        f.write("lora_parameters:\n  rank: %d\n  scale: 20.0\n  dropout: 0.0\n" % a.rank)
    batch = min(int(os.environ.get("TLM_MLX_BATCH", "4")), len(samples))
    args = ["--model", a.base_model, "--train", "--data", data_dir, "--iters", str(a.steps),
            "--learning-rate", repr(a.lr), "--batch-size", str(batch), "--val-batches", "1",
            "--steps-per-report", "1", "--steps-per-eval", str(a.steps + 1), "--save-every", str(a.steps + 1),
            "--adapter-path", a.adapter_out, "-c", cfg] + layers_args() + memory_args()
    if a.adapter_in != "none":
        args += ["--resume-adapter-file", os.path.join(a.adapter_in, "adapters.safetensors")]
    run_lora(args)
    print("RESULT train_loss=%.8g" % loss_on(a.base_model, a.adapter_out, samples))


def op_eval(a):
    print("RESULT loss=%.8g" % loss_on(a.base_model, a.adapter, lines_of(a.data)))


def op_distance(a):
    import mlx.core as mx

    def load(p):
        return {} if p == "none" else mx.load(os.path.join(p, "adapters.safetensors"))

    x, y = load(a.a), load(a.b)
    total = 0.0
    for k in set(x) | set(y):
        u, v = x.get(k), y.get(k)
        zero = mx.zeros_like(u if u is not None else v)
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
