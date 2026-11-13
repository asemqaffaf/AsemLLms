#!/usr/bin/env python3
"""Build the Hugging Face release folder for AsemLLM from exported models (see scripts/export.escript).

    python3 scripts/pack_release.py OUT REPO MLP_EXPORT_DIR [TRANSFORMER_EXPORT_DIR]

OUT/            MLP model: model.safetensors, config.json, vocab.json
OUT/transformer/ transformer model (if given)
OUT/asemllm.py, README.md, LICENSE

Every packed model is checked against the Erlang implementation's logits before anything is written as "done".
"""
import json, math, shutil, sys, importlib
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file

out, repo, mlp_src = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
tf_src = Path(sys.argv[4]) if len(sys.argv) > 4 else None
out.mkdir(parents=True, exist_ok=True)
TOKEN_RE = " ?[A-Za-z0-9']+| ?[^\\sA-Za-z0-9']|\\n|\\s"


def f32(x):
    return np.array(x, dtype=np.float32)


def pack(src, dest):
    """Write one model's files into `dest`; return (config, num_params, stats)."""
    dest.mkdir(parents=True, exist_ok=True)
    meta = json.loads((src / "meta.json").read_text())
    cfg, stats = meta["cfg"], meta["stats"]
    w = json.loads((src / "weights.json").read_text())
    V, T = cfg["vocab"], cfg["context"]
    if cfg.get("arch", "mlp") == "mlp":
        N, D, H = T, cfg["embed"], cfg["hidden"]
        tensors = {"embedding": f32(w["e"]), "w1": f32(w["w1"]), "b1": f32(w["b1"]),
                   "w2": f32(w["w2"]), "b2": f32(w["b2"])}
        assert tensors["embedding"].shape == (V, D) and tensors["w1"].shape == (H, N * D)
        config = {"model_type": "asemllm", "architecture": "embedding-mlp-language-model",
                  "vocab_size": V, "context_length": N, "embedding_dim": D, "hidden_dim": H, "activation": "tanh"}
    else:
        D, L, Hd, F = cfg["dim"], cfg["layers"], cfg["heads"], cfg["ff"]
        tensors = {"embedding": f32(w["e"]), "positional": f32(w["p"]), "lnf_g": f32(w["lnfg"]),
                   "lnf_b": f32(w["lnfb"]), "wout": f32(w["wout"]), "bout": f32(w["bout"])}
        names = {"ln1g": "ln1_g", "ln1b": "ln1_b", "wq": "wq", "wk": "wk", "wv": "wv", "wo": "wo",
                 "ln2g": "ln2_g", "ln2b": "ln2_b", "w1": "w1", "b1": "b1", "w2": "w2", "b2": "b2"}
        for i in range(1, L + 1):
            for erl, py in names.items():
                tensors[f"layers.{i - 1}.{py}"] = f32(w[f"{erl}.{i}"])
        assert tensors["embedding"].shape == (V, D) and tensors["positional"].shape == (T, D)
        config = {"model_type": "asemllm", "architecture": "transformer", "vocab_size": V, "context_length": T,
                  "dim": D, "heads": Hd, "layers": L, "ff_dim": F, "activation": "relu",
                  "norm": "pre-layernorm (eps 1e-5)", "positional": "learned"}
    params = sum(t.size for t in tensors.values())
    config.update({"tokenizer": meta["tokenizer"], "unk_token_id": 0, "num_parameters": params,
                   "tokenizer_regex": TOKEN_RE})
    save_file(tensors, str(dest / "model.safetensors"), metadata={"format": "np", "name": "AsemLLM"})
    (dest / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    shutil.copy(src / "vocab.json", dest / "vocab.json")
    return config, params, stats


mlp_cfg, mlp_params, mlp_stats = pack(mlp_src, out)
tf = pack(tf_src, out / "transformer") if tf_src else None
shutil.copy("/opt/homebrew/Cellar/erlang/28.4.2/LICENSE.txt", out / "LICENSE")

(out / "asemllm.py").write_text('''"""AsemLLM reference inference: numpy + safetensors only. Runs both model variants.

    python asemllm.py --chat "What is a good movie?"                    # MLP model (this folder)
    python asemllm.py --model transformer --chat "What is a good movie?"
    python asemllm.py --model transformer $'User: How do I cook pasta?\\\\nAssistant:'   # $'..' = real newline
"""
import json, re, sys
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file

TOKEN_RE = re.compile(rb" ?[A-Za-z0-9']+| ?[^\\sA-Za-z0-9']|\\n|\\s")


def _ln(x, g, b):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + 1e-5) * g + b


class AsemLLM:
    def __init__(self, path="."):
        p = Path(path)
        self.cfg = json.loads((p / "config.json").read_text())
        self.w = {k: v.astype(np.float64) for k, v in load_file(str(p / "model.safetensors")).items()}
        raw = json.loads((p / "vocab.json").read_text())
        self.tokens = [bytes.fromhex(t["hex"]) if isinstance(t, dict) else t.encode() for t in raw]
        self.index = {t: i for i, t in enumerate(self.tokens)}
        self.n = self.cfg["context_length"]
        self.arch = self.cfg["architecture"]
        self.pad = self.index.get(b"\\n", 0)

    def logits(self, ctx_ids):
        """Next-token logits. MLP: exactly `context_length` ids. Transformer: 1..`context_length` ids."""
        w = self.w
        if self.arch == "transformer":
            return self._transformer_logits(ctx_ids)
        x = np.concatenate([w["embedding"][i] for i in ctx_ids])
        return w["w2"] @ np.tanh(w["w1"] @ x + w["b1"]) + w["b2"]

    def _transformer_logits(self, ids):
        w, c = self.w, self.cfg
        T, D, H = len(ids), c["dim"], c["heads"]
        dh = D // H
        x = w["embedding"][ids] + w["positional"][:T]
        mask = np.triu(np.full((T, T), -np.inf), k=1)           # causal: position t sees only s <= t
        for i in range(c["layers"]):
            g = lambda k: w[f"layers.{i}.{k}"]
            xn = _ln(x, g("ln1_g"), g("ln1_b"))
            q, k_, v = (xn @ g(n).T for n in ("wq", "wk", "wv"))
            split = lambda m: m.reshape(T, H, dh).transpose(1, 0, 2)          # (heads, T, dh)
            s = split(q) @ split(k_).transpose(0, 2, 1) / np.sqrt(dh) + mask
            a = np.exp(s - s.max(-1, keepdims=True))
            a /= a.sum(-1, keepdims=True)
            ctx = (a @ split(v)).transpose(1, 0, 2).reshape(T, D)
            x = x + ctx @ g("wo").T
            xn = _ln(x, g("ln2_g"), g("ln2_b"))
            x = x + np.maximum(xn @ g("w1").T + g("b1"), 0.0) @ g("w2").T + g("b2")
        xf = _ln(x[-1], w["lnf_g"], w["lnf_b"])
        return w["wout"] @ xf + w["bout"]

    def generate(self, prompt, max_tokens=100, temperature=0.7, seed=None, stop=()):
        """Continue `prompt` (str). Returns the continuation. Never emits <unk>."""
        rng = np.random.default_rng(seed)
        ids = [self.index.get(t, 0) for t in TOKEN_RE.findall(prompt.encode())]
        if self.arch == "transformer":
            ctx = (ids or [self.pad])[-self.n:]
        else:
            ctx = ([self.pad] * max(0, self.n - len(ids)) + ids)[-self.n:]
        stops = [s.encode() for s in stop]
        out = b""
        for _ in range(max_tokens):
            z = self.logits(ctx)
            z[0] = -np.inf
            z = (z - z.max()) / max(temperature, 1e-6)
            p = np.exp(z - z.max())
            nxt = int(rng.choice(len(p), p=p / p.sum()))
            out += self.tokens[nxt]
            ctx = (ctx + [nxt])[-self.n:]
            cut = [out.find(s) for s in stops if s in out]
            if cut:
                out = out[:min(cut)]
                break
        return out.decode("utf-8", errors="replace")

    def chat(self, messages, **kw):
        """messages: [{"role": "user"|"assistant", "content": str}, ...]"""
        label = {"user": "User", "assistant": "Assistant"}
        prompt = "".join(f"{label[m['role']]}: {m['content']}\\n" for m in messages if m["role"] in label)
        return self.generate(prompt + "Assistant:", stop=("\\n", "User:"), **kw).strip()


if __name__ == "__main__":
    args = sys.argv[1:]
    sub = "."
    if args[:1] == ["--model"]:
        sub, args = args[1], args[2:]
    model = AsemLLM(Path(__file__).parent / sub)
    if args and args[0] == "--chat":
        print(model.chat([{"role": "user", "content": " ".join(args[1:])}]))
    else:
        text = " ".join(args) or "User: Hello!\\nAssistant:"
        print(text + model.generate(text, max_tokens=60))
''')


def row(name, stats):
    v, u, b = stats["val_loss"], stats["unigram"], stats["bigram"]
    return v, u, b


m_val, m_uni, m_bi = row("mlp", mlp_stats)
tf_section = ""
if tf:
    tcfg, tparams, tstats = tf
    t_val = tstats["val_loss"]
    tf_section = f"""
### Transformer variant (`transformer/`)

A small GPT-style decoder: **{tparams:,} parameters**, {tcfg['layers']} layers x {tcfg['heads']} heads, model dim {tcfg['dim']}
(MLP width {tcfg['ff_dim']}), learned positions, pre-LayerNorm, causal attention over a {tcfg['context_length']}-token window,
trained on the same data. Unlike the MLP it sees the whole prompt (up to {tcfg['context_length']} tokens).

| Held-out loss | Value | Perplexity |
|---|---|---|
| Transformer (best checkpoint of {tstats.get('steps', 0):,} steps) | {t_val:.2f} | {math.exp(t_val):.0f} |

The transformer overfits quickly on 138 KB: training loss kept falling while validation loss bottomed out early and
rose again, so the saved weights are the best validation checkpoint, not the last step. Its loss is averaged over every
position of held-out 32-token windows (including positions with little context), so it is **not directly comparable**
to the MLP's single-next-token number.
"""

tf_row = ("| Transformer | `transformer/` | " + f"{tf[1]:,} | {tf[0]['context_length']} tokens | GPT-style decoder (attention) |") if tf else ""
tf_cmd = 'python asemllm.py --model transformer --chat "What is a good movie?"    # transformer' if tf else ""

(out / "README.md").write_text(f"""---
license: apache-2.0
language:
- en
tags:
- text-generation
- tiny
- from-scratch
- erlang
- educational
pipeline_tag: text-generation
---

# AsemLLM

Tiny word-level language models written, trained and served entirely in **Erlang**, with no deep-learning framework:
supervised worker processes compute gradients in parallel. This repository holds the trained weights, the vocabulary and a
small numpy reference loader. **Source code (training, serving, web UI): https://github.com/asemqaffaf/AsemLLms**

> **These are toy models.** They learn the *style* of their training text (short `User:` / `Assistant:` dialogue) and
> produce fluent-looking English, but they do **not** understand questions, and answers are usually unrelated to them.
> They can state false things confidently. Do not use them for anything that needs correct answers.

## Two variants

| | Where | Parameters | Window | Architecture |
|---|---|---|---|---|
| MLP | repository root | {mlp_params:,} | {mlp_cfg['context_length']} tokens | embedding -> tanh hidden layer -> softmax (Bengio-style) |
{tf_row}

Both use the same tokenizer: words with their leading space, punctuation marks and newlines; the {mlp_cfg['vocab_size']}-token vocabulary
(top tokens seen at least twice, `<unk>` = id 0) is in `vocab.json`.

## Use it

```bash
pip install numpy safetensors huggingface_hub
huggingface-cli download {repo} --local-dir AsemLLM
cd AsemLLM
python asemllm.py --chat "What is a good movie?"                       # MLP
{tf_cmd}
```

```python
from asemllm import AsemLLM
m = AsemLLM(".")            # or AsemLLM("transformer")
print(m.chat([{{"role": "user", "content": "Hello!"}}], temperature=0.5, seed=1))
```

They use custom architectures, so they do **not** load in `transformers`, llama.cpp, Ollama or LM Studio.
Weights are float32 safetensors; `asemllm.py` is the reference implementation (verified against the Erlang code to ~1e-7 in logits).

## Source and other runtimes

The [GitHub repository](https://github.com/asemqaffaf/AsemLLms) has everything around these weights:

- the Erlang trainer and server (parallel gradient workers, web UI, JSON API) that produced them;
- KV-cache chat sessions, one supervised process per conversation, with logits identical to the stateless path;
- a Common Lisp (SBCL) port with a thread-parallel trainer, and a native C backend under the Erlang and Lisp runtimes;
- `tinyllm_tuner`, which proposes, scores and commits or rolls back fine-tuning rounds around any engine (for example MLX LoRA).

The weights here come from the Erlang trainer only. The tuner works on other engines' models, not on these.

## Later experiments (Common Lisp models, not published here)

After these weights, the GitHub repository added larger models trained with the Common Lisp engine (FP32, Apple Accelerate, one
M1 Pro CPU). Their weights are **not** in this repository; only the results are summarized. Both stages are negative or partial results.

**Stage 1** ([details](https://github.com/asemqaffaf/AsemLLms/blob/main/docs/results-stage1.md)): three transformers of 1.16M, 2.90M and
10.52M parameters, trained on the same FineWeb-Edu corpus (29.9M tokens). Test perplexity falls with size: 112.4, 87.8 and 80.8.
On a 90-question four-option set none answers better than chance (25%): 28.9%, 28.9% and 25.6%.

**Stage 2** ([details](https://github.com/asemqaffaf/AsemLLms/blob/main/docs/results-stage2.md),
[R7 analysis](https://github.com/asemqaffaf/AsemLLms/blob/main/docs/results-stage2-r7.md)): supervised fine-tuning of the 10.52M model
for passage question answering. The best model (R7) scores 45.8% (229/500) on the development set with four options (chance is 25%).
A simple word-overlap rule scores 62.0% on the same set, and the 90% target was not reached. Four controlled follow-up pilots did not
improve on R7, and the line was stopped on 2026-11-13
([final report](https://github.com/asemqaffaf/AsemLLms/blob/main/docs/results/stage2/mission-20261111/FINAL_REPORT.md)).
The 1,000-item test set has not been used to score any model, but its items are public in the repository.

A 3B-parameter model is a goal, not a result: at the measured speeds (Erlang ~0.4 GFLOP/s, Lisp ~14 GFLOP/s, no GPU) a
Chinchilla-sized 3B run would take thousands of years.

### Tuner check on a real model

A tuning round is committed only if held-out loss improves, a general probe set does not get worse, and the candidate beats a
control trained on a word-shuffled copy of the same data (`control => shuffle`). The control matters: part of any loss drop is
generic adaptation, so "loss fell" alone also passes garbage. In an earlier run without it, real data lowered held-out loss by
1.16 and garbage by 0.24, and the garbage round was committed.

Checked on Qwen3.5-4B (MLX 4-bit, LoRA) with 40 train and 60 held-out dialogues (no user question appears in both) and 51 probe
lines (facts, code, maths, other languages, instructions, prose). Losses are cross-entropy on those sets, not on AsemLLM data.

| Step | Result |
|---|---|
| baseline | held-out 2.971, probe 2.409 |
| garbage offered to the base model | rejected, `no_better_than_control` (it had lowered held-out loss to 2.73) |
| real data | committed, held-out 1.808, probe 1.789 |
| garbage after real data | rejected, weights unchanged |
| rollback to base | baseline scores reproduced exactly |

| Capability | Status |
|---|---|
| Real MLX training, commit, rollback to base | passed |
| Rejecting garbage training data | passed (this model, one run) |
| EUnit suite | passed (74 tests) |
| Rollback to an intermediate version on a real model | not tested |
| Better answers, not just lower loss | not shown |

The probe still falls with training (2.41 to 1.79) because it picks up the same generic effect, so it catches regressions, not
forgetting of knowledge. A gain on this chat data does not prove better answers.

## Training data

About {mlp_stats.get("tokens", 0):,} tokens (138 KB) of short, synthetic `User:` / `Assistant:` conversations on everyday topics
(cooking, gardening, travel, study, hobbies, ...), written by a locally run 4B-parameter instruction model.
No web scraping, no personal data. Text was filtered to well-formed dialogue lines and truncated generations were dropped.
Being machine-written, the data carries the repetitiveness and the mistakes of its source.

## Training (MLP)

- Word tokenizer, Adam (lr 0.005, linear decay to 10%), batch 128, gradient clipping 5, {mlp_stats.get("steps", 0):,} steps; best-validation checkpoint kept.
- Data-parallel training on one Apple M1 Pro (10 Erlang worker processes), about {mlp_stats.get("seconds", 0) / 60:.0f} minutes. Held-out set: last 10% of the text.

## Evaluation (held-out cross-entropy per token, lower is better)

| Model | Loss | Perplexity |
|---|---|---|
| Unigram baseline | {m_uni:.2f} | {math.exp(m_uni):.0f} |
| Bigram baseline (previous word) | {m_bi:.2f} | {math.exp(m_bi):.0f} |
| **AsemLLM MLP** | **{m_val:.2f}** | **{math.exp(m_val):.0f}** |
{tf_section}
With only 138 KB of text these models largely memorize phrases (MLP training loss fell to about 1.3 while validation loss stopped improving at {m_val:.2f}).

## Limitations

- Tiny data and tiny models: replies mostly do not relate to the prompt. English only; ASCII-centric vocabulary.
- No safety training; output reflects its machine-written training text.
- Not suitable for any real application. It exists to show how language models work, end to end, in Erlang.

## License

Apache-2.0 (see `LICENSE`).
""")

# ---- verify each packed model against the Erlang implementation -----------------------------
sys.path.insert(0, str(out))
asem = importlib.import_module("asemllm")


def check(dest, src, label):
    m = asem.AsemLLM(dest)
    worst = 0.0
    for tv in json.loads((src / "test_vectors.json").read_text()):
        py, er = m.logits(tv["context"]), np.array(tv["logits"])
        assert int(py.argmax()) == int(er.argmax()), f"{label}: argmax differs from Erlang"
        worst = max(worst, float(np.max(np.abs(py - er))))
    assert worst < 1e-4, f"{label}: logits differ from Erlang by {worst}"
    sample = m.chat([{"role": "user", "content": "What is a good movie?"}], temperature=0.5, seed=1)
    print(f"{label}: loader matches Erlang (max logit diff {worst:.2e}); sample: {sample!r}")


check(out, mlp_src, "mlp")
if tf:
    check(out / "transformer", tf_src, "transformer")
shutil.rmtree(out / "__pycache__", ignore_errors=True)
