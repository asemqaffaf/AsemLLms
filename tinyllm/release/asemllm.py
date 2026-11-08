"""AsemLLM reference inference: numpy + safetensors only. Runs both model variants.

    python asemllm.py --chat "What is a good movie?"                    # MLP model (this folder)
    python asemllm.py --model transformer --chat "What is a good movie?"
    python asemllm.py --model transformer $'User: How do I cook pasta?\\nAssistant:'   # $'..' = real newline
"""
import json, re, sys
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file

TOKEN_RE = re.compile(rb" ?[A-Za-z0-9']+| ?[^\sA-Za-z0-9']|\n|\s")


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
        self.pad = self.index.get(b"\n", 0)

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
        prompt = "".join(f"{label[m['role']]}: {m['content']}\n" for m in messages if m["role"] in label)
        return self.generate(prompt + "Assistant:", stop=("\n", "User:"), **kw).strip()


if __name__ == "__main__":
    args = sys.argv[1:]
    sub = "."
    if args[:1] == ["--model"]:
        sub, args = args[1], args[2:]
    model = AsemLLM(Path(__file__).parent / sub)
    if args and args[0] == "--chat":
        print(model.chat([{"role": "user", "content": " ".join(args[1:])}]))
    else:
        text = " ".join(args) or "User: Hello!\nAssistant:"
        print(text + model.generate(text, max_tokens=60))
