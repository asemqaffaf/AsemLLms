"""AsemLLM reference inference: numpy + safetensors only.

    python asemllm.py $'User: How do I cook pasta?\nAssistant:'   # $'...' makes \n a real newline
    python asemllm.py --chat "How do I cook pasta?"
"""
import json, re, sys
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file

TOKEN_RE = re.compile(rb" ?[A-Za-z0-9']+| ?[^\sA-Za-z0-9']|\n|\s")


class AsemLLM:
    def __init__(self, path="."):
        p = Path(path)
        self.cfg = json.loads((p / "config.json").read_text())
        w = load_file(str(p / "model.safetensors"))
        self.E, self.W1, self.b1, self.W2, self.b2 = (
            w[k].astype(np.float64) for k in ("embedding", "w1", "b1", "w2", "b2"))
        raw = json.loads((p / "vocab.json").read_text())
        self.tokens = [bytes.fromhex(t["hex"]) if isinstance(t, dict) else t.encode() for t in raw]
        self.index = {t: i for i, t in enumerate(self.tokens)}
        self.n = self.cfg["context_length"]
        self.pad = self.index.get(b"\n", 0)

    def logits(self, ctx_ids):
        """Next-token logits for exactly `context_length` token ids."""
        x = np.concatenate([self.E[i] for i in ctx_ids])
        h = np.tanh(self.W1 @ x + self.b1)
        return self.W2 @ h + self.b2

    def generate(self, prompt, max_tokens=100, temperature=0.7, seed=None, stop=()):
        """Continue `prompt` (str). Returns the continuation. Never emits <unk>."""
        rng = np.random.default_rng(seed)
        ids = [self.index.get(t, 0) for t in TOKEN_RE.findall(prompt.encode())]
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
            ctx = ctx[1:] + [nxt]
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
    model = AsemLLM(Path(__file__).parent)
    if args and args[0] == "--chat":
        print(model.chat([{"role": "user", "content": " ".join(args[1:])}]))
    else:
        text = " ".join(args) or "User: Hello!\nAssistant:"
        print(text + model.generate(text, max_tokens=60))
