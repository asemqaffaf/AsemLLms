#!/usr/bin/env python3
"""Make a tiny random Llama (a few thousand parameters) with a character tokenizer, plus train/heldout/probe
text files, so the whole tuner pipeline can be smoke-tested with MLX without downloading anything.

  python3 scripts/tiny_mlx_model.py OUT_DIR       ->  OUT_DIR/model, train.txt, heldout.txt, probe.txt

The model is random, so absolute losses mean nothing; what the smoke test checks is that the plumbing runs and
that training on text lowers the loss on similar text.
"""
import json
import os
import sys

import mlx.core as mx
from mlx_lm.models import llama
from tokenizers import Regex, Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

CHARS = list("abcdefghijklmnopqrstuvwxyz .,0123456789")


def main(out):
    model_dir = os.path.join(out, "model")
    os.makedirs(model_dir, exist_ok=True)
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    for c in CHARS:
        vocab[c] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex("."), "isolated")
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", bos_token="<s>", eos_token="</s>",
                            pad_token="</s>").save_pretrained(model_dir)
    cfg = {"model_type": "llama", "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 64,
           "num_attention_heads": 2, "num_key_value_heads": 2, "rms_norm_eps": 1e-5, "vocab_size": len(vocab),
           "max_position_embeddings": 128, "tie_word_embeddings": True, "rope_theta": 10000.0}
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(cfg, f)
    mx.random.seed(0)
    model = llama.Model(llama.ModelArgs.from_dict(cfg))
    mx.eval(model.parameters())
    model.save_weights(os.path.join(model_dir, "model.safetensors"))
    texts = {
        "train.txt": ["the cat sat on the mat. the cat sat on the mat."] * 8,
        "heldout.txt": ["the cat sat on the mat.", "the cat sat on the hat."] * 2,
        "probe.txt": ["one two three four five six seven.", "one two three four five six."] * 2,
    }
    for name, lines in texts.items():
        with open(os.path.join(out, name), "w") as f:
            f.write("\n".join(lines) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tiny")
