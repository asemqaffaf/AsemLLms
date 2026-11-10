# TLM1: flat weight file

One file per model, little endian, written by `tinyllm_tlm:write/3` (Erlang) and `save-tlm` (lisp/src/tlm.lisp),
read by `tlm_open` in `tlm_model.c` through `mmap`. Nothing is parsed or copied: tensors are used where they lie.

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | magic `TLM1` |
| 4 | 4 each | version (1), dtype (1 = f64), norm (0 = LayerNorm with bias, 1 = RMSNorm), flags (bit 0: biases present) |
| 20 | 4 each | vocab, ctx, dim, heads, layers, ff |
| 44 | 4 | n_tensors |
| 48 | 8 each | table offset (128), data offset, file size |
| 128 | 40 per tensor | name (24 bytes, NUL padded), rows u32, cols u32, absolute offset u64 |
| data offset | | tensors, each starting on a 256-byte boundary; the data section and the file end on 4096-byte boundaries |

Matrices are `(out, in)` row-major: row j holds the weights of output unit j, so `y = W x`. Vectors are `1 x n`.
Names: `tok` (vocab x dim), `pos` (ctx x dim), `lnf.g`, `lnf.b`, `out.w` (vocab x dim), `out.b`, and per layer l (from 0)
`l{l}.ln1.g`, `.ln1.b`, `.wq`, `.wk`, `.wv`, `.wo` (dim x dim), `.ln2.g`, `.ln2.b`, `.w1` (ff x dim), `.b1`, `.w2` (dim x ff), `.b2`.
`.b` norm biases exist only for norm 0; `out.b`, `.b1`, `.b2` only with the biases flag.

Why it is mmap-friendly: page-aligned data section, 256-byte aligned tensors (fine for SIMD, and for a Metal buffer wrapped
around the mapping with `newBufferWithBytesNoCopy`), and files are always replaced by rename, never rewritten in place
(overwriting a mapped file under a running process crashes it).

The reader validates magic, version, dtype, sizes, the table and every tensor's shape and bounds before it hands out a pointer.
Only f64 is implemented; f32/bf16 would be new dtype values (and would give up the bit-identical results).
The two runtimes' models are different architectures (Erlang: LayerNorm + biases; Lisp: RMSNorm, no biases); the format records which.
